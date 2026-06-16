"""Gather kernels: indirect row-indexed load via tl.descriptor_gather.

This is the core pattern behind paged KV-cache attention and embedding
lookups: given a 1D index tensor of row positions, read those rows from
a 2D source matrix into a contiguous output tile.

Three ``@triton.jit`` functions:

- :func:`gather_kernel`    — single-program: one kernel invocation
                              consumes the whole index array and writes
                              the whole output tile.
- :func:`gather_2d_kernel` — multi-program: tiled across a 2D core
                              grid, each ``(pid_m, pid_n)`` produces a
                              ``BLOCK_ROWS x BLOCK_COLS`` output tile.
                              Adds a ``BLOCK_ROWS`` constexpr; gathers
                              the full row width by column-tiling
                              instead of taking a fixed slice.
- :func:`gather_1d_kernel` — multi-program 1D-source gather:
                              ``out[i] = in[idx[i]]`` over a 1D source
                              vector, distributed across a 1D core grid
                              (``grid=[32]``). The source vector ``in[K]``
                              is modelled internally as a ``[K, 1]`` column
                              matrix because ``tt.descriptor_gather``
                              requires a rank-≥2 descriptor with a
                              leading-1 block dim — there is no rank-1
                              gather op in the dialect. The rank-2
                              ``[BLOCK_ROWS, 1]`` gather result is
                              reshaped to rank-1 ``[BLOCK_ROWS]`` before
                              storing back to a 1D output buffer.
"""

import triton
import triton.language as tl


@triton.jit
def gather_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    y_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_INDICES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Single-program gather: pull K_INDICES rows from a [M, N] source matrix.

    Vocabulary (matches ``docs/tiling_concepts.md``):

    - **Source matrix** ``in[M, N]`` — the data being read from. Conceptually
      a "vocabulary table" or "page pool"; rows are the gather targets.
    - **Index array** ``idx[K_INDICES]`` — a 1D ``i32`` array of row positions
      into the source. Each entry must satisfy ``0 <= idx[i] < M``.
    - **Result tile** ``out[K_INDICES, BLOCK_COLS]`` — gathered rows written
      contiguously. Row ``i`` of the result is the slice
      ``in[idx[i], y_offset : y_offset + BLOCK_COLS]``.
    - ``y_offset`` — column starting offset into the source row. Lets the
      kernel gather only a slice of each row (e.g. one head_dim worth of a
      KV cache row).

    Pythonic semantics::

        for i in range(K_INDICES):
            out[i, :] = in[idx[i], y_offset : y_offset + BLOCK_COLS]

    Single-program kernel (``tl.program_id`` is not used). The entire index
    array is consumed in one descriptor_gather call, which means
    ``K_INDICES`` must equal the total number of indices to gather. This is
    the minimal shape needed to exercise the gather lowering; multi-program
    tiling over a larger index array is left to a future variant.

    Constraints baked in by ``tt.descriptor_gather``:
      * The source descriptor's ``block_shape`` must have **exactly one row**
        — the gather is what fans the load out across many rows.
      * ``K_INDICES`` must be at least 8 (verifier).
      * ``BLOCK_COLS`` must be at least ``32 / bitwidth * 8`` (verifier);
        for ``f32`` that is 8 columns.

    Why the index array is loaded via a descriptor:
      The upstream Triton/GPU pattern ``tl.load(idx_ptr + tl.arange(...))``
      emits ``tt.splat`` + ``tt.addptr`` + ``tt.load`` on a tensor of
      ``!tt.ptr<i32>``. The Spyre ``LowerComputeOps`` pass cannot lower
      that (``linalg.fill`` rejects ``!tt.ptr`` operands). A descriptor
      load lowers cleanly through ``LowerDescriptorMemory``, so we use
      that pattern here. See ``docs/gather.md`` for the full discussion.
    """
    # 1. Load the K_INDICES row indices via a 1D descriptor.
    #    Workaround for the raw-pointer load gap — see docstring.
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[K_INDICES],
    )
    idx = idx_desc.load([0])

    # 2. Build a descriptor over the full source matrix. The block_shape
    #    MUST have a leading 1 — descriptor_gather fans this single-row
    #    template out across the K_INDICES rows named by `idx`.
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_COLS],
    )

    # 3. Gather: result[i, j] = in[idx[i], y_offset + j] for i in [0, K_INDICES),
    #    j in [0, BLOCK_COLS).
    result = in_desc.gather(idx, y_offset)

    # 4. Write the contiguous [K_INDICES, BLOCK_COLS] result tile.
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, BLOCK_COLS],
        strides=[BLOCK_COLS, 1],
        block_shape=[K_INDICES, BLOCK_COLS],
    )
    out_desc.store([0, 0], result)


@triton.jit
def gather_2d_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_INDICES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Gather tiled over a 2D grid of cores.

    Implements ``out[i, j] = in[idx[i], j]`` distributed across a 2D
    core grid. Each program instance ``(pid_m, pid_n)`` produces one
    ``BLOCK_ROWS x BLOCK_COLS`` tile of the output by gathering
    ``BLOCK_ROWS`` rows of the source matrix (selected by ``idx``) and
    a ``BLOCK_COLS``-wide column slice.

    Differences from :func:`gather_kernel`:

    - Two-axis ``tl.program_id`` (``(pid_m, pid_n)``); the single-program
      kernel uses no ``tl.program_id`` at all.
    - Adds a ``BLOCK_ROWS`` constexpr; the single-program kernel gathers
      all ``K_INDICES`` rows in one shot, so it has no row-tile size.
    - Output is the full ``[K_INDICES, N]`` matrix — every column is
      written. The kernel column-tiles its way across by using each
      tile's ``offset_n`` as the gather's column offset, so there is no
      ``y_offset`` argument.

    Grid dimensions are read via ``tl.num_programs(0)`` /
    ``tl.num_programs(1)`` and folded to compile-time constants by
    DistributeWork against ``SpyreOptions.grid``.

    Preconditions (not checked — variant shapes are picked to satisfy
    them; a violating variant would read/write out-of-bounds):
      - ``K_INDICES % BLOCK_ROWS == 0`` and ``N % BLOCK_COLS == 0``
        (tiles exactly cover the output).
      - ``cdiv(K_INDICES, BLOCK_ROWS) % num_programs(0) == 0`` and
        likewise for axis 1 (each core owns an integer number of tiles
        along each axis, so no masking/bounds-check is required).
      - ``max(idx) < M`` (no out-of-range row gather).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_m = tl.num_programs(0)
    grid_n = tl.num_programs(1)

    m_blocks = tl.cdiv(K_INDICES, BLOCK_ROWS)
    n_blocks = tl.cdiv(N, BLOCK_COLS)
    rows_per_core = tl.cdiv(m_blocks, grid_m)
    cols_per_core = tl.cdiv(n_blocks, grid_n)

    # Descriptors built once outside the loop (vector_add_2d pattern).
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[BLOCK_ROWS],
    )
    # Source matrix: block shape must be [1, BLOCK_COLS] — the
    # tt.descriptor_gather verifier requires a single-row template that
    # the gather fans out to BLOCK_ROWS rows via x_offsets.
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_COLS],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, N],
        strides=[N, 1],
        block_shape=[BLOCK_ROWS, BLOCK_COLS],
    )

    m_start = pid_m * rows_per_core
    n_start = pid_n * cols_per_core

    for m_sub in range(0, rows_per_core):
        for n_sub in range(0, cols_per_core):
            m_block = m_start + m_sub
            n_block = n_start + n_sub
            offset_m = m_block * BLOCK_ROWS
            offset_n = n_block * BLOCK_COLS

            idx = idx_desc.load([offset_m])
            tile = in_desc.gather(idx, offset_n)
            out_desc.store([offset_m, offset_n], tile)


@triton.jit
def gather_1d_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    K: tl.constexpr,
    K_INDICES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """1D-source gather distributed across a 1D core grid.

    Implements ``out[i] = in[idx[i]]`` where ``in`` is a 1D vector of
    length ``K`` and ``idx`` is a 1D index array of length ``K_INDICES``.
    The kernel runs on ``grid=[32]``; each ``tl.program_id(0)`` instance
    handles ``cdiv(K_INDICES, BLOCK_ROWS) / num_programs(0)`` index-tiles.

    Vocabulary:
      - ``K``         — length of the source vector ``in[K]``.
      - ``K_INDICES`` — number of indices to gather (length of ``idx``
                        and of the output buffer).
      - ``BLOCK_ROWS`` — index-tile size: each call to ``in_desc.gather``
                         consumes ``BLOCK_ROWS`` indices. Must be ≥ 8 per
                         the frontend gather verifier.

    Why a ``[K, 1]`` column-matrix view of ``in`` instead of a rank-1
    descriptor: ``tt.descriptor_gather`` requires the descriptor block
    to be at least rank-2 with a leading-1 dim (the indirect axis). A
    rank-1 descriptor like ``<BLOCK_ROWS x f32>`` is rejected by the
    op verifier (see
    ``test_lower_desc_memory.py::test_gather_rank1_block_rejected``).
    Modelling the same K elements as ``[K, 1]`` with ``block_shape=[1, 1]``
    satisfies the verifier: dim 0 (size K) is the indirect axis and dim 1
    (size 1) is the per-row scalar payload. The gather result is rank-2
    ``[BLOCK_ROWS, 1]``; we ``tl.reshape`` it to rank-1 ``[BLOCK_ROWS]``
    so the output buffer can stay 1D.

    ``y_offset`` is hardcoded to 0 — there is no column slicing on a
    width-1 row.

    Preconditions (not checked):
      - ``BLOCK_ROWS >= 8`` (frontend gather verifier).
      - ``K_INDICES % BLOCK_ROWS == 0`` and
        ``cdiv(K_INDICES, BLOCK_ROWS) % num_programs(0) == 0`` so each
        core owns an integer number of full tiles (no masking needed).
      - ``max(idx) < K`` (no out-of-range gather).
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    m_blocks = tl.cdiv(K_INDICES, BLOCK_ROWS)
    blocks_per_core = tl.cdiv(m_blocks, num_cores)

    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[BLOCK_ROWS],
    )
    # Source vector modelled as [K, 1]; block <1, 1> picks one scalar
    # per gathered row. The leading 1 in the block satisfies the
    # gather verifier's "block dim 0 == 1" rule.
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[K, 1],
        strides=[1, 1],
        block_shape=[1, 1],
    )
    # Output is 1D — same shape as idx.
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[BLOCK_ROWS],
    )

    m_start = pid * blocks_per_core
    m_end = tl.minimum(m_start + blocks_per_core, m_blocks)
    for m in range(m_start, m_end):
        offset_m = m * BLOCK_ROWS
        idx = idx_desc.load([offset_m])
        # Rank-2 [BLOCK_ROWS, 1] gather result.
        tile_2d = in_desc.gather(idx, 0)
        # Collapse to rank-1 [BLOCK_ROWS] so it matches out_desc.
        tile_1d = tl.reshape(tile_2d, [BLOCK_ROWS])
        out_desc.store([offset_m], tile_1d)
