"""Gather kernels: indirect row-indexed load via tl.descriptor_gather.

This is the core pattern behind block-indexed gather operations such as
KV-cache block fetch and embedding lookups: given a 1D index tensor of
row positions, read those rows from a source tensor into a contiguous
output tile.

Two ``@triton.jit`` functions for rank-2 sources:

- :func:`gather_kernel`    — single-program: one kernel invocation
                              consumes the whole index array and writes
                              the whole output tile.
- :func:`gather_2d_kernel` — multi-program: tiled across a 2D core
                              grid, each ``(pid_m, pid_n)`` produces a
                              ``BLOCK_ROWS x BLOCK_COLS`` output tile.
                              Adds a ``BLOCK_ROWS`` constexpr; gathers
                              the full row width by column-tiling
                              instead of taking a fixed slice.

Four ``@triton.jit`` functions for rank-N (N ≥ 3) sources:

- :func:`gather_3d_kernel`       — rank-3 source ``[M, BLOCK_SIZE, HEAD_DIM]``;
                                    block-fetch gather shape.
- :func:`gather_3d_group_kernel` — rank-3 source ``[M, NUM_GROUPS, HEAD_DIM]``
                                    with non-trivial ``y_offset`` selecting the
                                    group dimension.
- :func:`gather_4d_kernel`       — rank-4 source
                                    ``[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM]``;
                                    group dim permuted to physical dim 1.
- :func:`scatter_3d_kernel`      — write-back mirror of :func:`gather_3d_kernel`.
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
      * ``K_INDICES`` must be at least 8 (verifier; both backends).
      * ``BLOCK_COLS`` minimum depends on backend. On TMA/NVIDIA the
        verifier requires ``block_shape[1] >= 32 / bitwidth * 8`` (for
        ``f32`` that is 8 columns). On Spyre this check is skipped, so
        ``block_shape[1]`` may be as small as 1 — the rank-N variants
        rely on this (e.g. ``[1, 1, HEAD_DIM]``).

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


# ---------------------------------------------------------------------------
# Rank-N (N ≥ 3) gather / scatter kernels
# ---------------------------------------------------------------------------


@triton.jit
def gather_3d_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_INDICES: tl.constexpr,
):
    """Single-program 3D gather: pull K_INDICES blocks from [M, BLOCK_SIZE, HEAD_DIM].

    Implements::

        for i in range(K_INDICES):
            out[i, :, :] = in[idx[i], :, :]

    Source ``in[M, BLOCK_SIZE, HEAD_DIM]`` is a block-indexed KV-cache where
    each block is a ``[BLOCK_SIZE, HEAD_DIM]`` tile.  The index array
    ``idx[K_INDICES]`` names which blocks to fetch; the result is
    ``out[K_INDICES, BLOCK_SIZE, HEAD_DIM]``.

    Block shape ``[1, BLOCK_SIZE, HEAD_DIM]``: the leading 1 satisfies the
    ``tt.descriptor_gather`` requirement (one block per gather template).
    ``y_offset=0`` because dim 1 covers the full ``BLOCK_SIZE`` extent —
    no sub-block slicing.

    Constraints:
      - ``K_INDICES >= 8`` (verifier minimum on ``x_offsets.shape[0]``).
      - ``HEAD_DIM`` is a power of two.
    """
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[K_INDICES],
    )
    idx = idx_desc.load([0])

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, BLOCK_SIZE, HEAD_DIM],
        strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, BLOCK_SIZE, HEAD_DIM],
    )
    result = in_desc.gather(idx, 0)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, BLOCK_SIZE, HEAD_DIM],
        strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[K_INDICES, BLOCK_SIZE, HEAD_DIM],
    )
    out_desc.store([0, 0, 0], result)


@triton.jit
def gather_3d_group_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    group_idx,
    M: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_INDICES: tl.constexpr,
):
    """Single-program 3D gather with a non-trivial group selector.

    Source ``in[M, NUM_GROUPS, HEAD_DIM]``.  Each gathered block selects one
    group via the runtime scalar ``group_idx``, passed as ``y_offset`` to
    ``descriptor_gather``.  Block shape ``[1, 1, HEAD_DIM]``: the leading 1
    is the indirect dim, the second 1 means only the selected group slice
    is read per gather.

    Implements::

        for i in range(K_INDICES):
            out[i, 0, :] = in[idx[i], group_idx, :]

    Result shape: ``[K_INDICES, 1, HEAD_DIM]``.

    ``group_idx`` is non-zero in the default variant so the ``c_y`` capture
    on dim 1 is load-bearing, not a trivial zero — a regression that dropped
    ``c_y`` from the subscript would pass with ``group_idx=0`` but fail here.

    Note: this variant does NOT demonstrate the
    ``descriptor-gather-nd-subscripts`` pattern. That pattern pins the
    dim 1 (partial extent + ``y_offset``) vs. dim ≥ 2 (full extent, no
    offset) role split, which requires block dim 1 > 1. Here block dim 1
    is 1, so the role contrast is not visible. A separate fixture with
    block shape e.g. ``[1, TOKEN_DIM, HEAD_DIM]`` is needed to cover that
    pattern end-to-end.

    Constraints:
      - ``K_INDICES >= 8``.
      - ``HEAD_DIM`` is a power of two.
      - ``0 <= group_idx < NUM_GROUPS``.
    """
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[K_INDICES],
    )
    idx = idx_desc.load([0])

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, NUM_GROUPS, HEAD_DIM],
        strides=[NUM_GROUPS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, 1, HEAD_DIM],
    )
    result = in_desc.gather(idx, group_idx)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, 1, HEAD_DIM],
        strides=[HEAD_DIM, HEAD_DIM, 1],
        block_shape=[K_INDICES, 1, HEAD_DIM],
    )
    out_desc.store([0, 0, 0], result)


@triton.jit
def gather_4d_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    group_idx,
    NUM_BLOCKS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INNER_DIM: tl.constexpr,
    K_INDICES: tl.constexpr,
):
    """Single-program 4D gather: block_id × group × BLOCK_SIZE × INNER_DIM.

    Source ``in[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM]``.  Block
    shape ``[1, 1, BLOCK_SIZE, INNER_DIM]``: two leading 1s for the indirect
    and group dims; trailing two dims cover the full block extent.

    Implements::

        for i in range(K_INDICES):
            out[i, 0, :, :] = in[idx[i], group_idx, :, :]

    ``y_offset = group_idx`` selects dim 1 (NUM_GROUPS); dims 2 and 3 are
    direct with no offset.  Result shape:
    ``[K_INDICES, 1, BLOCK_SIZE, INNER_DIM]``.

    Constraints:
      - ``K_INDICES >= 8``.
      - ``INNER_DIM`` is a power of two.
      - ``0 <= group_idx < NUM_GROUPS``.
    """
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[K_INDICES],
    )
    idx = idx_desc.load([0])

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM],
        strides=[NUM_GROUPS * BLOCK_SIZE * INNER_DIM,
                 BLOCK_SIZE * INNER_DIM, INNER_DIM, 1],
        block_shape=[1, 1, BLOCK_SIZE, INNER_DIM],
    )
    result = in_desc.gather(idx, group_idx)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, 1, BLOCK_SIZE, INNER_DIM],
        strides=[BLOCK_SIZE * INNER_DIM, BLOCK_SIZE * INNER_DIM, INNER_DIM, 1],
        block_shape=[K_INDICES, 1, BLOCK_SIZE, INNER_DIM],
    )
    out_desc.store([0, 0, 0, 0], result)


@triton.jit
def scatter_3d_kernel(
    dst_ptr,
    data_ptr,
    idx_ptr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_INDICES: tl.constexpr,
):
    """Single-program 3D scatter: write K_INDICES blocks into [M, BLOCK_SIZE, HEAD_DIM].

    Write-back mirror of :func:`gather_3d_kernel`.  Reads a
    ``[K_INDICES, BLOCK_SIZE, HEAD_DIM]`` data tile from ``data_ptr`` and
    scatters it into the destination ``dst[M, BLOCK_SIZE, HEAD_DIM]`` at
    the row positions named by ``idx_ptr``.

    Implements::

        for i in range(K_INDICES):
            dst[idx[i], :, :] = data[i, :, :]

    The scatter descriptor block shape ``[1, BLOCK_SIZE, HEAD_DIM]`` mirrors
    the gather descriptor: the leading 1 is the indirect dim, and the
    trailing dims cover the full block extent.

    Constraints:
      - ``K_INDICES >= 8``.
      - ``HEAD_DIM`` is a power of two.
    """
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[K_INDICES],
    )
    idx = idx_desc.load([0])

    data_desc = tl.make_tensor_descriptor(
        data_ptr,
        shape=[K_INDICES, BLOCK_SIZE, HEAD_DIM],
        strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[K_INDICES, BLOCK_SIZE, HEAD_DIM],
    )
    data = data_desc.load([0, 0, 0])

    dst_desc = tl.make_tensor_descriptor(
        dst_ptr,
        shape=[M, BLOCK_SIZE, HEAD_DIM],
        strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, BLOCK_SIZE, HEAD_DIM],
    )
    dst_desc.scatter(data, idx, 0)
