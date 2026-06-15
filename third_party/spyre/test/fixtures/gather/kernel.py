"""Gather kernels: indirect row-indexed load via tl.descriptor_gather.

This is the core pattern behind indirect row-gather operations such as
embedding lookups and block-indexed fetches: given a 1D index tensor of
row positions, read those rows from a source tensor into a contiguous
output tile.

Three ``@triton.jit`` functions for rank-2 and rank-1 sources:

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

Four ``@triton.jit`` functions for rank-N (N ≥ 3) sources:

- :func:`gather_3d_kernel`       — rank-3 source ``[M, BLOCK_SIZE, HEAD_DIM]``;
                                    block-fetch gather shape.
- :func:`gather_3d_group_kernel` — rank-3 source ``[M, NUM_GROUPS, HEAD_DIM]``
                                    with non-trivial ``y_offset`` selecting the
                                    group dimension.
- :func:`gather_4d_kernel`       — rank-4 source
                                    ``[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM]``;
                                    group dim permuted to physical dim 1.
- :func:`gather_3d_partial_kernel`  — rank-3 source ``[M, NUM_TOKENS, HEAD_DIM]``
                                       with ``TOKEN_BLOCK | NUM_TOKENS``; block
                                       shape ``[1, TOKEN_BLOCK, HEAD_DIM]``.
                                       An ``scf.for`` sweeps the
                                       ``NUM_BLOCKS = NUM_TOKENS // TOKEN_BLOCK``
                                       windows along dim 1 with
                                       ``y_offset = b * TOKEN_BLOCK`` computed
                                       inside the loop, so each iteration is
                                       ``out[i, b*G:(b+1)*G, :] = in[idx[i], b*G:(b+1)*G, :]``.
- :func:`scatter_3d_kernel`      — write-back mirror of :func:`gather_3d_kernel`.
- :func:`scatter_3d_partial_kernel` — write-back mirror of
                                       :func:`gather_3d_partial_kernel`.
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
      kernel gather only a slice of each row rather than the full width.

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
def gather_kernel_spyre(
    in_ptr,
    out_ptr,
    idx_ptr,
    y_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_INDICES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
):
    """Spyre physical-layout variant of gather_kernel.

    Identical to gather_kernel except that in_desc and out_desc are
    annotated with Spyre stick-tiling layouts via tl.spyre_tensor_layout.
    idx_desc is not annotated (index arrays have no stick layout).

    IN_LAYOUT  — stick-tiling for in_ptr's full [M, N] extent.
    OUT_LAYOUT — stick-tiling for out_ptr's full [K_INDICES, BLOCK_COLS] extent.
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
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_COLS],
    )
    if IN_LAYOUT is not None and IN_LAYOUT != 0:
        tl.spyre_tensor_layout(in_desc, IN_LAYOUT)

    result = in_desc.gather(idx, y_offset)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, BLOCK_COLS],
        strides=[BLOCK_COLS, 1],
        block_shape=[K_INDICES, BLOCK_COLS],
    )
    if OUT_LAYOUT is not None and OUT_LAYOUT != 0:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

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

    Source ``in[M, BLOCK_SIZE, HEAD_DIM]`` is a block-indexed source tensor
    where each block is a ``[BLOCK_SIZE, HEAD_DIM]`` tile.  The index array
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
def gather_3d_partial_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    M: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_INDICES: tl.constexpr,
):
    """Single-program 3D gather: sweep ``NUM_TOKENS`` in ``TOKEN_BLOCK``-wide windows.

    Source ``in[M, NUM_TOKENS, HEAD_DIM]``; gather block shape
    ``[1, TOKEN_BLOCK, HEAD_DIM]`` with ``TOKEN_BLOCK | NUM_TOKENS``. An
    ``scf.for`` walks the ``NUM_BLOCKS = NUM_TOKENS // TOKEN_BLOCK`` windows
    along dim 1; on iteration ``b`` the gather reads
    ``in[idx[i], b*TOKEN_BLOCK:(b+1)*TOKEN_BLOCK, :]`` and writes that
    slice into ``out``. The full sweep reconstructs ``in[idx, :, :]``.

    Implements::

        for b in range(0, NUM_TOKENS // TOKEN_BLOCK):
            for i in range(K_INDICES):
                out[i, b*TOKEN_BLOCK:(b+1)*TOKEN_BLOCK, :] = (
                    in[idx[i], b*TOKEN_BLOCK:(b+1)*TOKEN_BLOCK, :]
                )

    Result shape: ``[K_INDICES, NUM_TOKENS, HEAD_DIM]``.

    Each loop iteration is a textbook instance of the partial-extent gather
    pattern (block dim 1 = ``TOKEN_BLOCK`` < shape dim 1 = ``NUM_TOKENS``,
    ``y_offset = b * TOKEN_BLOCK``, dim 2 read at full extent with no
    offset). The ``y_offset`` SSA value is produced *inside* the loop body
    by ``arith.muli %iv, TOKEN_BLOCK``, so the lowered
    ``ktdp.construct_indirect_access_tile`` lands inside the ``scf.for``
    region with its ``c_y`` operand captured from a loop-variant value.
    The descriptor's ``ktdp.construct_memory_view`` stays at function top —
    same hoisting rule as ``descriptor-placement-top-level``, but for the
    gather path. The fixture pins both placements via ``extra_checks``.

    What this variant pins beyond ``3d`` (full block extent on dim 1) and
    ``3d_group`` (block dim 1 = 1):
      - Strict partial extent on dim 1 (``TOKEN_BLOCK > 1`` *and*
        ``TOKEN_BLOCK < NUM_TOKENS``) — ``y_offset`` selects which
        ``TOKEN_BLOCK``-wide group to read.
      - In-loop ``y_offset`` capture — the lowered access tile must land
        inside the ``scf.for`` body and capture the loop-variant
        ``arith.muli`` result, not a hoisted constant.

    Constraints:
      - ``K_INDICES >= 8``.
      - ``HEAD_DIM`` and ``TOKEN_BLOCK`` are powers of two
        (``validate_block_shape``).
      - ``NUM_TOKENS % TOKEN_BLOCK == 0`` (loop sweep covers every element
        of dim 1 exactly once; oracle assumes exact tiling).
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
        shape=[M, NUM_TOKENS, HEAD_DIM],
        strides=[NUM_TOKENS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, TOKEN_BLOCK, HEAD_DIM],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, NUM_TOKENS, HEAD_DIM],
        strides=[NUM_TOKENS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[K_INDICES, TOKEN_BLOCK, HEAD_DIM],
    )

    NUM_BLOCKS: tl.constexpr = NUM_TOKENS // TOKEN_BLOCK
    for b in range(0, NUM_BLOCKS):
        y_offset = b * TOKEN_BLOCK
        win = in_desc.gather(idx, y_offset)
        out_desc.store([0, y_offset, 0], win)


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


@triton.jit
def scatter_3d_partial_kernel(
    dst_ptr,
    data_ptr,
    idx_ptr,
    M: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_INDICES: tl.constexpr,
):
    """Single-program 3D scatter: sweep ``NUM_TOKENS`` in ``TOKEN_BLOCK``-wide windows.

    Write-back mirror of :func:`gather_3d_partial_kernel`. Source
    ``data[K_INDICES, NUM_TOKENS, HEAD_DIM]``; an ``scf.for`` walks the
    ``NUM_BLOCKS = NUM_TOKENS // TOKEN_BLOCK`` windows along dim 1, on
    iteration ``b`` loading ``data[i, b*TOKEN_BLOCK:(b+1)*TOKEN_BLOCK, :]``
    and scattering it into
    ``dst[idx[i], b*TOKEN_BLOCK:(b+1)*TOKEN_BLOCK, :]``.

    Implements::

        for b in range(0, NUM_TOKENS // TOKEN_BLOCK):
            for i in range(K_INDICES):
                dst[idx[i], b*TOKEN_BLOCK:(b+1)*TOKEN_BLOCK, :] = (
                    data[i, b*TOKEN_BLOCK:(b+1)*TOKEN_BLOCK, :]
                )

    Same role-split rationale as :func:`gather_3d_partial_kernel`: the
    scatter descriptor's block shape ``[1, TOKEN_BLOCK, HEAD_DIM]`` mixes
    a partial-extent indexed dim (``TOKEN_BLOCK < NUM_TOKENS``) with a
    full-extent direct dim (``HEAD_DIM``), and the ``y_offset`` is
    computed inside the loop body so the lowered
    ``ktdp.construct_indirect_access_tile`` for the scatter lands inside
    the ``scf.for`` region.

    Constraints:
      - ``K_INDICES >= 8``.
      - ``HEAD_DIM`` and ``TOKEN_BLOCK`` are powers of two.
      - ``NUM_TOKENS % TOKEN_BLOCK == 0``.
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
        shape=[K_INDICES, NUM_TOKENS, HEAD_DIM],
        strides=[NUM_TOKENS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[K_INDICES, TOKEN_BLOCK, HEAD_DIM],
    )
    dst_desc = tl.make_tensor_descriptor(
        dst_ptr,
        shape=[M, NUM_TOKENS, HEAD_DIM],
        strides=[NUM_TOKENS * HEAD_DIM, HEAD_DIM, 1],
        block_shape=[1, TOKEN_BLOCK, HEAD_DIM],
    )

    NUM_BLOCKS: tl.constexpr = NUM_TOKENS // TOKEN_BLOCK
    for b in range(0, NUM_BLOCKS):
        y_offset = b * TOKEN_BLOCK
        win = data_desc.load([0, y_offset, 0])
        dst_desc.scatter(win, idx, y_offset)


@triton.jit
def gather_2d_index_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    y_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Single-program gather driven by a rank-2 (S0 x S1) index *grid*.

    The rank-K x_offsets relaxation: instead of a 1-D list of row
    indices, the index buffer is a 2-D grid (e.g. one page per
    ``(sequence, head)`` pair).  For each grid cell ``(i, j)`` the kernel
    gathers ``BLOCK_COLS`` columns of source row ``idx[i, j]``:

        result[i, j, :] = in[idx[i, j], y_offset : y_offset + BLOCK_COLS]

    Lowers to a single ``ktdp.construct_indirect_access_tile`` whose
    indirect base dim carries a 2-D address into the index view (see
    ``docs/impl-strategy-2d-x-offsets.md``), then a ``ktdp.load`` of the
    rank-3 ``[S0, S1, BLOCK_COLS]`` result tile.
    """
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[S0, S1],
        strides=[S1, 1],
        block_shape=[S0, S1],
    )
    idx = idx_desc.load([0, 0])                    # tensor<S0 x S1 x i32>

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_COLS],
    )
    data = in_desc.gather(idx, y_offset)           # tensor<S0 x S1 x BLOCK_COLS>

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[S0, S1, BLOCK_COLS],
        strides=[S1 * BLOCK_COLS, BLOCK_COLS, 1],
        block_shape=[S0, S1, BLOCK_COLS],
    )
    out_desc.store([0, 0, 0], data)


@triton.jit
def gather_scatter_2d_index_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    y_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Round-trip: gather a 2-D grid of rows from ``in`` and scatter them
    back into ``out`` at the *same* 2-D-indexed rows.

    Exercises both the rank-2 ``x_offsets`` gather and scatter lowerings
    against one shared index grid::

        data[i, j, :]                       = in[idx[i, j], y : y+BLOCK_COLS]
        out[idx[i, j], y : y+BLOCK_COLS]     = data[i, j, :]

    With ``out`` pre-zeroed and unique indices, after the kernel
    ``out[idx[i, j], y:y+BLOCK_COLS] == in[idx[i, j], y:y+BLOCK_COLS]`` and
    every other row of ``out`` stays zero.
    """
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[S0, S1],
        strides=[S1, 1],
        block_shape=[S0, S1],
    )
    idx = idx_desc.load([0, 0])                    # tensor<S0 x S1 x i32>

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_COLS],
    )
    data = in_desc.gather(idx, y_offset)           # tensor<S0 x S1 x BLOCK_COLS>

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_COLS],
    )
    out_desc.scatter(data, idx, y_offset)


@triton.jit
def gather_2d_index_3d_block_kernel(
    in_ptr,
    out_ptr,
    idx_ptr,
    h_offset,
    CACHE_SZ: tl.constexpr,
    HEAD: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Gather a rank-2 (BLOCK_B x BLOCK_L) index grid out of a rank-3 source table,
    producing a rank-4 result tile.

    Composes both relaxations — a 2-D ``x_offsets`` index grid *and* a
    rank-3 descriptor block ``[1, BLOCK_H, D]``.  The block's leading dim is
    **1** by the gather contract (dim 0 is the page axis, fanned out one
    page per index — enforced at semantic.py and Ops.cpp); the source's
    full page-pool size is ``M`` (``shape[0]``), not the block's dim 0.
    For each grid cell ``(i, j)`` the kernel reads ``BLOCK_H`` rows starting at
    a non-zero ``h_offset`` of the inner ``HEAD`` axis, full width ``D``, of
    source page ``idx[i, j]``:

        result[i, j, c, d] = in[idx[i, j], h_offset + c, d]
        for i in [0,BLOCK_B), j in [0,BLOCK_L), c in [0,BLOCK_H), d in [0,D)

    Lowers to a single ``ktdp.construct_indirect_access_tile`` whose one
    indirect base dim carries a 2-D address into the index view, the
    ``h_offset`` direct dim lands on the inner axis, and the trailing
    ``D2`` is a direct full-extent dim — then a ``ktdp.load`` of the
    rank-4 ``[BLOCK_B, BLOCK_L, BLOCK_H, D]`` result tile.

    The result block is stored at the ``[0,0,0,0]`` corner of a
    ``[B, L, BLOCK_H, D2]`` output table (the index grid's full extent on
    dims 0-1, the source's inner dims on 2-3).
    """
    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[B, L],
        strides=[L, 1],
        block_shape=[BLOCK_B, BLOCK_L],
    )
    idx = idx_desc.load([0, 0])                    # tensor<S0 x S1 x i32>

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[CACHE_SZ, HEAD, D],
        strides=[HEAD * D, D, 1],
        block_shape=[1, BLOCK_H, D],
    )
    data = in_desc.gather(idx, h_offset)           # tensor<S0 x S1 x C1 x D2>

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[B, L, HEAD, D],
        strides=[L * HEAD * D, HEAD * D, D, 1],
        block_shape=[BLOCK_B, BLOCK_L, BLOCK_H, D],
    )
    out_desc.store([0, 0, 0, 0], data)
