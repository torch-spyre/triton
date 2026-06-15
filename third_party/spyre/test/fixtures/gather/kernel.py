"""Gather kernels: indirect row-indexed load via tl.descriptor_gather.

This is the core pattern behind paged KV-cache attention and embedding
lookups: given a 1D index tensor of row positions, read those rows from
a 2D source matrix into a contiguous output tile.

Two ``@triton.jit`` functions:

- :func:`gather_kernel`    — single-program: one kernel invocation
                              consumes the whole index array and writes
                              the whole output tile.
- :func:`gather_2d_kernel` — multi-program: tiled across a 2D core
                              grid, each ``(pid_m, pid_n)`` produces a
                              ``BLOCK_ROWS x BLOCK_COLS`` output tile.
                              Adds a ``BLOCK_ROWS`` constexpr; gathers
                              the full row width by column-tiling
                              instead of taking a fixed slice.
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
def gather_kernel_spyre(
    in_ptr,
    out_ptr,
    idx_ptr,
    y_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_INDICES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
):
    """Spyre physical-layout variant of gather_kernel.

    Gathers K_INDICES full rows of the [M, N] source into a [K_INDICES, N]
    output, tiling the column dim into N // BLOCK_COLS chunks. in_desc and
    out_desc are annotated with Spyre stick-tiling layouts via
    tl.spyre_tensor_layout; idx_desc is not (index arrays have no stick layout).

    The inner col_stick loop is what exercises the multi-stick rewrite when
    BLOCK_COLS spans more than one stick: RewriteDescriptorLayout rescales the
    loop to stick granularity and the out_desc / in_desc tiles share that loop.

    IN_LAYOUT  — stick-tiling for in_ptr's full [M, N] extent.
    OUT_LAYOUT — stick-tiling for out_ptr's full [K_INDICES, N] extent.
    """

    pid_m = tl.program_id(0)
    grid_m = tl.num_programs(0)

    m_blocks = tl.cdiv(K_INDICES, BLOCK_ROWS)
    n_blocks = tl.cdiv(N, BLOCK_COLS)
    rows_per_core = tl.cdiv(m_blocks, grid_m)

    idx_desc = tl.make_tensor_descriptor(
        idx_ptr,
        shape=[K_INDICES],
        strides=[1],
        block_shape=[BLOCK_ROWS],
    )

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[1, BLOCK_COLS],
    )
    if IN_LAYOUT is not None and IN_LAYOUT != 0:
        tl.spyre_tensor_layout(in_desc, IN_LAYOUT)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[K_INDICES, N],
        strides=[N, 1],
        block_shape=[BLOCK_ROWS, BLOCK_COLS],
    )
    if OUT_LAYOUT is not None and OUT_LAYOUT != 0:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    m_start = pid_m * rows_per_core

    for m_sub in range(0, rows_per_core):
      for col_stick in range(n_blocks):
          m_block = m_start + m_sub
          offset_m = m_block * BLOCK_ROWS
          col_offset = col_stick * BLOCK_COLS

          idx = idx_desc.load([offset_m])
          result = in_desc.gather(idx, col_offset)
          out_desc.store([offset_m, col_offset], result)

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
