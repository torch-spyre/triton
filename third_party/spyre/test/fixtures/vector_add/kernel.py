"""Vector-add kernels: 1D, 2D, and 3D elementwise add.

Five ``@triton.jit`` functions exercising tensor descriptors at increasing
dimensionality. Two grid styles:

1D-grid kernels (``tl.program_id(0)`` only) — each core loops over its
  share of tiles in the outermost dimension:
- :func:`add_kernel`    — 1D: ``shape=[n_elements]``
- :func:`add_kernel_2d` — 2D: ``shape=[M, N]``
- :func:`add_kernel_3d` — 3D: ``shape=[M, N, P]``

Multi-axis grid kernels — each axis of the grid maps to one tensor
  dimension; no manual distribution loop is needed for those axes:
- :func:`add_kernel_2d_grid` — 2D grid: pid_0 → M-tile, pid_1 → N-tile
- :func:`add_kernel_3d_grid` — 3D grid: pid_0 → M-tile, pid_1 → N-tile,
                                         pid_2 → P-tile
"""

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[n_elements],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr,
        shape=[n_elements],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[n_elements],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )

    # Each core loops over its chunk of the sequence. tl.num_programs(0)
    # reports the grid's axis-0 size — folded to a compile-time constant
    # by DistributeWork against SpyreOptions.grid.
    num_cores = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, num_cores)
    start = pid * blocks_per_core
    end = tl.minimum(start + blocks_per_core, num_blocks)
    for i in range(start, end):
        offset = i * BLOCK_SIZE
        x = x_desc.load([offset])
        y = y_desc.load([offset])
        out_desc.store([offset], x + y)



@triton.jit
def add_kernel_2d(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    X_LAYOUT: tl.constexpr,
    Y_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
):
    """2D elementwise add: out[M, N] = x[M, N] + y[M, N].

    ``X_LAYOUT`` / ``Y_LAYOUT`` / ``OUT_LAYOUT`` are optional Spyre stick-tiling
    layouts for the matching descriptor; pass None to lower logically.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    if X_LAYOUT is not None:
        tl.spyre_tensor_layout(x_desc, X_LAYOUT)

    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    if Y_LAYOUT is not None:
        tl.spyre_tensor_layout(y_desc, Y_LAYOUT)

    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    if OUT_LAYOUT is not None:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    num_cores = tl.num_programs(0)
    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores)
    m_start = pid * m_blocks_per_core
    m_end = tl.minimum(m_start + m_blocks_per_core, m_blocks)

    for m in range(m_start, m_end):
        for n in range(0, n_blocks):
            offset_m = m * BLOCK_M
            offset_n = n * BLOCK_N
            x = x_desc.load([offset_m, offset_n])
            y = y_desc.load([offset_m, offset_n])
            out_desc.store([offset_m, offset_n], x + y)


@triton.jit
def add_kernel_2d_grid(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D grid: pid_0 distributes M-tiles, pid_1 distributes N-tiles.

    Each grid axis loops over its assigned tiles via a distribution loop,
    replacing the 1D-grid outer loops from :func:`add_kernel_2d`.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_cores_m = tl.num_programs(0)
    num_cores_n = tl.num_programs(1)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores_m)
    n_blocks_per_core = tl.cdiv(n_blocks, num_cores_n)
    m_start = pid_m * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)
    n_start = pid_n * n_blocks_per_core
    n_end   = tl.minimum(n_start + n_blocks_per_core, n_blocks)

    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            x = x_desc.load([m * BLOCK_M, n * BLOCK_N])
            y = y_desc.load([m * BLOCK_M, n * BLOCK_N])
            out_desc.store([m * BLOCK_M, n * BLOCK_N], x + y)


@triton.jit
def add_kernel_3d(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    P,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)

    stride_m = N * P
    stride_n = P

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N, P], strides=[stride_m, stride_n, 1],
        block_shape=[BLOCK_M, BLOCK_N, BLOCK_P],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N, P], strides=[stride_m, stride_n, 1],
        block_shape=[BLOCK_M, BLOCK_N, BLOCK_P],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N, P], strides=[stride_m, stride_n, 1],
        block_shape=[BLOCK_M, BLOCK_N, BLOCK_P],
    )

    num_cores = tl.num_programs(0)
    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    p_blocks = tl.cdiv(P, BLOCK_P)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores)
    m_start = pid * m_blocks_per_core
    m_end = tl.minimum(m_start + m_blocks_per_core, m_blocks)

    for m in range(m_start, m_end):
        for n in range(0, n_blocks):
            for p in range(0, p_blocks):
                offset_m = m * BLOCK_M
                offset_n = n * BLOCK_N
                offset_p = p * BLOCK_P
                x = x_desc.load([offset_m, offset_n, offset_p])
                y = y_desc.load([offset_m, offset_n, offset_p])
                out_desc.store([offset_m, offset_n, offset_p], x + y)


@triton.jit
def add_kernel_3d_grid(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    P,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """3D grid: pid_0 distributes M-tiles, pid_1 N-tiles, pid_2 P-tiles.

    Each grid axis loops over its assigned tiles via a distribution loop,
    replacing the 1D-grid outer loops from :func:`add_kernel_3d`.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)
    num_cores_m = tl.num_programs(0)
    num_cores_n = tl.num_programs(1)
    num_cores_p = tl.num_programs(2)

    stride_m = N * P
    stride_n = P

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N, P], strides=[stride_m, stride_n, 1],
        block_shape=[BLOCK_M, BLOCK_N, BLOCK_P],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N, P], strides=[stride_m, stride_n, 1],
        block_shape=[BLOCK_M, BLOCK_N, BLOCK_P],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N, P], strides=[stride_m, stride_n, 1],
        block_shape=[BLOCK_M, BLOCK_N, BLOCK_P],
    )

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    p_blocks = tl.cdiv(P, BLOCK_P)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores_m)
    n_blocks_per_core = tl.cdiv(n_blocks, num_cores_n)
    p_blocks_per_core = tl.cdiv(p_blocks, num_cores_p)
    m_start = pid_m * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)
    n_start = pid_n * n_blocks_per_core
    n_end   = tl.minimum(n_start + n_blocks_per_core, n_blocks)
    p_start = pid_p * p_blocks_per_core
    p_end   = tl.minimum(p_start + p_blocks_per_core, p_blocks)

    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            for p in range(p_start, p_end):
                x = x_desc.load([m * BLOCK_M, n * BLOCK_N, p * BLOCK_P])
                y = y_desc.load([m * BLOCK_M, n * BLOCK_N, p * BLOCK_P])
                out_desc.store([m * BLOCK_M, n * BLOCK_N, p * BLOCK_P], x + y)
