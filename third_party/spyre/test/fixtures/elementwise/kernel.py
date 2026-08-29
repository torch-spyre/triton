"""Elementwise kernels: 1D, 2D, and 3D shapes with OP dispatch.

Seven ``@triton.jit`` functions exercising tensor descriptors at increasing
dimensionality. Each takes ``OP: tl.constexpr`` and dispatches add/sub/mul/div.
Two grid styles:

1D-grid kernels (``tl.program_id(0)`` only) — each core loops over its
  share of tiles in the outermost dimension:
- :func:`elementwise_1d`    — 1D: ``shape=[n_elements]``
- :func:`elementwise_2d` — 2D: ``shape=[M, N]``
- :func:`elementwise_3d` — 3D: ``shape=[M, N, P]``

Multi-axis grid kernels — each axis of the grid maps to one tensor
  dimension; no manual distribution loop is needed for those axes:
- :func:`elementwise_2d_grid` — 2D grid: pid_0 → M-tile, pid_1 → N-tile
- :func:`elementwise_3d_grid` — 3D grid: pid_0 → M-tile, pid_1 → N-tile,
                                         pid_2 → P-tile
- :func:`elementwise_2d_scalar_dim` — 2D grid, but `M` is a scalar read
  from memory rather than a kernel argument; `N` is still chunked the
  same way as `M`. KTIR-structural only for now.

No-grid kernel — one tile, no distribution loop at all:
- :func:`elementwise_1d_device` — 1D, single tile; the only variant here
  that dbo-opt can lower all the way to a binary.

Scalar-load variant of the 1D kernel — same idea, one axis:
- :func:`elementwise_1d_scalar_dim` — 1D, but `n_elements` is a scalar read
  from memory rather than a kernel argument. KTIR-structural only for now.
"""

import triton
import triton.language as tl


@triton.jit
def elementwise_1d(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
    OP: tl.constexpr,
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
        if OP == "add":
            result = x + y
        elif OP == "sub":
            result = x - y
        elif OP == "mul":
            result = x * y
        else:
            result = x / y
        out_desc.store([offset], result)


@triton.jit
def elementwise_1d_scalar_dim(
    x_ptr,
    y_ptr,
    output_ptr,
    seqlen_ptr,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,
):
    """1D elementwise where `n_elements` is read from memory instead of passed directly.

    for now: KTIR-structural only, do not wire this into any end-to-end/DFIR test.
    """
    pid = tl.program_id(0)
    n_elements = tl.load(seqlen_ptr)

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

    num_cores = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, num_cores)
    start = pid * blocks_per_core
    end = tl.minimum(start + blocks_per_core, num_blocks)
    for i in range(start, end):
        offset = i * BLOCK_SIZE
        x = x_desc.load([offset])
        y = y_desc.load([offset])
        if OP == "add":
            result = x + y
        elif OP == "sub":
            result = x - y
        elif OP == "mul":
            result = x * y
        else:
            result = x / y
        out_desc.store([offset], result)


@triton.jit
def elementwise_2d(
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
    OP: tl.constexpr,
):
    """2D elementwise op: out[M, N] = x[M, N] OP y[M, N].

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
            if OP == "add":
                result = x + y
            elif OP == "sub":
                result = x - y
            elif OP == "mul":
                result = x * y
            else:
                result = x / y
            out_desc.store([offset_m, offset_n], result)


@triton.jit
def elementwise_2d_grid(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OP: tl.constexpr,
):
    """2D grid: pid_0 distributes M-tiles, pid_1 distributes N-tiles.

    Each grid axis loops over its assigned tiles via a distribution loop,
    replacing the 1D-grid outer loops from :func:`elementwise_2d`.
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
            if OP == "add":
                result = x + y
            elif OP == "sub":
                result = x - y
            elif OP == "mul":
                result = x * y
            else:
                result = x / y
            out_desc.store([m * BLOCK_M, n * BLOCK_N], result)


@triton.jit
def elementwise_2d_scalar_dim(
    x_ptr,
    y_ptr,
    output_ptr,
    seqlen_ptr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OP: tl.constexpr,
):
    """2D elementwise where `M` is read from memory instead of passed directly.

    for now: KTIR-structural only, do not wire this into any end-to-end/DFIR test.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_m = tl.num_programs(0)
    grid_n = tl.num_programs(1)
    M = tl.load(seqlen_ptr)

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
    m_blocks_per_core = tl.cdiv(m_blocks, grid_m)
    n_blocks_per_core = tl.cdiv(n_blocks, grid_n)
    m_start = pid_m * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)
    n_start = pid_n * n_blocks_per_core
    n_end   = tl.minimum(n_start + n_blocks_per_core, n_blocks)

    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            x = x_desc.load([m * BLOCK_M, n * BLOCK_N])
            y = y_desc.load([m * BLOCK_M, n * BLOCK_N])
            if OP == "add":
                result = x + y
            elif OP == "sub":
                result = x - y
            elif OP == "mul":
                result = x * y
            else:
                result = x / y
            out_desc.store([m * BLOCK_M, n * BLOCK_N], result)


@triton.jit
def elementwise_3d(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    P,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    OP: tl.constexpr,
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
                if OP == "add":
                    result = x + y
                elif OP == "sub":
                    result = x - y
                elif OP == "mul":
                    result = x * y
                else:
                    result = x / y
                out_desc.store([offset_m, offset_n, offset_p], result)


@triton.jit
def elementwise_3d_grid(
    x_ptr,
    y_ptr,
    output_ptr,
    M,
    N,
    P,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    OP: tl.constexpr,
):
    """3D grid: pid_0 distributes M-tiles, pid_1 N-tiles, pid_2 P-tiles.

    Each grid axis loops over its assigned tiles via a distribution loop,
    replacing the 1D-grid outer loops from :func:`elementwise_3d`.
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
                if OP == "add":
                    result = x + y
                elif OP == "sub":
                    result = x - y
                elif OP == "mul":
                    result = x * y
                else:
                    result = x / y
                out_desc.store([m * BLOCK_M, n * BLOCK_N, p * BLOCK_P], result)


@triton.jit
def elementwise_1d_device(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """Elementwise op over exactly one tile, with no distribution loop.

    The odd one out in this file: no ``tl.program_id``, no ``tl.num_programs``,
    no loop -- one ``BLOCK_SIZE``-wide tile that is the whole tensor. Every
    other kernel here carves work across the grid with an ``scf.for`` over the
    program id, and dbo-opt rejects the loop it outlines from that, so this is
    the only variant in the suite that reaches a Spyre *binary* rather than
    stopping at KTIR.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(y_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)
    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    y = y_desc.load([offset])
    if OP == "add":
        result = x + y
    elif OP == "sub":
        result = x - y
    elif OP == "mul":
        result = x * y
    else:
        result = x / y
    out_desc.store([offset], result)
