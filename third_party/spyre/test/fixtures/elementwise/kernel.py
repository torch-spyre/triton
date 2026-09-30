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

No-grid kernels — one tile, no distribution loop at all:
- :func:`elementwise_1d_device` — 1D, single tile; the only variant here
  that dbo-opt can lower all the way to a binary.
- :func:`dag_buffers_1d_device` — 1D, single tile, unary, and several
  computes rather than one, with every intermediate written to HBM and read
  back through a tensor descriptor the author declared. The computes form a
  DAG rather than a chain.
- :func:`chain_pooled_1d_device`, :func:`chain3_pooled_1d_device`,
  :func:`chain3_pooled2_1d_device`, :func:`dag_pooled_1d_device` — the same
  idea with all the intermediates in regions of **one** scratch buffer
  instead of one buffer each.

Scalar-load variant of the 1D kernel — same idea, one axis:
- :func:`elementwise_1d_scalar_dim` — 1D, but `n_elements` is a scalar read
  from memory rather than a kernel argument. KTIR-structural only for now.

No kernel here takes a ``DTYPE`` parameter, though several of their fixtures
sweep dtype. There would be nothing to do with one: the element type is fixed by
the pointer arguments and ``tl.make_tensor_descriptor`` takes it from there, so
the same source text compiles at fp16, fp32 and i32. Nor is it needed to tell
the compiled variants apart — the pointer types in the signature already give
them distinct hashes.

Four of these kernels did carry such a parameter, unread, until the fixture
framework stopped requiring it. Sweeping a dtype needs ``DTYPE`` in ``params``
for the factory hooks to read, and a ``params`` entry that was not a constexpr
used to reach ``run_cpu`` as a runtime scalar and be rejected as an unknown
kwarg — while a constexpr with no matching parameter is rejected by
``ASTSource`` (``ValueError: 'DTYPE' is not in list``). Taking the runtime
scalars from the variant's signature instead broke that chain: a param can now
drive a fixture without being an argument of anything.
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


@triton.jit
def elementwise_2d_device(
    x_ptr,
    y_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    X_LAYOUT: tl.constexpr,
    Y_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """2D elementwise op over a single tile, no distribution loop.

    Each of M×N tiles is the whole tensor; no tl.program_id loop, no
    tl.num_programs. dbo-opt accepts this because there is no scf.for
    for it to reject. Stick-on-N layouts physicalize the rank-2 tile
    to rank-3.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    if X_LAYOUT is not None:
        tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    if Y_LAYOUT is not None:
        tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    if OUT_LAYOUT is not None:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    offset_m = pid * BLOCK_M
    x = x_desc.load([offset_m, 0])
    y = y_desc.load([offset_m, 0])
    if OP == "add":
        result = x + y
    elif OP == "sub":
        result = x - y
    elif OP == "mul":
        result = x * y
    else:
        result = x / y
    out_desc.store([offset_m, 0], result)


# ---------------------------------------------------------------------------
# Author-declared buffers: every intermediate makes its own HBM round-trip.
#
# The kernels above compute one value from their loads and store it. The ones
# below compute several, and each intermediate is a *buffer the author declared*:
# a pointer argument with its own ``tl.make_tensor_descriptor``, its own
# ``tl.spyre_tensor_layout``, an explicit ``store`` and an explicit ``load``. No
# compute result is handed to another compute as a value; every one goes to HBM
# and comes back.
# ---------------------------------------------------------------------------


@triton.jit
def dag_buffers_1d_device(
    x_ptr,
    e_ptr,
    s_ptr,
    m_ptr,
    r_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = exp(x) * sqrt(exp(x)) + sqrt(x)``, one declared buffer per value.

    Four intermediates -- ``e = exp(x)``, ``s = sqrt(e)``, ``m = e * s`` and
    ``r = sqrt(x)`` -- each with its own pointer argument, descriptor, layout
    annotation, store and load. Loop-free and one tile per core.

    The computes form a DAG, not a chain, which is the point of choosing this
    shape. The interesting value is ``e``: it is read by the ``sqrt`` and by the
    multiply, so it becomes **one store and two loads of the same descriptor** --
    a value consumed by two later groups is one buffer read twice, not two
    buffers. ``x`` is likewise loaded twice, once for the ``exp`` and once for the
    other ``sqrt``, so no single load result is shared across two compute groups.
    That is the load-per-consumer rule, and a chain would not exercise it.

    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    e_desc = tl.make_tensor_descriptor(
        e_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    s_desc = tl.make_tensor_descriptor(
        s_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    m_desc = tl.make_tensor_descriptor(
        m_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    r_desc = tl.make_tensor_descriptor(
        r_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(e_desc, LAYOUT)
    tl.spyre_tensor_layout(s_desc, LAYOUT)
    tl.spyre_tensor_layout(m_desc, LAYOUT)
    tl.spyre_tensor_layout(r_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE

    # e = exp(x)
    x = x_desc.load([offset])
    e_desc.store([offset], tl.exp(x))

    # s = sqrt(e) -- first read of e
    e0 = e_desc.load([offset])
    s_desc.store([offset], tl.sqrt(e0))

    # m = e * s -- second read of e, joined with s
    e1 = e_desc.load([offset])
    s = s_desc.load([offset])
    m_desc.store([offset], e1 * s)

    # r = sqrt(x) -- second read of x
    x2 = x_desc.load([offset])
    r_desc.store([offset], tl.sqrt(x2))

    # out = m + r
    m = m_desc.load([offset])
    r = r_desc.load([offset])
    out_desc.store([offset], m + r)


# ---------------------------------------------------------------------------
# Pooled counterparts: one scratch pointer for all the intermediates.
#
# :func:`dag_buffers_1d_device` takes one pointer per intermediate, which is the
# plainest way to write it and the worst for footprint -- the launcher allocates
# every one, and its four are four whole-tensor allocations for values that are
# dead almost immediately. Here the author takes a single ``pool_ptr`` and puts
# every intermediate in a region of it.
# ---------------------------------------------------------------------------


@triton.jit
def chain_pooled_1d_device(
    x_ptr,
    pool_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = sqrt(exp(x))`` with the one intermediate in a pooled region.

    Liveness: one intermediate, ``exp(x)``, live from its store to the ``sqrt``'s
    load. Nothing else competes, so **one region** -- ``pool_ptr`` is used whole,
    and the pool is not carved at all. It exists as the floor of the family: if the
    pooled arm behaves differently from the per-intermediate arm here, the
    difference is the pooling itself and not the region reuse.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    r0_desc = tl.make_tensor_descriptor(
        pool_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(r0_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    r0_desc.store([offset], tl.exp(x))
    t = r0_desc.load([offset])
    out_desc.store([offset], tl.sqrt(t))


@triton.jit
def chain3_pooled_1d_device(
    x_ptr,
    pool_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = exp(sqrt(exp(x)))`` with both intermediates in **one** region.

    The interesting one. Three computes, so two intermediates::

        t0 = exp(x)     live from its store to the sqrt's load
        t1 = sqrt(t0)   live from its store to the final exp's load

    ``t0``'s last read is the ``sqrt`` that produces ``t1``, so the two live ranges
    are disjoint at the statement level and **one region** is the minimum an author
    can justify. That makes the middle compute ``load R -> sqrt -> store R``: a
    read and a write of the same region inside one schedule.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    r0_desc = tl.make_tensor_descriptor(
        pool_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(r0_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    r0_desc.store([offset], tl.exp(x))          # t0 -> R0
    t0 = r0_desc.load([offset])
    r0_desc.store([offset], tl.sqrt(t0))        # t1 -> R0, t0 now dead
    t1 = r0_desc.load([offset])
    out_desc.store([offset], tl.exp(t1))


@triton.jit
def chain3_pooled2_1d_device(
    x_ptr,
    pool_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = exp(sqrt(exp(x)))`` pooled into **two** regions, not one.

    The control for :func:`chain3_pooled_1d_device`. Same pool, same descriptor,
    but ``t0`` and ``t1`` get a region each, so no schedule reads and writes the
    same one. If the one-region variant misbehaves and this does not, the cause is
    the in-place reuse and not the pooling; if both behave the same, the reuse is
    free.

    The regions are **block offsets into one whole-pool descriptor**, not separate
    descriptors over ``pool_ptr + OFF``. That is forced, not preferred -- see the
    section comment above for the ``tt.addptr`` refusal.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    pool_desc = tl.make_tensor_descriptor(
        pool_ptr, shape=[2 * n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(pool_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    r0 = offset                    # region 0 base for this core
    r1 = n_elements + offset       # region 1 base for this core

    x = x_desc.load([offset])
    pool_desc.store([r0], tl.exp(x))            # t0 -> R0
    t0 = pool_desc.load([r0])
    pool_desc.store([r1], tl.sqrt(t0))          # t1 -> R1
    t1 = pool_desc.load([r1])
    out_desc.store([offset], tl.exp(t1))


@triton.jit
def dag_pooled_1d_device(
    x_ptr,
    pool_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = exp(x) * sqrt(exp(x)) + sqrt(x)`` with four intermediates in three
    pooled regions.

    Liveness, over the statement order of :func:`dag_buffers_1d_device`. Writing
    each intermediate's range as [statement that stores it, statement that last
    reads it]::

        1  store e          e = exp(x)
        2  load e, store s  s = sqrt(e)
        3  load e, load s, store m    m = e * s
        4  store r          r = sqrt(x)
        5  load m, load r, store out  out = m + r

        e : [1, 3]     s : [2, 3]     m : [3, 5]     r : [4, 5]

    ``e`` and ``s`` overlap on [2, 3], so they cannot share. ``m`` is stored at 3,
    where both ``e`` and ``s`` are still being read, so it cannot share with either
    -- ``{e, s, m}`` all contain statement 3 and that is a clique of three. Three
    is therefore the minimum, and it is achievable: ``r`` is stored at 4, after
    ``e``'s last read at 3, so ``r`` reuses ``e``'s region.

        R0 : e, then r      R1 : s      R2 : m

    Four intermediates, three regions, and one of them serving two values with
    disjoint ranges. The reuse here is *across* schedules -- ``e``'s last reader
    and ``r``'s writer are different computes -- which is a weaker claim than
    :func:`chain3_pooled_1d_device`'s reuse inside one schedule, and the two are
    kept apart so a failure names one of them.

    Two would need ``m`` to overwrite ``e`` or ``s`` in the very compute that reads
    them, which is that in-place case; it is deliberately not folded in here.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    pool_desc = tl.make_tensor_descriptor(
        pool_ptr, shape=[3 * n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(pool_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    r0 = offset                        # e, then r
    r1 = n_elements + offset           # s
    r2 = 2 * n_elements + offset       # m

    # 1. e = exp(x) -> R0
    x = x_desc.load([offset])
    pool_desc.store([r0], tl.exp(x))

    # 2. s = sqrt(e) -> R1; first read of e
    e0 = pool_desc.load([r0])
    pool_desc.store([r1], tl.sqrt(e0))

    # 3. m = e * s -> R2; second and last read of e, so R0 is free after this
    e1 = pool_desc.load([r0])
    s = pool_desc.load([r1])
    pool_desc.store([r2], e1 * s)

    # 4. r = sqrt(x) -> R0, reusing e's region
    x2 = x_desc.load([offset])
    pool_desc.store([r0], tl.sqrt(x2))

    # 5. out = m + r
    m = pool_desc.load([r2])
    r = pool_desc.load([r0])
    out_desc.store([offset], m + r)
