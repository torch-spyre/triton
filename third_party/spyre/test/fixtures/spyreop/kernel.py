"""spyreop kernels: three single-tile, loop-free Level D kernels, and two for the
``tl.spyre_op`` intrinsics at Level A-C.

The three Level D kernels are split by arity/dtype -- all take ``LAYOUT`` and have no
``tl.program_id``-driven distribution loop, only the one shape dbo-opt can
lower all the way to a binary (every looped kernel outlines an ``scf.for``
from its program-id distribution, which dbo-opt's scheduler rejects):

- :func:`spyreop_1d_device` -- unary, fp32. Takes ``OP: tl.constexpr`` and
  dispatches among the unary scalar float ops ``LowerSpyreOps.cpp`` converts
  unconditionally: ``sqrt``, ``rsqrt``, ``exp`` (``math.sqrt``/``math.rsqrt``/
  ``math.exp`` -> ``spyreop.sqrt``/``spyreop.rsqrt``/``spyreop.exp``).
- :func:`spyreop_realdiv_1d_device` -- binary, fp32, one op (``arith.divf`` ->
  ``spyreop.realdiv``), so no ``OP`` dispatch is needed.
- :func:`spyreop_addmul_1d_device` -- binary, i32. Takes ``OP: tl.constexpr``
  and dispatches ``add``/``mul`` (``arith.addi``/``arith.muli`` inside the
  ``linalg.generic`` ``convert_elementwise_to_linalg`` scalarizes this into ->
  ``spyreop.addi32toi32``/``spyreop.muli32toi32``).

See ``meta.py``'s module docstring for why there is no looped/Level-A/B
counterpart to any of these kernels.

And two kernels for the ``tl.spyre_op`` intrinsics, which dispatch ``OP`` among
``gelu``, ``silu`` and ``sigmoid``:

- :func:`spyreop_intrinsic_1d` -- 1D, distributed across the grid.
- :func:`spyreop_intrinsic_2d` -- 2D, with optional stick-tiling layouts.

What ``ktir_cpu`` runs for them is each intrinsic's FALLBACK: the ``ktir``
artifact holds the backend's registered body, inlined and hinted, and only the
``spyrecode`` stage replaces it with the intrinsic.
"""

import triton
import triton.language as tl


@triton.jit
def spyreop_1d_device(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """Unary spyreop over exactly one tile, with no distribution loop.

    No ``tl.program_id``, no ``tl.num_programs``, no loop -- one
    ``BLOCK_SIZE``-wide tile that is the whole tensor. That absence is what
    lets dbo-opt schedule it onto a binary.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    if OP == "sqrt":
        result = tl.sqrt(x)
    elif OP == "rsqrt":
        result = tl.rsqrt(x)
    else:
        result = tl.exp(x)
    out_desc.store([offset], result)


@triton.jit
def spyreop_realdiv_1d_device(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``x / y`` over exactly one tile, with no distribution loop. See
    :func:`spyreop_1d_device` -- same no-loop shape, binary instead of unary.
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
    out_desc.store([offset], x / y)


@triton.jit
def spyreop_addmul_1d_device(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """``x + y`` or ``x * y`` over exactly one tile, with no distribution
    loop. Same no-loop shape as :func:`spyreop_1d_device` /
    :func:`spyreop_realdiv_1d_device`, binary with an ``OP`` dispatch limited
    to ``add``/``mul`` -- the two int patterns ``LowerSpyreOps.cpp`` matches
    (``sub``/``div`` have no int spyreop intrinsic).
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
    else:
        result = x * y
    out_desc.store([offset], result)


@triton.jit
def spyreop_intrinsic_1d(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,
):
    """``tl.spyre_op(OP, x)`` over a 1D vector, distributed across the grid."""
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    num_cores = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, num_cores)
    start = pid * blocks_per_core
    end = tl.minimum(start + blocks_per_core, num_blocks)
    for i in range(start, end):
        offset = i * BLOCK_SIZE
        x = x_desc.load([offset])
        if OP == "gelu":
            result = tl.spyre_op("gelu", x)
        elif OP == "silu":
            result = tl.spyre_op("silu", x)
        else:
            result = tl.spyre_op("sigmoid", x)
        out_desc.store([offset], result)


@triton.jit
def spyreop_intrinsic_2d(
    x_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    X_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """``tl.spyre_op(OP, x)`` over ``[M, N]``, M-tiles distributed across the
    grid. ``X_LAYOUT`` / ``OUT_LAYOUT`` are optional stick-tiling layouts; None
    lowers logically."""
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    if X_LAYOUT is not None:
        tl.spyre_tensor_layout(x_desc, X_LAYOUT)
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
            if OP == "gelu":
                result = tl.spyre_op("gelu", x)
            elif OP == "silu":
                result = tl.spyre_op("silu", x)
            else:
                result = tl.spyre_op("sigmoid", x)
            out_desc.store([offset_m, offset_n], result)
