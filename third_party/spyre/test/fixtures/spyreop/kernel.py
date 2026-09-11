"""spyreop kernels: single-tile, no distribution loop, Level D only.

Three kernels, split by arity/dtype -- all take ``LAYOUT`` and have no
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
