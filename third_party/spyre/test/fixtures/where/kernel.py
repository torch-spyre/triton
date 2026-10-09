"""``tl.where`` kernels: a comparison, a select, and both with a spill.

Four ``@triton.jit`` functions, all loop-free and single-tile so they reach the
device tier:

- :func:`compare_1d_device` -- ``(x OP y).to(dtype)``, the comparison alone,
  writing 1.0 where it holds and 0.0 where it does not to a buffer.
- :func:`select_1d_device`  -- the select alone, over a mask the host wrote.
- :func:`where_1d_device`   -- both in one kernel, with the mask spilled to its
  own buffer between them.
- :func:`where_scalar_stick_device` -- :func:`where_1d_device` with ``y`` one
  scalar, read from a one-stick buffer the host filled.

``OP`` selects one of the six comparisons in :data:`COMPARISONS` at compile
time. Fixture inputs contain no NaNs.

:func:`where_1d_device` spills the mask rather than keeping it in registers
because a value with no descriptor has no ``tt.spyre_tensor_layout``.
``RewriteDescriptorLayoutGeneric`` leaves such a value logical while its
neighbours become stick-shaped, and bridges the two with a linearizing operand
map, ``(d0, d1) -> (d0 * S + d1)``. That map is not a permutation, and the
scheduler's elementwise fusion requires one, so the kernel is refused. Storing
the mask and loading it back gives it a descriptor, and every indexing map stays
the identity.

The select's condition is written ``m != 0``, not ``m > 0``. ``spyreop.select``
takes its true arm wherever the condition is non-zero, so a ``!= 0`` in front of
it is redundant and folds away, leaving one intrinsic for the pair. ``> 0`` does
not fold: it disagrees with "non-zero" on a negative lane, and a mask read back
from memory cannot be shown non-negative. Written ``> 0`` the kernel still
compiles, as two intrinsics.

The zero is ``tl.zeros`` at the mask's own dtype rather than a literal ``0.0``.
A bare Python float is fp32, so comparing an fp16 mask against it widens the
mask with ``arith.extf`` and compares at fp32.

No kernel takes a ``DTYPE`` parameter: the element type is fixed by the pointer
arguments, and ``meta.py`` sweeps it through the signature.
"""

import triton
import triton.language as tl


# Comparison names accepted by OP; meta.py supplies matching NumPy operators.
COMPARISONS = {
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "le": "<=",
    "lt": "<",
}


@triton.jit
def _compare(x, y, OP: tl.constexpr):
    """``x OP y`` for an ``OP`` named in :data:`COMPARISONS`.

    Resolved at compile time, so each specialization carries exactly one
    ``arith.cmpf``. An unknown name fails the compile rather than falling
    through to some default comparison.
    """
    if OP == "gt":
        result = x > y
    elif OP == "ge":
        result = x >= y
    elif OP == "eq":
        result = x == y
    elif OP == "ne":
        result = x != y
    elif OP == "le":
        result = x <= y
    elif OP == "lt":
        result = x < y
    else:
        tl.static_assert(False, "OP must be one of gt, ge, eq, ne, le, lt")
    return result


@triton.jit
def compare_1d_device(
    x_ptr,
    y_ptr,
    mask_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """``mask = (x OP y)`` over one tile: 1.0 where true and 0.0 where false,
    in the compared width.

    The device has no boolean type, so ``spyreop.compare`` answers with a number.
    The ``.to()`` is absorbed into that intrinsic, and the pair emits one op.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    mask_desc = tl.make_tensor_descriptor(
        mask_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(y_desc, LAYOUT)
    tl.spyre_tensor_layout(mask_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    y = y_desc.load([offset])
    mask_desc.store([offset], _compare(x, y, OP).to(x.dtype))


@triton.jit
def select_1d_device(
    mask_ptr,
    p_ptr,
    q_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    """``out = where(mask != 0, p, q)`` over a mask the host supplied.

    No comparison and no mask round trip precede the select, so a failure here
    is ``spyreop.select``'s. The ``!= 0`` folds away, leaving one intrinsic.
    """
    pid = tl.program_id(0)

    mask_desc = tl.make_tensor_descriptor(
        mask_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    p_desc = tl.make_tensor_descriptor(
        p_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(mask_desc, LAYOUT)
    tl.spyre_tensor_layout(p_desc, LAYOUT)
    tl.spyre_tensor_layout(q_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    mask = mask_desc.load([offset])
    p = p_desc.load([offset])
    q = q_desc.load([offset])
    zero = tl.zeros([BLOCK_SIZE], dtype=mask.dtype)
    out_desc.store([offset], tl.where(mask != zero, p, q))


@triton.jit
def where_1d_device(
    x_ptr,
    y_ptr,
    p_ptr,
    q_ptr,
    mask_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """``out = where(x OP y, p, q)`` in one kernel, with the mask spilled.

    ``mask_ptr`` is scratch rather than data: the module docstring says why the
    mask cannot stay in registers. The branches ``p``/``q`` are separate from the
    condition ``x``/``y`` so that a wrong-branch select is a wrong value; with
    ``where(x > y, x, y)`` a kernel returning ``max(x, y)`` would pass.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    p_desc = tl.make_tensor_descriptor(
        p_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    mask_desc = tl.make_tensor_descriptor(
        mask_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(y_desc, LAYOUT)
    tl.spyre_tensor_layout(p_desc, LAYOUT)
    tl.spyre_tensor_layout(q_desc, LAYOUT)
    tl.spyre_tensor_layout(mask_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE

    # Stage 1: the comparison, stored to its own buffer.
    x = x_desc.load([offset])
    y = y_desc.load([offset])
    mask_desc.store([offset], _compare(x, y, OP).to(x.dtype))

    # Stage 2: the select, over the mask read back. Reloading is what gives the
    # intermediate a descriptor, and with it a layout.
    mask = mask_desc.load([offset])
    p = p_desc.load([offset])
    q = q_desc.load([offset])
    zero = tl.zeros([BLOCK_SIZE], dtype=mask.dtype)
    out_desc.store([offset], tl.where(mask != zero, p, q))


@triton.jit
def where_scalar_stick_device(
    x_ptr,
    scalar_ptr,
    p_ptr,
    q_ptr,
    mask_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    STICK: tl.constexpr,
    LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """``out = where(x OP scalar, p, q)``, the scalar read from memory.

    ``x``, ``p``, ``q``, ``mask`` and ``out`` are ``[ROWS, STICK]``, one tile.
    ``STICK`` is the stick width at the pointers' dtype (64 lanes at fp16, 32 at
    fp32), so each row is one stick. ``scalar_ptr`` is ONE stick, ``[1, STICK]``,
    holding the scalar in every lane. The host fills it.

    The scalar comes from memory, not from ``tl.full``, because of what
    ``dbo-opt`` accepts. A constant tensor operand is folded by upstream
    ``FoldScalarOrSplatConstant`` (inside ``FuseComputeAndDataMovement``) into a
    scalar captured by the compare's body. The device's vector compare then
    finds an ``f16`` where it requires a vector, and ``dbo-opt`` refuses it.
    Loaded from a stick, the scalar is an ordinary input of the compare. The
    broadcast over ``ROWS`` ends up in that input's indexing map as
    ``(d0, d1, d2) -> (d0, 0, d2)``, which reads the same stick for every row
    and walks its lanes. That is the form torch-spyre's KTIR emitter produces
    for a scalar operand.

    The kernel is 2-D because the broadcast needs a row axis to repeat over. In
    1-D, a ``[STICK]`` operand has no shape that broadcasts to the tile.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[ROWS, STICK], strides=[STICK, 1], block_shape=[ROWS, STICK],
    )
    scalar_desc = tl.make_tensor_descriptor(
        scalar_ptr, shape=[1, STICK], strides=[STICK, 1], block_shape=[1, STICK],
    )
    p_desc = tl.make_tensor_descriptor(
        p_ptr, shape=[ROWS, STICK], strides=[STICK, 1], block_shape=[ROWS, STICK],
    )
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[ROWS, STICK], strides=[STICK, 1], block_shape=[ROWS, STICK],
    )
    mask_desc = tl.make_tensor_descriptor(
        mask_ptr, shape=[ROWS, STICK], strides=[STICK, 1], block_shape=[ROWS, STICK],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[ROWS, STICK], strides=[STICK, 1], block_shape=[ROWS, STICK],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(scalar_desc, LAYOUT)
    tl.spyre_tensor_layout(p_desc, LAYOUT)
    tl.spyre_tensor_layout(q_desc, LAYOUT)
    tl.spyre_tensor_layout(mask_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    # Stage 1: the comparison against the broadcast stick, stored to its own
    # buffer, as in where_1d_device.
    x = x_desc.load([0, 0])
    scalar = tl.broadcast_to(scalar_desc.load([0, 0]), [ROWS, STICK])
    mask_desc.store([0, 0], _compare(x, scalar, OP).to(x.dtype))

    # Stage 2: the select, over the mask read back.
    mask = mask_desc.load([0, 0])
    p = p_desc.load([0, 0])
    q = q_desc.load([0, 0])
    zero = tl.zeros([ROWS, STICK], dtype=mask.dtype)
    out_desc.store([0, 0], tl.where(mask != zero, p, q))
