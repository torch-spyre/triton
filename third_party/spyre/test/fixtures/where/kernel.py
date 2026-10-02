"""``tl.where`` kernels: a comparison, a select, and both with a spill.

Three ``@triton.jit`` functions, all loop-free and single-tile so they reach the
device tier:

- :func:`compare_1d_device` -- ``(x OP y).to(dtype)``, the comparison alone,
  writing its 1.0/0.0 answer to a buffer.
- :func:`select_1d_device`  -- the select alone, over a mask the host wrote.
- :func:`where_1d_device`   -- both in one kernel, with the mask spilled to its
  own buffer between them.

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
    """``mask = (x OP y)`` as 1.0/0.0 in the compared width, over one tile.

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
