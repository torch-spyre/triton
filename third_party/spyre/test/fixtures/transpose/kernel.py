"""Transpose / permute kernels on torch-spyre tensors in their device layout.

Each kernel loads one whole tensor through a descriptor, permutes it with
``tl.trans`` / ``tl.permute``, and stores the whole result. Nothing else is
computed, so the kernel is a pure data-movement copy. ``X_LAYOUT`` and
``Y_LAYOUT`` give each descriptor's physical layout as
``tl.spyre_tensor_layout`` takes it: the layout torch-spyre itself gives a
tensor of that rank (its last dim cut into sticks), so a tensor moved to the
device with ``.to("spyre")`` is already laid out the way the kernel declares.
"""

import triton
import triton.language as tl


@triton.jit
def trans2d_stick(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,
                  X_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = x.T`` for ``x`` of logical shape ``[M, N]``, ``y`` ``[N, M]``."""
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[M, N])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[N, M], strides=[M, 1],
                                       block_shape=[N, M])
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    x = x_desc.load([0, 0])
    y_desc.store([0, 0], tl.trans(x))


@triton.jit
def permute3d_stick(x_ptr, y_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                    OA: tl.constexpr, OB: tl.constexpr, OC: tl.constexpr,
                    X_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = x.permute(P0, P1, P2)`` for ``x`` of logical shape ``[A, B, C]``.

    ``[OA, OB, OC]`` is ``y``'s shape, ``[A, B, C]`` taken in the order
    ``(P0, P1, P2)``; it is a separate argument because a descriptor's shape
    must be a compile-time list.
    """
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[A, B, C], strides=[B * C, C, 1],
                                       block_shape=[A, B, C])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[OA, OB, OC],
                                       strides=[OB * OC, OC, 1],
                                       block_shape=[OA, OB, OC])
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    x = x_desc.load([0, 0, 0])
    y_desc.store([0, 0, 0], tl.permute(x, (P0, P1, P2)))
