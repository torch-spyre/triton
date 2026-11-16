"""Transpose / permute kernels on stick-tiled tensors in their device layout.

Each kernel loads one whole tensor through a descriptor, permutes it with
``tl.trans`` / ``tl.permute``, and stores the whole result. Nothing else is
computed, so the kernel is a pure data-movement copy. ``X_LAYOUT`` and
``Y_LAYOUT`` give each descriptor's physical layout as
``tl.spyre_tensor_layout`` takes it: the default layout of a tensor of that
rank (its last dim cut into sticks), so a tensor moved to the device with
``.to("spyre")`` is already laid out the way the kernel declares.
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
def permute3d_stick(x_ptr, y_ptr,
                    d0: tl.constexpr, d1: tl.constexpr, d2: tl.constexpr,
                    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                    output_d0: tl.constexpr, output_d1: tl.constexpr,
                    output_d2: tl.constexpr,
                    X_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = x.permute(P0, P1, P2)`` for ``x`` of logical shape ``[d0, d1, d2]``.

    ``[output_d0, output_d1, output_d2]`` is ``y``'s shape: ``[d0, d1, d2]``
    taken in the order ``(P0, P1, P2)``. It is a separate argument because a
    descriptor's shape must be a compile-time list.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[d0, d1, d2],
        strides=[d1 * d2, d2, 1],
        block_shape=[d0, d1, d2])
    y_desc = tl.make_tensor_descriptor(
        y_ptr,
        shape=[output_d0, output_d1, output_d2],
        strides=[output_d1 * output_d2, output_d2, 1],
        block_shape=[output_d0, output_d1, output_d2])
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    x = x_desc.load([0, 0, 0])
    y_desc.store([0, 0, 0], tl.permute(x, (P0, P1, P2)))


@triton.jit
def permute4d_stick(x_ptr, y_ptr,
                    d0: tl.constexpr, d1: tl.constexpr,
                    d2: tl.constexpr, d3: tl.constexpr,
                    P0: tl.constexpr, P1: tl.constexpr,
                    P2: tl.constexpr, P3: tl.constexpr,
                    output_d0: tl.constexpr, output_d1: tl.constexpr,
                    output_d2: tl.constexpr, output_d3: tl.constexpr,
                    X_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = x.permute(P0, P1, P2, P3)`` for ``x`` of shape ``[d0, d1, d2, d3]``.

    ``[output_d0, .., output_d3]`` is ``y``'s shape: ``[d0, d1, d2, d3]``
    taken in the order ``(P0, P1, P2, P3)``.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[d0, d1, d2, d3],
        strides=[d1 * d2 * d3, d2 * d3, d3, 1],
        block_shape=[d0, d1, d2, d3])
    y_desc = tl.make_tensor_descriptor(
        y_ptr,
        shape=[output_d0, output_d1, output_d2, output_d3],
        strides=[output_d1 * output_d2 * output_d3, output_d2 * output_d3,
                 output_d3, 1],
        block_shape=[output_d0, output_d1, output_d2, output_d3])
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    x = x_desc.load([0, 0, 0, 0])
    y_desc.store([0, 0, 0, 0], tl.permute(x, (P0, P1, P2, P3)))


@triton.jit
def permute5d_stick(x_ptr, y_ptr,
                    d0: tl.constexpr, d1: tl.constexpr,
                    d2: tl.constexpr, d3: tl.constexpr, d4: tl.constexpr,
                    P0: tl.constexpr, P1: tl.constexpr,
                    P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr,
                    output_d0: tl.constexpr, output_d1: tl.constexpr,
                    output_d2: tl.constexpr, output_d3: tl.constexpr,
                    output_d4: tl.constexpr,
                    X_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = x.permute(P0, ..., P4)`` for ``x`` of shape ``[d0, ..., d4]``.

    ``[output_d0, .., output_d4]`` is ``y``'s shape: ``[d0, .., d4]`` taken
    in the order ``(P0, ..., P4)``.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[d0, d1, d2, d3, d4],
        strides=[d1 * d2 * d3 * d4, d2 * d3 * d4, d3 * d4, d4, 1],
        block_shape=[d0, d1, d2, d3, d4])
    y_desc = tl.make_tensor_descriptor(
        y_ptr,
        shape=[output_d0, output_d1, output_d2, output_d3, output_d4],
        strides=[output_d1 * output_d2 * output_d3 * output_d4,
                 output_d2 * output_d3 * output_d4,
                 output_d3 * output_d4, output_d4, 1],
        block_shape=[output_d0, output_d1, output_d2, output_d3, output_d4])
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    x = x_desc.load([0, 0, 0, 0, 0])
    y_desc.store([0, 0, 0, 0, 0], tl.permute(x, (P0, P1, P2, P3, P4)))
