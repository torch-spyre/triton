"""Transpose / permute kernels on stick-tiled tensors in their device layout.

The ``trans2d_stick`` / ``permute{3,4,5}d_stick`` kernels load one whole
tensor through a descriptor, permute it with ``tl.trans`` / ``tl.permute``, and
store the whole result: a pure data-movement copy. The
``add*_trans{2,3}d_stick`` kernels at the end fuse that transpose into an add,
so an elementwise op reads operands whose stick dims disagree. Each
``*_LAYOUT`` constexpr (``X_LAYOUT``, ``Y_LAYOUT``, ``A_LAYOUT``, ..) gives one
descriptor's physical layout as ``tl.spyre_tensor_layout`` takes it: the
default layout of a tensor of that rank (its last dim cut into sticks), so a
tensor moved to the device with ``.to("spyre")`` is already laid out the way
the kernel declares.
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



# ---------------------------------------------------------------------------
# A transpose fused into an add
#
# Each kernel below adds tensors whose layouts disagree: some operands are
# stored in the output's dim order and some in the transposed order, so the add
# reads at least one operand through a transpose. TRANS_* flags (constexpr
# bools) say which: True means that operand is stored transposed relative to
# the output -- ``[N, M]`` for an ``[M, N]`` output at rank 2, ``[d0, d2, d1]``
# for a ``[d0, d1, d2]`` output at rank 3 -- and the kernel transposes it back
# before adding. Each descriptor's *_LAYOUT is the default stick layout of its
# rank, as for the copies above.
#
# Each kernel builds its descriptors in its own body rather than through a
# shared @triton.jit helper: a helper that takes a pointer and builds the
# descriptor survives the ttir stage as a tt.call, which ConvertFunctions then
# rejects (its !tt.ptr argument is still in use).
# ---------------------------------------------------------------------------


@triton.jit
def add_trans2d_stick(a_ptr, b_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,
                      TRANS_A: tl.constexpr, TRANS_B: tl.constexpr,
                      A_LAYOUT: tl.constexpr, B_LAYOUT: tl.constexpr,
                      Y_LAYOUT: tl.constexpr):
    """``y = a' + b'`` for ``y`` of shape ``[M, N]``, where ``a'`` is ``a.T`` when
    ``TRANS_A`` is set (``a`` then ``[N, M]``) and ``a`` otherwise, and likewise
    ``b'``.

    ``(TRANS_A, TRANS_B) = (False, True)`` is ``a + b.T``; ``(True, False)`` is
    ``a.T + b``.
    """
    if TRANS_A:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[N, M], strides=[M, 1],
                                           block_shape=[N, M])
    else:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1],
                                           block_shape=[M, N])
    if TRANS_B:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, M], strides=[M, 1],
                                           block_shape=[N, M])
    else:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[M, N], strides=[N, 1],
                                           block_shape=[M, N])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[M, N])
    tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    tl.spyre_tensor_layout(b_desc, B_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    a = a_desc.load([0, 0])
    b = b_desc.load([0, 0])
    if TRANS_A:
        a = tl.trans(a)
    if TRANS_B:
        b = tl.trans(b)
    y_desc.store([0, 0], a + b)


@triton.jit
def add3_trans2d_stick(a_ptr, b_ptr, c_ptr, y_ptr,
                       M: tl.constexpr, N: tl.constexpr,
                       TRANS_A: tl.constexpr, TRANS_B: tl.constexpr,
                       TRANS_C: tl.constexpr,
                       A_LAYOUT: tl.constexpr, B_LAYOUT: tl.constexpr,
                       C_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = a' + b' + c'`` for ``y`` of shape ``[M, N]``; an operand whose
    ``TRANS_*`` flag is set is stored ``[N, M]`` and transposed before the add.

    ``(False, True, True)`` is ``a + b.T + c.T``.
    """
    if TRANS_A:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[N, M], strides=[M, 1],
                                           block_shape=[N, M])
    else:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1],
                                           block_shape=[M, N])
    if TRANS_B:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, M], strides=[M, 1],
                                           block_shape=[N, M])
    else:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[M, N], strides=[N, 1],
                                           block_shape=[M, N])
    if TRANS_C:
        c_desc = tl.make_tensor_descriptor(c_ptr, shape=[N, M], strides=[M, 1],
                                           block_shape=[N, M])
    else:
        c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1],
                                           block_shape=[M, N])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[M, N])
    tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    tl.spyre_tensor_layout(b_desc, B_LAYOUT)
    tl.spyre_tensor_layout(c_desc, C_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    a = a_desc.load([0, 0])
    b = b_desc.load([0, 0])
    c = c_desc.load([0, 0])
    if TRANS_A:
        a = tl.trans(a)
    if TRANS_B:
        b = tl.trans(b)
    if TRANS_C:
        c = tl.trans(c)
    y_desc.store([0, 0], a + b + c)


@triton.jit
def add_self_trans2d_stick(x_ptr, y_ptr, N: tl.constexpr,
                           X_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = x + x.T`` for a square ``x`` of shape ``[N, N]``.

    ``x`` is loaded once, and that one value is read both as loaded and
    transposed, unlike :func:`add_trans2d_stick`, which loads its operands separately.
    """
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[N, N], strides=[N, 1],
                                       block_shape=[N, N])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[N, N], strides=[N, 1],
                                       block_shape=[N, N])
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    x = x_desc.load([0, 0])
    y_desc.store([0, 0], x + tl.trans(x))


@triton.jit
def add_trans3d_stick(a_ptr, b_ptr, y_ptr,
                      d0: tl.constexpr, d1: tl.constexpr, d2: tl.constexpr,
                      TRANS_A: tl.constexpr, TRANS_B: tl.constexpr,
                      A_LAYOUT: tl.constexpr, B_LAYOUT: tl.constexpr,
                      Y_LAYOUT: tl.constexpr):
    """``y = a' + b'`` for ``y`` of shape ``[d0, d1, d2]``, where an operand whose
    ``TRANS_*`` flag is set is stored ``[d0, d2, d1]`` and has its last two dims
    swapped (``tl.permute(.., (0, 2, 1))``) before the add."""
    if TRANS_A:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[d0, d2, d1],
                                           strides=[d2 * d1, d1, 1],
                                           block_shape=[d0, d2, d1])
    else:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[d0, d1, d2],
                                           strides=[d1 * d2, d2, 1],
                                           block_shape=[d0, d1, d2])
    if TRANS_B:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[d0, d2, d1],
                                           strides=[d2 * d1, d1, 1],
                                           block_shape=[d0, d2, d1])
    else:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[d0, d1, d2],
                                           strides=[d1 * d2, d2, 1],
                                           block_shape=[d0, d1, d2])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[d0, d1, d2],
                                       strides=[d1 * d2, d2, 1],
                                       block_shape=[d0, d1, d2])
    tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    tl.spyre_tensor_layout(b_desc, B_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    a = a_desc.load([0, 0, 0])
    b = b_desc.load([0, 0, 0])
    if TRANS_A:
        a = tl.permute(a, (0, 2, 1))
    if TRANS_B:
        b = tl.permute(b, (0, 2, 1))
    y_desc.store([0, 0, 0], a + b)


@triton.jit
def add3_trans3d_stick(a_ptr, b_ptr, c_ptr, y_ptr,
                       d0: tl.constexpr, d1: tl.constexpr, d2: tl.constexpr,
                       TRANS_A: tl.constexpr, TRANS_B: tl.constexpr,
                       TRANS_C: tl.constexpr,
                       A_LAYOUT: tl.constexpr, B_LAYOUT: tl.constexpr,
                       C_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = a' + b' + c'`` for ``y`` of shape ``[d0, d1, d2]``; an operand whose
    ``TRANS_*`` flag is set is stored ``[d0, d2, d1]`` and has its last two dims
    swapped before the add."""
    if TRANS_A:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[d0, d2, d1],
                                           strides=[d2 * d1, d1, 1],
                                           block_shape=[d0, d2, d1])
    else:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[d0, d1, d2],
                                           strides=[d1 * d2, d2, 1],
                                           block_shape=[d0, d1, d2])
    if TRANS_B:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[d0, d2, d1],
                                           strides=[d2 * d1, d1, 1],
                                           block_shape=[d0, d2, d1])
    else:
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[d0, d1, d2],
                                           strides=[d1 * d2, d2, 1],
                                           block_shape=[d0, d1, d2])
    if TRANS_C:
        c_desc = tl.make_tensor_descriptor(c_ptr, shape=[d0, d2, d1],
                                           strides=[d2 * d1, d1, 1],
                                           block_shape=[d0, d2, d1])
    else:
        c_desc = tl.make_tensor_descriptor(c_ptr, shape=[d0, d1, d2],
                                           strides=[d1 * d2, d2, 1],
                                           block_shape=[d0, d1, d2])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[d0, d1, d2],
                                       strides=[d1 * d2, d2, 1],
                                       block_shape=[d0, d1, d2])
    tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    tl.spyre_tensor_layout(b_desc, B_LAYOUT)
    tl.spyre_tensor_layout(c_desc, C_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    a = a_desc.load([0, 0, 0])
    b = b_desc.load([0, 0, 0])
    c = c_desc.load([0, 0, 0])
    if TRANS_A:
        a = tl.permute(a, (0, 2, 1))
    if TRANS_B:
        b = tl.permute(b, (0, 2, 1))
    if TRANS_C:
        c = tl.permute(c, (0, 2, 1))
    y_desc.store([0, 0, 0], a + b + c)


@triton.jit
def add_self_trans3d_stick(x_ptr, y_ptr, d0: tl.constexpr, N: tl.constexpr,
                           X_LAYOUT: tl.constexpr, Y_LAYOUT: tl.constexpr):
    """``y = x + x.permute(0, 2, 1)`` for ``x`` of shape ``[d0, N, N]``, loaded
    once and read both as loaded and with its last two dims swapped."""
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[d0, N, N],
                                       strides=[N * N, N, 1],
                                       block_shape=[d0, N, N])
    y_desc = tl.make_tensor_descriptor(y_ptr, shape=[d0, N, N],
                                       strides=[N * N, N, 1],
                                       block_shape=[d0, N, N])
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(y_desc, Y_LAYOUT)
    x = x_desc.load([0, 0, 0])
    y_desc.store([0, 0, 0], x + tl.permute(x, (0, 2, 1)))
