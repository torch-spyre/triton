import triton
import triton.language as tl


@triton.jit
def layernorm_single_tile(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
):
    """Row-wise LayerNorm over an ``[M, N]`` input, whole tensor in one tile.

    ``out[m, :] = (x[m, :] - mean[m]) / sqrt(var[m] + EPS) * weight + bias``

    Written in ordinary ``tl`` ops: two reductions (the mean, then the variance
    of the centred value), and the elementwise arithmetic around them. Nothing
    here names a Spyre intrinsic, so this is the slow path -- what a kernel
    compiles to with no ``tl.spyre_intrinsic`` in it, and the baseline any
    request-based version is compared with.

    ``weight`` and ``bias`` are ``[N]`` vectors, broadcast down the rows.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr, shape=[N], strides=[1], block_shape=[N],
    )
    bias_desc = tl.make_tensor_descriptor(
        bias_ptr, shape=[N], strides=[1], block_shape=[N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )

    x = x_desc.load([0, 0])
    weight = weight_desc.load([0])
    bias = bias_desc.load([0])

    mean = tl.sum(x, 1) / N
    centred = x - mean[:, None]
    var = tl.sum(centred * centred, 1) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    out = centred * rstd[:, None] * weight[None, :] + bias[None, :]
    out_desc.store([0, 0], out)


@triton.jit
def layernorm_mean(
    x_ptr,
    n_ptr,
    pool_ptr,
    mean_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TILE_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """The mean of each row of an ``[M, N]`` input: two compute groups, one pool.

    ``x`` is logical ``[M, N]``; the stick layout on N makes it physical
    ``[N / S, M, S]``, so the one reduction over N folds two physical dims.

        1  scale   x / N              into the pool, the same shape as x
        2  sum     sum over N         ``[M]``, stored across a stick into ``mean``

    The input is divided by the count before it is summed, so the sum is already
    the mean. The count is read from ``n_ptr``, one row of the input's width
    filled with it, and broadcast down the rows, as ``where``'s scalar is.
    """
    x_d = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
    pool_d = tl.make_tensor_descriptor(pool_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
    mean_w = tl.make_tensor_descriptor(mean_ptr, shape=[M], strides=[1], block_shape=[M])
    n_d = tl.make_tensor_descriptor(n_ptr, shape=[1, N], strides=[N, 1], block_shape=[1, N])
    tl.spyre_tensor_layout(x_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(n_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(pool_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(mean_w, STAT_LAYOUT)

    # G1: x divided by the count.
    n = tl.broadcast_to(n_d.load([0, 0]), [M, N])
    pool_d.store([0, 0], tl.fdiv(x_d.load([0, 0]), n))

    # G2: the sum over N, stored across a stick.
    mean_w.store([0], tl.sum(pool_d.load([0, 0]), axis=1))


@triton.jit
def layernorm_norm(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_ptr,
    eps_ptr,
    pool_ptr,
    mean_ptr,
    mean2_ptr,
    msq_ptr,
    sq_ptr,
    var_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    TILE_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """A whole LayerNorm per row of an ``[M, N]`` input, chained through HBM:
    ``out = (x - mean) / sqrt(var + eps) * weight + bias``. Nine compute groups.

    ``x`` is logical ``[M, N]``; the stick layout on N makes it physical
    ``[N / S, M, S]``, so one reduction over N folds two physical dims.

        1  scale   x / N                    into the pool
        2  mean    sum over N               stored across a stick, twice:
                                            into ``mean`` and ``mean2``
        3  square  pool * x                 ``x * x / N``, back into the pool
        4  msq     sum over N               stored across a stick
        5  sq      mean * mean2             ``[M, S]``, stick-wide
        6  var     msq - sq                 ``[M, S]``, stick-wide
        7  var     var + eps                ``[M, S]``, in place
        8  scale   rsqrt(var)               ``[M, S]``, stick-wide
        9  out     layernormnorm(x, mean, scale, weight, bias), a request

    Three choices shape it.

    * The square of the mean multiplies two copies held at two addresses. Group 2
      stores the mean into both, and group 5 multiplies them, so the multiply's two
      operands are two values: two loads of one address are merged into one.
    * The statistics are two plain ``tl.sum`` reductions over the pool, the input
      scaled by the count first, so the sums are the means.
    * The scale is plain ``tl``: a variance, an ``eps`` and ``rsqrt``. The count and
      ``eps`` are read from buffers filled with them, ``n_ptr`` and ``eps_ptr``.

    Each of groups 5 to 8 reads what it needs as lane 0 of the stick the group
    before it wrote. Group 9 brings every operand to the tile's ``[M, N]``: the
    mean and the scale are read as lane 0 of their sticks and spread along the
    row, and the weight and the bias are ``[1, N]`` rows read through the tile's
    stick layout and spread down the rows, as the count is.
    """
    x_d = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
    n_d = tl.make_tensor_descriptor(n_ptr, shape=[1, N], strides=[N, 1], block_shape=[1, N])
    eps_d = tl.make_tensor_descriptor(eps_ptr, shape=[1, S], strides=[S, 1], block_shape=[1, S])
    pool_d = tl.make_tensor_descriptor(pool_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
    mean_w = tl.make_tensor_descriptor(mean_ptr, shape=[M], strides=[1], block_shape=[M])
    msq_w = tl.make_tensor_descriptor(msq_ptr, shape=[M], strides=[1], block_shape=[M])
    mean2_w = tl.make_tensor_descriptor(mean2_ptr, shape=[M], strides=[1], block_shape=[M])
    mean2_r = tl.make_tensor_descriptor(mean2_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1])
    mean_r = tl.make_tensor_descriptor(mean_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1])
    msq_r = tl.make_tensor_descriptor(msq_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1])
    sq_w = tl.make_tensor_descriptor(sq_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, S])
    sq_r = tl.make_tensor_descriptor(sq_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1])
    var_w = tl.make_tensor_descriptor(var_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, S])
    var_r = tl.make_tensor_descriptor(var_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1])
    w_d = tl.make_tensor_descriptor(weight_ptr, shape=[1, N], strides=[N, 1], block_shape=[1, N])
    b_d = tl.make_tensor_descriptor(bias_ptr, shape=[1, N], strides=[N, 1], block_shape=[1, N])
    out_d = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
    scale_r = tl.make_tensor_descriptor(scale_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1])
    scale_w = tl.make_tensor_descriptor(scale_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, S])
    tl.spyre_tensor_layout(x_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(n_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(w_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(b_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(out_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(pool_d, TILE_LAYOUT)
    tl.spyre_tensor_layout(mean_w, STAT_LAYOUT)
    tl.spyre_tensor_layout(msq_w, STAT_LAYOUT)
    tl.spyre_tensor_layout(mean2_w, STAT_LAYOUT)

    # G1: x divided by the count.
    n = tl.broadcast_to(n_d.load([0, 0]), [M, N])
    pool_d.store([0, 0], tl.fdiv(x_d.load([0, 0]), n))

    # G2: the mean, stored twice.
    mean_w.store([0], tl.sum(pool_d.load([0, 0]), axis=1))
    mean2_w.store([0], tl.sum(pool_d.load([0, 0]), axis=1))

    # G3: x * x / N, back into the pool.
    pool_d.store([0, 0], pool_d.load([0, 0]) * x_d.load([0, 0]))

    # G4: the mean of the squares.
    msq_w.store([0], tl.sum(pool_d.load([0, 0]), axis=1))

    # G5: the square of the mean.
    m = tl.broadcast_to(mean_r.load([0, 0]), [M, S])
    m2 = tl.broadcast_to(mean2_r.load([0, 0]), [M, S])
    sq_w.store([0, 0], m * m2)

    # G6: the variance.
    q = tl.broadcast_to(msq_r.load([0, 0]), [M, S])
    s = tl.broadcast_to(sq_r.load([0, 0]), [M, S])
    var_w.store([0, 0], q - s)

    # G7: the variance plus eps, in place.
    v = tl.broadcast_to(var_r.load([0, 0]), [M, S])
    e = tl.broadcast_to(eps_d.load([0, 0]), [M, S])
    var_w.store([0, 0], v + e)

    # G8: one over the root.
    v2 = tl.broadcast_to(var_r.load([0, 0]), [M, S])
    scale_w.store([0, 0], tl.math.rsqrt(v2))

    # G9: the normalisation, a request.
    x9 = x_d.load([0, 0])
    m9 = tl.broadcast_to(mean_r.load([0, 0]), [M, N])
    s9 = tl.broadcast_to(scale_r.load([0, 0]), [M, N])
    w9 = tl.broadcast_to(w_d.load([0, 0]), [M, N])
    b9 = tl.broadcast_to(b_d.load([0, 0]), [M, N])
    out_d.store([0, 0], tl.spyre_intrinsic("spyreop.layernormnorm", x9, m9, s9, w9, b9))
