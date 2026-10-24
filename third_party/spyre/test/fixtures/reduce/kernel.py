
import triton
import triton.language as tl


# Three reductions are supported through ``OP: tl.constexpr``: ``sum``, ``max``
# and ``min``. They are the three that lower: each becomes a ``linalg.reduce``
# differing only in its combiner (addf/addi, maxnumf/maxsi, minnumf/minsi), and
# all three run on ``ktir_cpu`` at fp16, fp32 and i32. What separates them is the
# device: the dataflow scheduler resets a reduction accumulator to zero whatever
# the combiner is, so only ``sum`` has the right neutral element -- see the
# Level D banner in ``meta.py``.


@triton.jit
def reduce_spyre(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """Row reduce over the trailing axis: out[m] = OP(in[m, :]) for each row m.

    Distributes M-block rows across the grid. Each core iterates its
    assigned M-blocks, loads a [BLOCK_M, N] tile and reduces over the
    N axis to produce a [BLOCK_M] result.

    IN_LAYOUT  — stick-tiling for in_ptr's [M, N] extent
                 (stick-on-N: [(1,"floordiv",S), 0, (1,"mod",S)]).
    OUT_LAYOUT — stick-tiling for out_ptr's [M] extent
                 (stick: [(0,"floordiv",S), (0,"mod",S)]).
    Pass None for no layout annotation.
    """
    pid_m = tl.program_id(0)
    grid_m = tl.num_programs(0)

    m_blocks = tl.cdiv(M, BLOCK_M)
    rows_per_core = tl.cdiv(m_blocks, grid_m)

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, N],
    )
    if IN_LAYOUT is not None:
        tl.spyre_tensor_layout(in_desc, IN_LAYOUT)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M],
        strides=[1],
        block_shape=[BLOCK_M],
    )
    if OUT_LAYOUT is not None:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    m_start = pid_m * rows_per_core
    m_end   = tl.minimum(m_start + rows_per_core, m_blocks)
    for m_sub in range(m_start, m_end):
        a_tile = in_desc.load([m_sub * BLOCK_M, 0])
        if OP == "sum":
            reduced = a_tile.sum(1)
        elif OP == "max":
            reduced = tl.max(a_tile, 1)
        else:
            reduced = tl.min(a_tile, 1)
        out_desc.store([m_sub * BLOCK_M], reduced)


@triton.jit
def reduce_middle_axis_spyre(
    in_ptr,
    out_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_D0: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """Rank-3 reduce over the NON-TRAILING middle axis.

    out[d0, d2] = OP(in[d0, :, d2]) over the D1 axis.

    Distributes D0-block slabs across the grid, mirroring reduce_spyre: each
    core iterates its assigned D0-blocks, loads a [BLOCK_D0, D1, D2] tile and
    reduces over the middle (D1) axis to produce a [BLOCK_D0, D2] result.

    With three distinct extents, reducing the wrong axis fails loudly — the
    output shape changes. What the lowering does about the reduced axis not
    being trailing is: nothing. ``linalg.reduce`` names the axes it folds in a
    sorted ``dimensions`` list, so D1 is reduced where it sits and no
    ``linalg.transpose`` is emitted, at either grid and with or without a stick
    layout -- which is what ``meta.py`` asserts with
    ``assert_absent("linalg.transpose")``.

    The reduced axis is the *middle* one, so it is never the axis being
    blocked or distributed — the tiling above is orthogonal to what makes
    this case interesting.

    IN_LAYOUT  — stick-tiling for in_ptr's [D0, D1, D2] extent.
    OUT_LAYOUT — stick-tiling for out_ptr's [D0, D2] extent.
    Pass None for no layout annotation.
    """
    pid_0 = tl.program_id(0)
    grid_0 = tl.num_programs(0)

    d0_blocks = tl.cdiv(D0, BLOCK_D0)
    blocks_per_core = tl.cdiv(d0_blocks, grid_0)

    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[D0, D1, D2],
        strides=[D1 * D2, D2, 1],
        block_shape=[BLOCK_D0, D1, D2],
    )
    if IN_LAYOUT is not None:
        tl.spyre_tensor_layout(in_desc, IN_LAYOUT)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[D0, D2],
        strides=[D2, 1],
        block_shape=[BLOCK_D0, D2],
    )
    if OUT_LAYOUT is not None:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    d0_start = pid_0 * blocks_per_core
    d0_end   = tl.minimum(d0_start + blocks_per_core, d0_blocks)
    for d0_sub in range(d0_start, d0_end):
        a_tile = in_desc.load([d0_sub * BLOCK_D0, 0, 0])
        if OP == "sum":
            reduced = a_tile.sum(1)
        elif OP == "max":
            reduced = tl.max(a_tile, 1)
        else:
            reduced = tl.min(a_tile, 1)
        out_desc.store([d0_sub * BLOCK_D0, 0], reduced)


@triton.jit
def reduce_one_tile(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
    OP: tl.constexpr,
    AXIS: tl.constexpr,
):
    """The whole reduce in one tile: no ``tl.program_id``, no loop.

    The counterpart of ``elementwise_1d_device``, and it exists for the same
    reason: every other kernel here carves work across the grid with an
    ``scf.for`` over the program id, and dbo-opt rejects the loop that is
    outlined from it. Removing the loop is what makes any *reduce*-specific
    device refusal visible, rather than the loop refusal every variant shares.

    ``AXIS`` selects which axis of ``[M, N]`` is folded away, and it is a
    parameter because the two answers are not the same case once the input is
    stick-tiled on N -- which is where a reduce's device story is decided:

    - ``AXIS=0`` folds M, a *whole physical dimension*. The stick structure
      survives untouched, and because the output descriptor declares exactly the
      layout that leaves, the reduce is emitted at physical shape: one
      ``linalg.reduce`` whose surviving stick index is a batch dimension
      (``ins tensor<2x64x64> outs tensor<2x64> dimensions = [1]``), which is the
      shape torch-spyre's working ``sum`` emits. This one reaches a binary and
      launches.
    - ``AXIS=1`` folds N, the *stick* axis. Stick-on-N splits N across physical
      dimensions 0 and 2, so the reduce names both and collapses to a rank-1
      result that then has to be widened back for a rank-2 store. No batch
      dimension survives to carry the split. torch-spyre does not emit
      ``linalg.reduce`` for this at all -- its ``sum-onstick`` case uses
      ``linalg.generic`` with the maps written out, because the lanes are read as
      one dimension and written as another. It still stops in the scheduler; the
      Level D banner in ``meta.py`` says where.

    They stop in different places, which is the point of being able to ask for
    both. ``AXIS`` also makes the output extent vary, so the variant states no
    output shape: it follows from the axis.
    """
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M, N],
    )
    if IN_LAYOUT is not None:
        tl.spyre_tensor_layout(in_desc, IN_LAYOUT)

    # The extent that survives the reduce. Derived rather than passed so it
    # cannot contradict AXIS -- an output extent of M on an AXIS=0 reduce is a
    # shape error the descriptor would carry all the way to the store.
    OUT_EXTENT: tl.constexpr = N if AXIS == 0 else M

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[OUT_EXTENT],
        strides=[1],
        block_shape=[OUT_EXTENT],
    )
    if OUT_LAYOUT is not None:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    a_tile = in_desc.load([0, 0])
    if OP == "sum":
        reduced = a_tile.sum(AXIS)
    elif OP == "max":
        reduced = tl.max(a_tile, AXIS)
    else:
        reduced = tl.min(a_tile, AXIS)
    out_desc.store([0], reduced)


# ---------------------------------------------------------------------------
# The statistic chain: a reduce whose own output is read back by a later
# elementwise op, ``out = x - sum(x, axis)``. One kernel per axis, because the
# axis is baked into the broadcast that applies the statistic.
#
# A subtract rather than a divide, for the oracle: ``x - sum(x)`` passes the
# statistic's error through unamplified, where a sum of standard normals as a
# denominator can land near zero.
# ---------------------------------------------------------------------------


@triton.jit
def stat_chain_off_stick(
    x_ptr,
    stat_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STICK_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """``out[m, n] = x[m, n] - sum(x[:, n])``, the statistic through HBM.

    Two compute groups: G1 reduces over M and stores the ``[N]`` statistic, G2
    reads it back and applies it down every row. ``x`` is loaded once per group.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    stat_desc = tl.make_tensor_descriptor(
        stat_ptr, shape=[N], strides=[1], block_shape=[N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(out_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(stat_desc, STAT_LAYOUT)

    # G1: the column sum.
    x1 = x_desc.load([0, 0])
    stat_desc.store([0], tl.sum(x1, axis=0))

    # G2: every row shifted by it.
    x2 = x_desc.load([0, 0])
    s = stat_desc.load([0])
    out_desc.store([0, 0], x2 - s[None, :])


@triton.jit
def stat_chain_on_stick(
    x_ptr,
    stat_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STICK_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """``out[m, n] = x[m, n] - sum(x[m, :])``, the statistic through HBM.

    :func:`stat_chain_off_stick` with the reduce over N instead: G1 stores the
    ``[M]`` row sum, G2 reads it back and applies it across each row.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    stat_desc = tl.make_tensor_descriptor(
        stat_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(out_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(stat_desc, STAT_LAYOUT)

    # G1: the row sum.
    x1 = x_desc.load([0, 0])
    stat_desc.store([0], tl.sum(x1, axis=1))

    # G2: every row shifted by its sum.
    x2 = x_desc.load([0, 0])
    s = stat_desc.load([0])
    out_desc.store([0, 0], x2 - s[:, None])


# ---------------------------------------------------------------------------
# Softmax: the row statistic chain carried to its end, in two lengths.
#
#   max_shift_exp_on_stick   3 groups, 4 buffers   exp(x - max(x))
#   softmax_on_stick         6 groups, 7 buffers   the whole normalisation
#
# Every intermediate goes through its own buffer, since a value one group stores
# is read by a later group.
# ---------------------------------------------------------------------------


@triton.jit
def _maximum(a, b):
    """The combiner of a maximum reduce, for :func:`tl.reduce`."""
    return tl.maximum(a, b)


@triton.jit
def max_shift_exp_on_stick(
    x_ptr,
    max_ptr,
    diff_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STICK_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """``out[m, n] = exp(x[m, n] - max(x[m, :]))`` -- softmax's numerator.

    Three compute groups: the row maximum, the shift, the exponential. ``diff``
    and ``out`` are separate buffers because G3 reads what G2 stored.

    The maximum is a :func:`tl.reduce` with an explicit combiner, which keeps an
    fp16 input at fp16 where ``tl.max`` promotes it.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    max_desc = tl.make_tensor_descriptor(
        max_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    diff_desc = tl.make_tensor_descriptor(
        diff_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(diff_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(out_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(max_desc, STAT_LAYOUT)

    # G1: the row maximum.
    x1 = x_desc.load([0, 0])
    max_desc.store([0], tl.reduce(x1, 1, _maximum))

    # G2: every value shifted by its row's maximum.
    x2 = x_desc.load([0, 0])
    m = max_desc.load([0])
    diff_desc.store([0, 0], x2 - m[:, None])

    # G3: the exponential.
    d = diff_desc.load([0, 0])
    out_desc.store([0, 0], tl.exp(d))


@triton.jit
def softmax_on_stick(
    x_ptr,
    max_ptr,
    diff_ptr,
    exp_ptr,
    sum_ptr,
    recip_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STICK_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """A row softmax in six compute groups over seven buffers.

    :func:`max_shift_exp_on_stick` continued: the exponentials are summed, the
    total is inverted, and every exponential is scaled by its row's reciprocal.

        max     reduce over N        the largest value per row
        sub     x - max
        exp
        sum     reduce over N        the total of the exponentials
        recip   one over that total
        mul     exp * recip

    The exponentials go to memory because two groups read them.

    ``tl.fdiv`` of a same-dtype ``tl.full`` one, rather than ``1.0 / s``: ``/``
    and a Python float both promote fp16 to fp32. A numerator of exactly one is
    what ``LowerSpyreOps`` matches to emit ``spyreop.reciprocal``.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    max_desc = tl.make_tensor_descriptor(
        max_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    diff_desc = tl.make_tensor_descriptor(
        diff_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    exp_desc = tl.make_tensor_descriptor(
        exp_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    sum_desc = tl.make_tensor_descriptor(
        sum_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    recip_desc = tl.make_tensor_descriptor(
        recip_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(diff_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(exp_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(out_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(max_desc, STAT_LAYOUT)
    tl.spyre_tensor_layout(sum_desc, STAT_LAYOUT)
    tl.spyre_tensor_layout(recip_desc, STAT_LAYOUT)

    # G1: the row maximum.
    x1 = x_desc.load([0, 0])
    max_desc.store([0], tl.reduce(x1, 1, _maximum))

    # G2: every value shifted by its row's maximum.
    x2 = x_desc.load([0, 0])
    m = max_desc.load([0])
    diff_desc.store([0, 0], x2 - m[:, None])

    # G3: the exponential.
    d = diff_desc.load([0, 0])
    exp_desc.store([0, 0], tl.exp(d))

    # G4: the total of the exponentials.
    e1 = exp_desc.load([0, 0])
    sum_desc.store([0], tl.sum(e1, axis=1))

    # G5: one over that total.
    s = sum_desc.load([0])
    one = tl.full([M], 1.0, dtype=s.dtype)
    recip_desc.store([0], tl.fdiv(one, s))

    # G6: every exponential scaled by its row's reciprocal.
    e2 = exp_desc.load([0, 0])
    r = recip_desc.load([0])
    out_desc.store([0, 0], e2 * r[:, None])
