
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
                 (splat: [0, (0,"splat",S)]).
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
      dimensions 0 and 2, so the reduce names both and no stick index survives
      into the result. The output layout is then a *splat*, ``[M] -> [M, S]``:
      each row's value is written across a whole stick, the form torch-spyre's
      ``_single_arg_op_layout`` gives a reduction over the stick dim. This one
      reaches a binary and launches too.

    ``AXIS`` also makes the output extent vary, so the variant states no output
    shape: it follows from the axis.
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


@triton.jit
def reduce_middle_axis_one_tile(
    in_ptr,
    out_ptr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """:func:`reduce_middle_axis_spyre` in one tile: no ``tl.program_id``, no loop.

    out[d0, d2] = OP(in[d0, :, d2]) over the D1 axis, the whole ``[D0, D1, D2]``
    extent loaded as one block and the whole ``[D0, D2]`` result stored as one.

    Stuck on D2, the reduced axis D1 is a whole physical dimension of the input
    and D2's stick split survives into the output, so the reduce keeps the
    surviving stick index as a batch dimension -- the rank-3 counterpart of
    :func:`reduce_one_tile` at ``AXIS=0``.

    IN_LAYOUT  — stick-tiling for in_ptr's [D0, D1, D2] extent.
    OUT_LAYOUT — stick-tiling for out_ptr's [D0, D2] extent.
    Pass None for no layout annotation.
    """
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
        shape=[D0, D1, D2],
        strides=[D1 * D2, D2, 1],
        block_shape=[D0, D1, D2],
    )
    if IN_LAYOUT is not None:
        tl.spyre_tensor_layout(in_desc, IN_LAYOUT)

    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[D0, D2],
        strides=[D2, 1],
        block_shape=[D0, D2],
    )
    if OUT_LAYOUT is not None:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    a_tile = in_desc.load([0, 0, 0])
    if OP == "sum":
        reduced = a_tile.sum(1)
    elif OP == "max":
        reduced = tl.max(a_tile, 1)
    else:
        reduced = tl.min(a_tile, 1)
    out_desc.store([0, 0], reduced)


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
def stat_chain_off_stick_pinned(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STICK_LAYOUT: tl.constexpr,
    PIN_OFFSET: tl.constexpr,
):
    """``out[m, n] = x[m, n] - sum(x[:, n])``, the statistic in the scratchpad.

    :func:`stat_chain_off_stick` with its statistic descriptor replaced by a pin.
    The two groups are the sibling's: G1 folds M, G2 applies the result down every
    row. What differs is where the statistic is held between them -- a
    ``ct_local`` buffer at element ``PIN_OFFSET`` rather than ``stat_ptr`` -- so
    the kernel takes no statistic pointer and no ``STAT_LAYOUT``. The pin's buffer
    is the compiler's, and takes its layout from its neighbours.

    ``x`` is still loaded twice, once per group, for the sibling's reason.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(out_desc, STICK_LAYOUT)

    # G1: fold the non-stick axis, the statistic into the scratchpad.
    x1 = x_desc.load([0, 0])
    s = tl.sum(x1, axis=0)
    tl.spyre_pin(s, "ct_local", offset=PIN_OFFSET)

    # G2: read it back and apply it down every row.
    x2 = x_desc.load([0, 0])
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
def max_shift_exp_on_stick_pinned(
    x_ptr,
    max_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STICK_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
    PIN_OFFSET: tl.constexpr,
):
    """``out[m, n] = exp(x[m, n] - max(x[m, :]))``, the shift in the scratchpad.

    :func:`max_shift_exp_on_stick` with the ``diff`` descriptor replaced by a pin:
    G2's shifted ``[M, N]`` tile is held in a ``ct_local`` buffer at element
    ``PIN_OFFSET``, and G3 reads it from there. The pin's buffer takes
    ``STICK_LAYOUT`` from its neighbours, the tiles on either side of it.

    The statistic stays in HBM, through the sibling's one rank-1 descriptor.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    max_desc = tl.make_tensor_descriptor(
        max_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(out_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(max_desc, STAT_LAYOUT)

    # G1: the row maximum.
    x1 = x_desc.load([0, 0])
    max_desc.store([0], tl.reduce(x1, 1, _maximum))

    # G2: every value shifted by its row's maximum, into the scratchpad.
    x2 = x_desc.load([0, 0])
    m = max_desc.load([0])
    d = x2 - m[:, None]
    tl.spyre_pin(d, "ct_local", offset=PIN_OFFSET)

    # G3: the exponential, of the shift read back from the scratchpad.
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


@triton.jit
def softmax_on_stick_pinned(
    x_ptr,
    max_ptr,
    sum_ptr,
    recip_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STICK_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
    DIFF_OFFSET: tl.constexpr,
    EXP_OFFSET: tl.constexpr,
):
    """:func:`softmax_on_stick` with its two full-tile intermediates pinned.

    The same six groups. ``diff`` and ``exp`` -- the two ``[M, N]`` tiles -- are
    held in the scratchpad instead of HBM, each in a ``ct_local`` buffer of its
    own: ``diff`` at element ``DIFF_OFFSET`` and ``exp`` at ``EXP_OFFSET``. Each
    buffer is ``M * N`` elements, so the two offsets are at least that far apart
    and the buffers do not overlap. Both take ``STICK_LAYOUT`` from their
    neighbours.

    ``exp`` has TWO consumers, G4's sum and G6's multiply, and the pin gives each
    its own load of the one buffer.

    The three statistics stay in HBM, each through the sibling's one rank-1
    descriptor. Five buffers, where the sibling has seven.

    The reciprocal group is the sibling's verbatim, and its numerator of exactly
    one is load-bearing for the reason given there.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    max_desc = tl.make_tensor_descriptor(
        max_ptr, shape=[M], strides=[1], block_shape=[M],
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
    tl.spyre_tensor_layout(out_desc, STICK_LAYOUT)
    tl.spyre_tensor_layout(max_desc, STAT_LAYOUT)
    tl.spyre_tensor_layout(sum_desc, STAT_LAYOUT)
    tl.spyre_tensor_layout(recip_desc, STAT_LAYOUT)

    # G1: the row maximum.
    x1 = x_desc.load([0, 0])
    max_desc.store([0], tl.reduce(x1, 1, _maximum))

    # G2: every value shifted by its row's maximum, into the scratchpad.
    x2 = x_desc.load([0, 0])
    m = max_desc.load([0])
    d = x2 - m[:, None]
    tl.spyre_pin(d, "ct_local", offset=DIFF_OFFSET)

    # G3: the exponential, into the scratchpad for the two groups below.
    e = tl.exp(d)
    tl.spyre_pin(e, "ct_local", offset=EXP_OFFSET)

    # G4: the total of the exponentials.
    sum_desc.store([0], tl.sum(e, axis=1))

    # G5: one over that total.
    s = sum_desc.load([0])
    one = tl.full([M], 1.0, dtype=s.dtype)
    recip_desc.store([0], tl.fdiv(one, s))

    # G6: every exponential scaled by its row's reciprocal.
    r = recip_desc.load([0])
    out_desc.store([0, 0], e * r[:, None])


# ---------------------------------------------------------------------------
# The reduce's PRODUCER, through the scratchpad.
#
# The statistic chains above put a compute AFTER the reduce and hand the value
# through HBM, with a descriptor the kernel writes and reads itself. This puts one
# BEFORE it and hands the value through the scratchpad, with no descriptor at all:
# the intermediate is anonymous, and ``tl.spyre_pin`` is what names where it goes.
#
# The pin is also what makes it two compute groups. Unpinned, ``x * y`` would fuse
# into the reduce's own generic; pinned, the product is stored into a ``ct_local``
# view by the first group and loaded back from it by the second.
# ---------------------------------------------------------------------------


@triton.jit
def pinned_mul_sum(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
    PIN_OFFSET: tl.constexpr,
):
    """``out[n] = sum(x[:, n] * y[:, n])``, the product held in the scratchpad.

    Two compute groups handing one whole ``[M, N]`` tile through the scratchpad:
    the first multiplies and stores the product into a ``ct_local`` view at
    ``PIN_OFFSET``, the second loads it back and folds the non-stick axis.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    y_desc = tl.make_tensor_descriptor(
        y_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[N], strides=[1], block_shape=[N],
    )
    tl.spyre_tensor_layout(x_desc, IN_LAYOUT)
    tl.spyre_tensor_layout(y_desc, IN_LAYOUT)
    tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    # G1: the product, into the scratchpad.
    x = x_desc.load([0, 0])
    y = y_desc.load([0, 0])
    p = x * y
    tl.spyre_pin(p, "ct_local", offset=PIN_OFFSET)

    # G2: read it back and fold the non-stick axis.
    out_desc.store([0], tl.sum(p, axis=0))
