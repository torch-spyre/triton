
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


@triton.jit
def stat_chain_on_stick(
    x_ptr,
    stat_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    X_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
):
    """``out[m, n] = x[m, n] - sum(x[m, :])`` with the statistic through HBM.

    Two compute groups, and the point is the *second* one. ``reduce_one_tile``
    proved a stick-axis reduce can store its statistic replicated across a stick;
    this proves the other half -- that a later compute can read it back and apply
    it against a full tile. Between them they are softmax's G1 and G2.

    The statistic buffer carries **two descriptors over one pointer**, differing
    only in ``block_shape``:

    - ``stat_w`` is rank-1 ``[M]`` and marked with a *broadcast* layout, so the
      reduce stores each statistic across a whole stick -- physical ``[M, S]``.
    - ``stat_r`` is rank-2 ``[M, S]`` blocked ``[M, 1]`` and deliberately carries
      **no** layout, because its logical shape already *is* its physical one. Its
      rank-2 tile is what lets the consumer address a fixed lane.

    Reading a fixed lane is not a stylistic choice. The device replicates a
    statistic only 8 elements wide, not ``S``, so a read of the lane matching the
    consumer's own output would read memory nothing ever wrote. Lane 0 is written
    for every row, which is why the block is ``[M, 1]`` at offset 0.

    ``x`` is loaded twice, once per group: a load result may not be shared across
    two compute groups.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    # Store side: rank-1, broadcast. The pass derives the physical [M, S] strides,
    # so the stride declared here is the logical one and is not load-bearing.
    stat_w = tl.make_tensor_descriptor(
        stat_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    # Read side: the same bytes seen as [M, S], one lane wide.
    stat_r = tl.make_tensor_descriptor(
        stat_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, X_LAYOUT)
    tl.spyre_tensor_layout(stat_w, STAT_LAYOUT)
    tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)

    # G1: fold the stick axis, store the statistic stick-wide.
    x = x_desc.load([0, 0])
    stat_w.store([0], tl.sum(x, axis=1))

    # G2: read lane 0 of the statistic and apply it to the tile.
    x2 = x_desc.load([0, 0])
    stat = stat_r.load([0, 0])
    out_desc.store([0, 0], x2 - stat)


@triton.jit
def max_shift_exp_on_stick(
    x_ptr,
    max_ptr,
    diff_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    TILE_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """``out[m, n] = exp(x[m, n] - max(x[m, :]))`` -- softmax's numerator.

    :func:`stat_chain_on_stick` with two changes and no third: the reduce is a
    ``max`` instead of a ``sum``, and the shifted tile goes to a scratch buffer
    that a third group exponentiates rather than straight out. Three compute
    groups, four buffers.

    It exists as its own variant because it is the first kernel here where a
    statistic's CONSUMER is not the last group -- ``diff`` is both written and
    read, so the chain is three deep rather than two -- and because a max reduce
    plus an exp is the pair softmax needs before any of the normalisation does.

    ``diff`` and ``exp`` cannot be the same buffer, and neither can ``diff`` and
    ``out``: a group's store and a later group's load of the same buffer is the
    fence the scheduler splits on, so each intermediate needs its own.

    fp16 throughout, and not by preference either: the statistic read-back --
    one lane loaded and splatted across the stick -- is an fp16-only path in
    dbo-opt, so any chain whose reduce round-trips through HBM has to be fp16.
    Both frontend refusals that used to stand in the way (``tl.max`` promoting
    narrow floats, ``tl.exp`` rejecting fp16) are forked behind ``is_spyre()``;
    see ``test_frontend_guards.py`` for the fork and the accuracy trade it
    accepts.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    diff_desc = tl.make_tensor_descriptor(
        diff_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    # The statistic, in its two roles: written rank-1 through a broadcast layout
    # so each value lands across a whole stick, read back as the [M, S] the
    # broadcast made, one lane wide. See stat_chain_on_stick for why the read is
    # at lane 0 rather than at the consumer's own lane.
    max_w = tl.make_tensor_descriptor(
        max_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    max_r = tl.make_tensor_descriptor(
        max_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1],
    )
    tl.spyre_tensor_layout(x_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(diff_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(out_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(max_w, STAT_LAYOUT)

    # G1: the row maximum, stored stick-wide.
    x1 = x_desc.load([0, 0])
    max_w.store([0], tl.max(x1, axis=1))

    # G2: every value shifted by its row's maximum, so the exponential below
    # never sees a positive argument.
    x2 = x_desc.load([0, 0])
    m = max_r.load([0, 0])
    diff_desc.store([0, 0], x2 - m)

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
    S: tl.constexpr,
    TILE_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """A whole softmax over the stick axis: six compute groups, seven buffers.

    :func:`max_shift_exp_on_stick` continued to the end, and the end is the
    normalisation: the exponentials are summed, the total is inverted, and every
    exponential is scaled by its row's reciprocal. It is the shape of the
    hand-written reference chain, group for group --

        max     reduce the stick axis      the largest value per row
        sub     x - max                    read at the head of the stick
        exp     the intrinsic
        sum     reduce the stick axis      the total of the exponentials
        recip   one over that total        written back across a stick
        mul     exp * recip                read at the head of the stick

    -- and the buffer count is the binding constraint rather than an incidental
    fact: seven, which is every base address there is (the eighth segment holds
    the program). Nothing here can be given a scratch buffer it does not already
    have.

    Two things about the shape are not free choices:

    * The exponentials go to memory because TWO groups read them -- the sum
      reduces them and the multiply scales them -- and a load result may not be
      shared across groups.
    * The reciprocal is its own group producing a STICK-WIDE result, rather than
      a divide folded into the multiply. The multiply reads its statistic one
      lane wide like every other consumer here, so something has to have written
      that lane; a divide in the last group would instead need the sum broadcast
      into it, which is a second compute in a group that already has one.

    It reaches a binary, launches, and matches the fp16 oracle to 1.3e-2
    relative -- the variant's banner in ``meta.py`` has that group by group.

    The ``1.0`` below never reaches the device, and that is load-bearing rather
    than incidental: LowerSpyreOps sees a numerator of one and emits the UNARY
    ``spyreop.reciprocal``, so the literal dies in lowering. It has to. A float
    immediate is emitted as its IEEE bit pattern and read by the device as
    Spyre's 1-6-9 float, so an fp16 ``1.0`` would arrive as ``0.5``. Do not
    replace this with anything that keeps the constant alive.

    ``tl.fdiv`` and not ``one / s``, which is a real distinction and not style:
    ``/`` promotes fp16 to fp32 (``computation_type_impl`` in semantic.py, on the
    grounds that PTX has no native fp16 divide), and the promotion emits an
    ``arith.extf`` that no pass here lowers. ``tl.fdiv`` divides at the operand
    width, which is what the ``spyreop`` divide intrinsics want. The numerator is
    a same-dtype ``tl.full`` for the same reason -- a bare Python ``1.0`` is an
    fp32 scalar, and an fp32 numerator over an fp16 total is not the shape the
    reciprocal peephole (or a ``realdiv``) can take.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    diff_desc = tl.make_tensor_descriptor(
        diff_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    exp_desc = tl.make_tensor_descriptor(
        exp_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    # The two statistics, each in its two roles: written rank-1 through a
    # broadcast layout so a value lands across a whole stick, read back as the
    # [M, S] the broadcast made, one lane wide. See stat_chain_on_stick for why
    # the read is at lane 0 rather than at the consumer's own lane.
    max_w = tl.make_tensor_descriptor(
        max_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    max_r = tl.make_tensor_descriptor(
        max_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1],
    )
    sum_w = tl.make_tensor_descriptor(
        sum_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    sum_r = tl.make_tensor_descriptor(
        sum_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1],
    )
    # The reciprocal is the one statistic written WITHOUT a broadcast layout: it
    # is produced [M, S] already, by a compute whose output tile is stick-wide,
    # so its logical shape is its physical one and there is nothing to replicate.
    # Both descriptors are rank-2 over the same bytes; only the block differs.
    recip_w = tl.make_tensor_descriptor(
        recip_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, S],
    )
    recip_r = tl.make_tensor_descriptor(
        recip_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1],
    )
    tl.spyre_tensor_layout(x_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(diff_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(exp_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(out_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(max_w, STAT_LAYOUT)
    tl.spyre_tensor_layout(sum_w, STAT_LAYOUT)

    # G1: the row maximum, stored stick-wide.
    x1 = x_desc.load([0, 0])
    max_w.store([0], tl.max(x1, axis=1))

    # G2: every value shifted by its row's maximum.
    x2 = x_desc.load([0, 0])
    m = max_r.load([0, 0])
    diff_desc.store([0, 0], x2 - m)

    # G3: the exponential, to memory because two groups below read it.
    d = diff_desc.load([0, 0])
    exp_desc.store([0, 0], tl.exp(d))

    # G4: the total of the exponentials, stored stick-wide.
    e1 = exp_desc.load([0, 0])
    sum_w.store([0], tl.sum(e1, axis=1))

    # G5: one over that total, spread across the stick so G6 can read a lane.
    s = sum_r.load([0, 0])
    s_bcast = tl.broadcast_to(s, [M, S])
    one = tl.full([M, S], 1.0, dtype=s_bcast.dtype)
    recip_w.store([0, 0], tl.fdiv(one, s_bcast))

    # G6: every exponential scaled by its row's reciprocal.
    e2 = exp_desc.load([0, 0])
    r = recip_r.load([0, 0])
    out_desc.store([0, 0], e2 * r)
