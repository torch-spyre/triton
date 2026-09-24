
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
# elementwise op. Two kernels, one per axis, and ``sum`` in both.
#
# TWO KERNELS rather than one with an ``AXIS`` constexpr, unlike
# ``reduce_one_tile`` above, because the axis is not the only thing that differs.
# The off-stick arm reads its statistic back through the descriptor it wrote; the
# on-stick arm needs a SECOND descriptor over the same pointer, at a shape only
# the physical layout has. Different argument lists, and one of them has to be
# told the stick width.
#
# A SUBTRACT rather than a divide, for the ORACLE rather than for the lowering.
# ``x / sum(x)`` has a denominator that is a sum of standard normals and can land
# near zero, which would make the comparison a statement about catastrophic
# cancellation instead of about the chain. ``x - sum(x)`` passes the statistic's
# own error through unamplified, so an absolute tolerance sized in ulp of the
# statistic is the whole story.
#
# ``sum`` and not ``max``, and the reason is the DTYPE and not the combiner: at
# fp16 ``tl.max`` promotes to fp32 and emits an ``arith.extf`` that no pass here
# lowers. DropReductionInitFill gates on shape alone, so the combiner itself
# refuses nothing -- which makes the wall the promotion, and the promotion has
# nothing to do with reading a statistic back.
# ---------------------------------------------------------------------------


@triton.jit
def stat_chain_off_stick(
    x_ptr,
    stat_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TILE_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """``out[m, n] = x[m, n] - sum(x[:, n])``, the statistic through HBM.

    Two compute groups, and the second one is the point. ``reduce_one_tile`` at
    ``AXIS=0`` already proves the first: folding the non-stick axis leaves the
    stick split of N untouched, so the reduce is emitted at physical shape with
    the surviving stick index as a batch dimension, and it reaches a binary and
    launches. What is new here is a LATER compute reading that statistic back and
    applying it to a full tile -- softmax's G1 and G2, with the max replaced by a
    sum for the reason the banner above gives.

    ONE DESCRIPTOR for the statistic, written and then read, and that is the
    off-stick arm's whole advantage over its sibling. The statistic's surviving
    logical dim IS the stick dim, so its layout PARTITIONS -- ``[N]`` becomes
    ``[N/S, S]``, the element count unchanged -- and a partition is faithful to
    the dense ``[N]`` buffer the host staged. So the logical form is a correct
    description of the bytes, the same descriptor serves both roles, and the
    kernel never has to name the stick width. Compare ``stat_chain_on_stick``,
    which has to.

    ``x`` is loaded twice, once per group: a load result may not be shared across
    two compute groups.

    The layouts are not optional here the way they are on the kernels above. The
    arm exists to say what happens to a statistic chain under stick tiling, and
    with no annotation there is nothing for it to say.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    # The reduce folds M, so the statistic is over N -- a whole number of sticks.
    stat_desc = tl.make_tensor_descriptor(
        stat_ptr, shape=[N], strides=[1], block_shape=[N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(out_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(stat_desc, STAT_LAYOUT)

    # G1: fold the non-stick axis, store the statistic across the sticks of N.
    x1 = x_desc.load([0, 0])
    stat_desc.store([0], tl.sum(x1, axis=0))

    # G2: read the same descriptor back and apply it down every row.
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
    S: tl.constexpr,
    TILE_LAYOUT: tl.constexpr,
    STAT_LAYOUT: tl.constexpr,
):
    """``out[m, n] = x[m, n] - sum(x[m, :])``, the statistic through HBM.

    The sibling above with the axis moved onto the stick, and that one change
    costs the statistic its faithful logical form. ``reduce_one_tile_on_stick``'s
    splat variant already proves the write half: under stick-on-N the lanes being
    folded are read as one physical dim and the statistic is written as another,
    so a partitioning floordiv/mod split has nowhere to put the surviving extent
    and the layout has to SPLAT instead -- logical ``[M]`` becomes physical
    ``[M, S]``, one whole stick per statistic.

    TWO DESCRIPTORS OVER ONE POINTER, and the asymmetry between them is the
    non-obvious part:

    - ``stat_w`` is rank-1 ``[M]`` and carries the splat layout, so the reduce's
      rank-1 result is stored across a stick apiece.
    - ``stat_r`` is rank-2 ``[M, S]`` blocked ``[M, 1]`` and carries **no**
      layout, because its logical shape already is its physical one. That is what
      lets the consumer address a fixed lane, and it is why the read side is not
      annotated at all -- an annotated statistic view asked for a ``[M, 1]`` block
      is the refusal this sidesteps.

    Lane 0 rather than the consumer's own lane, which is a device fact rather
    than a style: the device replicates a statistic only 8 elements wide, not
    ``S``, so a read at the lane matching one's own output would read memory
    nothing ever wrote. Lane 0 is written for every row.

    ``S`` is therefore a constexpr the kernel has to be told, and it is a LEAK:
    the stick width is a property of the dtype and the hardware, and nothing else
    in this file needs it -- the layouts carry theirs inside themselves. Here the
    kernel has to restate it, in a descriptor over its own statistic buffer,
    because there is no way to say "the physical shape of that buffer" in the
    frontend. The off-stick sibling needs no such argument, which is what makes
    this an on-stick cost rather than a chain cost.

    ``x`` is loaded twice, once per group: a load result may not be shared across
    two compute groups.
    """
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    # Store side: rank-1, splat. The layout pass derives the physical [M, S]
    # strides, so the stride declared here is the logical one and is not
    # load-bearing.
    stat_w = tl.make_tensor_descriptor(
        stat_ptr, shape=[M], strides=[1], block_shape=[M],
    )
    # Read side: the same bytes seen as the [M, S] the splat made, one lane wide.
    stat_r = tl.make_tensor_descriptor(
        stat_ptr, shape=[M, S], strides=[S, 1], block_shape=[M, 1],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N],
    )
    tl.spyre_tensor_layout(x_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(out_desc, TILE_LAYOUT)
    tl.spyre_tensor_layout(stat_w, STAT_LAYOUT)

    # G1: fold the stick axis, store the statistic stick-wide.
    x1 = x_desc.load([0, 0])
    stat_w.store([0], tl.sum(x1, axis=1))

    # G2: read lane 0 of the statistic and apply it across each row.
    x2 = x_desc.load([0, 0])
    s = stat_r.load([0, 0])
    out_desc.store([0, 0], x2 - s)
