
import triton
import triton.language as tl


@triton.jit
def reduce_spyre(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IN_LAYOUT: tl.constexpr,
    OUT_LAYOUT: tl.constexpr,
):
    """Row-sum reduce: out[m] = sum(in[m, :]) for each row m.

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
        out_desc.store([m_sub * BLOCK_M], a_tile.sum(1))


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
):
    """Rank-3 reduce over the NON-TRAILING middle axis.

    out[d0, d2] = sum(in[d0, :, d2]) over the D1 axis.

    Distributes D0-block slabs across the grid, mirroring reduce_spyre: each
    core iterates its assigned D0-blocks, loads a [BLOCK_D0, D1, D2] tile and
    reduces over the middle (D1) axis to produce a [BLOCK_D0, D2] result.

    With three distinct extents, reducing the wrong axis fails loudly: the
    pass must rotate the reduced (D1) axis to the end via linalg.transpose
    before linalg.reduce, which always takes the trailing dims.

    The reduced axis is the *middle* one, so it is never the axis being
    blocked or distributed — the tiling above is orthogonal to what makes
    this case interesting, and the transpose requirement holds at any grid.

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
        out_desc.store([d0_sub * BLOCK_D0, 0], a_tile.sum(1))
