"""Inter-tile all_reduce kernel: multi-group ADD reduction across tiles.

A 2D kernel where the x-axis is reduced via all_reduce.  Each tile computes a
partial sum over its local block, then all tiles in the same reduction group
sum their partials together.  Every tile in a group receives the fully-reduced
value.

Requires ``tl.inter_tile`` (T006), which is not yet in the frontend.
"""

import triton
import triton.language as tl


@triton.jit
def inter_tile_add_kernel(
    x_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Each tile reduces its local row-block, then participates in cross-tile
    all_reduce along the x-axis so every tile in the group receives the sum.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    partial = x_desc.load([offset_m, offset_n])  # tensor<BLOCK_M x BLOCK_N x f32>

    # Cross-tile reduction: all tiles along the x-axis (pid_n axis) cooperate.
    # tl.inter_tile is not yet in the frontend (T006); this body is placeholder.
    result = tl.inter_tile(partial, axis='x', combiner='add', mode='all_reduce')

    out_desc.store([offset_m, offset_n], result)
