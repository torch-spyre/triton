"""Inter-tile all_reduce kernel: multi-group ADD reduction across tiles.

Flat-1D-grid kernel where each tile loads its row-block partial, then all
tiles in the same reduction group cooperate via ``tl.inter_tile`` to sum
their partials.  Every tile in a group receives the fully-reduced value.

Grid layout (8 tiles, flat ID 0-7):
  - 4 row-groups × 2 x-tiles per group.
  - ``work_slices[tile_id]["x"]`` is a group label (0–3); tiles with the
    same label cooperate.  ``W["x"] = 4`` (4 distinct labels), ``gsize = 2``.
    The 4 groups run independently.
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
    NUM_N_TILES: tl.constexpr,  # number of tiles along N (= x-axis group size)
    work_slices: tl.constexpr,  # {tile_id: {"x": slice_idx}} for all tiles
):
    """Each tile reduces its local row-block, then participates in a cross-tile
    all_reduce along the x-axis so every tile in the group receives the sum.

    Data layout (flat 1D grid of 8 tiles):
      tile_id = pid_m * NUM_N_TILES + pid_n
      pid_m  ∈ {0..3}   → selects the BLOCK_M row-block
      pid_n  ∈ {0..1}   → selects the BLOCK_N column-block
    """
    pid = tl.program_id(0)          # flat tile id ∈ {0..7}
    pid_m = pid // NUM_N_TILES      # row index
    pid_n = pid %  NUM_N_TILES      # column index (= x-axis slice index)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    # Each tile loads its column-block; shape = tensor<BLOCK_M x BLOCK_N x f32>.
    # The partial must have a unit leading dimension for the inter_tile_reduce
    # verifier (result rank = partial rank - 1).
    partial_2d = x_desc.load([offset_m, offset_n])  # tensor<BLOCK_M x BLOCK_N x f32>
    partial = tl.reshape(partial_2d, [1, BLOCK_M, BLOCK_N])  # unit leading dim

    # Cross-tile reduction: all tiles in the same row group (same pid_m)
    # cooperate.  work_slices carries the W/C metadata for the pass.
    result = tl.inter_tile(
        partial,
        axis="x",
        combiner="add",
        mode="all_reduce",
        work_slices=work_slices,
    )  # tensor<BLOCK_M x BLOCK_N x f32> (unit dim collapsed by the pass)

    out_desc.store([offset_m, offset_n], result)


@triton.jit
def inter_tile_add_kernel_f16(
    x_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    work_slices: tl.constexpr,
):
    """f16 variant of inter_tile_add_kernel."""
    pid = tl.program_id(0)
    pid_m = pid // NUM_N_TILES
    pid_n = pid %  NUM_N_TILES

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    partial_2d = x_desc.load([offset_m, offset_n])  # tensor<BLOCK_M x BLOCK_N x f16>
    partial = tl.reshape(partial_2d, [1, BLOCK_M, BLOCK_N])

    result = tl.inter_tile(
        partial,
        axis="x",
        combiner="add",
        mode="all_reduce",
        work_slices=work_slices,
    )

    out_desc.store([offset_m, offset_n], result)
