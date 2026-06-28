"""Inter-tile reduction kernels: all_reduce and reduce_to_one variants.

Kernels in this module:

all_reduce:
  inter_tile_add_kernel      -- ADD all_reduce (f32): flat 1D grid, 8 tiles,
                                4 row-groups × 2 x-tiles per group.
  inter_tile_add_kernel_f16  -- same as above but fp16.

reduce_to_one:
  matmul_splitk_kernel       -- split-K matmul C[M,N]=A[M,K]@B[K,N], K split
                                across NUM_IN_TILES tiles per output block;
                                reduce_to_one on the in-axis returns the sum
                                to pick₀.
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


@triton.jit
def matmul_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_IN_TILES: tl.constexpr,   # number of K-shard tiles per output block
    work_slices: tl.constexpr,    # list of {"out": pid_out, "in": pid_in} per tile
):
    """Split-K matmul: C[M,N] = A[M,K] @ B[K,N], K split across tiles.

    Grid layout (flat 1D, NUM_OUT_TILES × NUM_IN_TILES tiles):
      tile_id = pid_out * NUM_IN_TILES + pid_in
      pid_out — selects the [BLOCK_M, BLOCK_N] output block (distribution loop)
      pid_in  — selects the K-shard (fixed per core)

    Each tile accumulates its K-shard partial and contributes it to a
    reduce_to_one on the in-axis.  pick₀ (pid_in==0) writes the result to C.
    The outer loop over out_blocks handles arbitrary M for the fixed grid.
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    pid_out = pid // NUM_IN_TILES
    pid_in  = pid %  NUM_IN_TILES

    K_SHARD: tl.constexpr = K // NUM_IN_TILES // BLOCK_K  # BLOCK_K tiles per shard

    out_blocks = tl.cdiv(M, BLOCK_M)
    num_out_cores = num_cores // NUM_IN_TILES
    out_blocks_per_core = tl.cdiv(out_blocks, num_out_cores)
    out_start = pid_out * out_blocks_per_core
    out_end   = tl.minimum(out_start + out_blocks_per_core, out_blocks)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    k_start = pid_in * K_SHARD

    for out in range(out_start, out_end):
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(k_start, k_start + K_SHARD):
            a_tile = a_desc.load([out * BLOCK_M, k * BLOCK_K])
            b_tile = b_desc.load([k * BLOCK_K, 0])
            acc = tl.dot(a_tile, b_tile, acc)

        partial = tl.reshape(acc, [1, BLOCK_M, BLOCK_N])

        result = tl.inter_tile(
            partial,
            axis="in",
            combiner="add",
            mode="reduce_to_one",
            work_slices=work_slices,
        )

        if pid_in == 0:
            c_desc.store([out * BLOCK_M, 0], result)
