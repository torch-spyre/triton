"""Split-K matmul kernel: inter-tile reduction over the K dimension.

The K dimension is split across ``NUM_IN_TILES`` tiles.  Each tile
accumulates the dot products for its K-shard and contributes a partial
``[BLOCK_M × BLOCK_N]`` result.  A ``tl.inter_tile`` with
``mode="reduce_to_one"`` sums those partials; the designated consumer tile
(``in``-slice 0, i.e. ``pick₀``) writes the final result to C.

Grid layout (flat 1D, ``NUM_OUT_TILES × NUM_IN_TILES`` tiles):
  tile_id  = pid_out * NUM_IN_TILES + pid_in
  pid_out  ∈ {0 .. NUM_OUT_TILES - 1}  → selects the [BLOCK_M, BLOCK_N] output block
  pid_in   ∈ {0 .. NUM_IN_TILES  - 1}  → selects the K-shard

``work_slices`` maps each tile_id to ``{"out": pid_out, "in": pid_in}``.
The ``LowerInterTile`` pass reads W and C from the op attributes to derive
the reduction groups and ``pick₀``.

Only the tile with ``pid_in == 0`` stores to C (it is ``pick₀`` for each
out-group, since ``C[tile]["in"] == 0``).

Distribution loop: each tile handles ``out_blocks_per_core`` output blocks.
For M larger than ``NUM_OUT_TILES * BLOCK_M`` the trip count increases;
the fixed grid and ``work_slices`` are unchanged.
"""

import triton
import triton.language as tl


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
    work_slices: tl.constexpr,    # {tile_id: {"out": pid_out, "in": pid_in}}
):
    """Split-K matmul: C[M,N] = A[M,K] @ B[K,N], K split across tiles.

    Each tile handles one K-shard across all its assigned output blocks.
    The outer loop distributes output blocks across tiles; for each output
    block the tile accumulates its K-shard partial and reduces across the
    in-axis so pick₀ writes C.
    """
    pid = tl.program_id(0)           # flat tile id
    num_cores = tl.num_programs(0)
    pid_out = pid // NUM_IN_TILES    # base output-block index
    pid_in  = pid %  NUM_IN_TILES    # K-shard index

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
        # Accumulate over this tile's K-shard.
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(k_start, k_start + K_SHARD):
            a_tile = a_desc.load([out * BLOCK_M, k * BLOCK_K])
            b_tile = b_desc.load([k * BLOCK_K, 0])
            acc = tl.dot(a_tile, b_tile, acc)

        # Add a unit leading dimension so inter_tile_reduce can collapse it.
        partial = tl.reshape(acc, [1, BLOCK_M, BLOCK_N])

        # Cross-tile reduction over the in-axis: pick₀ (pid_in==0) receives sum.
        result = tl.inter_tile(
            partial,
            axis="in",
            combiner="add",
            mode="reduce_to_one",
            work_slices=work_slices,
        )

        # Only the designated consumer (pid_in == 0) stores to C.
        if pid_in == 0:
            c_desc.store([out * BLOCK_M, 0], result)
