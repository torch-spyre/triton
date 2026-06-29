"""Inter-tile reduction kernels: all_reduce and reduce_to_one variants.

Kernels in this module:

all_reduce:
  inter_tile_add_kernel      -- ADD all_reduce (f32): flat 1D grid, 8 tiles,
                                4 row-groups × 2 x-tiles per group.
  inter_tile_add_kernel_f16  -- same as above but fp16.
  softmax_inter_tile         -- row-wise softmax (f16): 32 tiles, 2 out-groups
                                × 16 mb-tiles; two all-reduces (rowmax, rowsum)
                                across the mb-cohort.

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
    work_slices: tl.constexpr,  # {tile_id: {"x": pid_m, "n": pid_n}} for all tiles
):
    """Each tile reduces its local row-block, then participates in a cross-tile
    all_reduce along the x-axis so every tile in the group receives the sum.

    Data layout (flat 1D grid of 8 tiles):
      tile_id = pid_m * NUM_N_TILES + pid_n
      pid_m  ∈ {0..3}   → selects the BLOCK_M row-block (= "x" group label)
      pid_n  ∈ {0..1}   → selects the BLOCK_N column-block (= "n" within-group)

    Coordinates are recovered from work_slices via tl.wk_slice_coord rather than
    the manual pid // / pid % radix, keeping the kernel topology-independent
    (spec E4).
    """
    pid_m = tl.wk_slice_coord(work_slices, "x")  # row index (group label)
    pid_n = tl.wk_slice_coord(work_slices, "n")  # column index

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
    pid_m = tl.wk_slice_coord(work_slices, "x")
    pid_n = tl.wk_slice_coord(work_slices, "n")

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
    NUM_IN_TILES: tl.constexpr,  # K-shards per output block
    work_slices: tl.constexpr,   # list of {"out": pid_out, "in": pid_in} per tile
):
    """Split-K matmul: C[M,N] = A[M,K] @ B[K,N], K split across tiles.

    Grid layout (flat 1D, NUM_OUT_TILES × NUM_IN_TILES tiles):
      tile_id = pid_out * NUM_IN_TILES + pid_in  ("out" outer / slowest)
      work_slices[t] = {out: t//NUM_IN_TILES, in: t%NUM_IN_TILES}
      pid_out — selects the [BLOCK_M, BLOCK_N] output block (fixed per tile)
      pid_in  — selects the K-shard (fixed per tile)

    The slice coordinates are recovered from ``work_slices`` itself via
    ``tl.wk_slice_coord`` rather than the manual ``pid // NUM_IN_TILES`` /
    ``pid % NUM_IN_TILES`` radix.  This keeps the kernel correct-by-construction
    for any ``work_slices`` topology (spec E4): there is no hard-coded radix to
    drift out of sync with the layout the pass reads.

    axis="out": groups are formed by tiles with the same "out" slice value.
    Each group contains NUM_IN_TILES K-shard tiles for one output block.
    reduce_to_one delivers the partial sum to pick₀ = tile with work_slices["in"]==0
    within each group, i.e., pid_in==0.  Those tiles write the result to C.

    Contiguous groups: out=0 → tiles {0..NUM_IN_TILES-1}, out=1 → next range, etc.
    """
    pid_out = tl.wk_slice_coord(work_slices, "out")
    pid_in  = tl.wk_slice_coord(work_slices, "in")

    K_SHARD: tl.constexpr = K // NUM_IN_TILES // BLOCK_K  # BLOCK_K tiles per shard

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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(k_start, k_start + K_SHARD):
        a_tile = a_desc.load([pid_out * BLOCK_M, k * BLOCK_K])
        b_tile = b_desc.load([k * BLOCK_K, 0])
        acc = tl.dot(a_tile, b_tile, acc)

    partial = tl.reshape(acc, [1, BLOCK_M, BLOCK_N])

    result = tl.inter_tile(
        partial,
        axis="out",
        combiner="add",
        mode="reduce_to_one",
        work_slices=work_slices,
    )

    if pid_in == 0:
        c_desc.store([pid_out * BLOCK_M, 0], result)


@triton.jit
def softmax_inter_tile(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_ROWS: tl.constexpr,   # rows per core (M / NUM_OUT_TILES)
    BLOCK_COLS: tl.constexpr,   # cols per core (N / NUM_MB_TILES)
    NUM_MB_TILES: tl.constexpr, # number of mb-tiles (column-block tiles per row-group)
    work_slices: tl.constexpr,  # list of {"mb": pid_mb, "out": pid_out} per tile
):
    """Row-wise softmax with two inter-tile all-reduces (rowmax, rowsum).

    Grid layout (flat 1D, NUM_OUT_TILES × NUM_MB_TILES tiles):
      tile_id = pid_out * NUM_MB_TILES + pid_mb
      pid_out — selects the BLOCK_ROWS row-block (row group)
      pid_mb  — selects the BLOCK_COLS column-block within the row-group

    axis="out": tiles sharing the same "out" label cooperate (16-core mb-cohort).
    The partial passed to tl.inter_tile has a unit leading dim [1, BLOCK_ROWS]
    which LowerInterTile collapses to [BLOCK_ROWS] in the result.

    Coordinates are recovered from work_slices via tl.wk_slice_coord rather than
    the manual pid // / pid % radix (spec E4).
    """
    pid_out = tl.wk_slice_coord(work_slices, "out")
    pid_mb  = tl.wk_slice_coord(work_slices, "mb")

    row0 = pid_out * BLOCK_ROWS
    col0 = pid_mb  * BLOCK_COLS

    in_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_ROWS, BLOCK_COLS],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_ROWS, BLOCK_COLS],
    )

    x = in_desc.load([row0, col0])
    x_f32 = x.to(tl.float32)

    local_max = tl.max(x_f32, axis=1)
    partial_max = tl.reshape(local_max, [1, BLOCK_ROWS])
    rowmax = tl.inter_tile(
        partial_max,
        axis="out",
        combiner="max",
        mode="all_reduce",
        work_slices=work_slices,
    )

    x_shifted = x_f32 - tl.reshape(rowmax, [BLOCK_ROWS, 1])
    exp_x = tl.exp(x_shifted)

    local_sum = tl.sum(exp_x, axis=1)
    partial_sum = tl.reshape(local_sum, [1, BLOCK_ROWS])
    rowsum = tl.inter_tile(
        partial_sum,
        axis="out",
        combiner="add",
        mode="all_reduce",
        work_slices=work_slices,
    )

    softmax_out = exp_x / tl.reshape(rowsum, [BLOCK_ROWS, 1])
    out_desc.store([row0, col0], softmax_out.to(tl.float16))
