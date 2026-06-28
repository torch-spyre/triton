"""SIGNATURE + VARIANTS for the matmul_splitk fixture.

Split-K matmul: C[M,N] = A[M,K] @ B[K,N] with the K dimension split across
``NUM_IN_TILES`` tiles per output block.  Each tile accumulates a K-shard
partial and contributes it to a ``tl.inter_tile`` reduce-to-one along the
``in``-axis.  The designated consumer tile (``in``-slice 0, ``pick₀``) writes
the final result to C.

Grid layout (flat 1D, ``NUM_OUT_TILES × NUM_IN_TILES`` tiles):
  tile_id = pid_out * NUM_IN_TILES + pid_in
  work_slices[tile_id] = {"out": pid_out, "in": pid_in}

Example (default variant): M=32, K=32, N=16, BLOCK_M=16, BLOCK_K=8, BLOCK_N=16
  NUM_OUT_TILES=2 (two BLOCK_M=16 output row blocks)
  NUM_IN_TILES=2  (K=32 split into two K-shards of K_SHARD=2 BLOCK_K tiles each)
  4 flat tiles total:
    tile 0: pid_out=0, pid_in=0  → K rows  0-15, output rows  0-15
    tile 1: pid_out=0, pid_in=1  → K rows 16-31, output rows  0-15
    tile 2: pid_out=1, pid_in=0  → K rows  0-15, output rows 16-31
    tile 3: pid_out=1, pid_in=1  → K rows 16-31, output rows 16-31

See ``fixtures/README.md`` for the field reference.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

_NUM_OUT_TILES = 2   # M / BLOCK_M
_NUM_IN_TILES  = 2   # K-shard count
_NUM_TILES     = _NUM_OUT_TILES * _NUM_IN_TILES  # 4 flat tiles

# work_slices: list indexed by tile_id; each entry is the full slice-index dict.
# Tiles with the same "in" value form a group (same out-block, cooperate on K-sum).
# tile_id = pid_out * NUM_IN_TILES + pid_in  →  group label = pid_out  →  in = pid_out
# Groups: {0,1} (in=0), {2,3} (in=1) — each out-block's K-shard tiles.
_WORK_SLICES = [
    {"out": t % _NUM_IN_TILES, "in": t // _NUM_IN_TILES}
    for t in range(_NUM_TILES)
]

# ---------------------------------------------------------------------------
# Reference oracle + input builder
# ---------------------------------------------------------------------------

def make_inputs(M: int, K: int, N: int, **_kw) -> dict:
    """Build A[M,K], B[K,N], C[M,N] buffers for matmul_splitk_kernel."""
    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)
    return {"a_ptr": a, "b_ptr": b, "c_ptr": c}


def run(inputs: dict) -> np.ndarray:
    """Oracle: standard 2D matmul."""
    return inputs["a_ptr"] @ inputs["b_ptr"]


# ---------------------------------------------------------------------------
# extra_checks: pin the exact KTIR lowering structure
# ---------------------------------------------------------------------------

def _extra_checks_default(tester) -> None:
    tester.assert_present("ktdp.inter_tile_produce")
    tester.assert_present("ktdp.inter_tile_reduce")

    # producer_tiles_per_group: (i)[g] with 1 dim, 1 symbol, 2 constraints
    tester.assert_integer_set(
        "ktdp.inter_tile_produce", "producer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=2,
    )
    tester.assert_integer_set(
        "ktdp.inter_tile_produce", "groups",
        num_dims=1, num_symbols=0, num_constraints=2,
    )
    tester.assert_result_type(
        "ktdp.inter_tile_produce", "ktdp.tile_future<tensor<1x16x16x",
    )

    # consumer_tiles_per_group: reduce_to_one — single equality constraint (pick₀ only)
    tester.assert_integer_set(
        "ktdp.inter_tile_reduce", "consumer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=1,
    )
    tester.assert_integer_set(
        "ktdp.inter_tile_reduce", "groups",
        num_dims=1, num_symbols=0, num_constraints=2,
    )
    tester.assert_result_type("ktdp.inter_tile_reduce", "tensor<16x16x")
    tester.assert_result("ktdp.inter_tile_reduce", shape=[16, 16])
    tester.assert_present("linalg.add", parent="ktdp.inter_tile_reduce")
    tester.assert_present("ktdp.yield_partial", parent="ktdp.inter_tile_produce")
    tester.assert_present("ktdp.yield_reduced", parent="ktdp.inter_tile_reduce")
    tester.assert_absent("tt.inter_tile_reduce")


# ---------------------------------------------------------------------------
# SIGNATURE
# ---------------------------------------------------------------------------

SIGNATURE = {
    "a_ptr":        "*fp32",
    "b_ptr":        "*fp32",
    "c_ptr":        "*fp32",
    "M":            "i32",
    "K":            "i32",
    "N":            "i32",
    "BLOCK_M":      "i32",
    "BLOCK_K":      "i32",
    "BLOCK_N":      "i32",
    "NUM_IN_TILES": "i32",
    "work_slices":  None,   # constexpr dict — not a KTIR runtime arg
}

# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["split-k", "inter-tile-reduce-to-one"],
        "summary": (
            "Split-K matmul: K split across 2 tiles per output block; "
            "reduce_to_one on the in-axis returns the sum to pick₀."
        ),
        "doc": (
            "Demonstrates the inter-tile ``reduce_to_one`` pattern on a "
            "2D matmul.  The K dimension is split across ``NUM_IN_TILES=2`` "
            "tiles; each tile accumulates a K-shard partial and contributes "
            "it to a cross-tile reduction.  Only the designated consumer tile "
            "(``in``-slice 0, ``pick₀``) writes the fully-reduced result to C.\n\n"
            "Grid: 4 flat tiles (2 output blocks × 2 K-shards).\n"
            "``work_slices`` is a compile-time constexpr encoding ``W`` and "
            "``C`` for the ``LowerInterTile`` pass."
        ),
        "kernel_fn":    kernel.matmul_splitk_kernel,
        "constexpr":    ["BLOCK_M", "BLOCK_K", "BLOCK_N", "NUM_IN_TILES", "work_slices"],
        "params": {
            "M":            [32],
            "K":            [32],
            "N":            [16],
            "BLOCK_M":      [16],
            "BLOCK_K":      [8],
            "BLOCK_N":      [16],
            "NUM_IN_TILES": [_NUM_IN_TILES],
            "work_slices":  [_WORK_SLICES],
        },
        "grid":         [_NUM_TILES],
        "parallel":     True,
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "c_ptr",
        "rtol":         1e-3,
        "extra_checks": _extra_checks_default,
    },
}
