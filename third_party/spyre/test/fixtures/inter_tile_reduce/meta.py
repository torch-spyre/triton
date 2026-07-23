"""SIGNATURE + VARIANTS for the inter_tile_reduce fixture.

Variants:

all_reduce (default, f16):
  Flat 1D grid of 8 tiles, 4 row-groups × 2 x-tiles per group.  Every tile
  loads a column-block partial; all tiles in the same group cooperate via
  ``tl.inter_tile(mode="all_reduce")`` so each receives the fully-reduced sum.
  ``work_slices[tile_id]["x"]`` is the group label (0–3); ``W["x"]=4``,
  ``gsize=2``.

reduce_to_one (splitk):
  Split-K matmul C[M,N]=A[M,K]@B[K,N].  K split across ``NUM_IN_TILES=2``
  tiles per output block; each tile accumulates its K-shard partial and
  contributes it via ``tl.inter_tile(mode="reduce_to_one")`` on the in-axis.
  Only pick₀ (``pid_in==0``) writes to C.  The outer distribution loop
  handles arbitrary M for the fixed grid.

See ``fixtures/README.md`` for the field reference.
"""

import functools

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

_NUM_M_GROUPS = 4   # independent row groups
_NUM_N_TILES  = 2   # tiles per group (gsize)
_NUM_TILES    = _NUM_M_GROUPS * _NUM_N_TILES  # 8 flat tiles

# work_slices: list indexed by tile_id; each entry is the full slice-index dict.
# tile_id = pid_m * _NUM_N_TILES + pid_n
#   "x" = pid_m  → group key (constant within group); groups {0,1}, {2,3}, {4,5}, {6,7}
#   "n" = pid_n  → reduction axis (tiles differing on "n" cooperate)
# Both coordinates are carried so the kernel can recover them via
# tl.wk_slice_coord instead of the manual pid // / pid % radix (spec E4).
_WORK_SLICES = [
    {"x": t // _NUM_N_TILES, "n": t % _NUM_N_TILES}
    for t in range(_NUM_TILES)
]

# ---------------------------------------------------------------------------
# Reference oracle + input builder
# ---------------------------------------------------------------------------

def make_inputs(M: int, N: int, BLOCK_M: int, BLOCK_N: int,
                NUM_N_TILES: int = _NUM_N_TILES,
                WORK_SLICES=None, **_kw) -> dict:
    """Build pointer-tensor inputs for inter_tile_add_kernel (f32)."""
    del BLOCK_M, BLOCK_N, NUM_N_TILES, WORK_SLICES, _kw
    rng = np.random.default_rng(42)
    x = rng.standard_normal((M, N)).astype(np.float32)
    output = np.zeros((M, N), dtype=np.float32)
    return {"x_ptr": x, "output_ptr": output}


def make_inputs_f16(M: int, N: int, BLOCK_M: int, BLOCK_N: int,
                    NUM_N_TILES: int = _NUM_N_TILES,
                    WORK_SLICES=None, **_kw) -> dict:
    """Build pointer-tensor inputs for inter_tile_add_kernel (f16)."""
    del BLOCK_M, BLOCK_N, NUM_N_TILES, WORK_SLICES, _kw
    rng = np.random.default_rng(42)
    x = rng.standard_normal((M, N)).astype(np.float16)
    output = np.zeros((M, N), dtype=np.float16)
    return {"x_ptr": x, "output_ptr": output}


def run(inputs: dict, BLOCK_M: int, BLOCK_N: int, NUM_N_TILES: int, **_kw) -> np.ndarray:
    """Oracle for this fixture's specific use case: column-block element-wise
    add-reduce across NUM_N_TILES tiles in each row-group.

    NOTE: This oracle is tuned to what ``tl.inter_tile(..., combiner="add",
    mode="all_reduce")`` *currently lowers to* for this fixture — a
    ``linalg.add`` over the ``1×BLOCK_M×BLOCK_N`` partials, collapsing the
    unit leading dim.  It does NOT implement a general row-sum: the result
    for each tile is the element-wise sum of all column-blocks in its group,
    written back to each tile's own ``BLOCK_M × BLOCK_N`` output region.

    If the ``tl.inter_tile`` API or the fixture grid changes this oracle must
    be updated to match.  Use ``extra_checks`` in VARIANTS to pin the exact
    KTIR structure (affine sets, combiner op) so regressions surface there
    before the numerical oracle becomes the only signal.
    """
    x = inputs["x_ptr"]
    M, N = x.shape
    NMG = M // BLOCK_M
    # reshape → (NMG, BLOCK_M, NUM_N_TILES, BLOCK_N)
    # sum over column-tile axis → (NMG, BLOCK_M, BLOCK_N)
    # replicate NUM_N_TILES times along the column axis → (M, N)
    x4 = x.reshape(NMG, BLOCK_M, NUM_N_TILES, BLOCK_N)
    col_sum = x4.sum(axis=2)
    return np.tile(col_sum, (1, 1, NUM_N_TILES)).reshape(M, N)


# ---------------------------------------------------------------------------
# f16 oracle (thin wrapper — same reshape logic, cast to f16)
# ---------------------------------------------------------------------------

def run_f16(inputs: dict, BLOCK_M: int, BLOCK_N: int, NUM_N_TILES: int, **_kw) -> np.ndarray:
    """f16 oracle: same column-block add-reduce as ``run``, cast to fp16."""
    return run(inputs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_N_TILES=NUM_N_TILES).astype(np.float16)


# ---------------------------------------------------------------------------
# extra_checks: pin the exact KTIR lowering structure
# ---------------------------------------------------------------------------

def _extra_checks_default(tester) -> None:
    """Pin the KTIR structure emitted by LowerInterTile for this fixture.

    Asserts the exact op pattern that ``tl.inter_tile(..., combiner="add",
    mode="all_reduce")`` lowers to, given:
      - partial shape  BLOCK_M × BLOCK_N
      - 4 groups of 2 tiles  (ngroups=4, gsize=2)
      - reduce-then-fold with linalg.add combiner

    The delivery op preserves shape: ``result_type == partial_type``;
    grouping lives in the affine sets, not in a tensor axis.  If the
    lowering changes (different combiner, different affine-set shape, or
    rank drift), this check will fail and prompt a review of both the
    pass and this oracle.
    """
    # Both produce and reduce ops must be present.
    tester.assert_present("ktdp.inter_tile_produce")
    tester.assert_present("ktdp.inter_tile_reduce")

    # Produce op: producer_tiles_per_group is (i)[g] with 1 symbol, 1 dim,
    # 2 constraints (lower + upper bound: g*gsize <= i <= g*gsize + gsize - 1).
    tester.assert_integer_set(
        "ktdp.inter_tile_produce", "producer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=2,
    )
    # groups set: (g) with 1 dim, 0 symbols, 2 constraints (0<=g<=ngroups-1).
    # Post PR-25: groups lives on the !ktdp.tile_future type, not on the op.
    tester.assert_tile_future_groups(
        "ktdp.inter_tile_produce",
        num_dims=1, num_symbols=0, num_constraints=2,
    )

    # Produce result type: tile_future carrying the 16×16 partial. The
    # tile_future syntax wraps the partial types in parens:
    # !ktdp.tile_future<(tensor<...>), groups=...>.
    tester.assert_result_type(
        "ktdp.inter_tile_produce", "ktdp.tile_future<(tensor<16x16x",
    )

    # Reduce op: still carries consumer_tiles_per_group; groups is inferred
    # from its !tile_future operand type.
    tester.assert_integer_set(
        "ktdp.inter_tile_reduce", "consumer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=2,
    )
    tester.assert_tile_future_groups(
        "ktdp.inter_tile_reduce",
        num_dims=1, num_symbols=0, num_constraints=2,
    )

    # Reduce result type equals the partial type (16×16), no rank reduction.
    tester.assert_result_type("ktdp.inter_tile_reduce", "tensor<16x16x")
    tester.assert_result(
        "ktdp.inter_tile_reduce", shape=[16, 16],
    )

    # Combiner region: linalg.add inside the reduce op's region.
    tester.assert_present("linalg.add", parent="ktdp.inter_tile_reduce")

    # yield_partial inside produce region; yield_reduced inside reduce region.
    tester.assert_present("ktdp.yield_partial", parent="ktdp.inter_tile_produce")
    tester.assert_present("ktdp.yield_reduced", parent="ktdp.inter_tile_reduce")

    # No raw inter_tile tt ops remain.
    tester.assert_absent("tt.inter_tile_reduce")


# ---------------------------------------------------------------------------
# SIGNATURE
# ---------------------------------------------------------------------------

SIGNATURE = {
    "x_ptr":       "*fp32",
    "output_ptr":  "*fp32",
    "M":           "i32",
    "N":           "i32",
    "BLOCK_M":     "i32",
    "BLOCK_N":     "i32",
    "NUM_N_TILES": "i32",
    "WORK_SLICES": None,   # constexpr dict — not a KTIR runtime arg
}

# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["all-reduce"],
        "summary": (
            "Multi-group ADD all_reduce (f32): 8 tiles × 4 row-groups × 2 "
            "x-tiles; every tile in a group receives the fully-reduced sum."
        ),
        "doc": (
            "A flat-1D-grid kernel (8 tiles) that loads a ``BLOCK_M × BLOCK_N`` "
            "column-block from a ``M × N`` matrix, then performs a cross-tile "
            "all_reduce along the x-axis.  Tiles are grouped into 4 row-groups "
            "of 2; ``LowerInterTile`` lowers the ``tl.inter_tile`` op to a "
            "``ktdp.inter_tile_produce`` + ``ktdp.inter_tile_reduce`` pair with "
            "a ``linalg.add`` combiner over the ``BLOCK_M×BLOCK_N`` partials.\n\n"
            "The result for each tile is the element-wise sum of the two "
            "column-blocks in its group; the delivery op preserves the "
            "partial shape (no rank reduction).  ``WORK_SLICES`` is a "
            "compile-time list; "
            "``LowerInterTile`` derives groups by equality of the slice dict."
        ),
        "kernel_fn":    kernel.inter_tile_add_kernel,
        "constexpr":    ["BLOCK_M", "BLOCK_N", "NUM_N_TILES", "WORK_SLICES"],
        "params": {
            "M":           [64],
            "N":           [32],
            "BLOCK_M":     [16],
            "BLOCK_N":     [16],
            "NUM_N_TILES": [_NUM_N_TILES],
            "WORK_SLICES": [_WORK_SLICES],
        },
        "grid":          [_NUM_TILES],
        "parallel":      False,  # one block per tile, no distribution loop
        "reference":     functools.partial(run, BLOCK_M=16, BLOCK_N=16, NUM_N_TILES=_NUM_N_TILES),
        "inputs":        make_inputs,
        "output_key":    "output_ptr",
        "extra_checks":  _extra_checks_default,
    },
    "f16": {
        "tags": ["all-reduce", "f16"],
        "summary": (
            "Multi-group ADD all_reduce (f16): same topology as default but "
            "accumulating in half precision."
        ),
        "doc": (
            "f16 variant of the default all_reduce fixture.  Same lowering "
            "structure as ``default`` but with ``fp16`` pointer types.  "
            "The numerical oracle computes in f32 then converts to f16; "
            "the kernel accumulates in f16, so ``rtol=1e-2`` is used."
        ),
        "kernel_fn":    kernel.inter_tile_add_kernel_f16,
        "SIGNATURE": {
            "x_ptr":       "*fp16",
            "output_ptr":  "*fp16",
            "M":           "i32",
            "N":           "i32",
            "BLOCK_M":     "i32",
            "BLOCK_N":     "i32",
            "NUM_N_TILES": "i32",
            "WORK_SLICES": None,
        },
        "constexpr":    ["BLOCK_M", "BLOCK_N", "NUM_N_TILES", "WORK_SLICES"],
        "params": {
            "M":           [64],
            "N":           [32],
            "BLOCK_M":     [16],
            "BLOCK_N":     [16],
            "NUM_N_TILES": [_NUM_N_TILES],
            "WORK_SLICES": [_WORK_SLICES],
        },
        "grid":          [_NUM_TILES],
        "parallel":      False,
        "reference":     functools.partial(run_f16, BLOCK_M=16, BLOCK_N=16, NUM_N_TILES=_NUM_N_TILES),
        "inputs":        make_inputs_f16,
        "output_key":    "output_ptr",
        "rtol":          1e-2,
        "extra_checks":  _extra_checks_default,
    },
}

# ---------------------------------------------------------------------------
# splitk variant — reduce_to_one
# ---------------------------------------------------------------------------

_SK_NUM_OUT_TILES = 2   # output blocks (M / BLOCK_M)
_SK_NUM_IN_TILES  = 2   # K-shard count
_SK_NUM_TILES     = _SK_NUM_OUT_TILES * _SK_NUM_IN_TILES  # 4

# tile_id = pid_out * _SK_NUM_IN_TILES + pid_in  ("out" outer / slowest)
# axis="in": tiles that differ on "in" (K-shard) but share the same "out" cooperate.
# Group out=0 → tiles {0,1}, group out=1 → tiles {2,3}  (contiguous ✓).
# reduce_to_one pick₀ = tile with C[t]["in"]==0 per group = tile 0 + tile 2.
# pick₀ tiles have work_slices["in"]==0 ↔ pid_in==0, so kernel guard matches.
# dfir_fixtures/splitk_M64_N512_K6144: out=8 groups × in=4 shards; same shape.
_SK_WORK_SLICES = [
    {"out": t // _SK_NUM_IN_TILES, "in": t % _SK_NUM_IN_TILES}
    for t in range(_SK_NUM_TILES)
]


def make_inputs_splitk(M: int, K: int, N: int, **_kw) -> dict:
    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)
    return {"a_ptr": a, "b_ptr": b, "c_ptr": c}


def run_splitk(inputs: dict) -> np.ndarray:
    return inputs["a_ptr"] @ inputs["b_ptr"]


def _extra_checks_splitk(tester) -> None:
    tester.assert_present("ktdp.inter_tile_produce")
    tester.assert_present("ktdp.inter_tile_reduce")
    tester.assert_integer_set(
        "ktdp.inter_tile_produce", "producer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=2,
    )
    # Post PR-25: groups lives on the !ktdp.tile_future type.
    tester.assert_tile_future_groups(
        "ktdp.inter_tile_produce",
        num_dims=1, num_symbols=0, num_constraints=2,
    )
    tester.assert_result_type(
        "ktdp.inter_tile_produce", "ktdp.tile_future<(tensor<16x16x",
    )
    # reduce_to_one: single equality constraint — only pick₀ consumes
    tester.assert_integer_set(
        "ktdp.inter_tile_reduce", "consumer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=1,
    )
    tester.assert_tile_future_groups(
        "ktdp.inter_tile_reduce",
        num_dims=1, num_symbols=0, num_constraints=2,
    )
    tester.assert_result_type("ktdp.inter_tile_reduce", "tensor<16x16x")
    tester.assert_result("ktdp.inter_tile_reduce", shape=[16, 16])
    tester.assert_present("linalg.add", parent="ktdp.inter_tile_reduce")
    tester.assert_present("ktdp.yield_partial", parent="ktdp.inter_tile_produce")
    tester.assert_present("ktdp.yield_reduced", parent="ktdp.inter_tile_reduce")
    tester.assert_absent("tt.inter_tile_reduce")


# ---------------------------------------------------------------------------
# softmax variant — two all-reduces (rowmax + rowsum) across mb-cohort
# ---------------------------------------------------------------------------

_SM_NUM_MB_TILES  = 16
_SM_NUM_OUT_TILES = 2
_SM_NUM_TILES     = _SM_NUM_MB_TILES * _SM_NUM_OUT_TILES  # 32

# tile_id = pid_out * _SM_NUM_MB_TILES + pid_mb
# axis="mb": tiles that differ on "mb" (column-block) but share "out" cooperate;
#   group g (out=g) owns tiles g*16 .. g*16+15 (contiguous).
_SM_WORK_SLICES = [
    {"mb": t % _SM_NUM_MB_TILES, "out": t // _SM_NUM_MB_TILES}
    for t in range(_SM_NUM_TILES)
]


def make_inputs_softmax(M: int, N: int, **_kw) -> dict:
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((M, N)).astype(np.float16)
    return {"input_ptr": x, "output_ptr": np.zeros((M, N), dtype=np.float16)}


def run_softmax(inputs: dict) -> np.ndarray:
    x = inputs["input_ptr"].astype(np.float32)
    x_shifted = x - x.max(axis=1, keepdims=True)
    num = np.exp(x_shifted)
    return (num / num.sum(axis=1, keepdims=True)).astype(np.float16)


def _extra_checks_softmax(tester) -> None:
    tester.assert_present("ktdp.inter_tile_produce")
    tester.assert_present("ktdp.inter_tile_reduce")
    tester.assert_count("ktdp.inter_tile_produce", 2)
    tester.assert_count("ktdp.inter_tile_reduce", 2)
    tester.assert_integer_set(
        "ktdp.inter_tile_reduce", "consumer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=2,
    )
    # Post PR-25: groups lives on the !ktdp.tile_future operand type.
    tester.assert_tile_future_groups(
        "ktdp.inter_tile_reduce",
        num_dims=1, num_symbols=0, num_constraints=2,
    )
    tester.assert_result_type("ktdp.inter_tile_reduce", "tensor<256x")
    tester.assert_absent("tt.inter_tile_reduce")


VARIANTS["softmax"] = {
    "tags": ["all-reduce", "double-all-reduce", "work-slices"],
    "summary": (
        "Row-wise softmax (f16): two all-reduces (rowmax + rowsum) across "
        "a 16-core mb-cohort; 32 tiles, 2 out-groups × 16 mb-tiles."
    ),
    "doc": (
        "Each core owns a [BLOCK_ROWS, BLOCK_COLS] column-block. "
        "Two cross-tile all-reduces over the mb-axis produce the true "
        "per-row max and sum from partial column-block values. The grid "
        "exactly covers [M, N] so no distribution loop is needed."
    ),
    "kernel_fn": kernel.softmax_inter_tile,
    "SIGNATURE": {
        "output_ptr":    "*fp16",
        "input_ptr":     "*fp16",
        "M":             "i32",
        "N":             "i32",
        "BLOCK_ROWS":    "i32",
        "BLOCK_COLS":    "i32",
        "NUM_MB_TILES":  "i32",
        "WORK_SLICES":   None,
    },
    "constexpr":   ["BLOCK_ROWS", "BLOCK_COLS", "NUM_MB_TILES", "WORK_SLICES"],
    "params": {
        "M":            [512],
        "N":            [1024],
        "BLOCK_ROWS":   [256],
        "BLOCK_COLS":   [64],
        "NUM_MB_TILES": [_SM_NUM_MB_TILES],
        "WORK_SLICES":  [_SM_WORK_SLICES],
    },
    "grid":        [_SM_NUM_TILES],
    "parallel":    False,
    "reference":   run_softmax,
    "inputs":      make_inputs_softmax,
    "output_key":  "output_ptr",
    "rtol":        1e-2,
    "extra_checks": _extra_checks_softmax,
}


VARIANTS["splitk"] = {
    "tags": ["reduce-to-one", "work-slices"],
    "summary": (
        "Split-K matmul (f32): K split across 2 in-tiles per output block; "
        "reduce_to_one on the in-axis returns the sum to pick₀."
    ),
    "doc": (
        "Demonstrates the inter-tile ``reduce_to_one`` pattern on a 2D matmul. "
        "The K dimension is split across ``NUM_IN_TILES=2`` K-shard tiles per "
        "output block; each tile accumulates its K-shard partial and contributes "
        "it to a cross-tile reduction.  Only pick₀ (``pid_in==0``) writes the "
        "result to C.\n\n"
        "Grid: 4 flat tiles (2 output blocks × 2 K-shards).  "
        "``WORK_SLICES[t] = {out: t//2, in: t%2}`` — ``out`` is outermost so "
        "tiles with the same ``out`` are contiguous: out=0 → tiles {0,1}, "
        "out=1 → {2,3}.  ``axis='in'`` names the reduction dim (K-shard); tiles "
        "differing only on ``in`` cooperate. pick₀ per group = tile with ``in==0`` "
        "(pid_in==0), which writes the result to the corresponding output block.\n\n"
        "The production split-K case (``dfir_fixtures/splitk_M64_N512_K6144``) "
        "has the same shape: 8 out-groups × 4 in-shards, with ``out`` outermost."
    ),
    "kernel_fn":    kernel.matmul_splitk_kernel,
    "SIGNATURE": {
        "a_ptr":         "*fp32",
        "b_ptr":         "*fp32",
        "c_ptr":         "*fp32",
        "M":             "i32",
        "K":             "i32",
        "N":             "i32",
        "BLOCK_M":       "i32",
        "BLOCK_K":       "i32",
        "BLOCK_N":       "i32",
        "NUM_IN_TILES":  "i32",
        "WORK_SLICES":   None,
    },
    "constexpr":    ["BLOCK_M", "BLOCK_K", "BLOCK_N", "NUM_IN_TILES", "WORK_SLICES"],
    "params": {
        "M":             [32],
        "K":             [32],
        "N":             [16],
        "BLOCK_M":       [16],
        "BLOCK_K":       [8],
        "BLOCK_N":       [16],
        "NUM_IN_TILES":  [_SK_NUM_IN_TILES],
        "WORK_SLICES":   [_SK_WORK_SLICES],
    },
    "grid":         [_SK_NUM_TILES],
    "parallel":     True,
    "reference":    run_splitk,
    "inputs":       make_inputs_splitk,
    "output_key":   "c_ptr",
    "rtol":         1e-3,
    "extra_checks": _extra_checks_splitk,
}
