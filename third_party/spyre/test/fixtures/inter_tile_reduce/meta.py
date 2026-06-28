"""SIGNATURE + VARIANTS for the inter_tile_reduce fixture.

Multi-group ADD all_reduce: a 1D-grid kernel where each flat tile (out of 8)
loads its column-block partial and participates in a cross-tile all_reduce
along the x-axis.  Tiles are grouped into 4 row-groups of 2; every tile in a
group receives the fully-reduced sum.

Grid layout (flat 1D, 8 tiles):
  tile 0: pid_m=0, pid_n=0  → x-slice 0
  tile 1: pid_m=0, pid_n=1  → x-slice 1
  tile 2: pid_m=1, pid_n=0  → x-slice 0
  ...
  tile 7: pid_m=3, pid_n=1  → x-slice 1

``work_slices`` is a list; tiles in the same group carry the same ``{"x": g}``
entry (group label).  ``W["x"]=4`` (4 distinct labels), ``gsize=2``.
``coreIdToWkSlice = [{x:0},{x:0},{x:1},{x:1},{x:2},{x:2},{x:3},{x:3}]``.

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
# Tiles in the same group carry the same slice dict (group label).
# tile_id = pid_m * _NUM_N_TILES + pid_n  →  group = pid_m  →  x-label = pid_m
# Groups: {0,1}, {2,3}, {4,5}, {6,7} (each pid_m pair)
_WORK_SLICES = [{"x": t // _NUM_N_TILES} for t in range(_NUM_TILES)]

# ---------------------------------------------------------------------------
# Reference oracle + input builder
# ---------------------------------------------------------------------------

def make_inputs(M: int, N: int, BLOCK_M: int, BLOCK_N: int,
                NUM_N_TILES: int = _NUM_N_TILES,
                work_slices=None, **_kw) -> dict:
    """Build pointer-tensor inputs for inter_tile_add_kernel (f32)."""
    del BLOCK_M, BLOCK_N, NUM_N_TILES, work_slices, _kw
    rng = np.random.default_rng(42)
    x = rng.standard_normal((M, N)).astype(np.float32)
    output = np.zeros((M, N), dtype=np.float32)
    return {"x_ptr": x, "output_ptr": output}


def make_inputs_f16(M: int, N: int, BLOCK_M: int, BLOCK_N: int,
                    NUM_N_TILES: int = _NUM_N_TILES,
                    work_slices=None, **_kw) -> dict:
    """Build pointer-tensor inputs for inter_tile_add_kernel (f16)."""
    del BLOCK_M, BLOCK_N, NUM_N_TILES, work_slices, _kw
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
      - partial shape  1 × BLOCK_M × BLOCK_N  (unit leading dim)
      - 4 groups of 2 tiles  (ngroups=4, gsize=2)
      - reduce-then-fold with linalg.add combiner

    These are structural invariants.  If the lowering changes (e.g. the
    pass emits a different combiner, collapses the unit dim differently,
    or uses a different affine-set shape), this check will fail and prompt
    a review of both the pass and this oracle.
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
    # groups set: (g) with 0 dims, 0 symbols, 2 constraints (0<=g<=ngroups-1).
    tester.assert_integer_set(
        "ktdp.inter_tile_produce", "groups",
        num_dims=1, num_symbols=0, num_constraints=2,
    )

    # Produce result type: tile_future carrying the 1×16×16 partial.
    tester.assert_result_type(
        "ktdp.inter_tile_produce", "ktdp.tile_future<tensor<1x16x16x",
    )

    # Reduce op: same affine-set shape for consumer_tiles_per_group / groups.
    tester.assert_integer_set(
        "ktdp.inter_tile_reduce", "consumer_tiles_per_group",
        num_dims=1, num_symbols=1, num_constraints=2,
    )
    tester.assert_integer_set(
        "ktdp.inter_tile_reduce", "groups",
        num_dims=1, num_symbols=0, num_constraints=2,
    )

    # Reduce result type: unit dim collapsed → 16×16 (not 1×16×16).
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
    "work_slices": None,   # constexpr dict — not a KTIR runtime arg
}

# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["inter-tile-all-reduce"],
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
            "a ``linalg.add`` combiner over the ``1×BLOCK_M×BLOCK_N`` partials.\n\n"
            "The result for each tile is the element-wise sum of the two "
            "column-blocks in its group (unit leading dim collapsed by the reduce "
            "result type).  ``work_slices`` is a compile-time list; "
            "``LowerInterTile`` derives groups by equality of the slice dict."
        ),
        "kernel_fn":    kernel.inter_tile_add_kernel,
        "constexpr":    ["BLOCK_M", "BLOCK_N", "NUM_N_TILES", "work_slices"],
        "params": {
            "M":           [64],
            "N":           [32],
            "BLOCK_M":     [16],
            "BLOCK_N":     [16],
            "NUM_N_TILES": [_NUM_N_TILES],
            "work_slices": [_WORK_SLICES],
        },
        "grid":          [_NUM_TILES],
        "parallel":      False,  # one block per tile, no distribution loop
        "reference":     functools.partial(run, BLOCK_M=16, BLOCK_N=16, NUM_N_TILES=_NUM_N_TILES),
        "inputs":        make_inputs,
        "output_key":    "output_ptr",
        "extra_checks":  _extra_checks_default,
    },
    "f16": {
        "tags": ["inter-tile-all-reduce", "f16"],
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
            "work_slices": None,
        },
        "constexpr":    ["BLOCK_M", "BLOCK_N", "NUM_N_TILES", "work_slices"],
        "params": {
            "M":           [64],
            "N":           [32],
            "BLOCK_M":     [16],
            "BLOCK_N":     [16],
            "NUM_N_TILES": [_NUM_N_TILES],
            "work_slices": [_WORK_SLICES],
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
