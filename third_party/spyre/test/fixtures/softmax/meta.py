"""SIGNATURE + VARIANTS + reference oracle + input generator for softmax.

Four variants, grouped under Level banners the way ``elementwise``,
``reduce`` and ``spyreop`` group theirs:

- **Level A/B — algorithm × shape on ``ktir_cpu``.** ``default``
  (single-tile), ``multi_tile`` and ``2pass``, all at ``M × N =
  1024 × 1024`` against the same oracle, differing only in how the row
  is tiled and swept: single tile with an in-tile reduce; 3-pass over
  N-tiles; 2-pass online (Milakov & Gimelshein). There is no Level C
  (layout) band — see the layout note below, which is why.
- **Level D — device.** ``one_tile_device``, the loop-free single-tile
  shape the other fixtures' device variants use. It is ``disabled``: the
  shape does not compile today, and the layout note below records the wall
  it hits first.

All four share ``make_inputs`` and ``run``.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np

from utils import sticksize

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(M: int, N: int, **_unused) -> dict:
    """Build ``[M, N]`` input + output buffers.

    Extra kwargs (``BLOCK_SIZE``, ``BLOCK_M``, ``BLOCK_N``) are ignored
    here — they shape tiling, not the data.
    """
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((M, N)).astype(np.float16)
    output = np.zeros((M, N), dtype=np.float16)
    return {"input_ptr": x, "output_ptr": output}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: row-wise softmax in f32, truncated to f16."""
    x = inputs["input_ptr"].astype(np.float32)
    x_shifted = x - x.max(axis=1, keepdims=True)
    num = np.exp(x_shifted)
    denom = num.sum(axis=1, keepdims=True)
    return (num / denom).astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE — module-level default (matches softmax_single_tile's arg list).
# Variants with a different arg list redeclare ``SIGNATURE`` inside their
# VARIANTS entry; the override replaces the module-level map wholesale.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "output_ptr": "*fp16",
    "input_ptr":  "*fp16",
    "M":          "i32",
    "N":          "i32",
    "BLOCK_SIZE": "i32",
}

#: ``softmax_one_tile_device``'s arg list: no BLOCK_SIZE (the block is the whole
#: tensor), and a layout per descriptor.
_SIG_DEVICE = {
    "output_ptr": "*fp16",
    "input_ptr":  "*fp16",
    "M":          "constexpr",
    "N":          "constexpr",
    "IN_LAYOUT":  "constexpr",
    "OUT_LAYOUT": "constexpr",
}


# ---------------------------------------------------------------------------
# Stick layout
#
# One helper, taking a dtype rather than a width, so the lane count follows from
# the element type at the single place it is written. Returns the
# ``("stick", layout)`` labelled pair, since a bare 3-element tuple in a
# ``params`` values list would be read as a ``(label, value)`` pair; the value
# inside stays a tuple because it reaches Triton as a constexpr.
# ---------------------------------------------------------------------------

def _stick_2d_on_n(dtype: str) -> tuple:
    """``[M, N]`` -> ``[ceil(N/S), M, S]``: stick on the row."""
    stick = sticksize({"p": f"*{dtype}"}, "p")
    return ("stick", ((1, "floordiv", stick), 0, (1, "mod", stick)))


# ---------------------------------------------------------------------------
# VARIANTS
#
# Level A/B and Level D, banner-separated below. The Level A/B trio all use the
# same input shape (M=1024, N=1024) and the same oracle, and split on tiling
# strategy:
#   - default = single_tile: full row in 1024-wide tile, in-tile reduce.
#   - multi_tile: 16 N-tiles of width 64, 3-pass per row.
#   - 2pass: 4-row × 64-col tiles, fused-max + online denom, 2-pass.
#
# 1D grid across all 32 cores: each of those three reads only
# tl.program_id(0) and partitions rows internally via an explicit
# rows_per_core loop.
#
# NO LEVEL C (layout on ktir_cpu) BAND, and the reason is the same one that
# keeps Level D disabled: annotating any of the three above with a
# tt.spyre_tensor_layout fails in RewriteDescriptorLayout, which physicalizes the
# loaded row to rank 3 while the broadcast of row_max stays rank 2, so
# `row - row_max` fails with "'arith.subf' op requires the same type for all
# operands and results". The pass does not re-derive broadcasts against the
# physical rank: `retypeChain` walks forward along operand 0 only, and the
# broadcast is a sibling operand of the subf whose producer traces back to the
# reduce, not to the physicalized load, so no forward walk can reach it.
#
# Splitting the reduced axis is NOT what makes this hard: stick-on-M (the
# non-reduced axis, with a multi-row block so the stick dim is not sub-stick)
# fails with the identical diagnostic. The reduce's `dimensions` are untouched
# there, so the stale broadcast target alone is sufficient to break it. Fixing
# just the broadcast re-derivation would therefore unblock a stick-on-M variant;
# multi-dim reduce is only additionally needed to split the reduced axis.
# ---------------------------------------------------------------------------

VARIANTS = {
    # -----------------------------------------------------------------------
    # Level A/B -- algorithm x shape on ktir_cpu
    #
    # Three tilings of the same function at one input shape, each with its own
    # structural claim. DTYPE is pinned at fp16 throughout: what these variants
    # are for is the sweep pattern, and every dtype tiles alike.
    # -----------------------------------------------------------------------
    "default": {
        # Single-tile: N fits in BLOCK_SIZE, no inner N-loop.
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce", "broadcast", "program-id-1d", "num-programs-fold"],
        "summary": (
            "Row-wise softmax where an entire row fits into a single "
            "on-chip tile — no inner N-loop."
        ),
        "doc": (
            "Computes row-wise softmax over an `M × N` input matrix. "
            "This variant assumes the row width `N` fits entirely into "
            "a single `BLOCK_SIZE`-wide tile, so each row is loaded "
            "once and the row-max / exponential / sum / normalize "
            "sequence runs in-tile with no outer N-loop. Rows are "
            "partitioned across the 32-core grid; each core walks its "
            "share of rows in a single pass.\n\n"
            "This is the simplest softmax variant, but also the most "
            "memory-constrained — it is only usable when `N` is small "
            "enough that a full row plus intermediates fit in the "
            "scratchpad."
        ),
        "kernel_fn":    kernel.softmax_single_tile,
        "constexpr":    ["BLOCK_SIZE"],
        "params":       {
            # M=[16,1000,1024]: absorbs few_rows (M=16) and nonaligned (M=1000).
            "M":          [16, 1000, 1024],
            "N":          [1024],
            "BLOCK_SIZE": [1024],
        },
        "grid":         [32],
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "output_ptr",
        # f16 softmax: computation is f32 internally but the result
        # round-trips through f16 for store and oracle — leave room.
        "rtol":         1e-3,
        "extra_checks": lambda t: (
            # In-tile reduce + broadcast: max and sum each emit one
            # linalg.reduce; the row_max - row subtraction emits a
            # broadcast from [1] to [1, BLOCK_SIZE].
            t.assert_count("linalg.reduce", 2),
            t.assert_present("linalg.broadcast"),
        ),
    },
    "multi_tile": {
        # 3-pass over n_tiles = N / BLOCK_N. Redeclares SIGNATURE because
        # multi_tile's kernel has BLOCK_N in place of BLOCK_SIZE.
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce", "program-id-1d", "num-programs-fold"],
        "summary": (
            "Row-wise softmax that tiles the row into `BLOCK_N`-wide "
            "chunks and sweeps the row three times."
        ),
        "doc": (
            "Computes row-wise softmax over an `M × N` input matrix "
            "when the row width exceeds what fits in a single tile. "
            "The row is split into `n_tiles = N / BLOCK_N` column "
            "chunks of width `BLOCK_N`, and each row is swept three "
            "times across its tiles:\n\n"
            "1. **row-max pass** — fold each chunk into a running max.\n"
            "2. **denominator pass** — compute `exp(x - row_max)` per "
            "chunk and fold into a running sum.\n"
            "3. **normalize pass** — recompute `exp(x - row_max) / "
            "denom` and store.\n\n"
            "This is slower than the single-tile variant but has no "
            "restriction on row width — the per-core working set is "
            "bounded by `BLOCK_N`, not by `N`."
        ),
        "kernel_fn": kernel.softmax_multi_tile,
        "SIGNATURE": {
            "output_ptr": "*fp16",
            "input_ptr":  "*fp16",
            "M":          "i32",
            "N":          "i32",
            "BLOCK_N":    "i32",
        },
        "constexpr": ["BLOCK_N"],
        "params":    {
            # M=[1000,1024]: absorbs multi_tile_nonaligned (M=1000).
            # BLOCK_N=[32,64]: absorbs multi_tile_small_block (BLOCK_N=32).
            "M": [1000, 1024], "N": [1024], "BLOCK_N": [32, 64],
        },
        "extra_checks": lambda t: (
            # Three nested scf.for in the kernel body: outer rows-per-core,
            # three inner N-tile passes (max, denom, normalize).
            t.assert_count("scf.for", 4, cmp="ge"),
        ),
    },
    "2pass": {
        # Online softmax: 2-pass, BLOCK_M × BLOCK_N tiled. Redeclares
        # SIGNATURE because 2pass adds BLOCK_M alongside BLOCK_N.
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce", "broadcast", "program-id-1d", "num-programs-fold"],
        "summary": (
            "Online (Milakov & Gimelshein) softmax: one fused "
            "max-plus-denom pass followed by a normalize pass."
        ),
        "doc": (
            "Computes row-wise softmax over an `M × N` input matrix "
            "using the online softmax algorithm — the row-max and the "
            "denominator are maintained together in a single pass that "
            "streams across the row's column tiles. When a new chunk "
            "produces a larger max, the running denominator is "
            "rescaled by `exp(old_max - new_max)` before folding the "
            "new chunk in. A second pass then normalizes and stores.\n\n"
            "The kernel is tiled as `BLOCK_M × BLOCK_N`, so multiple "
            "rows are processed in lockstep on each core. This is the "
            "fastest of the three variants when `N` exceeds a single "
            "tile but memory bandwidth, rather than compute, is the "
            "limiting factor."
        ),
        "kernel_fn": kernel.softmax_2pass,
        "SIGNATURE": {
            "output_ptr": "*fp16",
            "input_ptr":  "*fp16",
            "M":          "i32",
            "N":          "i32",
            "BLOCK_M":    "i32",
            "BLOCK_N":    "i32",
        },
        "constexpr": ["BLOCK_M", "BLOCK_N"],
        "params":    {"M": [1024], "N": [1024], "BLOCK_M": [4], "BLOCK_N": [64]},
        # Previously xfailed: the regex KTIRParser collapsed the multi-result
        # `%denom:2 = scf.for ... -> (T, T)` to a single value, so downstream
        # refs to %denom#0 / %denom#1 KeyError'd. MLIRFrontendParser exposes
        # per-result names and resolves them, so this now passes numerically.
        "extra_checks": lambda t: (
            # Pass 1 carries row_max + denom as iter_args through the
            # N-loop; pass 2 is a plain N-loop. The fused pattern emits
            # arith.mulf alongside the exp / sum that also appear in the
            # other variants.
            t.assert_present("math.exp", "arith.mulf", "arith.addf"),
        ),
    },

    # -----------------------------------------------------------------------
    # Level D -- device
    #
    # The shape a Spyre binary is built from: one tile that is the whole
    # tensor, one core, no tl.program_id, no loop, stick-tiled on both
    # descriptors. Same three ingredients as elementwise's ``1d_device``,
    # reduce's ``one_tile`` and spyreop's ``1d_device``, all of which reach a
    # binary and launch.
    #
    # DISABLED, because this one does not. It is the whole softmax rather than
    # one operation, and the row statistic is the part that does not lower:
    # subtracting a per-row maximum from the row it came from needs that one
    # value against the whole row, and that is what fails. The variant is kept
    # rather than left unwritten so the kernel exists, the intent is on record,
    # and the day it compiles this block is what gets deleted.
    #
    # No ``compiles_to_binary`` while it is disabled: that field is what
    # test_device_launch.py and the spyrecode fixtures parametrize over, and a
    # variant that skips param expansion has no ``param_values`` for them to
    # read. It goes in at the same time the ``disabled`` block comes out.
    #
    # ``params`` are spelled as separate single-name keys, not a group: a
    # disabled variant keeps its ``params`` raw and the group form is refused
    # for it at collection time.
    #
    # M = 64, N = 64 at fp16: one whole stick per row and 64 rows, so no ragged
    # extent and no sub-stick dimension. Nothing here is about padding.
    # -----------------------------------------------------------------------
    "one_tile_device": {
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "broadcast", "simplified:no-loop", "spyre-tensor-layout"],
        "summary": (
            "Row-wise softmax over a single stick-tiled tile with no "
            "distribution loop — the loop-free device shape. Does not compile "
            "today."
        ),
        "doc": (
            "Takes one `[M, N]` input and writes the row-wise softmax of it: "
            "for each row, subtract that row's maximum, exponentiate, and "
            "divide by the row's total. One tile, one core, no loop.\n\n"
            "The same arithmetic as the `default` variant with the "
            "distribution removed, so what it exercises is the softmax "
            "statistics themselves rather than how rows are handed out."
        ),
        "kernel_fn":  kernel.softmax_one_tile_device,
        "SIGNATURE":  _SIG_DEVICE,
        "constexpr":  ["M", "N", "IN_LAYOUT", "OUT_LAYOUT"],
        "params": {
            "M": [64],
            "N": [64],
            "IN_LAYOUT":  [_stick_2d_on_n("fp16")],
            "OUT_LAYOUT": [_stick_2d_on_n("fp16")],
        },
        "grid":       [1],
        # No tl.program_id, so DistributeWork has nothing to place and the
        # presence check would fail on a kernel that is correct.
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "output_ptr",
        "rtol":       1e-2,
        "atol":       5e-2,
        "disabled": {
            "reason":        "the row maximum cannot be applied back to the "
                             "row it was taken from once the tile is "
                             "stick-tiled. No tracking_test: the per-variant "
                             "form points at a single pass, and the evidence "
                             "here is a whole-pipeline compile. A runner over "
                             "every disabled variant -- the counterpart of the "
                             "global device test -- is what will pin it, to be "
                             "handled later.",
        },
    },
}
