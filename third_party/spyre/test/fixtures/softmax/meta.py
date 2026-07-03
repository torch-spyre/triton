"""SIGNATURE + VARIANTS + reference oracle + input generator for softmax.

Three variants — ``default`` (single-tile), ``multi_tile``, ``2pass``
— share the same reference oracle and input shape. They exercise
different KTIR patterns: single tile with in-tile reduce; 3-pass over
N-tiles; 2-pass online (Milakov & Gimelshein).

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np

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


# ---------------------------------------------------------------------------
# VARIANTS
#
# All three variants use the same input shape (M=1024, N=1024) and the
# same oracle. They split on tiling strategy:
#   - default = single_tile: full row in 1024-wide tile, in-tile reduce.
#   - multi_tile: 16 N-tiles of width 64, 3-pass per row.
#   - 2pass: 4-row × 64-col tiles, fused-max + online denom, 2-pass.
#
# 1D grid across all 32 cores: each kernel reads only tl.program_id(0)
# and partitions rows internally via an explicit rows_per_core loop.
# ---------------------------------------------------------------------------

VARIANTS = {
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
            "M":          [1024],
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
    "nonaligned": {
        # M=1000: rows_per_core=ceil(1000/32)=32 for most cores, but the last
        # core's range overshoots 1000 → tl.minimum clamp fires.
        "base":   "default",
        "params": {"M": [1000], "N": [1024], "BLOCK_SIZE": [1024]},
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
        "params":    {"M": [1024], "N": [1024], "BLOCK_N": [64]},
        "extra_checks": lambda t: (
            # Three nested scf.for in the kernel body: outer rows-per-core,
            # three inner N-tile passes (max, denom, normalize).
            t.assert_count("scf.for", 4, cmp="ge"),
        ),
    },
    "multi_tile_nonaligned": {
        # M=1000: rows_per_core clamp fires, same as nonaligned but on the
        # multi-tile kernel where N-tile inner loops also run.
        "base":   "multi_tile",
        "params": {"M": [1000], "N": [1024], "BLOCK_N": [64]},
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
}
