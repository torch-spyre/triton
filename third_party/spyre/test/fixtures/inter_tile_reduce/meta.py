"""SIGNATURE + VARIANTS for the inter_tile_reduce fixture.

Multi-group ADD all_reduce: a 2D kernel where every tile along the x-axis
contributes its partial sum to a cross-tile all_reduce.  Every tile in a
reduction group receives the fully-reduced value.

The only variant is currently ``disabled`` because ``tl.inter_tile`` (T006)
is not yet in the Triton frontend.  The tracking test is the structural
assertion for the lowered produce/reduce pair.

See ``fixtures/README.md`` for the field reference.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference oracle + input builder
# ---------------------------------------------------------------------------

def make_inputs(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> dict:
    """Build pointer-tensor inputs for inter_tile_add_kernel."""
    del BLOCK_M, BLOCK_N
    rng = np.random.default_rng(42)
    x = rng.standard_normal((M, N)).astype(np.float32)
    output = np.zeros((M, N), dtype=np.float32)
    return {"x_ptr": x, "output_ptr": output}


def run(inputs: dict) -> np.ndarray:
    """Oracle: each row broadcasts the row-sum to every element in that row.

    all_reduce along x (the N-axis): each element of the output equals the
    sum of its entire row, replicated across the row.
    """
    x = inputs["x_ptr"]
    return np.broadcast_to(x.sum(axis=1, keepdims=True), x.shape).copy()


# ---------------------------------------------------------------------------
# SIGNATURE
# ---------------------------------------------------------------------------

SIGNATURE = {
    "x_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "M":          "i32",
    "N":          "i32",
    "BLOCK_M":    "i32",
    "BLOCK_N":    "i32",
}

# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["inter-tile-all-reduce"],
        "summary": (
            "Multi-group ADD all_reduce: every tile along the x-axis "
            "contributes its partial and receives the fully-reduced sum."
        ),
        "doc": (
            "A 2D kernel that loads a ``BLOCK_M × BLOCK_N`` tile from a "
            "``M × N`` matrix, then performs a cross-tile all_reduce along "
            "the x-axis.  Each reduction group consists of all tiles that "
            "share the same y-position; every tile in a group receives the "
            "sum of all partials.  The oracle verifies that the output "
            "equals ``x.sum(axis=1, keepdims=True)`` broadcast across each "
            "row.\n\n"
            "Disabled until ``tl.inter_tile`` lands in the frontend (T006)."
        ),
        "kernel_fn":    kernel.inter_tile_add_kernel,
        "constexpr":    ["M", "N", "BLOCK_M", "BLOCK_N"],
        "params":       {"M": [64], "N": [32], "BLOCK_M": [16], "BLOCK_N": [16]},
        # 2D grid: 4 rows × 2 columns = 8 cores; two reduction groups of 2
        "grid":         [4, 2],
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "output_ptr",
        "disabled": {
            "reason": (
                "tl.inter_tile frontend (T006) not yet implemented; "
                "the kernel body references a non-existent tl.inter_tile call"
            ),
            "tracking_test": "test_lower_inter_tile.py::TestAllReduce",
        },
    },
}
