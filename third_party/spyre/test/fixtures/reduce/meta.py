"""SIGNATURE + VARIANTS + reference oracle + input generators for reduce.

Two shapes of reduce:

2D, trailing axis -- out[m] = sum(in[m, :]) over N, giving an M-vector.
- ``default``     -- static M/N, no layout annotation.
- ``spyre_stick`` -- input annotated stick-on-N; exercises the
                     RewriteDescriptorLayout source reduce path.

Rank-3, non-trailing middle axis -- out[d0, d2] = sum(in[d0, :, d2]).
- ``middle_axis``             -- no layout annotation.
- ``middle_axis_spyre_stick`` -- input annotated stick-on-D2, so the reduced
                                 axis is non-trailing in the physical tile and
                                 the pass must transpose it to the end.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import functools

import numpy as np

from . import kernel
from utils import sticksize


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(M: int, N: int, *, dtype=np.float32, **_unused) -> dict:
    """Build ``[M, N]`` input and ``[M]`` output buffers for reduce_spyre."""
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((M, N)).astype(dtype)
    out = np.zeros(M, dtype=dtype)
    return {"in_ptr": x, "out_ptr": out}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: row-sum reduce."""
    return inputs["in_ptr"].sum(axis=1)


def make_inputs_3d(D0: int, D1: int, D2: int, *, dtype=np.float32, **_unused) -> dict:
    """Build ``[D0, D1, D2]`` input and ``[D0, D2]`` output buffers."""
    rng = np.random.default_rng(seed=7)
    x = rng.standard_normal((D0, D1, D2)).astype(dtype)
    out = np.zeros((D0, D2), dtype=dtype)
    return {"in_ptr": x, "out_ptr": out}


def run_3d_middle(inputs: dict) -> np.ndarray:
    """NumPy oracle: reduce the middle (non-trailing) axis of a rank-3 input."""
    return inputs["in_ptr"].sum(axis=1)


# ---------------------------------------------------------------------------
# SIGNATURE — module-level default (matches reduce_spyre's arg list).
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "M":          "i32",
    "N":          "i32",
    "BLOCK_M":    "i32",
    "IN_LAYOUT":  "constexpr",
    "OUT_LAYOUT": "constexpr",
}

_SIG_SPYRE = {
    "in_ptr":     "*fp16",
    "out_ptr":    "*fp16",
    "M":          "i32",
    "N":          "i32",
    "BLOCK_M":    "i32",
    "IN_LAYOUT":  "constexpr",
    "OUT_LAYOUT": "constexpr",
}
_SS = functools.partial(sticksize, _SIG_SPYRE)

# Rank-3 middle-axis reduce: [D0, D1, D2] -> [D0, D2].
_SIG_3D = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "D0":         "i32",
    "D1":         "i32",
    "D2":         "i32",
    "IN_LAYOUT":  "constexpr",
    "OUT_LAYOUT": "constexpr",
}
_S3 = functools.partial(sticksize, _SIG_3D)


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "program-id-1d", "num-programs-fold"],
        "summary": "Row-sum reduce: out[m] = sum(in[m, :]), static M/N, no layout.",
        "kernel_fn": kernel.reduce_spyre,
        "SIGNATURE": SIGNATURE,
        "constexpr":  ["M", "N", "BLOCK_M", "IN_LAYOUT", "OUT_LAYOUT"],
        # BLOCK_M is fixed while M and N sweep: BLOCK_M interacts with the
        # grid partition (rows_per_core = cdiv(cdiv(M, BLOCK_M), grid)), so
        # sweeping both at once would conflate tiling and distribution.
        # M = 512 divides 16 evenly; 768 gives a non-multiple block count
        # (48 blocks over 32 cores -> ragged rows_per_core).
        "params": {
            "M": [512, 768], "N": [64, 256], "BLOCK_M": [16],
            "IN_LAYOUT": [None], "OUT_LAYOUT": [None],
        },
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr",
        "rtol":       1e-4,
        "extra_checks": lambda t: (
            t.assert_present("linalg.reduce"),
            t.assert_absent("tt.reduce"),
        ),
    },
    "spyre_stick": {
        # in_ptr  [M, N] stick-on-N: phys [N//S, M, N%S]
        # out_ptr [M]    stick:       phys [M//S, M%S]
        "base": "default",
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "program-id-1d", "spyre-tensor-layout"],
        "summary": (
            "Row-sum reduce with in_ptr stick-on-N and out_ptr 1D stick. "
            "Exercises the RewriteDescriptorLayout source reduce path."
        ),
        "SIGNATURE": _SIG_SPYRE,
        "params": {
            "M": [64], "N": [256], "BLOCK_M": [64],
            # in_ptr [M, N] stick-on-N: [N//S, M, N%S]
            "IN_LAYOUT":  [[(1, "floordiv", _SS("in_ptr")), 0, (1, "mod", _SS("in_ptr"))]],
            # out_ptr [M] stick: [M//S, M%S]
            "OUT_LAYOUT": [[(0, "floordiv", _SS("out_ptr")), (0, "mod", _SS("out_ptr"))]],
        },
        "grid":       [1],
        "data_layout": "host",
        "inputs":     functools.partial(make_inputs, dtype=np.float16),
        "rtol":       1e-2,
        "atol":       5e-2,
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            t.assert_present("linalg.reduce"),
        ),
    },
    # Rank-3 reduce over the NON-TRAILING middle axis: [D0, D1, D2] -> [D0, D2].
    # Numerical counterpart of the @reduce_middle_axis lit case. The three
    # extents are distinct, so reducing the wrong axis changes the output shape
    # and fails loudly.
    "middle_axis": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce"],
        "summary": (
            "Rank-3 reduce over the non-trailing middle axis "
            "(out[d0,d2] = sum(in[d0,:,d2])), no layout annotation."
        ),
        "kernel_fn":  kernel.reduce_middle_axis_spyre,
        "SIGNATURE":  _SIG_3D,
        "constexpr":  ["D0", "D1", "D2", "BLOCK_D0", "IN_LAYOUT", "OUT_LAYOUT"],
        "params": {
            # Distinct extents so a wrong-axis reduce cannot accidentally match.
            # BLOCK_D0 == D0 keeps grid=[1] a single full-tensor block, so the
            # reduce stays the only interesting structure at the default grid;
            # the middle_axis_grid variant below splits it across cores.
            "D0": [16], "D1": [96], "D2": [64], "BLOCK_D0": [16],
            "IN_LAYOUT": [None], "OUT_LAYOUT": [None],
        },
        "grid":       [1],
        "reference":  run_3d_middle,
        "inputs":     make_inputs_3d,
        "output_key": "out_ptr",
        "rtol":       1e-4,
        # linalg.reduce accumulates the 96 terms in a different order than
        # NumPy's sum, so fp32 drifts ~1e-5 absolute on a few elements.
        "atol":       1e-4,
        "extra_checks": lambda t: (
            t.assert_present("linalg.reduce"),
            t.assert_absent("tt.reduce"),
        ),
    },
    "middle_axis_spyre_stick": {
        # in_ptr stick-on-D2 (fp32 stick = 32, D2 = 64 = 2 sticks exactly):
        #   phys [D2//32, D0, D1, D2%32] = [2, 16, 96, 32]
        # The reduced axis (D1) is non-trailing in the physical tile, so the
        # pass must emit a linalg.transpose rotating it to the end before
        # linalg.reduce. A slot-index-derived permutation would be identity
        # here and would reduce the wrong axis.
        "base": "middle_axis",
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "spyre-tensor-layout"],
        "summary": (
            "Rank-3 middle-axis reduce with in_ptr stick-on-D2. The reduced "
            "axis is non-trailing, so the pass must transpose it to the end "
            "before linalg.reduce."
        ),
        "params": {
            "D0": [16], "D1": [96], "D2": [64], "BLOCK_D0": [16],
            "IN_LAYOUT": [[(2, "floordiv", _S3("in_ptr")), 0, 1,
                           (2, "mod", _S3("in_ptr"))]],
            "OUT_LAYOUT": [None],
        },
        "data_layout": "host",
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            t.assert_present("linalg.reduce"),
            # The reduced (D1) axis must be rotated to the trailing position.
            t.assert_present("linalg.transpose"),
        ),
    },
    # ---- Grid variation -----------------------------------------------------
    # `grid` is a top-level entry field, read once per variant, so it is not
    # sweepable through `params` — varying it means sibling variants. BLOCK_* is
    # held fixed and the extents vary, per the tiling/distribution separation
    # noted on `default`.
    "grid_8": {
        "base": "default",
        "summary": (
            "Row-sum reduce on 8 cores instead of 32 — same work, more rows "
            "per core, exercising a different DistributeWork split."
        ),
        "params": {
            "M": [512, 768], "N": [64], "BLOCK_M": [16],
            "IN_LAYOUT": [None], "OUT_LAYOUT": [None],
        },
        "grid": [8],
    },
    "middle_axis_grid": {
        # D0 = 16 over BLOCK_D0 = 4 gives 4 blocks, one per core at grid=[4],
        # so every core takes exactly one iteration of the outer loop. The
        # reduced axis (D1) is untouched by the split.
        "base": "middle_axis",
        "summary": (
            "Rank-3 middle-axis reduce distributed over a 4-core grid, "
            "exercising the program-id + outer-loop path."
        ),
        "params": {
            "D0": [16], "D1": [96], "D2": [64], "BLOCK_D0": [4],
            "IN_LAYOUT": [None], "OUT_LAYOUT": [None],
        },
        "grid": [4],
    },
}
