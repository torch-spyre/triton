"""SIGNATURE + VARIANTS + reference oracle + input generator for reduce.

Row-sum reduce: out[m] = sum(in[m, :]) over the N axis, producing a
vector of M elements.

Variants:
- ``default``     -- 2D input, static M/N, no layout annotation.
- ``spyre_stick`` -- input annotated stick-on-M; exercises the
                     RewriteDescriptorLayout source reduce path.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import functools

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

_DTYPE_MAP = {"fp32": np.float32, "fp16": np.float16}

def _np_dtype(sig, key):
    return _DTYPE_MAP[sig[key].lstrip("*")]

def _sticksize(sig, key):
    return 128 // np.dtype(_np_dtype(sig, key)).itemsize


def make_inputs(M: int, N: int, **_unused) -> dict:
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((M, N)).astype(np.float32)
    out = np.zeros(M, dtype=np.float32)
    return {"in_ptr": x, "out_ptr": out}


def run(inputs: dict) -> np.ndarray:
    return inputs["in_ptr"].sum(axis=1)


def make_inputs_fp16(M: int, N: int, **_unused) -> dict:
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((M, N)).astype(np.float16)
    out = np.zeros(M, dtype=np.float16)
    return {"in_ptr": x, "out_ptr": out}


def run_fp16(inputs: dict) -> np.ndarray:
    return inputs["in_ptr"].astype(np.float32).sum(axis=1).astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr":   "*fp32",
    "out_ptr":  "*fp32",
    "M":        "i32",
    "N":        "i32",
    "BLOCK_M":  "i32",
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
_SS = functools.partial(_sticksize, _SIG_SPYRE)


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
        "params": {
            "M": [512], "N": [64], "BLOCK_M": [16],
            "IN_LAYOUT": [0], "OUT_LAYOUT": [0],
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
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "program-id-1d", "spyre-tensor-layout"],
        "summary": (
            "Row-sum reduce with in_ptr stick-on-N and out_ptr 1D stick. "
            "Exercises the RewriteDescriptorLayout source reduce path."
        ),
        "kernel_fn": kernel.reduce_spyre,
        "SIGNATURE": _SIG_SPYRE,
        "constexpr":  ["M", "N", "BLOCK_M", "IN_LAYOUT", "OUT_LAYOUT"],
        "params": {
            "M": [64], "N": [256], "BLOCK_M": [64],
            # in_ptr [M, N] stick-on-N: [N//_S, M, N%_S]
            "IN_LAYOUT":  [[(1, "floordiv", _SS("in_ptr")), 0, (1, "mod", _SS("in_ptr"))]],
            # out_ptr [M] stick: [M//_S, M%_S]
            "OUT_LAYOUT": [[(0, "floordiv", _SS("out_ptr")), (0, "mod", _SS("out_ptr"))]],
        },
        "grid":       [1],
        "reference":  run_fp16,
        "inputs":     make_inputs_fp16,
        "output_key": "out_ptr",
        "rtol":       1e-2,
        "atol":       5e-2,
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            t.assert_present("linalg.reduce"),
        ),
    },
}
