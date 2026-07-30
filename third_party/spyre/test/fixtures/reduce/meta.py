"""SIGNATURE + VARIANTS + reference oracle + input generators for reduce.

Row-sum reduce: out[m] = sum(in[m, :]) over the N axis, producing a
vector of M elements.

Variants:
- ``default``     -- 2D input, static M/N, no layout annotation.
- ``spyre_stick`` -- input annotated stick-on-N; exercises the
                     RewriteDescriptorLayout source reduce path.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import functools

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def _sticksize(sig, key):
    """Compute the stick size (elements per 128-byte line) for a pointer arg."""
    _DTYPE_MAP = {"fp32": np.float32, "fp16": np.float16}
    np_dtype = _DTYPE_MAP[sig[key].lstrip("*")]
    return 128 // np.dtype(np_dtype).itemsize


def make_inputs(M: int, N: int, *, dtype=np.float32, **_unused) -> dict:
    """Build ``[M, N]`` input and ``[M]`` output buffers for reduce_spyre."""
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((M, N)).astype(dtype)
    out = np.zeros(M, dtype=dtype)
    return {"in_ptr": x, "out_ptr": out}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: row-sum reduce."""
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
}
