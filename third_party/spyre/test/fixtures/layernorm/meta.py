"""SIGNATURE + VARIANTS + reference oracle + input generator for layernorm.

Three variants:

``default``  a row-wise LayerNorm over an ``[M, N]`` input with the whole tensor in
             a single tile, in ordinary ``tl`` ops. The SLOW path, and the
             baseline.
``mean``     the mean of each row of a logical ``[M, N]`` input stuck on N, chained
             through HBM: the input divided by the count into a pool, then summed.
``norm``     a whole LayerNorm per row, chained through HBM in nine compute groups,
             the normalisation requested as the device's ``layernormnorm``.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np

from . import kernel
from utils import sticksize

#: Added to the variance before the square root; a kernel constexpr, so the
#: oracle reads the same value.
EPS = 1e-5


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(M: int, N: int, **_unused) -> dict:
    """``[M, N]`` input, ``[N]`` weight and bias, and a zeroed ``[M, N]`` output.

    The weight is not all ones and the bias not all zeros, so a kernel that
    dropped either would be wrong rather than coincidentally right.
    """
    rng = np.random.default_rng(seed=0)
    return {
        "x_ptr":      rng.standard_normal((M, N)).astype(np.float32),
        "weight_ptr": (1.0 + 0.25 * rng.standard_normal(N)).astype(np.float32),
        "bias_ptr":   (0.5 * rng.standard_normal(N)).astype(np.float32),
        "out_ptr":    np.zeros((M, N), dtype=np.float32),
    }


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``(x - mean) / sqrt(var + EPS) * weight + bias`` per row."""
    x = inputs["x_ptr"]
    mean = x.mean(axis=1, keepdims=True)
    centred = x - mean
    var = (centred * centred).mean(axis=1, keepdims=True)
    return (centred / np.sqrt(var + EPS) * inputs["weight_ptr"] + inputs["bias_ptr"]).astype(np.float32)


#: The ``eps`` the ``norm`` variant adds to the variance, and its oracle with it. It
#: is 2**-17, the value the device's own LayerNorm scale nudges the variance by.
DEVICE_EPS = 2.0 ** -17


def _stick(dtype: str) -> int:
    """Lanes to a stick at *dtype*: 64 at fp16."""
    return sticksize({"p": f"*{dtype}"}, "p")


SIGNATURE = {
    "x_ptr":      "*fp32",
    "weight_ptr": "*fp32",
    "bias_ptr":   "*fp32",
    "out_ptr":    "*fp32",
    "M":          "constexpr",
    "N":          "constexpr",
    "EPS":        "constexpr",
}


def make_inputs_mean(M, N, **_unused) -> dict:
    """``[M, N]`` in, an ``[M, N]`` scratch pool, and a zeroed ``[M]`` mean, at
    fp16. The mean is rank-1 on the host: the splat layout replicates it across a
    stick on the device, and the allocation comes from the compiled kernel's
    recorded footprint."""
    rng = np.random.default_rng(seed=0)
    return {
        "x_ptr":    rng.standard_normal((M, N)).astype(np.float16),
        "n_ptr":    np.full((1, N), N, dtype=np.float16),
        "pool_ptr": np.zeros((M, N), dtype=np.float16),
        "mean_ptr": np.zeros(M, dtype=np.float16),
    }


def run_mean(inputs: dict) -> np.ndarray:
    """NumPy oracle: the mean of each row."""
    return inputs["x_ptr"].astype(np.float32).mean(axis=1).astype(np.float16)


def _row_mean(dtype: str) -> tuple:
    """The ``("DTYPE", "S", "TILE_LAYOUT", "STAT_LAYOUT")`` row: x stuck on N, the
    mean splat across a stick."""
    s = _stick(dtype)
    return (dtype, s,
            ("stick", ((1, "floordiv", s), 0, (1, "mod", s))),
            ("splat", (0, (0, "splat", s))))


def make_inputs_norm(M, N, S, **_unused) -> dict:
    """:func:`make_inputs_mean` plus every buffer the nine groups chain through, the
    ``eps`` stick, a ``[1, N]`` weight and bias, and a zeroed ``[M, N]`` output.

    The second mean and the mean of the squares are rank-1 ``[M]``, stored through
    a splat layout as the mean is; ``sq``, ``var`` and ``scale`` are produced
    stick-wide, so their host shape is their device shape. The weight is not all
    ones and the bias not all zeros, so a kernel dropping either is wrong rather
    than coincidentally right.
    """
    rng = np.random.default_rng(seed=1)
    stick = lambda: np.zeros((M, S), dtype=np.float16)
    return {**make_inputs_mean(M, N),
            "eps_ptr":    np.full((1, S), DEVICE_EPS, dtype=np.float16),
            "mean2_ptr":  np.zeros(M, dtype=np.float16),
            "msq_ptr":    np.zeros(M, dtype=np.float16),
            "sq_ptr":     stick(),
            "var_ptr":    stick(),
            "scale_ptr":  stick(),
            "weight_ptr": (1.0 + 0.25 * rng.standard_normal((1, N))).astype(np.float16),
            "bias_ptr":   (0.5 * rng.standard_normal((1, N))).astype(np.float16),
            "out_ptr":    np.zeros((M, N), dtype=np.float16)}


def run_norm(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``(x - mean) / sqrt(var + DEVICE_EPS) * weight + bias`` per row."""
    x = inputs["x_ptr"].astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    var = (x * x).mean(axis=1, keepdims=True) - mean * mean
    w = inputs["weight_ptr"].astype(np.float32)
    b = inputs["bias_ptr"].astype(np.float32)
    return ((x - mean) / np.sqrt(var + np.float32(DEVICE_EPS)) * w + b).astype(np.float16)


VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce", "broadcast"],
        "summary": (
            "Row-wise LayerNorm over an `M x N` input in a single tile, in "
            "ordinary tl ops: the slow path, with no Spyre intrinsic."
        ),
        "doc": (
            "Computes `(x - mean) / sqrt(var + eps) * weight + bias` for each "
            "row of an `M x N` input, `weight` and `bias` being `[N]` vectors. "
            "The mean and the variance of the centred value are two `tl.sum` "
            "reductions, and the rest is elementwise. The whole tensor is one "
            "tile and one core, so there is no loop.\n\n"
            "It names no Spyre intrinsic, which makes it the baseline: what "
            "this compiles to is what a LayerNorm costs without a request for "
            "the device's `layernormscale` and `layernormnorm`."
        ),
        "kernel_fn":  kernel.layernorm_single_tile,
        "constexpr":  ["M", "N", "EPS"],
        "params":     {"M": [8], "N": [64], "EPS": [EPS]},
        "grid":       [1],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr",
        "rtol":       1e-4,
        "atol":       1e-5,
    },
    "mean": {
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "spyre-tensor-layout", "hbm-round-trip"],
        "summary": (
            "The mean of each row of a logical [M, N] input stuck on N: x / N "
            "into a pool, then one sum over N stored across a stick."
        ),
        "kernel_fn":  kernel.layernorm_mean,
        "SIGNATURE":  {"x_ptr": "*fp16", "n_ptr": "*fp16", "pool_ptr": "*fp16", "mean_ptr": "*fp16",
                       "M": "constexpr", "N": "constexpr",
                       "TILE_LAYOUT": "constexpr", "STAT_LAYOUT": "constexpr"},
        "constexpr":  ["M", "N", "TILE_LAYOUT", "STAT_LAYOUT"],
        "params": {
            ("DTYPE", "S", "TILE_LAYOUT", "STAT_LAYOUT"): [_row_mean("fp16")],
            # The target's 32 x 48 x 64 read as 48 rows of 32 * 64, on one core.
            "M": [48],
            "N": [2048],
        },
        "grid":        [1],
        "compiles_to_binary": True,
        "reference":   run_mean,
        "inputs":      make_inputs_mean,
        "output_key":  "mean_ptr",
        # The means of 2048 standard normals are about +-0.04, so the bound is
        # absolute and well below them: an output off by a factor of N lands near
        # zero and fails it. fp16 holds 1/2048 to about 2.4e-7 per element, and the
        # sum of 2048 such values gathers rounding to a few times 1e-4.
        "rtol":        0.0,
        "atol":        2e-3,
    },
    "norm": {
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "broadcast", "spyre-tensor-layout", "hbm-round-trip",
                 "spyre-intrinsic"],
        "summary": (
            "A whole LayerNorm per row: the scale in plain tl with rsqrt, then the "
            "normalisation as a layernormnorm request. Nine compute groups."
        ),
        "kernel_fn":  kernel.layernorm_norm,
        "SIGNATURE":  {"x_ptr": "*fp16", "weight_ptr": "*fp16", "bias_ptr": "*fp16",
                       "out_ptr": "*fp16", "n_ptr": "*fp16", "eps_ptr": "*fp16",
                       "pool_ptr": "*fp16", "mean_ptr": "*fp16", "mean2_ptr": "*fp16",
                       "msq_ptr": "*fp16",
                       "sq_ptr": "*fp16", "var_ptr": "*fp16", "scale_ptr": "*fp16",
                       "M": "constexpr", "N": "constexpr", "S": "constexpr",
                       "TILE_LAYOUT": "constexpr", "STAT_LAYOUT": "constexpr"},
        "constexpr":  ["M", "N", "S", "TILE_LAYOUT", "STAT_LAYOUT"],
        "params": {
            ("DTYPE", "S", "TILE_LAYOUT", "STAT_LAYOUT"): [_row_mean("fp16")],
            "M": [48],
            "N": [2048],
        },
        "grid":        [1],
        "compiles_to_binary": True,
        "reference":   run_norm,
        "inputs":      make_inputs_norm,
        "output_key":  "out_ptr",
        # The output is O(1); the bound carries the scale's fp16 error through
        # the normalisation.
        "rtol":        5e-2,
        "atol":        5e-2,
    },
}
