"""SIGNATURE + VARIANTS + reference oracle + input generator for the
``tl.spyre_op`` intrinsic requests.

One folder for the three intrinsics rather than one per function, like
``spyreop``: what is under test is the request mechanism -- trace, inline, tag --
and the three differ only in which fallback is traced.

``ktir_cpu`` runs the ``ktir`` artifact, so these variants check the registered
FALLBACKS (``backend/intrinsics.py``) against the mathematical definitions. That
the ``spyrecode`` stage replaces each with its intrinsic is pinned at the pass
level, in ``Transforms/LowerSpyreOps/request.mlir`` and
``spyre-triton-opt/stage-pipelines-spyre-op.mlir``.

The sweep is exactly the dtypes each intrinsic takes: gelu at fp16, silu and
sigmoid at fp16 and fp32. Any other cell is refused at trace time.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import math

import numpy as np
from dataclasses import dataclass

import conftest
from . import kernel
from utils import DTYPE_MAP


def _gelu(x):
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


_ORACLES = {
    "gelu": _gelu,
    "silu": lambda x: x * _sigmoid(x),
    "sigmoid": _sigmoid,
}


def make_inputs(n_elements, DTYPE="fp32", **_unused) -> dict:
    """``x`` as a ramp over [-4, 4), the range where all three functions bend,
    and a zeroed ``output``. Never random."""
    np_dtype = DTYPE_MAP[DTYPE]
    x = (np.arange(n_elements, dtype=np.float32) * (8.0 / n_elements) - 4.0)
    return {"x_ptr": x.astype(np_dtype),
            "output_ptr": np.zeros(n_elements, dtype=np_dtype)}


#: The fp32 arm; ``IntrinsicRequest.signature`` replaces it per ``DTYPE``. Declared
#: at module scope because discovery resolves a variant only when one exists.
SIGNATURE = {"x_ptr": "*fp32", "output_ptr": "*fp32",
             "n_elements": "i32", "BLOCK_SIZE": "i32"}


@dataclass(frozen=True)
class IntrinsicRequest(conftest.VariantFactory):
    """Pointer dtypes from ``DTYPE``; the oracle from ``OP``, computed in fp32
    and rounded to the buffer's dtype, which is what the fallback does."""

    def signature(self, DTYPE, **_):
        return {"x_ptr": f"*{DTYPE}", "output_ptr": f"*{DTYPE}",
                "n_elements": "i32", "BLOCK_SIZE": "i32"}

    def reference(self, OP, **_):
        oracle = _ORACLES[OP]

        def run(inputs):
            x = inputs["x_ptr"]
            return oracle(x.astype(np.float32)).astype(x.dtype)
        return run


VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static",
                 "program-id-1d", "spyre-op"],
        "summary": (
            "1D tl.spyre_op request across the grid, swept over each "
            "intrinsic at each dtype it takes."
        ),
        "kernel_fn":    kernel.intrinsic_request_1d,
        "factory":      IntrinsicRequest(),
        "constexpr":    ["BLOCK_SIZE", "OP"],
        "params": {
            ("OP", "DTYPE"): [
                ("gelu", "fp16"),
                ("silu", "fp16"), ("silu", "fp32"),
                ("sigmoid", "fp16"), ("sigmoid", "fp32"),
            ],
            "n_elements": [4096],
            "BLOCK_SIZE": [128],
        },
        "grid":         [32],
        "inputs":       make_inputs,
        "output_key":   "output_ptr",
        "rtol":         {"fp16": 1e-2, "fp32": 1e-5},
        "atol":         {"fp16": 2e-3, "fp32": 1e-6},
    },
}
