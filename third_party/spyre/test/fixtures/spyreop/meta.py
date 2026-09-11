"""SIGNATURE + VARIANTS + reference oracle + input generator for spyreop.

Named for the dialect/pass under test rather than one math function --
a deliberate departure from ``fixtures/README.md``'s usual one-folder-
per-function convention, so that ``LowerSpyreOps.cpp``'s unary float ops
sweep together as one ``OP`` axis instead of living in sibling folders.

Variants cover the 1D static shape on ``ktir_cpu`` (Level A/B) and a
single-tile, no-loop shape that reaches a real Spyre binary (Level D), each
sweeping ``OP`` across every unary scalar float op ``LowerSpyreOps.cpp``
converts *unconditionally* on type -- ``sqrt``, ``rsqrt``, ``exp``
(``math.sqrt``/``math.rsqrt``/``math.exp`` -> ``spyreop.sqrt``/
``spyreop.rsqrt``/``spyreop.exp``; see the pass's own module comment).

Two of that pass's other patterns are deliberately not swept here:

- ``arith.divf -> spyreop.realdiv`` is binary, not unary -- it needs a second
  operand and a different kernel shape (``elementwise``'s ``"div"`` arm
  already scalarizes a tensor ``/`` into ``arith.divf`` the same way this
  fixture's kernels scalarize into ``math.sqrt``/``math.rsqrt``/``math.exp``,
  so that path already has coverage under a different fixture name).
- ``arith.addi``/``arith.muli`` (i32/i64) are gated on being inside a
  ``linalg.generic`` body, not matched unconditionally on type the way the
  three float ops above are -- structurally a different kind of check (a
  containment predicate, not a type predicate), and, like ``divf``, already
  reachable through ``elementwise``'s int32 add/mul variants.

Level A/B run *before* ``_make_spyrecode``'s extra stage passes: the tensor
``tl.sqrt``/``tl.rsqrt``/``tl.exp`` scalarizes to a scalar ``math.<op>``
inside a ``linalg.generic`` body (via the always-on
``convert_elementwise_to_linalg`` fix), but nothing in the default KTIR
pipeline rewrites that to ``spyreop.<op>`` -- ``lower_spyre_ops`` only runs
as part of ``_SPYRECODE_STAGE_PASSES``, which is applied only to compiles
that go on to build a Spyre binary via ``dbo-opt`` (see
``backend/compiler.py``). So ``math.<op>`` is what the structural checks
below see, and it's also what ``ktir_cpu`` executes for the numerical check
-- ``ktir_cpu.ops.math_ops.MathOps`` implements ``sqrt``/``exp`` as plain
``np.sqrt``/``np.exp`` and ``rsqrt`` as ``1.0 / np.sqrt(x)``, confirmed
empirically (and matched exactly by this file's oracle, below).

fp32 only, throughout: upstream ``tl.sqrt``/``tl.rsqrt``/``tl.exp``
(``python/triton/language/math.py``) are each ``@_check_dtype(dtypes=
["fp32", "fp64"])`` and reject fp16 at compile time -- confirmed
empirically for all three. ``LowerSpyreOps.cpp`` matching a scalar f16
``math.<op>`` is therefore reachable only from a kernel that casts up to
f32 and back around the call (as ``softmax``'s kernels do for ``exp``),
which this fixture does not attempt.

The Level D variant is a real Spyre binary and does go through
``lower_spyre_ops``: that's the whole point of ``check_sqrt_e2e.py``
(sqrt only), which this variant's kernel (``spyreop_1d_device``)
generalizes to all three ops. The KTIR a structural test sees is still
pre-spyrecode, the same as ``elementwise``'s ``1d_device`` and ``reduce``'s
``one_tile`` -- so it is not asserted here either; the real, post-lowering
``spyreop.<op>`` is what ``check_sqrt_e2e.py`` and the ``ktir-spyreop-sqrt.mlir``
lit test cover for ``sqrt``.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np
from dataclasses import dataclass

import conftest
from . import kernel
from utils import sticksize, DTYPE_MAP


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
#
# One generator: the named maker below only gives it a shape, and the
# framework calls ``inputs`` with the variant's whole ``params`` dict.
# ---------------------------------------------------------------------------

# NumPy oracles indexed by op name. rsqrt is spelled the same way
# ktir_cpu.ops.math_ops.MathOps implements it (1.0 / np.sqrt(x)), not
# np.reciprocal(np.sqrt(x)) or scipy's rsqrt -- so the two match exactly
# rather than merely within tolerance.
_NUMPY_OPS = {
    "sqrt":  np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "exp":   np.exp,
}

# math/spyreop dialect op names, indexed the same way, for the structural
# checks below.
_MATH_OP = {"sqrt": "math.sqrt", "rsqrt": "math.rsqrt", "exp": "math.exp"}
_SPYREOP_OP = {"sqrt": "spyreop.sqrt", "rsqrt": "spyreop.rsqrt", "exp": "spyreop.exp"}


def _make_inputs(shape, *, dtype="fp32") -> dict:
    """``x`` / zeroed ``output`` buffers of *shape*. Never random, never zero.

    ``x`` is a ramp over [0.1, 1.1) -- comfortably away from 0, where sqrt's
    and rsqrt's relative error grows fastest (and where their derivatives
    are unbounded), the same offset ``check_sqrt_e2e.py`` uses. ``exp``
    stays well-conditioned over the same range, so one input maker covers
    every op this fixture sweeps.
    """
    np_dtype = DTYPE_MAP[dtype]
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    total = int(np.prod(shape))
    x = (np.arange(total, dtype=np.float32) / total + 0.1).astype(np_dtype)
    return {"x_ptr": x.reshape(shape),
            "output_ptr": np.zeros(shape, dtype=np_dtype)}


def make_inputs(n_elements, DTYPE="fp32", **_unused) -> dict:
    return _make_inputs(n_elements, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Factory -- SpyreOp(VariantFactory)
#
# OP determines the oracle (sqrt/rsqrt/exp all share one kernel, one
# SIGNATURE and one input maker, so only ``reference`` needs to vary per
# combination).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpyreOp(conftest.VariantFactory):
    """Factory for spyreop variants that sweep ``OP``."""

    def reference(self, OP, **_):
        op = _NUMPY_OPS[OP]

        def oracle(inputs):
            return op(inputs["x_ptr"])
        return oracle


def _make_op_checks(OP, **_):
    """math.<op> present, spyreop.<op> absent -- lower_spyre_ops does not run
    outside _SPYRECODE_STAGE_PASSES, see the module docstring."""
    math_op, spyreop_op = _MATH_OP[OP], _SPYREOP_OP[OP]

    def checks(t):
        t.assert_present(math_op)
        t.assert_absent(spyreop_op)
    return checks


# ---------------------------------------------------------------------------
# SIGNATURE -- dtype per @triton.jit arg. Purely types; values live in the
# variant's ``params`` dict and ``constexpr`` list selects which of them
# get baked into TTIR.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "x_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
}

# The 1d_device variant adds LAYOUT; the shape args otherwise match SIGNATURE.
_SIG_DEVICE = {**SIGNATURE, "LAYOUT": "constexpr"}


def _stick_of(dtype: str) -> int:
    return sticksize({"p": f"*{dtype}"}, "p")


def _stick_1d(dtype: str) -> tuple:
    """Labelled 1D stick layout at *dtype*, ``[n]`` -> ``[ceil(n/S), S]``."""
    stick = _stick_of(dtype)
    return ("stick", ((0, "floordiv", stick), (0, "mod", stick)))


# ---------------------------------------------------------------------------
# VARIANTS
#
# Two per-variant knobs:
#   - ``constexpr`` : list of arg names to bake in as Triton constexprs.
#                     Each variant declares the full list explicitly; no
#                     subset overrides of the default's list.
#   - ``params``    : dict of arg name -> list of values, crossed to one
#                     registry entry per combination. A key may instead be a
#                     tuple of names whose value is a list of rows, sweeping
#                     those names jointly -- see ``fixtures/README.md``.
# ---------------------------------------------------------------------------

VARIANTS = {
    # -----------------------------------------------------------------------
    # Level A/B -- shape + op x ktir_cpu correctness
    #
    # n_elements is a constexpr, baked into TTIR as a literal, the same as
    # elementwise's "default". OP sweeps sqrt/rsqrt/exp -- the factory
    # supplies the per-op oracle; inputs and SIGNATURE are shared.
    # -----------------------------------------------------------------------
    "default": {
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static",
                  "program-id-1d", "num-programs-fold"],
        "summary": (
            "1D unary spyreop `out = OP(x)` over a fully-static vector, "
            "partitioned across the 32-core grid. Sweeps OP across "
            "sqrt/rsqrt/exp."
        ),
        "doc": (
            "Takes one 1D input vector `x` of length `n_elements` and "
            "writes `out = OP(x)` to an output vector of the same "
            "length, for `OP` in `{sqrt, rsqrt, exp}`. Each of the 32 "
            "cores runs one program that carves out its share of the "
            "vector (a contiguous run of `BLOCK_SIZE`-wide tiles) and "
            "streams through it in a single pass.\n\n"
            "`n_elements` is baked in at compile time, so the tensor "
            "descriptors carry a fully-static shape (`memref<4096xf32>`)."
        ),
        "kernel_fn":    kernel.spyreop_1d,
        "factory":      SpyreOp(),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "OP"],
        "params": {
            "n_elements": [4096], "BLOCK_SIZE": [1024], "DTYPE": ["fp32"],
            "OP": ["sqrt", "rsqrt", "exp"],
        },
        "grid":         [32],
        "inputs":       make_inputs,
        "output_key":   "output_ptr",
        "rtol":         1e-6,
        "extra_checks": _make_op_checks,
    },

    # -----------------------------------------------------------------------
    # Level D -- device
    #
    # compiles_to_binary, so test_device_launch.py and the spyrecode lit
    # tests pick this up. Loop-free and one tile total -- dbo-opt rejects the
    # scf.for the looped variant above outlines from its program-id
    # distribution. Sweeps the same OP set as "default".
    # -----------------------------------------------------------------------
    "1d_device": {
        # sqrt's arm mirrors check_sqrt_e2e.py's hand-written
        # sqrt_kernel_1d_device exactly -- that script is what first drove
        # this shape through LowerSpyreOps.cpp end to end, on real hardware,
        # before this fixture folder existed. rsqrt/exp share its shape.
        "base": None,   # prevent inheriting reference/inputs from default
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D unary spyreop over a single tile, no distribution loop, "
            "fp32, stick-tiled. Sweeps OP across sqrt/rsqrt/exp."
        ),
        "kernel_fn":    kernel.spyreop_1d_device,
        "factory":      SpyreOp(),
        "SIGNATURE":    _SIG_DEVICE,
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT", "OP"],
        "params": {
            "n_elements": [128], "BLOCK_SIZE": [128], "DTYPE": ["fp32"],
            "LAYOUT": [_stick_1d("fp32")],
            "OP": ["sqrt", "rsqrt", "exp"],
        },
        "grid":         [1],
        # No tl.program_id distribution loop, so DistributeWork has nothing
        # to place and the presence check would fail on a kernel that is
        # correct.
        "parallel":     False,
        "compiles_to_binary": True,
        "inputs":       make_inputs,
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
        "extra_checks": None,
    },
}
