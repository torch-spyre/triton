"""SIGNATURE + VARIANTS + reference oracle + input generator for spyreop.

Named for the dialect/pass under test rather than one math function --
a deliberate departure from ``fixtures/README.md``'s usual one-folder-
per-function convention, so that ``LowerSpyreOps.cpp``'s float ops sweep
together as one fixture instead of living in sibling folders.

Level D only -- no ktir_cpu (Level A/B) variants, and that is deliberate,
not an oversight:

``ktir_cpu`` (the numerical interpreter Level A/B runs against) has no
``spyreop`` dialect handler at all -- confirmed by inspecting the installed
package (``ls`` its ``dialects/`` only lists ``arith``, ``math``, ``linalg``,
``tensor``, ``scf``, ``ktdp``; grepping the whole package for
``spyreop``/``SpyreOp`` returns nothing). And ``lower_spyre_ops`` (the pass
this fixture is named for) only runs as part of ``_SPYRECODE_STAGE_PASSES``
(see ``backend/compiler.py``), applied only to compiles that go on to build a
real Spyre binary via ``dbo-opt`` -- i.e. only to Level D compiles. Put those
two facts together: a Level A/B variant here can only ever see
``math.<op>``/``arith.divf``/``arith.addi``/``arith.muli`` -- pre-
``lower_spyre_ops`` -- no matter how it is built, and ``ktir_cpu`` could not
execute ``spyreop.<op>``/``spyreop.realdiv``/``spyreop.addi32toi32``/
``spyreop.muli32toi32`` even if that pass did run underneath it. Such a
variant would numerically verify ``convert_elementwise_to_linalg``'s
scalarization (a different pass) while asserting nothing about
``LowerSpyreOps.cpp`` -- the one thing this fixture exists to cover. Hence
Level D only: every variant below is ``compiles_to_binary`` and its
correctness comes from running the resulting binary on real hardware, the
only place any of those plain-dialect ops actually becomes their ``spyreop``
counterpart.

Three variants, split by arity/dtype (matching ``kernel.py``):

- ``default`` sweeps ``OP`` across every *unary* scalar float op
  ``LowerSpyreOps.cpp`` converts unconditionally on type -- ``sqrt``,
  ``rsqrt``, ``exp`` (``math.sqrt``/``math.rsqrt``/``math.exp`` ->
  ``spyreop.sqrt``/``spyreop.rsqrt``/``spyreop.exp``; see the pass's own
  module comment).
- ``realdiv`` covers the one *binary* float op the pass converts
  unconditionally on type -- ``arith.divf -> spyreop.realdiv``. Kept as a
  separate variant (own kernel, own inputs) rather than folded into the
  ``OP`` sweep above because the arity differs: a second operand needs a
  second tensor descriptor, a second SIGNATURE entry, and a second input
  array (not just another branch in an existing kernel body).
- ``addmul`` sweeps ``OP`` across ``add``/``mul`` on i32 -- the pattern
  ``LowerSpyreOps.cpp`` matches only *inside* a ``linalg.generic`` body
  (``arith.addi``/``arith.muli`` -> ``spyreop.addi32toi32``/
  ``spyreop.muli32toi32``), a containment predicate rather than the
  unconditional type match the float ops above get. ``ConvertElementwiseToLinalg``
  supplies that containment for free -- it scalarizes any tensor-level Triton
  op into a ``linalg.generic`` body before this pass ever runs, int tensors
  included -- so the kernel body is written the same plain way as the other
  two (no explicit ``linalg.generic``, no manual scalarization).

  Before this variant existed, nothing in the repo drove an int32 tensor
  add/mul through a real Spyre binary compile: ``elementwise``'s Level D
  variants (``1d_device``/``2d_device``) only sweep fp16/fp32, and its Level
  B variant that does sweep i32 (``1d_compute``) never reaches this pass for
  the same reason Level A/B can't here (see below) -- ``lower_spyre_ops``
  only runs at the spyrecode stage, which Level B compiles never reach.
  ``LowerSpyreOps.cpp``'s own pass-level lit test
  (``test/Conversion/lower-spyre-ops.mlir``) already covers the containment
  predicate itself precisely, positive and negative cases both (i32/i64
  inside a ``linalg.generic`` converts -- i64 only for add, no
  ``muli64toi64`` intrinsic exists; i16 and anything outside a
  ``linalg.generic`` does not); this variant's job is proving a real Triton
  kernel's tensor int32 add/mul reaches that same shape end to end, on real
  hardware. i64 is not swept here for the same reason ``muli64toi64`` isn't
  in the lit test's positive cases -- an i64 row would only exercise the
  ``add`` arm's intrinsic and leave ``mul`` an (already-covered) "survives"
  case, asymmetric with the i32 row for no real gain.

``default``/``realdiv`` are fp32 only: upstream ``tl.sqrt``/``tl.rsqrt``/
``tl.exp`` (``python/triton/language/math.py``) are each
``@_check_dtype(dtypes=["fp32", "fp64"])`` and reject fp16 at compile time --
confirmed empirically for all three. ``LowerSpyreOps.cpp`` matching a scalar
f16 ``math.<op>`` is therefore reachable only from a kernel that casts up to
f32 and back around the call (as ``softmax``'s kernels do for ``exp``),
which this fixture does not attempt. ``addmul`` is i32 -- see its own
paragraph above for why i64 is not swept alongside it.

Every variant's structural test still sees pre-spyrecode KTIR (the same
limitation ``elementwise``'s ``1d_device`` and ``reduce``'s ``one_tile``
have), so ``extra_checks`` is ``None`` throughout -- there is currently no
hook here that captures the post-``lower_spyre_ops`` KTIR to assert
``spyreop.<op>``/``spyreop.realdiv``/``spyreop.addi32toi32``/
``spyreop.muli32toi32`` actually appears. The pass's own correctness at that
level (that the rewrite fires, and fires only where it should) is what
``test/Conversion/lower-spyre-ops.mlir`` covers precisely, at the pass
level; what this fixture adds on top is that a real Triton kernel, run
through the full pipeline on real hardware, produces the right numbers.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np
from dataclasses import dataclass

import conftest
from . import kernel
from utils import sticksize, DTYPE_MAP


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input makers
# ---------------------------------------------------------------------------

# NumPy oracles indexed by op name. rsqrt is spelled the same way
# ktir_cpu.ops.math_ops.MathOps implements it (1.0 / np.sqrt(x)), not
# np.reciprocal(np.sqrt(x)) or scipy's rsqrt -- kept for the numeric match,
# even though ktir_cpu itself is no longer in this fixture's path (it never
# executes spyreop.<op> -- see the module docstring).
_NUMPY_OPS = {
    "sqrt":  np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "exp":   np.exp,
}


def _make_inputs(shape, *, dtype="fp32") -> dict:
    """``x`` / zeroed ``output`` buffers of *shape*. Never random, never zero.

    ``x`` is a ramp over [0.1, 1.1) -- comfortably away from 0, where sqrt's
    and rsqrt's relative error grows fastest (and where their derivatives
    are unbounded). ``exp`` stays well-conditioned over the same range, so
    one input maker covers every unary op this fixture sweeps.
    """
    np_dtype = DTYPE_MAP[dtype]
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    total = int(np.prod(shape))
    x = (np.arange(total, dtype=np.float32) / total + 0.1).astype(np_dtype)
    return {"x_ptr": x.reshape(shape),
            "output_ptr": np.zeros(shape, dtype=np_dtype)}


def make_inputs(n_elements, DTYPE="fp32", **_unused) -> dict:
    return _make_inputs(n_elements, dtype=DTYPE)


def _make_realdiv_inputs(shape, *, dtype="fp32") -> dict:
    """``x``/``y``/zeroed ``output`` buffers of *shape*, for ``x / y``.

    ``x`` is the same [0.1, 1.1) ramp ``_make_inputs`` uses; ``y`` is a
    second, disjoint ramp over [1.1, 2.1) -- comfortably away from 0 (no
    division by, or near, zero) and never equal to ``x`` (no accidental
    ``x / y == 1`` everywhere).
    """
    np_dtype = DTYPE_MAP[dtype]
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    total = int(np.prod(shape))
    ramp = np.arange(total, dtype=np.float32) / total
    x = (ramp + 0.1).astype(np_dtype)
    y = (ramp + 1.1).astype(np_dtype)
    return {"x_ptr": x.reshape(shape), "y_ptr": y.reshape(shape),
            "output_ptr": np.zeros(shape, dtype=np_dtype)}


def make_realdiv_inputs(n_elements, DTYPE="fp32", **_unused) -> dict:
    return _make_realdiv_inputs(n_elements, dtype=DTYPE)


def _realdiv_reference(inputs):
    return inputs["x_ptr"] / inputs["y_ptr"]


_NUMPY_INT_OPS = {"add": np.add, "mul": np.multiply}


def _make_addmul_inputs(shape, *, dtype="i32") -> dict:
    """``x``/``y``/zeroed ``output`` int buffers of *shape*, for ``x + y``/
    ``x * y``. Both ramps are folded to [1, 16] (``idx % 16 + 1``, offset by
    5 between ``x`` and ``y``) regardless of *shape* -- the largest possible
    product is 16*16, comfortably inside int32, so a passing ``mul`` case
    can't be masked by wraparound.
    """
    np_dtype = DTYPE_MAP[dtype]
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    total = int(np.prod(shape))
    idx = np.arange(total, dtype=np.int64)
    x = ((idx % 16) + 1).astype(np_dtype)
    y = (((idx + 5) % 16) + 1).astype(np_dtype)
    return {"x_ptr": x.reshape(shape), "y_ptr": y.reshape(shape),
            "output_ptr": np.zeros(shape, dtype=np_dtype)}


def make_addmul_inputs(n_elements, DTYPE="i32", **_unused) -> dict:
    return _make_addmul_inputs(n_elements, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Factories -- SpyreOp/SpyreIntOp(VariantFactory)
#
# OP determines the oracle in both cases (each variant shares one kernel,
# one SIGNATURE and one input maker across its OP sweep, so only
# ``reference`` needs to vary per combination). realdiv has no OP sweep, so
# it uses a plain "reference" function instead -- see the "realdiv" variant
# below.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpyreOp(conftest.VariantFactory):
    """Factory for the unary float spyreop variant ("default")."""

    def reference(self, OP, **_):
        op = _NUMPY_OPS[OP]

        def oracle(inputs):
            return op(inputs["x_ptr"])
        return oracle


@dataclass(frozen=True)
class SpyreIntOp(conftest.VariantFactory):
    """Factory for the binary int spyreop variant ("addmul")."""

    def reference(self, OP, **_):
        op = _NUMPY_INT_OPS[OP]

        def oracle(inputs):
            return op(inputs["x_ptr"], inputs["y_ptr"])
        return oracle


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
    "LAYOUT":     "constexpr",
}

# realdiv is binary: a second input pointer, same dtype as x_ptr.
_SIG_REALDIV = {
    "x_ptr":      "*fp32",
    "y_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
    "LAYOUT":     "constexpr",
}

# addmul is binary and i32, not fp32.
_SIG_ADDMUL = {
    "x_ptr":      "*i32",
    "y_ptr":      "*i32",
    "output_ptr": "*i32",
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
    "LAYOUT":     "constexpr",
}


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
    # Unary spyreop, device. compiles_to_binary, so test_device_launch.py
    # and the spyrecode lit tests pick this up. Loop-free and one tile
    # total -- dbo-opt rejects the scf.for a looped kernel would outline
    # from its program-id distribution (see the module docstring for why
    # there is no looped/Level-A/B counterpart at all). Sweeps OP across
    # sqrt/rsqrt/exp.
    # -----------------------------------------------------------------------
    "default": {
        "base": None,
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

    # -----------------------------------------------------------------------
    # Binary spyreop (realdiv), device. Same role as "default", for the one
    # binary float op instead of the unary OP sweep -- own kernel, own
    # SIGNATURE (adds y_ptr), own inputs/reference (no OP, so no factory
    # needed).
    # -----------------------------------------------------------------------
    "realdiv": {
        "base": None,
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D binary spyreop `out = x / y` over a single tile, no "
            "distribution loop, fp32, stick-tiled."
        ),
        "kernel_fn":    kernel.spyreop_realdiv_1d_device,
        "SIGNATURE":    _SIG_REALDIV,
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT"],
        "params": {
            "n_elements": [128], "BLOCK_SIZE": [128], "DTYPE": ["fp32"],
            "LAYOUT": [_stick_1d("fp32")],
        },
        "grid":         [1],
        "parallel":     False,
        "compiles_to_binary": True,
        "inputs":       make_realdiv_inputs,
        "reference":    _realdiv_reference,
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
        "extra_checks": None,
    },

    # -----------------------------------------------------------------------
    # Binary spyreop (addi/muli), i32. Same role as "realdiv" for the int
    # add/mul patterns -- own kernel, own SIGNATURE/dtype, own inputs. Sweeps
    # OP across add/mul, so it uses a factory like "default" rather than a
    # fixed "reference" like "realdiv".
    #
    # KTIR-structural only for now, unlike its "default"/"realdiv" siblings:
    # no `compiles_to_binary`, so `test_device_launch.py` does not pick this
    # one up (see `compilable_example` in conftest.py, which parametrizes
    # strictly over `compiles_to_binary` entries). Real hardware genuinely
    # cannot compile it today -- dbo-opt's own verifier rejects the output
    # descriptor's int32 store: `ktdf.data_transfer` requires its source and
    # destination to share an element type, but the store side produces a
    # *signed* `si32` memref for an int32 output tensor against a *signless*
    # `i32` FIFO slot type. `LowerDescriptorMemory.cpp` already documents and
    # reconciles this exact signedness split on the *load* path
    # (`resolveIndexView`'s "two sides describe the same storage but
    # disagree on signedness" comment) -- that reconciliation evidently
    # doesn't extend to stores. Confirmed unrelated to `LowerSpyreOps.cpp`:
    # the failure is inside dbo-opt itself, well after this pass runs.
    # Revisit `compiles_to_binary` here once that store-side gap is fixed.
    # -----------------------------------------------------------------------
    "addmul": {
        "base": None,
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D binary spyreop over a single tile, no distribution loop, "
            "i32, stick-tiled. Sweeps OP across add/mul. KTIR-structural "
            "only -- real hardware can't compile an int32 descriptor store "
            "yet (see the comment above)."
        ),
        "kernel_fn":    kernel.spyreop_addmul_1d_device,
        "factory":      SpyreIntOp(),
        "SIGNATURE":    _SIG_ADDMUL,
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT", "OP"],
        "params": {
            "n_elements": [128], "BLOCK_SIZE": [128], "DTYPE": ["i32"],
            "LAYOUT": [_stick_1d("i32")],
            "OP": ["add", "mul"],
        },
        "grid":         [1],
        "parallel":     False,
        "inputs":       make_addmul_inputs,
        "output_key":   "output_ptr",
        "rtol":         0,
        "atol":         0,
        "extra_checks": None,
    },
}
