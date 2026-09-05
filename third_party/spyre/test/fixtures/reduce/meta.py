"""SIGNATURE + VARIANTS + reference oracle + input generator for reduce.

Two reduce shapes:

2D, trailing axis    ``out[m] = OP(in[m, :])`` over N, giving an M-vector.
Rank-3, middle axis  ``out[d0, d2] = OP(in[d0, :, d2])`` -- the reduced axis is
                     not the trailing one, which is the whole point of the case.

Three reductions lower: ``sum``, ``max`` and ``min``, dispatched by
``OP: tl.constexpr``. All three reach ``linalg.reduce`` and run on ``ktir_cpu``.
One combination reaches a Spyre binary and launches -- ``one_tile`` at
``AXIS=0``, the loop-free shape folding the non-stick axis. The Level D banner
records why that one and not the others.

The variants are grouped under Level A-D banners, each of which says what its
level is for and what it deliberately does not vary. See ``fixtures/README.md``
for the field reference and the discovery rules.
"""

import numpy as np
from dataclasses import dataclass

import conftest
from . import kernel
from utils import sticksize, DTYPE_MAP


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
#
# One generator. The two named makers below only give it a shape: the framework
# calls ``inputs`` with the variant's whole ``params`` dict, and the 2D and
# rank-3 variants spell their shape with different argument names.
# ---------------------------------------------------------------------------

def _make_inputs(shape, *, dtype="fp32", axis=1) -> dict:
    """Input of *shape*, and a zeroed output of *shape* with *axis* dropped.

    The output shape follows from the input's and the axis, so no variant states
    it a second place it could get wrong. *axis* defaults to 1 because that is
    what every variant reduces -- N in ``[M, N]``, D1 in ``[D0, D1, D2]`` --
    except ``one_tile``, which sweeps it.

    *dtype* is a :data:`DTYPE_MAP` key, read from the variant's ``params`` rather
    than defaulted here, so the buffers cannot drift from the pointer types
    ``SIGNATURE`` declares.

    Integers take their own branch. ``standard_normal`` cast to int32 truncates
    to a field of -1, 0 and 1, where every row's max is 1 and every row's min is
    -1 -- a reduce could be reducing the wrong axis and still match.
    """
    np_dtype = DTYPE_MAP[dtype]
    shape = tuple(shape)
    rng = np.random.default_rng(seed=0)
    if np.issubdtype(np_dtype, np.integer):
        x = rng.integers(-100, 100, size=shape).astype(np_dtype)
    else:
        x = rng.standard_normal(shape).astype(np_dtype)
    out_shape = shape[:axis] + shape[axis + 1:]
    return {"in_ptr": x, "out_ptr": np.zeros(out_shape, dtype=np_dtype)}


def make_inputs(M, N, DTYPE="fp32", AXIS=1, **_unused) -> dict:
    """``[M, N]`` in, ``[M]`` out at ``AXIS=1`` and ``[N]`` out at ``AXIS=0``.

    ``AXIS`` is read from ``params`` like ``DTYPE`` is, so the buffer the oracle
    is compared against cannot disagree with the axis the kernel folds.
    """
    return _make_inputs((M, N), dtype=DTYPE, axis=AXIS)


def make_inputs_3d(D0, D1, D2, DTYPE="fp32", **_unused) -> dict:
    """``[D0, D1, D2]`` in, ``[D0, D2]`` out."""
    return _make_inputs((D0, D1, D2), dtype=DTYPE)


#: NumPy oracle per op name. The same three serve both ranks and either axis.
_NUMPY_OPS = {"sum": np.sum, "max": np.max, "min": np.min}


def _oracle(OP, axis=1):
    """NumPy oracle for *OP* over *axis*, in the input's own dtype.

    The cast keeps the oracle in the kernel's dtype, so a tolerance is not a
    measure of NumPy's promotion rules. What it does NOT do is make the oracle
    the same arithmetic as the kernel: ``np.sum`` over a float16 array reduces in
    float16 here (measured), but in its own summation order, and two fp16 orders
    over 64 terms differ by several ulp of the result. That difference, not the
    dtype, is what ``one_tile``'s ``atol`` is sized for -- see the note there.
    """
    def run(inputs):
        x = inputs["in_ptr"]
        return _NUMPY_OPS[OP](x, axis=axis).astype(x.dtype)
    return run


#: The oracle for the op Level A pins. A module-level name because Level A
#: declares ``reference`` literally rather than through the factory.
run = _oracle("sum")


# ---------------------------------------------------------------------------
# SIGNATURE
#
# One entry per ``@triton.jit`` argument, per ``fixtures/README.md``. Built from
# a shape and a dtype rather than written out per variant, which is what stops an
# argument going missing: omitting one is harmless only while it is a constexpr
# in every variant of that shape, since constexpr values come from ``params``.
# Flip it to a runtime argument and ``_resolve_variant`` would drop it from the
# ABI and ``run_cpu`` would report a missing kwarg.
# ---------------------------------------------------------------------------

_SHAPE_ARGS = {
    "2d":       {"M": "i32", "N": "i32", "BLOCK_M": "i32",
                 "IN_LAYOUT": "constexpr", "OUT_LAYOUT": "constexpr"},
    "3d":       {"D0": "i32", "D1": "i32", "D2": "i32", "BLOCK_D0": "i32",
                 "IN_LAYOUT": "constexpr", "OUT_LAYOUT": "constexpr"},
    # reduce_one_tile has no BLOCK_M: the block *is* the tensor. It does have
    # AXIS, which the other two do not -- they fold axis 1 by construction.
    "one_tile": {"M": "i32", "N": "i32",
                 "IN_LAYOUT": "constexpr", "OUT_LAYOUT": "constexpr",
                 "AXIS": "constexpr"},
}


def _signature(shape: str, dtype: str) -> dict:
    """The full argument list of the *shape* kernel, with pointers at *dtype*."""
    return {"in_ptr": f"*{dtype}", "out_ptr": f"*{dtype}", **_SHAPE_ARGS[shape]}


#: Module-level default -- reduce_spyre's arg list at fp32, which is what Level A
#: pins.
SIGNATURE = _signature("2d", "fp32")

_SIG_3D = _signature("3d", "fp32")


# ---------------------------------------------------------------------------
# Stick layouts
#
# Every helper takes a *dtype*, never a width: the width follows from the dtype
# and each call site had the dtype in hand, so taking a width would put the
# chance of pairing a dtype with the wrong stick at every one of them.
#
# Each returns the ``("stick", layout)`` pair rather than the bare layout, so a
# layout has one spelling in this file and the only tuples written into ``params``
# are rows: ``_normalise_param_list`` reads any tuple in a values list as a
# ``(label, value)`` pair, and a 3-element layout is not one. The value inside
# stays a tuple because it reaches Triton as a constexpr and has to be hashable.
# ---------------------------------------------------------------------------

def _stick_of(dtype: str) -> int:
    """Lanes to a stick at *dtype*: 64 at fp16, 32 at fp32 and i32."""
    return sticksize({"p": f"*{dtype}"}, "p")


def _stick_1d(dtype: str) -> tuple:
    """``[M]`` -> ``[ceil(M/S), S]``. The reduce *output* layout."""
    stick = _stick_of(dtype)
    return ("stick", ((0, "floordiv", stick), (0, "mod", stick)))


def _stick_2d_on_n(dtype: str) -> tuple:
    """``[M, N]`` -> ``[ceil(N/S), M, S]``: stick on the reduced axis.

    The reduced axis is split across the leading and trailing physical dims, so
    this is the source-reduce path through ``RewriteDescriptorLayout``.
    """
    stick = _stick_of(dtype)
    return ("stick", ((1, "floordiv", stick), 0, (1, "mod", stick)))


def _stick_3d_on_d2(dtype: str) -> tuple:
    """``[D0, D1, D2]`` -> ``[ceil(D2/S), D0, D1, S]``: stick on the *unreduced*
    trailing axis, which leaves the reduced axis (D1) non-trailing in the
    physical tile.
    """
    stick = _stick_of(dtype)
    return ("stick", ((2, "floordiv", stick), 0, 1, (2, "mod", stick)))


def _stick_on_n_row(dtype: str, n_sticks: int) -> tuple:
    """One row for the ``("DTYPE", "N", "IN_LAYOUT", "OUT_LAYOUT")`` group: an
    ``[M, N]`` reduce *n_sticks* sticks wide at *dtype*.

    ``N`` and both layouts take their width from the one *dtype* argument, so a
    row cannot pair a 128-lane ``N`` with a 32-lane layout. No width is written
    here or at the call site: *n_sticks* says how many, the dtype says how wide.

    Worked out, because the entry that reaches the kernel is a nest of tuples and
    nothing else here shows it. ``_stick_on_n_row("fp16", n_sticks=2)`` at ``M=64``
    is the row::

        DTYPE      = "fp16"                                  # stick S = 64
        N          = 128                                     # 2 sticks of 64
        IN_LAYOUT  = ((1, "floordiv", 64), 0, (1, "mod", 64))
        OUT_LAYOUT = ((0, "floordiv", 64),    (0, "mod", 64))

    ``OUT_LAYOUT`` is the same idea over a 1-D output: ``[X]`` ->
    ``[X // 64, X % 64]``. It does not depend on which axis was reduced, so one
    layout serves both -- only the surviving extent differs, and the kernel
    derives that. At ``M=64, N=128``: reducing N leaves M, so ``[64]`` ->
    ``memref<1x64>``; reducing M leaves N, so ``[128]`` -> ``memref<2x64>``.
    """
    return (dtype, n_sticks * _stick_of(dtype),
            _stick_2d_on_n(dtype), _stick_1d(dtype))


def _stick_on_d2_row(dtype: str, n_sticks: int) -> tuple:
    """One row for the ``("DTYPE", "D2", "IN_LAYOUT")`` group -- the rank-3
    counterpart of :func:`_stick_on_n_row`.

    ``OUT_LAYOUT`` is outside the group because it stays ``None``: it follows
    from nothing, and a row is for values that follow from the dtype.
    """
    return (dtype, n_sticks * _stick_of(dtype), _stick_3d_on_d2(dtype))


# ---------------------------------------------------------------------------
# Factory — Reduce(VariantFactory)
#
# Supplies the three fields that vary with the swept combination: the signature
# (from DTYPE), the oracle (from OP) and the input maker (from the shape). A
# variant carrying it must not also declare the literal fields, inherited ones
# included, which is why each one below sets ``"base": None``.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reduce(conftest.VariantFactory):
    """``shape`` names a ``_SHAPE_ARGS`` entry: the argument list that joins the
    signature, and which input maker gives the buffers their rank.

    Nothing here has anything to say about the layout a variant sticks -- that
    follows from ``DTYPE``, and a ``params`` group states it beside the dtype
    without a factory in sight.
    """
    shape: str = "2d"

    def signature(self, DTYPE, **_):
        return _signature(self.shape, DTYPE)

    def reference(self, OP, AXIS=1, **_):
        return _oracle(OP, AXIS)

    def inputs(self, **_):
        return make_inputs_3d if self.shape == "3d" else make_inputs



# ---------------------------------------------------------------------------
# VARIANTS
#
# Two per-variant knobs:
#   - ``constexpr`` : list of arg names to bake in as Triton constexprs. Each
#                     variant declares the full list explicitly; no subset
#                     overrides of the default's list.
#   - ``params``    : dict of arg name -> list of values, crossed to one registry
#                     entry per combination. A key may instead be a tuple of
#                     names whose value is a list of rows, sweeping those names
#                     jointly -- see ``fixtures/README.md``.
# ---------------------------------------------------------------------------

VARIANTS = {
    # -----------------------------------------------------------------------
    # Level A -- shape and distribution
    #
    # DTYPE and OP are pinned to fp32/sum throughout: these variants are about
    # descriptor shape and how the work divides across cores, and sweeping the
    # other two axes would multiply keys without covering anything shape-related.
    # No layout annotation either -- that is Level C.
    # -----------------------------------------------------------------------
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "program-id-1d", "num-programs-fold"],
        "summary": "Row-sum reduce: out[m] = sum(in[m, :]), static M/N, no layout.",
        "kernel_fn": kernel.reduce_spyre,
        "SIGNATURE": SIGNATURE,
        "constexpr":  ["M", "N", "BLOCK_M", "IN_LAYOUT", "OUT_LAYOUT", "OP"],
        # BLOCK_M is fixed while M and N sweep: BLOCK_M interacts with the
        # grid partition (rows_per_core = cdiv(cdiv(M, BLOCK_M), grid)), so
        # sweeping both at once would conflate tiling and distribution.
        # M = 512 divides 16 evenly; 768 gives a non-multiple block count
        # (48 blocks over 32 cores -> ragged rows_per_core).
        "params": {
            "M": [512, 768], "N": [64, 256], "BLOCK_M": [16],
            "IN_LAYOUT": [None], "OUT_LAYOUT": [None],
            "DTYPE": ["fp32"], "OP": ["sum"],
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
    # ---- Grid variation -----------------------------------------------------
    # ``grid`` is a top-level entry field, read once per variant, so it is not
    # sweepable through ``params`` -- varying it means sibling variants. BLOCK_* is
    # held fixed and the extents vary, per the tiling/distribution separation
    # noted on ``default``.
    "grid_8": {
        "base": "default",
        "summary": (
            "Row-sum reduce on 8 cores instead of 32 — same work, more rows "
            "per core, exercising a different DistributeWork split."
        ),
        "params": {
            "M": [512, 768], "N": [64], "BLOCK_M": [16],
            "IN_LAYOUT": [None], "OUT_LAYOUT": [None],
            "DTYPE": ["fp32"], "OP": ["sum"],
        },
        "grid": [8],
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
        "constexpr":  ["D0", "D1", "D2", "BLOCK_D0", "IN_LAYOUT", "OUT_LAYOUT",
                       "OP"],
        "params": {
            # Distinct extents so a wrong-axis reduce cannot accidentally match.
            # BLOCK_D0 == D0 keeps grid=[1] a single full-tensor block, so the
            # reduce stays the only interesting structure at the default grid;
            # the middle_axis_grid variant below splits it across cores.
            "D0": [16], "D1": [96], "D2": [64], "BLOCK_D0": [16],
            "IN_LAYOUT": [None], "OUT_LAYOUT": [None],
            "DTYPE": ["fp32"], "OP": ["sum"],
        },
        "grid":       [1],
        "reference":  run,
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
            "DTYPE": ["fp32"], "OP": ["sum"],
        },
        "grid": [4],
    },


    # -----------------------------------------------------------------------
    # Level B -- compute
    #
    # The op x dtype product on ktir_cpu: what a reduce *computes*. Deliberately
    # the simplest shape in the file -- 2D, static, one tile on one core, no
    # layout -- so the combiner is the only thing that differs between entries.
    #
    # All nine entries pass. None carries ``compiles_to_binary``, and not because
    # of the arithmetic: these distribute over the program id and dbo-opt refuses
    # the loop that outlines from that. The one reduce that does reach a binary is
    # the loop-free ``one_tile`` at ``AXIS=0`` (Level D).
    # -----------------------------------------------------------------------
    "2d_compute": {
        # No base: the factory produces SIGNATURE, reference and inputs, and a
        # literal field beside the hook that produces it is a collection-time
        # error -- including one inherited from ``default``, since the merge
        # grammar cannot delete a key.
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "program-id-1d"],
        "summary": (
            "2D trailing-axis reduce across fp16/fp32/i32 and sum/max/min. "
            "Sweeps the OP × DTYPE product to cover ktir_cpu correctness."
        ),
        "kernel_fn":    kernel.reduce_spyre,
        "factory":      Reduce(shape="2d"),
        "constexpr":    ["M", "N", "BLOCK_M", "IN_LAYOUT", "OUT_LAYOUT", "OP"],
        "params": {
            "DTYPE":      ["fp16", "fp32", "i32"],
            "OP":         ["sum", "max", "min"],
            "M":          [64], "N": [64], "BLOCK_M": [64],
            "IN_LAYOUT":  [None], "OUT_LAYOUT": [None],
        },
        "grid":         [1],
        "output_key":   "out_ptr",
        # Sized for the loosest arm, measured rather than guessed. Eight of the
        # nine are bit-exact: max and min at every dtype, because a maximum is a
        # selection and reordering the comparisons cannot change it, and i32 sum,
        # because integer addition is associative. Only the two float sums drift,
        # from linalg.reduce accumulating the 64 terms in an order it does not
        # promise to share with NumPy's -- fp32 by 1.9e-6 absolute (2.1e-5
        # relative), fp16 by 1.6e-2 (4.9e-2), against row sums up to 24.
        #
        # So it is fp16 sum that sets both numbers, and the fp32 arm is checked
        # far more loosely than it could be. One tolerance covers all nine
        # because rtol/atol are fields, not params; the alternative is nine
        # variants, or a tolerance hook, which is mechanism.
        "rtol":         1e-2,
        "atol":         5e-2,
        "extra_checks": lambda t: (
            t.assert_present("linalg.reduce"),
            t.assert_absent("tt.reduce"),
        ),
    },


    # -----------------------------------------------------------------------
    # Level C -- layout
    #
    # Stick physicalization without going to a binary: what the descriptor
    # rewrite does to a reduce when the reduced axis is, and is not, the one
    # being stuck. OP is pinned to sum -- the combiner is Level B's business and
    # the layout rewrite never looks at it.
    #
    # One dtype per variant, each spelled once, in the row that carries
    # everything following from it. A second dtype row would be one line and is
    # not there for one reason: rtol/atol are fields, not params, so both rows
    # would share the tolerance the wider dtype needs, and the fp32 arm would be
    # checked more loosely than it can be.
    # -----------------------------------------------------------------------
    "spyre_stick": {
        # in_ptr  [M, N] stick-on-N: phys [ceil(N/S), M, S]
        # out_ptr [M]    stick:      phys [ceil(M/S), S]
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "program-id-1d", "spyre-tensor-layout"],
        "summary": (
            "Row-sum reduce with in_ptr stick-on-N and out_ptr 1D stick. "
            "Exercises the RewriteDescriptorLayout source reduce path."
        ),
        "kernel_fn":  kernel.reduce_spyre,
        "factory":    Reduce(shape="2d"),
        "constexpr":  ["M", "N", "BLOCK_M", "IN_LAYOUT", "OUT_LAYOUT", "OP"],
        "params": {
            # N and both layouts follow from DTYPE, so they share its row: four
            # sticks is 256 lanes at fp16. M and BLOCK_M are outside the group
            # because they follow from nothing.
            ("DTYPE", "N", "IN_LAYOUT", "OUT_LAYOUT"): [
                _stick_on_n_row("fp16", n_sticks=4),
            ],
            "M": [64], "BLOCK_M": [64], "OP": ["sum"],
        },
        "grid":        [1],
        "data_layout": "host",
        "output_key":  "out_ptr",
        "rtol":        1e-2,
        "atol":        5e-2,
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            t.assert_present("linalg.reduce"),
        ),
    },
    "middle_axis_spyre_stick": {
        # in_ptr stick-on-D2 (fp32 stick = 32, D2 = 64 = 2 sticks exactly):
        #   phys [D2//32, D0, D1, D2%32] = [2, 16, 96, 32]
        # The reduced axis (D1) is non-trailing in the physical tile, and that is
        # the case: linalg.reduce takes a sorted `dimensions` list, so D1 is named
        # where it sits and no transpose is emitted. A slot-index-derived
        # permutation would be identity here and would reduce the wrong axis.
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "spyre-tensor-layout"],
        "summary": (
            "Rank-3 middle-axis reduce with in_ptr stick-on-D2. The reduced "
            "axis is non-trailing in the physical tile, and stays where it is: "
            "linalg.reduce names it rather than rotating it to the end."
        ),
        "kernel_fn":  kernel.reduce_middle_axis_spyre,
        "factory":    Reduce(shape="3d"),
        "constexpr":  ["D0", "D1", "D2", "BLOCK_D0", "IN_LAYOUT", "OUT_LAYOUT",
                       "OP"],
        "params": {
            ("DTYPE", "D2", "IN_LAYOUT"): [
                _stick_on_d2_row("fp32", n_sticks=2),
            ],
            "D0": [16], "D1": [96], "BLOCK_D0": [16],
            "OUT_LAYOUT": [None], "OP": ["sum"],
        },
        "grid":        [1],
        "data_layout": "host",
        "output_key":  "out_ptr",
        "rtol":        1e-4,
        # linalg.reduce accumulates the 96 terms in a different order than
        # NumPy's sum, so fp32 drifts ~1e-5 absolute on a few elements.
        "atol":        1e-4,
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            t.assert_present("linalg.reduce"),
            # No transpose: the reduced axis is named where it sits.
            t.assert_absent("linalg.transpose"),
        ),
    },


    # -----------------------------------------------------------------------
    # Level D -- device
    #
    # The only reduce shape in this fixture with a device story: loop-free and
    # stick-tiled, which is elementwise's Level D shape. Every other variant here
    # outlines an scf.for from its program-id distribution and dbo-opt rejects
    # that loop, so a device result on any of them would be a statement about
    # distribution rather than about reduce.
    #
    # TWO VARIANTS RATHER THAN ONE AXIS SWEEP, because the two axes have
    # different device stories and a variant is the unit that can say so. What
    # divides them is not params -- ``compiles_to_binary`` and ``atol`` are
    # fields, so one variant cannot carry two answers -- exactly the constraint
    # that keeps Level C at two variants. Collapse them back into a single
    # ``AXIS: [0, 1]`` sweep the day the on-stick arm reaches the device too.
    #
    # The first wall is shared by both, and neither variant asserts it:
    #
    #   V1 only supports add/mul/sub/reduce compute ops; found unsupported
    #   compute op
    #
    # naming the ``linalg.fill`` that LowerComputeOps puts on the reduction's
    # ``outs``. ``linalg.reduce`` itself is in that allowlist; the neutral-element
    # fill beside it is not. ``DropReductionInitFill`` removes exactly that fill,
    # and ``_make_spyrecode`` installs it unconditionally, out of
    # ``_SPYRECODE_STAGE_PASSES`` -- so what reaches the
    # device is the emission with the fill already gone, which is what makes it
    # match torch-spyre's, whose emitter never writes one.
    #
    # DTYPE and OP are pinned in both: the fill and then the shape are what
    # decide the outcome, and every dtype and every combiner produce those alike.
    # -----------------------------------------------------------------------

    # Folds M, the NON-stick axis -- a whole physical dimension. The stick split
    # of N survives, so RewriteDescriptorLayout physicalizes the reduce's output
    # and the surviving stick index rides along as a batch dimension of the one
    # linalg.reduce (``ins tensor<2x64x64> outs tensor<2x64> dimensions = [1]``),
    # with ktdp.store consuming it directly. That is the shape torch-spyre's
    # working ``sum`` emits, and it is the one reduce here that reaches a binary
    # and launches. ``rewrite-descriptor-layout-reduce-batch-dim.mlir`` pins the
    # emitted form.
    "one_tile": {
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "simplified:no-loop", "spyre-tensor-layout"],
        "summary": (
            "Sum over a single stick-tiled tile, folding the non-stick axis, no "
            "distribution loop — the one reduce that reaches a Spyre binary, "
            "with the surviving stick index as a batch dimension."
        ),
        "kernel_fn":  kernel.reduce_one_tile,
        "factory":    Reduce(shape="one_tile"),
        "constexpr":  ["M", "N", "IN_LAYOUT", "OUT_LAYOUT", "OP", "AXIS"],
        "params": {
            ("DTYPE", "N", "IN_LAYOUT", "OUT_LAYOUT"): [
                _stick_on_n_row("fp16", n_sticks=2),
            ],
            # M = 64 and N = 128, so folding M leaves N as two whole sticks at
            # fp16. A ragged extent would be testing stick padding, which is not
            # what this is for.
            "M": [64], "OP": ["sum"], "AXIS": [0],
        },
        "grid":        [1],
        # No tl.program_id, so DistributeWork has nothing to place and the
        # presence check would fail on a kernel that is correct.
        "parallel":    False,
        "data_layout": "host",
        "compiles_to_binary": True,
        "output_key":  "out_ptr",
        "rtol":        1e-2,
        # Set by the DEVICE arm, and looser than the elementwise-shaped 5e-2 the
        # other stick variants use because a reduce accumulates where an
        # elementwise op does not: 5e-2 is only 3.2 ulp at this output's
        # magnitude, and the device lands past it.
        #
        # Sized in ulp rather than picked. The sums reach 24, where fp16 ulp is
        # 0.015625, so 0.25 is 16 of them. Measured against the fp16 oracle on
        # this exact input (seed 0, M=64, N=128):
        #
        #     device     0.1367   =  8.75 ulp   <- what sets this
        #     ktir_cpu   0.0205   =  1.31 ulp
        #
        # 8.75 ulp is what a 64-term reordering predicts -- the drift grows as
        # sqrt(64) = 8 ulp -- so the device is behaving, and 16 ulp is that with
        # a factor of two. Not the worst case, which is 64 ulp = 1.0, and not a
        # number fitted to the element that failed at 5e-2.
        #
        # It is this device's accumulation order that differs, not fp16 order in
        # general: NumPy's fp16 sum and a plain sequential fp16 sum agree here
        # bit for bit, so there is no oracle-side order to match. Widening the
        # oracle to an fp32 accumulator does not close the gap either.
        #
        # The on-stick sibling below keeps 5e-2: it never runs on the device, so
        # nothing there asks for this. That is what the split buys.
        #
        # An output that was never written is caught by test_device_launch's own
        # nonzero assertion, not by a tolerance.
        "atol":        2.5e-1,
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            t.assert_present("linalg.reduce"),
            # The zero init fill is still here, and that is correct at this
            # stage: DropReductionInitFill runs in _make_spyrecode, so the KTIR a
            # structural test sees is the KTIR before the binary path repairs it.
            # Asserted rather than left unsaid because its absence would mean the
            # pass had moved back into the pipeline every path crosses.
            #
            # It is re-emitted at the PHYSICAL shape here rather than dropped,
            # because a reduce's `outs` is read by its payload and the neutral
            # element is what makes that well defined.
            t.assert_present("linalg.fill"),
            # No stick loop -- the whole point of this variant, and what its
            # device story turns on. `parallel: False` above says there is no
            # distribution loop; this says there is no stick loop either.
            t.assert_absent("scf.for"),
        ),
    },

    # Folds N, the STICK axis, which stick-on-N splits across physical dims 0
    # and 2. The reduce names both and collapses to a rank-1 result that needs
    # widening for the rank-2 store, and dbo-opt stops on ktdf.data_transfer
    # having a rank-2 dest against a 1-result dest_map. No batch dimension
    # survives to carry the split, so the sibling above's path does not apply --
    # and torch-spyre does not emit linalg.reduce for this case at all, using a
    # linalg.generic with the maps written out, which currently fails there too.
    # So there is no working emission to match yet.
    #
    # That refusal is why this variant carries no ``compiles_to_binary``. It is
    # not asserted anywhere: what is under test is that the reduce lowers and
    # computes the right numbers on ktir_cpu at this axis.
    "one_tile_on_stick": {
        "base": "one_tile",
        "summary": (
            "The same single-tile sum folding the stick axis instead. Two "
            "physical dims are reduced, no batch dimension survives, and it "
            "stops in the scheduler — so ktir_cpu only."
        ),
        "params": {
            ("DTYPE", "N", "IN_LAYOUT", "OUT_LAYOUT"): [
                _stick_on_n_row("fp16", n_sticks=2),
            ],
            # Folding N leaves M = 64, one whole stick at fp16.
            "M": [64], "OP": ["sum"], "AXIS": [1],
        },
        # Shallow merge replaces the whole field, so the device arm's
        # compiles_to_binary is not inherited -- but say so rather than relying
        # on a reader knowing that: this arm does not reach a binary.
        "compiles_to_binary": False,
        # Back to the elementwise-shaped tolerance. Nothing here runs on the
        # device, and on ktir_cpu this arm drifts 0.0122 = 0.78 ulp, so 5e-2
        # (3.2 ulp) is already generous. Inheriting the sibling's 0.25 would
        # check it 20x looser than it needs for no reason.
        "atol":        5e-2,
    },
}
