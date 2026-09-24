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

Level D has a SECOND GROUP, and it is a reduce's CONSUMER rather than a fourth
reduction: ``out = x - sum(x, axis)``, a statistic written to HBM and read back by
a later elementwise op, at both axes. Softmax's first two groups. Both arms reach
a binary and launch; the on-stick one is checkable only on the device, and that
group's banner says why.

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
# The statistic chain — inputs and oracle
#
# ``out = x - sum(x, axis)``: a reduce whose own statistic is read back by a
# later elementwise op. Three buffers, and the statistic is an intermediate --
# ``output_key`` is ``out_ptr``, so nothing checks the statistic directly and it
# is checked through what the second group did with it, which is the point.
#
# Both helpers take the axis rather than reading it off a variant, so the buffer
# shapes and the oracle's reduction axis come from one value. The kernels bake
# their axis in (there is no ``AXIS`` argument), so that value lives on the
# factory and reaches both from there.
# ---------------------------------------------------------------------------

def _make_inputs_stat_chain(M, N, DTYPE="fp16", *, axis) -> dict:
    """``[M, N]`` in, an ``[M]`` or ``[N]`` statistic, an ``[M, N]`` out.

    The statistic's extent is the one the reduce leaves, so no variant states it
    a second place it could contradict the axis. Its host shape is the LOGICAL
    one even when the layout replicates: what the device needs is derived from the
    compiled kernel's recorded footprint, not from here -- see ``device_alloc_from``
    in ``test_device_launch.py``.

    Floats only, unlike :func:`_make_inputs`: both arms of this shape sum, and a
    sum through HBM is not a path any integer variant takes, so an integer branch
    would be dead code. The dtype still comes from ``params`` as it does there.

    fp16 is the default because it is the only dtype BOTH arms reach a binary at.
    The off-stick arm has a working fp32 device variant; the on-stick one cannot
    have one, and the reason is the tool rather than this fixture -- the lane-0
    read that spreads a statistic across its stick becomes a
    ``vectorchain.shuffle``, which ``dbo-opt`` takes at 2 bytes and refuses at 4.
    """
    np_dtype = DTYPE_MAP[DTYPE]
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((M, N)).astype(np_dtype)
    stat_extent = N if axis == 0 else M
    return {"x_ptr":    x,
            "stat_ptr": np.zeros(stat_extent, dtype=np_dtype),
            "out_ptr":  np.zeros((M, N), dtype=np_dtype)}


def _stat_chain_inputs(axis):
    """The input maker for a chain folding *axis*, as the framework calls it."""
    def make(M, N, DTYPE="fp16", **_unused) -> dict:
        return _make_inputs_stat_chain(M, N, DTYPE, axis=axis)
    return make


def _stat_chain_oracle(axis):
    """``x - sum(x, axis)`` in the input's own dtype, broadcast back over *axis*.

    In the kernel's dtype for the reason :func:`_oracle` gives, and with the same
    caveat: this is not the kernel's arithmetic. The statistic is summed in the
    input's dtype but in a different summation order, and the subtract passes that
    difference straight through -- so the variant's ``atol`` is sized in ulp of the
    STATISTIC, not of the output.
    """
    def run(inputs):
        x = inputs["x_ptr"]
        stat = np.sum(x, axis=axis).astype(x.dtype)
        spread = stat[None, :] if axis == 0 else stat[:, None]
        return (x - spread).astype(x.dtype)
    return run


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


def _signature_stat_chain(dtype: str, axis: int) -> dict:
    """A statistic chain's arg list -- three pointers over three buffers.

    Not a ``_SHAPE_ARGS`` entry, because the argument list is not a fixed row:
    the on-stick arm takes the stick width ``S`` and the off-stick arm does not,
    so the shape it would key on is the axis, and the axis is the one thing these
    two kernels do not share. Built from it directly instead.

    Order matters and is the kernels' declaration order: the pointer ordinals are
    what ``device_layouts`` is keyed by, so a signature listing them differently
    would pair a recorded layout with the wrong buffer.
    """
    sig = {"x_ptr": f"*{dtype}", "stat_ptr": f"*{dtype}", "out_ptr": f"*{dtype}",
           "M": "i32", "N": "i32"}
    if axis == 1:
        sig["S"] = "i32"
    return {**sig, "TILE_LAYOUT": "constexpr", "STAT_LAYOUT": "constexpr"}


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


def _stick_1d_splat(dtype: str) -> tuple:
    """``[M]`` -> ``[M, S]``: the statistic *replicated* across a stick.

    The counterpart of :func:`_stick_1d`, and the difference is the whole point.
    ``_stick_1d`` *partitions* M's elements over two physical dims, so the element
    count is unchanged and one statistic lands in one lane. This one *splats*:
    physical dim 1 is a replication of width S, so each statistic occupies a whole
    stick. That is the form a stick-axis reduce has to store, because the lanes it
    reduced are read as one dim and written as another -- a floordiv/mod split,
    which keeps the element count, has nowhere to put the surviving extent.

    Note the layout is what targets ``[M, S]``; the host buffer stays logical
    ``[M]``, and the allocation is derived from the compiled kernel's own recorded
    footprint rather than restated here. See ``device_alloc_from`` in
    ``test_device_launch.py``.
    """
    return ("splat", (0, (0, "splat", _stick_of(dtype))))


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


def _stick_on_n_row_splat(dtype: str, n_sticks: int) -> tuple:
    """:func:`_stick_on_n_row` with the *splat* output layout.

    Same input side, so the reduce folds the same axis; only where the statistic
    lands differs. At ``M=64, N=128`` fp16 this is::

        IN_LAYOUT  = ((1, "floordiv", 64), 0, (1, "mod", 64))   # [64,128] -> [2,64,64]
        OUT_LAYOUT = (0, (0, "splat", 64))                      # [64]     -> [64,64]
    """
    return (dtype, n_sticks * _stick_of(dtype),
            _stick_2d_on_n(dtype), _stick_1d_splat(dtype))


def _stick_on_d2_row(dtype: str, n_sticks: int) -> tuple:
    """One row for the ``("DTYPE", "D2", "IN_LAYOUT")`` group -- the rank-3
    counterpart of :func:`_stick_on_n_row`.

    ``OUT_LAYOUT`` is outside the group because it stays ``None``: it follows
    from nothing, and a row is for values that follow from the dtype.
    """
    return (dtype, n_sticks * _stick_of(dtype), _stick_3d_on_d2(dtype))


def _stat_chain_row_on_stick(dtype: str, n_sticks: int) -> tuple:
    """One row for the on-stick chain's
    ``("DTYPE", "N", "S", "TILE_LAYOUT", "STAT_LAYOUT")`` group.

    :func:`_stick_on_n_row_splat` plus the stick width, taken from the same one
    *dtype* argument that built both layouts -- so the ``S`` the kernel declares
    its statistic read view with cannot be a different width from the splat that
    wrote it. That agreement is the whole reason ``S`` comes through a row rather
    than being written beside ``M``.

    The OFF-stick arm needs no row of its own: its two layouts are exactly
    :func:`_stick_on_n_row`'s pair, and only the argument names they land on
    differ (``TILE_LAYOUT``/``STAT_LAYOUT`` rather than ``IN_LAYOUT``/
    ``OUT_LAYOUT``), which is a property of the group KEY and not of the row.
    """
    dt, n, tile, stat = _stick_on_n_row_splat(dtype, n_sticks)
    return (dt, n, _stick_of(dtype), tile, stat)


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


@dataclass(frozen=True)
class StatChain(conftest.VariantFactory):
    """The three combination-dependent fields of a statistic chain variant.

    ``axis`` is the one thing that separates the two arms, and it is a factory
    field rather than a ``params`` entry because the kernels bake their axis in --
    there is no ``AXIS`` argument for a param to fill, and a param stating it
    beside a ``kernel_fn`` that already implies it would be a second place to get
    it wrong. From here it reaches the signature (whether ``S`` is an argument),
    the oracle (which axis it sums) and the input maker (which extent the
    statistic gets), so all four agree by construction.
    """
    axis: int = 0

    def signature(self, DTYPE, **_):
        return _signature_stat_chain(DTYPE, self.axis)

    def reference(self, **_):
        return _stat_chain_oracle(self.axis)

    def inputs(self, **_):
        return _stat_chain_inputs(self.axis)



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
        # No "data_layout". It selected the NAMED RewriteDescriptorLayout's
        # "device"/"host" stride mode, and that pass roots on a
        # tt.spyre_tensor_layout op. tl.spyre_tensor_layout authors
        # tts.tensor_layout now, so the named pass no-ops on every kernel in this
        # tree and the option reached nothing. The generic pass that physicalizes
        # these -- in the spyrecode stage -- has no equivalent option and needs
        # none: a caller wanting the logical form reads the ktir artifact, which
        # is logical. Removed rather than left as dead config, because conftest
        # forwards any key naming a SpyreOptions field and the field still
        # exists, so it would have kept being passed and kept doing nothing.
        "output_key":  "out_ptr",
        "rtol":        1e-2,
        "atol":        5e-2,
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
        "output_key":  "out_ptr",
        "rtol":        1e-4,
        # linalg.reduce accumulates the 96 terms in a different order than
        # NumPy's sum, so fp32 drifts ~1e-5 absolute on a few elements.
        "atol":        1e-4,
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
    # TWO GROUPS OF VARIANTS, at that one shape, and the split is what the reduce
    # is asked to do rather than another axis of the ladder:
    #
    #   the reduce alone        ``one_tile`` and its two on-stick siblings: where a
    #                           statistic LANDS, at either axis and at either
    #                           output layout.
    #   the reduce's consumer   ``stat_chain_*``: the same statistic READ BACK by a
    #                           later compute group, ``out = x - sum(x, axis)``.
    #                           Softmax's G1 and G2. Each arm's G1 is verbatim one
    #                           of the three above, so everything that is new is
    #                           G2.
    #
    # Both groups belong at this level and not at a new one: the ladder's levels
    # are what a band VARIES -- shape, compute, layout, device -- and the chain
    # varies none of those. It is this level's shape with a second group bolted on,
    # aimed at the same tier. The second group's own banner is below,
    # above ``stat_chain_off_stick``.
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
    # and the spyrecode stage installs it unconditionally, out of
    # ``buildSpyrecodePipeline``'s own list -- so what reaches the
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

    # The same stick-axis fold as the sibling above, storing its statistic
    # REPLICATED across a stick instead of split across one -- and that is what
    # reaches a binary where the split form does not.
    #
    # The split cannot work and it is structural, not a tuning question. Under
    # stick-on-N the lanes being folded are read as one physical dim and the
    # statistic is written as another, so _stick_1d's floordiv/mod -- which
    # PARTITIONS, keeping the element count -- has nowhere to put the surviving
    # extent. A splat coord op REPLICATES: logical [M] becomes physical [M, S],
    # one whole stick per statistic. The emitted reduce is then byte-identical to
    # the form torch-spyre's own KTIR backend emits for torch.sum(x, dim=-1):
    #
    #   indexing_maps  = [(d0,d1,d2,d3) -> (d0,d1,d2),
    #                     (d0,d1,d2,d3) -> (d1,d3)]
    #   iterator_types = ["reduction", "parallel", "reduction", "parallel"]
    #
    # d3 appears in no input map, which is what replicates: for a row, every
    # output lane receives the same reduced value. rebuild-reduction.mlir pins
    # that form.
    #
    # No pass change was needed for it. The "a stick dim cannot be sub-stick"
    # check keys on the mod coord op, and a splat dim is not one, so it never
    # fires here. It was expected to be the wall on the READ side of a statistic
    # (a block shape of [M, 1] under a stick layout), which softmax needs and this
    # does not -- and the chain group below, which does need it, found that it is
    # not: a read view carrying NO layout has no stick dim to be sub-stick, and
    # the [M, 1] tile it takes is accepted.
    #
    # The replication is 8 elements wide on the DEVICE, not S. That is a device
    # fact with no representation in the layout: the buffer still has to span
    # M x S because S is the stride, and the host tensor addresses lane 0 only,
    # which is what makes the oracle comparison correct. Reading the lane matching
    # one's own output would read memory nothing wrote.
    "one_tile_on_stick_splat": {
        "base": "one_tile",
        "summary": (
            "The stick-axis sum again, storing its statistic SPLAT across a "
            "stick instead of split across one. The split form cannot be "
            "scheduled -- the lanes are reduced as one physical dim and would be "
            "written as another -- and the splat form is what the reference "
            "chains store. This is the arm that reaches a binary."
        ),
        "params": {
            ("DTYPE", "N", "IN_LAYOUT", "OUT_LAYOUT"): [
                _stick_on_n_row_splat("fp16", n_sticks=2),
            ],
            "M": [64], "OP": ["sum"], "AXIS": [1],
        },
        "compiles_to_binary": True,
        # No "device_alloc" and nothing restating OUT_LAYOUT. The output buffer
        # cannot be staged with `.to("spyre")` -- that path allocates the host
        # element count, 128 bytes here, against the 8192 the splat layout
        # addresses -- and the harness gets the right number by asking the
        # COMPILED KERNEL for it, from metadata["device_layouts"]. Which buffers
        # need that treatment follows from the recorded layouts too, so this
        # variant declares nothing about it. See device_alloc_from in
        # test_device_launch.py.
        "atol":        2.5e-1,
    },


    # ---- Level D, second group: the reduce's CONSUMER ----------------------
    #
    # A reduce whose own statistic is read back by a later elementwise op:
    # ``out = x - sum(x, axis)``, softmax's G1 and G2 with the max replaced by a
    # sum. The variants above prove a reduce can WRITE a statistic at either axis;
    # these two are the first thing here to read one back, which is the dependency
    # every normalisation is built on and which nothing in this tier covered.
    #
    # TWO ARMS, one per axis, as above and for the same reason -- and the finding
    # is that the axis is NOT where they differ. Each arm's G1 is verbatim one of
    # the variants above that already reaches the device (``one_tile`` for the
    # off-stick arm, ``one_tile_on_stick_splat`` for the on-stick one), so
    # everything below is about G2.
    #
    # BOTH REACH A BINARY AND LAUNCH, and what made that true is one pass rather
    # than anything about either arm. Reading a statistic back and applying it to a
    # full tile is a Triton BROADCAST, and a broadcast used to reach the layout pass
    # as a linalg.generic of its own that computes nothing -- it yields its input
    # and only its maps differ. That op has no descriptor, so no layout marker, so
    # RewriteDescriptorLayoutGeneric left its result LOGICAL while its producer and
    # consumer were both physical, and the consumer's operand map was left to bridge
    # the two. It bridged it with a LINEARIZATION, identically in both arms
    # (`(d0, d1, d2) -> (d1, d0 * 64 + d2)`), and dbo-opt aborted on the off-stick
    # arm inside upstream's own fusion pass and refused the on-stick arm's [M, 1]
    # statistic load.
    #
    # FoldDataMovementGenerics closes it, in the spyrecode stage ahead of the layout
    # pass: the coordinate change becomes an operand map the layout pass restates at
    # physical rank like any other. What each arm now emits, and the checkpoint
    # worth keeping because it is the whole reason the arms compile:
    #
    #   off-stick   the broadcast FOLDS, consumer operand map
    #               `(d0, d1, d2) -> (d0, d2)`, and no linearizing map anywhere
    #   on-stick    the unit-dim collapse in front of the broadcast is ABSORBED,
    #               giving `(d0, d1, d2) -> (d1, 0)` -- a statistic read at a
    #               constant lane, which is what case 4 of
    #               ``rebuild-reduction.mlir`` states -- and no
    #               tensor.collapse_shape survives
    #
    # That fixture's input is hand-written POST-fold IR, and this group is what now
    # produces the same form end to end.
    #
    # The walls the reduce family's own banners predicted did NOT fire, and both
    # non-firings are worth having recorded:
    #
    #   - The sub-stick refusal on the statistic READ. Level D's banner names a
    #     block shape of [M, 1] under a stick layout as the wall softmax will hit.
    #     The read view carries no layout at all, so there is no stick dim to be
    #     sub-stick, and it physicalizes to itself: memref<64x64> with a
    #     !ktdp.access_tile<64x1>. The pattern works.
    #   - G1 itself. Both arms emit exactly the reduce their Level D sibling does,
    #     the on-stick one byte-identical to case 2 of ``rebuild-reduction.mlir``
    #     (["reduction", "parallel", "reduction", "parallel"], output map
    #     (d0,d1,d2,d3) -> (d1,d3)).
    #
    # One alternative spelling was tried for the on-stick arm and REJECTED, which
    # is worth a line because it looks better than it is: reading the statistic
    # back through the rank-1 [M] descriptor that wrote it, annotated, and letting
    # Triton broadcast. It is logically faithful, so it passes on ktir_cpu (0.031
    # measured) -- but it physicalizes to a load of the WHOLE stick,
    # tensor<64x64xf16>, and the device replicates a statistic only 8 elements
    # wide, so lanes 8 and up are memory nothing wrote. Green in this tier and
    # wrong on hardware is the one outcome worse than red, so the arm below reads
    # lane 0 through an unannotated rank-2 view instead, and pays for it in this
    # tier rather than on the device.
    #
    # fp16 and sum in both, pinned rather than swept -- ``sum`` because at fp16
    # ``tl.max`` promotes to fp32 and emits an ``arith.extf`` nothing here lowers,
    # and fp16 because the statistic read-back is an fp16-only path downstream. See
    # kernel.py's banner. Neither is a free choice, so neither is a param.
    # -----------------------------------------------------------------------

    # Folds M, the NON-stick axis, so the statistic survives on the stick dim and
    # its layout PARTITIONS. A partition keeps the element count, so the logical
    # [N] buffer is a faithful description of the bytes -- which is why this arm
    # needs only ONE statistic descriptor, needs no stick width, and is checked
    # numerically on ktir_cpu as well as on the device.
    "stat_chain_off_stick": {
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "simplified:no-loop", "spyre-tensor-layout", "hbm-round-trip"],
        "summary": (
            "out[m, n] = x[m, n] - sum(x[:, n]) with the statistic through HBM: "
            "a non-stick-axis reduce, then a second compute group reading the "
            "statistic back and applying it down every row."
        ),
        "kernel_fn":  kernel.stat_chain_off_stick,
        "factory":    StatChain(axis=0),
        "constexpr":  ["M", "N", "TILE_LAYOUT", "STAT_LAYOUT"],
        "params": {
            # _stick_on_n_row's pair unchanged -- the tile is stick-on-N and the
            # statistic is the 1-D partition that leaves. Only the names differ
            # from Level C's, and a name is a property of the key.
            ("DTYPE", "N", "TILE_LAYOUT", "STAT_LAYOUT"): [
                _stick_on_n_row("fp16", n_sticks=2),
            ],
            # M = 64, N = 128: two whole sticks at fp16, nothing ragged.
            "M": [64],
        },
        "grid":        [1],
        # No tl.program_id, so DistributeWork has nothing to place and there is no
        # scf.for for dbo-opt to refuse. The chain reaches a binary because
        # FoldDataMovementGenerics folds the statistic broadcast into its consumer's
        # operand map; see the group banner above for the map it lands on.
        "compiles_to_binary": True,
        "output_key":  "out_ptr",
        # An ABSOLUTE bound alone, and rtol is 0 deliberately rather than omitted.
        # `x - sum(x)` has elements near zero, where a relative bound says nothing,
        # and at the other end |out| reaches 25.5 -- so an rtol of 1e-2 like the
        # siblings carry would contribute 0.25 and be the LOOSER of the two bounds,
        # which is the opposite of what a second bound is for.
        "rtol":        0.0,
        # Sized in ulp of the STATISTIC, which is where the error comes from: the
        # subtract passes it through unamplified. The column sums reach 24.03,
        # where fp16 ulp is 0.015625, and a 64-term reordering predicts the drift
        # growing as sqrt(64) = 8 ulp = 0.125.
        #
        # 0.25 is that prediction with a factor of two -- 16 ulp, not the worst
        # case of 64 ulp = 1.0. Same rule and same answer as ``one_tile``. A
        # summation order that differs from the oracle's puts the drift just
        # outside the predicted bound, so the bound rather than the prediction is
        # what this carries; it is one COLUMN that moves, which is the statistic
        # passing through unamplified rather than the subtract adding anything.
        "atol":        2.5e-1,
    },

    # The same chain at fp32, which is a DIAGNOSTIC before it is coverage: it says
    # whether the sibling's drift is the accumulation order or something
    # structural. The reduce folds M either way, so both arms sum the same 64
    # terms and only the precision differs -- so if the drift is a reordering it
    # falls by the ratio of the ulps and a bound near 1e-5 holds, and if it stays
    # at the fp16 absolute size then the cause is not arithmetic at all.
    #
    # Second job, not incidental: a stick is 32 lanes at fp32 against 64 at fp16,
    # so the absorber and the layout pass are exercised at a different stick width
    # on the same kernel. N follows from the dtype -- `_stick_on_n_row` takes its
    # width from there, so two sticks is 64 elements here where it was 128.
    "stat_chain_off_stick_fp32": {
        "base": "stat_chain_off_stick",
        "summary": (
            "out[m, n] = x[m, n] - sum(x[:, n]) at fp32: the same non-stick-axis "
            "reduce through HBM, at half the stick width and a quarter the ulp."
        ),
        "params": {
            ("DTYPE", "N", "TILE_LAYOUT", "STAT_LAYOUT"): [
                _stick_on_n_row("fp32", n_sticks=2),
            ],
            # M = 64, N = 64: two whole sticks at fp32, nothing ragged.
            "M": [64],
        },
        # The sibling's rule at this dtype: ulp of the statistic, times the
        # sqrt(64) reordering factor, times two. The column sums reach the same
        # 24.03, where fp32 ulp is 1.907e-6, so 16 ulp is 3e-5.
        #
        # This arm ANSWERED the question it was added for. Against the fp16
        # sibling, the drift falls by the ratio of the two ulps rather than staying
        # at the fp16 absolute size -- so it is the summation order and not a
        # mis-addressed element, which would have moved the same distance at either
        # precision. It is also spread across the output rather than concentrated,
        # which is the other thing a mis-addressed statistic would not do.
        "atol":        3e-5,
    },

    # Folds N, the STICK axis, so the statistic must be SPLAT and the logical form
    # stops describing the bytes. Everything this arm costs over its sibling
    # follows from that one fact: a second descriptor, a stick width the kernel has
    # to be told, and no ktir_cpu arm. The DEVICE arm is where it is checked, and
    # that is not a compromise -- the device is the tier whose buffer the [M, S]
    # read view actually describes.
    "stat_chain_on_stick": {
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static", "reduce",
                 "simplified:no-loop", "spyre-tensor-layout", "hbm-round-trip"],
        "summary": (
            "out[m, n] = x[m, n] - sum(x[m, :]) with the statistic through HBM: "
            "a stick-axis reduce storing it splat across a stick, then a second "
            "compute group reading lane 0 of it back and applying it to the tile."
        ),
        "kernel_fn":  kernel.stat_chain_on_stick,
        "factory":    StatChain(axis=1),
        "constexpr":  ["M", "N", "S", "TILE_LAYOUT", "STAT_LAYOUT"],
        "params": {
            # S joins the row so the width the read view declares is the width the
            # splat wrote. See _stat_chain_row_on_stick.
            ("DTYPE", "N", "S", "TILE_LAYOUT", "STAT_LAYOUT"): [
                _stat_chain_row_on_stick("fp16", n_sticks=2),
            ],
            # M = 64 is one whole stick at fp16, so the statistic is one stick of
            # rows and nothing is padded.
            "M": [64],
        },
        "grid":        [1],
        # Same as the sibling: no tl.program_id, so no loop to refuse, and what
        # gets it past the layout pass is FoldDataMovementGenerics ABSORBING the
        # unit-dim collapse in front of the statistic broadcast. See the group
        # banner for the map.
        "compiles_to_binary": True,
        "output_key":  "out_ptr",
        # Sized by the sibling's rule: the row sums reach 27.09, fp16 ulp there is
        # 0.015625, and a 128-term reordering predicts sqrt(128) = 11.3 ulp = 0.18.
        # 0.25 is 16 ulp, the same bound the splat sibling carries.
        "rtol":        0.0,
        "atol":        2.5e-1,
        # WHY THE NUMERICAL ARM CANNOT PASS, and it is a tier boundary rather than
        # a bug to fix here. ``ktir_cpu`` executes the ``ktir`` stage's artifact,
        # which is LOGICAL -- physicalization is in the ``spyrecode`` stage (see
        # fixtures/README.md). The read view's [M, S] shape is a statement about the
        # PHYSICAL buffer: it exists because the splat layout made one stick per
        # statistic. Logically the buffer is a dense [M], so a read at stride S
        # addresses element m*S of an M-element buffer and only m = 0 is the value
        # the writer wrote. The interpreter does not refuse it; it returns other
        # numbers (max |err| 27.1, which is the size of the statistic itself).
        #
        # The DEVICE arm has no such problem and passes: physicalization happens in
        # the spyrecode stage, so the buffer a launch reads really is the [M, S] the
        # splat made and lane 0 really is where the statistic was written. So the
        # xfail below is a statement about ONE tier, not about the arm -- which is
        # the reverse of what it said while neither tier could check it.
        #
        # raises=AssertionError, not left open: the failure that is expected is the
        # comparison's. A compile that stops working would raise RuntimeError out
        # of setup_method, and an open xfail would absorb that too and say nothing.
        # This way the xfail asserts "it lowers and it runs, and the numbers are
        # wrong for a stated structural reason".
        "xfail_numerical": {
            "reason": "the [M, S] statistic read view is a physical-layout shape, "
                      "and ktir_cpu executes the logical ktir artifact, where the "
                      "statistic buffer is a dense [M]; the device arm checks this "
                      "arm instead",
            "strict": True,
            "raises": AssertionError,
        },
    },
}
