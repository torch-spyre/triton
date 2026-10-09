"""SIGNATURE + VARIANTS + reference oracle + input generators for ``where``.

Four variants. The first three split ``tl.where(x OP y, p, q)`` into the parts
that can fail independently:

- ``compare`` -- the comparison alone, storing its mask (1.0 where true, 0.0
  where false).
  ``compare_device`` is its fp32 row on the device tier.
- ``select``  -- the select alone, over a mask the host wrote.
- ``spilled`` -- both in one kernel, with the mask spilled to its own buffer.

The fourth, ``spilled_scalar``, is ``spilled`` with ``y`` replaced by one scalar.
The host fills that scalar into a one-stick buffer, and the kernel broadcasts it
over its rows.

All four sweep fp16 and fp32. All but ``select`` also sweep ``OP`` over every
comparison in ``kernel.COMPARISONS``; ``select`` has no comparison of its own to
sweep. ``select``, ``spilled``, ``spilled_scalar`` and ``compare_device`` are
``compiles_to_binary``; ``compare`` at fp16 is not, and its entry says why.

The branch values are kept separate from the condition, which is what makes a
wrong branch visible. Written ``where(x > y, x, y)`` the kernel computes a
maximum, so an implementation that ignored the mask and returned ``max(x, y)``
would pass. Separate ``p``/``q`` buffers make a wrong branch a wrong value.
Each lane of ``p`` and ``q`` also holds its own value, so a select that reads a
branch from another lane -- a layout error -- gives a wrong answer too; with
every lane equal it would not. Every variant compares exactly
(``rtol``/``atol`` 0): a select copies its branch values without arithmetic, so
any difference at all is a wrong answer.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import operator

import numpy as np
from dataclasses import dataclass

import conftest
from . import kernel
from utils import sticksize, DTYPE_MAP


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input makers
# ---------------------------------------------------------------------------

# Where the branch ramps start: p counts up from +100, q down from -100, so the
# two never meet and a wrong branch cannot pass as rounding.
_P_BASE = 100.0
_Q_BASE = -100.0


# The NumPy operator for each ``OP`` name in ``kernel.COMPARISONS``. On floats
# these agree with the ordered predicates the kernel compiles to for every
# non-NaN input, and the inputs below hold no NaN.
_NUMPY_COMPARISONS = {
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "le": operator.le,
    "lt": operator.lt,
}
assert _NUMPY_COMPARISONS.keys() == kernel.COMPARISONS.keys(), (
    "the oracle's comparisons and the kernel's must name the same set")

# ``y - x`` on successive lanes. One of each relation, so every comparison is
# true on some lanes and false on others. The equal lane is what tells ``>``
# from ``>=`` and ``<`` from ``<=``, and without it ``==`` would never be true.
_Y_MINUS_X = (-1.0, 0.0, 1.0)


def _condition_inputs(n, np_dtype):
    """``x``/``y`` for the comparison, cycling y < x, y == x, y > x lane by lane.

    ``x`` is the integer ramp ``-n/2 .. n/2 - 1`` and ``y`` differs from it by
    ``_Y_MINUS_X``, cycled. Every value is a small integer, exact in fp16 (whose
    integers are exact up to 2048) as well as fp32, so an equal lane is equal
    after the conversion to the device dtype, and an unequal lane is a whole unit
    apart -- no near-tie that a reduced-precision comparison could break the
    other way. Deterministic, so a failure reproduces without a seed.

    Cycling the relation per lane, rather than in blocks, makes a comparison
    that read the wrong lane give a wrong answer. The period 3 is coprime with
    both stick widths (64 at fp16, 32 at fp32), so the pattern does not realign
    at a stick boundary and hide a stick-level layout error.
    """
    if n > 2 * 2048:
        raise ValueError(
            f"n={n} would take the ramp past +/-2048, where fp16 stops "
            f"representing every integer and the equal lanes stop being exact")
    x = np.arange(n, dtype=np.float64) - n // 2
    y = x + np.resize(np.asarray(_Y_MINUS_X), n)
    return x.astype(np_dtype), y.astype(np_dtype)


def _branch_inputs(n, np_dtype) -> dict:
    """``p``/``q`` as ramps, one distinct value per lane.

    ``p[i] = 100 + i`` and ``q[i] = -100 - i``: every value appears in one lane
    of one buffer, so reading either branch from the wrong lane changes the
    output. All are integers, exact in fp16 up to magnitude 2048.
    """
    largest = abs(_P_BASE) + (n - 1)
    if largest > 2048:
        raise ValueError(
            f"n={n} would take the branch ramps to +/-{largest:g}, past 2048, "
            f"where fp16 stops representing every integer")
    lane = np.arange(n, dtype=np.float64)
    return {"p_ptr": (_P_BASE + lane).astype(np_dtype),
            "q_ptr": (_Q_BASE - lane).astype(np_dtype)}


def make_compare_inputs(n_elements, DTYPE="fp16", **_unused) -> dict:
    np_dtype = DTYPE_MAP[DTYPE]
    x, y = _condition_inputs(n_elements, np_dtype)
    return {"x_ptr": x, "y_ptr": y,
            "mask_ptr": np.zeros(n_elements, dtype=np_dtype)}


def make_select_inputs(n_elements, DTYPE="fp16", **_unused) -> dict:
    """The mask is written by the host as 1.0 or 0.0, isolating the select.

    It alternates lane by lane rather than running in blocks, so a select that
    read the wrong lane -- a layout error rather than a logic one -- shows.
    """
    np_dtype = DTYPE_MAP[DTYPE]
    mask = np.zeros(n_elements, dtype=np_dtype)
    mask[::2] = 1.0
    return {"mask_ptr": mask, **_branch_inputs(n_elements, np_dtype),
            "output_ptr": np.zeros(n_elements, dtype=np_dtype)}


def make_where_inputs(n_elements, DTYPE="fp16", **_unused) -> dict:
    """``mask_ptr`` is scratch the kernel spills through; only ``output_ptr``
    is checked."""
    np_dtype = DTYPE_MAP[DTYPE]
    x, y = _condition_inputs(n_elements, np_dtype)
    return {"x_ptr": x, "y_ptr": y, **_branch_inputs(n_elements, np_dtype),
            "mask_ptr": np.zeros(n_elements, dtype=np_dtype),
            "output_ptr": np.zeros(n_elements, dtype=np_dtype)}


# The scalar ``spilled_scalar`` compares against. It is non-integral so that it
# exercises the path a general scalar takes, not only one of the four integers
# 0..3, which dcc can encode as instruction constants. It is exact in fp16, in
# DF16 and in fp32, so the equal lanes stay equal.
_SCALAR = 1.5


def make_where_scalar_inputs(ROWS, STICK, DTYPE="fp16", **_unused) -> dict:
    """Inputs for ``spilled_scalar``: ``[ROWS, STICK]`` tiles, one scalar stick.

    ``x`` is ``_SCALAR`` minus ``_Y_MINUS_X``, cycled lane by lane, so the
    scalar is below, equal to and above ``x`` in turn. That is the same per-lane
    relation ``_condition_inputs`` gives ``y`` and ``x``. Every value is a
    multiple of 0.5 in ``[0.5, 2.5]``, exact at both dtypes. ``scalar_ptr`` is
    one stick, ``[1, STICK]``, with ``_SCALAR`` in every lane, because the
    kernel reads every lane of it.
    """
    np_dtype = DTYPE_MAP[DTYPE]
    n = ROWS * STICK
    x = _SCALAR - np.resize(np.asarray(_Y_MINUS_X), n)
    branches = {name: array.reshape(ROWS, STICK)
                for name, array in _branch_inputs(n, np_dtype).items()}
    return {"x_ptr": x.astype(np_dtype).reshape(ROWS, STICK),
            "scalar_ptr": np.full((1, STICK), _SCALAR, dtype=np_dtype),
            **branches,
            "mask_ptr": np.zeros((ROWS, STICK), dtype=np_dtype),
            "output_ptr": np.zeros((ROWS, STICK), dtype=np_dtype)}


def make_where_scalar_oracle(OP):
    """-> the ``spilled_scalar`` oracle for comparison *OP*.

    ``scalar_ptr`` is ``[1, STICK]`` and broadcasts over the rows, as
    ``tl.broadcast_to`` does in the kernel. Like ``make_where_oracle``, it
    recomputes the comparison rather than reading ``mask_ptr``.
    """
    compare = _NUMPY_COMPARISONS[OP]

    def run_where_scalar(inputs) -> np.ndarray:
        return np.where(compare(inputs["x_ptr"], inputs["scalar_ptr"]),
                        inputs["p_ptr"], inputs["q_ptr"])
    return run_where_scalar


def make_compare_oracle(OP):
    """-> the ``compare`` oracle for comparison *OP*: its mask, 1.0 where true
    and 0.0 where false."""
    compare = _NUMPY_COMPARISONS[OP]

    def run_compare(inputs) -> np.ndarray:
        return compare(inputs["x_ptr"], inputs["y_ptr"]).astype(
            inputs["x_ptr"].dtype)
    return run_compare


def run_select(inputs) -> np.ndarray:
    return np.where(inputs["mask_ptr"] != 0, inputs["p_ptr"], inputs["q_ptr"])


def make_where_oracle(OP):
    """-> the ``spilled`` oracle for comparison *OP*.

    Recomputes the comparison rather than reading ``mask_ptr``, so a wrong mask
    on the device makes this disagree instead of being inherited.
    """
    compare = _NUMPY_COMPARISONS[OP]

    def run_where(inputs) -> np.ndarray:
        return np.where(compare(inputs["x_ptr"], inputs["y_ptr"]),
                        inputs["p_ptr"], inputs["q_ptr"])
    return run_where


# ---------------------------------------------------------------------------
# Factory -- Where(VariantFactory)
#
# The SIGNATURE varies per combination: every pointer takes the swept DTYPE.
# So does the oracle, for a variant that sweeps OP. The input makers read DTYPE
# from ``params`` themselves.
# ---------------------------------------------------------------------------

_PTR = {"fp16": "*fp16", "fp32": "*fp32"}

_SHAPE_ARGS = {
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
    "LAYOUT":     "constexpr",
}


@dataclass(frozen=True)
class Where(conftest.VariantFactory):
    """Factory for a where variant with no comparison of its own (``select``).

    ``ptrs`` names the kernel's pointer arguments in declaration order. The
    variant sets its ``reference`` as a literal field.
    """
    ptrs: tuple

    def signature(self, DTYPE, **_):
        return {**{n: _PTR[DTYPE] for n in self.ptrs}, **_SHAPE_ARGS}


@dataclass(frozen=True)
class WhereComparing(Where):
    """Factory for a where variant whose kernel compares, and so sweeps ``OP``.

    ``oracle`` builds the reference for one ``OP``: ``make_compare_oracle`` or
    ``make_where_oracle``. The variant must not also set ``reference``;
    ``conftest._apply_factory`` refuses a field set both ways.
    """
    oracle: object = None

    def signature(self, DTYPE, OP, **_):
        return {**super().signature(DTYPE), "OP": "constexpr"}

    def reference(self, OP, **_):
        return self.oracle(OP)


@dataclass(frozen=True)
class WhereScalarStick(WhereComparing):
    """Factory for ``spilled_scalar``, whose kernel is 2-D.

    Its shape arguments are ``ROWS``/``STICK`` rather than the 1-D variants'
    ``n_elements``/``BLOCK_SIZE``.
    """

    def signature(self, DTYPE, OP, **_):
        return {**{n: _PTR[DTYPE] for n in self.ptrs},
                "ROWS": "i32", "STICK": "i32",
                "LAYOUT": "constexpr", "OP": "constexpr"}


# ---------------------------------------------------------------------------
# SIGNATURE -- dtype per @triton.jit arg. The module-level one is the default
# the registry needs; every variant's factory overrides it per DTYPE.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "x_ptr":      "*fp16",
    "y_ptr":      "*fp16",
    "mask_ptr":   "*fp16",
    **_SHAPE_ARGS,
    "OP":         "constexpr",
}


def _stick_1d(dtype: str) -> tuple:
    """Labelled 1D stick layout at *dtype*, ``[n]`` -> ``[ceil(n/S), S]``."""
    stick = sticksize({"p": f"*{dtype}"}, "p")
    return ("stick", ((0, "floordiv", stick), (0, "mod", stick)))


def _stick_2d_on_n(dtype: str) -> tuple:
    """Labelled stick-on-N layout at *dtype*, ``[M, N]`` -> ``[ceil(N/S), M, S]``.

    The same layout ``fixtures/elementwise/meta.py`` uses for its 2-D kernels.
    Every ``spilled_scalar`` operand takes it, the ``[1, S]`` scalar stick
    included, so all of them physicalize the same way.
    """
    stick = sticksize({"p": f"*{dtype}"}, "p")
    return ("stick", ((1, "floordiv", stick), 0, (1, "mod", stick)))


def _scalar_stick_row(dtype: str) -> tuple:
    """One ``("DTYPE", "LAYOUT", "STICK", "ROWS")`` row for ``spilled_scalar``.

    ``STICK`` and the layout both take their width from *dtype*, so a row cannot
    pair one dtype's stick with another's layout. ``ROWS`` keeps the tile at the
    other variants' 128 elements. 128 is a whole number of sticks at both dtypes:
    2 rows of 64 at fp16, 4 rows of 32 at fp32.
    """
    stick = sticksize({"p": f"*{dtype}"}, "p")
    if _N_ELEMENTS % stick:
        raise ValueError(
            f"{_N_ELEMENTS} elements is not a whole number of {stick}-lane "
            f"sticks at {dtype}")
    return (dtype, _stick_2d_on_n(dtype), stick, _N_ELEMENTS // stick)


# ---------------------------------------------------------------------------
# VARIANTS
#
# All variants are loop-free and one tile total, which the device tier requires:
# dbo-opt rejects the scf.for a program-id distribution loop outlines.
#
# n_elements=128 is two sticks at fp16 (64 per stick) and four at fp32 (32).
# ---------------------------------------------------------------------------

# Elements per tile, for every variant.
_N_ELEMENTS = 128

_TAGS = [
    "descriptor-load-static", "descriptor-store-static",
    "simplified:no-loop", "spyre-tensor-layout", "where",
]

_PARAMS = {
    # One row per dtype, because the layout's stick width follows from it.
    ("DTYPE", "LAYOUT"): [
        ("fp16", _stick_1d("fp16")),
        ("fp32", _stick_1d("fp32")),
    ],
    "n_elements": [_N_ELEMENTS],
    "BLOCK_SIZE": [_N_ELEMENTS],
}

# Every comparison the kernels accept, for the variants that compare.
_OPS = list(kernel.COMPARISONS)

_COMPARE_PARAMS = {**_PARAMS, "OP": _OPS}

VARIANTS = {
    # -----------------------------------------------------------------------
    # The comparison alone, at the compute tier for both dtypes. Its fp32 row
    # also runs on the device, as ``compare_device``; the fp16 row cannot.
    # The fp16 kernel compiles and launches, and its mask is correct as a mask
    # -- every lane where the predicate holds is non-zero, which is all
    # spyreop.select asks of a condition. What is wrong is the mask's value:
    # at fp16 the device stores 0.5 where the oracle expects 1.0, a DF16
    # encoding issue; fp32 is unaffected. ``spilled``
    # covers the fp16 comparison on the device, since a select only asks that
    # the mask be non-zero.
    # -----------------------------------------------------------------------
    "compare": {
        "base": None,
        "tags": _TAGS,
        "summary": (
            "1D (x OP y) over a single stick-tiled tile, no distribution loop. "
            "Sweeps fp16/fp32 and every OP."
        ),
        "kernel_fn":    kernel.compare_1d_device,
        "factory":      WhereComparing(ptrs=("x_ptr", "y_ptr", "mask_ptr"),
                                       oracle=make_compare_oracle),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT", "OP"],
        "params":       _COMPARE_PARAMS,
        "grid":         [1],
        "inputs":       make_compare_inputs,
        "output_key":   "mask_ptr",
        "rtol":         0.0,
        "atol":         0.0,
    },

    # -----------------------------------------------------------------------
    # ``compare``'s fp32 row on the device. At fp32 the mask comes back as the
    # 1.0 (true) and 0.0 (false) the oracle expects.
    # -----------------------------------------------------------------------
    "compare_device": {
        "base": "compare",
        "summary": (
            "1D fp32 (x OP y) over a single stick-tiled tile on the device, "
            "no distribution loop."
        ),
        "params": {
            # The parent's group with its fp32 row alone. Redeclared in full
            # because ``params`` merges wholesale.
            ("DTYPE", "LAYOUT"): [
                ("fp32", _stick_1d("fp32")),
            ],
            "n_elements": [128],
            "BLOCK_SIZE": [128],
            "OP": _OPS,
        },
        "compiles_to_binary": True,
    },

    # -----------------------------------------------------------------------
    # The select alone, over a host-written mask: no comparison and no mask
    # round trip precede it, so a failure here is spyreop.select's.
    # -----------------------------------------------------------------------
    "select": {
        "base": None,
        "tags": _TAGS,
        "summary": (
            "1D where(mask != 0, p, q) over a host-written mask, single tile, "
            "no distribution loop. Sweeps fp16/fp32."
        ),
        "kernel_fn":    kernel.select_1d_device,
        "factory":      Where(ptrs=("mask_ptr", "p_ptr", "q_ptr", "output_ptr")),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT"],
        "params":       _PARAMS,
        "grid":         [1],
        "compiles_to_binary": True,
        "inputs":       make_select_inputs,
        "reference":    run_select,
        "output_key":   "output_ptr",
        "rtol":         0.0,
        "atol":         0.0,
    },

    # -----------------------------------------------------------------------
    # Comparison and select in one kernel, the mask spilled between them
    # (kernel.py's module docstring says why it cannot stay in registers).
    # -----------------------------------------------------------------------
    "spilled": {
        "base": None,
        "tags": _TAGS,
        "summary": (
            "1D where(x OP y, p, q) in one kernel, the mask spilled to its own "
            "buffer between compare and select. Sweeps fp16/fp32 and every OP."
        ),
        "kernel_fn":    kernel.where_1d_device,
        "factory":      WhereComparing(ptrs=("x_ptr", "y_ptr", "p_ptr", "q_ptr",
                                             "mask_ptr", "output_ptr"),
                                       oracle=make_where_oracle),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT", "OP"],
        "params":       _COMPARE_PARAMS,
        "grid":         [1],
        "compiles_to_binary": True,
        "inputs":       make_where_inputs,
        "output_key":   "output_ptr",
        "rtol":         0.0,
        "atol":         0.0,
    },
    # -----------------------------------------------------------------------
    # ``spilled`` with ``y`` a scalar, read from a one-stick buffer the host
    # filled and broadcast over the tile's rows. Written with ``tl.full`` the
    # scalar is folded into the compare's body as a captured constant, and
    # dbo-opt's vector compare refuses it (operand #1 "must be vector ... but
    # got 'f16'"). kernel.where_scalar_stick_device's docstring has the
    # mechanism.
    # -----------------------------------------------------------------------
    "spilled_scalar": {
        "base": None,
        "tags": _TAGS,
        "summary": (
            "2D where(x OP scalar, p, q), the scalar a host-filled stick "
            "broadcast over the rows, mask spilled; single tile, no "
            "distribution loop. Sweeps fp16/fp32 and every OP."
        ),
        "kernel_fn":    kernel.where_scalar_stick_device,
        "factory":      WhereScalarStick(
                            ptrs=("x_ptr", "scalar_ptr", "p_ptr", "q_ptr",
                                  "mask_ptr", "output_ptr"),
                            oracle=make_where_scalar_oracle),
        "constexpr":    ["ROWS", "STICK", "LAYOUT", "OP"],
        "params": {
            ("DTYPE", "LAYOUT", "STICK", "ROWS"): [
                _scalar_stick_row("fp16"),
                _scalar_stick_row("fp32"),
            ],
            "OP": _OPS,
        },
        "grid":         [1],
        "compiles_to_binary": True,
        "inputs":       make_where_scalar_inputs,
        "output_key":   "output_ptr",
        "rtol":         0.0,
        "atol":         0.0,
    },
}
