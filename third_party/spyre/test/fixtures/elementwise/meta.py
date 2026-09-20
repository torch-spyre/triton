"""SIGNATURE + VARIANTS + reference oracle + input generators for elementwise.

Variants cover 1D/2D/3D shapes (Level A, OP pinned to "add") and
op × dtype correctness on ktir_cpu (Level B, 1d_compute sweeps 3×4=12 combos).

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import functools
import operator

import numpy as np
from dataclasses import dataclass

import conftest
from . import kernel
from utils import sticksize, DTYPE_MAP


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input makers
#
# One generator. The named makers below only give it a shape: the framework
# calls `inputs` with the variant's whole `params` dict, and variants spell
# their shape with different argument names.
# ---------------------------------------------------------------------------

def _make_inputs(shape, *, dtype="fp32", nonzero_y=False) -> dict:
    """``x`` / ``y`` / zeroed ``output`` buffers of *shape*. Never random.

    Floats take ``x = sin`` and ``y = cos`` over the flattened index, keeping
    both inside [-1, 1] so fp16 stays well conditioned. Integers take a ramp
    against a constant, small enough that ``mul`` stays clear of the int32
    boundary.

    ``nonzero_y`` lifts ``y`` off zero, which ``div`` needs -- ``cos`` crosses
    zero and the quotient there is meaningless. It is a property of the buffers
    rather than of one op, because a variant sweeping ``OP`` shares one input
    maker across all of its ops.

    *dtype* is a :data:`DTYPE_MAP` key. It was previously spelled three ways in
    this file -- ``"f32"``, ``"fp32"`` and ``np.float32`` -- each with its own
    lookup, so passing one maker's spelling to another raised.
    """
    np_dtype = DTYPE_MAP[dtype]
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    total = int(np.prod(shape))
    if np.issubdtype(np_dtype, np.integer):
        x = np.arange(1, total + 1, dtype=np_dtype)
        y = np.full(total, 3, dtype=np_dtype)
    else:
        t = np.arange(total, dtype=np.float32)
        x = np.sin(t * 2.0 * np.pi / total).astype(np_dtype)
        y = (np.cos(t * 2.0 * np.pi / total)
             + (2.0 if nonzero_y else 0.0)).astype(np_dtype)
    return {"x_ptr": x.reshape(shape), "y_ptr": y.reshape(shape),
            "output_ptr": np.zeros(shape, dtype=np_dtype)}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``x + y``. Works for any shape."""
    return inputs["x_ptr"] + inputs["y_ptr"]


# Each maker takes the shape argument names its variants use and ignores the
# rest of ``params``. ``DTYPE`` is read from ``params`` rather than defaulted
# separately, which is what stops the buffers' dtype drifting from the pointer
# types ``SIGNATURE`` declares.

def make_inputs(n_elements, DTYPE="fp32", nonzero_y=False, **_unused) -> dict:
    return _make_inputs(n_elements, dtype=DTYPE, nonzero_y=nonzero_y)


def make_inputs_2d(M, N, DTYPE="fp32", nonzero_y=False, **_unused) -> dict:
    return _make_inputs((M, N), dtype=DTYPE, nonzero_y=nonzero_y)


def make_inputs_3d(M, N, P, DTYPE="fp32", **_unused) -> dict:
    return _make_inputs((M, N, P), dtype=DTYPE)


def make_inputs_scalar_dim(n_elements=4096, DTYPE="fp32", **_unused) -> dict:
    """1D, plus the length as a rank-1 buffer.

    ``n_elements`` is a default here and not a ``params`` entry: the kernel
    reads it from ``seqlen_ptr`` instead of taking it as an argument, so a
    ``params`` entry would leak into ``run_cpu``'s kwargs and fail its
    unknown-kwarg check.
    """
    inputs = _make_inputs(n_elements, dtype=DTYPE)
    inputs["seqlen_ptr"] = np.array([n_elements], dtype=np.int32)
    return inputs


def make_inputs_2d_scalar_dim(N, M=32, DTYPE="fp32", **_unused) -> dict:
    """2D, plus ``M`` as a rank-1 buffer -- see :func:`make_inputs_scalar_dim`."""
    inputs = _make_inputs((M, N), dtype=DTYPE)
    inputs["seqlen_ptr"] = np.array([M], dtype=np.int32)
    return inputs


# The declared-buffer kernels take one input pointer, not two, so they get their own
# maker and their own oracles rather than a mode of the pair above. A ramp over
# [0.1, 1.1): away from 0, where sqrt's relative error grows fastest, and small
# enough that exp applied twice stays well inside fp32.

def make_inputs_unary(n_elements, DTYPE="fp32", scratch=(), **_unused) -> dict:
    """``x`` plus a zeroed ``output``, and one zeroed buffer per *scratch* name.

    *scratch* is what the declared-buffer variants need: each intermediate is a
    kernel pointer argument there rather than a value, so it needs a real
    allocation. Zeroed, and the same length and dtype as ``x`` -- the kernel writes
    every element of each before reading it, so the initial contents only matter in
    that a nonzero one could mask a store that never happened.
    """
    np_dtype = DTYPE_MAP[DTYPE]
    total = int(n_elements)
    x = (np.arange(total, dtype=np.float32) / total + 0.1).astype(np_dtype)
    inputs = {"x_ptr": x, "output_ptr": np.zeros(total, dtype=np_dtype)}
    for name in scratch:
        inputs[name] = np.zeros(total, dtype=np_dtype)
    return inputs


# Named here rather than in the variant so the scratch names sit next to the maker
# that has to honour them, and so a variant's ``inputs`` field stays a plain callable
# the framework calls with ``**param_values``.
make_inputs_dag_buffers = functools.partial(
    make_inputs_unary, scratch=("e_ptr", "s_ptr", "m_ptr", "r_ptr"))


def make_inputs_pooled(n_elements, DTYPE="fp32", regions=1, **_unused) -> dict:
    """``x`` plus a zeroed ``output``, and one zeroed scratch *pool*.

    The pooled kernels take a single ``pool_ptr`` and address *regions* blocks of
    ``n_elements`` within it, so the allocation here is ``regions * n_elements``
    long -- one array where the per-intermediate variant takes *regions* of them.
    That difference in the allocation is the footprint the pooled arm exists to
    show; the region count itself is the kernel's liveness claim, and it is spelled
    in the kernel's docstring.
    """
    np_dtype = DTYPE_MAP[DTYPE]
    inputs = make_inputs_unary(n_elements, DTYPE=DTYPE)
    total = int(n_elements)
    return {"x_ptr": inputs["x_ptr"],
            "pool_ptr": np.zeros(int(regions) * total, dtype=np_dtype),
            "output_ptr": inputs["output_ptr"]}


# One partial per pooled variant, naming that kernel's region count rather than its
# pointers -- the pool is one pointer whatever the count.
make_inputs_pooled1 = functools.partial(make_inputs_pooled, regions=1)
make_inputs_pooled2 = functools.partial(make_inputs_pooled, regions=2)
make_inputs_pooled3 = functools.partial(make_inputs_pooled, regions=3)


def run_chain(inputs: dict) -> np.ndarray:
    return np.sqrt(np.exp(inputs["x_ptr"]))


def run_chain3(inputs: dict) -> np.ndarray:
    return np.exp(np.sqrt(np.exp(inputs["x_ptr"])))


def run_dag(inputs: dict) -> np.ndarray:
    x = inputs["x_ptr"]
    e = np.exp(x)
    return e * np.sqrt(e) + np.sqrt(x)


# ---------------------------------------------------------------------------
# Level B factory — Elementwise(VariantFactory)
# ---------------------------------------------------------------------------

# Shape-arg dicts for each rank's runtime signature (no pointer types — those
# come from DTYPE). These are the non-constexpr, non-pointer runtime args.
_SHAPE_ARGS = {
    1: {"n_elements": "i32"},      # elementwise_1d
    2: {},                         # 2d_device: M/N are constexprs, no runtime shape args
}

# NumPy oracles indexed by op name.
_NUMPY_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}

# Pointer type strings for each DTYPE.
_PTR = {"fp16": "*fp16", "fp32": "*fp32", "i32": "*i32"}


@dataclass(frozen=True)
class Elementwise(conftest.VariantFactory):
    """Factory for elementwise variants that sweep OP and/or DTYPE.

    ``rank`` selects the shape: which ``_SHAPE_ARGS`` entry joins the runtime
    signature, and which input maker is used. Nothing here has anything to say
    about the layout a variant sticks -- that follows from ``DTYPE``, and a
    ``params`` group states it beside the dtype without a factory in sight.
    """
    rank: int = 1

    def signature(self, DTYPE, **_):
        ptrs = {n: _PTR[DTYPE] for n in ("x_ptr", "y_ptr", "output_ptr")}
        return {**ptrs, **_SHAPE_ARGS[self.rank]}

    def reference(self, OP, **_):
        def oracle(inputs):
            x, y = inputs["x_ptr"], inputs["y_ptr"]
            if OP == "div" and x.dtype == np.int32:
                return (x.astype(np.float32) / y.astype(np.float32)).astype(np.int32)
            return _NUMPY_OPS[OP](x, y).astype(x.dtype)
        return oracle

    def inputs(self, **_):
        """Shape from ``rank``; ``DTYPE`` the maker reads from ``params`` itself.

        ``nonzero_y`` is unconditional because these variants sweep ``OP``, and
        one of the ops they sweep is ``div``.
        """
        return functools.partial({1: make_inputs, 2: make_inputs_2d}[self.rank],
                                 nonzero_y=True)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg. Purely types; values live in the
# variant's ``params`` dict and ``constexpr`` list selects which of them
# get baked into TTIR.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "x_ptr":      "*fp32",
    "y_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
}


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

_SIG_2D = {
    "x_ptr":      "*fp32",
    "y_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "M":          "i32",
    "N":          "i32",
    "BLOCK_M":    "i32",
    "BLOCK_N":    "i32",
}

# elementwise_2d carries the optional layout constexprs; elementwise_2d_grid
# (which shares _SIG_2D) does not.
_SIG_2D_LAYOUT = {
    **_SIG_2D,
    "X_LAYOUT":   "constexpr",
    "Y_LAYOUT":   "constexpr",
    "OUT_LAYOUT": "constexpr",
}

_SIG_TENSORS_FP16 = {
    "x_ptr":      "*fp16",
    "y_ptr":      "*fp16",
    "output_ptr": "*fp16",
}

# The layout signature in fp16 -- the union of the two above, which is all it has
# ever been. Spelled as a merge so a change to the shape arguments or the layout
# constexprs reaches it, instead of leaving it silently behind. The pointer
# overrides come last but the key order is _SIG_2D_LAYOUT's, since re-assigning an
# existing key keeps its position, and for a signature that order is the argument
# order.
_SIG_2D_SPYRE = {**_SIG_2D_LAYOUT, **_SIG_TENSORS_FP16}


def _stick_of(dtype: str) -> int:
    """Lanes to a stick at *dtype*: 64 at fp16, 32 at fp32 and i32."""
    return sticksize({"p": f"*{dtype}"}, "p")


# Both layout helpers take a *dtype*, not a width, because every call site had one
# and passed _stick_of(dtype) -- which put the chance of pairing a dtype with the
# wrong width at each of them. And both return the ``("stick", layout)`` pair rather
# than the bare layout, so a layout has one spelling everywhere in this file and the
# only tuples written into ``params`` are rows. Handing the bare tuple over meant
# wrapping it in ``list()`` at every unlabelled site: _normalise_param_list reads any
# tuple in a values list as a (label, value) pair, and a 3-element layout is not one,
# so it raised rather than mislabelled. The value inside the pair stays a tuple
# because it reaches Triton as a constexpr and has to be hashable.

def _stick_1d(dtype: str) -> tuple:
    """Labelled 1D stick layout at *dtype*, ``[n]`` -> ``[ceil(n/S), S]``.

    Physical dim 0 is which stick, dim 1 the lane within it.
    """
    stick = _stick_of(dtype)
    return ("stick", ((0, "floordiv", stick), (0, "mod", stick)))


def _stick_2d_on_n(dtype: str) -> tuple:
    """Labelled stick-on-N layout at *dtype*, ``[M, N]`` -> ``[ceil(N/S), M, S]``.

    Every operand physicalizes identically under this, so an elementwise op stays
    elementwise on the rank-3 tiles and no transpose is synthesized.
    """
    stick = _stick_of(dtype)
    return ("stick", ((1, "floordiv", stick), 0, (1, "mod", stick)))


def _stick_on_n_row(dtype: str, n_sticks: int) -> tuple:
    """One row for the ``("DTYPE", "N", "BLOCK_N", *_LAYOUT)`` group: a rank-2
    shape *n_sticks* sticks wide at *dtype*, with the layout that matches it.

    ``N``, ``BLOCK_N`` and the layout all take their width from the one *dtype*
    argument, so a row cannot pair a 128-lane ``N`` with a 32-lane layout -- the
    mismatch dbo-opt rejects with an error naming neither operand.
    Spelling the layout beside its dtype is what let the two disagree; a row makes
    it unrepresentable, with no hook to gate the fixtures that get the property.

    No width is written here or at the call site: *n_sticks* says how many, the
    dtype says how wide. Named for the row it makes rather than for a size.
    (This used to point at ``_SS = functools.partial(sticksize, _SIG_SPYRE)`` in
    ``fixtures/reduce/meta.py`` as the counter-example; that fixture now derives
    its stick widths from a dtype the same way and the partial is gone.)
    """
    n = n_sticks * _stick_of(dtype)
    layout = _stick_2d_on_n(dtype)
    return (dtype, n, n, layout, layout, layout)

_SIG_3D = {
    "x_ptr":      "*fp32",
    "y_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "M":          "i32",
    "N":          "i32",
    "P":          "i32",
    "BLOCK_M":    "i32",
    "BLOCK_N":    "i32",
    "BLOCK_P":    "i32",
}

_SIG_1D_SCALAR = {
    "x_ptr":      "*fp32",
    "y_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "seqlen_ptr": "*i32",
    "BLOCK_SIZE": "i32",
}

_SIG_2D_SCALAR = {
    "x_ptr":      "*fp32",
    "y_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "seqlen_ptr": "*i32",
    "N":          "i32",
    "BLOCK_M":    "i32",
    "BLOCK_N":    "i32",
}

# The declared-buffer kernels are unary -- one input pointer, no ``OP`` -- plus one
# extra pointer per scratch buffer they declare. ``_SIG_1D_UNARY`` is that shape
# with no scratch at all, and the helper inserts the scratch pointers into its key
# order rather than respelling the whole dict, so a change to the shape arguments or
# to ``LAYOUT`` reaches every one of them. The key order is the argument order, and
# the scratch pointers go after ``x_ptr`` and before ``output_ptr``, the way the
# kernels declare them.
_SIG_1D_UNARY = {
    "x_ptr":      "*fp32",
    "output_ptr": "*fp32",
    "n_elements": "i32",
    "BLOCK_SIZE": "i32",
    "LAYOUT":     "constexpr",
}


def _sig_1d_buffers(*names: str) -> dict:
    return {"x_ptr": "*fp32",
            **{name: "*fp32" for name in names},
            **{k: v for k, v in _SIG_1D_UNARY.items() if k != "x_ptr"}}


_SIG_1D_DAG_BUFFERS = _sig_1d_buffers("e_ptr", "s_ptr", "m_ptr", "r_ptr")

# The pooled arm takes exactly one extra pointer whatever its region count, so one
# signature serves all of them -- which is the point of the arm.
_SIG_1D_POOLED = _sig_1d_buffers("pool_ptr")


VARIANTS = {
    # -----------------------------------------------------------------------
    # Level A -- shape
    #
    # OP and DTYPE are pinned to add/fp32 throughout: these variants are about
    # descriptor shape, and sweeping the other two axes would multiply keys
    # without covering anything shape-related.
    # -----------------------------------------------------------------------

    # 1D
    "default": {
        # Static-shape flavor (PR #82): n_elements is a constexpr baked
        # into TTIR as the literal 2097152.
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "num-programs-fold"],
        "summary": (
            "1D elementwise add `C = A + B` over a fully-static vector, "
            "partitioned across the 32-core grid."
        ),
        "doc": (
            "Takes two 1D input vectors `A` and `B` of length "
            "`n_elements` and writes `C = A + B` to an output vector of "
            "the same length. Each of the 32 cores runs one program "
            "that carves out its share of the vector (a contiguous run "
            "of `BLOCK_SIZE`-wide tiles) and streams through it in a "
            "single pass.\n\n"
            "`n_elements` is baked in at compile time, so the tensor "
            "descriptors carry a fully-static shape "
            "(`memref<2097152xf32>`). This is the simplest kernel in "
            "the set — one axis, no inner reduction, no cross-core "
            "communication."
        ),
        "kernel_fn":    kernel.elementwise_1d,
        "constexpr":    ["n_elements", "BLOCK_SIZE", "OP"],
        "params":       {
            # n_elements=[1024,2097152,2097153]: absorbs single_block (1024)
            # and nonaligned (2097153).
            "n_elements": [1024, 2097152, 2097153], "BLOCK_SIZE": [1024],
            "DTYPE": ["fp32"], "OP": ["add"],
        },
        # 1D kernel (only tl.program_id(0)) on the 32-core Spyre grid.
        "grid":         [32],
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "output_ptr",
    },
    "dynamic": {
        # PR #86: flip n_elements from constexpr to runtime i32. Produces
        # memref<?xf32> in KTIR. Inherits ``params`` and everything else
        # from default; only ``constexpr`` and the structural check change.
        "tags": ["descriptor-load-dynamic", "descriptor-store-dynamic", "program-id-1d", "num-programs-fold"],
        "summary": (
            "1D elementwise add where the vector length is a runtime "
            "argument, not a compile-time constant."
        ),
        "doc": (
            "Same computation as the static 1D add, but `n_elements` "
            "arrives as a runtime `i32` argument instead of being "
            "baked in at compile time. The resulting KTIR descriptors "
            "carry a dynamic extent (`memref<?xf32>`), so the same "
            "compiled kernel can run on any input length — each core "
            "computes its own per-tile work based on the runtime "
            "value."
        ),
        "constexpr":    ["BLOCK_SIZE", "OP"],
    },
    "dynamic_small": {
        # Different shape: verifies the compiled dynamic kernel runs at a
        # smaller n_elements than the static default.
        "base":   "dynamic",
        "params": {"n_elements": [4096], "BLOCK_SIZE": [1024], "DTYPE": ["fp32"], "OP": ["add"]},
    },
    "dynamic_from_scalar_load": {
        "base":         "dynamic",
        "kernel_fn":    kernel.elementwise_1d_scalar_dim,
        "SIGNATURE":    _SIG_1D_SCALAR,
        "constexpr":    ["BLOCK_SIZE", "OP"],
        "params":       {"BLOCK_SIZE": [1024], "OP": ["add"]},
        "inputs":       make_inputs_scalar_dim,
        "tags": [
            "descriptor-load-dynamic-from-scalar-load",
            "descriptor-store-dynamic", "program-id-1d", "num-programs-fold",
        ],
        "summary": (
            "1D elementwise add where `n_elements` is read from memory "
            "via a scalar `tl.load`, then used as a tensor descriptor's "
            "dynamic shape."
        ),
        "doc": (
            "not yet wired into any dataflow-scheduler/DFIR "
            "flow (`kDynamic` has no `AddressAssignment` / "
            "`NormalizeGridTo1D` path yet). `n_elements` is not a kernel "
            "argument here; it is read from `seqlen_ptr` with a scalar "
            "`tl.load`. This is the 1D counterpart of "
            "`2d_dynamic_from_scalar_load` — the simplest form of the "
            "single-element 1-D scalar-read chain (`construct_memory_view` "
            "/ `construct_access_tile` / `ktdp.load` / `tensor.extract`) "
            "feeding the dynamic-shape path via an `arith.index_cast` "
            "bridge, producing `memref<?xf32>`. Reuses the same oracle "
            "as `dynamic` (via `make_inputs_scalar_dim`)."
        ),
    },

    # 2D
    "2d": {
        # M=[512,520]: absorbs 2d_nonaligned (M=520, m_blocks=33 → clamp fires).
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "num-programs-fold"],
        "summary": (
            "2D elementwise add over an `M × N` matrix, tiled in both "
            "axes across the 32-core grid."
        ),
        "doc": (
            "Takes two `M × N` matrices `A` and `B` and writes "
            "`C = A + B`. Each core sweeps a strip of row-tiles of "
            "height `BLOCK_M` and walks across the full row of "
            "`BLOCK_N`-wide column tiles. Tile counts are computed with "
            "`cdiv` so a trailing partial tile works correctly, and the "
            "inner loops clamp their bounds against `M` and `N`.\n\n"
            "`M` and `N` are compile-time constants here, so the "
            "descriptor shape is fully static (`memref<512x32xf32>`)."
        ),
        "kernel_fn":    kernel.elementwise_2d,
        "SIGNATURE":    _SIG_2D_LAYOUT,
        "constexpr":    ["M", "N", "BLOCK_M", "BLOCK_N",
                         "X_LAYOUT", "Y_LAYOUT", "OUT_LAYOUT", "OP"],
        "params":       {
            # M=[512,520]: absorbs 2d_nonaligned (M=520).
            "M": [512, 520], "N": [32], "BLOCK_M": [16], "BLOCK_N": [16],
            "X_LAYOUT": [None], "Y_LAYOUT": [None], "OUT_LAYOUT": [None],
            "DTYPE": ["fp32"], "OP": ["add"],
        },
        "inputs":       make_inputs_2d,
    },
    "2d_dynamic": {
        "base":      "2d",
        "tags":      ["descriptor-load-dynamic", "descriptor-store-dynamic", "program-id-1d", "num-programs-fold"],
        "summary": (
            "2D elementwise add where both `M` and `N` are runtime "
            "arguments."
        ),
        "doc": (
            "Same tiling structure as the static 2D add, but `M` and "
            "`N` arrive as runtime `i32` arguments. The descriptor "
            "lowers to `memref<?x?xf32>`, so the compiled kernel runs "
            "unchanged across a range of matrix shapes."
        ),
        "constexpr":    ["BLOCK_M", "BLOCK_N",
                         "X_LAYOUT", "Y_LAYOUT", "OUT_LAYOUT", "OP"],
    },
    "2d_dynamic_alt": {
        # Different N than the static 2d sibling: confirms the compiled
        # dynamic kernel runs at a shape distinct from its static sibling.
        "base":   "2d_dynamic",
        "params": {
            "M": [256], "N": [64], "BLOCK_M": [16], "BLOCK_N": [16],
            "X_LAYOUT": [None], "Y_LAYOUT": [None], "OUT_LAYOUT": [None],
            "DTYPE": ["fp32"], "OP": ["add"],
        },
    },
    "2d_dynamic_from_scalar_load": {
        "base":         "2d",
        "kernel_fn":    kernel.elementwise_2d_scalar_dim,
        "SIGNATURE":    _SIG_2D_SCALAR,
        "constexpr":    ["N", "BLOCK_M", "BLOCK_N", "OP"],
        "params":       {"N": [32], "BLOCK_M": [16], "BLOCK_N": [16], "OP": ["add"]},
        "inputs":       make_inputs_2d_scalar_dim,
        # 2D grid: [4, 8] = 32 cores. N is chunked across grid_n the same
        # way the runtime M is chunked across grid_m (elementwise_2d_grid's
        # distribution pattern), rather than walking a full row per core.
        "grid":         [4, 8],
        "tags": [
            "descriptor-load-dynamic-from-scalar-load",
            "descriptor-store-dynamic", "program-id-2d", "num-programs-fold",
        ],
        "summary": (
            "2D elementwise add over a 2D grid where `M` is read from "
            "memory via a scalar `tl.load`, then used as a tensor "
            "descriptor's dynamic shape; `N` is chunked the same way."
        ),
        "doc": (
            "not yet wired into any dataflow-scheduler/DFIR "
            "flow (`kDynamic` has no `AddressAssignment` / "
            "`NormalizeGridTo1D` path yet). `M` is not a kernel argument "
            "here; it is read from `seqlen_ptr` with a scalar `tl.load`. "
            "Uses a genuine 2D grid (`pid_m`/`pid_n`, `grid_m`/`grid_n`, "
            "matching `elementwise_2d_grid`'s naming and distribution "
            "pattern) so `N` is chunked across cores the same way the "
            "runtime `M` is, instead of walking a full row per core. "
            "This exercises the single-element 1-D scalar-read chain "
            "(`construct_memory_view` / `construct_access_tile` / "
            "`ktdp.load` / `tensor.extract`) feeding the dynamic-shape "
            "path via an `arith.index_cast` bridge, producing "
            "`memref<?x32xf32>`. Reuses the same `x + y` oracle as `2d` "
            "(via `make_inputs_2d_scalar_dim`)."
        ),
    },

    # 3D
    "3d": {
        # M=[64,65,256]: absorbs 3d_nonaligned (M=65) and 3d_active_cores (M=256).
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "num-programs-fold"],
        "summary": (
            "3D elementwise add over an `M × N × P` tensor with "
            "explicit stride arithmetic."
        ),
        "doc": (
            "Takes two rank-3 tensors of shape `M × N × P` and writes "
            "`C = A + B`. The kernel computes strides explicitly "
            "(`stride_m = N * P`, `stride_n = P`) and sweeps nested "
            "tile loops along all three axes. All three dimensions are "
            "compile-time constants, producing a fully-static "
            "descriptor (`memref<64x32x16xf32>`)."
        ),
        "kernel_fn":    kernel.elementwise_3d,
        "SIGNATURE":    _SIG_3D,
        "constexpr":    ["M", "N", "P", "BLOCK_M", "BLOCK_N", "BLOCK_P", "OP"],
        "params":       {
            # M=[64,65,256]: absorbs 3d_nonaligned (M=65) and 3d_active_cores (M=256).
            "M": [64, 65, 256], "N": [32], "P": [16],
            "BLOCK_M": [8], "BLOCK_N": [8], "BLOCK_P": [8],
            "OP": ["add"],
        },
        "inputs":       make_inputs_3d,
    },
    "3d_dynamic": {
        "base":      "3d",
        "tags":      ["descriptor-load-dynamic", "descriptor-store-dynamic", "program-id-1d", "num-programs-fold"],
        "summary": (
            "3D elementwise add where `M`, `N`, `P` are all runtime "
            "arguments."
        ),
        "doc": (
            "Same tiling structure as the static 3D add, but all three "
            "dimensions arrive as runtime `i32` arguments. The "
            "descriptor lowers to `memref<?x?x?xf32>`."
        ),
        "constexpr":    ["BLOCK_M", "BLOCK_N", "BLOCK_P", "OP"],
    },

    # multi-axis grid
    "2d_grid": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-2d", "num-programs-fold"],
        "summary": (
            "2D grid: pid_0 distributes M-tiles, pid_1 distributes N-tiles, "
            "each with a distribution loop."
        ),
        "doc": (
            "Same elementwise add as `2d`, but uses a 2D program grid. Each "
            "axis distributes its tiles via a loop: `pid_0` covers M, `pid_1` "
            "covers N. The 2D grid replaces the 1D-grid outer loops."
        ),
        "kernel_fn":    kernel.elementwise_2d_grid,
        "SIGNATURE":    _SIG_2D,
        "constexpr":    ["M", "N", "BLOCK_M", "BLOCK_N", "OP"],
        # 2D grid: [4, 8] = 32 cores
        "params":       {"M": [256], "N": [128], "BLOCK_M": [16], "BLOCK_N": [16], "OP": ["add"]},
        "grid":         [4, 8],
        "inputs":       make_inputs_2d,
    },
    "2d_grid_dynamic": {
        "base":      "2d_grid",
        "tags":      ["descriptor-load-dynamic", "descriptor-store-dynamic", "program-id-2d", "num-programs-fold"],
        "summary": (
            "2D grid with runtime `M` and `N`: distribution loop structure, "
            "dynamic descriptor shapes."
        ),
        "doc": (
            "Same as `2d_grid` but `M` and `N` are runtime `i32` arguments. "
            "Descriptors lower to `memref<?x?xf32>`."
        ),
        "constexpr":    ["BLOCK_M", "BLOCK_N", "OP"],
    },
    "3d_grid": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-3d", "num-programs-fold"],
        "summary": (
            "3D grid: pid_0 distributes M-tiles, pid_1 N-tiles, pid_2 P-tiles, "
            "each with a distribution loop."
        ),
        "doc": (
            "Same elementwise add as `3d`, but uses a 3D program grid. Each "
            "axis distributes its tiles via a loop: `pid_0` covers M, `pid_1` "
            "covers N, `pid_2` covers P."
        ),
        "kernel_fn":    kernel.elementwise_3d_grid,
        "SIGNATURE":    _SIG_3D,
        "constexpr":    ["M", "N", "P", "BLOCK_M", "BLOCK_N", "BLOCK_P", "OP"],
        # 3D grid: [2, 4, 4] = 32 cores
        "params":       {
            "M": [64], "N": [32], "P": [16],
            "BLOCK_M": [8], "BLOCK_N": [8], "BLOCK_P": [8],
            "OP": ["add"],
        },
        "grid":         [2, 4, 4],
        "inputs":       make_inputs_3d,
    },
    "3d_grid_dynamic": {
        "base":      "3d_grid",
        "tags":      ["descriptor-load-dynamic", "descriptor-store-dynamic", "program-id-3d", "num-programs-fold"],
        "summary": (
            "3D grid with runtime `M`, `N`, `P`: distribution loop structure, "
            "dynamic descriptor shapes."
        ),
        "doc": (
            "Same as `3d_grid` but `M`, `N`, `P` are runtime `i32` arguments. "
            "Descriptors lower to `memref<?x?x?xf32>`."
        ),
        "constexpr":    ["BLOCK_M", "BLOCK_N", "BLOCK_P", "OP"],
    },


    # -----------------------------------------------------------------------
    # Level B -- compute
    #
    # The op x dtype product on ktir_cpu. Deliberately the simplest shape in
    # the file -- 1D, static, one tile, no layout -- so arithmetic is the only
    # thing that differs between its entries. LowerSpyreOps only runs
    # at the spyrecode stage (buildSpyrecodePipeline), after ktir_cpu's own
    # module is already built, so this variant's math/arith ops stay in their
    # plain dialect spelling and every combo is reachable end to end.
    # -----------------------------------------------------------------------
    "1d_compute": {
        # No base: prevent inheriting `reference` and `inputs` from `default`
        # (factory hooks produce them; a literal field + hook on the same
        # variant is a collection-time error).
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static",
                 "program-id-1d", "elementwise-compute"],
        "summary": (
            "1D elementwise op across fp16/fp32/i32 and add/sub/mul/div. "
            "Sweeps the OP × DTYPE product to cover ktir_cpu correctness."
        ),
        "kernel_fn":    kernel.elementwise_1d,
        "factory":      Elementwise(rank=1),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "OP"],
        "params": {
            # The full 3x4 product. (i32, div) is the one cell that exercises a
            # tensor-typed arith cast: an i32 division is not an integer op in
            # Triton, so it goes through float and the kernel carries
            # arith.sitofp / divf / fptosi on tensors. Those reach ktir_cpu
            # un-wrapped now that ConvertElementwiseToLinalg is in the spyrecode
            # stage, which is a case ktir-cpu's handlers only gained recently --
            # so this cell is also the suite's floor on that dependency.
            "DTYPE":      ["fp16", "fp32", "i32"],
            "OP":         ["add", "sub", "mul", "div"],
            "n_elements": [128],
            "BLOCK_SIZE": [128],
        },
        "grid":         [1],
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
    },


    # -----------------------------------------------------------------------
    # Level C -- layout
    #
    # Stick physicalization without going to a binary. Bases on Level A's 2d,
    # so this section reads after it.
    # -----------------------------------------------------------------------
    "2d_spyre_stick": {
        # Elementwise add with all three operands stick-on-N. Every operand
        # physicalizes identically, so the add stays a pure elementwise op on
        # rank-3 tiles [N//S, M, N%S] = [2, 64, 64] and no transpose or
        # reduction loop is synthesized.
        "base": "2d",
        "tags": ["descriptor-load-static", "descriptor-store-static",
                 "program-id-1d", "spyre-tensor-layout"],
        "summary": (
            "2D elementwise add with x/y/out all annotated stick-on-N. "
            "Exercises the layout path on a pure elementwise kernel."
        ),
        "SIGNATURE": _SIG_2D_SPYRE,
        "params": {
            # fp16 stick = 64, so N = 128 is exactly 2 sticks.
            "M": [64], "N": [128], "BLOCK_M": [64], "BLOCK_N": [128],
            "X_LAYOUT":   [_stick_2d_on_n("fp16")],
            "Y_LAYOUT":   [_stick_2d_on_n("fp16")],
            "OUT_LAYOUT": [_stick_2d_on_n("fp16")],
            "DTYPE": ["fp16"], "OP": ["add"],
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
        "rtol":        1e-2,
        "atol":        5e-2,
    },
    "2d_spyre_stick_fp32": {
        # The same layout path one stick width down: fp32 sticks are 32 lanes, so
        # the physical view is [2, 64, 32] rather than [2, 64, 64]. Level D covers
        # fp32 stick-tiling on device; this covers it in KTIR and on ktir_cpu,
        # where the numbers are checked rather than just the shapes.
        #
        # Its ``params`` would collapse into one of Level D's dtype groups, and its
        # SIGNATURE differs from its sibling's only in the pointer dtype the
        # factory already derives. What keeps the two apart is rtol/atol: an fp32
        # add is exact and is checked as such, while its fp16 sibling is not.
        # Those are fields, not params, and the params grammar has nothing to say
        # about them -- collapsing the pair would mean a tolerance hook, growing
        # the mechanism, or weakening the fp32 check.
        "base": "2d_spyre_stick",
        "summary": (
            "2D fp32 elementwise add with x/y/out all annotated stick-on-N. "
            "The fp32 arm of the layout path: 32 lanes to a stick, not 64."
        ),
        "SIGNATURE": _SIG_2D_LAYOUT,
        "params": {
            # fp32 stick = 32, so N = 64 is exactly 2 sticks.
            "M": [64], "N": [64], "BLOCK_M": [64], "BLOCK_N": [64],
            "X_LAYOUT":   [_stick_2d_on_n("fp32")],
            "Y_LAYOUT":   [_stick_2d_on_n("fp32")],
            "OUT_LAYOUT": [_stick_2d_on_n("fp32")],
            "DTYPE": ["fp32"], "OP": ["add"],
        },
        "rtol":        1e-6,
        "atol":        0.0,
    },


    # -----------------------------------------------------------------------
    # Level D -- device
    #
    # compiles_to_binary, so test_device_launch.py and the spyrecode lit tests
    # pick these up. Loop-free and one tile per core -- dbo-opt rejects the
    # scf.for every other variant outlines from its program-id distribution.
    #
    # One variant per rank, not per dtype: the stick size and everything that
    # follows from it are grouped with DTYPE in ``params``, one row per dtype, so
    # the fp32 arm is a row rather than a variant that repeats the fp16 one.
    # -----------------------------------------------------------------------
    "1d_device": {
        # The device variants in the suite that reach a Spyre binary. Everything
        # else distributes tiles with an scf.for over the program id, and dbo-opt
        # rejects the loop it outlines from that; this kernel has no loop --
        # one tile that is the whole tensor.
        #
        # That makes it the fixture the spyrecode-stage tests compile, which is
        # why it lives here rather than inline in a test module.
        "base": None,   # prevent inheriting reference/inputs from default
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D elementwise over a single tile, no distribution loop. Sweeps "
            "fp16/fp32 x add/sub/mul -- the arms dbo-opt accepts."
        ),
        "kernel_fn":    kernel.elementwise_1d_device,
        "factory":      Elementwise(rank=1),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT", "OP"],
        "params": {
            # One row per dtype, because the layout's stick width follows from it.
            # Written out rather than built: at rank 1 there is no shape to keep
            # in step, so a row is just the dtype and the layout it implies.
            ("DTYPE", "LAYOUT"): [
                ("fp16", _stick_1d("fp16")),
                ("fp32", _stick_1d("fp32")),
            ],
            "OP":         ["add", "sub", "mul"],
            "n_elements": [128],
            "BLOCK_SIZE": [128],
        },
        "grid":         [1],
        "compiles_to_binary": True,
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
    },
    "1d_device_grid2": {
        # The multi-core counterpart of 1d_device: still one tile per core and
        # still no distribution loop, but two cores each taking half the
        # vector.  n_elements=128 over BLOCK_SIZE=64 is exactly two fp16 sticks,
        # one per core, so core i owns stick i.
        "base": "1d_device",
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "program-id-1d", "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D fp16 elementwise across two cores, one fp16 stick each, "
            "no distribution loop. Sweeps add/sub/mul."
        ),
        "grid": [2],
        "params": {
            # The parent's group with its fp16 row alone: this variant is about
            # work division, not dtype. Redeclared in full because ``params``
            # merges wholesale, so a partial override would leave the fp32 row in.
            ("DTYPE", "LAYOUT"): [
                ("fp16", _stick_1d("fp16")),
            ],
            "OP":         ["add", "sub", "mul"],
            "n_elements": [128],
            "BLOCK_SIZE": [64],   # 64 elements = 1 fp16 stick per core
        },
    },
    "2d_device": {
        "base": None,   # prevent inheriting reference/inputs from default
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "2D elementwise over a single tile, stick-on-N layout. Sweeps "
            "fp16/fp32 x add/sub/mul. First fixture to put a rank-3 "
            "physicalized layout and a multi-element 2D tile on hardware."
        ),
        "kernel_fn":    kernel.elementwise_2d_device,
        "factory":      Elementwise(rank=2),
        "constexpr":    ["M", "N", "BLOCK_M", "BLOCK_N",
                         "X_LAYOUT", "Y_LAYOUT", "OUT_LAYOUT", "OP"],
        "params": {
            # N, BLOCK_N and the three layouts all follow from DTYPE, so they
            # share its rows -- two sticks is 128 lanes at fp16 and 64 at fp32,
            # and no row can say one while its layout says the other. M and
            # BLOCK_M are outside the group because they follow from nothing.
            ("DTYPE", "N", "BLOCK_N",
             "X_LAYOUT", "Y_LAYOUT", "OUT_LAYOUT"): [
                _stick_on_n_row("fp16", n_sticks=2),
                _stick_on_n_row("fp32", n_sticks=2),
            ],
            "OP":       ["add", "sub", "mul"],
            "M":        [64],
            "BLOCK_M":  [64],
        },
        "grid":         [1],
        "compiles_to_binary": True,
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
    },

    # The declared-buffer arm of Level D: the same single-tile, loop-free shape,
    # but several computes rather than one, with every intermediate written to HBM
    # and read back through a descriptor the *author* declared 
    "1d_device_dag_buffers": {
        # One buffer per intermediate -- e, s, m, r -- which is the plainest way to
        # write this and the root of the arm. The DAG rather than a chain, because
        # `e` is read twice: one store and two loads of the same descriptor. A value
        # consumed by two later groups is one buffer read twice, not two buffers, and
        # a chain would not exercise that rule.
        #
        # base: None -- the arm shares nothing with the binary elementwise variants
        # above (they sweep OP through a factory, and a factory hook beside the
        # literal field it produces is a collection-time error).
        "base": None,
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D `out = exp(x) * sqrt(exp(x)) + sqrt(x)` over a single tile, with "
            "all four intermediates in explicit HBM buffers. `e` is read twice."
        ),
        "doc": (
            "Takes one 1D input vector `x` of length `n_elements`, four scratch "
            "vectors of the same length, and writes "
            "`out = exp(x) * sqrt(exp(x)) + sqrt(x)`. One tile, one core, no "
            "loop.\n\n"
            "`e = exp(x)`, `s = sqrt(e)`, `m = e * s` and `r = sqrt(x)` each get "
            "their own descriptor, store and load. `e` is stored once and loaded "
            "twice -- by the `sqrt` and by the multiply -- and `x` is loaded twice, "
            "so no load result is shared across two compute groups."
        ),
        "kernel_fn":    kernel.dag_buffers_1d_device,
        "SIGNATURE":    _SIG_1D_DAG_BUFFERS,
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT"],
        "params": {
            "n_elements": [128], "BLOCK_SIZE": [128], "DTYPE": ["fp32"],
            "LAYOUT": [_stick_1d("fp32")],
        },
        "grid":         [1],
        "compiles_to_binary": True,
        "reference":    run_dag,
        "inputs":       make_inputs_dag_buffers,
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
    },

    # The pooled sub-arm: the same idea with the intermediates in regions of ONE
    # scratch buffer instead of taking a pointer each. 
    "1d_device_chain_pooled": {
        # The floor of the sub-arm: one intermediate, one region, so the pool is used
        # whole and no reuse is claimed. Its job is to separate "pooling works" from
        # "region reuse works" -- if this one fails, nothing further means anything.
        "base": "1d_device_dag_buffers",
        "summary": (
            "1D `out = sqrt(exp(x))` over a single tile, with the intermediate in "
            "one region of a pooled scratch pointer."
        ),
        "doc": (
            "Takes one 1D input vector `x` of length `n_elements`, one scratch "
            "*pool* pointer, and writes `out = sqrt(exp(x))`. One tile, one core, "
            "no loop.\n\n"
            "The pooled counterpart of the per-intermediate spelling, with the "
            "scratch pointer treated as a pool the author carves. There is only one "
            "intermediate here so the pool holds one region and nothing is reused; "
            "the variant exists so that a failure in the reusing variants can be "
            "attributed to the reuse rather than to the pooling."
        ),
        "kernel_fn":    kernel.chain_pooled_1d_device,
        "SIGNATURE":    _SIG_1D_POOLED,
        "inputs":       make_inputs_pooled1,
        "reference":    run_chain,
    },
    "1d_device_chain_pooled_grid2": {
        # The case that tests whether an author needs per-core pinning at all. The
        # region gets the WHOLE-tensor shape and is accessed at `pid * BLOCK_SIZE`,
        # exactly as `x_desc` and `out_desc` are, so the two cores round-trip
        # disjoint blocks of one region. Nothing about the source says which core
        # owns which block; the descriptor's block access does.
        "base": "1d_device_chain_pooled",
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "program-id-1d", "simplified:no-loop", "spyre-tensor-layout",
        ],
        "summary": (
            "1D `out = sqrt(exp(x))` across two cores, with the intermediate in "
            "one whole-tensor region of a pooled scratch pointer."
        ),
        "grid":         [2],
        "params": {
            # 128 elements over BLOCK_SIZE=64 is two blocks of two fp32 sticks each
            # (a stick is 32 elements at fp32), one block per core.
            "n_elements": [128], "BLOCK_SIZE": [64], "DTYPE": ["fp32"],
            "LAYOUT": [_stick_1d("fp32")],
        },
    },
    "1d_device_chain3_pooled": {
        # The one genuine unknown in the sub-arm. Both intermediates share ONE
        # region, because t0's last reader is the compute that produces t1, so the
        # ranges are disjoint at the statement level. The consequence is that the
        # middle schedule does `load R0 -> sqrt -> store R0`.
        #
        # There is a double-buffering pass in the pipeline. Whether it may issue the
        # store into R0 before the pipelined load out of R0 has retired is not
        # something the source can say, so this is written the minimal way and
        # measured; `1d_device_chain3_pooled2` is the two-region control that
        # separates an in-place hazard from a pooling problem.
        "base": "1d_device_chain_pooled",
        "summary": (
            "1D `out = exp(sqrt(exp(x)))` over a single tile, with both "
            "intermediates sharing one region of a pooled scratch pointer."
        ),
        "doc": (
            "Takes one 1D input vector `x` of length `n_elements`, one scratch "
            "*pool* pointer of the same length, and writes "
            "`out = exp(sqrt(exp(x)))`. One tile, one core, no loop.\n\n"
            "`t0 = exp(x)` and `t1 = sqrt(t0)` have disjoint live ranges -- `t0`'s "
            "last reader is the compute that produces `t1` -- so one region is the "
            "minimum an author can justify, and the middle compute both loads from "
            "and stores to it. Whether reading and writing one region inside a "
            "single schedule survives the double-buffering pass is what this "
            "variant measures; `1d_device_chain3_pooled2` is the two-region "
            "control."
        ),
        "kernel_fn":    kernel.chain3_pooled_1d_device,
        "inputs":       make_inputs_pooled1,
        "reference":    run_chain3,
    },
    "1d_device_chain3_pooled2": {
        # Control for the above: same pool, two regions, so no schedule reads and
        # writes the same one. Kept as a variant rather than a comment because the
        # comparison only means something if both are run by the same suite on the
        # same day.
        "base": "1d_device_chain3_pooled",
        "summary": (
            "1D `out = exp(sqrt(exp(x)))` over a single tile, with the two "
            "intermediates in two regions of a pooled scratch buffer."
        ),
        "doc": (
            "Takes one 1D input vector `x` of length `n_elements`, one scratch "
            "*pool* buffer of twice that length, and writes "
            "`out = exp(sqrt(exp(x)))`. One tile, one core, no loop.\n\n"
            "The control for `1d_device_chain3_pooled`: identical pooling, but "
            "`t0` and `t1` get a region each, so no compute loads from and stores "
            "to the same region. A difference between the two is the in-place "
            "reuse and nothing else.\n\n"
            "The regions are block offsets into one whole-pool descriptor. That is "
            "forced rather than chosen -- a descriptor per region would need "
            "pointer arithmetic on the pool argument, which the memory passes do "
            "not consume."
        ),
        "kernel_fn":    kernel.chain3_pooled2_1d_device,
        "inputs":       make_inputs_pooled2,
    },
    "1d_device_dag_pooled": {
        # Four intermediates, THREE regions. The minimum, not a round number:
        # e:[1,3] s:[2,3] m:[3,5] r:[4,5] over the kernel's statement order, and
        # {e, s, m} all contain statement 3, so three is forced. Achievable because
        # r is stored after e's last read, so R0 carries e and then r.
        # The full derivation is in the kernel's docstring, next to the code it
        # constrains.
        "base": "1d_device_chain_pooled",
        "summary": (
            "1D `out = exp(x) * sqrt(exp(x)) + sqrt(x)` over a single tile, with "
            "four intermediates in three pooled regions."
        ),
        "doc": (
            "Takes one 1D input vector `x` of length `n_elements`, one scratch "
            "*pool* pointer of three times that length, and writes "
            "`out = exp(x) * sqrt(exp(x)) + sqrt(x)`. One tile, one core, no "
            "loop.\n\n"
            "`1d_device_dag_buffers` takes four scratch pointers for `e`, `s`, `m` "
            "and `r`; this takes one pool and puts them in three regions. Three is "
            "the minimum: `e`, `s` and `m` are all live at the compute that reads "
            "`e` and `s` and writes `m`, so no two of them can share, while `r` is "
            "written after `e`'s last read and reuses `e`'s region. The reuse here "
            "is across schedules, unlike `1d_device_chain3_pooled`'s, which is "
            "inside one."
        ),
        "kernel_fn":    kernel.dag_pooled_1d_device,
        "inputs":       make_inputs_pooled3,
        "reference":    run_dag,
    },

}
