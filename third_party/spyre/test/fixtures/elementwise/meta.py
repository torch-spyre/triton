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
# extra_checks factories (for variants with multi-value params that contain
# shapes in the check).  Each factory accepts **combo and returns a
# (tester)->None function.
# ---------------------------------------------------------------------------

def _make_1d_checks(n_elements, BLOCK_SIZE, **_):
    def checks(t):
        if n_elements != BLOCK_SIZE:
            t.assert_result("ktdp.construct_memory_view", shape_not=[BLOCK_SIZE])
    return checks


def _make_2d_checks(M, N, **_):
    def checks(t):
        t.assert_result_type("ktdp.construct_memory_view", f"memref<{M}x{N}xf32>")
    return checks

def _make_3d_checks(M, N, P, **_):
    def checks(t):
        t.assert_result_type("ktdp.construct_memory_view", f"memref<{M}x{N}x{P}xf32>")
    return checks

def _make_1d_scalar_dim_checks(**_):
    def checks(t):
        # Single-element 1-D scalar-read chain: construct_memory_view
        # <memref<1xi32>> -> construct_access_tile<1xindex> -> ktdp.load
        # -> tensor.extract.
        t.assert_result_type("ktdp.construct_memory_view", "memref<1xi32>")
        t.assert_result_type("ktdp.construct_access_tile", "<1xindex>")
        t.assert_present("tensor.extract")
        # arith.index_cast bridges the extracted i32 to index before it
        # feeds the descriptor's dynamic size operand.
        t.assert_operand("ktdp.construct_memory_view", 1,
                         defined_by="arith.index_cast", type_substr="index")
        # The descriptor shape lowers to a dynamic memref.
        t.assert_result_type("ktdp.construct_memory_view", "memref<?xf32>")
    return checks


def _make_2d_scalar_dim_checks(N, **_):
    def checks(t):
        # Single-element 1-D scalar-read chain: construct_memory_view
        # <memref<1xi32>> -> construct_access_tile<1xindex> -> ktdp.load
        # -> tensor.extract.
        t.assert_result_type("ktdp.construct_memory_view", "memref<1xi32>")
        t.assert_result_type("ktdp.construct_access_tile", "<1xindex>")
        t.assert_present("tensor.extract")
        # arith.index_cast bridges the extracted i32 to index before it
        # feeds the descriptor's dynamic size operand.
        t.assert_operand("ktdp.construct_memory_view", 1,
                         defined_by="arith.index_cast", type_substr="index")
        # The descriptor shape lowers to a dynamic memref.
        t.assert_result_type("ktdp.construct_memory_view", f"memref<?x{N}xf32>")
    return checks


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input makers
# ---------------------------------------------------------------------------

def make_inputs(n_elements: int, BLOCK_SIZE: int,
                *, dtype="f32", **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    Takes the same parameter names as ``VARIANTS[...]["params"]`` so the
    framework can pass them positionally as kwargs. ``BLOCK_SIZE`` is
    unused here but accepted so the signature matches the full param
    set (keeps things uniform for kernels that use it in tensor
    construction).

    Returns only pointer/tensor args keyed by SIGNATURE name. Runtime
    scalars (e.g. ``n_elements`` in the dynamic variant) are added by
    the framework from ``params`` before calling ``run_cpu``.
    """
    del BLOCK_SIZE  # not used in input generation, but part of the param set
    np_dtype = {"f32": np.float32, "f16": np.float16}[dtype]
    t = np.arange(n_elements, dtype=np.float32)
    x = np.sin(t * 2.0 * np.pi / n_elements).astype(np_dtype)
    y = np.cos(t * 2.0 * np.pi / n_elements).astype(np_dtype)
    output = np.zeros(n_elements, dtype=np_dtype)
    return {"x_ptr": x, "y_ptr": y, "output_ptr": output}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``x + y``. Works for any shape."""
    return inputs["x_ptr"] + inputs["y_ptr"]


def make_inputs_scalar_dim(BLOCK_SIZE: int, dtype: str = "f32", n_elements: int = 4096, **_unused) -> dict:
    """1D inputs for elementwise_1d_scalar_dim.

    `n_elements` is not part of the kernel's SIGNATURE (it is read from
    `seqlen_ptr`, not passed as an arg), so it is a keyword default here
    rather than a `params` entry — same reasoning as
    `make_inputs_2d_scalar_dim`. Delegates to ``make_inputs`` for the
    x/y/output buffers and adds the scalar read as a rank-1 buffer of
    length 1.
    """
    inputs = make_inputs(n_elements, BLOCK_SIZE, dtype=dtype)
    inputs["seqlen_ptr"] = np.array([n_elements], dtype=np.int32)
    return inputs


def make_inputs_2d(M: int, N: int, BLOCK_M: int, BLOCK_N: int,
                   *, dtype=np.float32, **_unused) -> dict:
    """Build 2D pointer-tensor inputs for elementwise_2d."""
    del BLOCK_M, BLOCK_N
    total = M * N
    t = np.arange(total, dtype=np.float32)
    x = np.sin(t * 2.0 * np.pi / total).astype(dtype).reshape(M, N)
    y = np.cos(t * 2.0 * np.pi / total).astype(dtype).reshape(M, N)
    output = np.zeros((M, N), dtype=dtype)
    return {"x_ptr": x, "y_ptr": y, "output_ptr": output}


def make_inputs_2d_scalar_dim(N: int, BLOCK_M: int, BLOCK_N: int, M: int = 32, **_unused) -> dict:
    """2D inputs for elementwise_2d_scalar_dim.

    `M` is not part of the kernel's SIGNATURE (it is read from
    `seqlen_ptr`, not passed as an arg), so it is a keyword default here
    rather than a `params` entry — a `params` entry would otherwise leak
    into ``run_cpu``'s kwargs and fail its "unknown kwarg" check, since
    `elementwise_2d_scalar_dim` has no `M` parameter. Delegates to
    ``make_inputs_2d`` for the x/y/output buffers and adds the scalar
    read as a rank-1 buffer of length 1.
    """
    inputs = make_inputs_2d(M, N, BLOCK_M, BLOCK_N)
    inputs["seqlen_ptr"] = np.array([M], dtype=np.int32)
    return inputs


def make_inputs_3d(
    M: int, N: int, P: int,
    BLOCK_M: int, BLOCK_N: int, BLOCK_P: int,
    **_unused,
) -> dict:
    """Build 3D pointer-tensor inputs for elementwise_3d."""
    del BLOCK_M, BLOCK_N, BLOCK_P
    total = M * N * P
    t = np.arange(total, dtype=np.float32)
    x = np.sin(t * 2.0 * np.pi / total).astype(np.float32).reshape(M, N, P)
    y = np.cos(t * 2.0 * np.pi / total).astype(np.float32).reshape(M, N, P)
    output = np.zeros((M, N, P), dtype=np.float32)
    return {"x_ptr": x, "y_ptr": y, "output_ptr": output}


# ---------------------------------------------------------------------------
# Level B factory — Elementwise(VariantFactory)
# ---------------------------------------------------------------------------

# Shape-arg dicts for each rank's runtime signature (no pointer types — those
# come from DTYPE). These are the non-constexpr, non-pointer runtime args.
_SHAPE_ARGS = {
    1: {"n_elements": "i32"},      # elementwise_1d
    # 2D and 3D are not used by Level B (1d_compute), kept for Stage 3
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


def _make_inputs_compute(n_elements: int, BLOCK_SIZE: int,
                         *, dtype: str, op: str, **_unused) -> dict:
    """Input generator for 1d_compute: dtype/op-aware.

    Floats: x = sin, y = cos + 2 (keeps div away from zero and gives
    interesting values). i32: small positive integers so mul stays in range
    and floordiv is exact.
    """
    np_dtype = DTYPE_MAP[dtype]
    n = n_elements
    if dtype == "i32":
        x = np.arange(1, n + 1, dtype=np_dtype)
        y = np.full(n, 3, dtype=np_dtype)
    else:
        t = np.arange(n, dtype=np.float32)
        x = np.sin(t * 2.0 * np.pi / n).astype(np_dtype)
        y = (np.cos(t * 2.0 * np.pi / n) + 2.0).astype(np_dtype)
    output = np.zeros(n, dtype=np_dtype)
    return {"x_ptr": x, "y_ptr": y, "output_ptr": output}


@dataclass(frozen=True)
class Elementwise(conftest.VariantFactory):
    """Factory for elementwise variants that sweep OP and/or DTYPE.

    ``rank`` selects which _SHAPE_ARGS entry to use for the runtime signature.
    Only rank=1 is used in Stage 2 (1d_compute); rank=2 is for Stage 3 (2d_device).
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

    def inputs(self, OP, DTYPE, **_):
        return functools.partial(_make_inputs_compute, dtype=DTYPE, op=OP)


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
#   - ``params``    : dict of arg name -> list of values. Single-element
#                     lists for now; the Cartesian expansion is deferred.
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

_S2 = functools.partial(sticksize, _SIG_TENSORS_FP16)

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

VARIANTS = {
    # --- 1D variants ---
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
        "constexpr":    ["n_elements", "BLOCK_SIZE", "DTYPE", "OP"],
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
        "extra_checks": _make_1d_checks,
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
        "constexpr":    ["BLOCK_SIZE", "DTYPE", "OP"],
        "extra_checks": lambda t: (
            # Dynamic path: construct_memory_view must carry a dynamic
            # dimension (memref<?x...>) — the whole point of this variant
            # is exercising the dynamic-shape lowering through
            # LowerDescriptorMemory.
            t.assert_result_type("ktdp.construct_memory_view", "memref<?x"),
        ),
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
        "extra_checks": _make_1d_scalar_dim_checks,
    },
    # --- Level B: compute correctness (op × dtype, ktir_cpu only) ---
    "1d_compute": {
        # No base: prevent inheriting `reference` and `inputs` from `default`
        # (factory hooks produce them; a literal field + hook on the same
        # variant is a collection-time error).
        "base": None,
        "tags": ["descriptor-load-static", "descriptor-store-static",
                 "program-id-1d", "elementwise-compute"],
        "summary": (
            "1D elementwise op across fp16/fp32/i32 and add/sub/mul/div. "
            "Sweeps the OP × DTYPE product to cover ktir_cpu correctness; "
            "div and i32 stop here pending #107."
        ),
        "kernel_fn":    kernel.elementwise_1d,
        "factory":      Elementwise(rank=1),
        "constexpr":    ["n_elements", "BLOCK_SIZE", "DTYPE", "OP"],
        "params": {
            "DTYPE":      ["fp16", "fp32", "i32"],
            "OP":         ["add", "sub", "mul", "div"],
            "n_elements": [128],
            "BLOCK_SIZE": [128],
        },
        "grid":         [1],
        "parallel":     False,
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
    },
    # --- 2D variants ---
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
            "OP": ["add"],
        },
        "inputs":       make_inputs_2d,
        "extra_checks": _make_2d_checks,
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
        "extra_checks": lambda t: (
            t.assert_result_type("ktdp.construct_memory_view",
                                 "memref<?x?xf32>"),
        ),
    },
    "2d_dynamic_alt": {
        # Different N than the static 2d sibling: confirms the compiled
        # dynamic kernel runs at a shape distinct from its static sibling.
        "base":   "2d_dynamic",
        "params": {
            "M": [256], "N": [64], "BLOCK_M": [16], "BLOCK_N": [16],
            "X_LAYOUT": [None], "Y_LAYOUT": [None], "OUT_LAYOUT": [None],
            "OP": ["add"],
        },
    },
    "1d_device": {
        # The only variant in the suite that reaches a Spyre binary. Everything
        # else distributes tiles with an scf.for over the program id, and dbo-opt
        # rejects the loop it outlines from that; this kernel has no loop and no
        # program id -- one tile that is the whole tensor.
        #
        # That makes it the fixture the spyrecode-stage tests compile, which is
        # why it lives here rather than inline in a test module.
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "simplified:no-loop", "spyre-tensor-layout"
        ],
        "summary": (
            "Elementwise add over a single tile, with no grid distribution "
            "loop -- the one variant dbo-opt can lower to a binary."
        ),
        "doc": (
            "Takes two `M` fp16 inputs in spyre tensor 2D layouts "
            "and writes `C = A + B` in one "
            "`tl.program_id`, no `tl.num_programs`, no loop."
        ),
        "kernel_fn":    kernel.elementwise_1d_device,
        "SIGNATURE":    _SIG_TENSORS_FP16,
        "constexpr":    ["n_elements", "BLOCK_SIZE", "LAYOUT", "OP"],
        "params":       {
            "n_elements": [128],
            "BLOCK_SIZE": [128],
            # LAYOUT is a constexpr that enters the Triton cache key, so it
            # must be hashable -- a list is not. Stored as tuple of tuples.
            # A labelled ("name", value) pair is required: the framework's
            # tuple-detection sees any tuple element in the values list and
            # would misread the raw inner tuple as a (label, value) pair. The
            # label also becomes the suffix when sweeping: adding a second
            # layout produces elementwise__1d_device[LAYOUT=stick] and
            # elementwise__1d_device[LAYOUT=other].
            "LAYOUT":   [("stick", ((0, "floordiv", _S2("x_ptr")), (0, "mod", _S2("x_ptr"))))],
            "OP": ["add"],
        },
        "grid":         [1],
        "parallel":     False,
        # dbo-opt can lower this one all the way to a Spyre binary, so the
        # spyrecode-stage tests compile it. Declared here rather than named in
        # conftest so the property travels with the variant, like `parallel`.
        "compiles_to_binary": True,
        "reference":    run,
        # BLOCK_SIZE is actually not used, stub it out
        "inputs":       functools.partial(make_inputs, dtype="f16"),
        "output_key":   "output_ptr",
        "rtol":         1e-2,
        "atol":         5e-2,
        "extra_checks": None, # for the device examples we disable the checks
    },
    "1d_device_grid2": {
        # The multi-core counterpart of 1d_device: still one tile per corelet and
        # still no distribution loop, but two corelets each taking half the
        # vector.  n_elements=128 over BLOCK_SIZE=64 is exactly two fp16 sticks,
        # one per corelet, so corelet i owns stick i.
        "base": "1d_device",
        "tags": [
            "descriptor-load-static", "descriptor-store-static",
            "program-id-1d", "simplified:no-loop", "spyre-tensor-layout"
        ],
        "summary": (
            "Elementwise add distributed over two corelets, one stick each, "
            "with no distribution loop."
        ),
        "doc": (
            "Takes two 128-element fp16 vectors in a 2-stick spyre tensor "
            "layout and writes `C = A + B` across a `grid = [2]`.  Each corelet "
            "reads `tl.program_id(0)` and offsets by `pid * BLOCK_SIZE`, so it "
            "owns exactly one 64-element stick; there is no `tl.num_programs` "
            "and no `scf.for`.\n\n"
        ),
        "grid": [2],
        "params":       {
            "n_elements": [128],
            "BLOCK_SIZE": [64],
            "LAYOUT":   [("stick", ((0, "floordiv", _S2("x_ptr")), (0, "mod", _S2("x_ptr"))))],
            "OP": ["add"],
        },
    },
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
            # fp16 stick = 64; N = 128 = 2 sticks exactly.
            "M": [64], "N": [128], "BLOCK_M": [64], "BLOCK_N": [128],
            "X_LAYOUT":   [[(1, "floordiv", _S2("x_ptr")), 0,
                            (1, "mod", _S2("x_ptr"))]],
            "Y_LAYOUT":   [[(1, "floordiv", _S2("y_ptr")), 0,
                            (1, "mod", _S2("y_ptr"))]],
            "OUT_LAYOUT": [[(1, "floordiv", _S2("output_ptr")), 0,
                            (1, "mod", _S2("output_ptr"))]],
            "OP": ["add"],
        },
        "grid":        [1],
        "data_layout": "host",
        "inputs":      functools.partial(make_inputs_2d, dtype=np.float16),
        "rtol":        1e-2,
        "atol":        5e-2,
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            # All three operands share the rank-3 physical view [2, 64, 64].
            t.assert_result_type("ktdp.construct_memory_view", "2x64x64xf16"),
        ),
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
        "extra_checks": _make_2d_scalar_dim_checks,
    },
    # --- 3D variants ---
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
        "extra_checks": _make_3d_checks,
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
        "extra_checks": lambda t: (
            t.assert_result_type("ktdp.construct_memory_view",
                                 "memref<?x?x?xf32>"),
        ),
    },
    # --- 2D grid variants ---
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
        "extra_checks": lambda t: (
            t.assert_result_type("ktdp.construct_memory_view",
                                 "memref<256x128xf32>"),
        ),
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
        "extra_checks": lambda t: (
            t.assert_result_type("ktdp.construct_memory_view",
                                 "memref<?x?xf32>"),
        ),
    },
    # --- 3D grid variants ---
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
        "extra_checks": lambda t: (
            t.assert_result_type("ktdp.construct_memory_view",
                                 "memref<64x32x16xf32>"),
        ),
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
        "extra_checks": lambda t: (
            t.assert_result_type("ktdp.construct_memory_view",
                                 "memref<?x?x?xf32>"),
        ),
    },
}
