"""SIGNATURE + VARIANTS + reference oracle + input generators for vector_add.

Six variants — 1D static/dynamic, 2D static/dynamic, 3D static/dynamic —
share the same NumPy oracle (``x + y``). They exercise tensor descriptors
at increasing dimensionality and static-vs-dynamic shape lowering.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import functools

import numpy as np

from . import kernel
from utils import sticksize


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


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(n_elements: int, BLOCK_SIZE: int, dtype: str = "f32") -> dict:
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


def make_inputs_2d(M: int, N: int, BLOCK_M: int, BLOCK_N: int,
                   *, dtype=np.float32, **_unused) -> dict:
    """Build 2D pointer-tensor inputs for add_kernel_2d."""
    del BLOCK_M, BLOCK_N
    total = M * N
    t = np.arange(total, dtype=np.float32)
    x = np.sin(t * 2.0 * np.pi / total).astype(dtype).reshape(M, N)
    y = np.cos(t * 2.0 * np.pi / total).astype(dtype).reshape(M, N)
    output = np.zeros((M, N), dtype=dtype)
    return {"x_ptr": x, "y_ptr": y, "output_ptr": output}


def make_inputs_3d(
    M: int, N: int, P: int,
    BLOCK_M: int, BLOCK_N: int, BLOCK_P: int,
) -> dict:
    """Build 3D pointer-tensor inputs for add_kernel_3d."""
    del BLOCK_M, BLOCK_N, BLOCK_P
    total = M * N * P
    t = np.arange(total, dtype=np.float32)
    x = np.sin(t * 2.0 * np.pi / total).astype(np.float32).reshape(M, N, P)
    y = np.cos(t * 2.0 * np.pi / total).astype(np.float32).reshape(M, N, P)
    output = np.zeros((M, N, P), dtype=np.float32)
    return {"x_ptr": x, "y_ptr": y, "output_ptr": output}



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

# add_kernel_2d carries the optional layout constexprs; add_kernel_2d_grid
# (which shares _SIG_2D) does not.
_SIG_2D_LAYOUT = {
    **_SIG_2D,
    "X_LAYOUT":   "constexpr",
    "Y_LAYOUT":   "constexpr",
    "OUT_LAYOUT": "constexpr",
}

_SIG_2D_SPYRE = {
    "x_ptr":      "*fp16",
    "y_ptr":      "*fp16",
    "output_ptr": "*fp16",
    "M":          "i32",
    "N":          "i32",
    "BLOCK_M":    "i32",
    "BLOCK_N":    "i32",
    "X_LAYOUT":   "constexpr",
    "Y_LAYOUT":   "constexpr",
    "OUT_LAYOUT": "constexpr",
}


_S2 = functools.partial(sticksize, _SIG_2D_SPYRE)

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
        "kernel_fn":    kernel.add_kernel,
        "constexpr":    ["n_elements", "BLOCK_SIZE"],
        "params":       {
            # n_elements=[1024,2097152,2097153]: absorbs single_block (1024)
            # and nonaligned (2097153).
            "n_elements": [1024, 2097152, 2097153], "BLOCK_SIZE": [1024],
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
        "constexpr":    ["BLOCK_SIZE"],
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
        "params": {"n_elements": [4096], "BLOCK_SIZE": [1024]},
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
        "kernel_fn":    kernel.add_kernel_2d,
        "SIGNATURE":    _SIG_2D_LAYOUT,
        "constexpr":    ["M", "N", "BLOCK_M", "BLOCK_N",
                         "X_LAYOUT", "Y_LAYOUT", "OUT_LAYOUT"],
        "params":       {
            # M=[512,520]: absorbs 2d_nonaligned (M=520).
            "M": [512, 520], "N": [32], "BLOCK_M": [16], "BLOCK_N": [16],
            "X_LAYOUT": [None], "Y_LAYOUT": [None], "OUT_LAYOUT": [None],
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
                         "X_LAYOUT", "Y_LAYOUT", "OUT_LAYOUT"],
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
        "kernel_fn":    kernel.add_kernel_3d,
        "SIGNATURE":    _SIG_3D,
        "constexpr":    ["M", "N", "P", "BLOCK_M", "BLOCK_N", "BLOCK_P"],
        "params":       {
            # M=[64,65,256]: absorbs 3d_nonaligned (M=65) and 3d_active_cores (M=256).
            "M": [64, 65, 256], "N": [32], "P": [16],
            "BLOCK_M": [8], "BLOCK_N": [8], "BLOCK_P": [8],
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
        "constexpr":    ["BLOCK_M", "BLOCK_N", "BLOCK_P"],
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
        "kernel_fn":    kernel.add_kernel_2d_grid,
        "SIGNATURE":    _SIG_2D,
        "constexpr":    ["M", "N", "BLOCK_M", "BLOCK_N"],
        # 2D grid: [4, 8] = 32 cores
        "params":       {"M": [256], "N": [128], "BLOCK_M": [16], "BLOCK_N": [16]},
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
        "constexpr":    ["BLOCK_M", "BLOCK_N"],
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
        "kernel_fn":    kernel.add_kernel_3d_grid,
        "SIGNATURE":    _SIG_3D,
        "constexpr":    ["M", "N", "P", "BLOCK_M", "BLOCK_N", "BLOCK_P"],
        # 3D grid: [2, 4, 4] = 32 cores
        "params":       {
            "M": [64], "N": [32], "P": [16],
            "BLOCK_M": [8], "BLOCK_N": [8], "BLOCK_P": [8],
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
        "constexpr":    ["BLOCK_M", "BLOCK_N", "BLOCK_P"],
        "extra_checks": lambda t: (
            t.assert_result_type("ktdp.construct_memory_view",
                                 "memref<?x?x?xf32>"),
        ),
    },
}
