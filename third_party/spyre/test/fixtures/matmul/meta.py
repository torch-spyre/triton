"""SIGNATURE + VARIANTS + reference oracle + input generators for matmul.

Eight variants exercise ``tt.dot`` → ``linalg.matmul`` at increasing complexity.
Two grid styles:

1D-grid variants (loop-distributed):
- ``default``       -- 2D static
- ``dynamic``       -- 2D dynamic
- ``bmm``           -- batched 3D static
- ``bmm_dynamic``   -- batched 3D dynamic

Multi-axis grid variants (one tile per core):
- ``2d_grid``           -- 2D grid, static
- ``2d_grid_dynamic``   -- 2D grid, dynamic
- ``bmm_3d_grid``       -- 3D grid BMM, static
- ``bmm_3d_grid_dynamic`` -- 3D grid BMM, dynamic

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input makers
# ---------------------------------------------------------------------------

def make_inputs(M: int, K: int, N: int, **_unused) -> dict:
    """Build ``[M,K]``, ``[K,N]``, ``[M,N]`` buffers for matmul_kernel."""
    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)
    return {"a_ptr": a, "b_ptr": b, "c_ptr": c}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: standard 2D matmul."""
    return inputs["a_ptr"] @ inputs["b_ptr"]


def make_inputs_bmm(B: int, M: int, K: int, N: int, **_unused) -> dict:
    """Build ``[B,M,K]``, ``[B,K,N]``, ``[B,M,N]`` buffers for bmm_matmul_kernel."""
    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal((B, M, K)).astype(np.float32)
    b = rng.standard_normal((B, K, N)).astype(np.float32)
    c = np.zeros((B, M, N), dtype=np.float32)
    return {"a_ptr": a, "b_ptr": b, "c_ptr": c}


def run_bmm(inputs: dict) -> np.ndarray:
    """NumPy oracle: batched matmul via ``@`` broadcasting."""
    return inputs["a_ptr"] @ inputs["b_ptr"]


# ---------------------------------------------------------------------------
# SIGNATURE — module-level default (matches matmul_kernel's arg list).
# ---------------------------------------------------------------------------

SIGNATURE = {
    "a_ptr":   "*fp32",
    "b_ptr":   "*fp32",
    "c_ptr":   "*fp32",
    "M":       "i32",
    "K":       "i32",
    "N":       "i32",
    "BLOCK_M": "i32",
    "BLOCK_K": "i32",
    "BLOCK_N": "i32",
}

_SIG_BMM = {
    "a_ptr":   "*fp32",
    "b_ptr":   "*fp32",
    "c_ptr":   "*fp32",
    "B":       "i32",
    "M":       "i32",
    "K":       "i32",
    "N":       "i32",
    "BLOCK_B": "i32",
    "BLOCK_M": "i32",
    "BLOCK_K": "i32",
    "BLOCK_N": "i32",
}

_SIG_2D_GRID = {
    "a_ptr":   "*fp32",
    "b_ptr":   "*fp32",
    "c_ptr":   "*fp32",
    "M":       "i32",
    "K":       "i32",
    "N":       "i32",
    "BLOCK_M": "i32",
    "BLOCK_K": "i32",
    "BLOCK_N": "i32",
}

_SIG_BMM_3D_GRID = {
    "a_ptr":   "*fp32",
    "b_ptr":   "*fp32",
    "c_ptr":   "*fp32",
    "B":       "i32",
    "M":       "i32",
    "K":       "i32",
    "N":       "i32",
    "BLOCK_B": "i32",
    "BLOCK_M": "i32",
    "BLOCK_K": "i32",
    "BLOCK_N": "i32",
}

_SIG_BMM_ADDPTR = {
    "a_ptr":   "*fp32",
    "b_ptr":   "*fp32",
    "c_ptr":   "*fp32",
    "B":       "i32",
    "M":       "i32",
    "K":       "i32",
    "N":       "i32",
    "BLOCK_M": "i32",
    "BLOCK_K": "i32",
    "BLOCK_N": "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    # --- 2D variants ---
    "default": {
        # Static: M, K, N baked in as constexprs → memref<MxKxf32>.
        # M=512 so ceil(M/BLOCK_M)/32 = ceil(32/32) = 1 m-block per core.
        "tags": ["descriptor-load-static", "descriptor-store-static", "dot", "program-id-2d", "num-programs-fold"],
        "summary": (
            "Standard 2D matmul `C = A · B` with `M`, `K`, `N` known "
            "at compile time."
        ),
        "doc": (
            "Takes an `M × K` matrix `A` and a `K × N` matrix `B` and "
            "writes `C = A · B` into an `M × N` output. The kernel "
            "tiles the M and N axes of the output across the 32-core "
            "grid: each core is responsible for one or more "
            "`BLOCK_M × BLOCK_N` output tiles, accumulates partial "
            "products along the `K` axis in a `BLOCK_K`-sized inner "
            "loop, and writes the result back in one pass.\n\n"
            "`M`, `K`, `N` are compile-time constants, so the "
            "descriptors lower to fully-static shapes "
            "(`memref<512x64xf32>` for `A`, etc.). The core operator is "
            "a `tl.dot` per inner iteration, which the Spyre pipeline "
            "lowers to `linalg.matmul`."
        ),
        "kernel_fn":    kernel.matmul_kernel,
        "constexpr":    ["M", "K", "N", "BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "params":       {
            "M": [512], "K": [64], "N": [256],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "grid":         [32],
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "c_ptr",
        # fp32 matmul accumulation order differs from NumPy's @ — allow ~1% drift.
        "rtol":         1e-2,
        "atol":         1e-3,
        "extra_checks": lambda t: (
            t.assert_present("linalg.matmul"),
            t.assert_absent("tt.dot"),
            t.assert_result_type("ktdp.construct_memory_view", "memref<512x64xf32>"),
        ),
    },
    "nonaligned": {
        # M=520: m_blocks=33, not divisible by 32 cores.
        # ceil(33/32)=2 for core 0; tl.minimum clamp fires on the last core.
        "base":   "default",
        "params": {
            "M": [520], "K": [64], "N": [256],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "extra_checks": lambda t: (
            t.assert_present("linalg.matmul"),
            t.assert_absent("tt.dot"),
            t.assert_result_type("ktdp.construct_memory_view", "memref<520x64xf32>"),
        ),
    },
    "dynamic": {
        # Dynamic: M, K, N are runtime i32 → memref<?x?xf32>.
        "tags": ["descriptor-load-dynamic", "descriptor-store-dynamic", "dot", "program-id-2d", "num-programs-fold"],
        "summary": (
            "Same matmul as above, but with `M`, `K`, `N` passed as "
            "runtime arguments."
        ),
        "doc": (
            "The static matmul reshaped to take `M`, `K`, `N` as "
            "runtime `i32` arguments. Tile sizes `BLOCK_M`, `BLOCK_K`, "
            "`BLOCK_N` remain compile-time constants — only the matrix "
            "dimensions are runtime. The descriptors lower to "
            "`memref<?x?xf32>`, so the compiled kernel runs unchanged "
            "for any `(M, K, N)` that fits the scratchpad tile budget."
        ),
        "constexpr":    ["BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "extra_checks": lambda t: (
            t.assert_present("linalg.matmul"),
            t.assert_result_type("ktdp.construct_memory_view", "memref<?x?xf32>"),
        ),
    },
    "dynamic_nonaligned": {
        # M=500: non-divisible by 32 → clamp fires in the dynamic-bounds path.
        "base":   "dynamic",
        "params": {
            "M": [500], "K": [64], "N": [256],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
    },
    # --- BMM (batched) variants ---
    "bmm": {
        # BMM static: uses 3D descriptors tiled in batch dimension.
        "tags": ["descriptor-load-static", "descriptor-store-static", "dot", "program-id-1d", "num-programs-fold"],
        "kernel_fn":    kernel.bmm_matmul_kernel,
        "SIGNATURE":    _SIG_BMM,
        "constexpr":    ["B", "M", "K", "N", "BLOCK_B", "BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "params":       {
            "B": [4], "M": [128], "K": [32], "N": [64],
            "BLOCK_B": [1], "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "reference":    run_bmm,
        "inputs":       make_inputs_bmm,
        "output_key":   "c_ptr",
        "rtol":         1e-2,
        "extra_checks": lambda t: (
            t.assert_present("linalg.batch_matmul"),
            t.assert_absent("tt.dot"),
        ),
    },
    "bmm_dynamic": {
        # BMM dynamic: B, M, K, N are runtime i32.
        "tags": ["descriptor-load-dynamic", "descriptor-store-dynamic", "dot", "program-id-1d", "num-programs-fold"],
        "kernel_fn":    kernel.bmm_matmul_kernel,
        "SIGNATURE":    _SIG_BMM,
        "constexpr":    ["BLOCK_B", "BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "params":       {
            "B": [4], "M": [128], "K": [32], "N": [64],
            "BLOCK_B": [1], "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "reference":    run_bmm,
        "inputs":       make_inputs_bmm,
        "output_key":   "c_ptr",
        "rtol":         1e-2,
        "extra_checks": lambda t: (
            t.assert_present("linalg.batch_matmul"),
            t.assert_absent("tt.dot"),
        ),
    },
    "bmm_nonaligned": {
        # B=5: b_blocks=5, bm_blocks=5*8=40, not divisible by 32 cores.
        # tl.minimum clamp fires on the static bm distribution.
        "base":   "bmm",
        "params": {
            "B": [5], "M": [128], "K": [32], "N": [64],
            "BLOCK_B": [1], "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
    },
    # --- 2D grid variant ---
    "2d_grid": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "dot", "program-id-2d"],
        "summary": (
            "2D grid matmul: pid_0 distributes M-tiles, pid_1 distributes N-tiles, "
            "each via a loop over its assigned tile range."
        ),
        "doc": (
            "Same `C = A · B` computation as the static matmul, but uses a "
            "2D program grid. Each axis loops over its assigned tiles: `pid_0` "
            "covers M-tiles and `pid_1` covers N-tiles. The K dimension is the "
            "innermost reduction loop. A distribution loop is always present "
            "so the kernel handles any input size for a fixed grid and tile size."
        ),
        "kernel_fn":    kernel.matmul_kernel_2d_grid,
        "SIGNATURE":    _SIG_2D_GRID,
        "constexpr":    ["M", "K", "N", "BLOCK_M", "BLOCK_K", "BLOCK_N"],
        # 2D grid: [4, 8] = 32 cores; each core loops over its M- and N-tile strip
        "params":       {
            "M": [256], "K": [64], "N": [128],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "grid":         [4, 8],
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "c_ptr",
        "rtol":         1e-2,
        "extra_checks": lambda t: (
            t.assert_present("linalg.matmul"),
            t.assert_result_type("ktdp.construct_memory_view", "memref<256x64xf32>"),
        ),
    },
    "2d_grid_dynamic": {
        "tags": ["descriptor-load-dynamic", "descriptor-store-dynamic", "dot", "program-id-2d"],
        "summary": (
            "2D grid matmul with runtime `M`, `K`, `N`: same distribution-loop "
            "structure, dynamic descriptor shapes."
        ),
        "doc": (
            "Same as `2d_grid` but `M`, `K`, `N` arrive as runtime `i32` "
            "arguments. Descriptors lower to `memref<?x?xf32>`."
        ),
        "kernel_fn":    kernel.matmul_kernel_2d_grid,
        "SIGNATURE":    _SIG_2D_GRID,
        "constexpr":    ["BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "params":       {
            "M": [256], "K": [64], "N": [128],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "grid":         [4, 8],
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "c_ptr",
        "rtol":         1e-2,
        "extra_checks": lambda t: (
            t.assert_present("linalg.matmul"),
            t.assert_result_type("ktdp.construct_memory_view", "memref<?x?xf32>"),
        ),
    },
    "2d_grid_nonaligned": {
        # M=260: m_blocks=17, not divisible by 4 M-cores → clamp fires.
        "base":   "2d_grid",
        "params": {
            "M": [260], "K": [64], "N": [128],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "extra_checks": lambda t: (
            t.assert_present("linalg.matmul"),
            t.assert_result_type("ktdp.construct_memory_view", "memref<260x64xf32>"),
        ),
    },
    # --- BMM 3D grid variants ---
    "bmm_3d_grid": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "dot", "program-id-3d"],
        "summary": (
            "3D grid BMM: pid_0 distributes B-tiles, pid_1 M-tiles, pid_2 N-tiles, "
            "each with a distribution loop."
        ),
        "doc": (
            "Same batched matmul as `bmm`, but uses a 3D program grid. Each axis "
            "distributes its tiles across cores via a loop: `pid_0` covers B, "
            "`pid_1` covers M, `pid_2` covers N. The K dimension is the innermost "
            "reduction loop."
        ),
        "kernel_fn":    kernel.bmm_matmul_kernel_3d_grid,
        "SIGNATURE":    _SIG_BMM_3D_GRID,
        "constexpr":    ["B", "M", "K", "N", "BLOCK_B", "BLOCK_M", "BLOCK_K", "BLOCK_N"],
        # 3D grid: [2, 4, 4] = 32 cores
        "params":       {
            "B": [4], "M": [64], "K": [32], "N": [64],
            "BLOCK_B": [1], "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "grid":         [2, 4, 4],
        "reference":    run_bmm,
        "inputs":       make_inputs_bmm,
        "output_key":   "c_ptr",
        "rtol":         1e-2,
        "atol":         1e-4,
        "extra_checks": lambda t: (
            t.assert_present("linalg.batch_matmul"),
            t.assert_absent("tt.dot"),
        ),
    },
    "bmm_3d_grid_dynamic": {
        "tags": ["descriptor-load-dynamic", "descriptor-store-dynamic", "dot", "program-id-3d"],
        "summary": (
            "3D grid BMM with runtime `B`, `M`, `K`, `N`: same distribution loop "
            "structure, dynamic descriptor shapes."
        ),
        "doc": (
            "Same as `bmm_3d_grid` but `B`, `M`, `K`, `N` arrive as runtime "
            "`i32` arguments. Descriptors lower to `memref<?x?x?xf32>`."
        ),
        "kernel_fn":    kernel.bmm_matmul_kernel_3d_grid,
        "SIGNATURE":    _SIG_BMM_3D_GRID,
        "constexpr":    ["BLOCK_B", "BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "params":       {
            "B": [4], "M": [64], "K": [32], "N": [64],
            "BLOCK_B": [1], "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "grid":         [2, 4, 4],
        "reference":    run_bmm,
        "inputs":       make_inputs_bmm,
        "output_key":   "c_ptr",
        "rtol":         1e-2,
        "atol":         1e-4,
        "extra_checks": lambda t: (
            t.assert_present("linalg.batch_matmul"),
            t.assert_absent("tt.dot"),
        ),
    },
    "bmm_3d_grid_nonaligned": {
        # B=5: b_blocks=5, not divisible by 2 B-cores → clamp fires on B axis.
        "base":   "bmm_3d_grid",
        "params": {
            "B": [5], "M": [64], "K": [32], "N": [64],
            "BLOCK_B": [1], "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
    },
    # --- BMM addptr variants (disabled: tt.addptr-into-descriptor gap) ---
    # These exercise the per-batch pointer arithmetic pattern
    # (a_ptr + b_idx * stride) feeding tl.make_tensor_descriptor — not yet
    # lowered by LowerDescriptorMemory. Pinned by TestAddptrIntoDescriptor.
    "bmm_addptr": {
        "kernel_fn":    kernel.bmm_matmul_kernel_addptr,
        "SIGNATURE":    _SIG_BMM_ADDPTR,
        "constexpr":    ["B", "M", "K", "N", "BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "params":       {
            "B": [4], "M": [128], "K": [32], "N": [64],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "reference":    run_bmm,
        "inputs":       make_inputs_bmm,
        "output_key":   "c_ptr",
        "disabled": {
            "reason":        "tt.addptr into tt.make_tensor_descriptor not "
                             "yet lowered by LowerDescriptorMemory",
            "tracking_test": "test_lower_desc_memory.py::"
                             "TestAddptrIntoDescriptor",
        },
    },
    "bmm_addptr_dynamic": {
        "kernel_fn":    kernel.bmm_matmul_kernel_addptr,
        "SIGNATURE":    _SIG_BMM_ADDPTR,
        "constexpr":    ["BLOCK_M", "BLOCK_K", "BLOCK_N"],
        "params":       {
            "B": [4], "M": [128], "K": [32], "N": [64],
            "BLOCK_M": [16], "BLOCK_K": [16], "BLOCK_N": [16],
        },
        "reference":    run_bmm,
        "inputs":       make_inputs_bmm,
        "output_key":   "c_ptr",
        "disabled": {
            "reason":        "tt.addptr into tt.make_tensor_descriptor not "
                             "yet lowered by LowerDescriptorMemory",
            "tracking_test": "test_lower_desc_memory.py::"
                             "TestAddptrIntoDescriptor",
        },
    },
}
