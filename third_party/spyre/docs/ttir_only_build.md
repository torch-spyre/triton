# TRITON_BUILD_TTIR_ONLY

The `TRITON_BUILD_TTIR_ONLY` CMake flag builds libtriton with only the TritonIR
dialect, skipping all GPU-specific code. This is useful for the Spyre backend
which only lowers TritonIR → KTDP and never touches GPU dialects.

## Motivation

A full Triton build compiles TritonGPU, Gluon, Instrumentation, NVIDIA, and AMD
dialects plus their LLVM/MLIR dependencies. When working with git worktrees that
share a single venv, each worktree needs its own `libtriton.so` — and the full
build is expensive. This flag cuts the build significantly by excluding everything
Spyre doesn't need.

## Build time

Clean build from scratch (no ccache, `MAX_JOBS=4`, Apple M-series):

| Build | Source files | Wall time |
|-------|-------------|-----------|
| TTIR-only | 45 | ~46s |
| Full (all dialects) | 200+ | ~4–5 min |

## Usage

```bash
TRITON_BACKENDS=spyre \
TRITON_BUILD_PROTON=OFF \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_GSAN=OFF -DTRITON_BUILD_TTIR_ONLY=ON" \
MAX_JOBS=4 \
uv run python setup.py build_ext --inplace 2>&1 | tee /tmp/triton-build.log
rm -rf python/triton.egg-info
```

## What gets skipped

| Component | Path | Status | Notes |
|-----------|------|--------|-------|
| TritonIR dialect | `lib/Dialect/Triton/` | ✅ Built | Core dialect — Spyre's TTIR→KTDP lowering depends on it |
| TritonIR transforms | `lib/Dialect/Triton/Transforms/` | ✅ Built | Combine, loop unroll, CSE, etc. used by the Spyre pass pipeline |
| TritonGPU dialect | `lib/Dialect/TritonGPU/` | ❌ Skipped | |
| TritonNvidiaGPU dialect | `lib/Dialect/TritonNvidiaGPU/` | ❌ Skipped | |
| TritonInstrument dialect | `lib/Dialect/TritonInstrument/` | ❌ Skipped | |
| Gluon dialect | `lib/Dialect/Gluon/` | ❌ Skipped | |
| GPU conversions | `lib/Conversion/` | ❌ Skipped | |
| Analysis | `lib/Analysis/` | ❌ Skipped | Depends on TritonGPU types (`MemDescType`) |
| LLVM IR target | `lib/Target/LLVMIR/` | ✅ Built | Provides `BreakStructPhiNodesPass` and DI passes needed by `llvm.cc`; GPU link deps (`TritonGPUToLLVM`, NVVM, ROCDL) are stripped |
| Tools | `lib/Tools/` | ✅ Built | `PluginUtils`, `LayoutUtils`, etc. — no GPU dependencies |
| NVIDIA backend | `third_party/nvidia/` | ❌ Skipped | Filtered from `TRITON_CODEGEN_BACKENDS` |
| AMD backend | `third_party/amd/` | ❌ Skipped | Filtered from `TRITON_CODEGEN_BACKENDS` |
| Spyre backend | `third_party/spyre/` | ✅ Built | The whole point — `TritonToKTIR` / `KTDPTransforms` / `SpyreTransforms` and the `TritonSpyre` plugin |
| Proton dialect | `third_party/proton/Dialect/` | ❌ Skipped | |
| CLI tools (`triton-opt`, etc.) | `bin/` | ❌ Skipped | `RegisterTritonDialects.h` pulls in all GPU dialects |
| Example plugins | `examples/plugins/` | ❌ Skipped | |
| C++ unit tests | `unittest/` | ❌ Skipped | |
| Lit tests | `test/` | ❌ Skipped | |
| `gluon_ir.cc` bindings | `python/src/gluon_ir.cc` | ❌ Skipped | Entirely GPU-specific pybind11 wrappers |
| `linear_layout.cc` bindings | `python/src/linear_layout.cc` | ❌ Skipped | Python bindings for LinearLayout; references `TritonGPU` layout attributes |
| LLVM GPU codegen libs | `LLVMNVPTXCodeGen`, `LLVMAMDGPUCodeGen` | ❌ Skipped | |
| MLIR GPU dialect libs | `MLIRNVVMDialect`, `MLIRGPUDialect`, etc. | ❌ Skipped | |

## Files modified to support this flag

### CMake files

| File | Conditionals | What is changed |
|------|-------------|-----------------|
| `CMakeLists.txt` (root) | 8 | Defines option + compile definition, filters nvidia/amd from backends, skips proton dialect, excludes GPU MLIR/LLVM link libs, excludes `gluon_ir.cc`/`linear_layout.cc` from sources, skips example plugins, skips `bin/`/`test/`/`unittest/` |
| `lib/CMakeLists.txt` | 1 | Skips `Analysis/` |
| `lib/Dialect/CMakeLists.txt` | 1 | Skips TritonGPU, TritonNvidiaGPU, TritonInstrument, Gluon subdirectories |
| `lib/Conversion/CMakeLists.txt` | 1 | Skips TritonToTritonGPU, TritonGPUToLLVM, TritonInstrumentToLLVM |
| `lib/Target/LLVMIR/CMakeLists.txt` | 1 | Strips GPU link deps (`TritonGPUToLLVM`, `MLIRNVVMToLLVM`, `MLIRNVVMToLLVMIRTranslation`, `MLIRROCDLToLLVMIRTranslation`) |
| `include/triton/Dialect/CMakeLists.txt` | 1 | Skips GPU/Gluon/Instrument tablegen targets |
| `include/triton/Conversion/CMakeLists.txt` | 1 | Skips GPU conversion tablegen targets |

### C++ files (`#ifndef TRITON_BUILD_TTIR_ONLY` guards)

| File | Guards | What is guarded |
|------|--------|-----------------|
| `python/src/passes.cc` | 8 | GPU pass includes, `init_triton_analysis`, `init_triton_passes_ttgpuir`, `init_gluon_passes`, `add_convert_to_ttgpuir` wrapper, `add_nvvm_to_llvm` wrapper |
| `python/src/ir.cc` | 7 | GPU dialect includes/aliases, `getTensorDescMetadata`, GPU dialect registry insertions in `load_dialects`, `create_barrier` binding |
| `python/src/main.cc` | 2 | `init_gluon_ir` and `init_linear_layout` declarations and calls |
| `python/src/llvm.cc` | 1 | `InitializeNativeTarget` instead of `InitializeAllTargets` (avoids NVPTX/AMDGPU codegen libs) |

### Python files

- **`python/triton/tools/__init__.py`** — `try/except` guard around
  `LinearLayout` import (the C++ module is excluded from the TTIR-only build)
