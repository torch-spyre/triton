# Triton-Spyre

Triton-Spyre is a Triton fork that adds an experimental backend for IBM Spyre.
The backend lowers Triton TTIR into KTIR, so Triton kernels can be used as a
frontend for the downstream Spyre compiler stack.

The main path under development is:

```text
Triton Python kernel -> TTIR -> KTIR -> downstream Spyre compiler stack
```

This repo is based on upstream [triton-lang/triton](https://github.com/triton-lang/triton).

## Status

The Spyre backend is an early public development path. It is useful for compiler
development, KTIR validation, and kernel-lowering experiments. It is not yet a
drop-in replacement for upstream Triton GPU execution.

Current focus:

- building Triton with only the Spyre backend enabled,
- lowering selected TTIR patterns to KTIR,
- validating generated KTIR with both structural and numerical tests,
- tracking remaining gaps with focused tests and pattern references.

Known limitations:

- local execution and benchmarking are not supported by the Spyre driver,
- numerical validation through `ktir-cpu` is a work in progress; some lowering
  patterns are not yet covered and are tracked by focused tests,
- some fixtures intentionally document unsupported lowering patterns,
- the KTIR MLIR bindings (`mlir_ktdp`) are built from this repo's
  `ktir-mlir-frontend` submodule and differ from Triton's own C++ bindings.

## Repository Layout

Key Spyre-specific areas:

- `third_party/spyre/backend/`: Triton backend registration, compiler stages,
  and driver stub.
- `third_party/spyre/include/` and `third_party/spyre/lib/`: KTIR lowering
  passes and C++ bindings exposed through Triton.
- `third_party/spyre/test/`: structural and numerical tests for the Spyre
  lowering path.
- `third_party/spyre/test/fixtures/`: Triton kernel fixtures used by the test
  framework.
- `third_party/spyre/docs/patterns/`: generated pattern reference for supported
  and intentionally unsupported lowering cases.
- `third_party/spyre/ktir-mlir-frontend/`: KTIR MLIR frontend submodule.

## Install

The Spyre backend is the default build target, so a plain `pip install`
produces a Spyre-only Triton out of the box — no environment variables and no
GPU toolchains required. Python 3.12 or newer is recommended.

> **First build downloads LLVM (~900 MB).** A Spyre-only build resolves LLVM
> from a public GitHub Releases asset in `ktir-mlir-frontend` on first use — no
> token required. The download is cached in `~/.cache/ktir-mlir/` so subsequent
> builds are instant. See [LLVM Build Dependency](#llvm-build-dependency) for
> details.

### Install directly from Git

The simplest path. `uv` fetches the repository (submodule included) and builds
it; no manual clone is needed. Use it editable or not:

```bash
# Non-editable
uv pip install "triton[spyre-test] @ git+https://github.com/torch-spyre/triton.git"

# Editable (uv keeps the checkout under its source tree)
uv pip install -e "git+https://github.com/torch-spyre/triton.git#egg=triton[spyre-test]"
```

### Install from a Git checkout

Recommended for development, since submodule and native-build issues are easier
to inspect and debug from a local clone:

```bash
git clone --recurse-submodules https://github.com/torch-spyre/triton.git
cd triton

uv venv .venv --python 3.12 --python-preference managed --seed
export UV_PROJECT_ENVIRONMENT=.venv

uv pip install -e ".[spyre-test]"
```

Setting `UV_PROJECT_ENVIRONMENT` lets every `uv` command target the venv
without activating it; export it in your shell rc to make it persistent.
Activating the venv (`source .venv/bin/activate`) instead works too.

If the repository was cloned without submodules, the install step initializes
the required Spyre submodule for you. Initializing it explicitly makes any
submodule or native-build failure easier to diagnose:

```bash
git submodule update --init --recursive
```

### Backend selection

The Spyre backend is built on its own; it is not combined with the GPU
backends in a single build. To build the inherited upstream GPU backends
instead, set `TRITON_BACKENDS` explicitly (e.g. `TRITON_BACKENDS=nvidia` or
`TRITON_BACKENDS=nvidia,amd`); see the build-variable table below.

The default Spyre-only install:

1. builds only the Spyre backend (the default when `TRITON_BACKENDS` is unset),
2. auto-enables `TRITON_BUILD_TTIR_ONLY=ON`,
3. auto-disables `TRITON_BUILD_PROTON`,
4. initializes the KTIR MLIR frontend submodule,
5. resolves LLVM from the frontend's artifact store (see Pre-Downloading below),
6. builds Triton's C++ extension with the Spyre bindings,
7. installs the local `triton` Python package,
8. installs test dependencies when `[spyre-test]` is requested.

## Faster Rebuilds

For iterative development, install the build dependencies into the venv once
and use `--no-build-isolation`. This avoids rebuilding in a throwaway
environment and makes incremental rebuilds faster and more predictable.

```bash
uv pip install $(uv run python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['build-system']['requires']))")
uv pip install -e ".[spyre-test]" --no-build-isolation
```

Common build variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRITON_BACKENDS` | `spyre` | Backend(s) to build. Spyre builds on its own; set to `nvidia` / `amd` (or `nvidia,amd`) for the inherited GPU backends instead. |
| `TRITON_BUILD_TTIR_ONLY` | auto `ON` when no GPU backends are built | Skip GPU dialects for faster compiler-only builds. |
| `TRITON_BUILD_PROTON` | auto `OFF` when no GPU backends are built | Skip the profiler when it is not needed. |
| `MAX_JOBS` | `2 * cpu_count` | Limit parallel compilation jobs. |
| `TRITON_BUILD_WITH_CCACHE` | `ON` when ccache is available | Enable or disable ccache. |

## Tests

Run the full Spyre validation suite with:

```bash
uv run pytest third_party/spyre/test -s --tb=short
```

The suite contains both structural lowering tests and numerical checks through
`ktir-cpu`. Numerical coverage is a work in progress; a few lowering patterns
are not yet supported and are tracked by focused tests. To run only the
structural lowering tests:

```bash
uv run pytest third_party/spyre/test -s --tb=short -k "not numerical"
```

Useful narrower commands:

```bash
uv run pytest third_party/spyre/test/test_lower_desc_memory.py -s --tb=short
uv run pytest third_party/spyre/test/test_lower_compute_ops.py -s --tb=short
uv run pytest third_party/spyre/test/test_distribute_work.py -s --tb=short
uv run pytest third_party/spyre/test/test_ktir_examples.py -s --tb=short
```

## KTIR CPU Dependency

The optional `spyre-test` extra installs `ktir-cpu` from
`torch-spyre/ktir-cpu@main`. It provides the numerical interpreter used by the
Spyre test suite. Treat it as a development dependency rather than part of a
stable user-facing package contract.

It is installed **without** its `[mlir-frontend]` extra on purpose: that extra
would pin the `ktir-mlir-frontend` to `ktir-cpu`'s own commit, which differs
from this repo's `third_party/spyre/ktir-mlir-frontend` submodule. The
numerical tests need the `mlir_ktdp` bindings (`MLIRFrontendParser`) built from
*our* submodule so they match the lowering under test — see the parser note in
`third_party/spyre/test/conftest.py`.

## LLVM Build Dependency

A Spyre-only build does not use upstream Triton's prebuilt LLVM blob. Instead,
`setup.py` resolves LLVM from the `ktir-mlir-frontend`'s artifact store by
running `third_party/spyre/ktir-mlir-frontend/scripts/setup_mlir.py`, which
reads the pinned hash from `cmake/llvm-hash-spyre.txt` and fetches the matching
build from `torch-spyre/ktir-mlir-frontend`. The resolved path is passed to
CMake as `LLVM_SYSPATH`.

To point the build at a prebuilt LLVM and skip this step entirely, set
`LLVM_SYSPATH` yourself — `setup.py` only runs `setup_mlir.py` when
`LLVM_SYSPATH` is unset:

```bash
export LLVM_SYSPATH=/path/to/llvm
```

The pinned LLVM hash lives in `cmake/llvm-hash-spyre.txt`.

To point the build at a prebuilt nlohmann/json include tree, set `JSON_SYSPATH`
(inherited from upstream Triton):

```bash
export JSON_SYSPATH=/path/to/json/include
```

Note: upstream's `TRITON_OFFLINE_BUILD` does **not** make a Spyre build fully
offline — `setup_mlir.py` resolves LLVM from the `ktir-mlir-frontend` artifact
store independently of that flag. To build without network access, pre-place
LLVM and set `LLVM_SYSPATH` (and `JSON_SYSPATH`) yourself.

## Upstream Triton Changes

Most Spyre-specific additions live under `third_party/spyre/`. A small number
of upstream Triton files are modified to make the in-tree Spyre backend build
cleanly. These changes are guarded so the inherited NVIDIA and AMD GPU paths
keep building unchanged when those backends are selected.

Search for these markers when rebasing or auditing local changes:

```text
# --- START --- added for spyre
# --- END --- added for spyre
# --- added for spyre
```

Current upstream-file touch points include:

| File | Spyre-specific change |
| --- | --- |
| `setup.py` | Default `TRITON_BACKENDS` to `spyre`; auto TTIR-only / Proton defaults; resolve LLVM via `setup_mlir.py`; Spyre-only package discovery; `spyre-test` extra. |
| `CMakeLists.txt` | Guard GPU dialect / blob logic behind the TTIR-only build. |
| `python/src/main.cc` | Register empty `gluon_ir` / `linear_layout` pybind modules so `import triton` works in TTIR-only builds. |
| `python/triton/experimental/gluon/__init__.py`, `.../language/__init__.py` | Guard GPU-only arch shim imports absent from a Spyre-only wheel. |
| `include/triton/Dialect/Triton/IR/Dialect.h`, `lib/Target/LLVMIR/LLVMDIUtils.cpp` | Source compatibility with the Spyre LLVM pin (`cmake/llvm-hash-spyre.txt`). |

## Related Documentation

- `third_party/spyre/docs/patterns/index.md`: generated KTIR lowering pattern
  reference.
- `third_party/spyre/docs/ttir_only_build.md`: details of the TTIR-only build
  mode used by Spyre-only builds.
- `third_party/spyre/test/fixtures/README.md`: fixture framework for kernel
  examples and per-variant expectations.
- `third_party/spyre/ktir-mlir-frontend/README.md`: KTIR MLIR frontend
  submodule documentation.

## License

This fork retains Triton's MIT license at the repository root. The KTIR MLIR
frontend submodule has its own Apache-2.0 license. See the license files in the
root repository and submodule for details.
