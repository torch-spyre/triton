# Triton + Spyre Backend

This is a Triton fork that adds an experimental **Spyre** backend lowering
Triton TTIR → KTIR (KTDP dialect). The Spyre backend is the **default** build
target. Full user-facing docs live in `README.md`; this file captures what
you need to work in the repo effectively.

## Install / build (quick reference)

A plain install builds Spyre-only — no `TRITON_BACKENDS` needed:

```bash
uv pip install -e ".[spyre-test]"            # editable + test deps
bash install-ktdp-mlir-bindings.sh           # install KTDP MLIR bindings (generated above)
uv run pytest third_party/spyre/test         # run the suite
```

The two-step install is intentional: `mlir_ktdp` (the KTIR MLIR Python bindings)
must be compiled against Triton's own MLIR to avoid duplicate-MLIR-global-state
crashes. `setup.py` generates `install-ktdp-mlir-bindings.sh` with the correct
`MLIR_DIR` baked in during the first step.

Faster iterative rebuilds (install build deps once, then skip isolation):

```bash
uv pip install $(uv run python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['build-system']['requires']))")
uv pip install -e ".[spyre-test]" --no-build-isolation
bash install-ktdp-mlir-bindings.sh
```

Use `UV_PROJECT_ENVIRONMENT=/path/to/venv` (Python 3.12+) so `uv` targets your
venv without activating it.

### Documented but may not be immediately obvious

- **No `GIT_PAT` needed.** A Spyre-only build resolves LLVM from the
  `ktir-mlir-frontend` artifact store via `setup_mlir.py`, which downloads the
  pinned LLVM build from a public GitHub Releases asset (no auth required).
  `GIT_PAT`/`GITHUB_TOKEN` is only consulted as a fallback for the
  token-gated Actions-artifact path, which should not be needed in normal use.
- **LLVM is NOT the upstream Triton blob.** `setup.py` runs
  `third_party/spyre/ktir-mlir-frontend/scripts/setup_mlir.py`, reading the pin
  from `cmake/llvm-hash-spyre.txt` (not `cmake/llvm-hash.txt`). It only runs
  when `LLVM_SYSPATH` is unset — set `LLVM_SYSPATH` to point at a prebuilt LLVM
  and skip the fetch.
- **`TRITON_OFFLINE_BUILD` does not make a Spyre build offline** — `setup_mlir.py`
  fetches LLVM independently of that flag. For offline, pre-place LLVM and set
  `LLVM_SYSPATH` (and `JSON_SYSPATH`).
- Build the inherited GPU backends instead with `TRITON_BACKENDS=nvidia` (or
  `nvidia,amd`). Spyre is not combined with GPU backends in one build.

### Build variables

| Variable | Default (Spyre-only) | Purpose |
|----------|---------------------|---------|
| `TRITON_BACKENDS` | `spyre` | Backends to build; set to `nvidia`/`amd` for GPU paths |
| `TRITON_BUILD_TTIR_ONLY` | auto `ON` when no GPU backends | Skip GPU dialects (faster) |
| `TRITON_BUILD_PROTON` | auto `OFF` when no GPU backends | Skip profiler |
| `MAX_JOBS` | 2 * cpu_count | Parallel compile jobs |
| `TRITON_BUILD_WITH_CCACHE` | `ON` when ccache available | Use ccache |
| `LLVM_SYSPATH` / `JSON_SYSPATH` | unset | Point at prebuilt LLVM / json includes |

## Pinned dependencies

- **LLVM**: `cmake/llvm-hash-spyre.txt`, fetched from
  `torch-spyre/ktir-mlir-frontend`'s artifact store by `setup_mlir.py`.
- **ktir-cpu**: pinned in `setup.py` under `extras_require["spyre-test"]` to
  `git+https://github.com/torch-spyre/ktir-cpu@main`. It is the numerical
  interpreter for the test suite (treat as a dev dependency, not a stable
  contract). Installed **without** its `[mlir-frontend]` extra on purpose —
  that extra would pin `ktir-mlir-frontend` to ktir-cpu's own commit, which
  differs from this repo's `third_party/spyre/ktir-mlir-frontend` submodule.
  The numerical tests need `mlir_ktdp` (`MLIRFrontendParser`) built from *our*
  submodule so it matches the lowering under test (see the parser note in
  `third_party/spyre/test/conftest.py`).

To refresh ktir-cpu to the latest `main`:

```bash
uv pip install -e ".[spyre-test]" --force-reinstall --no-deps
```

## Upstream modifications

Changes to upstream Triton files are guarded so the inherited GPU paths still
build when those backends are selected. Search markers when rebasing/auditing:

```
# --- START --- added for spyre
# --- END --- added for spyre
# --- added for spyre
```

Two guard mechanisms, picked by *when* the code runs:

- **Compile-time (C++)** — `#ifdef TRITON_BUILD_TTIR_ONLY` (auto-defined for the
  Spyre build). Use for anything in the C++ library: dialect verifiers, LLVM-pin
  shims. See `Dialect.h`, `LLVMDIUtils.cpp`, `Ops.cpp`.
- **Runtime (Python frontend)** — `TRITON_BUILD_TTIR_ONLY` is meaningless in the
  frontend, which runs op construction before any pass. Detect the backend at
  runtime via `triton.language.target_info` (next to `is_cuda`/`is_hip`):
  `is_spyre()` for behavior that *forks* by backend
  (`if is_spyre(): <relaxed> else: <strict>`), and `@requires_backend("spyre")`
  for ops that *only exist* for Spyre (raises on any other backend; resolves at
  call time). Prefer these over open-coding
  `driver.active.get_current_target().backend`.

Current upstream touch points:

| File | What was added |
|------|---------------|
| `setup.py` | Default `TRITON_BACKENDS=spyre`; auto TTIR-only/Proton; resolve LLVM via `setup_mlir.py`; Spyre-only package discovery; `spyre-test` extra; `--recursive` submodule init |
| `CMakeLists.txt` | Guard GPU dialect / blob logic behind the TTIR-only build |
| `python/src/main.cc` | Register empty `gluon_ir` / `linear_layout` pybind modules so `import triton` works in TTIR-only builds |
| `python/triton/experimental/gluon/__init__.py`, `.../language/__init__.py` | Guard GPU-only arch shim imports absent from a Spyre-only wheel |
| `include/triton/Dialect/Triton/IR/Dialect.h`, `lib/Target/LLVMIR/LLVMDIUtils.cpp` | Source compatibility with the Spyre LLVM pin |
| `python/triton/language/target_info.py` | Runtime frontend backend guards: `is_spyre()` predicate + `requires_backend()` decorator |

## Where the Spyre code lives

- `third_party/spyre/backend/compiler.py` — `SpyreBackend`. `add_stages()`
  registers **three** stages, and which one you are looking at is most of
  debugging this backend:

  | Stage | In → out | Method | Notes |
  |-------|----------|--------|-------|
  | `ttir` | Triton IR → Triton IR | `_make_ttir` | The standard upstream optimization passes. |
  | `ktir` | Triton IR → KTIR | `_make_ktir` | The lowering. Also reads `metadata["name"]` and infers the base addresses, both of which have to happen before `ConvertFunctions` retypes the entry point. |
  | `spyrecode` | KTIR → a loadable binary | `_make_spyrecode` | A KTIR → KTIR round trip that shapes the module for the device, then `dbo-opt`. Returns the export directory as ZIP bytes. |

  The first two are pure IR-to-IR; only `spyrecode` shells out. Each artifact is
  cached under the stage name and reachable as `asm["<stage>"]` on a
  `CompiledKernel`.

  **The two IR pipelines are registered MLIR pass pipelines**, built once in C++
  (`lib/Pipeline.cpp`) and reachable as `spyre-triton-opt --spyre-ttir-to-ktir`
  and `--spyre-prepare-spyrecode`. So a lit test can drive a whole stage rather
  than one pass (`test/Pipeline/stage-pipelines.mlir`), and the module `dbo-opt`
  receives can be reproduced by hand — see below.
- The C++ passes live in three libraries, split by what each pass's subject is.
  See the spyre / spyre-ktir agents for the pass pipeline.
  - `third_party/spyre/lib/Conversion/TritonToKTIR/` — passes that cross a
    dialect boundary, taking `tt` into KTIR (`ktdp`, linalg, tensor, func,
    spyreop). `ktdp` is one dialect; KTIR is the language it composes with.
  - `third_party/spyre/lib/Dialect/KTDP/Transforms/` — passes whose subject is
    KTDP's own abstractions: memory views, access tiles.
  - `third_party/spyre/lib/Transforms/` — passes acting on upstream structure or
    the whole program, not on a dialect's own abstractions.
  - `third_party/spyre/lib/Dialect/KTDP/Utils/` — shared helpers; kept out of the
    peer libraries so neither has to depend on the other.
- `Passes.td` under the matching `include/` directory for each of the three —
  authoritative per-pass contracts (input/lowering/output). Note a pass's
  `dependentDialects` there is a second source of link dependencies alongside
  its `#include`s.
- `third_party/spyre/test/` — structural + numerical tests. The lit tree mirrors
  `lib/`; `fixtures/` holds the kernel examples (vector_add, softmax, matmul,
  gather).
- `third_party/spyre/ktir-mlir-frontend/` — KTIR MLIR frontend submodule
  (provides the `mlir_ktdp` bindings; supplies LLVM).

## Seeing what the compiler did

Three environment variables, and they are the whole story — **two of the three are
upstream Triton's, not ours**, so do not go looking for a Spyre-specific dump flag.

| Variable | Whose | Gives you |
|----------|-------|-----------|
| `MLIR_ENABLE_DUMP` | upstream | The module before each pass, all three stages |
| `TRITON_KERNEL_DUMP` + `TRITON_DUMP_DIR` | upstream | One artifact file per stage |
| `TRITON_SPYRE_DBO_DEBUG` | ours | `dbo-opt`'s own tree, inside the archive. On by default |

**Between passes.** All three stages, since `_make_ttir` / `_make_ktir` /
`_make_spyrecode` each call `pm.enable_debug()` — one call per pass manager, and a
new pass manager needs its own or it is silently quiet.

```bash
MLIR_ENABLE_DUMP=1 python kernel.py           # every pass, every kernel
MLIR_ENABLE_DUMP=my_kernel python kernel.py   # a function name narrows it
```

You get **"before" only** on a successful run: upstream passes
`printAfterOnlyOnFailure=true`. An "IR Dump After" line therefore means *that pass
failed* — read it as a diagnostic, not as ordinary output.

**Per-stage artifacts.** `TRITON_ALWAYS_COMPILE=1` is not optional: a cache hit
returns before any stage runs, so on a warm cache the dump directory stays empty
and nothing says why.

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump TRITON_ALWAYS_COMPILE=1 python kernel.py
```

**The module `dbo-opt` was handed.** Already recorded, on every compile: `dbo-opt`
writes a `debug/` tree that `_make_spyrecode` packs into the archive, and its first
entry *is* that module. Unpack `asm["spyrecode"]` (a ZIP) and read it — which is
why no flag of ours exists for this.

**Reproducing that module without the tool.** Run the two registered pipelines in
sequence, with the options the compile recorded in its `metadata` (grid, data
layout, base addresses):

```bash
spyre-triton-opt kernel.ttir \
  --spyre-ttir-to-ktir="grid=32" \
  --spyre-prepare-spyrecode="bind-base-addresses base-addresses=0,4294967296"
```

Use the **dumped** `kernel.ttir`, i.e. the `ttir` stage's *output*.
`--spyre-ttir-to-ktir` does not inline, so raw `ASTSource.make_ir` output fails on
a surviving `tt.call` from any `tl.*` helper. There is no flag for the `ttir` stage
because it is upstream Triton's passes. Omit `bind-base-addresses` for the
symbolic mode, which is what a launch uses.

## Tests

```bash
uv run pytest third_party/spyre/test                    # full suite
uv run pytest third_party/spyre/test -k "not numerical" # structural only
uv run lit build/cmake.*/third_party/spyre/test -v      # lit/FileCheck tests

# fast loop on one python-driven lit test -- no cmake, no re-configure
T=third_party/spyre/test/python/segment-addresses.py
PYTHONPATH=python:third_party/spyre uv run python $T | ./python/triton/FileCheck $T
```

Numerical coverage is a work in progress; known gaps are strict-xfail'd and
missing oracles skip, so the suite stays green while catching regressions.

`test_device_launch.py` launches on hardware, in the pytest process itself, and a
Spyre device admits one opener for that process's whole lifetime. It is in the
pytest suite rather than under lit precisely because pytest runs one process
sequentially — that is what serializes device access. 

### Lit tests and `spyre-triton-opt`

`spyre-triton-opt` registers both Triton (TTIR) and KTDP dialects/passes.
It lives in `third_party/spyre/bin/` and is built by default. Lit tests
live in `third_party/spyre/test/` alongside the pytest suite: `.mlir` for IR, and
`.py` under `test/python/` for what is Python rather than IR (the address policy,
the option surface, the `spyrecode` stage, the driver surface).
`test/python/lit.local.cfg` adds the suffix and sets the `PYTHONPATH` those need,
since lit does not load `conftest.py`.

**Tests needing `dbo-opt`.** `test/lit.cfg.py` forwards `TRITON_SPYRE_DBO_OPT` and
`TRITON_SPYRE_DEVICE` (they are not in lit's default whitelist) and defines a
`dbo-opt` feature; `spyrecode-compile-test.py` requires it. So a machine without the
tool reports `Unsupported`, which is *visible* — unlike a `pytest.skip`, which would
be invisible because pytest exits 0 when it skips and lit would call that `Passed`.

That feature re-spells the rule in `resolve_dbo_opt()` rather than importing it
(which would pull triton into lit's config phase): a value with a path separator is
literal, a bare name goes through `PATH`. Keep the two in step. And note lit scans a
test file's *whole* text for directives, so writing `REQUIRES` followed by a colon in
a docstring creates a second, malformed one and the test comes out `Unresolved`.

**Generating FileCheck patterns** — use `utils/generate-test-checks.py`
(from upstream LLVM) to auto-generate CHECK lines from printed IR:

```bash
# Roundtrip test (parse → print → parse → print → FileCheck):
spyre-triton-opt foo.mlir | python utils/generate-test-checks.py --source foo.mlir -i

# Pass output test (e.g. lower-compute-ops):
spyre-triton-opt foo.mlir --lower-compute-ops | python utils/generate-test-checks.py --source foo.mlir -i
```

The `-i` flag edits the source file in-place, inserting CHECK lines above
each function. The RUN line must already be present in the file before running
the script. For `-verify-diagnostics` tests (error checking), write
`expected-error` annotations manually — there is no auto-generator for those.
