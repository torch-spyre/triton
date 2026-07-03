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

- **First build needs `GIT_PAT`.** A Spyre-only build resolves LLVM from the
  `ktir-mlir-frontend` artifact store via `setup_mlir.py`, which currently
  needs a GitHub token: `export GIT_PAT=<token>`. The failure message when it's
  missing is *not* obvious (it surfaces during LLVM fetch). Tracked in
  ktir-mlir-frontend#24.
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

- `third_party/spyre/backend/compiler.py` — `SpyreBackend`; `add_stages()`
  defines the `ttir` and `ktir` stages. `_make_ktir` runs the KTDP passes.
- `third_party/spyre/lib/Dialect/KTDP/Transforms/` — the lowering passes
  (C++). See the spyre / spyre-ktir agents for the pass pipeline.
- `third_party/spyre/include/Dialect/KTDP/Transforms/Passes.td` — authoritative
  per-pass contracts (input/lowering/output).
- `third_party/spyre/test/` — structural + numerical tests; `fixtures/` holds
  the kernel examples (vector_add, softmax, matmul, gather).
- `third_party/spyre/ktir-mlir-frontend/` — KTIR MLIR frontend submodule
  (provides the `mlir_ktdp` bindings; supplies LLVM).

## Tests

```bash
uv run pytest third_party/spyre/test                    # full suite
uv run pytest third_party/spyre/test -k "not numerical" # structural only
```

Numerical coverage is a work in progress; known gaps are strict-xfail'd and
missing oracles skip, so the suite stays green while catching regressions.

## Documentation conventions

- **Use GitHub-compatible LaTeX math.** GitHub renders `$...$` (inline) and
  `$$...$$` (display block) via KaTeX. Stick to the KaTeX subset it supports:
  - **Use**: `\lfloor`, `\rfloor`, `\lceil`, `\rceil`, `\cdot`, `\times`,
    `\mod`, `\div`, `^`, `_`, `\frac`, `\sum`, `\prod`, `\in`, `\to`,
    `\mathbb`, `\mathbf`, `\text{short label}`, `\begin{aligned}...\end{aligned}`.
  - **Avoid** (renders as raw text or breaks layout on GitHub): `\;` `\,` `\!`
    spacing hacks, `\quad` / `\qquad`, `\bmod` (use `\mod` instead),
    `\operatorname{}`, `\DeclareMathOperator`, `\newcommand`, multi-line `align`
    without `aligned`, and bare `\\` outside an environment.
