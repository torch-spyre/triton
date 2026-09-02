# Spyre kernel test fixtures

Per-kernel folder structure for Triton kernels that exercise the Spyre
TTIR → KTIR pipeline in the test suite.

## Layout

```
<name>/
  kernel.py       # @triton.jit functions
  meta.py         # SIGNATURE + VARIANTS + reference oracle + input generator
  README.md       # (optional) what this kernel exercises
```

`test/conftest.py::_load_examples` globs `test/fixtures/*/meta.py`,
imports each as a package-qualified module (so `from . import kernel`
works), and expands each `VARIANTS` dict into registry entries. The
`TestExample` class in `test/test_ktir_examples.py` parametrizes over
every discovered variant.

Each folder holds one **mathematical function**. Different
implementations of the same function (algorithms, shape flavors,
static-vs-dynamic) are variants in one `meta.py`; they share the
reference oracle and input generator. Different functions
(`softmax` vs `log_softmax`) go in sibling folders.

## `VARIANTS` discovery rules

- `"default"` is the full META dict.
- Other keys are **shallow-merge deltas** over a base variant — every
  key the variant omits inherits wholesale from the base. There is no
  partial-override rule for nested fields; a variant that wants to
  change `constexpr` (or `params`) replaces the whole list / dict.
- The base is `"default"` unless the variant declares `"base": "<name>"`,
  which names another variant in the same `VARIANTS` dict, or
  `"base": None`, which opts out of the fallback entirely (the variant's
  dict is used as-is, with no merge). `"base"` is consumed at load time
  and does not appear in the registry entry (`"base": None` is the one
  exception — see the field reference below). Circular chains are caught
  at collection time.
- The `"default"`-fallback is implicit and applies to *every* variant that
  omits `"base"`, not just ones that look like they want to inherit
  something. If `"default"` sets a field like `parallel: False` with a
  `summary` justifying it, every sibling variant silently inherits that
  justification too unless it sets its own or opts out with
  `"base": None`. Prefer keeping `"default"` a representative, unexotic
  case for exactly this reason — a `"default"` that is itself an edge
  case makes every accidental non-opt-out sibling misrepresent itself.
- Registry keys: `<folder>` for the default variant and for
  single-variant kernels; `<folder>__<variant>` for every other entry.
  e.g. `vector_add`, `vector_add__dynamic`.

## Field reference

| Field | Type | Role |
|---|---|---|
| `kernel_fn` | `@triton.jit` function | Compiled on demand via `compile_to_ttir` → `make_ktir_mod`. |
| module-level `SIGNATURE` | `dict[str, str]` | Dtype per `@triton.jit` arg. Pure types — no values. Declared at module scope in `meta.py`, not inside `VARIANTS`. Used by every variant that doesn't redeclare it. |
| variant `SIGNATURE` | `dict[str, str]` | Optional per-variant override. Replaces the module-level map wholesale — use when the variant's kernel has a different arg list (e.g. softmax's `multi_tile` has `BLOCK_N` where `single_tile` has `BLOCK_SIZE`). |
| `base` | `str \| None` | Optional. Name of another variant in the same `VARIANTS` dict to use as the merge base instead of `"default"`. Consumed at load time; not stored in the registry entry. `None` opts out of the implicit `"default"`-fallback entirely — use this when the variant must not inherit something `"default"` sets (e.g. a `SIGNATURE` it doesn't share, or a `parallel`/`summary` pair that wouldn't apply to it). Unlike a named base, an explicit `"base": None` currently *does* survive into the resolved registry entry as a stray key (a minor asymmetry in the loader — harmless since nothing reads it back). |
| `constexpr` | `list[str]` | Which arg names are Triton constexprs for this variant. Each variant declares the full list explicitly (no partial override over default's list). Values for constexprs come from `params`. |
| `params` | `dict[str, list[Any]]` | Single source of truth for argument values. Lists today carry one element each; future Cartesian expansion (one registry entry per product) is deferred — when it lands, the `constexpr` vs runtime partition stays the same per expansion. |
| `grid` | `list[int]` | Per-axis partition of the 32-core Spyre grid. One entry per `tl.program_id` axis the kernel reads; `prod(grid)` equals the hardware core count. Defaults to the backend's `(32,)` (1D on all cores) when omitted. |
| `reference` | `(inputs) -> np.ndarray` | NumPy oracle for the numerical test. Omit for structure-only variants. Defined alongside `VARIANTS` in the same `meta.py`. |
| `inputs` | `(**param_values) -> {"arg_name": np.array, ...}` | Pointer/tensor input generator. Called with kwargs matching `params` keys; returns pointer/tensor args only. Runtime scalars (params that aren't in `constexpr`) are merged in by the framework. |
| `output_key` | `str` | Which `inputs` key holds the output buffer compared against `reference(inputs)`. |
| `func_name` | `str` | KTIR function name for `ktir_cpu`. Defaults to `kernel_fn.__name__`. |
| `parallel` | `bool`, default `True` | Set `False` for single-program kernels that do not call `tl.program_id` — skips the DistributeWork-presence check in `TestExample`. When `False`, also set `summary` (below) to justify it — and remember `"default"`-fallback means every sibling without its own `"base"` inherits both fields together unless it opts out. |
| `tags` | `list[str]` | Free-form categorization strings (e.g. `"descriptor-gather"`, `"1core"`). Consumed by `scripts/gen_patterns_docs.py` to group variants into the generated `docs/patterns/*.md` — see that script's `--check` mode, which CI runs to catch stale generated docs. |
| `summary` | `str` | Human-readable justification, surfaced in the `pytest.skip` reason when `parallel: False` (see `test_work_distribution` in `test_ktir_examples.py`). Explains *why* the variant is legitimately single-program rather than leaving that to be inferred. Because of the `"default"`-fallback rule above, a `summary` set on `"default"` is inherited by every sibling that doesn't set its own — which is exactly the failure mode that motivated documenting `"base": None` as an opt-out. |
| `extra_checks` | `(tester) -> None` | Optional. Runs alongside the shared structural suite for variant-specific assertions (e.g. `memref<?x` only in the dynamic variant). |
| `factory` | `VariantFactory` | Optional. Supplies the fields that vary with the swept `params` combination. Subclass `VariantFactory` (`test/conftest.py`) and override `signature()`, `reference()` or `inputs()`; each is called per combination with the combination as kwargs and returns that field's value, or `None` to leave it unset. Declaring a hook and the literal field it produces on one variant is an error. |
| `xfail_numerical` | `str \| dict` | Optional. `str` is shorthand for `{"reason": str, "strict": True}`; `dict` is forwarded to `pytest.mark.xfail(**d)` (so `raises=ValueError` etc. work). Attached at collection time so failures show as `XFAIL`, not `SKIP`. Use this when the kernel compiles but the numerical comparison fails (e.g. `ktir_cpu` can't parse a dynamic memref shape). |
| `disabled` | `dict` | Optional. `{"reason": str, "tracking_test": "file.py::ClassName"}`. Marks a variant as unable to compile through the TTIR→KTIR pipeline today. Every structural and numerical test skips with `reason`. `tracking_test` points at the single-pass test that pins the underlying gap (e.g. a `test_lower_desc_memory.py` class asserting the expected verification failure). The meta-test `test_disabled_variants_tracking_tests_exist` fails if `tracking_test` no longer resolves, so a closed gap can't leave a stale `disabled` block behind. Use this instead of `xfail` when the kernel does not yet compile — it keeps the failure documented in one place (the tracking test) rather than duplicated across every structural test. |

## Test groups in `TestExample`

`TestExample` (in `test/test_ktir_examples.py`) splits its methods into
three categories. See the class docstring for details.

1. **Pipeline invariants** — kernel-agnostic KTIR properties (no `tt.*`
   ops, `ktdp.*` ops present, memref types, DistributeWork ran). Runs
   uniformly over every variant.
2. **Per-variant structural hook** — `test_extra_checks` calls the
   variant's `extra_checks` callable.
3. **Numerical** — `test_numerical` runs the kernel on `ktir_cpu` and
   compares to the NumPy oracle.
