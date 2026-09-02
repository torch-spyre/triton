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
  which names another variant in the same `VARIANTS` dict. The `"base"`
  key is consumed at load time and does not appear in the registry entry.
  Circular chains are caught at collection time.
- Registry keys: `<folder>` for the default variant and for
  single-variant kernels; `<folder>__<variant>` for every other entry.
  e.g. `vector_add`, `vector_add__dynamic`.

## Field reference

| Field | Type | Role |
|---|---|---|
| `kernel_fn` | `@triton.jit` function | Compiled on demand via `compile_to_ttir` → `make_ktir_mod`. |
| module-level `SIGNATURE` | `dict[str, str]` | Dtype per `@triton.jit` arg. Pure types — no values. Declared at module scope in `meta.py`, not inside `VARIANTS`. Used by every variant that doesn't redeclare it. |
| variant `SIGNATURE` | `dict[str, str]` | Optional per-variant override. Replaces the module-level map wholesale — use when the variant's kernel has a different arg list (e.g. softmax's `multi_tile` has `BLOCK_N` where `single_tile` has `BLOCK_SIZE`). |
| `base` | `str` | Optional. Name of another variant in the same `VARIANTS` dict to use as the merge base instead of `"default"`. Consumed at load time; not stored in the registry entry. |
| `constexpr` | `list[str]` | Which arg names are Triton constexprs for this variant. Each variant declares the full list explicitly (no partial override over default's list). Values for constexprs come from `params`. |
| `params` | `dict[str \| tuple[str, ...], list[Any]]` | Single source of truth for argument values. A key naming one argument maps to a list of values; more than one is **swept** — one registry entry per value, its key suffixed `[name=value]`. Write a value as `(label, value)` to be named by the label instead, for values whose repr would make a key unreadable (a stick layout); a labelled value is named even when it is the only one. A key that is a **tuple of names** maps to a list of *rows* — one value per name, positionally — and sweeps those names jointly: only the rows written are enumerated. Use it to state a functional dependency, since the `N` a dtype implies sits on the dtype's own row and so cannot disagree with it, and to skip an invalid combination, by not listing its row. A group column is named in the key when it varies across rows or was labelled; a group's row count says nothing about a column held constant. The product runs across keys, and each combination partitions into `constexpr` and runtime the same way. |
| `grid` | `list[int]` | Per-axis partition of the 32-core Spyre grid. One entry per `tl.program_id` axis the kernel reads; `prod(grid)` equals the hardware core count. Defaults to the backend's `(32,)` (1D on all cores) when omitted. |
| `reference` | `(inputs) -> np.ndarray` | NumPy oracle for the numerical test. Omit for structure-only variants. Defined alongside `VARIANTS` in the same `meta.py`. |
| `inputs` | `(**param_values) -> {"arg_name": np.array, ...}` | Pointer/tensor input generator. Called with kwargs matching `params` keys; returns pointer/tensor args only. Runtime scalars (params that aren't in `constexpr`) are merged in by the framework. |
| `output_key` | `str` | Which `inputs` key holds the output buffer compared against `reference(inputs)`. |
| `func_name` | `str` | KTIR function name for `ktir_cpu`. Defaults to `kernel_fn.__name__`. |
| `parallel` | `bool`, default `True` | Set `False` for single-program kernels that do not call `tl.program_id` — skips the DistributeWork-presence check in `TestExample`. |
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
