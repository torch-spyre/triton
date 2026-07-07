# Sweep grammar design

## Problem

Every `params` value is currently restricted to a 1-element list.  Adding
coverage for a second shape (e.g. `M=520` alongside `M=512`) requires a
new named variant with a duplicated params dict.  That is the same
grep-and-patch problem that `"base"` solved for the `constexpr` list.

## Proposed grammar

A `params` value may be a list with more than one element:

```python
"params": {
    "M":       [512, 520],   # two values → two registry entries
    "K":       [64],
    "N":       [256],
    "BLOCK_M": [16],
    "BLOCK_K": [16],
    "BLOCK_N": [16],
}
```

`_load_examples` expands multi-value lists into the full Cartesian product
and emits one registry entry per combination.

## Registry key naming

Single-element params (no expansion) keep the existing key format:
`<folder>` or `<folder>__<variant>`.

For expanded entries the suffix `[<k>=<v>, ...]` is appended, listing
only the params that have more than one value in the variant's own
`params` dict (not inherited ones):

```
matmul                         # M=[512] only — no suffix
matmul__dynamic[M=512]         # M=[512, 520] — suffix even for first value
matmul__dynamic[M=520]
```

The bracket suffix uses only the swept params, sorted alphabetically, to
keep keys stable regardless of dict insertion order.

## Interaction with `"base"`

`"base"` resolution happens first, before expansion.  The merged params
dict (base + delta) is what gets expanded.  Only params whose list length
> 1 in the **variant's own delta** (not the inherited base) contribute to
the key suffix — inherited single-value params are silent.

Example: `dynamic` overrides `constexpr` only, inheriting `params` from
`default`.  If `default` later gains `"M": [512, 520]`, then `dynamic`
also expands over those two M values — and the suffix appears on both.

Open question: should the suffix include inherited multi-value params or
only the ones declared in the variant's own delta?  Two options:

**Option A — suffix only own-delta multi-value params.**
Pros: suffix is minimal and stable when base changes.
Cons: two variants expanded from the same base produce identical suffixes,
making the registry key ambiguous.

**Option B — suffix all multi-value params in the merged dict.**
Pros: keys are always unambiguous.
Cons: suffix can be verbose when base has many sweep params.

Recommendation: **Option B**.  Ambiguous keys are a correctness bug;
verbosity is a readability inconvenience.  The suffix only appears when
expansion actually happens, so single-value params never contribute noise.

## `_load_examples` implementation sketch

```python
import itertools

def _expand_params(params: dict) -> list[dict]:
    """Return list of flattened {name: scalar} dicts, one per product point."""
    names = list(params)
    values = [params[n] for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]

def _sweep_key_suffix(merged_params: dict) -> str:
    """Suffix string for params with more than one value, sorted by name."""
    swept = sorted(
        (k, v) for k, v in merged_params.items() if len(v) > 1
    )
    if not swept:
        return ""
    return "[" + ", ".join(f"{k}={v}" for k, vlist in swept for v in [None]) + "]"
    # (placeholder — actual impl iterates over the specific combo values)
```

The real loop inside `_load_examples`:

```python
combos = _expand_params(merged_params)
is_sweep = any(len(v) > 1 for v in merged_params.values())

for combo in combos:
    entry = dict(merged)
    entry["params"] = {k: [combo[k]] for k in merged_params}

    # Build suffix from swept params in this combo
    if is_sweep:
        swept_names = sorted(k for k, v in merged_params.items() if len(v) > 1)
        suffix = "[" + ", ".join(f"{k}={combo[k]}" for k in swept_names) + "]"
    else:
        suffix = ""

    base_key = folder if vname == "default" else f"{folder}__{vname}"
    key = base_key + suffix

    if module_sig:
        runtime, constexprs, param_values = _resolve_variant(
            module_sig, entry, kernel_name=f"{folder}::{vname}{suffix}"
        )
        entry["signature"]    = runtime
        entry["constexprs"]   = constexprs
        entry["param_values"] = param_values
    registry[key] = entry
```

`_resolve_variant` is unchanged — it still enforces 1-element lists on
the already-flattened per-combo entry.

## `extra_checks` and sweep

`extra_checks` lambdas may need to vary by combo (e.g. `assert_result_type`
checks a static memref shape that differs by `M`).  Two options:

**Option A — extra_checks receives the combo dict.**
Change the call site from `entry["extra_checks"](tester)` to
`entry["extra_checks"](tester, entry["param_values"])`.  Lambdas that
don't need it ignore the second arg with `**_`.

**Option B — extra_checks stays `(tester) -> None`; swept variants that
need shape-specific checks declare their own `extra_checks`.**
Works today because `extra_checks` is already per-variant.  For a static
memref check in a swept static variant, the lambda closes over the
specific combo values injected into `entry` at expansion time.

Recommendation: **Option B** for now.  Option A is a breaking change to
all existing `extra_checks` lambdas.  If a swept static variant needs a
per-combo structural check, it can use a factory:

```python
def _make_checks(M, K):
    def checks(t):
        t.assert_result_type("ktdp.construct_memory_view", f"memref<{M}x{K}xf32>")
    return checks

# in VARIANTS:
"extra_checks": _make_checks(M, K),  # called at expansion time
```

This requires extra_checks to be a callable that is re-evaluated per
combo, which is already the case when it's a lambda that closes over
loop variables — but Python's late-binding means the factory pattern is
safer.

## Constraints preserved

- `_resolve_variant` still validates that every `constexpr` name has a
  value in the (now single-combo) `params`.
- The `"base"` resolution is unchanged.
- Single-element lists remain valid and produce no suffix — fully backward
  compatible.
- `disabled` variants are skipped before expansion, so a disabled variant
  with sweep params does not produce many skipped entries.

## What does NOT change

- `constexpr` lists — these are not swept.  A name is either a constexpr
  or a runtime arg for a given variant; sweep grammar only varies the
  *values*, not the partition.
- `SIGNATURE` — not swept.
- `grid` — not swept (would require a different kernel invocation).
- `reference` / `inputs` / `output_key` — not swept (the oracle is
  per-variant, not per-combo).  If the oracle depends on shape, `inputs`
  already receives `**param_values` so it adapts automatically.

## Known limitation: `reference` cannot see `param_values`

The `reference` calling convention is `(inputs: dict) -> np.ndarray`.
It only receives the runtime buffer dict returned by `inputs(**param_values)`,
not `param_values` itself.  Oracles that need constexpr values (e.g.
`run_2d_index_3d_block` needs `BLOCK_B`, `BLOCK_L`, `BLOCK_H` to slice
the output corner correctly) must receive them via `functools.partial` at
variant definition time.  This means those values are duplicated between
`params` and the partial call — a maintenance hazard that the `"base"`
mechanic does not eliminate.

The fix is to change the calling convention to
`reference(inputs, param_values)` and update all oracles.  That is a
broader change deferred to a follow-up; it is orthogonal to the sweep
grammar.
