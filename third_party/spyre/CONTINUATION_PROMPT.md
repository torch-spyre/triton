## Continuation prompt

We are implementing `CONVERGENCE_PLAN.md` step by step in
`third_party/spyre/lib/Dialect/KTDP/Transforms/RewriteDescriptorLayout.cpp`.

**S1–S6 are DONE.** Branch `inbue-metadata-clean`. Last commit `f38e79a34`.
**S7 is NEXT.** The working tree is at the green `f38e79a34` baseline
(461 passed / 27 skipped) — start S7 cleanly on top of it.

### Authoritative spec: CONVERGENCE_PLAN.md "Step S7"

Read the S7 section of `CONVERGENCE_PLAN.md` first. This file only summarizes.

### S7 in one paragraph

With S6 done, the parallel loops no longer carry iter_args (they carry only
the `fullCInit` assembly accumulator, which canonicalize folds away for
single-output-stick). `emitParallelNest` was a workaround for the old model
where parallel loops needed to escape values; it now just sets each parallel
IV to `c0` and calls body inline (trip-1 assumption). The goal is to delete
`emitParallelNest` and route the parallel prefix through the generic `emitNest`
with `iterArgs = {}`, which emits a result-less `scf.for` with a bare
`scf.yield`. Verify: for current fixtures (all parallel trip-1), canonicalize
folds the parallel loop away, so the static golden diff stays empty.

### Build + test

```bash
uv pip install -e ".[spyre-test]" --no-build-isolation 2>&1 | tail -3
uv pip install -e /Users/flim/AI/ktir-cpu.worktrees/missing-ops --no-build-isolation 2>&1 | tail -1
uv run pytest third_party/spyre/test --no-header -q 2>&1 | tail -5
```

(The triton rebuild reinstalls the pinned ktir-cpu; the second line restores
the local `missing-ops` ktir-cpu with `tensor.extract_slice` support.)

### Acceptance

- Full suite green (461 passed / 27 skipped).
- `emitParallelNest` deleted.
- Static golden diff empty (trip-1 parallel loops fold away).
- No changes outside `RewriteDescriptorLayout.cpp`.

### Key files

- Implementation: `third_party/spyre/lib/Dialect/KTDP/Transforms/RewriteDescriptorLayout.cpp`
- Authoritative S7 spec: `third_party/spyre/CONVERGENCE_PLAN.md` (Step S7 section)
- Tests: `uv run pytest third_party/spyre/test`
