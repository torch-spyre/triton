# Convergence plan — code → design docs, tests green throughout

## Progress (as of session end)

| Step | Status | Notes |
|------|--------|-------|
| S1   | DONE   | Golden net captured at `/tmp/golden/` (4 variants). Baseline: 461 passed / 27 skipped. |
| S2   | DONE   | `-2` role deleted. `classify` rewritten right-to-left (R-c). New `reduceDims` field. `reduceInner` added to `opSliceDims` during walk so 2D-tile machinery unchanged. Golden diff empty. Commit: `07f4e0846`. |
| S3   | DONE   | `buildExtract` + hand-rolled B extract + `ks*KA` deleted. Unified `extractOpSlice` lambda: A's `reduceLoop` dims indexed by their own IVs; B's `reduceInner` (K-flat) offset by reduction IV, size=1 stick. Golden diff empty. Commit: `0e73345e9`. |
| S4   | DONE   | `strides`/`sizes` (`SmallVector<OpFoldResult>`) added to `OperandPlan`. `getMemViewForOperand` + `threadStridesAndSizes` helpers added. Wired into `dispatchMatmul`. Not yet consumed (S5 flips switch). Golden diff empty. Commit: `2748a1fa5`. Note: no consistency assert between `plan.sizes` (full descriptor extents) and `coords.physShape` (block extents) — see TODO(S5) comment in `threadStridesAndSizes`. |
| S5   | TODO   | Switch `extractOpSlice` to use `plan.strides` for extract strides; use `plan.sizes` for dynamic trip counts in `collectLoops`. Remove last `coords.op` reads from classify floor/lane detection. Static golden diff empty; dynamic diff expected non-empty (offsets become threaded SSA). |
| S6   | TODO   | Home-level accumulator (R-e): parallel loops carry nothing; `acc0` init inside parallel leaf; reduction loops carry acc. Replaces `cVal` mechanism. IR shape changes for `spyre_stick_k`. |
| S7   | TODO   | Unify parallel nest into `emitNest`; delete `emitParallelNest`. Only if parallel loops carry nothing after S6. |
| S8   | TODO   | Apply S2-S5 to `emitStoreStage` (same `-2`-free model + `iv*laneSize` R-a violation at :1022). |

Goal: remove the 7 inconsistencies (see PROBLEMS_SUMMARY.md) between
`RewriteDescriptorLayout.cpp` and `PLAN_DATA_STRUCTURES.md` /
`PLAN_PSEUDOCODE.md` / `UNIFIED_RECURSION_STEPS.md`, keeping
`uv run pytest third_party/spyre/test` green after **every** step.

Principle for staying green: each step is a **behavior-preserving** rewrite —
the emitted IR stays semantically identical (often byte-identical) until the
final cleanup steps. We change the *model* first (internal), then *delete* the
now-dead workarounds, never both at once.

## Inconsistency → step map

| # | Inconsistency | Step |
|---|---------------|------|
| 1 | `-2` role exists; design has only `-1` | S2 |
| 2 | `classify` buckets by floordiv/mod op codes (R-a) | S2, S5 |
| 3 | No threaded strides/sizes in OperandPlan | S4 |
| 4 | B's `ks*KA` chunked-K-flat branch | S3 |
| 5 | Parallel loops carry an iter_arg (R-e) | S6 |
| 6 | Source stage emits insert_slice assembly | S6 |
| 7 | `emitParallelNest` doesn't emit loops | S7 |

## Baseline decision (do first)

The working tree has the background agent's P1 fix (init reduction with `cVal`)
+ `meta.py` fixture additions (K=128, dynamic variants). Decide:

- **Keep** the agent's K=128 fixture + `cVal` fix as baseline (suite green now),
  converge on top. ← assumed by this plan.
- **Revert** to pre-agent and converge from there (changes what "green" means).

If keeping: S6 will *replace* the `cVal` mechanism with the proper R-e home-level
accumulator; until then it stays as the thing keeping `spyre_stick_k` green.

---

## Step S1 — characterization tests (safety net, no code change)

Before touching the model, lock current IR shape so refactors can't silently
drift. Dump round-trip KTIR for `spyre_stick_parallel`, `spyre_stick_k`,
`spyre_stick_parallel_dynamic`, `spyre_stick_k_dynamic` to a golden dir.

```bash
for v in spyre_stick_parallel spyre_stick_k \
         spyre_stick_parallel_dynamic spyre_stick_k_dynamic; do
  uv run python third_party/spyre/scripts/dump_round_trip.py \
    --filter "matmul__${v}\$" --dest /tmp/golden
done
```

After each later step, re-dump and `diff` against `/tmp/golden`. A non-empty diff
must be *intentional* (only S6 changes IR shape). Keeps us honest beyond "tests
pass."

**Green check:** `uv run pytest third_party/spyre/test --no-header -q`.

---

## Step S2 — delete the `-2` role; derive loop-vs-consumed by right-to-left (R-c)

Internal model change, IR-identical. (#1, part of #2)

- `buildDimRoles`: stop emitting `-2`. Roles become `>= 0` or `-1` only (a
  K-stick dim is just `-1` like any reduction dim).
- `classify`: rewrite bucketing per R-c. Walk dims **right-to-left**:
  - innermost dim → `lane`.
  - `-1` dims → `reduceDims`; the **rightmost** is `reduceInner` (consumed),
    the rest are `reduceLoop` (loops).
  - `>= 0` stick-index dims → `floorDims` (parallel loops).
  - residual `>= 0` (e.g. M lane consumed by matmul) → `opSliceDims`.
- `OperandPlan`: rename/realign fields to match PLAN_DATA_STRUCTURES
  (`reduceDims`, `reduceInner`, `reduceLoop`, `floorDims`, `opSliceDims`).
- `collectLoops`: reduction loops come from `aPlan.reduceLoop` (the outer `-1`
  dims), which now equals the old `-2` set — so the emitted loops are identical.

Crucial: the set of dims that become loops must be **exactly the same** as
before (old `-2` dims == new "outer `-1`" dims). Verify with the golden diff
(empty).

To locate floor-vs-lane this step MAY still read `coords.op` (R-c-only); full
R-a removal of op codes is S5. This keeps S2 small.

**Green check + golden diff empty.**

---

## Step S3 — unify A/B extraction; delete B's `ks*KA` branch (R-b)

Behavior-preserving once R-b holds. (#4)

- Add `extractOpSlice(plan, loops)` matching PLAN_DATA_STRUCTURES §extract:
  per phys dim — floor/reduceLoop dim → `(iv, 1)`; reduceInner of the other
  operand → matching reduction iv, one stick; lane/opSlice → `(0, full)`.
- Route **both** A and B through it. Delete the hand-rolled B extract and its
  `ks*KA` / size-`KA` arithmetic.
- Risk: today B's K-flat is sliced `KA`-wide at `ks*KA`; the one-stick-wide R-b
  form must produce the **same** elements. For the current fixtures KA == stick
  size == 64 and the K-stick loop advances one stick per iter, so `ks*KA` ==
  `ks*64` == "the ks-th stick". Confirm the golden diff is empty (or only
  cosmetic SSA-name changes). If the dynamic fixture's B slice differs, that
  signals the stride threading (S4) is actually required first — if so, reorder
  S3 after S4.

**Green check + golden diff empty/cosmetic.**

---

## Step S4 — thread physical strides/sizes into OperandPlan (R-a addressing)

Internal; prepares to delete `physShape` math. (#3)

- Extend `OperandPlan` with `strides` / `sizes` (`SmallVector<OpFoldResult>`)
  read from the physical `construct_memory_view` (the memViewOp Phase 1 built).
- Populate them in `dispatchMatmul` / `dispatchStore` where the marker + memView
  are in hand.
- Do **not** yet switch the extract to use them — just thread + assert they
  match the `physShape`-derived values currently used (static case: equal;
  dynamic: the threaded value is the SSA the loop should use). This is a
  no-op-behavior add that surfaces mismatches early.

**Green check + golden diff empty** (nothing consumes strides yet).

---

## Step S5 — switch extract/insert to threaded strides; drop op-code reads (full R-a)

Now flip consumers to the threaded values and remove floordiv/mod from the hot
path. (#2 remainder)

- `extractOpSlice` / sink `insertOpSlice`: use `plan.strides` for the
  `extract/insert_slice` strides; trips from `plan.sizes` (ceildiv already baked
  for dynamic floor dims). Remove `iv*laneSize` (sink, :1022) and `physShape[p]`
  size math in favor of threaded extents.
- `classify`: remove the last `coords.op` reads — floor/lane/reduce located
  positionally + by role only.
- Golden diff now **expected non-empty** for the *dynamic* fixtures (offsets
  become threaded SSA instead of recomputed) but must stay **empty for static**
  (strides fold to the same constants). Numerical tests are the real gate here.

**Green check.** Static golden diff empty; dynamic diff reviewed.

---

## Step S6 — R-e home-level accumulator; remove parallel iter-arg + source insert_slice

The real fix for #5 and #6; replaces the agent's `cVal` workaround.

- `emitMatmulStage` per PLAN_DATA_STRUCTURES §source pseudocode:
  - parallel nest: `iterArgs = {}` (carries nothing).
  - at the home level (inside parallel leaf): `acc0 = zero(tensor<M,Nl>)`.
  - reduction nest: `emitNest(reduceLoops, {acc0}, matmulBody)`; matmulBody does
    `matmul(aS, bS, acc)` accumulating into the carried acc.
  - result = the reduction's final acc = **logical C**; RAUW `mm` with it.
  - **delete** the `fullCInit` parallel iter-arg and the source-stage
    `tensor.insert_slice` assembly path.
- This is where IR shape changes for `spyre_stick_k`: the K reduction now carries
  its own home-level acc (not `cVal`, not a parallel iter-arg). Confirm
  post-canonicalize the K-reduction `scf.for ... iter_args(%acc = <zero>)` and
  the matmul/add reads `%acc`, per UNIFIED_RECURSION_STEPS §Step 6.
- Multi-output-stick (parallel trip > 1) stays guarded with `emitError`
  (unchanged scope).

**Green check** including `spyre_stick_k` numerical; golden diff intentional.

---

## Step S7 — unify the parallel nest back into `emitNest` (delete `emitParallelNest`)

Cleanup of #7, once S6 made parallel loops carry nothing.

- With `iterArgs = {}`, the generic `emitNest` already emits a result-less
  `scf.for` with a bare `scf.yield`. Use it for the parallel prefix too and
  delete `emitParallelNest`.
- Caveat that birthed `emitParallelNest`: a value computed in a result-less
  `scf.for` can't escape (P3). Under R-e the per-stick C is the reduction's
  result *consumed inside* the parallel body (handed to the sink stage via the
  logical SSA + placement rule), not read after the parallel loop — so escaping
  is no longer needed. Verify: for single-output-stick, canonicalize still folds
  the trip-1 parallel loop so static golden diff stays empty.
- If escaping *is* still needed for the current RAUW wiring, keep
  `emitParallelNest` but document it as the "trip-1 inline" specialization of
  `emitNest` rather than a separate mechanism. (Decide when we get here.)

**Green check + static golden diff empty.**

---

## Step S8 — sink stage (separate, after matmul is converged)

Apply S2–S5 results to `emitStoreStage` (it has the same `-2`-free model needs
and `iv*laneSize` R-a violation, :1022). Done last per your phasing.

---

## Ordering summary

```
S1 golden net
  └ S2 delete -2 (R-c)            [IR-identical]
      └ S3 unify extract (R-b)    [IR-identical / cosmetic]
          └ S4 thread strides     [IR-identical, threaded unused]
              └ S5 use strides (R-a) [static identical, dynamic intentional]
                  └ S6 R-e accumulator  [IR change: the real fix]
                      └ S7 unify parallel nest [static identical]
S8 sink stage (repeat S2–S5 there)
```

Each arrow = "previous green before starting next." Revert any step whose golden
diff is unexpectedly non-empty and investigate before proceeding.
