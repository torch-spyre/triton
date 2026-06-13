# Convergence plan — code → design docs, tests green throughout

## Progress (as of session end)

| Step | Status | Notes |
|------|--------|-------|
| S1   | DONE   | Golden net captured at `/tmp/golden/` (4 variants). Baseline: 461 passed / 27 skipped. |
| S2   | DONE   | `-2` role deleted. `classify` rewritten right-to-left (R-c). New `reduceDims` field. `reduceInner` added to `opSliceDims` during walk so 2D-tile machinery unchanged. Golden diff empty. Commit: `07f4e0846`. |
| S3   | DONE   | `buildExtract` + hand-rolled B extract + `ks*KA` deleted. Unified `extractOpSlice` lambda: A's `reduceLoop` dims indexed by their own IVs; B's `reduceInner` (K-flat) offset by reduction IV, size=1 stick. Golden diff empty. Commit: `0e73345e9`. |
| S4   | DONE   | `strides`/`sizes` (`SmallVector<OpFoldResult>`) added to `OperandPlan`. `getMemViewForOperand` + `threadStridesAndSizes` helpers added. Wired into `dispatchMatmul`. Not yet consumed (S5 flips switch). Golden diff empty. Commit: `2748a1fa5`. Note: no consistency assert between `plan.sizes` (full descriptor extents) and `coords.physShape` (block extents) — see TODO(S5) comment in `threadStridesAndSizes`. |
| S5   | DONE   | Per-dim `SliceKind` (now named `StickIndex`/`StickifiedBlock`/`WholeBlock`) + `stickSize` in `OperandPlan`. `extractOpSlice` + slice-type calc are pure lookups. Multi-stick B K-flat fix (size=stickSize, offset=k*stickSize, (owner,physDim) IV lookup). Renamed `physShape`→`physBlock` (it is always the block extent, never the full tensor shape). Cross-operand fix: `reconcilePlans()` demotes `StickifiedBlock`→`WholeBlock` when neither operand has a `reduceLoop` (single-stick matmul has K-block > stick but no reduction loop). Commit `a884007b5`. **Rejected (do not re-attempt):** (a) `plan.strides` in extract strides (element strides, always 1). (b) `plan.sizes` for `collectLoops` trips (wrong source — full descriptor extent). (c) removing `coords.op` from floor detection (floordiv can sit at any phys position). |
| S5-blocker | ktir-cpu | `spyre_stick_k_dynamic` numerical needs ktir-cpu to resolve dynamic `tensor.extract_slice`/`insert_slice` offsets (the B K-flat offset is a runtime `k*stickSize`). The interpreter reads only the static offset array and uses the kDynamic sentinel → empty slice. Documented in ktir-cpu's `DYNAMIC_SLICE_OFFSET_PROBLEM.md`; fixed separately on the ktir-cpu side. Structural tests are green. |
| S6   | IN PROGRESS | Rescale the EXISTING loops to physical stick granularity. **Phase 1** now rescales each enclosing logical loop (`new_trip = old_trip * (BLOCK_d // stickSize)`, step 1) and wires the IV straight into the physical stick-index coord operand, discarding the `muli`+`divsi`/`remsi` chain. **Phase 2** (`emitMatmulStage`) does 2D slicing + reconciles the accumulator: `matmul(outs=acc)` into the existing `%arg5` iter_arg directly (single output stick) or extract/insert a sub-block (multi output stick); no `linalg.add`; RAUW old `mm`→new result, erase `mm`. Keep ALL loops (erasing the M-loop kills the `descriptor_store` inside it). Old baseline (`a884007b5`) inits acc with `cVal` → resets each K-tile → `spyre_stick_k` numerically wrong. See S6 section + CONTINUATION_PROMPT.md. |
| S7   | TODO   | Unify parallel nest into `emitNest`; delete `emitParallelNest`. Only if parallel loops carry nothing after S6. |
| S8   | TODO   | Apply S2-S5 + S6 loop-replacement to `emitStoreStage` (same `-2`-free model + `iv*laneSize` R-a violation at :1022). |
| S9   | TODO   | Factor the proven S6 loop-replacement machinery into an op-agnostic core; matmul + store as leaf-body instances. Deferred until concrete paths are green. |

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

## Step S5 — per-dim SliceKind; multi-stick B K-flat (commit `48e632b5f`)

The original S5 framing ("switch to `plan.strides`, trips from `plan.sizes`,
drop `coords.op` positionally") was based on assumptions that did not survive
contact with the code. What S5 actually delivered, and what was rejected:

### Delivered

- **Per-dim `SliceKind`** (`StickIndex` / `SlicedStick` / `Full`) added to
  `OperandPlan`, assigned in `classify()` from local geometry. `extractOpSlice`
  and the 2D slice-type computation switch on it — pure lookups, no loop-set
  inspection or cross-operand reasoning at slice time. Replaces the ad-hoc
  `reduceInnerIsSliced` bool + separate `reduceIV` parameter.
- **`stickSize`** field (= `physShape[lane]`, the Mod/lane extent). The slice
  width for stick-width dims; distinct from `physShape[dim]` (full extent =
  n_sticks × stickSize when BLOCK > stick).
- **Multi-stick B K-flat fix** (the real bug, exposed by retuning
  `spyre_stick_k_dynamic` to BLOCK_K=128, stick 64 → 2 K-sticks):
  - `SlicedStick` ⇔ `reduceInner` extent > `stickSize` (decided in classify).
  - size = `stickSize` (one stick), not `physShape[reduceInner]` (full K-flat).
  - offset = `reduceIV * stickSize` (was the raw stick index).
  - IV lookup matches `(owner, physDim)` so A's single reduction IV no longer
    leaks into B's floor dim when they share a physical index.

### Rejected (do NOT re-attempt without re-deriving)

- **`plan.strides` for `extract_slice` strides** — wrong. `tensor.extract_slice`
  strides are *element* strides within the tensor (always 1); physical memory
  strides live on the `construct_memory_view` (Phase 1). Threading `plan.strides`
  into the extract caused OOB (`4032 >= 64`). Kept stride-1.
- **`plan.sizes` for trip counts in `collectLoops`** — wrong source.
  `plan.sizes` (from the memViewOp) is the *full descriptor extent* (all
  K-sticks); the loop trip is the *block extent* (`physShape[p]`, = sticks per
  access tile). Using `plan.sizes` double-counted with the outer Triton K-loop.
  `Loop::trip` stays `int64_t` from `physShape[p]`; tile/block sizes are
  constexpr so it is always static (a genuinely dynamic block size would need
  the trip threaded from the *access tile* shape, not the memView — not needed
  by any fixture).
- **Removing `coords.op` from `classify` floor detection** — no positional rule
  replaces it. A floordiv (floor) dim can sit at any physical position; the
  current fixtures only *happen* to put it first. Distinguishing floor from
  opSlice still needs `coords.op[p] == FloorDiv`.

### Blocker for the dynamic numerical gate

`spyre_stick_k_dynamic`'s numerical test needs the **ktir-cpu** interpreter to
resolve dynamic `tensor.extract_slice` offsets — the B K-flat offset is a
runtime `k*stickSize`, and the interpreter currently reads only the static
offset array (using the kDynamic sentinel → empty slice). The emitted KTIR is
correct. Documented in ktir-cpu `DYNAMIC_SLICE_OFFSET_PROBLEM.md`; fixed on the
ktir-cpu side, not here. Structural tests pass; static numerical tests pass.

---

## Step S6 — rescale the existing loops to physical stick granularity

The real fix for #5 and #6. The split of work between the two phases moves:

- **Phase 1** (`rewriteOnePhysicalize` / `rewriteAccessTile`) now **rescales each
  enclosing logical loop to physical stick granularity and rewires its IV
  directly into the stick-index coord operand**, discarding the logical-offset
  arithmetic. Previously it left the loop untouched and computed `divsi`/`remsi`
  off the logical IV; that is replaced.
- **Phase 2** (`emitMatmulStage`, source) does the **2D slicing + accumulator
  reconcile**; (`emitStoreStage`, sink) does the **scatter**.

### Why (the decomposition that drives everything)

The physical layout always splits a stickified logical dim `d` into:
- a **stick-index** dim (FloorDiv, extent `BLOCK_d // stickSize`) — iterated, and
- a **lane** dim (Mod, the **innermost** physical dim, extent `stickSize`) —
  consumed WHOLE, never iterated.

The lane being innermost-and-whole is invariant (it is the stick's contiguous
lane the matmul reads). So Phase 1 only ever has to wire **stick-index** dims to
loops; the lane is handled by Phase 2's slice.

A single logical loop over `d` steps in units of `BLOCK_d`. Physically that dim
has `BLOCK_d // stickSize` sticks per tile, so the physical iteration count
**expands by `BLOCK_d // stickSize`**. The cleanest realization is to **rescale
the existing loop in place** (NOT nest an inner loop):

```
new_trip = old_trip * (BLOCK_d // stickSize)      // old_trip = num tiles along d
new_step = 1
new IV   = global stick index along d  ──►  feeds the stick-index coord operand directly
```

The `i * BLOCK_d` logical offset disappears: the loop IV `i` now feeds straight
in **as the stick index**. No `muli`, no `divsi`/`remsi` for looped stick dims.

`BLOCK_d` and `stickSize` are both available on the marker coord map at Phase 1
time (`stickSize = phys_arg` of the Mod/lane dim; `BLOCK_d` from the logical
block shape). The factor is 1 for the single-stick fixtures (loop unchanged) and
2 for `spyre_stick_k_dynamic` (BLOCK_K=128, stick=64 → trip doubles).

### Pass-input IR for `matmul__spyre_stick_k`

```mlir
scf.for %arg3 ... {                                       // M-loop: parallel, NO iter_args/results
  %9 = scf.for %arg4 = 0 to 2 step 1 iter_args(%arg5 = %cst) -> (tensor<64x64xf32>) {  // K-loop: reduction
    %12 = muli %arg4, 64                                   // logical K offset  ← DISCARDED in S6
    %13 = descriptor_load %2[..., %12]                     // A logical
    %14 = descriptor_load %3[%12, ...]                     // B logical
    %15 = tt.dot %13, %14, %arg5                            // → linalg.matmul, accumulates %arg5
    scf.yield %15
  }
  descriptor_store %4[...], %9                             // sink consumes K-loop result
}
```

### Phase 1 — rescale loop + rewire IV (concrete)

For each stickified logical dim `d` that has an enclosing `scf::ForOp`
(found by IV-tracing the access-tile coord operand back to the loop's IV via
`traceToMLIRBlockArg`):

1. Compute `factor = BLOCK_d // stickSize`.
2. **Rescale the loop**: set lower bound 0, upper bound `old_trip * factor`,
   step 1. (When `old_trip` is a runtime SSA value, emit `old_trip * factor`;
   when static, fold it.)
3. **Rewire the stick-index coord operand** of the physical
   `construct_access_tile` to be the (rescaled) loop IV directly. Drop the
   `muli %arg, BLOCK_d` + `divsi`/`remsi` chain that previously fed it.
4. The lane coord operand is not derived from the IV — Phase 2 slices it whole.
5. Retype the load to physical rank (as today). `retypeChain` still stops at `mm`.

`applyIndex` (FloorDiv→divsi / Mod→remsi) stays in the codebase for stick dims
that have **no** enclosing loop (trip-1 / inline dims), but is NOT used on the
looped path.

For `spyre_stick_k`: K-loop `%arg4 = 0 to 2` is already stick-granular
(`factor = 64//64 = 1`), so the trip is unchanged; `%arg4` is wired straight into
A's stick-index coord and B's K coord, and `%12 = muli %arg4, 64` is discarded.

### Phase 2 — slice + accumulator reconcile (source)

After Phase 1 the loads are physical-rank and the existing reduction loop is
stick-granular, carrying the **existing** `%arg5` (= `dense<0>` C tile).
`emitMatmulStage` then:

1. Extract the 2D physical matmul tile from each load via `extractOpSlice`
   (lane taken WHOLE; residual matmul dims per `SliceKind`), using the rescaled
   loop IVs for stick-index offsets. Transpose if the reduction dim precedes the
   parallel dim.
2. **Reconcile the accumulator shape.** The matmul produces a per-iteration
   `[M_tile, N_tile]` sub-block.
   - **Single output stick** (M, N each one stick — current `spyre_stick_k`):
     `[M_tile,N_tile] == ` the logical C iter_arg shape, so
     `linalg.matmul(ins={aS,bS}, outs=%arg5)` writes the existing iter_arg
     directly.
   - **Multi output stick** (e.g. A stick-on-M M=128, B stick-on-N N=128):
     `extract_slice` the `[M_tile,N_tile]` sub-block out of the `[M,N]` iter_arg,
     `matmul(outs=subAcc)`, `insert_slice` back. The reshape lives ENTIRELY in
     Phase 2 (only Phase 2 knows the matmul tile geometry).
3. `linalg.matmul` is `result = outs + ins[0]@ins[1]`, so passing the iter_arg
   (or its sub-slice) as `outs` IS the accumulation — **no `linalg.add`, no
   fresh-zero matmul.**
4. RAUW the old `mm` result with the new matmul result; **erase `mm`**. The old
   `tt.dot`-derived matmul becomes dead; DCE removes it. **No loops are erased**
   (erasing the M-loop would kill the `descriptor_store` inside it).

### Sink stage

`emitStoreStage` is NOT modified by S6. It consumes the logical C the source
stage leaves behind (the existing reduction loop's result). No accumulator to
manage. (Sink loop rescale is S8.)

### Open implementation questions (resolve while coding)

- Loop rescale API: mutate the existing `scf::ForOp` bounds/step in place
  (`setLowerBound`/`setUpperBound`/`setStep`) vs. build a replacement loop. In
  place is preferred — it keeps the iter_args, results, body, and the enclosing
  M-loop + `descriptor_store` valid with no RAUW.
- Single-output-stick keeps the existing `%arg5` iter_arg unchanged; only the
  matmul `outs` wiring changes. Multi-output-stick adds extract/insert around it.
- Parallel (M/N) loops are trip-1 for the current fixtures and already fold —
  for them `factor==1` so the rescale is a no-op; do not special-case.

### IR shape change

For `spyre_stick_k`: trip unchanged (`factor==1`); `%arg4` feeds the physical
stick-index coords directly; `%12 = muli %arg4, 64` and the `divsi`/`remsi` are
gone. Post-canonicalize confirm the EXISTING K-loop now reads `%arg5` as the
matmul `outs`:
`scf.for %arg4 ... iter_args(%acc = <zero>) { ... linalg.matmul(..., outs=%acc) ... }`
(not a bare `%cst`, not a `linalg.add`); the old `tt.dot`-derived matmul is DCE'd.

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

Apply the S6 loop-rescale model + S2–S5 results to the sink. `emitStoreStage`
has the same `-2`-free needs and the `iv*laneSize` R-a violation (it computes the
logical-C offset by hand instead of letting a rescaled stick-index loop feed the
coord). Phase 1 already rescales the store's enclosing loops (same code path as
the source); Phase 2's sink then scatters the logical C into physical D using the
rescaled IVs as stick indices (no `iv*laneSize` arithmetic). Done last per the
phasing — only after the matmul source path is green.

---

## Ordering summary

```
S1 golden net
  └ S2 delete -2 (R-c)            [IR-identical]
      └ S3 unify extract (R-b)    [IR-identical / cosmetic]
          └ S4 thread strides     [IR-identical, threaded unused]
              └ S5 use strides (R-a) [static identical, dynamic intentional]
                  └ S6 rescale loops to stick granularity + R-e accumulator [IR change: the real fix]
                      └ S7 unify parallel nest [static identical]
S8 sink stage (repeat S2–S5 there)
```

Each arrow = "previous green before starting next." Revert any step whose golden
diff is unexpectedly non-empty and investigate before proceeding.


## NOTES

### Rejected S6 approaches (kept as warnings — do NOT re-attempt)

1. **Emit a fresh nest, erase the logical loops.** Walk up from `mm` to the
   outermost enclosing `scf.for`, emit a new physical nest before it, RAUW the
   outermost loop's results, erase `mm` + all enclosing loops. **Rejected:** the
   `descriptor_store` lives *inside* the M-loop body; the M-loop has no
   iter_args/results to RAUW, so erasing it deletes the store the sink stage
   needs. Also duplicates structure.

2. **Keep the logical loop + `divsi`/`remsi`, append a new accumulator iter_arg
   in Phase 2.** Leaves Phase 1's `muli %arg, BLOCK` + `divsi`/`remsi` coord
   chain in place and only mutates the loop's iter_args. **Rejected in favor of
   the rescale model:** carrying the logical-offset arithmetic is dead weight,
   and the multi-stick (BLOCK > stick) case then needs an extra inner loop. The
   rescale model folds the iteration expansion into the existing loop's trip and
   discards the offset arithmetic entirely — one mechanism for all stick counts.

### Settled S6 approach (what to implement) — see the S6 step section above

The physical layout splits each stickified logical dim into a **stick-index**
dim (FloorDiv, iterated) and a **lane** dim (Mod, innermost, taken whole). So:

- **Phase 1** rescales each enclosing logical loop to stick granularity
  (`new_trip = old_trip * (BLOCK_d // stickSize)`, step 1) and wires the IV
  directly into the stick-index coord operand — discarding the
  `muli`+`divsi`/`remsi` chain. The lane is never iterated. `factor` is 1 for the
  single-stick fixtures (loop unchanged), 2 for `spyre_stick_k_dynamic`.
- **Phase 2 source** slices the 2D physical tile (lane whole), reconciles the
  accumulator (`matmul(outs=existing %arg5)` for single output stick;
  extract/insert a sub-block for multi output stick — the reshape depends on the
  `extract_slice` extents, which only Phase 2 knows), and erases `mm`. No
  `linalg.add`, no loops erased.
- **Phase 2 sink** unchanged (S8 will rescale the sink loops).
