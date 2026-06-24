# Fusing Triton Kernels — Multi-Module KTIR Design

Authors: @tnakaike, @fabianlim 

> **Status:** standalone design spec. It is about the step *after* per-kernel
> Triton→KTIR lowering: combining several independently-lowered kernels into KTIR
> and eliding the HBM round-trips between them where it is legal. All facts needed
> to read it are inlined here.

## Code references — three repos

This doc touches three distinct codebases. Every code path below is tagged with
one of these so it is unambiguous where it lives:

| Tag | Repo | What lives there (relevant to this doc) |
|---|---|---|
| **[triton]** | `torch-spyre/triton`, paths under `third_party/spyre/` | The TTIR→KTIR lowering passes (C++): `LowerDescriptorMemory`, `LowerComputeOps`, `RewriteDescriptorLayout`, `ConvertFunctions`, `DistributeWork` — all in `third_party/spyre/lib/Dialect/KTDP/Transforms/`. |
| **[ktir-mlir-frontend]** | `torch-spyre/ktir-mlir-frontend` | The `ktdp` MLIR **dialect** definitions: `ktdp.construct_memory_view`, `construct_access_tile`, `load`/`store`, `get_compute_tile_id`, and the inter-tile ops `inter_tile_produce` / `inter_tile_reduce` + `!ktdp.tile_future` (PR #25). |
| **[torch-spyre]** | `torch-spyre/torch-spyre`, paths under `torch_spyre/_inductor/` | The Inductor backend that decides work distribution and residency: `work_division.py`, `pass_utils.py` (`PerCoreView`), `scratchpad/utils.py` (`get_ncores_for_buffers`), `fusion.py`, `coarse_tile.py`, `loop_info.py`. Also the origin of the work-slice metadata (`numWkSlicesPerDim` / `coreIdToWkSlice`). |

> Note: in the local checkout the `ktdp` dialect is consumed by **[triton]**'s
> passes but defined in **[ktir-mlir-frontend]** (a git submodule under
> `third_party/spyre/ktir-mlir-frontend`, which may be empty until initialized).

---

## 1. Goal & scope

Several Triton kernels — each authored with its **own** work slicing, Spyre
tensor layouts, and core assignment — should be compilable together, with the
compiler eliding HBM round-trips between a producer kernel and its consumer where
the geometry permits.

The design lands on a **multi-module** organization: each compute group becomes
its own KTIR module (mirroring the one-SDSC-per-module backend decomposition),
the groups are sequenced by a top-level function, and keeping an intermediate
on-chip (eliding its HBM round-trip) is a *local* optimization that adds an
inter-tile data-movement op at a function boundary — the compute functions
themselves stay separate.

The feature has four moving parts:

1. **Per-kernel lowering (reused, unchanged).** Each `@triton.jit` kernel runs
   through the five existing TTIR→KTIR passes **[triton]**
   (`third_party/spyre/lib/Dialect/KTDP/Transforms/`): `LowerDescriptorMemory`,
   `LowerComputeOps`, `RewriteDescriptorLayout`, `ConvertFunctions`,
   `DistributeWork`. Each kernel keeps its own descriptors,
   `tl.spyre_tensor_layout` annotations, tile sizes, and grid.
2. **Combine: one module per compute group, sequenced by a top-level function** — §3.
3. **Elide HBM round-trips where legal** — §4, using **[ktir-mlir-frontend]**'s
   `ktdp.inter_tile_produce` / `inter_tile_reduce`, gated by a work-slice
   geometry-compatibility check (§4.2).
4. **Fusion hints** (`coreIdToWkSlice`-style) to make the geometry match where
   it otherwise wouldn't — §5.

**Non-goals.** This doc does not redesign per-kernel lowering, does not pick the
cross-core reduction *algorithm* (unichain/bichain/etc. — decided far below, in
the DFIR backend), and does not introduce a scheduler. It is the *stitching*
layer.

---

## 2. Background facts this design rests on (inlined)

### 2.1 Each lowered kernel is already physical

`RewriteDescriptorLayout` **[triton]** rewrites every descriptor annotated with
`tl.spyre_tensor_layout` into a physical, stick-tiled `ktdp.construct_memory_view`
**[ktir-mlir-frontend]** — e.g. a logical `[M,N]` stick-on-N tensor becomes a
rank-3 `memref<[N/64], M, 64>`. Consequence for fusion: a producer kernel's
**output** view and a consumer kernel's **input** view for a shared tensor are
*already in the same physical (stick) coordinate system* and are directly
comparable. The fuser does no physicalization of its own.

### 2.2 Work-slice geometry: how work is cut, and who owns each piece

Work distribution is computed in **[torch-spyre]** (`work_division.py`) and
carried as two structures per op:

- `numWkSlicesPerDim` — per named axis, how many independent slices the iteration
  space is cut into (e.g. `{mb: 32}` = 32 M-row slices; `{x: 32}` = 32 N-column
  slices).
- `coreIdToWkSlice` — per physical core ID, which slice index it owns along each
  axis (e.g. core 7 → `{mb: 7}`).

These two together are *the* description of how a kernel spreads a tensor across
the 32 cores. **Two kernels can hand a tensor over in on-chip scratchpad only if
these agree on the shared tensor.**

### 2.3 HBM always composes; on-chip (LX) hand-off breaks on mismatch

This is the single correctness gate the whole design turns on:

- **HBM is the universal fallback.** HBM is globally shared and per-core HBM
  addresses are baked into each op's view independently. A producer with one
  geometry and a consumer with a *different* geometry always compose correctly
  through HBM — each side simply indexes its own addresses. No communication op,
  no error. (This is why today, an `add` that splits M and a `matmul` that splits
  N can chain at all: `add` stores to HBM, `matmul` reloads from HBM.)
- **LX (per-core scratchpad) breaks on geometry mismatch.** LX is local to a
  core — a core sees only its own slice. If the producer wrote slice *p* on a
  core and the consumer expects slice *q ≠ p* on that same core, the data is
  silently wrong. So an LX hand-off is legal *only* when producer and consumer
  agree on the per-core slice of the shared tensor.

**[torch-spyre]** already encodes exactly this decision for buffers shared
*between ops within a kernel* (deciding HBM-vs-LX residency) — see §4.2. The
elision step reuses that machinery across a *kernel* boundary.

### 2.4 Inter-tile produce/reduce (the replacement for a round-trip)

**[ktir-mlir-frontend]** PR #25 adds inter-tile communication ops that let
compute tiles exchange/combine data **without** passing through HBM:

- `!ktdp.tile_future<…>` — a single-use value connecting a producer to a
  consumer.
- `ktdp.inter_tile_produce` — a producer region that yields partial results
  across a group of tiles as a future (terminator `ktdp.yield_partial`).
- `ktdp.inter_tile_reduce` — a combiner region (2N args, `identity` operands)
  that merges those partials into a collapsed result (terminator
  `ktdp.yield_reduced`).

Only two modes are implemented and verified today: **all-reduce** (consumer tile
set `C == P` producer tile set) and **reduce-to-one** (`|C| == 1`). The verifier
checks `C ⊆ P` and rejects anything else as `"unsupported"` (not invalid) —
broadcast, reduce-scatter, gather/scatter are spec'd but not yet implemented.

Because `!ktdp.tile_future` is an SSA value, producer and consumer must live in
the **same region** — MLIR isolation (`func.func` and `builtin.module` are both
`IsolatedFromAbove`) forbids an SSA value from crossing a function or module
boundary. This single fact is what shapes the whole architecture in §3–§4.

**Alternative: `ktdp.construct_distributed_memory_view`.** The `ktdp` dialect
**[ktir-mlir-frontend]** also has a *view-based* way to express a distributed
tensor, predating the explicit ops above. `ktdp.construct_distributed_memory_view`
declares that a tensor is partitioned across cores (along a named axis, by a core
count); the cross-core combine is then left **implicit** — encoded as a pattern of
distributed views + `ktdp.store`s that the backend reverse-engineers into the
reduction. The two mechanisms trade off the same way as everywhere in this stack:

| | `inter_tile_produce`/`reduce` (PR #25) | `construct_distributed_memory_view` |
|---|---|---|
| Intent | **explicit** — a named combiner region | **implicit** — a pattern the backend infers |
| Result type | a non-distributed value (the reduced tensor) | a distributed `memref` view; combine inferred from stores |
| Backend work | reads the op directly | must recognize the view+store pattern |
| Maturity | new; all-reduce / reduce-to-one only | pre-existing, already lowered |

This doc uses the explicit ops for case (b) (§4.2) because the intent is
unambiguous, but a `construct_distributed_memory_view` over the shared LX region
is the equivalent view-based encoding and is the fallback where the explicit op's
mode is `"unsupported"` (§8 item 2). Both are `ktdp` ops in the same dialect —
neither changes the §3–§4 structure.

### 2.5 Grid is stamped per-function today

`DistributeWork` **[triton]** lowers `tt.get_program_id` / `tt.get_num_programs`
to constants and stamps a single `grid` attribute on the enclosing `func.func`.
Its current contract (per its own source comments) is **one grid per pass run —
every function in the module must share the same grid rank/shape**, with a noted
future direction of a per-function `spyre.grid` attribute for mixed-grid modules.
`ktdp.get_compute_tile_id` **[ktir-mlir-frontend]** reads that grid to produce the
core ID. The multi-module architecture (§3) sidesteps the "must share one grid"
constraint by putting each grid in its own module.

---

## 3. Architecture: one module per compute group, sequenced by a top-level function

**The primary structure is multi-module.** Each compute group — the KTIR analog
of one SDSC — becomes its **own** module, nested under a top-level module. One
sequencer function, named after the fused op, runs the groups in dependency
order. Intermediates between groups live in HBM by default.

```
builtin.module @fused_matmul {
  builtin.module @g0 attributes {grid = [16]} { func.func @add  ... }   // a+b
  builtin.module @g1 attributes {grid = [32]} { func.func @fill ... }
  builtin.module @g2 attributes {grid = [32]} { func.func @bmm  ... }
  builtin.module @g3 attributes {grid = [32]} { func.func @comb ... }
  builtin.module @g4 attributes {grid = [32]} { func.func @addd ... }   // y+d
  // sequencer @fused_matmul: runs g0 → g1 → … → g4, intermediates in HBM
}
```

### 3.1 Why this is the right baseline

- **1:1 with the backend.** Each SDSC already lowers to its own DFIR module
  today, and a bundle (`bundle.mlir`) sequences the N SDSCs. "One KTIR module per
  compute group" makes the KTIR decomposition match the SDSC/DFIR decomposition
  exactly — the module boundary *is* the SDSC boundary.
- **The grid problem dissolves (the Q1 answer).** `grid` (core count +
  distribution) becomes a **per-module** attribute, so groups with different core
  counts coexist with no contract change. `DistributeWork` **[triton]** runs once
  *per module*, on exactly the grid that module needs — its current
  one-grid-per-run contract is honoured, not relaxed. The 16-core `add` is simply
  its own module alongside the 32-core groups. `ktdp.get_compute_tile_id`
  **[ktir-mlir-frontend]** reads the enclosing module's grid.
- **Composition is always legal.** Because each inter-group tensor goes through
  HBM, any pair of groups composes regardless of geometry (§2.3). Correctness is
  never in question; elision (§4) is a pure optimization on top.

### 3.2 The sequencer function

"Modules call each other" is **not** literal. MLIR's `func.call` takes a flat
symbol reference resolved in the nearest symbol table, and a `builtin.module` is
itself an isolated symbol table — so a function in one module cannot `func.call`
a function in another. Instead, one top-level function — named after the fused op
(`@fused_matmul`, `@ffn`, …) — runs the groups in order. It does no compute of
its own: it is logical, carries grid `[1]`, and maps to no hardware compute op.
Its backend analog is `bundle.mlir`, which only orders the per-SDSC dispatches.
Data passes from one group to the next through HBM tensors named at the top
level, not through SSA values.

### 3.3 The cost, and how data stays on-chip

Routing every intermediate through HBM costs one round-trip per intermediate. To
keep an intermediate on-chip (in LX) instead, two consecutive groups use the
inter-tile data-movement ops (§2.4). **This does not change the multi-module
structure and does not merge the compute functions.** The inter-tile
produce+reduce pair is self-contained: it sits at the **start or end of a compute
function**, or in a small **dedicated data-movement function** — the same shape
as a `restickify`. The data it moves lives in LX and is addressed by a
`ktdp.construct_memory_view` **[ktir-mlir-frontend]**, not by an SSA value that
would have to cross a function boundary. §4 says when this is legal and what the
op looks like.

---

## 4. Keeping an intermediate on-chip (eliding the round-trip)

Two consecutive groups can pass a tensor through LX instead of HBM — eliding the
round-trip — when they slice that tensor the same way across cores. The decision
is one check (§4.1); when it passes there are two shapes of data movement (§4.2),
neither of which merges the compute functions.

### 4.1 The check: do both groups slice the tensor the same way?

Decide HBM vs on-chip with the **same predicate [torch-spyre] already uses to
decide HBM-vs-LX residency** for a buffer shared between two ops:

```
compatible :=
    PerCoreView(producer writes T) == PerCoreView(consumer reads T)
```

- `PerCoreView` (`torch_spyre/_inductor/pass_utils.py`) **[torch-spyre]** captures
  `(device_dim_index, split_factor)` pairs keyed by the tensor's *device*
  dimension — comparable across two groups even though each names its axes locally
  (`mb`/`x`/`out`/`in`).
- `get_ncores_for_buffers()` (`torch_spyre/_inductor/scratchpad/utils.py`)
  **[torch-spyre]** returns `-1` when any pair disagrees.

Mapping to KTIR: the producer's output and the consumer's input are each a
`ktdp.construct_memory_view` **[ktir-mlir-frontend]** over the tensor. If the
`PerCoreView`s match, the consumer's view addresses the same per-core LX slices
the producer wrote, so the consumer reads LX directly and the `ktdp.store` /
`ktdp.load` pair is dropped. If they disagree (`-1`), the tensor stays in HBM and
each side uses its own addresses — today's "mismatched buffer falls back to HBM,
no comm op, no error."

This is **not new analysis** — it is the existing LX-residency analysis applied
between two groups instead of two ops. Both views are already stick-tiled (§2.1),
so the `PerCoreView`s come straight from the `construct_memory_view` shapes.

Two groups with **different grids** are essentially always incompatible here
(different `coreIdToWkSlice`), so they stay separate HBM-connected modules,
consistent with §3. On-chip data movement is for groups that share a grid (or are
made to, via §5).

### 4.2 Two shapes of on-chip data movement

**(a) Same slicing, no combine — pure elision.** A pointwise consumer that reads
the tensor with the producer's slicing just reads the producer's LX slices. No
inter-tile op is needed at all; the `store`/`load` pair is simply deleted.

**(b) Cross-core combine — `inter_tile_reduce` at the consumer's start.** When
the consumer contracts over the dimension the producer sliced, each core holds a
*partial* result that must be summed across cores. The consumer function **begins
with** `ktdp.inter_tile_reduce` **[ktir-mlir-frontend]** over the LX partials
(all-reduce or reduce-to-one, §2.4), then runs its own compute. The reduce is the
consumer's first op — the function keeps its own body.

**FFN worked through** (`down(act(up(x)))`, tensor-parallel):

| Step | Op | Slicing | Between this step and the next |
|---|---|---|---|
| up | `H = X @ W_up` | `W_up` column-sharded → `H` is `{x: N-split}` | — |
| act | `A = act(H)` | elementwise, same `{x}` slicing | **(a)** same slicing → reads `H` from LX, no op |
| down | `Y = A @ W_down` | contracts the sharded dim | **(b)** `down` begins with `inter_tile_reduce` over the per-core partials |

So up / act / down stay **three separate functions**; only `down` gains a leading
`inter_tile_reduce`. We do not merge the bodies.

At this KTIR layer the producer/consumer express only the **intent** (a combiner
region); the cross-core reduction *algorithm* (ring topology, in-flight vs
staged) is chosen far below in the DFIR backend. KTIR never names a ring.

---

## 5. Fusion hints (`coreIdToWkSlice` and friends)

The §4.1 check is conservative: it keeps data on-chip only when the two groups
already slice the tensor the same way. Hints let the author (or Inductor
**[torch-spyre]**) *make* them agree. The hint surface is the one that already
exists for single kernels, reused between groups:

- **`work_div` / `tiles` on the shared axis** — force producer and consumer to
  split the tensor the same way (e.g. both split M by the same factor), so their
  `PerCoreView`s match by construction. This directly changes `numWkSlicesPerDim`
  **[torch-spyre]** and is the primary lever.
- **`coreIdToWkSlice` pin (placement)** — beyond *how many* slices, *which core*
  owns *which* slice. LX is per-core, so the producing and consuming core for a
  given slice must be the **same physical core** (§2.3). Pinning `coreIdToWkSlice`
  identically on both groups guarantees slice *s* is produced and consumed on the
  same core.

Hints are advisory and correctness-preserving: if a hint can't be honored (it
would violate the consumer's own layout), the tensor stays in HBM. A hint never
makes an incompatible tensor silently use LX — the §4.1 check runs *after* hints
are applied and is the final gate.

---

## 6. Do we need a Triton-level inter-tile surface? (issue #20)

`torch-spyre/triton` **issue #20** proposes a Triton-language surface for
cross-core reduction: `tl.inter_tile_produce` / `tl.inter_tile_reduce` Python
builtins → `tt.inter_tile_*` ops → `ktdp.inter_tile_*` **[ktir-mlir-frontend]**.
The author writes the grouping by hand:

```python
partial = tl.sum(block, axis=0)
fut = tl.inter_tile_produce(partial, producer_groups=[[0, 1, 2, 3]])
reduced = tl.inter_tile_reduce(fut, consumer_groups=[[0, 1, 2, 3]],
                               identity=tl.zeros([64], tl.float16),
                               combine=lambda l, r: l + r)
if pid == 0:                       # reduce-to-one; drop the guard for all-reduce
    OUT_desc.store([0], reduced)
```

**For this design, we do not need it.** Everything the `tl.` builtins ask the
author to spell out, torch-spyre **already computes**:

| `tl.inter_tile_*` argument (issue #20) | Already in torch-spyre |
|---|---|
| `producer_groups` / `consumer_groups` | the core cohorts in `coreIdToWkSlice` **[torch-spyre]** |
| `if pid == 0` (reduce-to-one) vs no guard (all-reduce) | implied by whether the consumer needs the result on one core or all |
| `combine` lambda + `identity` | the op being distributed (the reduction torch-spyre is already lowering) |

So the fuser reads `PerCoreView` / `numWkSlicesPerDim` / `coreIdToWkSlice` and
**synthesizes** the `ktdp.inter_tile_*` ops itself (§4.2, case b). Re-stating the
same grouping at the `tl.` level would be redundant for kernels that came through
Inductor — the intent is *derivable*, not something the author must supply.

**The combiner is arbitrary — but it is still captured, in the SDSC JSON.** The
strongest argument for a `tl.` surface is #20's point that the combiner is
open-ended: not just `sum`, but argmax, online-softmax-style merges, etc. The
worry is that an inferred path can only handle a fixed set of named reductions.
That worry does not apply to the Inductor path, because the combiner there is
**not free-form text the author wrote** — it is the op Inductor lowered, and
torch-spyre serializes that op (with its work split) into one **SDSC JSON per
compute op** **[torch-spyre]**. The reduction the SDSC describes is whatever
`linalg.reduce` / op chain produced it; an "arbitrary" combiner that came through
Inductor is therefore already pinned down as a concrete compute description, not
lost. Concretely, the fuser recovers every `ktdp.inter_tile_reduce` component from
one SDSC:

| `ktdp.inter_tile_reduce` needs | SDSC JSON field **[torch-spyre]** |
|---|---|
| producer / consumer groups (`C ⊆ P`) | `coreIdToWkSlice_` cohorts + `numWkSlicesPerDim_` split factors |
| reduce-to-one vs all-reduce | whether one core or all hold the output slice (the `out`/`x` axis split vs the reduced `in`/`mb` axis) |
| `combine` region | the reduction op's `ComputeNode` `type_` (e.g. `ADD` for matmul psum, `max`/`add` for softmax) in the op's schedule tree |
| `identity` | that combiner's identity (the `linalg.fill` / accumulator init the op already carries, e.g. `0.0` for sum) |

This is exactly the split-K case (§7.2): the `genericpartialreduction` SDSC for
the K-combine is an `ADD` over the `in`-axis cohorts — the fuser reads the op and
the cohorts and emits the matching `inter_tile_reduce`. A softmax distributed over
its reduction axis is the same shape with a `max` (then `add`) combiner. No author
lambda; the SDSC already says what to combine and how.

**Hand-written kernels: combiner design left open.** A standalone Triton kernel
that did **not** go through torch-spyre Inductor has no `work_division` metadata —
nobody computed `coreIdToWkSlice` for it, and no SDSC pins down its combiner. There
the author *is* the only source of both the grouping and the combiner, and a
genuinely arbitrary combiner (argmax, online-softmax) may be one no Inductor op
emits, so it cannot be recovered from an SDSC the way the table above describes. **We do
not commit to a surface for this case here.** Issue #20's `tl.inter_tile_*` builtins (author writes
the `combine` lambda + `producer_groups`) are one option; an out-of-band annotation
is another. This mirrors the existing two-path split — the Inductor path derives
layout/work-division for you (`tl.spyre_tensor_layout` is the analog), the standalone
path makes the author state it — but the exact combiner-carrying surface for
hand-written cross-core reductions is **out of scope for this design and left
open**.

**Net:** for the fusion scope here (stitching Inductor-lowered kernels), reuse
`PerCoreView` and the SDSC — the grouping *and* combiner are derivable, so no `tl.`
surface is needed. Issue #20 is **orthogonal**: it serves hand-written kernels,
whose combiner design we deliberately leave open. Both converge on the same
`ktdp.inter_tile_*` ops, so neither blocks the other.

---

## 7. Examples

### 7.1 `add` → `matmul` — incompatible by default, on-chip with a hint

```
k0:  X = A + B        grid=[32], M-row split  → numWkSlicesPerDim {mb: 32}
k1:  Y = X @ C        grid=[32], N-col split   → numWkSlicesPerDim {x: 32}
```

`k0` and `k1` become modules `@g0`/`@g1` (both `grid=[32]`) sequenced by the
top-level function.

- **Default.** `PerCoreView(g0 writes X) = {(M, 32)}`;
  `PerCoreView(g1 reads X) = {(N, 32)}` **[torch-spyre]** → disagree → `X` stays
  in HBM. Correct: the two kernels slice `X` along different axes.
- **With a hint.** `work_div={"M": 32}` on both (or an identical `coreIdToWkSlice`
  pin) makes both `{(M, 32)}` → compatible → case (a): the `ktdp.store X` /
  `ktdp.load X` pair is dropped and `g1` reads `X` from LX. Each core keeps its
  M-row slice across both kernels; `X` never touches HBM.

### 7.2 Split-K matmul — `inter_tile_reduce` inside one group

`C = A @ B`, K split 4 ways and N split 8 ways across 32 cores:
`numWkSlicesPerDim {in: 4, out: 8}` **[torch-spyre]**. Each core computes a
*partial* `C` over its K-slice; the 4 cores sharing an N-slice (the K-cohort) must
sum their partials. That sum is a single `ktdp.inter_tile_reduce`
**[ktir-mlir-frontend]** over the `in` (K) axis, emitted **inside** the matmul
function — no second group, no body merge. `coreIdToWkSlice` places each K-cohort
on adjacent cores so the reduction is local.

If a pointwise epilogue (e.g. bias-add) consumes the reduced `C` with the same
`{out: 8}` N-slicing, that is case (a): it reads `C` from LX directly. This is the
same structure as the FFN `down` step (§4.2b), one layer up.

### 7.3 The 5-phase example, mapped to this design

`y = (a + b) @ c + d` (`M=64, K=128, N=256`, `tiles={K:2, M:4}`) lowers to 5
compute groups:

| Group | Op | Cores | Placement |
|---|---|---|---|
| g0 | `add` (a+b) | **16** (M/4 × K/2 → `[16,64]` per-tile space) | own module, `grid=[16]` |
| g1 | `fill` (zero-init accumulator) | 32 | own module, `grid=[32]` |
| g2 | `batchmatmul` | 32 | own module, `grid=[32]` |
| g3 | `combine` (K-reduction add) | 32 | own module, `grid=[32]` |
| g4 | `add` (y+d) | 32 | own module, `grid=[32]` |

Five modules, five grids, one sequencer — no module-wide grid needed. The 16-core
`g0` hands `a+b` to the matmul through HBM, which it would need anyway: `g0`'s
`{mb:16}` doesn't match the matmul's `{x:32}`, so the §4.1 check fails. The
`fill→batchmatmul→combine` chain (g1–g3, all on the `{x:32}` accumulator) is where
on-chip data movement could pay off — the candidate to evaluate first.

---

## 8. Open questions & risks

1. **Cross-module sequencing semantics.** The sequencer (§3.2) is not
   `func.call`. The exact top-level construct — nested `builtin.module`s + an
   explicit sequence op, vs. separate module artifacts stitched by the existing
   `bundle.mlir` machinery — needs to be pinned against what the DFIR backend's
   bundle step actually consumes.
2. **`inter_tile_*` [ktir-mlir-frontend] covers only all-reduce / reduce-to-one
   today.** Broadcast, reduce-scatter, gather/scatter are spec'd but
   unimplemented; any data movement needing those keeps the tensor in HBM until
   they land. The fuser must treat the verifier's `"unsupported"` as "stay in
   HBM," never as an error.
3. **`PerCoreView` [torch-spyre] between two groups is an assumption, not a tested
   path.** `get_ncores_for_buffers` is exercised today across *ops within a
   kernel*. Using it across *groups* should be validated on a fixture before it is
   trusted as the gate.
4. **Placement vs count.** Matching `numWkSlicesPerDim` is necessary but not
   sufficient — the *same core* must own each slice on both sides (§2.3). The hint
   design (§5) must pin `coreIdToWkSlice` placement, not just the split factor, or
   on-chip data movement will produce per-core-wrong data.
5. **Cross-function LX residency.** Case (a) and the data-movement function (§3.3)
   rely on a tensor's LX address persisting from the producer function to the
   consumer function. The allocator must keep that LX region live across the group
   boundary (the producer must not free it, the consumer must rebuild the same
   `construct_memory_view`). Validate this against how the backend currently
   scopes LX allocations; HBM (§3) is always the safe fallback.
