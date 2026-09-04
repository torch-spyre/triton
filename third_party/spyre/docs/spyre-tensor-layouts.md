# RewriteDescriptorLayout: Reconciling Kernels with Device Tensor Layouts

`RewriteDescriptorLayout` reconciles a logical Triton kernel against a device
tensor layout that is already stickified. The layout comes from a user
annotation, `tl.spyre_tensor_layout`, on each tensor descriptor; the pass does
not choose or infer layouts, it only rewrites the kernel body so every op sees
the physical shape the annotation demands. It runs after
`LowerDescriptorMemory`, `LowerScalarLoad`, and `LowerComputeOps` have already
lowered the kernel to KTDP.

## Definitions and assumptions

A **stick** is the hardware's contiguous innermost memory unit, fixed at
**128 bytes**. Elements per stick is derived from the element size, not a
constant: `128 / itemsize`, which gives **32 for fp32** and **64 for fp16**
(`STICK_BYTES` and `sticksize()` in `test/utils.py`). Layouts and this doc
always write the stick size as `S`; the lit fixtures use `S=64` on f32
tensors as a test convenience, not a hardware value — do not generalize from
it.

**Physicalization** (stick expansion) splits one logical dim `d` into two
physical dims, `d floordiv S` and `d mod S`. It is a coordinate-map rewrite of
how a dim is indexed, not data movement — the underlying buffer is
unchanged. Physical extents follow from the split: a `floordiv` dim gets
extent `ceil(N/S)` (covers a partial boundary stick when `S` does not divide
`N`), a `mod` dim gets extent `S`, and an identity dim keeps its logical
extent. Worked example: a logical `[M, N]` tensor stick-split on `N` becomes
physical `[ceil(N/S), M, S]`.

The **lane** is the physical dim carrying the `mod` role. It is found by
searching for that role; it is not necessarily the innermost dim.

The annotation itself, `tl.spyre_tensor_layout(desc, layout)`, takes one
entry per physical dim, each either `src` (identity), `(src, "floordiv", S)`,
or `(src, "mod", S)`, where `src` is the logical source dim. It lowers to
three i64 arrays on the descriptor: `phys_src`, `phys_op` (0 = identity, 1 =
floordiv, 2 = mod), and `phys_arg` (the `S` operand, unused for identity). The
docstring example (`spyre_tensor_layout` in `python/triton/language/core.py`):

```python
# desc describes a [M, N] logical tensor; physical layout is
# [ceil(N/64), M, 64] -- N stick-split, M and the stick dim untouched.
tl.spyre_tensor_layout(desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])
```

Assumptions the pass relies on: the author supplies the layout and stick
size, the pass never infers them; a logical dim spans at most two physical
dims — one `floordiv` and one `mod` — enforced by
`SpyreTensorLayoutOp::verify()`; and a `mod` dim cannot be sub-stick, i.e. its
extent must equal `S`.

## Phases

The pass runs in three phases, stated as a contract per phase.

**Phase 1 — physicalize the loads.**
Input: descriptors carrying layout markers, and the logical `ktdp.load` chain
below them.
Lowering: for each marker, rebuild the memory view and access tile at physical
shape and retype the `ktdp.load`, recording each value it physicalizes. Phase 1
stops there. It does not push the physical type into consuming ops: an op's
result type is Phase 2's decision, and setting it here would leave the op with a
physical operand and a logical result that no contract covers.
Output: physical loads, and the set of values Phase 1 made physical; every
consuming op still logical.

**Phase 2 — decide each op.** Two sub-steps: decide every type, then rewrite.

*2A, analysis.* Seed a worklist from the values Phase 1 physicalized and
propagate forward, recording for each reachable value the physical type it will
carry. Mutate nothing. Each op answers for itself through a
`PhysicalPropagationPattern`, registered by
`populatePhysicalPropagationPatterns` — an untaught op matches no pattern, is
reported, and gets the conservative answer. Resolution is transitive and carries
a visit stack, so a value already being resolved is a cycle and fails rather than
diverging.

Most rules are a function of the operand alone, but not all: whether a
`linalg.reduce` produces a physical result is a question about the layout its
result is *stored under*, so that rule walks forward to the store, the mirror of
what the store pattern does backward. Nothing about that walk mutates IR, and it
reads only what Phase 1 produced, so it stays an analysis.

*2B, rewrite.* Walk the ops and act on the recorded types rather than on the
current IR. An operand's physicality is a lookup: resolved, or predicted and not
yet reached (defer to a later visit), or absent and therefore logical.

Output: every op's operands and result agree. Markers stay live throughout,
since the decision reads each operand's coordinate map from its marker.

Seeding the walk is what scopes it. Membership answers "is this value reachable
from something Phase 1 physicalized", so an op on an unannotated path is never
visited and needs no per-op guard. Without that scoping a local shape rule
misfires on ops that legitimately change shape — `tt.expand_dims`, `tt.reshape`,
`tt.broadcast` and their siblings all have one tensor operand, one tensor result
and differing shapes, and retyping one corrupts it. Scoping per op instead, by
tracing an operand backward to its load, answers the same question once per
query and cannot see past an op with two tensor operands, where the walk has no
single producer to follow.

**Phase 3 — cleanup.** Erase the markers and any now-dead bridge casts.

### Where the phases live

Phase and file do not coincide, which is worth stating because three of the
init helpers sit in Phase 2B's file and one of them is called from 2A.

```
  markers (tt.spyre_tensor_layout)
        │
   ┌────▼──────────────────────────────────────────────────────────┐
   │ PHASE 1   RewriteDescriptorLayout.cpp                         │
   │   memory view + access tiles + ktdp.load retyped              │
   └────┬───────────────────────────────────────────────┬──────────┘
        │ physicalValues (the roots)                    │ physMemViewToMarker
   ┌────▼───────────────────────────────────────────────▼──────────┐
   │ PHASE 2A  PhysicalTypeAnalysis.cpp        (mutates nothing)   │
   │   worklist forward from the roots; one                        │
   │   PhysicalPropagationPattern per op kind                      │
   └────┬──────────────────────────────────────────────────────────┘
        │ PhysicalTypeMap ──▶ ctx.physicalTypeAnalysis   read, const
        │                 ◀── ctx.physicalTypes          write, one method
   ┌────▼──────────────────────────────────────────────────────────┐
   │ PHASE 2B  ContractionSynthesis.cpp, Classify.cpp              │
   │   SourceOpSpec → dispatchSource → classify →                  │
   │   reconcileOperandSet → resolveOperand → emitNarrowStage      │
   └────┬──────────────────────────────────────────────────────────┘
        │ verifyPhysicalTypeAgreement: physicalValues ⊆ PhysicalTypeMap
        ▼  PHASE 3   erase markers, bridge casts, dead logical views
```

Two crossings are deliberate and worth knowing:

- `canRebuildPhysicalInit` is defined in `ContractionSynthesis.cpp` (2B) and
  called from `PhysicalTypeAnalysis.cpp` (2A). It is a pure predicate, and 2A
  asks the emitter's own question so the decision and the emission cannot drift.
- `PhysicalTypeCarryForward` is 2B's only write into 2A's map, a handle holding
  it privately behind one method. `ctx.physicalTypeAnalysis` stays a pointer to
  const. Symmetrically, the 2A rules take `const MarkerByMemView &` — Phase 1's
  marker map — rather than the whole `PassContext`, whose `physicalValues` is a
  non-const reference member and would let an analysis rule write 2B's state.
  The phase split is only worth something if neither side can reach across.

The 2A/2B vocabulary is this doc's and `Passes.td`'s, not the driver's:
`RewriteDescriptorLayout.cpp` says "Phase 2A" and never "Phase 2B", its second
block is labelled just "Phase 2", and its file-header staged-model comment lists
only Phase 1 and Phase 3. Reconciling that is unfinished business.

## Operand reconciliation

Where several annotated operands feed one op, their splits must agree. The trip
count of a stick loop is a property of the operand set, not of one operand, and
two operands that disagree on it — different trip counts, or parallel splits on
different output axes — cannot be reconciled. That agreement is one question,
answered once per op, before its output is decided.

That agreement is `reconcileOperandSet`, which also yields the trip counts the
emitters need.

Separately, each operand needs its own resolution: the permutation bringing its
physical dims into the order the op consumes, and the extent of each op-tile
dim. That is `resolveOperand`, per operand and independent of the others, taking
the direction of the conversion so a store's data tile is resolved by the same
code as a matmul's operand rather than by a second implementation.

## The output decision

With inputs physical, Phase 2 fires on the op and decides what to do with the
**output**. Two outcomes:

| Output | Action |
|---|---|
| Can be physicalized | Physicalize it. The op consumes and produces physical shape; no loop. |
| Cannot be physicalized | Leave it logical and emit a loop that bridges physical inputs to the logical result. |

Whether the output can be physicalized *at all* is a property of the op, not of
the shapes:

| Op | Output can be physical | Why |
|---|---|---|
| elementwise (`arith`, `linalg.generic`) | Yes | Rank-agnostic; the physical type propagates unchanged. |
| `linalg.reduce` | Yes | `dimensions` is a list, so a stick-split reduced axis is expressible in one op, and a *surviving* one rides along as a batch dim. |
| `ktdp.store` | Yes | `AnyTensor`; the verifier checks only that data-tile and access-tile shapes agree. |
| `linalg.matmul`, `linalg.batch_matmul` | No | Contracts exactly one `K` axis, so a split `K` needs accumulation across sticks; and its init extents must match the A/B slice extents, so a surviving multi-stick axis is scattered by an outer loop rather than carried. |

An annotated output is what makes the first outcome available: the store's
access tile is physicalized alongside it, so the two sides agree. A reduce whose
output descriptor carries no marker takes the second outcome — the result stays
`tensor<64xf32>` and an `scf.for` accumulates into it.

The first outcome, for a reduce over a stick-split axis with physical input
`[2, 64, 64]` and physical output `[64]`:

```mlir
%r = linalg.reduce ins(%phys : tensor<2x64x64xf32>) outs(%acc : tensor<64xf32>) dimensions = [0, 2]
```

This is legal because `dimensions` is `DenseArrayStrictlySorted` with no
adjacency or trailing-position requirement.

### Which physical shape: the output axis space

For a reduce, "can be physicalized" leaves a second question the table above does
not answer: *which* physical shape. Reducing away a logical axis deletes every
physical dim sourced from it and leaves the rest in place, so the operand's layout
**induces** one — the surviving physical dims, in order, with their coord ops and
args intact, against the surviving axes renumbered. If the output descriptor
declares exactly that layout, the reduce is emitted at it directly and the store
has nothing left to do. If it declares anything else, or nothing, the result stays
logical and the store's widen stage builds the physical form afterwards.

#### Why this had to become an explicit choice

A `role` answers "which output axis does this physical dim feed", and every source
op numbered those axes per surviving *logical* dim. That silently assumed one
surviving logical axis implies one output axis — true for every op the pass was
built for, and false the moment a surviving logical axis is stick-split, because
then one logical axis occupies two physical dims and both survive.

For `[M=64, N=128]` fp16 stick-split on `N`, physically `[2, 64, 64]`, folding `M`:

| physical dim | logical | survives? |
|---|---|---|
| 0, `N floordiv 64` | N | yes — stick index |
| 1, `M` | M | no — reduced |
| 2, `N mod 64` | N | yes — lane |

Under logical numbering dims 0 and 2 both take role 0, and the accumulator is
built as `accDims[role] = extent`, so they collide: the last write wins and the
accumulator comes out rank-1. The stick-tiled `tensor<2x64xf16>` the store wants
is not merely unbuilt, it is inexpressible. The old code was not wrong, it was
complete for its inputs; what it lacked was a way to say that this op's output
axes are counted differently.

That choice is named `OutputAxisSpace`, and it is what a `role` numbers positions
in:

- **Logical** — one output axis per surviving *logical* dim. The two physical dims
  of a stick-split axis share one role, so a surviving stick-index dim is not an
  output axis at all: it is a scatter dim, sliced away at extent 1 or scattered by
  an outer loop. Every matmul-like op is here.
- **Physical** — one output axis per surviving *physical* dim, in physical order. A
  surviving stick-index dim is an output axis of its own, i.e. a batch dim of the
  emitted op, so one logical axis can occupy two output axes (its stick index and
  its lane) — which is exactly what a role numbered per logical dim cannot
  express, and why the accumulator is keyed by output axis rather than by role.

What makes the second space cheap is that **roles stay unique in both.** Only the
numbering of survivors changes, never whether a dim survives, so
`accDims[role] = extent` and the transpose permutation's uniqueness assumption
are untouched — the accumulator code did not change at all. `canonicalAxes` keeps
answering *whether* a logical dim survives, which is space-independent;
`buildDimRoles` takes the space and numbers the survivors.

The space is a property of the op **instance**, not the op kind: the same
`linalg.reduce` lands in either space depending on what its result is stored
under. So the pattern does not choose. Phase 2A decides and the pattern reads the
verdict — `ReducePropagation` computes the induced layout and compares it against
the output descriptor's marker, all three coordinate arrays and the access-tile
shape. That is the same direction "decide physical types before rewriting" set,
and it is what the pass's own older comment already said the rule had to be: a
result is physical only under a layout of its own, which a reduce result acquires
only when the output descriptor is annotated.

One predicate, `isScatterDim(role, coordOp, space)`, decides for both `classify()`
and any caller building a target order from a marker directly, so the two cannot
disagree about which dims are excluded. It is named for the decision it makes —
and specifically for what the dim's loop then *does*, scatter — rather than for a
property of the dim, because it takes three facts to reach: the dim survives, it
is a stick index (`isFloorCoord`), and the space is Logical so it has no output
axis to occupy. Its sibling bucket, `reduceLoopDims`, is sliced identically and
differs only in that its loop accumulates rather than scatters — see the bucket
table below.

Worked, for a logical `[M=64, N=128]` fp16 stick-split on `N` (`S=64`, physical
`[2, 64, 64]` = stick index, `M`, lane) reducing **M**, a whole physical
dimension, into `[128]` declared with the same split (physical `[2, 64]`):

```mlir
%acc = tensor.empty() : tensor<2x64xf16>
%r = linalg.reduce ins(%tile : tensor<2x64x64xf16>) outs(%acc : tensor<2x64xf16>) dimensions = [1]
ktdp.store %r, %out_tile : tensor<2x64xf16>, <2x64xindex>
```

One reduce, no loop, no slicing, and `ktdp.store` consuming the result. The
accumulator is rebuilt at the physical shape rather than reused: it is a different
physicalization of the same logical tensor than the init the op arrived with, not
a slab of it. Only a `tensor.empty` or a `linalg.fill` over one can be rebuilt —
which is what `LowerComputeOps` emits for a reduction's `outs` — since anything
else holds values a rebuild would drop; that keeps the Logical answer.

Reducing `N` instead, the stick axis itself, consumes both of its physical dims,
so nothing stick-split survives, the induced layout is a single identity dim, and
a `[64]` output declaring a split does not match it. That is the Logical answer,
and it is the `dimensions = [0, 2]` form above.

#### The two stick-index buckets: scatter vs accumulate

`classify()` puts stick-index dims in one of two buckets, and what separates them
is *not* how they are sliced — both get `SliceKind::StickIndex`, one stick per
iteration of their own loop. What differs is what the loop does with the slice:

| bucket | which dims | the loop |
|---|---|---|
| `scatterDims` | surviving (`role >= 0`) stick indices | **scatters** — each iteration writes a different slice of the output |
| `reduceLoopDims` | reduced (`role == -1`) dims past the first, since `opInnerDim` takes that one | **accumulates** — every iteration folds into the same accumulator |

Naming one of them for the `floordiv` coordinate they *both* carry hid exactly
that, and stopped being descriptive once a surviving floor coordinate could land
inside the op tile as a batch dim. A floor *coordinate* is still one:
`isFloorCoord` and `CoordOp::FloorDiv` name the coordinate, not a fate.

The distinction is load-bearing twice. `RewriteStorePattern` reads it as its two
preconditions — an empty `scatterDims` means nothing to scatter, a non-empty
`reduceLoopDims` means a reduction the store cannot express. And
`absorbReduceLoopDims` folds only the accumulating bucket into the op tile:
absorbing a reduce axis says nothing about where the output axes are, so
`scatterDims` stay outside it either way.

#### What `SourceOpSpec` carries

Two absorption-ish fields sit on it and they answer different questions, from
different places, at different times:

- `absorbReduceLoopDims` — can this op **kind** fold its whole reduce axis set
  into one emitted op? True for `linalg.reduce`, whose `dimensions` takes a
  sorted list; false for matmul, which contracts one axis at a time. Fixed at the
  pattern.
- `outputAxes` — does **this instance's** output get physical axes? Supplied by
  Phase 2A, with `outputMarker` beside it: the layout the result carries, null in
  the Logical space where the result has no layout of its own.

They were briefly one flag, and merging them is wrong in both directions — a
reduce can absorb its reduce axes while still being Logical.

#### Invariants the Physical space had to re-establish

- **`targetOrder` ⟷ `opTileDims`, 1:1 in physical order.** A surviving stick
  index can now be an op-tile dim, so both sides must exclude the same dims.
  `RewriteReducePattern` calls `buildDimRoles` itself and filters with the same
  `isScatterDim` in the same space `classify()` will use, rather than re-deriving
  the rule.
- **Idempotence.** See the section below: `dimensions` can now name a physical
  position inside logical range, which the older guard cannot see.
- **The subset invariant.** `verifyPhysicalTypeAgreement` asserts that everything
  Phase 2B found physical, 2A predicted. This matters because 2B reads *absence*
  from the map as "genuinely logical" and commits on that basis, so an
  under-claiming analysis would silently mis-lower. A minted replacement
  postdates the analysis, so the minting pattern carries the decision across with
  `PhysicalTypeCarryForward::carryForward` and containment holds by construction
  — with no exemption in the check, which keeps its full strength for Phase 1's
  roots and for what the elementwise and transpose patterns record. The assertion
  is live in the default build (`TritonRelBuildWithAsserts`); it is the library's
  only `#ifndef NDEBUG`, deliberately not an `LLVM_DEBUG` (runtime-gated, so it
  would run only when asked) and not a bare `assert` (which drops the condition
  but not the walk feeding it).
- **Store ordering.** `RewriteStorePattern` could fire first, build a widen loop
  for a logical result, then have the reduce replace that value underneath it. It
  defers while its data tile is *predicted* physical but not yet registered — the
  analysis-shaped sibling of the `pendingElementwiseRetype` deferral.
- **The init operand.** Rebuilt at physical shape rather than retyped, because
  nothing physical flows *into* an init — it is a root the forward propagation
  cannot reach, so there is no retype to ride and the shape has to be stated.
  What that leaves behind, the userless logical `tensor.empty`, is not this pass's
  problem: `_make_ktir` canonicalizes and CSEs immediately after, nothing between
  consumes it, and the emitted KTIR is byte-identical either way. A standalone
  `--rewrite-descriptor-layout` run therefore shows it, which is why
  `rewrite-descriptor-layout-reduce-batch-dim.mlir` canonicalizes in its RUN line
  — only against the folded module can it assert `CHECK-NOT: tensor.empty`.

#### Designs rejected

- **A third array (`accAxisOfTarget`) beside `targetOrder`.** Identical to
  `targetOrder` for every existing op — a whole indexing concept whose only
  distinct use is one new case.
- **Absorbing surviving scatter dims unconditionally for reduce.** A simpler flag
  with wrong answers: a rank-3 middle-axis reduce would emit a rank-3 result
  against a logical rank-2 store, and two lit tests regress. The decision
  genuinely depends on the output descriptor.
- **Special-casing the reduce in `emitNarrowStage`, or mutating it in place.**
  Provably safe in this one configuration, but it carves a per-op path through
  shared code, and the next op with a stick-split surviving axis would need its
  own.
- **Loosening the subset invariant**, broadly or by exempting minted values.
  Something has to give — with nothing done the assertion fires — but every
  exemption spends the check to buy it, and a flag saying "do not check me"
  spends it where it is hardest to notice.

#### Extending this to another source op

A new source op needs `absorbReduceLoopDims` for its kind and, if its result can
carry a layout of its own, a `PhysicalPropagationPattern` stating which output
layout its operand's layout induces. It does **not** need to touch the
accumulator or the permutation code — that is what keeping roles unique bought.
The case a per-logical-dim role provably cannot express is the rank-4 case in
`test/Conversion/rewrite-descriptor-layout-reduce-batch-dim.mlir`: two batch dims,
a stick index and an untouched dim. If that test ever needs a special case, the
abstraction has sprung a leak.

## One conversion path

Narrowing a physical operand into an op tile and widening an op tile back into
physical shape are the same conversion in opposite directions: both transpose,
both drive a counted loop, both slice and insert. They are one path taking a
direction, not two emitters — a store's op is the identity and its conversion is
the widening, so nothing about a sink needs its own code. `emitCountedLoop`
emits the `scf.for`, inlining when the trip count is one, and its callers differ
only in what the body yields: an accumulator carry for a reduction, an
`insert_slice` into a container for a scatter.

A conversion is emitted only where there is work. A slice covering a whole
tensor at unit stride is not emitted, and neither is a memory view that no
access tile references once the physical one supersedes it.

## Shape polymorphism of the memory ops

`ktdp.load` and `ktdp.store` take `AnyTensor` and their verifiers check only
that data-tile and access-tile shapes agree — no rank cap, no dim placement
rule (`KTDP_LoadOp`, `KTDP_StoreOp` in `KTDP.td`). By contrast `tt.dot` caps
operand rank at 2 or 3 at parse time, and `linalg.reduce` requires a strictly
sorted `dimensions` list. Memory ops constrain almost nothing, compute ops
constrain tightly — this is why a store absorbs whatever physical shape reaches
it while a matmul cannot.

## Idempotence

Phase 2 runs under a greedy pattern driver that re-enqueues an op whenever a
neighbour it feeds or consumes is rewritten, so ops are visited repeatedly
until a fixpoint. Every pattern's match condition must therefore be
falsified by its own rewrite. Where the output is left logical and a loop is
emitted, this follows from the rank change. Where the output is physicalized in
place the rank does not change, so the pattern must test whether the work is
already done — a reduce whose `dimensions` already covers the physical reduce
dims does not match. Running the pass on its own output is a no-op, and a lit
fixture asserts it by diffing the two.

A reduce emitted in the Physical output axis space needs a second test, because
neither of those catches it: it consumes the physical `ktdp.load` unsliced, so
every other match condition still holds, and its `dimensions` can name a physical
position that is *within* logical range (a single surviving reduce axis at
physical position 1, say), so "already covers the physical reduce dims" cannot see
it. What does is the registration: a result the rewrite recorded as physical is a
result this pattern already produced.

Recording that result is also what the Phase 2A subset invariant needs. A source
pattern *replaces* its op, minting a value that did not exist when the analysis
ran, so that value has no entry of its own; the pattern hands the analysis the one
fact it is missing, that the replacement carries the decision already made for the
value it replaced. Containment then holds by construction rather than by exempting
the new value from the check.

The driver's convergence result is checked, so a pattern that stays matchable
fails the pass with a diagnostic instead of spinning to the iteration cap and
reporting success. Note what that does not catch: if every pattern declines, the
worklist drains and the driver calls that a fixpoint, so a permanent deferral is
reported as success. Deferral must therefore be a prediction the analysis
supports, never a guess.

## Proposed: a backward requirement analysis

**Not implemented.** This section states a design and the questions it still owes
answers to, so the next person does not have to re-derive it.

### The shape of it

Phase 2A asks one question, forward: *what layout does this value have?* The reduce
showed there is a second question — *what layout is wanted of it?* — and today that
one is answered by a forward rule walking forward again, to the store, to fetch a
fact it cannot receive. That is `findStoreDestination`, and it is the tell.

Make it a direction instead. Two facts per value:

| fact | direction | seeded from |
|---|---|---|
| **have** | forward | the markers on physicalized `ktdp.load`s (Phase 1's roots) |
| **want** | backward | the markers on the `ktdp.store`s the values reach |

The decision at an op becomes a comparison of the two *at the same value*, rather
than a derivation from one of them. Agreement means physical; disagreement, or a
`want` with no satisfiable `have`, means logical and a bridge — which is what
happens today, reached deliberately rather than by fallthrough.

### The backward rules

Given a requirement `R` on an op's result, what does it require of each operand?
This is the part worth getting right, and the rules are not symmetric with the
forward ones.

| op | requirement induced on the operand |
|---|---|
| elementwise / single-tensor shape-preserving | `R` unchanged — rank-agnostic, so it passes straight through |
| `linalg.transpose` | `R` permuted by the inverse permutation |
| `linalg.reduce` | the **surviving** dims must match `R`; the reduced dims are unconstrained |
| `linalg.matmul`, `linalg.batch_matmul` | nothing crosses to the operands. The op is fixed at logical rank, so its result cannot carry `R`; `R` is discharged *here*, as the widen the store already emits. The operands are still narrowed, from their own `have` — the requirement does not have to cross to establish that |
| reshape, broadcast, expand_dims, collapse_shape | none — the physical dim count changes, so no requirement can cross |
| `ktdp.load` | terminus: this is where `want` meets `have` |
| `tensor.empty` / `linalg.fill` as a DPS init | `R` **is** the shape to build at — the fact the forward pass structurally cannot supply |

### The hard part: a requirement is partial, not a layout

A `want` cannot be represented as "a layout". For a reduce, `R` constrains only the
surviving axes — the reduced axis's own split is invisible to `R`, because it is
folded away and nothing downstream can see it. So a requirement is a **partial**
map: some physical dims constrained, others free. Modelling it as a full layout
either over-constrains (rejecting operands that would have worked) or needs a
sentinel value doing the same job less legibly.

The consequence is worth stating, because it is easy to expect otherwise: the
reduce decision stays a *comparison of two facts*, not a derivation from one.
Backward supplies `R` over the survivors, forward supplies the operand's actual
layout, and agreement is "the operand's layout, restricted to the survivors,
induces `R`". That is today's equality test — with `R` arriving instead of being
fetched.

### What it collapses

- `findStoreDestination` and `lookupMarkerFromTile`'s use from 2A: gone. The walk
  is what the direction gives for free.
- `populatePhysicalPropagationPatterns` stops taking `const MarkerByMemView &`, and
  every rule becomes a pure function of the op and the incoming fact. Narrowing
  that parameter from `PassContext` closed a real hole, but the parameter exists
  because the direction is wrong for one rule; this removes the cause.
- The init, for a reduce. A requirement reaches `tensor.empty`/`linalg.fill`, so
  `rebuildPhysicalInit` becomes a **retype** rather than a mint — the same thing
  `RewriteElementwisePattern` already does — and `canRebuildPhysicalInit` stops
  needing to be asked across the 2A/2B boundary. Reduce-scoped on purpose:
  `dispatchSource` rebuilds an init only in the Physical space, and a matmul is
  always Logical, so its accumulator is a slab of the wider container and is minted
  at slice extents either way. Nothing is left dead behind the
  rewrite, so the question `eraseDeadProducers` used to answer does not return.
- The markers themselves do **not** go away, and neither does
  `physMemViewToMarker`: the markers are the only source of layout truth, and 2B
  still resolves one from an access tile while rewriting. What changes is that the
  store's marker is held at the seed rather than walked to.

### What it owes answers to

- **Conflicting requirements.** A value reaching two stores with different layouts
  has two incompatible `want`s. Inexpressible today. The answer must be detect and
  fall back to logical plus bridges — not pick one.
- **Cycles.** The backward worklist needs the visit-stack detection the forward one
  already has, for the same reason.
- **The subset invariant.** `verifyPhysicalTypeAgreement` relates `physicalValues`
  to the forward map. A second map needs either its own check or a restatement
  covering both.
- **`want` without a satisfiable `have`.** A store requiring a layout of a value
  produced by an op that cannot be physical. This is the bridge case, and it should
  be the explicit answer of a rule rather than what happens when no rule fires —
  the driver reports a drained worklist as a fixpoint, so a permanent deferral
  reads as success.
- **Ordering.** Backward seeds from stores whose access tile Phase 1 physicalized,
  so both analyses still run before Phase 2B and the phase structure is unchanged.
- **A tie-break where both answers are legal.** Agreement/disagreement is a
  legality test, and it says nothing about a value with **no** `want` at all, where
  more than one answer verifies and the difference is cost. A free-typed op needs a
  rule for choosing, not just for rejecting.
- **A join at multi-operand `arith.*` ops.** The elementwise rule hands the same
  `R` to every operand, but nothing states that the operands' `have`s must agree
  with each other. `reconcileOperandSet` does that for source ops; a mixed pair at
  an `arith` op is a parse failure rather than a cost, and no rule covers it.

## Rejected inputs

| Input | Rejected because |
|---|---|
| A `mod` dim whose block extent is smaller than `S` | A `mod` dim cannot be sub-stick. |
| A stick-split on the indirect (row) dim of a gather | The indirect dim indexes rows one at a time and cannot be split across sticks. |
| Operands sharing a stick-split contraction axis where not all carry a marker | Cross-stick accumulation requires every operand on that axis to agree on the split. |
| A physical operand whose dim order differs from the op's by other than a permutation | Only a transpose is recognized; any other relation has no defined conversion. |
| A source-op operand produced by a reshape, expand_shape, collapse_shape or broadcast | Neither a physical value nor a plain logical one. A reshape changes the physical dim count while the marker's coordinate arrays are indexed per physical dim, so type and layout stop describing the same thing; a broadcast's target shape would have to be re-derived against the physical rank, which the pass does not yet do. Treating either as logical would silently compute the wrong slice. |
