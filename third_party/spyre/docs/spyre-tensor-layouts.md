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

Whether the output can be physicalized is a property of the op, not of the
shapes:

| Op | Output can be physical | Why |
|---|---|---|
| elementwise (`arith`, `linalg.generic`) | Yes | Rank-agnostic; the physical type propagates unchanged. |
| `linalg.reduce` | Yes | `dimensions` is a list, so a stick-split reduced axis is expressible in one op. |
| `ktdp.store` | Yes | `AnyTensor`; the verifier checks only that data-tile and access-tile shapes agree. |
| `linalg.matmul`, `linalg.batch_matmul` | No | Contracts exactly one `K` axis, so a split `K` needs accumulation across sticks. |

An annotated output is what makes the first outcome available: the store's
access tile is physicalized alongside it, so the two sides agree. A reduce whose
output descriptor carries no marker takes the second outcome — the result stays
`tensor<64xf32>` and an `scf.for` accumulates into it.

`linalg.reduce` and `ktdp.store` are therefore one category, differing only in
which outcome their output lands in. There is no separate sink path.

The first outcome, for a reduce over a stick-split axis with physical input
`[2, 64, 64]` and physical output `[64]`:

```mlir
%r = linalg.reduce ins(%phys : tensor<2x64x64xf32>) outs(%acc : tensor<64xf32>) dimensions = [0, 2]
```

This is legal because `dimensions` is `DenseArrayStrictlySorted` with no
adjacency or trailing-position requirement.

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

The driver's convergence result is checked, so a pattern that stays matchable
fails the pass with a diagnostic instead of spinning to the iteration cap and
reporting success. Note what that does not catch: if every pattern declines, the
worklist drains and the driver calls that a fixpoint, so a permanent deferral is
reported as success. Deferral must therefore be a prediction the analysis
supports, never a guess.

## Rejected inputs

| Input | Rejected because |
|---|---|
| A `mod` dim whose block extent is smaller than `S` | A `mod` dim cannot be sub-stick. |
| A stick-split on the indirect (row) dim of a gather | The indirect dim indexes rows one at a time and cannot be split across sticks. |
| Operands sharing a stick-split contraction axis where not all carry a marker | Cross-stick accumulation requires every operand on that axis to agree on the split. |
| A physical operand whose dim order differs from the op's by other than a permutation | Only a transpose is recognized; any other relation has no defined conversion. |
| A source-op operand produced by a reshape, expand_shape, collapse_shape or broadcast | Neither a physical value nor a plain logical one. A reshape changes the physical dim count while the marker's coordinate arrays are indexed per physical dim, so type and layout stop describing the same thing; a broadcast's target shape would have to be re-derived against the physical rank, which the pass does not yet do. Treating either as logical would silently compute the wrong slice. |
