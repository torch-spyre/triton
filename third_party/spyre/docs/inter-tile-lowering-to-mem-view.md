# Expression of Inter-Tile Communications and Lowering to Distributed Memory Views

`ktdp.construct_distributed_memory_view` is an existing KTIR feature. It composes
per-core partitions into a single memref whose domain is the union of their coordinate
sets, and its lowering is expected to resolve a global index into "selection of the
appropriate underlying memref and a local coordinate computation, potentially producing
explicit address calculation **and communication** when required by the target
architecture."

This document is about the two things that do not exist yet: **how a Triton kernel
expresses an inter-tile communication** (§2), and **how that expression lowers onto the
view** (§3). The view itself is the target, not the subject — where the design would
change it, §8 says so.

It is **one lowering target, not the whole subject**. `LowerInterTile` gains a second mode
and the existing `tt.inter_tile_reduce` path is left alone for now; §11 says how the two differ
and why that one is expected to be deprecated.

The communication in question is a **scratchpad relayout**: a tensor moving between
two ownership arrangements while it stays resident in the scratchpad. Both sides are
scratchpad-allocated and the
two work divisions differ — a producer and a consumer disagree about which core holds
which region, and the data has to be redistributed without a round trip through global
memory. That is why the kernel body needs scratchpad descriptors even when its entry inputs do
not.

Scope is the copy family — gather, scatter, all-to-all, relocation and broadcast — and,
because one view expresses both, cross-core reduction as well: #20's subject arrives here
as a *use* of the same mechanism rather than a mechanism of its own (§7).

State of play: the view has **no producer and no lowering** today — every reference to it
is the dialect definition, the README, and Dialect round-trip tests. That makes it cheap
to change, which is what §8's first decision turns on.

## 1. Definitions and assumptions

A **partition** is one core's share of a tensor: a coordinate region, plus the core
whose scratchpad holds it. In KTDP it is a `ktdp.construct_memory_view` carrying a
`coordinate_set` — the region in the tensor's global index space — and a `memory_space`
naming the holder.

```mlir
%p3 = ktdp.construct_memory_view %off, sizes: [64, 64], strides: [64, 1] {
    coordinate_set = #region_3,
    memory_space   = #ktdp.memory_space<ct_local, ct_id = 3>
} : memref<64x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
```

Two things about that type are load-bearing. The `ct_id` is part of it, so two
partitions in different cores' scratchpads have *different types* — that difference is the only
thing distinguishing otherwise identical operands. And the shape is static, so
partitions of differing shape cannot share one SSA value.

A **distributed view** composes partitions into a single memref whose domain is the
union of their coordinate sets. It moves nothing; it establishes a map from a global
coordinate to the partition holding it, plus the local coordinate within that partition —
the behaviour quoted at the top of this document.

A compose **grows the shares** along each dimension the work was divided on: a share's
extent there is replaced by the union over partitions. A share holding 128 rows of a 512-row
tensor composes to 512 rows, and a gathered 512×128 is still 512×128. The rank is never
changed, and the composed domain is a true statement of where data lives.

**`work_slices`** is a map from tile id to the region that tile holds. Each value is one
coordinate: one key per dimension the work was divided along, mapping to a slice index on it.
Every value carries the same keys.

```python
work_slices = {k:     {"out": k} for k in range(8)}    # region k on tile k
work_slices = {2 * k: {"mb": k}  for k in range(16)}   # region k on tile 2k
```

**Tile ids need not be contiguous**, and nothing derives them — they are the keys. The values
plus `axes` give each region's coordinates but say nothing about where a region lives, while
the compose must emit one view per region with a distinct `ct_id`.

The second case is a measured one, and it is what forces this. A `{mb: 512, in: 4096}` tensor is
held as sixteen regions of `{mb: 32}`, one owner each, and the owners are the **even** tiles:
`0, 2, 4 … 30`. The odd tiles hold no region of it at all — several of them receive without ever
producing. Neither positional form reaches that. Sixteen entries would claim the holders are
`0 .. 15`, wrong for every entry but the first; thirty-two entries would need sixteen holes, and
since every value carries the same keys there is no spelling for "nothing".

This is what KTDP's attribute has always been called — `coreIdToWkSlice`, tile id to work
slice — now meant literally rather than implied by position.

**Dim ids are the opposite: dense from 0.** Every index under a division key runs from 0 with
no gaps, which is what lets the slice count be read off rather than declared — it is one more
than the largest index appearing for that key anywhere in the map. That count does two things:
with the extents it turns a coordinate into a region, and with the share's own extent it gives
the composed domain, `count × share extent` on each divided dimension, which is how the
composed shape is known without being passed.

Eight regions, divided four ways on `x` and two ways on `n`, one region per tile:

```python
work_slices = {t: {"x": t // 2, "n": t % 2} for t in range(8)}

# tile 0 -> {"x": 0, "n": 0}     tile 4 -> {"x": 2, "n": 0}
# tile 1 -> {"x": 0, "n": 1}     tile 5 -> {"x": 2, "n": 1}
# tile 2 -> {"x": 1, "n": 0}     tile 6 -> {"x": 3, "n": 0}
# tile 3 -> {"x": 1, "n": 1}     tile 7 -> {"x": 3, "n": 1}
```

`x` has four slices and `n` has two — one more than the largest index under each key, and
neither count written down. With extents `{x: 512, n: 128}` the shares are 128 and 64 wide, so
tile 5, at `{x: 2, n: 1}`, holds `x[256:384]` by `n[64:128]`. That projection is all the
lowering needs, and §3 does exactly it.

Only the **source** side has a map. The destination arrangement is the offsets each instance
passes when it reads — see §2.

The **stick** is the hardware's contiguous innermost unit, 128 bytes, so `S = 128 /
itemsize` — 64 for fp16. Written `S` throughout, as in
[spyre-tensor-layouts.md](spyre-tensor-layouts.md).

### Assumptions this design relies on

1. **A kernel is one fused unit and its entry inputs live in global memory.** This is a
   property of where a fused region can begin. It says nothing about the kernel *body*,
   which is where a relayout appears — see §2.
2. **The lowering is pull.** §4.
3. **A source view's regions are disjoint or identical.** A slice table cannot express a
   partial overlap; identical means the holders are replicas. §5.
4. **This design composes no destination view.** Each destination holder writes into its
   own scratchpad, so there is nothing to compose on that side. That is a statement about
   what the lowering emits, not a restriction on KTDP: a `ktdp.store` through an access
   tile taken from a distributed view is not forbidden today, and nothing here proposes
   forbidding it. §4.
5. **Provenance carries order-freedom.** A value read from a distributed view carries no
   order guarantee, and the same fact identifies a fold over partitions as a cross-core
   reduction. §7, and §8's second decision.
6. **Scratchpad offsets reach the kernel as metadata**, not as arguments and not as
   constants: a distributed view can involve up to 32 distinct indices, so an argument per
   index is impractical.
7. **The participant set is statically known.** Every instance that contributes must reach
   the constructor with a value. Unconditional is fine, and so is a predicate on
   the tile id; a data-dependent predicate is not.

## 2. What the kernel provides

One call, `tl.make_distributed_descriptor`, whose result is a **descriptor** — not a
tensor — read like any other descriptor at the use site:

```python
partial    = x_desc.load([offset_m, offset_n])              # my share, a tensor
whole_desc = tl.make_distributed_descriptor(
    partial, work_slices=SRC, axes=[...], block_shape=[...],
)                                                          # a descriptor
mine       = whole_desc.load([dest_offset_m, dest_offset_n])   # my share under DEST
```

| Piece | Where it lives | What it carries |
|---|---|---|
| my **share** | `partial`, a value | the data this instance contributes |
| tile id → **region** | `work_slices` | which tile holds which region |
| dim → **key** | `axes` | which tensor dimension each partition key indexes |
| **extent** of one access | `block_shape` | what I take per `.load()` |
| memory-space **kind** | `tl.spyre_tensor_layout` | `global` \| `ct_local` |
| the N **views** | the lowering | `construct_memory_view` + `ct_id`, then the compose |

The kernel body names holders but never enumerates the grid: it states which tile holds each
region and nothing more — no branching on tile id, no grid shape, no knowledge of what any
other instance does. The lowering turns those statements into one `construct_memory_view` per
region with its `ct_id` (§3), which is what keeps the global vantage out of a kernel that
otherwise knows only its own `program_id`.

**There is no destination table.** Two things state that side, and neither is an input to the
compose or a verification target.

The **set** — which instances end up holding a copy — is stated by a guard: no guard means
every instance, and a guard on the tile id names a subset. The **arrangement** — which region
each of them holds — is the offset that instance passes to `.load()`, computed however the
kernel likes, whether plain arithmetic like `pid // 4` or a lookup of its own. That a kernel
may use a table does not make one part of this interface.

`construct_access_tile` takes **runtime** base indices, so the lowering passes the offset
straight through and never has to recover a table from it.

**Replication is not expressed at all**, and that is the point: the factor *is* the number of
instances that pass the same offset. Four instances resolving to one region means that region
has four holders. Nothing declares it, and irregular replication needs nothing either.

**Each pattern is a use, not a mode.** Gather reads the assembled region; scatter indexes a
slice; broadcast indexes the same region on every instance; a fold reduces over the
reduction dimension (§7). There is no `mode` enum and nothing new per pattern.

### What is new about this op

`make_tensor_descriptor` takes a **pointer** — the same value in every instance.
`make_distributed_descriptor` takes a **tensor** — a different value in every instance — so
its result depends on all of them. It is a collective constructor, and that is the one
genuinely new thing here. The *authoring* vantage stays per-instance: no instance names a
`ct_id`, none enumerates cores, and each writes only its own offsets. What spans instances
is the op's meaning.

Two consequences worth stating rather than discovering. The call is an implicit barrier over
the participant set, which is why assumption 7 exists. And `.load()` on the result is a
*transfer*, syntactically identical to a local descriptor read — the uniformity is the point
of the design, and the cost is that a communication no longer stands out at the use site.

### Memory space

`tl.spyre_tensor_layout` gains a memory-space attribute. The op is already a result-less
marker "consumed and erased by the lowering", carrying three parallel `DenseI64ArrayAttr`s
for the physical layout, so a memory space is one more attribute on an op whose job is
already to carry device-side facts about a descriptor:

```
let arguments = (ins
  TT_TensorDescType:$desc,
  DenseI64ArrayAttr:$phys_src,
  DenseI64ArrayAttr:$phys_op,
  DenseI64ArrayAttr:$phys_arg,
  <enum>:$memory_space          // new
);
```

The vocabulary is `global` and `ct_local`, matching `Ktdp_MemorySpaceKind`, which already
has exactly those two kinds plus an optional `ct_id`. It does **not** go on
`make_tensor_descriptor`, which is upstream generic.

This is needed from the start, not deferred. A relayout is scratchpad on both sides by
definition, so a kernel expressing one needs scratchpad descriptors immediately; assumption 1
constrains the signature, not the body.

## 3. Phases

The lowering runs in three phases, stated as a contract per phase. Its inputs are the
source map, `axes`, `block_shape`, the tensor's extents and the memory-space attribute;
its output is the redistributed data resident in each consumer's own scratchpad.

**Phase 1 — build the source view.**
Input: the source map, `axes`, the extents, and the memory space.
Lowering: turn each partition's coordinate into the region it owns, exactly as the §1
example does — slice width is the extent divided by the slice count, and the coordinate
picks which slice, with `axes` saying which tensor dimension each key indexes. Read the
holder off the entry's key (§1). Emit one
`construct_memory_view` per partition with the region as `coordinate_set` and the holder as
`ct_id`, then compose them with `construct_distributed_memory_view`.
Output: one distributed view whose domain is the composed whole, and which knows for every
coordinate which tile holds it.

The emitted views differ **only** in `coordinate_set` and `ct_id`; offsets, sizes and
strides are identical across them. Those two attributes are the entire content of the
distribution.

**Phase 2 — place the access.**
Input: the distributed view, and the offsets this instance passed to `.load()`.
Lowering: build a `construct_access_tile` on the view anchored at those offsets, with
`block_shape` as its extent. The offsets may be runtime values; the op takes dynamic base
indices, so nothing needs to be static here.
Output: an access tile naming the coordinates this instance will consume — and nothing has
moved yet. `construct_memory_view` and `construct_access_tile` are `Pure`: they
materialize addressing, not access, so a core may name another core's scratchpad without
anything crossing a core boundary.

**Phase 3 — transfer, and land if this is a copy.**
Input: the access tile.
Lowering: one `ktdp.load`, which **is** the transfer. Where the access tile's region spans
several partitions the lowering resolves it into per-partition transfers, so fan-in does
not multiply the load count — it stays one load. For a **copy** the load is followed by a
`ktdp.store` into a plain `ct_local` view with no `ct_id` — the executing tile's own
scratchpad — and that landing is mandatory: a received tile must be resident before a
compute unit can read it. A **reduction** is exempt, because the folded value can feed the
next computation directly. §7 turns on that exemption.
Output: for a copy, the data in place for the next operation to read locally; for a
reduction, a value.

Worked example for phase 1. Eight partitions, `out` divided eight ways, `x` uncut:

```
source map            {0: {"out": 0}, 1: {"out": 1}, ..., 7: {"out": 7}}
extents               {out: 512, x: 64}
slice width           512 / 8 = 64 on out; x is whole
=> partition 3 owns   out[192:256] by x[0:64], held by the tile at {"out": 3}
```

### Where the phases live

`LowerInterTile` gains a second mode. The existing path — `tt.inter_tile_reduce` to
`ktdp.inter_tile_produce` plus a delivery op — is not removed yet; §11 says when it should be.

## 4. The pull model

A distributed view is composed over the **source** partitions. Consumers read; producers
do not write remotely.

This is not merely a convention. Under pull a destination holder writes only into its own
scratchpad, so the destination side needs no distributed view and nothing has to be
composed there. One consequence worth naming: replication needs no expression of its own,
because a region needed by several consumers is just several reads, and reads do not
conflict.

A destination view remains *possible* where destination regions are disjoint, which would
make a relayout a view-to-view copy, and nothing in KTDP stops a store through an access
tile taken from one. It is never *necessary* here, and it is *meaningless* where
destinations are replicated: a written view would have two writers for one coordinate with
nothing saying which wins, whereas a view that is only read tolerates replicas because
either holder returns the same bytes. So this design composes none — and proposes no rule
against anyone else composing one.

Push would need a core to write into another core's scratchpad. Cross-core **reads** are
what the interconnect is described as supporting; remote writes are unverified. If they
exist, pull versus push becomes a performance question rather than an expressiveness one.

## 5. Several holders, and why that is welcome

A region may be held by more than one tile. That is expressible, it occurs, and it should
**not** be forbidden — it is information the backend can use.

**What the input format can and cannot produce.** Each partition gets one slice index per
divided dimension, so its region is one box. For any two partitions the boxes are therefore
either **identical** — every index equal, which is replication — or **disjoint**, because
differing slices do not intersect. There is no third case: a partially overlapping pair
would need a partition to own something other than a whole slice, or to own two of them, and
a slice table can express neither.

That is a stronger guarantee than a measurement. The dangerous shape is partial overlap,
where two holders share only some coordinates and could disagree about those. A slice
table **cannot describe it**, so the lowering cannot emit it, whatever future schedules
look like. The only overlap reachable is exact replication.

**Replication is a scheduling asset.** With a region held by two tiles and two consumers
needing it, the backend can pair them off:

```
region held by tiles 0 and 1;  tiles 2 and 3 both need it

  one holder only        free choice
  2 <- 0                 2 <- 0   ┐ two different source ports,
  3 <- 0   serialised    3 <- 1   ┘ transfers in parallel
```

So source selection should stay the backend's, and a canonical rule — "always read from
the lowest tile id" — would be actively worse than no rule: it funnels every reader onto
one holder and discards the parallelism the replication provides.

Note the contrast with a fold's result holder (§7). A reader has several valid holders and
the choice is worth leaving open; a fold's result has none to choose from, which is why the
kernel names it with a guard instead of the language stipulating one.

**What replication does not carry.** A slice table records ownership and cannot assert
*agreement*. Replicas and copies that have diverged have byte-identical declarations, so
"read from any holder" is sound exactly while the holders agree — which holds when a
region is written once by its holders and thereafter only read. It would stop holding if
something mutated one copy in place.

For reference, in the catalog pinned by torch-spyre#4300 every coordinate has exactly one
holder on the read side: no source region lists more than one owner, and no two distinct
regions overlap. Replication appears on the destination side — 97 of 130 records, up to 32
holders — where each holder writes and reads its own copy locally.

## 6. Why ownership stays tabulated

What the measured patterns rule out is deriving ownership from **axis counts** — an axis
name and a slice count, which is the form a named collective takes (§11). Ownership
is frequently strided rather than contiguous. In the measured records the cores feeding one
destination region sit two apart (cores 0 and 2, then 4 and 6, and so on), or eight apart
(0, 8, 16 and 24), or are drawn only from the even-numbered cores. No axis count reproduces
those, because the core-to-region assignment is not a function of how many pieces an axis
was cut into.

Note that `axes` is not that form. It maps a tensor dimension to a partition key and says
nothing about counts or holders; the striding lives in the coordinates, where a table can
carry it.

A closed form *does* reach them — with `mod` and `floordiv`, affine sets express strides of
that kind directly, and a survey finds a closed form for the consumer-to-source map in all
130 records. The table is kept anyway for two reasons: it is **always** expressible, so an
irregular pattern that no closed form covers still has somewhere to go; and it needs **no
inference**, where recovering a formula from 32 elements can fail, or over-generalise
silently into one that is right for the cases inspected and wrong for one that was not.

A table is also enough for what the lowering needs, which is one *concrete*
`coordinate_set` per partition — a box — not a parameterized family.

## 7. Cross-core reduction is a use of the same view

#20 tracks cross-core reduction. Here it is not a separate mechanism and needs nothing new:
compose the **inputs**, then write the reduction once over the composed axis. The fold is an
ordinary `tl.sum` or `tl.dot` over an ordinary axis, and the only inter-tile construct is the
descriptor.

**A reduction — `sum` over a distributed axis.** `x` is split on `n`, one share per instance:

```python
x_share = x_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N])   # [BLOCK_M, BLOCK_N]

SRC   = [{"n": i} for i in range(P)]
whole = tl.make_distributed_descriptor(
    x_share, work_slices=SRC,
    axes=[None, "n"],                            # composed: [BLOCK_M, P * BLOCK_N]
    block_shape=[BLOCK_M, P * BLOCK_N],
)
total = tl.sum(whole.load([0, 0]), axis=1)       # [BLOCK_M] -- all-reduce
```

`n` is a real dimension of the composed tensor, so reducing axis 1 reduces global `n` — the
local extent and the P partitions together, in one operation. Every instance runs it, which is
what makes it an all-reduce.

**A contraction — `matmul` with split K.** K appears in both operands at different positions,
so each operand gets its own descriptor:

```python
a_share = a_desc.load([0, pid_k * BLOCK_K])      # [BLOCK_M, BLOCK_K] -- my K-slice of A
b_share = b_desc.load([pid_k * BLOCK_K, 0])      # [BLOCK_K, BLOCK_N] -- my K-slice of B

SRC     = [{"k": i} for i in range(P)]
a_whole = tl.make_distributed_descriptor(
    a_share, work_slices=SRC,
    axes=[None, "k"],                            # composed: [BLOCK_M, P * BLOCK_K]
    block_shape=[BLOCK_M, P * BLOCK_K],
)
b_whole = tl.make_distributed_descriptor(
    b_share, work_slices=SRC,
    axes=["k", None],                            # composed: [P * BLOCK_K, BLOCK_N]
    block_shape=[P * BLOCK_K, BLOCK_N],
)
c = tl.dot(a_whole.load([0, 0]), b_whole.load([0, 0]))      # [BLOCK_M, BLOCK_N]
```

Two descriptors are what make K's differing position a non-issue: `axes=[None, "k"]` for A and
`axes=["k", None]` for B, each naming K where that operand carries it. `tl.dot` then contracts
the full K, and the cross-core reduction *is* the contraction.

The identifiers are arbitrary — `"n"` and `"k"` could be any strings — but which key is named
is not: it selects the division the composed axis spans.

**Nothing is composed from partials**, in either example, and that is what keeps the design
small. A partial has its reduction dimension already eliminated, so P partials cover the *same*
coordinates: the compose would emit P identical `coordinate_set`s differing only in `ct_id`,
which is the format's spelling for replicas, and §5 would then permit returning one addend
where their sum was meant. Composing inputs never reaches that state, because the shares are
disjoint on the divided dimension.

**The cost is a reassociation the backend has to find.** As written, every instance addresses
the whole composed input — all of `x` in the first example, all of A and B in the second. The
efficient schedule, reducing or multiplying the local slice and then combining P results, is a
reassociation that provenance permits (below). It is a heavier transform for `tl.dot` than for
`tl.sum`, since one dot must become P dots and a sum, and that is the main risk this form
carries.

`reduce_scatter` indexes before it folds, so instance `j` reads only its own slice of the
composed domain and reduces that.

**Which axis is reduced decides whether this is a collective.** A fold is cross-core exactly
when a reduced axis is one that `axes` named. Reducing an axis `axes` did not name is a gather
followed by an ordinary local reduction, not a collective at all.

### The destination set

Who ends up holding the result is not in the fold. `all_reduce` and `reduce_to_one` have
*identical* reduction expressions and differ only in how many instances want the answer. For
a copy that question answers itself: every destination holder ends with a mandatory
`ktdp.store` into its own scratchpad (§3), so the landings say who the holders are. A
reduction has no landing — the result may feed the next computation directly — so nothing in
the IR names them.

The kernel states it with a guard, and one mechanism spans the whole lattice:

```python
total = tl.sum(grp.load([0, 0, 0]), axis=0)          # all: every instance holds it

if tl.program_id(0) == 0:                            # one
    total = tl.sum(grp.load([0, 0, 0]), axis=0)

if tl.program_id(0) < 4:                             # a subset
    total = tl.sum(grp.load([0, 0, 0]), axis=0)
```

The guard reaches KTIR as an `scf.if` on `ktdp.get_compute_tile_id`, enclosing the access
tile, the load and the fold, so non-holders issue no transfer. Two things make an enclosing
construct the right carrier rather than an attribute. The saving is in the *transfer*, so
whatever carries it has to enclose the load. And `construct_access_tile` — the one op in that
chain that could plausibly hold an attribute — is general addressing machinery usable for any
tile: its subject is where data *is*, and who *executes* is a different question.

One dependency this creates, worth stating rather than discovering: the pipeline is
branch-averse today. `LowerScalarLoad` documents that no runtime branch is ever emitted for a
mask, and `DistributeWork` handles pids nested in an `scf.if` only defensively. A conditional
region around a load and a fold has to be acceptable downstream.

The direction of the default matters. Writing no guard means every instance holds the result,
which is definite and often what was meant — but it is also the expensive reading, reached by
writing nothing: a ring all-reduce takes 2(P−1) steps where reduce-to-one takes P−1, and the
fold runs on every instance rather than along one path. That asymmetry is why the guard is
part of the design and not an optimization hint. §8, decision 3.

### What provenance carries

That the reduced value came from a distributed view
is one def-use step away, and the composed partitions carry distinct `ct_id`s in their
memref types — so the fact is in the types, not in a name. It grants the *freedom* to
reorder (§8, decision 2) and supplies the *recognition* that this is a cross-core fold, from
the same observation. Algorithm selection therefore stays the backend's: ring, tree,
recursive halving or direct transfers, chosen by P and message size. A named mode would
pre-commit to one lowering; this does not.

Nor does the composed value ask for P blocks of scratchpad. It is a tensor in SSA form, not
an allocation, so a load feeding a reduction is ordinary producer-consumer dataflow, and
whether it is ever materialized is a bufferization outcome.

Note that reductions are absent from the pinned catalog only because it records scratchpad
relayouts specifically. They exist in the workload: an OpSpec can carry a matmul or
reduction that splits the reduction dimension.

## 8. Open decisions

1. **Whether the N partition operands can be replaced.** Enumerating every share is the
   cost that stands out. Two candidates. An **ownership relation** — an affine set relating
   coordinates to holding tiles — reaches the measured patterns, keeps holder and shape in
   the type, and stays statically checkable, at the cost of new attribute machinery. A
   **dynamic `ct_id`** is a far smaller change, but `ct_id` is part of the memref type, so
   partitions of different cores would share one type and the operand list would stop
   carrying ownership. Neither changes the Triton surface, and both change KTDP itself, so
   either lands upstream in `ktir-mlir-frontend` rather than here.
2. **May a kernel specify the reduction order?** Proposed: **no** — and stated as a
   property of the *value* rather than as a restriction on the *loop*. A tensor read from a
   distributed view carries no order guarantee, so a backend may re-associate whatever the
   kernel wrote. The alternative was to reject a written-out accumulation, on the grounds
   that `for k: acc += load(dview[k])` fixes a left-to-right fold that fp16's
   non-associativity then makes illegal to re-associate. That reading does not survive the
   pipeline: a Triton `for` becomes `scf.for` in the `tt` layer, and #62 proposes
   converting every `scf.for` into a `generic`, at which point the ordering is gone
   regardless of what the kernel wrote. Better to say the order was never promised than to
   police a form whose ordering the next pass discards. One case this does not cover:
   order-freedom is associativity *plus* commutativity, and while `add` / `max` / `mul`
   have both, a custom reducer region need not, and **argmax** is associative while its
   result depends on traversal order on ties.
3. **What to do when a fold's result is computed everywhere and wanted in one place.**
   Writing no guard means all-reduce, which is definite and frequently intended, so refusing
   it would reject valid kernels. The suspect form is narrower: a fold whose result is live
   *only* under a guard — all-reduce semantics with P−1 dead copies, and almost certainly
   meant as reduce-to-one. Proposed: sink the fold and its load into the guard wherever the
   value has that single conditional use, and diagnose where sinking is blocked, because a
   missed sink is silent, correct, and pays all-reduce cost. Sinking is legal when the
   predicate is analyzable — a comparison on the tile id, so the surviving instances are
   derivable — and nothing writes the source between the load and the guard.

## 9. Designs rejected

**Exposing the compose itself in Triton.** `construct_distributed_memory_view` takes memref
*values*, so surfacing *it* would put all N partitions in one function as SSA values — a
global vantage forced by the operand list, which collides with SPMD. A descriptor avoids
that by taking one value, this instance's share, plus a table, and leaving the enumeration
to the lowering.

**Emulating scratchpad-to-scratchpad through global memory.** Functionally it works, and the
descriptor-layout machinery would carry the form change. But if every intermediate goes
through global memory there is nothing left for a relayout to do — that round trip is
exactly what this exists to remove. It would pass expression and numerical checks while
failing on emitted accesses, with every number green.

**Deriving ownership from an axis and a count.** §6.

**A mode enum per communication pattern.** §11.

## 10. Rejected inputs

The lowering should refuse, rather than guess:

- **A tile → coordinate table whose length does not match the launch grid.** `prod(grid)
  == len(work_slices)` is a launch obligation the kernel body cannot enforce. It binds that
  role only — the source map has one entry per region, which may be far fewer than the
  grid.
- **A work slice table with differing key sets** across its elements — every element of a
  `work_slices` list must have identical keys.
- **An `axes` entry naming a key no partition dict carries**, or a partition key that
  `axes` never places. The two have to agree, and neither can be inferred from the other.
- **A source key outside the launch grid.** The key is the holder (§1), and a tile the launch
  never creates names a share nobody holds.
- **A gap in a dimension's indices.** Dim ids are dense from 0 (§1); a missing index leaves a
  region with no holder and makes the slice count wrong.
- **A `ct_id` on a `global` memory space.** `ct_id` is meaningful only for `ct_local`.
- **A reducer region that is not known to be commutative**, where the reduction is over a
  distributed view. This is decision 2's residual case: order-freedom needs commutativity
  as well as associativity, and a custom region can supply neither. With no order to
  promise, the lowering should say so rather than quietly pick one.

## 11. How this differs from `tl.inter_tile`, which it supersedes

`tl.inter_tile(x, axis, combiner, mode, work_slices=...)` is the shipped surface for cross-core
reduction: one named collective call, a `mode` enum, and a tensor result. It is expected to be
**deprecated**, because every mode it offers is a use of the descriptor (§7) while the copy
family is not expressible through it at all.

**Its `work_slices` is a different shape, and that is the difference to keep straight.** There
it is a positional list with one element per tile, dense over the grid, so
`len(work_slices) == prod(grid)` is a launch obligation and position carries the tile id;
`tl.wk_slice_coord` reads it at `program_id`, letting a kernel recover its own coordinate
without hand-coding a radix. That form works because the list is **complete** — every tile
appears, so nothing is left implicit. The map of §1 is never complete: only holders appear, and
they need not be contiguous. A complete enumeration may use position; a partial one must name
its keys, which is why the two have different shapes rather than this being a matter of taste.

**What lowers there today.** `all_reduce` and `reduce_to_one`. `reduce_scatter` and `broadcast`
are accepted by the Python surface and rejected by the pass
([`LowerInterTile.cpp:340-346`](../lib/Dialect/KTDP/Transforms/LowerInterTile.cpp)), as are
custom combiner regions (`:356`), non-contiguous groups (`:180-183`) and non-uniform `pick0`
layouts for `reduce_to_one` (`:253-255`). Two of the four modes it declares do not work.

**What naming the collective buys, and does not.** It buys directness: `mode` maps onto the
produce/delivery pair with nothing to recognize, and the K-split matmul reduce ring is validated
through it at `SENCORES` 4 and 8. It does not buy expressiveness. Provenance already identifies
a fold (§7) without committing to one lowering — a mode picks the delivery pair, where
provenance leaves ring, tree and direct transfer available — and a mode per pattern does not
scale to the copy family, where gather, scatter, all-to-all, relocation and broadcast would each
need one while the descriptor needs none.

**When to retire it.** When the descriptor path passes the tests `tl.inter_tile` passes now —
not before, and without extending it in the meantime. It works today and the descriptor path
lowers not at all, so sequencing is the only argument for keeping it.
