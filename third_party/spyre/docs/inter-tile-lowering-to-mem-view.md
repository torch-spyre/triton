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
and the existing `tt.inter_tile_reduce` path stays exactly as it is; §11 says what that
path is and why it is not the design.

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

A **work slice table** is a list of dicts. Each dict holds one grid coordinate: one key
per dimension the work was divided along, mapping to a slice index on it. Every element
carries the same keys. It is what `tl.inter_tile` already accepts as `work_slices`.

The same kind of list appears in **two roles**, and they differ in length:

| Role | Indexed by | Length | Read by |
|---|---|---|---|
| tile → coordinate | tile id | `prod(grid)` | `tl.wk_slice_coord` |
| partition → region | partition number | number of partitions | the compose |

The first says where each tile sits in the grid; every tile needs an element, so its length
is a launch obligation. The second says which regions a view is composed from — one element
per **region**, not per tile. A tensor held by 8 of 32 cores gives a list of 8, and a
broadcast source gives a list of 1. Conflating the two would force every source
distribution to span the whole grid, which is the case broadcast is not.

A partition's **holder** is found by matching the two: the tile whose coordinate dict equals
the partition's dict is the core whose scratchpad holds it. So both lists are in play even
though only the second is passed to the compose.

The slice count along a dimension is not declared alongside the table; it is one more than
the largest index appearing for that key anywhere in the list. That count is what turns a
coordinate into a region.

Eight tiles, work divided four ways on `x` and two ways on `n`:

```python
work_slices = [{"x": t // 2, "n": t % 2} for t in range(8)]

# tile 0 -> {"x": 0, "n": 0}     tile 4 -> {"x": 2, "n": 0}
# tile 1 -> {"x": 0, "n": 1}     tile 5 -> {"x": 2, "n": 1}
# tile 2 -> {"x": 1, "n": 0}     tile 6 -> {"x": 3, "n": 0}
# tile 3 -> {"x": 1, "n": 1}     tile 7 -> {"x": 3, "n": 1}
```

`x` has four slices and `n` has two — one more than the largest index under each key, and
neither count written down. With extents `{x: 512, n: 128}` the slices are 128 and 64 wide,
so tile 5, at `{x: 2, n: 1}`, owns `x[256:384]` by `n[64:128]`. That projection is all the
lowering needs from the table, and §3 does exactly it.

A redistribution needs only the **source** list. The destination arrangement is the offsets
each instance passes when it reads — see §2.

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
   `tl.wk_slice_coord`; a data-dependent predicate is not.

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
| partition → **region** | `work_slices` | which regions the view is composed from |
| dim → **key** | `axes` | which tensor dimension each partition key indexes |
| **extent** of one access | `block_shape` | what I take per `.load()` |
| memory-space **kind** | `tl.spyre_tensor_layout` | `global` \| `ct_local` |
| the N **views** | the lowering | `construct_memory_view` + `ct_id`, then the compose |

The kernel body needs **no** `ct_id` and no knowledge of the grid. Holder identities are
derived during lowering by matching partition coordinates against the launch table (§1),
which is what keeps the global vantage out of a kernel that only knows its own
`program_id`.

**There is no destination table.** The destination arrangement is the offset each instance
passes to `.load()`, computed from its own `tl.wk_slice_coord`. Every instance reads a
different region of the same composed descriptor, and that difference *is* the relayout.
This is not a convenience: `construct_access_tile` takes **runtime** base indices, so the
lowering passes the offset straight through and never has to recover a table from it.

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
partition list, `axes`, `block_shape`, the tensor's extents and the memory-space attribute;
its output is the redistributed data resident in each consumer's own scratchpad.

**Phase 1 — build the source view.**
Input: the partition list, `axes`, the extents, and the memory space.
Lowering: turn each partition's coordinate into the region it owns, exactly as the §1
example does — slice width is the extent divided by the slice count, and the coordinate
picks which slice, with `axes` saying which tensor dimension each key indexes. Resolve the
holder by matching that coordinate against the launch table. Emit one
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
partition list        [{"out": 0}, {"out": 1}, ..., {"out": 7}]
extents               {out: 512, x: 64}
slice width           512 / 8 = 64 on out; x is whole
=> partition 3 owns   out[192:256] by x[0:64], held by the tile at {"out": 3}
```

### Where the phases live

`LowerInterTile` gains a second mode. The existing path — `tt.inter_tile_reduce` to
`ktdp.inter_tile_produce` plus a delivery op — is not removed. §11.

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

## 6. Why the partition list stays a table

What the measured patterns rule out is deriving ownership from **axis counts** — an axis
name and a slice count, which is the form `tl.inter_tile`'s `axis` parameter has. Ownership
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

if tl.wk_slice_coord(WS, "p") == 0:                  # one
    total = tl.sum(grp.load([0, 0, 0]), axis=0)

if tl.wk_slice_coord(WS, "p") < 4:                   # a subset -- no mode spells this
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
   predicate is analyzable — `tl.wk_slice_coord`'s column is `constexpr`, so the surviving
   instances are derivable — and nothing writes the source between the load and the guard.

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
  role only — a partition → region list is as long as there are regions, which may be
  fewer than the grid.
- **A work slice table with differing key sets** across its elements — every element of a
  `work_slices` list must have identical keys.
- **An `axes` entry naming a key no partition dict carries**, or a partition key that
  `axes` never places. The two have to agree, and neither can be inferred from the other.
- **A partition coordinate matching no tile.** The holder is found by matching against the
  launch table (§1); an unmatched coordinate names a share nobody holds.
- **A `ct_id` on a `global` memory space.** `ct_id` is meaningful only for `ct_local`.
- **A reducer region that is not known to be commutative**, where the reduction is over a
  distributed view. This is decision 2's residual case: order-freedom needs commutativity
  as well as associativity, and a custom region can supply neither. With no order to
  promise, the lowering should say so rather than quietly pick one.

## 11. `tl.inter_tile`, the alternative

`tl.inter_tile(x, axis, combiner, mode, work_slices=...)` is the shipped surface: one named
collective call, a `mode` enum, one work-slice table, and a tensor result.

**What lowers today.** `all_reduce` and `reduce_to_one`. `reduce_scatter` and `broadcast`
are accepted by the Python surface and rejected by the pass
([`LowerInterTile.cpp:340-346`](../lib/Dialect/KTDP/Transforms/LowerInterTile.cpp)), as are
custom combiner regions (`:356`), non-contiguous groups (`:180-183`) and non-uniform `pick0`
layouts for `reduce_to_one` (`:253-255`).

**Its advantage.** It names the collective, so nothing has to be recognized: `mode` maps
directly onto the produce/delivery pair, and the K-split matmul reduce ring is validated
through it at `SENCORES` 4 and 8.

**Why it is not the design.** Naming buys less than it appears to, because provenance
already identifies the fold (§7) and does so without committing to one lowering — a mode
picks the delivery pair, where provenance leaves ring, tree and direct transfer all
available. And naming does not scale to the copy family: gather, scatter, all-to-all,
relocation and broadcast would each need a mode, where a descriptor needs none. Two of the
four modes already declared do not lower.

**Why it stays anyway.** Sequencing, not design. It works and is validated; the descriptor
path lowers not at all. Retire it when the descriptor path passes the same tests — not
before, and without extending it in the meantime.
