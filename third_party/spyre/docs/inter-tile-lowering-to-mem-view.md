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

It is **one lowering target, not the whole subject**. The cross-tile reduction it
replaces, `tl.inter_tile`, has been removed; §11 says what that surface was and why it is
not the design.

The communication in question is a **scratchpad relayout**: a tensor moving between
two ownership arrangements while it stays resident in the scratchpad. Both sides are
scratchpad-allocated and the
two work divisions differ — a producer and a consumer disagree about which core holds
which region, and the data has to be redistributed without a round trip through global
memory. That is why the kernel body needs scratchpad placement even when its entry inputs
do not.

Scope is the copy family — gather, scatter, all-to-all, relocation and broadcast — and,
because one view expresses both, cross-core reduction as well: #20's subject arrives here
as a *use* of the same mechanism rather than a mechanism of its own (§7).

**Both sides of a relayout are composed in KTIR.** The source is a distributed view the
kernel reads from; the destination is a distributed view each receiving tile stores into,
at a region of its own. The Triton surface states the source; the lowering identifies the
destination from what the kernel states about who receives what (§3).

State of play: nothing in tree produces or lowers the view yet. Thirty-five relayouts a
real model performs have been written in this form as KTIR, from the model's recorded
scratchpad placements, and they parse and verify; they are the evidence behind several
statements below, and are cited by what they show rather than by name.

## 1. Definitions and assumptions

A **partition** is one core's share of a tensor: a coordinate region, plus the core
whose scratchpad holds it. In KTDP it is a `ktdp.construct_memory_view` carrying a
`coordinate_set` — the region in the composed view's index space — and a `memory_space`
naming the holder. KTIR does not require a particular layout; the KTIR in this document
is written in the physical one, the form `RewriteDescriptorLayout` gives a memory view: a
`[64, 128]` share sticked on its last dimension is `[2, 64, 64]`, and its coordinate set
and access tiles are in that same physical index space.

```mlir
%p3 = ktdp.construct_memory_view %off, sizes: [2, 64, 64], strides: [4096, 64, 1] {
    coordinate_set = #region_3,
    memory_space   = #ktdp.memory_space<ct_local, ct_id = 3>
} : memref<2x64x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
```

Two things about that type are load-bearing. The `ct_id` is part of it, so two
partitions in different cores' scratchpads have *different types* — that difference is the only
thing distinguishing otherwise identical operands. And the shape is static, so
partitions of differing shape cannot share one SSA value.

A **distributed view** composes partitions into a single memref whose domain is the
union of their coordinate sets. It moves nothing; it establishes a map from each element
of the composed index space to the partition holding it, plus its position within that
partition —
the behaviour quoted at the top of this document. The view maps each **element** — each
point of its index space — to *one* partition, so **the partitions of one view are
disjoint**. Since each partition has one holder and each tile holds at most one partition
of a view, every element of the view is held by one tile. An **access**, a block of
elements named by an access tile, may still span several partitions; that is what a gather
is (§3, phase 4). The op leaves overlapping coordinate sets unspecified, and this design
never emits them.

The **gather dims** of a composed view are the dimensions along which its partitions are
laid out. Every partition has the same box shape, and on each gather dim the boxes sit side
by side, so the composed extent there is the box's extent times the number of positions,
and every other dimension keeps the box's extent. The rank is never changed. A share
holding 128 rows of a 512-row tensor composes to 512 rows, and a tensor split along two
dimensions has two gather dims: 32 shares of `[1, 192]` laid out 8 by 4 compose to
`[8, 768]`.

**`work_slices`** says which tile holds which region. It is a **dense list with one entry
per tile**, so `len(work_slices) == prod(grid)` and the position is the holder: entry `k`
is the region tile `k` holds, as a dict with one key per dimension the work was divided
along, mapping to a slice index on it. Entry `k` is **`None`** where tile `k` holds no
region of this view, and **`{}`** where it holds the whole of it undivided, which only a
single holder can. Every non-`None` entry carries the same keys, and no two entries are
equal, since no two partitions overlap.

The slice count along a dimension is not declared alongside the list; it is one more than
the largest index appearing for that key anywhere in it. So **the indices are dense even
where the positions are not**: under each key they run `0 .. n-1` with no gap, which is
what keeps the composed view contiguous however sparse its holders are.

Eight tiles, work divided four ways on `x` and two ways on `n`:

```python
work_slices = [{"x": 0, "n": 0}, {"x": 0, "n": 1}, {"x": 1, "n": 0}, {"x": 1, "n": 1},
               {"x": 2, "n": 0}, {"x": 2, "n": 1}, {"x": 3, "n": 0}, {"x": 3, "n": 1}]
```

`x` has four slices and `n` has two — one more than the largest index under each key, and
neither count written down. With extents `{x: 512, n: 128}` the slices are 128 and 64 wide,
so tile 5, at `{x: 2, n: 1}`, owns `x[256:384]` by `n[64:128]`. That projection is all the
lowering needs from the list, and §3 does exactly it.

Holders need not be `0 .. N-1`. A `[512, 64]` tensor held as 8 regions by every fourth
tile is

```python
work_slices = [{"mb": 0}, None, None, None, {"mb": 1}, None, None, None,
               {"mb": 2}, None, None, None, {"mb": 3}, None, None, None,
               {"mb": 4}, None, None, None, {"mb": 5}, None, None, None,
               {"mb": 6}, None, None, None, {"mb": 7}, None, None, None]
```

and such holders occur: across the 130 relayouts of the recorded model, many owner sets
are strided, by 2, 4, 8 or 16, and none is irregular.

Two words are kept apart throughout. A **coordinate** is what an access names: the
position in a composed view at which it is anchored, which with `block_shape` names a block
of elements. An **offset** is an address in a scratchpad — where a partition's buffer
starts. A single point of a view is an **element**.

The **stick** is the hardware's contiguous innermost unit, 128 bytes, so `S = 128 /
itemsize` — 64 for fp16. Written `S` throughout.

### Assumptions this design relies on

1. **A kernel is one fused unit and its entry inputs live in global memory.** This is a
   property of where a fused region can begin. It says nothing about the kernel *body*,
   which is where a relayout appears — see §2.
2. **Nothing writes remotely.** Cross-core reads move the data; every store lands in the
   executing tile's own scratchpad. §4.
3. **A view's partitions are disjoint.** A slice list cannot express a partial overlap, and
   with no two entries equal there is no full one either. §5.
4. **Both sides are composed, and the destination is identified, not stated.** The
   kernel says who receives (an `if`), what each receives (the coordinates it reads) and
   where it lands (a pin); the lowering composes the destination from that. §3.
5. **Provenance carries order-freedom.** A value read from a distributed view carries no
   order guarantee, and the same fact identifies a fold over partitions as a cross-core
   reduction. §7.
6. **A partition's scratchpad offset is one offset, the same on every core**, stated by the
   pin of the value that partition holds. The partitions of one side then differ only in
   `coordinate_set` and `ct_id`. §8 records where recorded placements do not fit this.
7. **Who takes part is statically known.** Every tile holding a region of a source must
   reach the constructor with its share, and every `if` that selects tiles is a function of
   `tl.program_id` and constants. A data-dependent condition is not.

## 2. What the kernel provides

One call, `tl.make_distributed_descriptor`, whose result is a **descriptor** — not a
tensor — read like any other descriptor at the use site. It composes the shares each tile
has pinned into its own scratchpad; reading it at the coordinates of the region a tile
ends up holding is the transfer, and pinning the result is the landing:

```python
x     = x_desc.load([pid * 16, 0])
share = tl.exp(x)                                       # my share, computed on-chip
tl.spyre_pin(share, "ct_local", offset=SRC_OFF)         # where my share lives
whole = tl.make_distributed_descriptor(
    share, work_slices=SRC, axes=["mb", None], block_shape=[...],
)                                                       # a descriptor
if <I receive>:
    mine = whole.load([...])                            # the transfer
    tl.spyre_pin(mine, "ct_local", offset=DST_OFF)      # where it lands
```

| Piece | Where it lives | What it carries |
|---|---|---|
| my **share** | `share`, a value | the data this tile contributes |
| its **offset** | the pin on the share | where the share sits in this tile's scratchpad |
| tile → **region** | `work_slices` | which tile holds which region; `None` for none |
| dim → **key** | `axes` | which tensor dimension each key indexes |
| **extent** of one access | `block_shape` | what a tile takes per `.load()` |
| who **receives** | an `if` on `tl.program_id` | the tiles that load and land |
| what each **receives** | the `.load()` coordinates | the region of the composed view it reads |
| where it **lands** | the pin on the loaded value | the offset in the receiving tile's own scratchpad |
| the **views** | the lowering | one `construct_memory_view` per partition with `ct_id`, then the compose, on both sides |

The physical layout is stated once per descriptor, by `tl.spyre_tensor_layout`; the
kernel otherwise works in logical coordinates.

The kernel body needs **no** `ct_id` and no knowledge of the grid beyond the list. A
region's holder is its position in `work_slices`, which is what keeps the global vantage
out of a kernel that otherwise knows only its own `program_id`.

**There is no destination table.** The destination is stated by three things the kernel
already writes: the `if` that selects the receiving tiles, the coordinates each of them
reads, and the pin that lands what it read. Every receiver reads a region of the same
composed descriptor, and that difference *is* the relayout. The coordinates and the `if`
condition must be **static per tile**: the backend pairs each read with the tile that holds
what it reads, and the lowering composes the destination, from exactly that per-tile
evaluation.

**Each pattern is a use, not a mode.** Gather reads the assembled region; scatter indexes a
slice; broadcast reads the same region on several tiles; a fold reduces over the
reduction dimension (§7). There is no `mode` enum and nothing new per pattern.

### What is new about this op

`make_tensor_descriptor` takes a **pointer** — the same value in every instance.
`make_distributed_descriptor` takes a **tensor** — a different value in every instance — so
its result depends on all of them. It is a collective constructor, and that is the one
genuinely new thing here. The *authoring* vantage stays per-instance: no instance names a
`ct_id`, none enumerates cores, and each writes only its own coordinates. What spans
instances is the op's meaning.

One consequence worth stating rather than discovering: `.load()` on the result is a
*transfer*, syntactically identical to a local descriptor read — the uniformity is the point
of the design, and the cost is that a communication no longer stands out at the use site.

### Where the share lives

A share is placed by `tl.spyre_pin`, as
[lx-placement.md](lx-placement.md) sets out under "Composing pinned shares across
cores": the pin is what gives each partition its scratchpad offset, and nothing else
supplies one. A landing is pinned for the same reason and one more: a tile received over
the interconnect has to be resident in the scratchpad before a compute unit can read it,
so a copy's landing cannot stay in registers. The pin is therefore required on both.

**What a pin does, whatever produced the value.** A pin names a value, and its annotation
is attached to the op that has that value as its result — a compute such as `tl.exp`, or
the load of an HBM descriptor, alike; the kind of op makes no difference. The value is
stored into the scratchpad at the pin's offset directly after that op, the earliest point
it exists, and every use of it then reads it back from there. That store is not an extra
copy: a relayout moves data over the scratchpad interconnect, so a share has to be
resident in the scratchpad before it can be sent, and the store is how it gets there.
Where the pinned value was loaded from global memory, what is placed is the loaded copy;
the tensor in global memory is unaffected.

## 3. Phases

The lowering runs in four phases, stated as a contract per phase. Its inputs are
`work_slices`, `axes`, `block_shape`, the share's shape, the pins, and the `if` and
coordinates of each access; its output is the redistributed data resident in each
receiver's own scratchpad.

**Phase 1 — build the source view.**
Input: `work_slices`, `axes`, the share's shape, and the share's pin.
Lowering: turn each non-`None` entry into the region it denotes, exactly as the §1 example
does — slice width is the share's extent, and the index picks which slice, with `axes`
saying which tensor dimension each key indexes. The holder is the entry's position. Emit
one `construct_memory_view` per partition with the region as `coordinate_set`, the holder
as `ct_id`, and the pin's offset, then compose them with
`construct_distributed_memory_view`.
Output: one distributed view whose domain is the composed whole, and which knows for every
coordinate which tile holds it.

The emitted views differ **only** in `coordinate_set` and `ct_id`; offsets, sizes and
strides are identical across them. Those two attributes are the entire content of the
distribution.

**Phase 2 — place the access.**
Input: the distributed view, and the coordinates each tile passes to `.load()`.
Lowering: build a `construct_access_tile` on the view anchored at those coordinates, with
`block_shape` as its extent. The coordinates are evaluated per tile, and must fold to
constants there.
Output: an access tile naming the coordinates this tile will consume — and nothing has
moved yet. `construct_memory_view` and `construct_access_tile` are `Pure`: they
materialize addressing, not access, so a core may name another core's scratchpad without
anything crossing a core boundary.

**Phase 3 — compose the destination.**
Input: the `if` around each landing, the coordinates and block of the load it lands, and
the landing's pin.
Lowering: per tile, the `if` says whether it receives and the load says the box it
receives, its coordinate folded to a constant. Whether two receivers' boxes are the same is
decided by those values, not by whether the coordinate mentions `program_id`:
`[(pid // 4) * 64, 0]` on 32 receivers gives each box four receivers.

Where every receiver's box is distinct — a permutation, a scatter, an all-to-all, or a
gather onto one tile — the destination is determined. Its gather dims are found dimension
by dimension: a dimension on which the receivers' starts take more than one value is a
gather dim, one on which they all agree is not. On each gather dim the starts must be
spaced by the block's extent and every combination of them must occur exactly once, so the
boxes tile their union; the destination is that union, in its own coordinates starting at
0. A permutation of 32 regions of `[1, 192]` laid out over `j` and `y`, loaded at
`[pid % 8, (pid // 8) * 192]`, has starts `0 .. 7` on `j` and `0, 192, 384, 576` on `y`:
gather dims `j` and `y`, 8 × 4 = 32 boxes.

Where several receivers receive the **same** box — a broadcast, or a gather onto several
tiles — the starts say nothing about where the copies go, and their copies are laid out
apart along a dimension, which then becomes a gather dim; the receivers of one box are laid
out in ascending tile id. A destination can have both: 8 distinct boxes with four receivers
each has a gather dim from its starts, and the copies need one more. Emit one `construct_memory_view` per receiver, at
the landing pin's offset, with the receiver as `ct_id`, and compose them.
Output: the composed destination, and for every receiver the coordinate of its own region.

**Where the copies go is the lowering's choice.** The kernel does not state it: two
receivers that read the same box have no coordinate that tells their copies apart. Any
placement is correct provided that

1. nothing reads the composed destination as a global view — a later relayout composes the
   landings again, from its own `work_slices`, so the placement is never observed;
2. the choice is deterministic, so the same kernel lowers to the same KTIR;
3. the result is a valid composed view: the destination's regions are disjoint and tile its
   composed type exactly.

The third rules out one case: if the boxes of a destination have **differing** numbers of
copies, no placement tiles a box, and the kernel is rejected (§10).

**Phase 4 — transfer and land.**
Input: the access tile and the composed destination.
Lowering: one `ktdp.load`, which **is** the transfer. Where the access tile's region spans
several partitions the lowering resolves it into per-partition transfers, so fan-in does
not multiply the load count — it stays one load. For a **copy** the load is followed by a
`ktdp.store` into the composed destination, at the receiving tile's own region, and both
sit under an `scf.if` on `ktdp.get_compute_tile_id` that admits the receivers. The landing
is mandatory, for the reason §2 gives. A **reduction** is exempt, because the folded value
can feed the next computation directly. §7 turns on that exemption.
Output: for a copy, the data in place for the next operation to read locally; for a
reduction, a value.

### Checks before the analysis

The lowering checks what the kernel states before it analyzes it, and reports a failure
against the Triton line that caused it, so that a kernel the design does not admit is never
mistaken for a lowering error:

- every `if` condition and every access coordinate folds to a constant on every tile;
- every block lies inside its composed view;
- every copy's landing is pinned, at the same offset on every receiver;
- `work_slices` is well formed (§10).

The analysis of phase 3 runs only on a kernel that passes, and its result — the receivers,
their boxes and their regions in the destination — is stated in the IR before KTIR is
emitted, so that it can be tested on its own rather than only through the KTIR it produces.

### A worked example

A `[4, 64]` fp16 tensor, sticked on its last dimension, held as one row per tile on a
grid of 4, and gathered whole onto the two even tiles:

```python
SRC = [{"mb": 0}, {"mb": 1}, {"mb": 2}, {"mb": 3}]

x_desc = tl.make_tensor_descriptor(x_ptr, shape=[4, 64], strides=[64, 1], block_shape=[1, 64])
tl.spyre_tensor_layout(x_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])
share = x_desc.load([pid, 0])
tl.spyre_pin(share, "ct_local", offset=SRC_OFF)

whole = tl.make_distributed_descriptor(share, work_slices=SRC, axes=["mb", None],
                                       block_shape=[4, 64])      # composed: [4, 64]
if pid % 2 == 0:
    mine = whole.load([0, 0])
    tl.spyre_pin(mine, "ct_local", offset=DST_OFF)
```

Physically the tensor is `(out // 64, mb, out % 64) = [1, 4, 64]`. The source is four views
of `[1, 1, 64]`, one per tile, composed to `[1, 4, 64]`. Tiles 0 and 2 receive the same box,
the whole tensor, so the destination has two copies to lay apart. Laid out along `mb`:

```mlir
// copy r on tile 2r covers mb [4r, 4r + 4)
%d0 = ktdp.construct_memory_view %dst, sizes: [1, 4, 64], strides: [256, 64, 1]
    {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : ...
%d1 = ktdp.construct_memory_view %dst, sizes: [1, 4, 64], strides: [256, 64, 1]
    {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>} : ...
%to = ktdp.construct_distributed_memory_view (%d0, %d1 : ...) : memref<1x8x64xf16>

scf.if %part {                                   // tid % 2 == 0
  %rt  = ktdp.construct_access_tile %from[%c0, %c0, %c0] {...}
  %val = ktdp.load %rt
  %wt  = ktdp.construct_access_tile %to[%c0, %o_mb, %c0] {...}   // %o_mb = 4 * (tid / 2)
  ktdp.store %val, %wt
}
```

and laid out along `out // 64` instead, every line is the same except:

```mlir
// copy r on tile 2r covers out // 64 = r
%to = ktdp.construct_distributed_memory_view (%d0, %d1 : ...) : memref<2x4x64xf16>
  %wt  = ktdp.construct_access_tile %to[%q, %c0, %c0] {...}      // %q = tid / 2
```

with `#dst1` naming `d0 = 1` rather than `d1 ∈ [4, 8)`. Both are correct: the views are
disjoint and tile the composed type, the guard admits exactly tiles 0 and 2, and each stores
into its own region. They agree on everything the kernel states and differ only in what it
does not, which is why the choice is the lowering's.

### Where the phases live

In the `ktir` stage, after `LowerComputeOps` has turned the share into linalg/tensor
values. That is the slot the removed `LowerInterTile` pass held (§11); nothing else in the
stage lowers inter-tile communication.

## 4. Nothing writes remotely

Data crosses a core boundary only by a read: a tile reads another tile's region of the
source. A store always lands in the executing tile's own region of the destination. The
destination is composed, but no tile writes any region of it but its own.

Push would need a core to write into another core's scratchpad. Cross-core **reads** are
what the interconnect is described as supporting; remote writes are unverified. If they
exist, pull versus push becomes a performance question rather than an expressiveness one.

**Ordering is the backend's.** Each scratchpad-to-scratchpad transfer is synchronized by the
backend: a sender waits until its receiver has read before its buffer may be reused, and a
transfer does not start before the data it sends has been written. Inside one kernel, the
passes up to KTIR do not reorder loads and stores on scratchpad-resident data, so the order
the kernel wrote is the order the backend sees.

## 5. One holder per element

Every element of a composed view lies in exactly one partition, and so has one holder. That
follows from the input format and from the op.

**What the input format can and cannot produce.** Each entry gets one slice index per
divided dimension, so its region is one box. For any two entries the boxes are therefore
either **identical** — every index equal — or **disjoint**, because differing slices do not
intersect. Equal entries are refused, so only the disjoint case remains. There is no third
case: a partially overlapping pair would need a partition to own something other than a
whole slice, or to own two of them, and a slice list can express neither. The dangerous
shape is partial overlap, where two holders share only some elements and could disagree
about those. A slice list **cannot describe it**, so the lowering cannot emit it, whatever
future schedules look like.

**Which holder a read reaches is the kernel's.** An element has one holder, so the block a
tile reads fixes the tiles it reads from — one, or several where the block spans
partitions. The backend chooses nothing among holders. A kernel that wants its readers
spread over several copies of the same data reads different coordinates — and to be read
from, copies must be composed as distinct regions, or all but one of them listed as `None`.

**Copies are distinct regions.** Where several tiles receive the same data, each copy is
its own region of the composed destination (§3), not one region with several holders. In
the recorded model replication is everywhere on the destination side — 97 of 130 records,
up to 32 holders — and the recorded placements describe each such destination as one region
held by several cores. That is a statement about where the bytes are, which this design
keeps; how the copies are laid out in a composed view is a separate question the record
does not answer, and §3 leaves to the lowering. On the read side every element has exactly
one holder in all 130.

## 6. Why ownership stays a table

What the measured patterns rule out is deriving ownership from **axis counts** — an axis
name and a slice count, which is the form the removed `tl.inter_tile`'s `axis` parameter took. Ownership
is frequently strided rather than contiguous. In the measured records the cores feeding one
destination region sit two apart (cores 0 and 2, then 4 and 6, and so on), or eight apart
(0, 8, 16 and 24), or are drawn only from the even-numbered cores. No axis count reproduces
those, because the core-to-region assignment is not a function of how many pieces an axis
was cut into. The dense list reaches them by `None` at every tile that holds nothing.

Note that `axes` is not that form. It maps a tensor dimension to a key and says nothing
about counts or holders; the striding lives in the positions, where a list can carry it.

A closed form *does* reach them — with `mod` and `floordiv`, affine sets express strides of
that kind directly, and a survey finds a closed form for the consumer-to-source map in all
130 records. The table is kept for one reason: an irregular pattern, which no closed form
covers, still has somewhere to go. None of the 130 is irregular, so this is room for a case
not yet seen rather than a case in hand. Where a closed form exists it is no harder for the
compiler — an affine set is analyzed per tile into the same boxes — and whether to admit
one alongside the table is §8's second decision.

## 7. Cross-core reduction is a use of the same view

#20 tracks cross-core reduction. Here it is not a separate mechanism and needs nothing new:
compose the **inputs**, then write the reduction once over the composed axis. The fold is an
ordinary `tl.sum` or `tl.dot` over an ordinary axis, and the only inter-tile construct is the
descriptor. The grid below is `[P]`.

**A reduction — `sum` over a distributed axis.** `x` is split on `n`, one share per tile:

```python
x_share = x_desc.load([0, pid * BLOCK_N])               # [BLOCK_M, BLOCK_N]
tl.spyre_pin(x_share, "ct_local", offset=X_OFF)

SRC   = [{"n": i} for i in range(P)]
whole = tl.make_distributed_descriptor(
    x_share, work_slices=SRC,
    axes=[None, "n"],                            # composed: [BLOCK_M, P * BLOCK_N]
    block_shape=[BLOCK_M, P * BLOCK_N],
)
total = tl.sum(whole.load([0, 0]), axis=1)       # [BLOCK_M] -- all-reduce
```

`n` is a real dimension of the composed tensor, so reducing axis 1 reduces global `n` — the
local extent and the P partitions together, in one operation. Every tile runs it, which is
what makes it an all-reduce.

**A contraction — `matmul` with split K.** K appears in both operands at different positions,
so each operand gets its own descriptor:

```python
a_share = a_desc.load([0, pid * BLOCK_K])        # [BLOCK_M, BLOCK_K] -- my K-slice of A
b_share = b_desc.load([pid * BLOCK_K, 0])        # [BLOCK_K, BLOCK_N] -- my K-slice of B
tl.spyre_pin(a_share, "ct_local", offset=A_OFF)
tl.spyre_pin(b_share, "ct_local", offset=B_OFF)

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
coordinates: the compose would need P overlapping partitions, which a view does not admit
(§1). Composing inputs never reaches that state, because the shares are disjoint on the
divided dimension.

**The cost is a reassociation the backend has to find.** As written, every tile addresses
the whole composed input — all of `x` in the first example, all of A and B in the second. The
efficient schedule, reducing or multiplying the local slice and then combining P results, is a
reassociation that provenance permits (below). It is a heavier transform for `tl.dot` than for
`tl.sum`, since one dot must become P dots and a sum, and that is the main risk this form
carries.

`reduce_scatter` indexes before it folds, so tile `j` reads only its own slice of the
composed domain and reduces that.

**Which axis is reduced decides whether this is a collective.** A fold is cross-core exactly
when a reduced axis is one that `axes` named. Reducing an axis `axes` did not name is a gather
followed by an ordinary local reduction, not a collective at all.

### The destination set

Who ends up holding the result is not in the fold. `all_reduce` and `reduce_to_one` have
*identical* reduction expressions and differ only in how many tiles want the answer. For a
copy, the `if` around the landing states it, and the lowering composes the destination from
it (§3). A reduction has no landing — the result may feed the next computation directly —
so the `if` goes around the fold:

```python
total = tl.sum(grp.load([0, 0, 0]), axis=0)          # all: every tile holds it

if tl.program_id(0) == 0:                            # one
    total = tl.sum(grp.load([0, 0, 0]), axis=0)

if tl.program_id(0) < 4:                             # a subset -- no mode spells this
    total = tl.sum(grp.load([0, 0, 0]), axis=0)
```

The guard reaches KTIR as an `scf.if` on `ktdp.get_compute_tile_id`, enclosing the access
tile, the load and the fold, so non-holders issue no transfer. Two things make an enclosing
construct the right carrier rather than an attribute. The saving is in the *transfer*, so
whatever carries it has to enclose the load. And `construct_access_tile` — the one op in that
chain that could plausibly hold an attribute — is general addressing machinery usable for any
tile: its subject is where data *is*, and who *executes* is a different question. Whether a
copy's composed destination still needs its guard in KTIR is §8's third decision.

One dependency this creates, worth stating rather than discovering: the pipeline is
branch-averse today. `LowerScalarLoad` documents that no runtime branch is ever emitted for a
mask, and `DistributeWork` handles pids nested in an `scf.if` only defensively. A conditional
region around a load and a fold has to be acceptable downstream.

The direction of the default matters. Writing no guard means every tile holds the result,
which is definite and often what was meant — but it is also the expensive reading, reached by
writing nothing: a ring all-reduce takes 2(P−1) steps where reduce-to-one takes P−1, and the
fold runs on every tile rather than along one path. That asymmetry is why the guard is
part of the design and not an optimization hint.

**A fold whose result is wanted in one place is sunk into the guard.** Writing no guard
means all-reduce, which is definite and frequently intended, so refusing it would reject
valid kernels. The suspect form is narrower: a fold whose result is live *only* under a
guard — all-reduce semantics with P−1 dead copies, and almost certainly meant as
reduce-to-one. The lowering sinks the fold and its load into the guard wherever the value
has that single conditional use, and diagnoses where sinking is blocked, because a missed
sink is silent, correct, and pays all-reduce cost. Sinking is legal when the guard's
condition is a function of `tl.program_id` and constants, so the surviving tiles are known,
and nothing writes the source between the load and the guard.

### What provenance carries

That the reduced value came from a distributed view
is one def-use step away, and the composed partitions carry distinct `ct_id`s in their
memref types — so the fact is in the types, not in a name. It grants the *freedom* to
reorder (below) and supplies the *recognition* that this is a cross-core fold, from
the same observation. Algorithm selection therefore stays the backend's: ring, tree,
recursive halving or direct transfers, chosen by P and message size. A named mode would
pre-commit to one lowering; this does not.

**The reduction order is undefined, and a kernel cannot specify it.** That is a property of
the *value* rather than a restriction on the *loop*: a tensor read from a distributed view
carries no order guarantee, so a backend may re-associate whatever the kernel wrote. The
alternative was to reject a written-out accumulation, on the grounds that
`for k: acc += load(dview[k])` fixes a left-to-right fold that fp16's non-associativity then
makes illegal to re-associate. That reading does not survive the pipeline: a Triton `for`
becomes `scf.for` in the `tt` layer, and #62 proposes converting every `scf.for` into a
`generic`, at which point the ordering is gone regardless of what the kernel wrote. Better to
say the order was never promised than to police a form whose ordering the next pass
discards.

**The order is undefined even where the reducer is not commutative.** Order-freedom is
associativity *plus* commutativity, and while `add`, `max` and `mul` have both, a custom
reducer region need not, and **argmax** is associative while its result depends on
traversal order on ties. Such a fold is still accepted, and its result is whatever the
order the backend chose gives: an argmax over a distributed view may return any of the tied
indices. A kernel that needs one particular answer has to make the reducer order-free
itself — break argmax's ties by index, for instance — rather than rely on an order.

Nor does the composed value ask for P blocks of scratchpad. It is a tensor in SSA form, not
an allocation, so a load feeding a reduction is ordinary producer-consumer dataflow, and
whether it is ever materialized is a bufferization outcome.

Note that reductions are absent from the recorded relayouts only because those record
scratchpad relayouts specifically. They exist in the workload: an OpSpec can carry a matmul
or reduction that splits the reduction dimension.

## 8. Open decisions

Grouped by where each would be settled: in the Triton surface and its lowering, or in
KTIR itself, which lands upstream in `ktir-mlir-frontend`.

### In Triton and its lowering

1. **Scratchpad offsets that differ between cores.** Assumption 6 has one offset per side,
   the same on every core, and the pin states one. The recorded model's placements do not
   always fit: on 49 of the 260 sides of its 130 relayouts the scratchpad offset differs
   between cores — one source puts core 0 at 512 and the other 31 cores at 768. A
   `construct_memory_view` takes its own offset, so KTIR can say this; what the Triton
   surface would need, and whether the planner can always choose a common offset instead,
   is open.

### In KTIR

2. **Whether the N partition operands can be replaced.** Enumerating every share is the
   cost that stands out. Two candidates. An **ownership relation** — an affine set relating
   coordinates to holding tiles — reaches the measured patterns, keeps holder and shape in
   the type, and stays statically checkable, at the cost of new attribute machinery. A
   **dynamic `ct_id`** is a far smaller change, but `ct_id` is part of the memref type, so
   partitions of different cores would share one type and the operand list would stop
   carrying ownership. Neither changes the Triton surface.
3. **Whether a copy's composed destination needs its guard in KTIR.** The destination
   views state their holders by `ct_id`, so a tile that holds no view — tiles 1 and 3 in
   §3's worked example — is known statically, and the compiler could emit no store for it.
   Under SPMD, though, an unguarded `ktdp.store` runs on every tile, and on such a tile it
   lands in another tile's view: a remote write. Dropping the guard therefore needs a
   dialect rule: a store through a distributed view is performed only by the holder of the
   view it lands in. That removes the `scf.if` the downstream pipeline would otherwise have
   to accept, at the cost of turning a wrongly computed region into a silent no-op instead
   of a rejected remote write.

## 9. Designs rejected

**Exposing the compose itself in Triton.** `construct_distributed_memory_view` takes memref
*values*, so surfacing *it* would put all N partitions in one function as SSA values — a
global vantage forced by the operand list, which collides with SPMD. A descriptor avoids
that by taking one value, this instance's share, plus a list, and leaving the enumeration
to the lowering.

**Emulating scratchpad-to-scratchpad through global memory.** Functionally it works, and the
descriptor-layout machinery would carry the form change. But if every intermediate goes
through global memory there is nothing left for a relayout to do — that round trip is
exactly what this exists to remove. It would pass expression and numerical checks while
failing on emitted accesses, with every number green.

**Deriving ownership from an axis and a count.** §6.

**A mode enum per communication pattern.** §11.

**A distributed descriptor on the destination too.** The appendix.

## 10. Rejected inputs

The lowering should refuse, rather than guess:

- **A `work_slices` whose length is not `prod(grid)`.** The position is the holder, so the
  list has one entry per tile.
- **Non-`None` entries with differing key sets**, or **two equal entries** — the second
  would make two partitions overlap.
- **A gap in a key's indices.** `[{"mb": 0}, None, {"mb": 2}, None, ...]` counts three
  slices and leaves `mb = 1` with no holder.
- **An `axes` entry naming a key no entry carries**, or a key that `axes` never places. The
  two have to agree, and neither determines the other.
- **An access coordinate, or an `if` condition that selects tiles, that is not static per
  tile.** Both are evaluated per tile to pair senders with receivers and to compose the
  destination.
- **A block outside its composed view.**
- **A copy's landing that is not pinned**, or pinned at different offsets on different
  receivers.
- **A destination whose boxes have differing numbers of copies.** No placement tiles a box
  (§3).
- **`tl.wk_slice_coord` reached by a tile whose entry is `None`.** A tile that holds no
  region has no coordinate on any axis. It is checked per tile once `program_id` is folded;
  an `if` that keeps such tiles away from the call makes the kernel valid.
- **A `ct_id` on a `global` memory space.** `ct_id` is meaningful only for `ct_local`.

A reducer that is not commutative is **not** rejected: the order of a fold over a
distributed view is undefined whatever the reducer (§7).

## 11. `tl.inter_tile`, the removed alternative

`tl.inter_tile(x, axis, combiner, mode, work_slices=...)` was the earlier surface: one
named collective call, a `mode` enum, one work-slice table, and a tensor result. It
authored a `tts.inter_tile_reduce` op, which a `LowerInterTile` pass expanded into a
`ktdp.inter_tile_produce` plus a delivery op. The builtin, the op and the pass have been
removed; the `ktdp` ops are KTIR's and remain, and `tl.wk_slice_coord`, which reads the
same work-slice table, stays.

**What it lowered.** `all_reduce` and `reduce_to_one`. `reduce_scatter` and `broadcast`
were accepted by the Python surface and rejected by the pass, as were custom combiner
regions, non-contiguous groups and non-uniform `pick0` layouts for `reduce_to_one`.

**Its advantage.** It named the collective, so nothing had to be recognized: `mode` mapped
directly onto the produce/delivery pair, and the K-split matmul reduce ring was validated
through it at `SENCORES` 4 and 8.

**Why it is not the design.** Naming buys less than it appears to, because provenance
already identifies the fold (§7) and does so without committing to one lowering — a mode
picks the delivery pair, where provenance leaves ring, tree and direct transfer all
available. And naming does not scale to the copy family: gather, scatter, all-to-all,
relocation and broadcast would each need a mode, where a descriptor needs none. Two of the
four modes it declared did not lower.

**Why it was removed before the descriptor path.** Keeping it meant keeping a second
lowering for inter-tile communication in the slot the descriptor path needs, and a second
op in the `tts` dialect that the descriptor path does not build on. The K-split ring it
validated has no Triton-level surface until the descriptor path lowers a reduction (§7).

## Appendix: a distributed descriptor on the destination too

#243 proposed stating the destination the way the source is stated: a second
`make_distributed_descriptor`, built from a scratchpad descriptor rather than a pinned
value, with its own `work_slices` naming who holds each destination region, and a
`.store()` into it. §3's gather would read

```python
DST = [{"mb": 0}, None, {"mb": 1}, None]
dst_whole = tl.make_distributed_descriptor(dst_desc, work_slices=DST, axes=["mb", None],
                                           block_shape=[4, 64])  # composed: [8, 64]
if pid % 2 == 0:
    dst_whole.store([(pid // 2) * 4, 0], whole.load([0, 0]))
```

It expresses the same transfers: given a pinned landing and a static `if`, the design
above composes the same destination. Its one objective difference is in what is checked.
There, a forgotten or too-wide `if` is a compile error, because a tile whose `DST` entry is
`None` would store; here the same mistake is legal and silent — every tile loads and lands,
the lowering identifies all of them as receivers, the numbers stay right, and the extra
remote reads and scratchpad stay, since a pinned store cannot be proven dead. Getting the
intent itself wrong, `DST` and the `if` wrong the same way, passes in both.

It was not adopted because that waste arises only from a mistake in the kernel: a kernel
with the right `if` gets the same code under both designs. What it adds is a check against
an author's error, not a capability, and it would replace the pinned share that
`make_distributed_descriptor` takes and that [lx-placement.md](lx-placement.md) builds on.
It is worth revisiting if silent waste from a forgotten `if` turns out to matter in
practice.
