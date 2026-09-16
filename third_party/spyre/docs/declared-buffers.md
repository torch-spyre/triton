# Declared buffers: giving an intermediate a place to live

## Motivation

The Spyre dataflow scheduler takes one compute per schedule, so a chain of computes must be split
into groups shaped `load(s) → one compute → store`, with every intermediate round-tripped through
memory. Two passes do that today. The **splitting pass** chooses where the boundaries fall and routes
each intermediate through memory, giving every resulting group its own memory view, access tiles and
index arithmetic. The **layout pass** assigns a physical, stick-tiled type to every value on a
layout-annotated chain, along with the indexing maps and iterator types that go with it. The layout
pass runs first, and that ordering is the problem: an intermediate has no descriptor when it runs,
therefore no layout annotation, therefore the pass has to invent one by copying a neighbouring
operand's. This document proposes that the author instead *declare* where each intermediate lands, in
the vocabulary the frontend already has for descriptors, so that a boundary is an edge in the IR
rather than the output of an analysis and there is no layout left to invent.

Both are named by role throughout, since it is the job that matters and not the name. The layout pass
is `RewriteDescriptorLayout` in the tree today. The splitting pass runs late, in the spyrecode stage,
and is being added separately; nothing below depends on what it ends up being called.

## Surface syntax

The construct is an annotation applied to a value, not a wrapper around a subgraph: it declares
a buffer for that value, and that declaration is what ends one schedule and begins the next. It
is deliberately near-identical to `tl.spyre_tensor_layout`, so this section starts with the one
thing the two do not share.

### Two ops, and what separates them

The two ops differ in one thing, and everything else about them follows from it:

> A descriptor **names memory that exists**. A declaration **brings memory into existence.**

`tl.spyre_tensor_layout` describes a buffer the caller allocated and passed in, so it can be
cross-checked against that buffer — its verifier takes the logical rank from the descriptor's block
type, and the layout pass reads sizes and strides off the memory view that already exists.
`tl.spyre_buffer` defines a buffer that does not exist, so there is nothing to check against: the
layout together with the value's own type *is* the definition.

Every difference deduces from that. There is no pointer, so nothing supplies the base address or
the element type, and both must be stated or allocated. Nothing exists yet, so the value has to be
committed to memory, which is what ends a schedule. And where the buffer is `global`, the launcher
has to be told about it, so an argument is appended.

|  | `tl.spyre_tensor_layout(desc, …)` | `tl.spyre_buffer(value, …)` |
|---|---|---|
| attaches to | a tensor descriptor | a value |
| the buffer | exists; the caller allocated it | does not exist; the compiler must |
| base address from | a `!tt.ptr` already in the signature | stated, or allocated |
| element type from | the descriptor's own type | stated, defaulting to the value's |
| strides from | the descriptor's stride operands | derived row-major |
| ends a schedule | no | yes |

**The argument list is a consequence, not the distinction.** A declared `global` buffer causes one
argument to be appended to the entry function per buffer, which the launcher fills — the mechanism
the splitting pass already uses, reported through its spill-buffer metadata. So a softmax whose author
wrote two pointers compiles to an entry function taking seven. But a `ct_local` buffer appends no
argument and allocates no tensor, so this is true of one memory space and not the other, which is
why it cannot carry the distinction on its own.

### The physical description, spelled once

Three facts describe how a tensor sits on the device, and all three mean the same thing on either
op, so they are spelled once and passed to both:

```python
PHYS = tl.spyre_placement([0, (1, "splat", 64)],     # coordinate entries, one per physical dim
                          memory_space="global",      # or "ct_local"
                          arrangement="standard")     # or "staggered", "EXX2"

tl.spyre_tensor_layout(x_desc, PHYS)                  # the caller's buffer
tl.spyre_buffer(row_max, PHYS)                        # the compiler's buffer
```

Anything a pointer would otherwise have supplied is **not** in there, because a descriptor has no
use for it — those are keyword arguments on `tl.spyre_buffer`, where they are always meaningful:

```python
tl.spyre_buffer(pair, PHYS, dtype=tl.fp16_fused, also_readable_as=[tl.float16])
tl.spyre_buffer(tmp, PHYS_LOCAL, address=0x100)
```

This is deliberately the only spelling: a bare entry list is not accepted where a placement is
expected. An earlier draft took either, and the result was two ways to say one thing with no way for
a reader to tell whether a bare list was the same kind of object as a placement. Existing kernels
and lit tests that pass a list to `tl.spyre_tensor_layout` are migrated.

The entries are the ones `tl.spyre_tensor_layout` accepts today, plus one:

| entry | meaning |
|---|---|
| `src` (bare int) | `phys_idx = logical_idx[src]` |
| `(src, "floordiv", d)` | `phys_idx = logical_idx[src] // d` |
| `(src, "mod", m)` | `phys_idx = logical_idx[src] % m` |
| `(src, "splat", w)` | logical dim `src` (extent 1) replicated across `w` physical elements |

**The `splat` entry does not exist yet.** The first three are accepted today; `splat` is the one this
construct adds, and it is what makes a rank-reducing statistic come back out across a whole stick.
See Prerequisites for what has to be true before it means anything.

Two things it is worth not confusing. Every HBM tensor is *stick-tiled*, and that is the
`floordiv`/`mod` pair, not this entry — `splat` is about a logical dim of extent 1 occupying a
stick's width, which is a different question from how a tensor is tiled. And `splat` is unrelated to
`tl.broadcast_to`, to `tt.broadcast`, and to `tl.inter_tile`'s broadcast mode, all of which operate
on values rather than on layouts; avoiding that collision is why it is not called `broadcast`.

### Why this is not one op

The two ops carry the same placement and are the same shape — a result-less marker, erased by the
lowering — so merging them is the obvious question. The answer is no, for a reason that is a property
of an existing type rather than a matter of taste: `!tt.tensordesc` is parameterised by its block
shape and its element type, so a descriptor's access extent is *in its type*. A declared buffer needs
two extents on one base address — the full stick to store a statistic, one index to read it — and
layernorm's half-view of a fused pair is a third. Routing a declaration through a descriptor would
therefore have to commit in the type to one of those and treat the others as exceptions, which means
an author writing the extent this design derives.

A second reason, smaller but practical: types are invisible at the Python call site. One op name
would make the line that brings memory into existence and ends a schedule textually identical to the
line that annotates a buffer the caller passed in.

What *should* be shared is the vocabulary rather than the op, and that is what the placement is — in
the dialect too, as one attribute definition and one rank-parameterised verifier helper carried by
both ops, rather than each declaring three parallel arrays and re-checking the same rules. That is
also what makes `memory_space` and `arrangement` a one-place addition instead of the same field added
twice.

### The name

Three candidates were considered, all replacing the original proposal's `spyre_operator`.

**`tl.spyre_buffer` — chosen.** It names what the op brings into existence, which is the thing that
separates it from `spyre_tensor_layout`. It reads as a verb at the call site and does not overstate:
the op does not compute, schedule or move anything.

**`tl.spyre_checkpoint` — rejected, narrowly.** Names the effect — the value is committed to
memory and a schedule ends there — which reads well and is arguably the more consequential half.
Rejected because it says nothing about bringing memory into existence, which is the distinction.

**`tl.spyre_internal_layout` — rejected.** Maximally parallel to `spyre_tensor_layout`, which
would make the shared vocabulary obvious. Rejected because it presents itself as a pure
annotation and hides both effects — the allocation and the boundary.

### Naming an intrinsic, and why that is a separate gap

There is no frontend surface for a spyreop today, and the reason is structural: at frontend time
only TTIR exists, and the KTDP dialect appears later in the pipeline. Spyreops are reached
*indirectly* — the author writes ordinary Triton and `LowerSpyreOps` converts in the ktir stage,
so `tl.sqrt` becomes `spyreop.sqrt`, `/` becomes `spyreop.realdiv`, and integer add and multiply
become their `spyreop` equivalents. The `spyreop` fixture is written exactly that way.

That covers every intrinsic with an ordinary-Triton spelling and none of the others.
`exx2_fused` and the `layernorm*_fused` family have no such spelling, so conversion cannot reach
them, and both worked examples below need them. So a frontend surface is required, and it must
produce a **TTIR** op: minimally one name-carrying `tt.spyre_op` that a ktir pass turns into the
real `spyreop.*`.

The intrinsic is named by **the dialect binding, not by a string**, so arity and operand types come
from the op definition rather than from a table we maintain:

```python
from mlir_ktdp.dialects import spyreop

pair = tl.spyre_op(spyreop.exx2_fused, x, axis=1)
```

The generated builders are used here as *symbols*, not called: they build KTDP ops and would need an
MLIR context, which does not exist at trace time. The frontend reads the op's name and its declared
arity off the binding and puts them on `tt.spyre_op`. An unknown or wrong-arity intrinsic then fails
at the frontend rather than in a later pass, and nothing has to be kept in step with the dialect by
hand.

Two consequences worth stating. Tracing a kernel gains a hard dependency on `mlir_ktdp`, which is
installed by a separate step, so a kernel using an intrinsic will not trace without it. And this
construct assumes the surface exists when it does not yet; whether it belongs in this proposal is in
Open decisions.

### The op is a marker with no result

The op annotates a value rather than producing one, which is what keeps it usable on a compute with
more than one result. `tl.spyre_buffer` returns nothing and the author does not rebind, exactly as
`tl.spyre_tensor_layout` does not — that op is declared with no result and no traits, and is a
marker consumed and erased by the lowering. Multiple outputs then need no new spelling: one
declaration per result.

```python
vals, idxs = tl.spyre_op(spyreop.topk, x, k=4)
tl.spyre_buffer(vals, VAL_PHYS)
tl.spyre_buffer(idxs, IDX_PHYS)
```

## Semantics

`tl.spyre_buffer` says: this value is written to a buffer of its own with the given layout, and
read back by each consumer through an access tile derived from that layout. It makes four things
determinate that are guessed today — where the schedule splits, what the intermediate's physical
type is, what indexing map every operand of the compute gets, and what element arrangement the
buffer holds. **No affine map is ever author-written or synthesised; every map is projected out of
declared coordinate entries.**

### Prerequisites

Two capabilities have to exist before any of this means anything, and neither does yet. They are
stated as properties rather than as work items, since what matters is the property.

1. **Every compute is lowered as a `linalg.generic`.** Today `LowerComputeOps` emits named `linalg`
   ops for some computes, and a named op's identity carries layout information that its indexing
   maps then do not have to. Once every compute is a generic, an op's identity says nothing about
   layout and its maps say everything — which is the precondition for maps being derivable from
   declared entries at all.
2. **The layout pass decides physical types from indexing maps rather than per-op rules,** and admits
   a `splat` coordinate entry. Today it carries one rule per op kind, keyed on what the op *is*,
   which stops working the moment every op is a generic.

This construct is a frontend for the second capability's input. It should not be implemented ahead
of it, and the derivation below assumes both.

### The group graph

A declared buffer ends one group and begins the next. A group is therefore the maximal region of
dataflow between declared buffers (or between a descriptor load and one, or one and a descriptor
store), and one-compute-per-schedule becomes the statement that each such region holds exactly
one compute — checkable by a verifier rather than discovered late.

A declared buffer is a buffer, not a value copy: a value consumed by two later groups is one
buffer read twice, which is what softmax's `exp` needs. Nothing is shared between groups. Each
reconstructs its own memory view, access tiles, `tensor.empty` and index arithmetic, because a
stage's schedule is extracted into its own module and anything reachable from two stages is a use
that extraction cannot resolve. Replication is the splitting pass's job, not the author's.

### Element arrangement

A value has an arrangement as well as a layout — how its coordinates sit within a stick — and a
declared buffer is where that would otherwise be lost. Three matter here, drawn from
`ElementArrangement`, which every `SpyreTensorLayout` already carries:

| arrangement | what it means |
|---|---|
| `standard` | sequential element order — the default |
| `staggered` | values correct, within-stick position not matching logical order |
| `EXX2` | a reduction putting **two values in the stick** rather than one |

The last is a different kind of fact from the middle one, so V10 is two rules. `staggered` **reorders**
a stick, and an op is safe on it only if it never consults within-stick position, so combining it with
a `standard` operand is not; the fix is an explicit rearrangement, which like any other mismatch is
its own compute group with its own buffer. `EXX2` instead changes **how many values a stick holds**,
so its rule is about reads: the pair, or one of its two values, and nothing past the second.

Of the two targets, softmax is `standard` throughout and layernorm is not — its `pair` buffer is an
`exx2` reduction, and that is what lets the buffer be read either as the pair or as one half. Neither
target reorders, since a dtype cast is what produces `staggered` and neither has one.

`staggered` is the
[FP32 element-arrangement RFC](https://github.com/torch-spyre/RFCs/blob/main/2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md)'s
term for the two conversion-produced orderings. The enum holds more than these three; a frontend
spelling of it should not renumber, since the integer encoding is stable by contract.

### Reading a buffer as another type

A buffer may declare additional types it can be read as, and a consumer selects by its own
operand type:

```python
tl.spyre_buffer(pair, PAIR, dtype=tl.fp16_fused,
                also_readable_as=[tl.float16])          # or [(tl.float16, 1)] for the 2nd half
```

A consumer whose operand slot is `fp16_fused` gets the pair view; one whose slot is `f16` gets
the `f16` view — the same base address, the same physical shape and strides, a different element
type. An entry may name a component index, so a specific half of a fused pair is addressable,
and several entries may share a type at different components. A consumer asking for a type not
on the list is an error rather than a silently emitted view.

This is a reinterpretation, not a conversion: it costs a second memory view and no compute.

## The declaration grammar

The grammar is one production for the physical description both ops take and one for the declaration
that applies it, with the provenance-dependent fields as keywords on the declaration. Everything
about how a buffer is *read* is derived from how it was written, except the list of alternative
element types, which is stated because a consumer may want one the producer did not.

```
placement  ::= "tl.spyre_placement" "(" layout
                       [ "," "memory_space" "=" space       ]       // default "global"
                       [ "," "arrangement"  "=" arrangement ]       // default "standard"
                   ")"

layout     ::= "[" entry { "," entry } "]"              // one entry per physical dim
entry      ::= int                                      // identity
             | "(" int "," "floordiv" "," int ")"
             | "(" int "," "mod"      "," int ")"
             | "(" int "," "splat"    "," int ")"        // write across w, read one index

declare    ::= "tl.spyre_buffer" "(" value "," placement
                       [ "," "dtype"            "=" type    ]       // else the value's
                       [ "," "also_readable_as" "=" "[" alias { "," alias } "]" ]
                       [ "," "address"          "=" address ]       // else compiler-supplied
                   ")"

space       ::= "global" | "ct_local"
arrangement ::= "standard" | "staggered" | "EXX2"
alias       ::= type | "(" type "," int ")"             // type, and which component
address     ::= int                                     // one offset in the named space
              | "[" int { "," int } "]"                  // one per partition

type        ::= an ordinary Triton element type, such as tl.float16
              | a Spyre opaque element type, such as a fused pair
```

`tl.spyre_tensor_layout(desc, placement)` takes the same placement and nothing else.

Fields, and what each is for. The first three are the placement, shared with descriptors; the last
three are keywords on the declaration, because a descriptor takes them from its pointer or its type
and so has nothing to state:

- **`layout`** — one entry per physical dimension, identical in form and lowered representation to
  `tl.spyre_tensor_layout`'s. The logical shape is the annotated value's own shape; it is not
  restated.
- **`memory_space`** — lowers to the `#ktdp.memory_space<global>` / `<ct_local>` attribute that
  every `ktdp.construct_memory_view` carries. The inter-tile design already adds this field to
  `tl.spyre_tensor_layout`, which is why it belongs in the shared placement rather than here.
- **`arrangement`** — `standard`, `staggered` or `EXX2`, defaulting to `standard`; see Element
  arrangement.
- **`dtype`** — the buffer's element type, defaulting to the value's. This is the primary source of
  that type rather than an override, because there is no pointer to take it from; it is how a fused
  pair type is named, as in layernorm's `!spyreop.fp16_fused`.
- **`also_readable_as`** — the other element types this buffer may be read as, each optionally
  naming a component index.
- **`address`** — an offset within the memory space that `memory_space` names, or one offset per
  partition; see below. Omitting it leaves the compiler to supply one.

The cases the two verified targets and the scratchpad future require:

| case | spelling |
|---|---|
| plain tile buffer | `tl.spyre_buffer(v, TILE)` |
| statistic, written across the stick, read one index | `tl.spyre_buffer(v, STAT)` where `STAT`'s entries end `(1, "splat", 64)` |
| fused pair, also read as one half | `tl.spyre_buffer(v, PAIR, dtype=tl.fp16_fused, also_readable_as=[tl.float16])` where `PAIR` is `arrangement="EXX2"` |
| scratchpad buffer at a known offset | `tl.spyre_buffer(v, STAT_LOCAL, address=0x100)` |
| one offset per partition | `tl.spyre_buffer(v, SHARE_LOCAL, address=[0x0, 0x100, ...])` |

### `memory_space` says which memory; `address` says where in it

These are two different questions, and keeping them apart is what makes both fields simple. This
section also says how a list of addresses reaches KTIR, since a list is the one form whose lowering
is not a straight attribute copy.

Nothing about `address` is per core. A scratchpad is a flat per-core space, and two buffers in it sit
at two offsets: the two scratchpad allocations observed in a pointwise probe carry `0` and `256`,
uniform across cores. That is two buffers at two offsets, and it is what a scalar `address` states.
The per-core variation in the same probe is on the global side, where the values are symbolic handles
the launcher patches — not something an author states.

A **list** of addresses is for the distributed case: one offset per *partition*, parallel to the
partition table that `tl.make_distributed_descriptor` composes over. Indexed by partition, not by
core — a list as long as there are regions, which for a source held by 8 of 32 cores is 8 and for a
broadcast source is 1. That distinction matters: a grid-indexed list would be the global vantage the
inter-tile design rejects, whereas a partition table is surface that design already accepts.

The lowering needs nothing new. `ktdp.construct_memory_view` takes its base as a single `index`
offset operand, so the lowering emits **one view per address** rather than looking for one view that
can carry several. That composes with the group graph's no-sharing rule, where each group already
reconstructs its own memory view, so several views over one declared buffer is the established idiom
rather than a new burden.

Two costs of the list form, both worth stating before it is relied on. It amends the inter-tile
design, whose phase 1 asserts that partition views differ *only* in their coordinate set and holder,
with offsets and strides identical across them — a partition-varying address makes offsets differ
too. And nothing measured needs it: the scratchpad evidence is two buffers with two scalars, not one
distributed buffer. So the list form is admitted in the grammar and checked by the verifier, and the
lowering should report it as unsupported rather than silently emitting per-partition views.

### Composing a declared buffer across cores

A scratchpad relayout needs a tensor that is compiler-allocated *and* read at author-chosen offsets,
which neither this construct nor `tl.make_distributed_descriptor` provides alone. They compose by
ordering: the declaration says where each core's piece lives, the descriptor says how the pieces
compose and which part this instance reads.

```python
share = ...                                                    # my per-core piece
tl.spyre_buffer(share, SHARE_LOCAL)                            # where it lives
whole = tl.make_distributed_descriptor(share, work_slices=SRC, axes=[None, "n"])
mine  = whole.load([0, my_offset])                             # the relayout
```

That ordering also fills a gap on the other side. The distributed design's phase 1 emits one memory
view per partition with a holder and a coordinate set, and its scratchpad offsets arrive as metadata
— but nothing in it says who reserved the scratchpad, and its first assumption explicitly declines to
answer, constraining the signature rather than the body. The declaration is what answers it, so the
base address reaching phase 1 is stated rather than presupposed.

**The two features cannot collide.** A `splat` entry requires `global` (V12), and a relayout is
scratchpad on both sides by definition, so a declared buffer that is composed across cores never
carries a `splat` entry. Its layout is ordinary stick tiling. That matters because it means the
two-access-tile asymmetry — the reason an author-written access extent is unacceptable elsewhere in
this design — does not arise in the distributed case at all: a composed descriptor's `block_shape`
has no derived extent to contradict.

So `block_shape` on `tl.make_distributed_descriptor` is safe as it stands. Replacing it with a named
granularity — this instance's share, the composed whole, or one slice along a partition key, each
derivable from the share's shape and the slice counts the partition table already implies — would be
an ergonomic improvement to that design rather than a correctness requirement of this one.

## Worked example 1 — layernorm, three groups, a fused pair

This traces the construct against `@LayerNorm_1`, a target with `grid = [2]` and six base
addresses (`x, pair, sc, w, b, out`) that compiles and runs. It is the harder example because one
buffer is written as a fused pair and read back both as the pair and as one half of it, which is
what `also_readable_as` exists for.

```python
M, N = 48, 2048
S    = 64
X_PHYS = tl.spyre_placement([(1, "floordiv", S), 0, (1, "mod", S)])   # [48, 2048] -> [32, 48, 64]
STAT   = tl.spyre_placement([0, (1, "splat", S)])                     # [48, 1]    -> [48, 64]

x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
tl.spyre_tensor_layout(x_desc, X_PHYS)
x = x_desc.load([0, 0])

# group 1 — exx2, reducing N away, leaving a stick per row with two values at its head
PAIR = tl.spyre_placement([0, (1, "splat", S)], arrangement="EXX2")
pair = tl.spyre_op(spyreop.exx2_fused, x, axis=1)
tl.spyre_buffer(pair, PAIR, dtype=tl.fp16_fused, also_readable_as=[tl.float16])

# group 2 — layernormscale: reads the pair at one index, writes stick-wide
sc = tl.spyre_op(spyreop.layernormscale_fused, pair)
tl.spyre_buffer(sc, STAT)

# group 3 — layernormnorm; its sq slot is f16, so it gets the f16 view of the same buffer
out_desc.store([0, 0], tl.spyre_op(spyreop.layernormnorm, x, pair, sc, w, b))
```

Two declarations, two buffers, three groups. Of the target's six base addresses, four are the
author's (`x, w, b, out`) and two are declared (`pair, sc`). Both are `global`, and the arrangements
differ: `sc` is an ordinary one-value-per-stick statistic, so `standard`, while `pair` is what an
`exx2` reduction produces and carries **two values per stick**, so `EXX2`. That arrangement is the
reason the buffer can be read both as the pair and as one of its halves.

**The `pair` buffer's memory view is derived from `PAIR`, not stated.** Physical dim 0 is the
logical row, extent 48; physical dim 1 is the stick, extent `S = 64`; row-major strides
follow. That is `sizes: [48, 64], strides: [64, 1]`, and with `dtype` the element type is
`!spyreop.fp16_fused` — the target's `memref<48x64x!spyreop.fp16_fused>`. The `f16` entry in
`also_readable_as` produces the target's second view over the same base as `memref<48x64xf16>`.

| group | compute | ins → map | outs → map | iterators | access tiles |
|---|---|---|---|---|---|
| 1 | `exx2_fused` | `x ⟨32×24×64⟩` → `#map_red_in` | `pair ⟨24×64⟩` fused → `#map_red_out` | `[reduction, parallel, reduction, parallel]` | store `⟨24×64⟩` |
| 2 | `layernormscale_fused` | `pair ⟨24×1⟩` fused → `#map_splat` | `sc ⟨24×64⟩` f16 → `#map2d` | `[parallel, parallel]` | load `⟨24×1⟩`, store `⟨24×64⟩` |
| 3 | `layernormnorm` | `x` → `#map`; `sq ⟨24×1⟩` f16 → `#map_stat`; `sc ⟨24×1⟩` → `#map_stat`; `w`, `b` | `out ⟨32×24×64⟩` → `#map` | `[parallel ×3]` | load `⟨24×1⟩` ×2, store `⟨32×24×64⟩` |

Group 2 matches the target in full — maps, iterators and both access tiles. Group 1's shapes and
element types match; its maps and iterator vector are not printed in the target IR, though they
are identical in form to softmax group 1, which is. Group 3's `sq` view and its `⟨24×1⟩` `f16`
read match; whether the target reads `sc` at one index or stick-wide is not shown, and the `w`
and `b` layouts are not given.

**Where the derivation does the work.** Group 2 is entirely mechanical: the participants are
`pair` (read) and `sc` (written), both `STAT`-shaped, so the iteration space is 2-D — row, then
the splat dim at its write extent of 64. `sc` is written across it, giving `#map2d`; `pair`
is read with its splat dim fixed, giving `#map_splat = (d0,d1) -> (d0,0)`. Neither map was
written by the author.

**The two access tiles on one buffer.** `pair` is stored `⟨24×64⟩` in group 1 and loaded `⟨24×1⟩`
in group 2, from one base address. That asymmetry is what a `splat` entry means — the dim is
written across its full extent and read at a single constant index, so one buffer carries two access
tiles — and it is why a descriptor, which carries a single `block_shape`, cannot stand in for a
declared buffer.

## Worked example 2 — softmax, six groups

This traces the construct against `@Softmax_1`, `grid = [2]`, seven base addresses
(`x, max, diff, exp, sum, recip, out`), logical `[48, 2048]` f16, which runs correctly on
hardware. It is the example showing the construct covers ordinary arithmetic rather than only
named intrinsics, and where the derived maps can be checked against six groups at once.

```python
M, N = 48, 2048
S    = 64
TILE = tl.spyre_placement([(1, "floordiv", S), 0, (1, "mod", S)])   # [48, 2048] -> [32, 48, 64]
STAT = tl.spyre_placement([0, (1, "splat", S)])                    # [48, 1]    -> [48, 64]

x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
tl.spyre_tensor_layout(x_desc, TILE)
out_desc = tl.make_tensor_descriptor(o_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
tl.spyre_tensor_layout(out_desc, TILE)

tile = x_desc.load([0, 0])

row_max = tl.max(tile, axis=1, keep_dims=True)       # group 1
tl.spyre_buffer(row_max, STAT)

diff = tile - row_max                                # group 2
tl.spyre_buffer(diff, TILE)

ex = tl.exp(diff)                                    # group 3
tl.spyre_buffer(ex, TILE)

den = tl.sum(ex, axis=1, keep_dims=True)             # group 4
tl.spyre_buffer(den, STAT)

rcp = tl.spyre_op(spyreop.reciprocal, den)                 # group 5
tl.spyre_buffer(rcp, STAT)

out_desc.store([0, 0], ex * rcp)                     # group 6
```

Compare with the fixture as it stands: the arithmetic is unchanged except that the divide is
split into a reciprocal and a multiply, and five declarations are added. Five declared buffers
plus the author's `x` and `out` give the target's seven base addresses.

That split is not forced by the construct. `ex / den` would lower to `spyreop.realdiv`, a single
compute, and so would be one group like any other. It is written as a reciprocal and a multiply
because that is what the verified target does — `@Softmax_1` has `spyreop.reciprocal` producing a
stick-wide result and a separate multiply after it. Why the target takes the two-compute form rather
than one `realdiv` is an open question; the example follows the measured IR rather than guessing.

| group | loads → map | compute | outs → map | iterators | store |
|---|---|---|---|---|---|
| 1 | `x ⟨32×24×64⟩` → `#map_red_in` | `arith.maximumf` | `⟨24×64⟩` → `#map_red_out` | `[reduction, parallel, reduction, parallel]` | `max ⟨24×64⟩` |
| 2 | `x` → `#map`; `max ⟨24×1⟩` → `#map_stat` | `arith.subf` | `⟨32×24×64⟩` → `#map` | `[parallel ×3]` | `diff ⟨32×24×64⟩` |
| 3 | `diff` → `#map` | `spyreop.exp` | `⟨32×24×64⟩` → `#map` | `[parallel ×3]` | `exp ⟨32×24×64⟩` |
| 4 | `exp` → `#map_red_in` | `arith.addf` | `⟨24×64⟩` → `#map_red_out` | `[reduction, parallel, reduction, parallel]` | `sum ⟨24×64⟩` |
| 5 | `sum ⟨24×1⟩` → `#map_splat` | `spyreop.reciprocal` | `⟨24×64⟩` → `#map2d` | `[parallel, parallel]` | `recip ⟨24×64⟩` |
| 6 | `exp` → `#map`; `recip ⟨24×1⟩` → `#map_stat` | `arith.mulf` | `⟨32×24×64⟩` → `#map` | `[parallel ×3]` | `out ⟨32×24×64⟩` |

Every map and iterator vector in this table is *derived* by the four-step rule in Semantics rather
than written by the author. Working through the three distinct shapes:

**Groups 1 and 4 — the reduces.** Participants are a `TILE`-shaped input and a `STAT`-shaped
output. The input contributes `d0` (stick, 32), `d1` (row, 24) and `d2` (lane, 64) at their own
positions. The output's splat dim is not in that space, so it is appended as `d3` at its
write extent of 64. The reduced logical axis is `N`, which `TILE` splits across two physical
dims — `d0` via `floordiv` and `d2` via `mod` — so both of those positions are `reduction` and
`d1`, `d3` are `parallel`. That yields `[reduction, parallel, reduction, parallel]`,
`#map_red_in = (d0,d1,d2,d3) -> (d0,d1,d2)` and `#map_red_out = (d0,d1,d2,d3) -> (d1,d3)`,
matching the target. The two reduction loops land at positions 0 and 2 because that is where the
dims they reduce sit.

**Groups 2, 3 and 6 — the elementwise ones.** A `TILE`-shaped value participates, so the
iteration space is its 3-D physical tile, all parallel. `TILE` operands and results project
through the identity. `STAT` operands — `max` in group 2, `recip` in group 6 — contribute their
row dim, which is the iteration space's `d1`, and fix their splat dim, giving
`#map_stat = (d0,d1,d2) -> (d1, 0)` and a `⟨24×1⟩` access tile.

**Group 5 — the reciprocal.** Only `STAT`-shaped values participate, so the iteration space is
2-D, and the derivation is the same as layernorm group 2: `#map_splat` in, `#map2d` out.

Groups 1, 4 and 5 all write `STAT`-shaped buffers, and all three are read back the same way, at one
fixed index — whether the compute filled every lane or only the head of the stick.

## The TTIR representation, and which pass reads what

The information enters as one TTIR marker op carrying attributes, and is read by two passes at
the two stages they already occupy — the property that made the annotation form preferable to a
region. Nothing new is created in the ktir or spyrecode pipelines; two existing passes stop
guessing and start reading.

**The layout pass, in the ktir stage**, is the primary consumer — the one that assigns a physical
type to every value on a layout-annotated chain, and the indexing maps and iterator types that go
with it. It reads the declared entries off every value that carries them instead of copying a
neighbouring operand's layout, and derives the maps from them.

**The splitting pass, in the spyrecode stage**, is the second. It keeps its late position and its
duty of giving each group its own memory view, access tiles, `tensor.empty` and index arithmetic, but
stops *choosing* where the boundaries fall, splitting at `tt.spyre_buffer` markers only
and materializing each buffer from the declaration's own attributes. That division is why the
annotation works where a region does not — the layout must be known before the layout pass runs,
while the per-group duplication must happen after the last CSE, since every duplicated op is `Pure`
and a CSE would merge them back. One declaration serves both, read twice.

### The op

The op carries the declaration and nothing else — no result, no region, no operands beyond the
annotated value:

```mlir
tt.spyre_buffer %v {
  phys_src = [0, 1], phys_op = [0, 3], phys_arg = [0, 64],   // the placement, as
  memory_space     = #ktdp.memory_space<global>,             //   spyre_tensor_layout carries it
  arrangement      = #tt.spyre_arrangement<standard>,
  element_type     = f16,                                    // declaration-only from here
  also_readable_as = [],                                     // (type, component) pairs
  address          = []                                      // scalar, or one per partition
} : tensor<48x1xf16>
```

The first three attributes are the placement and are carried identically by `tt.spyre_tensor_layout`;
the last three exist only here, because a descriptor takes them from its pointer or its type. Having
no result, the op is not `Pure` and nothing removes it, and two declarations on distinct values are
structurally distinct without needing an effect annotation.

The placement should be one attribute definition with one rank-parameterised verifier helper shared
by both ops, rather than three parallel arrays declared and re-checked twice. Collapsing the triple
into a single named attribute is the cleaner end state and changes `spyre_tensor_layout`'s printed
form, so it costs a rewrite across the lit tests that reference it — worth doing when a third op
carries a placement, not before.

## Verifier rules

These are the rules that make one-compute-per-schedule and the declared layouts checkable at
the frontend, with a diagnostic naming a source line. Op verifiers get the local rules; a
function-level pass gets the whole-graph ones, since an op verifier should not walk a function.

Local, in the op verifier:

- **V1 — entry closure.** Every entry is `identity`, `floordiv`, `mod` or `splat`, so a
  linearizing or composite expression is unrepresentable.
- **V2 — rank canonicalization.** No layout may yield a leading extent-1 physical dim; the
  frontend collapses it and the verifier rejects a survivor.
- **V3 — constant read index.** A read of a splat dim is a compile-time constant. A
  dim-dependent index is already unrepresentable by V1; this states the invariant.
- **V4 — no unresolved layout.** There is no "no layout" state: a value that crosses a memory
  boundary either carries a declaration or is an error. Where a genuine unknown is needed it is an
  explicit placeholder attribute plus a verifier that errors on survivors — never an absent
  attribute that some later pass is expected to repair.
- **V5 — alias consistency.** An `also_readable_as` entry may change the element type and name a
  component, but not the physical shape or strides, and its component index must lie inside the
  declared element type.
- **V12 — a `splat` entry requires `global`.** The convention it expresses is an HBM one, so a
  placement carrying a `splat` entry with `memory_space = "ct_local"` is an error. This holds on
  either op, since the rule is about the memory space rather than about what the placement is
  attached to.
- **V13 — a well-formed address.** A scalar address is non-negative. A list is non-empty with
  distinct non-negative entries, and requires `ct_local`, since a `global` buffer's addresses come
  from the launcher. Checking a list's *length* is not an op-verifier job — the partition count lives
  on `tl.make_distributed_descriptor` — so that check is V14 below.

Whole-graph, in a `-verify-compute-groups` pass:

- **V6 — one compute per group.** On every path between two boundaries — declaration to
  declaration, descriptor load to declaration, or declaration to descriptor store — there is
  exactly one compute. The diagnostic names the extra computes with their source locations.
- **V7 — every compute-to-compute edge is declared.** A compute whose result feeds another compute
  with no intervening declaration is an error naming both. This is the rule that rejects bare
  chained kernels.
- **V8 — reduction position.** For a group whose compute reduces logical axis `a`,
  `iterator_types` is `reduction` at exactly the positions of `a`'s physical dims and `parallel`
  elsewhere. Permutations of the parallel dims are unconstrained.
- **V9 — no sharing between groups.** No `ktdp.construct_memory_view`,
  `construct_access_tile`, `tensor.empty` or index computation is reachable from two groups.
- **V10 — arrangement compatibility**, two rules, because the arrangements are two kinds of fact. For
  a *reordering* — `staggered` — a group combining it with a `standard` operand, without broadcast, is
  an error, as is any op consulting within-stick position on it; the diagnostic should name the
  rearrangement that would fix it. For a *multiplicity* — `EXX2` — the rule is instead about how the
  stick is read: a consumer of an `EXX2` buffer takes either the pair or one of its two values, and
  anything that would read past the second is an error.
- **V14 — an address list matches its partition table.** Where a declared buffer is composed by
  `tl.make_distributed_descriptor`, an `address` list has one entry per partition and corresponds in
  order, so its length is checked against that op's partition table rather than against the launch
  grid. This is the frontend diagnostic with a source location; the lowering keeps its own backstop
  for hand-written IR.
- **V11 — declared read types.** Every consumer of a declared buffer takes either its `dtype` or
  one of its `also_readable_as` entries.

V9 states as a rule what the splitting pass currently gets from pass *position* — installed last, with
the only canonicalize/CSE in that stage on the other arm of a conditional. Today that invariant is
defended by a comment rather than a check.

## Rejection behaviour

Bare chained kernels are rejected, because boundary inference does not work. What decides whether
that is a disruptive change is how narrowly the rejection is scoped, and it is scoped twice over: by
the kernel opting in at all, and then by V7 within it.

The outer scope is the presence of `tl.spyre_tensor_layout` annotations. A kernel carrying at least
one has opted into the Spyre placement discipline, and is where the rules apply; a kernel with no
Spyre annotations anywhere is never rejected, so nothing that compiles today can start failing. That
is strictly safer than scoping on V7 alone, and it costs nothing, because a kernel that declares
layouts for its descriptors is exactly the kernel whose intermediates are worth declaring too.

### What is rejected

Inside that scope, the rule that fires is V7: a compute whose result feeds another compute with no
declaration between them. A kernel with a single compute has no such edge, so the `elementwise`,
`reduce`, `matmul`, `gather` and `spyreop` fixtures are unaffected and need no annotation whether
they carry layout annotations or not. The chained kernels — the four softmax variants — are what the
rule catches, and they are the ones that today either fail or depend on the inference path.

### The diagnostic

The diagnostic has to name the undeclared edges and say what to write in their place, or the author
learns only that the kernel is rejected.

```
error: 'softmax_one_tile_device' declares Spyre tensor layouts and has a chain of 5 computes
       on one dataflow path, but declares no compute-group boundaries. One compute is
       admitted per schedule, and boundary inference is not used for such a kernel.
       Declare each intermediate with tl.spyre_buffer to say where it lands.
 note: arith.maximumf feeds arith.subf with no declaration between them
 note: 3 further undeclared edges (arith.subf -> math.exp, math.exp -> arith.addf,
       arith.addf -> arith.divf)
```

Each note carries the source location of the ops it names, and every edge it names is fixable by
adding a declaration. The final `arith.divf` needs no rewriting to be admissible: it lowers to
`spyreop.realdiv`, one compute, so it becomes a group of its own like every other link in the chain.

## Open decisions

Four questions would change the surface; the rest would change only the implementation.

- **Does the soft rejection scope keep layout invention alive indefinitely?** Scoping rejection to
  kernels carrying `tl.spyre_tensor_layout` means an unannotated chained kernel is still admitted, so
  something must still place its intermediates and today's inference cannot be deleted on the
  strength of this construct alone. Keep it permanently, keep it and check its output against the
  same verifier, or widen the scope later and delete it.
- **How much of the `address` field is real?** Three unsettled at once: whether the per-partition
  list is worth admitting when nothing measured needs it and it amends the inter-tile design's claim
  that partition views differ only in coordinate set and holder; whether its entries are positional
  or keyed by partition coordinate, which would reuse that design's own matching and turn V14 into a
  key-set comparison; and whether the backend's `[core, corelet, …]` key needs granularity below the
  core, since every entry seen is `[n, 0, 0]`.
- **Is multiple output arity the direction or the exception?** The top-k discussion reaches for a
  fused opaque type carrying value *and* index — layernorm's fused pair again — while its iterative
  formulation has `2k` results on one `linalg.generic`. This surface takes both, one declaration per
  result; which is expected decides which path deserves the better support.
- **Is `tl.spyre_op` part of this construct or a change of its own?** Both worked examples need a way
  to name an intrinsic with no ordinary-Triton spelling and none exists, so it is a prerequisite
  rather than a component.

Smaller, and implementation-facing: whether `arrangement` should be inferred from the producing op
rather than stated; whether declarations are legal inside `scf.for`, which no target exercises;
whether V6's "one compute" counts `linalg` ops or every op with a tensor result; and whether a
declared buffer ever needs strides other than row-major.

One loose end that is a question about the hardware rather than about this design: **why does
`@Softmax_1` divide as reciprocal-then-multiply?** `arith.divf` lowers to `spyreop.realdiv`, one
compute, so `ex / den` would be a single group. The target has a reciprocal and a multiply anyway,
and the worked example follows it. If that form is not required, the example loses a group.

## What this does not solve

The construct fixes what the scheduler is handed, not what the scheduler does. Three of these look
like they ought to fall out of it and do not.

- **It does not schedule, and it does not allocate addresses.** The declared graph makes each
  buffer's live range computable — `exp` is written by group 3 and read by groups 4 and 6 — but no
  reuse is implemented, so a chain with more simultaneously-live buffers than there are addresses
  still fails.
- **It does not implement `ct_local` buffers.** The surface expresses them and the memory views are
  expressible, but the channel is not: the splitting pass appends one argument per buffer for the
  launcher to fill, which is what a `global` allocation wants and not what a scratchpad offset is.
- **It does not retire the layout pass**, which still physicalizes descriptors and synthesises
  contractions. What it stops doing is *inventing* a layout — but only inside the rejection scope,
  where every intermediate reaching memory carries a declaration. Outside it, invention stays
  reachable, which is the price of the soft rule.
- **It does not make bare chained kernels work.** Inference remains broken; inside the scope the
  failure becomes a diagnostic on a source line, and outside it nothing changes.
- **It does not implement rearrangement.** `arrangement` is declarable and V10 rejects an illegal
  combination, but the relayout op that would fix one is not proposed here.
- **It says nothing about loops**, which is the largest gap between the construct as specified and
  the fixtures as written, and nothing about dtype conversion — though when a conversion intrinsic
  exists it is a compute like any other, so it needs a group and a buffer and no new surface, and it
  is where `staggered` first becomes real.
