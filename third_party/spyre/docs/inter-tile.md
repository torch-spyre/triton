# Inter-Tile Reduction and Collectives

**Status:** design note for the `tl` → `tt` → `ktdp` lowering of cross-tile collectives.

When a kernel's iteration space is sliced across tiles (cores), some results must be
combined *across* those tiles before the kernel can finish: a contraction split across
tiles leaves each tile with a partial sum; a reduction split across tiles leaves each tile
with a partial max/sum; a value computed on one tile must sometimes be replicated to all of
them. None of these are expressible with the per-tile compute ops alone — each tile sees
only its own data. This note designs the construct that fills that gap, and how it lowers.

---

## 1. The layering

```
┌──────────────────────────────────────────────────────────────┐
│  tl  (Python kernel surface)                                  │
│  intent only: "reduce this across <axis> with <combiner>"     │
│  work-slice shape supplied as a HINT, not hand-written        │
└───────────────────────────────┬──────────────────────────────┘
                                │ Triton → TTIR
┌───────────────────────────────▼──────────────────────────────┐
│  tt  (TTIR)                                                   │
│  metadata op + function attributes (numWkSlicesPerDim,        │
│  coreIdToWkSlice) describing the work-slice → tile map        │
└───────────────────────────────┬──────────────────────────────┘
                                │ TTIR → KTIR  (LowerInterTile pass)
┌───────────────────────────────▼──────────────────────────────┐
│  ktdp  (KTIR)                                                 │
│  future-based producer/consumer ops (produce + delivery):     │
│  consume / reduce / reduce_scatter, with explicit tile groups │
└───────────────────────────────┬──────────────────────────────┘
                                │ schedule + dataflow lowering
┌───────────────────────────────▼──────────────────────────────┐
│  the hard floor                                               │
│  no collective exists here — only per-tile send/recv plus a   │
│  local combine. The algorithm (chain topology, in-flight vs   │
│  staged) is chosen by a scheduling pass, not named in the op. │
└──────────────────────────────────────────────────────────────┘
```

The bottom layer is the immovable constraint. There is no "reduce across tiles" primitive
at the lowest IR: a collective is always materialized as a concrete set of per-tile programs
that send sticks to one another and run a local combine on the receiving side. Everything
above exists to let the frontend *not* write that out by hand, and to give a scheduling pass
one well-scoped place to choose the algorithm.

---

## 2. The two upstream proposals

Two proposals shape this design. They agree on the semantics and disagree on the surface.

### 2.1 A produce/reduce pair with a combiner lambda

The first proposal exposes a **pair** of `tl` ops:

- `inter_tile_produce(partial, producer_groups=[[...]])` → a *future*. Each tile contributes
  its local partial to a producer group.
- `inter_tile_reduce(fut, consumer_groups=, identity=, combine=)` → the reduced result. A
  **combiner lambda** (`lambda lhs, rhs: lhs + rhs`) generalizes the op past a fixed menu of
  reductions — sum, max, argmax, and online-softmax-style combines all fit the same shape.

Two important ideas come from here and survive into this design:

- **The combiner is a region, not an enum.** A lambda keeps the op open-ended.
- **Reduce-to-one vs all-reduce is a consumer-side property**, expressed by guarding the
  writer (e.g. only one tile in the group keeps/writes the result) rather than by a separate op.

The cost: the author hand-writes core-ID groups (`[[0, 1, 2, 3]]`). That couples kernels to a
physical tile layout the frontend should not have to know.

### 2.2 A future-based producer/consumer split

The second proposal generalizes the pair into one production op feeding several delivery ops,
threaded by an explicit in-flight value (a `tile_future`):

| Pattern | Production op | Delivery op |
|---|---|---|
| **Broadcast** | `inter_tile_produce` | `inter_tile_consume` |
| **Reduce** | `inter_tile_produce` | `inter_tile_reduce` |
| **Reduce-scatter** | `inter_tile_produce` | `inter_tile_reduce_scatter` |

All patterns share one producer; only the delivery differs. The shape of the result follows
from the relationship between the producer and consumer tile groups:

- **all-reduce** — consumer set equals producer set (every contributing tile receives the result),
- **reduce-to-one** — the consumer set is a single tile,
- **reduce-to-subset** — a subset receives the result.

Broadcast is the combiner-less case: a value produced on one tile is delivered to a consumer
group with no combine. Multi-arity reductions (e.g. value + index for argmax) are supported by
producing more than one partial.

This is the model this design targets at the KTIR layer.

### 2.3 The tension

The first proposal is *full-blown*: the author supplies core-ID groups and the combiner region
directly. The second is *structured* but still group-explicit. We want the frontend to express
**intent** — what is reduced, along which axis, with which combiner — and let the lowering derive
the tile groups from work-slice metadata. The question is how much of the producer/consumer split
to surface at `tl`/`tt`, and how much to leave to lowering.

---

## 3. Op shape: fused metadata op vs. surfaced split

### Approach A — single fused metadata op at `tl`/`tt`

The frontend emits one lightweight op carrying intent (`axis`, `combiner`, `mode`). The
TTIR→KTIR pass **expands** it into the producer/consumer pair (`produce` + the right delivery op).

### Approach B — surface the producer/consumer split at `tl`/`tt`

The frontend emits the `produce`/`deliver` pair itself, mirroring the KTIR ops — but each side
references an **axis** instead of explicit core-ID groups, keeping it metadata-style.

| Criterion | Fused op (A) | Surfaced split (B) |
|---|---|---|
| Frontend complexity | One op, three attributes. | Two ops + a future value to thread. |
| Matches the kernel mental model | "reduce x across axis" — direct. | Author reasons about produce/deliver phases. |
| Broadcast / multi-consumer | A `mode`/combiner-less variant. | First-class — `produce` + `consume`. |
| Decoupled producer & consumer placement | Harder — one op fixes both ends. | Easier — distinct producer/consumer groups. |
| TTIR→KTIR lowering | Non-trivial: expand one op into the pair. | Closer to 1:1 with the KTIR ops. |
| Maps cleanly to the KTIR future model | Via expansion. | Directly. |

**Recommendation: Approach A for the `tl`/`tt` surface.** The producer/consumer split is a real
and useful structure, but it belongs *below* the frontend. A kernel author thinks "reduce this
across the tile axis," not "produce a future, then deliver it." The split, the future, and the
tile groups are all synthesized in one well-scoped lowering pass. Approach B should be revisited
if decoupled producer/consumer placement (e.g. produce on one tile group, consume on a disjoint
one) becomes a common frontend need rather than a lowering-internal detail.

The remainder of this note follows Approach A.

---

## 4. The `tl` surface (metadata style)

A single op covers all four delivery modes (reduce, reduce-scatter, broadcast), so it is named
`tl.inter_tile` rather than `..._reduce` — the `mode` argument, not the op name, picks the
semantic:

```python
reduced = tl.inter_tile(
    x,                       # this tile's local partial
    axis="in",               # which work-slice dimension to reduce across
    combiner=lambda a, b: a + b,   # generic combine (or a shorthand, see below)
    mode="all_reduce",       # all_reduce | reduce_to_one | reduce_scatter | broadcast
)
```

- **`axis`** names a *work-slice* dimension, not a tensor axis. It identifies the set of tiles
  that cooperate (the ones that differ only along this dimension). It does **not** name physical
  core IDs.
- **`combiner`** is a lambda by default — preserving the generic-combine property: `a + b` (sum),
  `tl.maximum(a, b)` (max), an argmax-style select over a (value, index) pair, an online-softmax
  combine. Shorthands (`combiner="add"`, `"max"`, `"mul"`) are accepted for the common cases and
  carry a known identity.
- **`mode`** selects the delivery semantic:
  - `all_reduce` — every cooperating tile ends with the result.
  - `reduce_to_one` — only a designated tile holds the result.
  - `reduce_scatter` — the result is split back across the cooperating tiles.
  - `broadcast` — no combine; a value on one tile is replicated to the group (the combiner-less
    delivery).

What is **not** an argument: the core-ID groups and the core→slice map. The author never writes
`[[0, 1, 2, 3]]`. The work-slice shape that determines the groups is supplied as a **hint**
through the existing hint mechanism (the same path that already feeds `numWkSlicesPerDim` /
`coreIdToWkSlice`; see `work-slicing.md`). The reduction op only references the *name* of the
axis that hint established.

### Examples (schematic)

Each example assumes a work-slice hint has named the cooperating axis. The
`numWkSlicesPerDim` it produces is shown in a comment so the `axis=` argument is easy to trace:
the count on the named axis is the number of tiles that combine.

A contraction split across tiles — each tile has a partial product, summed across tiles:

```python
# hint ⇒ numWkSlicesPerDim = {out: 16, in: 2}   → axis "in" = 2 tiles cooperate per output slice
psum = tl.dot(a_slice, b_slice)
full = tl.inter_tile(psum, axis="in", combiner=lambda a, b: a + b,
                     mode="all_reduce")
```

A row-reduction split across tiles — local max, then max across tiles:

```python
# hint ⇒ numWkSlicesPerDim = {row: 16, out: 2}  → axis "row" = 16 tiles cooperate per output slice
local_max = tl.max(x, axis=1)
g_max = tl.inter_tile(local_max, axis="row", combiner=lambda a, b: tl.maximum(a, b),
                      mode="all_reduce")
```

A value replicated from its producing tile to the rest of the group:

```python
# hint ⇒ numWkSlicesPerDim = {row: 16, out: 2}  → axis "row" names the group to broadcast within
shared = tl.inter_tile(scalar, axis="row", combiner=None, mode="broadcast")
```

---

## 5. The `tt` op and its function attributes

At TTIR the op stays lightweight, but it is **not self-contained** — it reads function-level
attributes that describe how the iteration space is sliced across tiles.

```mlir
func.func @kernel(...)
    attributes {
      grid = [..., ...],
      numWkSlicesPerDim = {x = 1, out = ..., in = ...},
      coreIdToWkSlice   = { ... }      // core → slice-index map (see §8)
    } {
  ...
  %result = tt.inter_tile_reduce %partial,
              axis = "in",
              combiner = #tt<add>,            // shorthand, or a region (below)
              mode = #tt<all_reduce>
            : tensor<...> -> tensor<...>
  ...
}
```

**Attributes**

| Attribute | Meaning | Required |
|---|---|---|
| `axis` | Names a key in `numWkSlicesPerDim`; the reduction cooperates across tiles that differ only on this dimension. | yes |
| `combiner` | Shorthand (`#tt<add>` / `#tt<max>` / `#tt<mul>`) or a region for custom combines. | yes (unless `mode = broadcast`) |
| `mode` | `all_reduce` / `reduce_to_one` / `reduce_scatter` / `broadcast`. | yes |
| `scatter_dimension` | For `reduce_scatter`: result axis to split along. | only for `reduce_scatter` |

**Region-form combiner** (the lambda, lowered):

```mlir
%v, %i = tt.inter_tile_reduce %partial_v, %partial_i,
           axis = "in",
           identity(%neg_inf, %neg_one),     // required for region combiners (see §9)
           combiner = {
             ^bb0(%va: f16, %vb: f16, %ia: i32, %ib: i32):
               %gt   = arith.cmpf ogt, %va, %vb : f16
               %mv   = arith.select %gt, %va, %vb : f16
               %mi   = arith.select %gt, %ia, %ib : i32
               tt.yield %mv, %mi : f16, i32
           },
           mode = #tt<all_reduce>
         : tensor<...>, tensor<...> -> tensor<...>, tensor<...>
```

**Verifier rules**

- `axis` must name a key in `numWkSlicesPerDim` with count > 1 (a degenerate count of 1 means no
  cooperation is required and the op folds away).
- A region-form combiner must be accompanied by explicit `identity` operands (see §9).
- `scatter_dimension` is present iff `mode = reduce_scatter`.
- The function must carry the work-slice attributes; without them the op cannot be lowered.

---

## 6. TTIR → KTIR lowering

A single pass (`LowerInterTile`) converts the metadata op into the future-based producer/consumer
ops. It performs four steps.

**1. Derive the tile groups from work-slice metadata.** `axis` plus `numWkSlicesPerDim[axis]`
gives the group size (how many tiles cooperate); the remaining dimensions give the number of
groups. `coreIdToWkSlice` resolves *which* tiles land in each group (see §8). The pass emits the
`producer_tiles_per_group` and `consumer_tiles_per_group` sets the KTIR ops expect.

**2. Select the delivery op and consumer set from `mode`.**

| `mode` | KTIR delivery op | consumer set vs producer set |
|---|---|---|
| `all_reduce` | `inter_tile_reduce` | consumer = producer |
| `reduce_to_one` | `inter_tile_reduce` | consumer = single tile |
| `reduce_scatter` | `inter_tile_reduce_scatter` | consumer = producer, result sliced |
| `broadcast` | `inter_tile_consume` | producer = single tile, consumer = group |

**3. Synthesize the future and the regions.** The `%partial` becomes the producer's contribution;
`inter_tile_produce` yields a `tile_future`; the delivery op consumes it. A shorthand combiner
expands into a reducer region with a known identity; a region-form combiner is transcribed
(its `tt.yield` becomes the KTIR reduced-yield) and its supplied identity operands are attached.

**4. Derive the result type.** For `all_reduce` / `reduce_to_one`, the within-group tile axes
collapse out of the partial type. For `reduce_scatter`, the `scatter_dimension` extent is further
divided by the group size. For `broadcast`, the result type equals the produced type.

### Walkthrough

Take a contraction split 2-ways across the `in` axis, with the work-slice attributes:

```
grid = [16, 2]                            → 32 tiles total
numWkSlicesPerDim = {x: 1, out: 16, in: 2}
```

**Step 1 — group structure.** `axis = "in"` and `numWkSlicesPerDim["in"] = 2` give a group size
of 2; the product of the other dimensions gives the group count:

```
group_size = numWkSlicesPerDim["in"]      = 2
num_groups = prod(other dims) = 1 * 16    = 16
```

`coreIdToWkSlice` resolves which two tiles share each output slice (the pair that agrees on `out`
and differs on `in`), yielding 16 disjoint groups of 2 tiles. The affine sets the KTIR ops expect:

```mlir
// participating tiles for group g: tiles [2g, 2g+1]
#group_tiles = affine_set<(i)[g] : (i - 2*g >= 0, -i + 2*g + 1 >= 0)>
// 16 groups, indices 0..15
#all_groups  = affine_set<(g) : (g >= 0, -g + 15 >= 0)>
```

**Step 2 — TTIR input:**

```mlir
%psum = linalg.matmul ins(%a, %b) : tensor<BLOCK_MxBLOCK_Nxf16>
%full = tt.inter_tile_reduce %psum, axis = "in",
          combiner = #tt<add>, mode = #tt<all_reduce>
        : tensor<BLOCK_MxBLOCK_Nxf16> -> tensor<BLOCK_MxBLOCK_Nxf16>
```

**Step 3 — KTIR output** (producer + reduce delivery; `#tt<add>` expands to a zero identity and a
sum reducer; `consumer = producer` ⇒ all-reduce):

```mlir
%zero = arith.constant 0.0 : f16
%id   = linalg.fill ins(%zero) outs(%init) : tensor<BLOCK_MxBLOCK_Nxf16>

%fut = ktdp.inter_tile_produce %psum
         producer_tiles_per_group = #group_tiles
       : tensor<BLOCK_MxBLOCK_Nxf16> -> !ktdp.tile_future<BLOCK_MxBLOCK_Nxf16>

%full = ktdp.inter_tile_reduce %fut
          identity(%id : tensor<BLOCK_MxBLOCK_Nxf16>)
          consumer_tiles_per_group = #group_tiles      // = producer ⇒ all-reduce
          groups = #all_groups
          : !ktdp.tile_future<BLOCK_MxBLOCK_Nxf16> -> tensor<BLOCK_MxBLOCK_Nxf16>
reducer {
  ^bb0(%lhs: tensor<BLOCK_MxBLOCK_Nxf16>, %rhs: tensor<BLOCK_MxBLOCK_Nxf16>):
    %s = linalg.add ins(%lhs, %rhs) outs(%init) : tensor<BLOCK_MxBLOCK_Nxf16>
    ktdp.yield_reduced %s : tensor<BLOCK_MxBLOCK_Nxf16>
}
```

`reduce_to_one` differs only in that `consumer_tiles_per_group` names a single tile;
`broadcast` drops the reducer region and uses `inter_tile_consume`; `reduce_scatter` uses the
scatter delivery op and a sliced result type. The chain topology and whether the combine happens
in-flight or after a staging step are **not** decided here — they are left to the scheduling pass
below KTIR.

---

## 7. Example patterns

These show how each reduction pattern maps to the `tl` op and the `tt` op it lowers to. The
DFIR realization (chain topology, where the local combine runs) is deliberately omitted — it is
chosen by the scheduling pass below KTIR and is not visible at this layer.

### 7.1 Contraction sum — ADD, all-reduce

A K-split contraction: each tile holds a partial product, summed across the `in` axis. This is
the §6 walkthrough.

```python
# numWkSlicesPerDim = {out: 16, in: 2}
full = tl.inter_tile(psum, axis="in", combiner=lambda a, b: a + b, mode="all_reduce")
```
→ `tt.inter_tile_reduce ... combiner = #tt<add>, mode = #tt<all_reduce>`

### 7.2 Row max / row sum — MAX or ADD, all-reduce

A reduction whose axis is split across tiles: reduce locally, then combine the per-tile partials.
Max and sum differ only in the combiner (and its identity, −inf vs 0).

```python
# numWkSlicesPerDim = {row: 16, out: 2}
local_max = tl.max(x, axis=1)
g_max = tl.inter_tile(local_max, axis="row", combiner=lambda a, b: tl.maximum(a, b),
                      mode="all_reduce")
# ... and the sum half of the same reduction:
g_sum = tl.inter_tile(local_sum, axis="row", combiner=lambda a, b: a + b, mode="all_reduce")
```
→ `combiner = #tt<max>` and `combiner = #tt<add>` respectively.

### 7.3 Mean — sum across tiles, then scale

The cross-tile part is a plain sum; the in-tile portion (the per-tile partial) and the final
divide are ordinary local compute and need no inter-tile op.

```python
# numWkSlicesPerDim = {row: 2, out: 16}
g_sum = tl.inter_tile(local_sum, axis="row", combiner=lambda a, b: a + b, mode="all_reduce")
mean  = g_sum / count
```

### 7.4 Argmax — multi-arity region combiner

A (value, index) pair reduced together. The lambda takes two operands per role and returns both;
because it is a custom combiner, the lowering requires an explicit identity (−inf, −1).

```python
# value + index reduced as one unit across the "in" axis
def argmax_combine(va, ia, vb, ib):
    gt = va > vb
    return tl.where(gt, va, vb), tl.where(gt, ia, ib)

v, i = tl.inter_tile((val, idx), axis="in", combiner=argmax_combine,
                     identity=(float("-inf"), -1), mode="all_reduce")
```
→ a region-form `tt.inter_tile_reduce` with two partials, two identities, and a four-argument
reducer region (see §5).

### 7.5 Broadcast — replicate, no combine

A value produced on one tile delivered to the rest of its group; `combiner=None` selects the
combiner-less delivery.

```python
# numWkSlicesPerDim = {row: 1, out: 32}  → input not split; replicate to all
shared = tl.inter_tile(scalar, axis="row", combiner=None, mode="broadcast")
```
→ lowers to `ktdp.inter_tile_produce` + `ktdp.inter_tile_consume` (no reducer region).

### 7.6 No cooperation — folds away

When every named axis has count 1 on the reduction dimension (embarrassingly parallel work), the
op is unnecessary and the verifier/canonicalizer folds it away — the frontend need not special-case
this. There is simply no `tl.inter_tile` call to emit.

---

## 8. `coreIdToWkSlice` — the connective metadata

The reduction op names an **axis**, an abstract work-slice dimension. An axis alone does not say
*which tiles* cooperate. Two pieces of metadata close that gap:

- **`numWkSlicesPerDim`** — how many slices each named dimension is cut into. The count on the
  reduction axis is the **group size**; the product of the others is the **number of groups**.
- **`coreIdToWkSlice`** — the map from each tile to its slice index along every dimension. This is
  what tells the lowering that, say, the tiles sharing an output slice but differing along `in`
  are exactly the ones that must combine. Without it, "reduce across `in`" cannot be turned into a
  concrete set of cooperating tiles.

This is the same role the scheduling-level description played in the older pipeline: the work-slice
map was the single source of truth for which tile owned which slice, and every downstream step read
it. The reduction construct does not replace that map — it *depends* on it.

**Where it lives at each layer:**

- **`tl`** — supplied as a **hint**. The author does not write the core→slice map; they attach a
  work-slicing hint (the same mechanism that already establishes `numWkSlicesPerDim` /
  `coreIdToWkSlice`; see `work-slicing.md` §4). The reduction call only references the resulting
  axis *name*.
- **`tt`** — carried as **function attributes** (`numWkSlicesPerDim`, `coreIdToWkSlice`) on the
  enclosing `func.func`. The `LowerInterTile` pass reads them to materialize the producer/consumer
  tile groups. This is the layer where the abstract axis becomes a concrete tile set.

So the metadata is threaded, not invented: a hint at the Python surface becomes a function
attribute in TTIR, which the lowering consumes to build the KTIR tile groups.

---

## 9. Open questions

1. **Identity for custom combiners.** Shorthands carry a known identity (sum→0, max→−inf, mul→1).
   Region-form combiners cannot, in general, have their neutral element inferred — so the design
   **requires** explicit `identity` operands on the op when the combiner is a region, and the
   verifier enforces it. (Synthesizing the identity from the region body is possible only in
   restricted cases and is not relied upon.)
2. **Broadcast: its own op or a combiner-less reduce?** Broadcast is modeled here as
   `mode = broadcast` (a combiner-less delivery that lowers to the KTIR consume op). An alternative
   is a dedicated `tl.inter_tile_broadcast`. The combiner-less mode keeps the surface small;
   a dedicated op would be more self-documenting. Left open pending frontend feedback.
3. **Reduce-scatter dimension naming.** `scatter_dimension` is a positional index into the result
   tensor (a property of the result, not of the work-slice distribution), whereas `axis` is a
   work-slice name. This deliberate asymmetry should be confirmed against the KTIR scatter op's
   convention.
4. **Revisiting Approach B.** If kernels need to place producers and consumers on *disjoint* tile
   groups (beyond all-reduce / reduce-to-one / reduce-scatter), surfacing the producer/consumer
   split at the frontend (Approach B, §3) becomes more attractive. Until then the fused op is
   simpler and the split stays a lowering-internal concern.
