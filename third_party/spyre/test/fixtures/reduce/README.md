# reduce

Reductions over one axis with tensor descriptors: `out = OP(in, axis=1)`.
Exercises the **one program per core** idiom — each of the 32 Spyre cores runs
one program that loops over its share of the outer axis — and the axis
bookkeeping the reduce itself needs, which is what separates this fixture from
`elementwise`.

Three reductions are supported via `OP: tl.constexpr`: `sum`, `max`, `min`. All
three lower to a `linalg.reduce` and differ only in the combiner.

Both shapes reduce **axis 1**, which is why one input generator and one oracle
serve the whole fixture: the output shape is the input's with axis 1 dropped.

## Variant hierarchy

```
Level D  device               loop-free, stick-tiled; 2 variants, 1 launches
         one_tile              1 key (folds the NON-stick axis; compiles_to_binary)
         one_tile_on_stick     1 key (folds the STICK axis; ktir_cpu only)

Level C  layout                stick physicalization, one dtype per variant
         spyre_stick               1 key (fp16, stick on the REDUCED axis)
         middle_axis_spyre_stick   1 key (fp32, stick on an UNREDUCED axis)

Level B  compute               fp16/fp32/i32 × {sum,max,min}
         2d_compute            9 keys (ktir_cpu only)

Level A  shape/distribution    fp32, sum  (DTYPE and OP pinned)
         default, grid_8, middle_axis, middle_axis_grid    8 keys
```

## Variants

### Level A — shape and distribution (OP="sum", DTYPE=fp32)

No layout annotation anywhere here; that is Level C.

- **default** (`reduce[M=..., N=...]`) — 4 keys. `out[m] = sum(in[m, :])` over a
  static `M × N`, distributed across 32 cores. `M` and `N` sweep while `BLOCK_M`
  is held fixed: `BLOCK_M` interacts with the grid partition
  (`rows_per_core = cdiv(cdiv(M, BLOCK_M), grid)`), so sweeping both would
  conflate tiling with distribution. `M = 768` gives a block count that is not a
  multiple of the core count, so `rows_per_core` comes out ragged.
- **grid_8** (`reduce__grid_8[M=...]`) — 2 keys. The same work on 8 cores instead
  of 32: more rows per core, a different `DistributeWork` split. `grid` is a
  top-level field read once per variant, so it is not sweepable through `params`
  — varying it means sibling variants.
- **middle_axis** (`reduce__middle_axis`) — rank-3 reduce over the **non-trailing**
  middle axis, `out[d0, d2] = sum(in[d0, :, d2])`. The three extents are distinct,
  so a reduce that folded the wrong axis would change the output shape and fail
  loudly. Numerical counterpart of the `@reduce_middle_axis` lit case.
- **middle_axis_grid** (`reduce__middle_axis_grid`) — the same rank-3 reduce split
  four ways: `D0 = 16` over `BLOCK_D0 = 4` is one block per core at `grid=[4]`.
  The reduced axis is untouched by the split, which is the point.

### Level B — compute correctness (OP × DTYPE sweep)

- **2d_compute** (`reduce__2d_compute[DTYPE=..., OP=...]`) — 9 keys sweeping
  `fp16/fp32/i32` × `sum/max/min`. The simplest shape in the file (2D, one tile,
  one core, no layout) so the combiner is the only thing that varies. All nine
  pass on ktir_cpu, and eight of them bit-exactly: `max` and `min` are selections,
  and integer addition is associative. Only the float sums drift, which is what
  sets the tolerance — see the comment on the variant.

  None of them carries `compiles_to_binary`, and unlike `elementwise`'s Level B
  that is not about the arithmetic: they distribute over the program id and
  dbo-opt refuses the loop that outlines from that. The one reduce that does
  reach a binary is the loop-free `one_tile` at `AXIS=0` (Level D).

### Level C — layout (stick physicalization, on ktir_cpu)

The two arms are chosen for where the stick lands relative to the reduced axis.

- **spyre_stick** (`reduce__spyre_stick[IN_LAYOUT=stick, OUT_LAYOUT=stick]`) —
  fp16. `in_ptr` is stick-on-N, so the **reduced** axis is the one split across
  the leading and trailing physical dims (`[M, N]` → `[ceil(N/S), M, S]`); this is
  the source-reduce path through `RewriteDescriptorLayout`. `out_ptr` is a 1D
  stick.
- **middle_axis_spyre_stick** (`reduce__middle_axis_spyre_stick[IN_LAYOUT=stick]`)
  — fp32. `in_ptr` is stick-on-D2, an **unreduced** axis, which leaves the reduced
  axis (D1) non-trailing in the physical tile `[2, 16, 96, 32]`. Nothing is done
  about that: `linalg.reduce` names the axes it folds in a sorted `dimensions`
  list, so D1 is reduced where it sits and no `linalg.transpose` is emitted. The
  variant asserts the transpose's absence, and a permutation derived from slot
  indices would be the identity here and reduce the wrong axis.

One dtype per variant, spelled once, in the `params` row that carries `N` (or
`D2`) and the layouts that follow from it. A second dtype row would be one line;
what keeps it out is that `rtol`/`atol` are fields rather than params, so both
rows would share the tolerance the wider dtype needs and the fp32 arm would be
checked more loosely than it can be.

### Level D — device

**Two variants, one of which launches.** Both are the whole reduce in a single
stick-tiled tile with no `tl.program_id` and no loop — elementwise's Level D
shape, the one that does reach a binary — so what they hit is reduce's own
obstacle rather than the `scf.for` refusal every other variant here shares.

They are two variants rather than one `AXIS` sweep because what divides them is
`compiles_to_binary` and `atol`, which are *fields*: one variant cannot carry two
answers. That is the same constraint that keeps Level C at two variants. Collapse
them into a single `AXIS: [0, 1]` sweep the day the on-stick arm reaches the
device too.

The first obstacle is shared, and neither variant asserts it. dbo-opt stops on

```
V1 only supports add/mul/sub/reduce compute ops; found unsupported compute op
```

naming the neutral-element `linalg.fill` that `LowerComputeOps` puts on the
reduction's `outs`. `linalg.reduce` is itself in that allowlist; the fill beside
it is not. `DropReductionInitFill` removes exactly that fill, and the backend runs
it unconditionally in `_make_spyrecode`, out of `_SPYRECODE_STAGE_PASSES` — the
binary path only, since nothing that stops at KTIR cares. With it, our emission
matches torch-spyre's, whose emitter never writes a fill in the first place.

It lives on the binary path rather than in the pipeline every path crosses because
it is a requirement of dbo-opt, not of the IR — and because the pass refuses any
combiner whose identity is not zero: the scheduler resets a reduction accumulator
to zero whatever the combiner is, so it reports rather than miscompile. On the
KTIR path a `max` reduce still lowers and still runs on `ktir_cpu`.

Past that the two diverge, on what happens to the stick split:

- **one_tile** (`reduce__one_tile[IN_LAYOUT=stick, OUT_LAYOUT=stick]`) — folds M,
  the non-stick axis, deleting a whole physical dimension. The split of N
  survives, and because `OUT_LAYOUT` declares exactly the layout that leaves,
  `RewriteDescriptorLayout` physicalizes the reduce's output and the surviving
  stick index becomes a **batch dimension** of the one `linalg.reduce`:
  `ins tensor<2x64x64> outs tensor<2x64> dimensions = [1]`, with `ktdp.store`
  consuming the result directly. That is the shape torch-spyre's working `sum`
  emits, and it reaches a binary and launches.

  It used to walk the stick axis with an `scf.for` and a slice per stick, plus a
  second loop to re-tile for the store, and dbo-opt rejected the loop.
  `Conversion/rewrite-descriptor-layout-reduce-batch-dim.mlir` pins the form that
  replaced it; `docs/spyre-tensor-layouts.md` is the design.

  Its `atol` is 0.25 rather than the elementwise-shaped 5e-2, set by the device
  arm and sized in ulp: the sums reach 24 where fp16 ulp is 0.015625, the device
  drifts 0.1367 = 8.75 ulp against the fp16 oracle, and 0.25 is 16 ulp. 8.75 is
  what a 64-term reordering predicts (`sqrt(64)` = 8), so the device is behaving.
  It is this device's accumulation order that differs, not fp16 order in general —
  NumPy's fp16 sum and a plain sequential fp16 sum agree here bit for bit.

- **one_tile_on_stick** (`reduce__one_tile_on_stick[IN_LAYOUT=stick, OUT_LAYOUT=stick]`)
  — folds N, the stick axis itself, which stick-on-N splits across physical dims 0
  and 2. Both are reduced, so no batch dimension survives to carry the split: the
  result collapses to rank 1 and then fails on a rank-2 store against a 1-result
  `dest_map`. torch-spyre emits this case as a `linalg.generic` instead, and
  currently fails it too, so there is no working emission to match.

  That refusal is not asserted anywhere: what this arm tests is that the reduce
  lowers and computes the right numbers on `ktir_cpu`. It keeps `atol` at 5e-2,
  since nothing here runs on the device and it drifts only 0.0122 = 0.78 ulp —
  inheriting the sibling's 0.25 would check it 20x looser than it needs.

Both assert the absence of `scf.for` structurally: no distribution loop and no
stick loop.
