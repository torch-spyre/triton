# softmax_optimization

Three Triton kernels over one measured softmax, written to be diffed. They differ only
in **who decides where the intermediates live**, and the set is meant to make that
division of labour visible rather than to declare a winner.

| kernel | who allocates | groups | full-tile scratchpad materializations |
|---|---|---:|---:|
| `softmax_opspec` | the author, everywhere | 6 | 3 — staged `x`, `sub`, `exp` |
| `softmax_fused` | the author stages `x`, the compiler does the rest | 5 | 2 — staged `x`, `e` |
| `softmax_fused_auto` | the compiler, everywhere | 5 | 2 in principle — see below |

**Status: draft, deliberately inert.** `tl.spyre_pin` does not exist,
`tl.spyre_tensor_layout` takes neither a memory space nor an element arrangement yet
(see [`spyre-tensor-layout-ea.md`](../../../docs/spyre-tensor-layout-ea.md) and
triton#164), and there is no `meta.py`, so `conftest.py::_load_examples` never
discovers this folder. Nothing runs and nothing fails.

## The reference artifact

`sdsc_output_code.py` is the generated wrapper for this case, copied verbatim from
`op-level-tests/gpt-oss-20b/SDSC/torch.nn.functional.softmax-1x64x11x2049/output_code.py`
so the six OpSpecs, their `allocation` dicts and their `device_layout` are readable
beside the kernels rather than only in a tree nobody else has. It is evidence, not
code: nothing imports it, `conftest.py` globs `meta.py` and so never sees it, and it
would not run here in any case.

One edit, so the file carries no path that resolves only on one machine: five
`SourceLoc` entries named an absolute `/home/...` path to `runner.py` and now name it
relative to the repository root. Nothing else was touched.

## The case

`op-level-tests/gpt-oss-20b/SDSC/torch.nn.functional.softmax-1x64x11x2049`, the
largest softmax in that corpus by 11×.

```
logical        [1, 64, 11, 2049] fp16, 1,442,496 elements
device_size    [64, 11, 33, 1, 64]   stride_map [22539, 2049, 64, -1, 1]
iteration      d0 = 64 over 32 cores, d1 = 11, d2 = 2049
per core       [2, 11, 2049] logical, [2, 11, 33, 64] physical, ~91 KiB
reduction      the last axis, 2049
```

The kernels squeeze the leading extent-1 logical dim, so their descriptors are rank 3
where the record is rank 4. Nothing else about the geometry differs.

## What a group is

> A **group** is a maximal region of dataflow whose operations share one iteration
> space — the same extents **and the same iterator types** — bounded by
> materializations. It lowers to one `linalg.generic` after elementwise fusion, inside
> one function, which becomes one three-stage pipeline; on the SDSC path, one SDSC.

The iterator-types half matters here and is easy to miss. `exp(x - m)` and
`sum(e, axis=2)` have the *same extents*, `[2, 11, 2049]`, and are still different
groups, because `d2` is `parallel` for the first and `reduction` for the second.

## What the SDSC path emits

One kernel, one bundle, **six SDSCs**, from `ir_post_fusion.txt` and the six
`sdsc_N.json` files:

| SDSC | op | result | where it lands |
|---|---|---|---|
| 0 | `identity` | `buf5`, full tile | `lx` |
| 1 | `max` (reduction) | `buf0`, `[1,64,11,1]` | `lx` |
| 2 | `sub` | `buf1`, full tile | `lx` |
| 3 | `exp` | `buf2`, full tile | `lx` |
| 4 | `sum` (reduction) | `buf3`, `[1,64,11,1]` | `lx` |
| 5 | `realdiv` | `buf4`, full tile | `hbm` (the output) |

Two things worth knowing before treating this as a case to improve, because both cut
against the obvious reading:

**The intermediates are already on-chip.** Every allocation in the wrapper is
`{'lx': 0}` or `{'lx': 2816}` — two slots, alternating, so the planner is already
double-buffering in the scratchpad. Only the input and the output touch HBM. There is
no HBM round-trip here to remove.

**And it fits comfortably.** 704 rows over 32 cores is 22 rows per core, ~91 KiB
against 2 MiB of LX.

So the cost is not memory space and not HBM traffic. It is **passes over the data**.

## What fusing removes, and what it does not

**One removal.** `sub` and `exp` are pointwise on one iteration space, so with no pin
between them they are one `linalg.generic` and `x - m` never reaches memory. The
six-SDSC shape comes from the OpSpecs arriving pre-split, not from the scheduler
declining to fuse them — Step 2 of `ConstructThreeStagePipeline` calls
`linalg::fuseElementwiseOps` and would have fused them had they arrived together.

Four things that stay, none of them a missed opportunity:

- **the staging of `x` — and this is a correction worth reading.** An earlier draft of
  this README called `identity` "a full tile stored and reloaded for nothing". It is
  not. `max` and `sub` both need the whole tile and cannot share a group, so the tile
  is read **twice either way**. Staging chooses where from:

  | | HBM reads of `x` | scratchpad traffic |
  |---|---:|---|
  | with the stage | **1** | 1 store + 2 reads |
  | without it | **2** | none |

  Given how much dearer an HBM read is than a scratchpad access, the stage is the right
  call. `identity` does come from a clone in the traced graph, but it is not idle.
- **both reductions are boundaries**, because each eliminates the last axis.
- **`e` is materialized**, and needs no pin to be: `sum` and the divide both read it,
  and a value cannot cross a group boundary in a register — schedules are extracted
  into separate modules.
- **`m` and `s` are materialized**, being reduction outputs.

## Why five is the floor today, and why it is a floor rather than a law

Two backend limitations set it, and both are worth naming as limitations:

- **A reduction group holds exactly one arithmetic op.**
  `KTDFLowToDFIR/LinalgLowering.cpp:282` — *"reduction linalg.generic body must have
  exactly one compute op"*. So nothing fuses into `sum` or `max`, not even the `exp`
  immediately before it.
- **Only elementwise fuses with elementwise.** Step 2's own comment names the rest as
  future work: *"TODO: we may want do non-element-wise fusion as well such as matmul
  followed by add."*

So the fusible opportunity in this kernel is exactly the pointwise runs, of which there
is one. Neither limitation is architectural, and if either is lifted the same *source*
yields fewer groups — which is the point of the next section.

## The three kernels are three divisions of labour

Not three quality levels. What separates them is who decides, and the consequence is
about time rather than about this softmax.

**`softmax_opspec` encodes a schedule.** Every buffer is stated, so it produces these
six groups now and these six groups after any backend improvement. That is what you
want when reproducing a measured path, and not what you want in a kernel you keep.

**`softmax_fused` is a hybrid**, and its one pin is doing the compiler's job. Staging a
value that already has a backing store is an allocation decision, and nothing today
promotes a twice-read input into the scratchpad — the planner places buffers that
already exist rather than creating one. The SDSC path gets the same staging by accident
of a graph-level clone, not by a planner that noticed two readers.

**`softmax_fused_auto` states only the arithmetic**, and it is the only one of the three
whose output improves when the backend does. A fuser that learned reduction-plus-
elementwise would give it fewer groups from the same source; a planner that learned to
promote a twice-read input would give it the staging for free. Today it is the one that
reads `x` from HBM twice, which is a real cost and a real argument for the other two —
but it is a cost with an owner, where the other two are workarounds with a shelf life.

## Two hardware costs, measured, that no kernel here changes

**Padding.** 2049 occupies 33 sticks of 64, so 63 lanes per row are unused and the last
stick of every row is 1/64 useful while costing a full stick of reduce work. The stick
is the unit of transfer, so this is a hardware property rather than something a kernel
can spend differently. All three pay it identically.

**The statistics blowup, and this is the first number we have for it.** `buf0` and
`buf3` are logically `[1, 64, 11, 1]` — 704 values — and the record gives each
`device_size = [11, 1, 1, 1, 64, 64]`, which is **45,056 slots. 64×.** That is the price
of an extent-1 axis on a machine whose smallest addressable unit is a stick: one value
per row occupies a whole stick, so a rank-reducing statistic comes back out stick-wide.
All three pay it twice, and it is what a `splat` coordinate entry exists to describe
rather than to fix.

## What this set does not attempt

**Online softmax**, which would be two passes rather than five by carrying a running
max and sum. A different algorithm rather than a scheduling difference, and `tl.max` /
`tl.sum` do not express it, so it belongs in its own example if it is wanted.

**Anything measured on hardware.** Group counts and materializations are countable from
the artifacts; time is not, and nothing here claims it. In particular, which of
`softmax_fused` and `softmax_fused_auto` is faster *today* is a measurement — the
argument above is about who should own the decision, not about the number.

## When `meta.py` is added

It needs `SIGNATURE`, `VARIANTS` with `params` and `constexpr`, a NumPy `reference`
oracle and an `inputs` generator. Add `__init__.py` at the same time; `_import_meta`
imports the folder as a package, which is also why this directory is spelled with an
underscore. `grid` is `[32]`.

One variant per kernel with a shared oracle, since all three compute the same function
— which makes the set a correctness test of the fusion and the staging as well as a
demonstration of them.
