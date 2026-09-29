# Element arrangement on `tl.spyre_tensor_layout`

One attribute, on an op that already exists. A layout says where a tensor's coordinates go; an
arrangement says how the values sit inside a stick once they get there, and today nothing at the
Triton level says it. This document proposes the surface for it and states what the compiler owes
in return.

Scope note. Memory space and scratchpad addresses are **not** here. They belong with
`tl.spyre_pin`, which annotates a value and so has both a type and a place to put an address; see
that proposal. This document is only the arrangement, which belongs on the descriptor for the same
reason the layout does — it is a property of the memory the descriptor names.

Companion to [`spyre-tensor-layouts.md`](spyre-tensor-layouts.md), which documents how descriptor
layouts are physicalized. That document is the mechanism; this one adds one fact to what a
descriptor carries.

## The attribute

```python
tl.spyre_tensor_layout(
    desc,
    [(1, "floordiv", 64), 0, (1, "mod", 64)],   # coordinate entries, one per physical dim
    element_arrangements=["standard", "standard", "staggered"],   # one per physical dim
)
```

| value | meaning |
|---|---|
| `standard` | sequential element order — the default, and what every entry is unless stated |
| `staggered` | values correct, within-stick position not matching logical order — what a dtype conversion produces |

Values come from `ElementArrangement`, which every `SpyreTensorLayout` already carries. The enum
holds more than these two and its integer encoding is stable by contract, so a frontend spelling
must not renumber it. `staggered` is the
[FP32 element-arrangement RFC](https://github.com/torch-spyre/RFCs/blob/main/2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md)'s
term for the two conversion-produced orderings.

Grouping it with the coordinate entries into a value type would earn its keep if two ops took the
same group. With one op there is nothing to share, so it is a keyword argument.

## One entry per physical dim, not one per tensor

An arrangement is a fact about a *dimension*, and it does not survive being detached from one. If
a `staggered` tensor is transposed, the staggered structure no longer aligns with the dim that
carried it, so naming the arrangement without naming the dim is not enough to interpret the
result. Position supplies the dim, exactly as it does for the coordinate entries.

In practice only the innermost physical dim — the stick — is ever non-standard today, so the list
is `standard` everywhere but one entry. The array is not for expressing several arrangements at
once; it is for keeping the fact attached to the right dim when a permutation moves it.

A single value per tensor is simpler and sufficient for every case that exists today, and
insufficient the moment a permutation moves the sticked dim. That is the trade, and it is the
first open item below.

## Who states it, and who derives it

**On a descriptor, the author states it.** A descriptor names memory the author placed, and the
arrangement of that memory is a fact about it that nothing in the kernel can recover. Same
standing as the coordinate entries.

**On an intermediate, the compiler derives it**, and no annotation is needed. The arrangement of a
computed value follows from the op that produced it — a dtype conversion yields `staggered`, an
ordinary arithmetic op preserves what it read — in the same way the physical layout follows from
the operand's, which [`spyre-tensor-layouts.md`](spyre-tensor-layouts.md)'s "Which physical shape:
the output axis space" already establishes for a reduce. An annotation on an intermediate would
carry no information the compiler does not already have.

## The check the compiler owes

**An element-arrangement mismatch is an error, and this is the one diagnostic this proposal asks
for.** The reason is that no fallback exists.

Contrast the layout case, where one does. Where the induced layout differs from what a store's
descriptor declares, `ReducePropagation` returns `failure()`, the value stays logical, and the
store's widen stage builds the physical form. That is a complete answer, just not the direct one,
so the difference costs a stage rather than correctness.

Nothing plays that role for an arrangement. A `staggered` operand read as though it were
`standard` produces wrong values, with nothing to signal it and no stage that repairs an
ordering. So the obligation is a positive one: every consumer's arrangement requirement met by
what its operand carries, checked across the whole graph, reported where it is not.

Two things that obligation is **not**:

- **not "every operand of an op has the same arrangement".** Different ops require different
  arrangements of their operands and produce different arrangements of their results. The rule is
  per-operand against that op's requirement, not agreement among operands.
- **not the author's job to satisfy by hand.** The author states what a descriptor carries and
  writes an explicit conversion where one is needed; finding an unmet requirement is the
  compiler's.

## Open items

- **`EXX2` is not in the value set.** It is a third `ElementArrangement` — a reduction putting two
  values in a stick rather than one — and layernorm's fused-pair buffer is written that way, which
  is what lets it be read as the pair or as one half. Excluding it means that target cannot be
  expressed. Including it raises a question that is about element types rather than about
  arrangement: how a consumer selects which half. That question is why it is deferred here rather
  than settled.
- **Per-dim or scalar**, as above. The transpose case is the whole of the argument for per-dim,
  and no target exercises it today.
- **An arrangement does not survive a shape change.** `ReshapePropagation` declines for
  `expand_shape`, `collapse_shape` and `reshape`, because collapsing a physical dim range would
  fuse the stick index with a data dim while the marker still describes the original dims; its own
  comment names itself as the one place a fix would go. Transposing a `staggered` tensor is the
  same obstacle from the arrangement side, which is worth noting because the per-dim spelling
  above makes the *fact* expressible without making the *propagation* work.
- **Rearrangement is not proposed.** A mismatch can be reported and cannot be fixed: the op that
  would convert one arrangement to another does not exist. So the diagnostic's advice is "write a
  conversion", and there is nothing to write it with. Whether that makes the check premature or
  makes the op urgent is the question.
- **Who performs the check.** The forward and backward layout analyses are the natural home —
  `PhysicalTypeAnalysis` propagates, `RequirementAnalysis` records what is wanted — but neither
  carries an arrangement today.

## Relationship to other proposals

- [`spyre-tensor-layouts.md`](spyre-tensor-layouts.md) — the mechanism. Its output decision and
  backward requirement analysis are what make an intermediate's arrangement derivable rather than
  something the author must state.
- **`tl.spyre_pin`** — memory space and scratchpad address. Both are properties of a *value* and
  its placement rather than of a descriptor's memory, and pinning a value avoids a typing problem
  this op cannot: a descriptor's element type is taken from its base pointer, so an address cannot
  be given to a descriptor without also giving it a dtype from somewhere.
- [`inter-tile-lowering-to-mem-view.md`](inter-tile-lowering-to-mem-view.md) — unaffected by this
  attribute. Its per-partition views carry a layout each, and a relayout does not change an
  arrangement.
