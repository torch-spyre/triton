# Intermediates in the scratchpad

## The problem

The scheduler admits one compute per local schedule, so a value handed from one
compute to the next cannot stay in registers — it has to go through memory. Until
recently that memory was always off-chip: every producer of
`ktdp.construct_memory_view` in the tree hardcoded the global space, so an
intermediate that never needed to leave the chip paid two off-chip DMAs.

The scratchpad is on-chip and much closer.

Most intermediates are also anonymous. In

```python
out_desc.store([0], tl.add(tl.exp(x), y))
```

nothing names `tl.exp(x)`, and it still has to land somewhere. So this cannot be a
feature the author opts into — whatever places intermediates has to place them
whether or not the author can point at them.

## Assumptions

Six, and the fifth is the one the rest of the document rests on.

1. **Off-chip round trips are out of scope.** An author who wants an intermediate
   in HBM writes a `tl.make_tensor_descriptor` for it, with a
   `tl.spyre_tensor_layout`, and an explicit `store` and `load`. That reaches a
   binary and matches its oracle on hardware. This document is only about the
   scratchpad.
2. **The compiler places every intermediate, named or not.** This is the baseline,
   not a fallback for authors who write no annotation.
3. **A pin is an optional override**, available only where the author named the
   value. `y = tl.exp(x)` can be pinned; `tl.exp(x)` inside a larger expression
   cannot.
4. **A constant or an affine expression in `tl.program_id`, with `tl.constexpr`
   coefficients.** A constant is the same on every core, which is the natural case:
   the scheduler's address-assignment pass replaces each allocation with a literal
   constant in the single local-schedule body every core executes. An affine form
   `BASE + pid * STRIDE` gives each core a different offset within its own private
   scratchpad. It is the form that captures where a previous schedule left things:
   `STRIDE` equals the prior computation's tile size, so the address set is
   determined by that layout rather than chosen freely. Both forms keep the address
   set finite and enumerable — `{BASE + i*STRIDE : i < grid}` — so alignment,
   capacity and disjointness remain checkable. The disjointness check covers the
   schedules of one kernel (one artifact, one address-assignment run); two artifacts
   assign addresses from zero independently and are never co-resident.
5. **Unverified — the proposal is conditional on it.** Does scratchpad content
   outside the allocator's own pool survive from one local schedule to the next?
   Each schedule deliberately restarts its allocator at zero, on the stated
   grounds that a program has the memories it allocates from to itself. That
   governs allocations the allocator made; a pinned region is not one of them, so
   the reset does not discard it — but nor does it promise anything about it.
   Nobody here has established which. If the answer is no, no address channel
   helps and the value has to stay inside one schedule.
6. **The path is blocked downstream.** A hand-written `ct_local` memory view makes
   the scheduler abort — not diagnose — with `LLVM ERROR: ktdf.stage ... is
   missing required 'applicable_units' attribute`. The FIFO slot at the producer
   position keeps the raw memory-space attribute instead of resolving to a
   hardware load unit name, so the stage is assigned no units and a later pass
   fatals. Annotating the space in the memref result type as well fails
   identically, so it is not a spelling problem. Everything below is a design for
   when that unblocks.

## The proposal

### The pass

A pass that, for every compute-to-compute edge, inserts a scratchpad memory view,
a store after the producer and a load before each consumer — each group
constructing its own view and access tile, since nothing may be reachable from two
groups.

The author writes nothing:

```python
x = x_desc.load([0])
out_desc.store([0], tl.add(tl.exp(x), y))     # tl.exp(x) placed by the compiler
```

This is the whole of what is required to stop paying off-chip round trips for
on-chip values. Everything after this section is optional surface.

### The pin

Where the author has named a value, they may override the compiler's placement:

```python
e = tl.exp(x)
tl.spyre_pin(e, "ct_local", address=4096)     # constant
tl.spyre_pin(e, "ct_local", address=BASE + pid * STRIDE)  # affine in program_id
out_desc.store([0], tl.sqrt(e))
```

The pin states no layout — the buffer is the compiler's, so there is nothing for
the author to agree with, and the physical type is the producing compute's result
type.

The pin cannot become an attribute the way `tts.tensor_layout` does. A value pin's
only carrier is the op producing the value, and `convert-elementwise-to-linalg` and
`linalg-generalize-named-ops` replace that op, dropping a discardable attribute in
silence. The buffer must therefore be materialized while the marker is still
present.

### The disjointness check

**A precondition of the design, not a hardening step.** Wherever a pinned address
comes from, the build must check that every pinned range is disjoint from every
schedule's allocation pool. This is checkable after the fact, because the assigned
offsets are literal constants in the schedule bodies: read them out, compare, and
fail the build on an intersection.

The point is that this can only be done by the build, not by the author. An author
cannot know any schedule's high-water mark. Observed pools start at zero and run to
offsets like 512, 3072 and 98304, so a plausible-looking hand-picked number is
exactly the dangerous kind. Without the check, pinning is unsound whoever chose the
number.

### Composing pinned shares across cores

[inter-tile-lowering-to-mem-view.md](inter-tile-lowering-to-mem-view.md) proposes
`tl.make_distributed_descriptor`, which composes each core's share of a tensor into
one descriptor read at author-chosen offsets. Its first assumption is that a kernel's
**entry inputs live in global memory**, and its examples build the share from a
global load. But a scratchpad relayout is on-chip on both sides, and its lowering
emits one memory view per partition carrying a holder and a base address — so it
needs a scratchpad address per share, and nothing in that design supplies one. It
presupposes an allocation it cannot request.

The pin is what supplies it, and the shape that shows this most clearly is a kernel
with **no entry inputs at all**:

```python
@triton.jit
def relayout():                                        # no inputs
    share = ...                                        # produced on-chip
    tl.spyre_pin(share, "ct_local", address=BASE + pid * STRIDE)  # LX offset
    whole = tl.make_distributed_descriptor(share, work_slices=SRC, axes=[None, "n"])
    mine  = whole.load([0, my_offset])                 # my region under the new division
    out   = tl.exp(mine)                               # and on into the next compute
```

The empty signature is not a way around that assumption. It constrains what an entry
input may *be*, not that there must be one, so a kernel with none satisfies it
trivially. And it is the case that isolates the question: were the share loaded from
global memory, the address in play would be the global one and the scratchpad
question would not arise. With no inputs, the pin is the only possible source of the
address.

Nor is a zero-argument entry function an exotic shape for the backend — the
baked-address mode already produces one, replacing every pointer argument with a
constant because the scheduler requires it.

**One address, not one per core.** Every instance runs the same kernel; what varies
per partition is the coordinate set, which is the composition's business. So the pin
stays a scalar even here.

## What the pin is for

Composition is the answer, with `tl.make_distributed_descriptor` as the concrete
consumer. That op composes one `ktdp.construct_memory_view` per partition, each
needing a base address, and the pin is the only source for it. Note this is not an
override — the compiler has no prior placement to override, because the value has to
sit where the composition agrees it sits. Coordination and disagreement remain
without a concrete consumer.

## Open: does anything receive a scratchpad-resident tile

Whether a kernel ever needs to *receive* a scratchpad-resident tile rather than
produce one — if so, the pin is not sufficient, since an address would have to be
passed in. And what such a kernel returns: under the pull model each destination
writes into its own scratchpad, so possibly nothing crosses the signature in either
direction.
