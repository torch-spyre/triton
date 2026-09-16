# Pinning an intermediate to the scratchpad

## The problem

The Spyre dataflow scheduler admits one compute per local schedule. A kernel that computes
anything in more than one step therefore cannot hand a value from one compute to the next as an
SSA value — the value has to be written to memory by one schedule and read back by the next.
**Which** memory decides what that costs. Off-chip, it is a DMA out and a DMA back per
intermediate. On-chip, in the per-core scratchpad, it is neither.

Every chained kernel in the tree round-trips through off-chip memory today, and no kernel has
ever placed an intermediate in the scratchpad. This document proposes the surface that would let
one, and it is a narrow surface: **the author marks a value as living in the scratchpad, and the
compiler emits the round trip.** The author writes no store, no load, no descriptor and no
address.

That leaves one genuinely open question, which is most of what follows: an on-chip buffer needs a
scratchpad offset, and nothing in the pipeline currently produces one. The offset has to arrive
from somewhere, and this document argues it arrives from **build configuration** rather than from
kernel source, weighs the two channels available for that, and states the two soundness
conditions the whole approach rests on — one of which is unverified and would sink it.

**None of this works today, and the block is not in the frontend.** Nothing in the tree emits the
scratchpad memory space, and the scheduler hard-aborts when handed a hand-written scratchpad
view. The details and a checkable unblock criterion are below; the surface is specified anyway,
because the specification is what makes the block worth clearing.

Out of scope, and mentioned only so a reader does not go looking: a frontend surface for naming
intrinsics, and element arrangement within a stick. Neither bears on placement.

### Three names for one memory

Worth having up front, because a reader tracing this through diagnostics will meet all three. The
hardware calls the on-chip scratchpad **LX**. The KTDP memory-space attribute this branch pins
spells it **`ct_local`**, with an optional `ct_id` naming whose scratchpad is meant. The
scheduler's address-assignment pass groups allocations under **L1**. This document says LX for
the memory and `ct_local` when quoting IR. Another revision of the KTDP dialect renames the
attribute, so treat the spelling as versioned and the concept as not.

## The surface: a pin, not a round trip

The author's whole contribution is one marker on one value:

```python
tl.spyre_pin(t, "ct_local", name="t")
```

It says: *this value lives in the scratchpad.* It has no result, so the author does not rebind —
the same shape as the existing `tl.spyre_tensor_layout` marker, which is result-less and erased by
the lowering. Everything else is the compiler's: the memory view, the access tiles, the store that
ends one schedule, the load that begins the next, and the offset.

### Why the author does not write the round trip

This is the opposite of the choice the off-chip case makes, and the difference is worth stating
rather than leaving to be inferred, because it is the reason a pin is the right surface here and
would be the wrong one there.

**Off-chip needs no new surface, because a descriptor is already a complete carrier.**
`tl.make_tensor_descriptor` supplies a base address, the logical extents and the strides;
`tl.spyre_tensor_layout` supplies the physical layout on top of it. An author who wants an
intermediate in off-chip memory can therefore declare a pointer for it, describe it exactly as
they describe any input, and write the `store` and the `load` by hand — and that works. Every
part of the round trip has a spelling already.

**LX has no such carrier.** There is no pointer, so no descriptor; nothing outside the
compilation allocated the memory, so nothing supplies a base address; and there is no launcher
argument for one to arrive in. An author-written LX round trip would therefore require inventing
a scratchpad descriptor — a new op taking extents, a layout and an address with no pointer behind
it — and then requiring the author to use it correctly on both sides of a schedule boundary they
cannot see.

So the division is deliberate: **author-written LX round trips are deferred**, and not as an
oversight. If a scratchpad descriptor is ever wanted, it is wanted for a different reason — a
cross-core relayout, where the author genuinely does need to name regions — and it should be
designed for that reason. Nothing here forecloses it, and nothing here needs it.

The consequence is that the compiler owns more of the LX path than it owns of the off-chip one.
That is the cost, and it is where every difficulty below comes from.

### What the pin does not state

Two omissions that are load-bearing.

**No layout.** An off-chip descriptor's layout has to be stated because it must agree with a
buffer somebody else allocated. A pinned value's buffer is the compiler's, so there is nothing to
agree with: the physical type of the scratchpad buffer is the physical type the producing compute
already produced. That is the natural reading, and it is unverified — nothing has yet put a
physicalized tensor through a scratchpad round trip to see whether the resulting memref type is
accepted.

**No `ct_id`.** A pinned intermediate is in the executing core's own scratchpad, which is the
no-`ct_id` form. A kernel naming another core's scratchpad is performing a cross-core
communication, which is a different subject with a design of its own; the two uses of the same
attribute do not overlap.

A reader who wants the wider context: `declared-buffers.md` proposes a general marker for
schedule-boundary buffers, of which a pin would be one case, and works through how layouts and
indexing maps are derived from such a marker. This pin is specified standalone because the LX
question does not depend on any of that, and nothing below assumes it.

## A pin in the smallest chain

Two computes on one tile, no grid, is the smallest kernel with an intermediate at all, and it is
the shape that currently lowers all the way to a binary. This section shows what the author writes
and what the compiler would have to emit. **The first is hypothetical surface; the second is a
sketch of a target, not output anyone has seen.**

```python
@triton.jit
def scale_then_bias(x_ptr, out_ptr, BLOCK: tl.constexpr):
    x_desc   = tl.make_tensor_descriptor(x_ptr,   shape=[BLOCK], strides=[1], block_shape=[BLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[BLOCK], strides=[1], block_shape=[BLOCK])
    tl.spyre_tensor_layout(x_desc,   TILE)
    tl.spyre_tensor_layout(out_desc, TILE)

    t = x_desc.load([0]) * 2.0            # compute 1
    tl.spyre_pin(t, "ct_local", name="t") # t lives in the scratchpad
    out_desc.store([0], t + 1.0)          # compute 2
```

The kernel takes two pointers, not three. There is no allocation for `t` on the caller's side, no
launcher argument for it, and no DMA for it. The author has said one thing.

What the compiler emits is two local schedules — compute-group extraction puts each in its own
module — joined by one scratchpad region:

```mlir
// schedule 1 -- compute 1, landing t in the scratchpad
%z    = arith.constant 0 : index
%off  = arith.constant 917504 : index                     // <-- the whole question: who chose this?
%tv   = ktdp.construct_memory_view %off, sizes: [64, 64], strides: [64, 1] {
          coordinate_set = #whole,
          memory_space   = #ktdp.memory_space<ct_local>    // no ct_id: this core's own
        } : memref<64x64xf16>
%tt   = ktdp.construct_access_tile %tv[%z, %z] { ... } : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
ktdp.store %v1, %tt : tensor<64x64xf16>, <64x64xindex>

// schedule 2 -- compute 2, reading it back through its own view and tile
%z2   = arith.constant 0 : index
%off2 = arith.constant 917504 : index                     // the same number, in a different module
%tv2  = ktdp.construct_memory_view %off2, ... same attributes ... : memref<64x64xf16>
%tt2  = ktdp.construct_access_tile %tv2[%z2, %z2] { ... } : ...
%v2   = ktdp.load %tt2 : <64x64xindex> -> tensor<64x64xf16>
```

Three features of that sketch drive the rest of the document. The view carries no `ct_id`, so the
region is the executing core's own and nothing enumerates cores. The offset is a **literal
constant**, and it is duplicated rather than shared, because no operation may be reachable from
two compute groups. And the same number has to appear in **two separate modules** and denote the
same storage — which is the soundness question, addressed after the mechanism.

## What is blocked, and how to tell when it has unblocked

Two independent blocks sit between the surface above and the IR below it. They are in different
repositories and will clear separately, and only the second is hard.

**Nothing in the tree emits the scratchpad space.** No pass produces
`#ktdp.memory_space<ct_local>`. Three sites produce a `ktdp.construct_memory_view`, and all three
are global by construction: `LowerDescriptorMemory` and `LowerScalarLoad` each build a fresh view
with the global kind written into the call alongside `ct_id = -1`, and `RewriteDescriptorLayout`
rebuilds a view when it physicalizes a layout, copying the space off the view it replaces — so it
propagates faithfully and can only ever reproduce what the other two made. This half is small and
well understood, and it is not what is blocking anything.

**Fed a `ct_local` view, the scheduler hard-aborts.** Not a recoverable diagnostic:

```
LLVM ERROR: ktdf.stage ... is missing required 'applicable_units' attribute
```

The traced mechanism: the FIFO slot at the producer position retains the raw memory-space
attribute instead of resolving it to a hardware load-unit name; the stage is consequently assigned
no applicable units; and a later pass — the one that maps parallel loops onto hardware instances,
which unions `applicable_units` over a pipeline's immediate stages — treats the absence as a
violated upstream invariant and calls `report_fatal_error`. That contract is documented on the
applicable-units analysis itself, so the fatal is deliberate where it fires. The defect is
upstream of it, in whatever should have resolved the attribute to a unit name.

**It is not a spelling problem.** Annotating the memory space in the memref result type as well as
in the attribute dictionary fails identically.

**LX is supported piecewise, not end to end.** `ct_local` does appear in the scheduler's unit
tests for individual passes, which is why it is easy to conclude the space works. Those tests
never run the pass that aborts. Every verified end-to-end reference chain uses the global space.

### The unblock criterion

Checkable without reading any of the above:

1. A hand-written KTIR chain whose intermediate is a `ct_local` view with no `ct_id`, stored by
   one compute group and loaded by the next, reaches a `spyreCodeDir` instead of aborting.
2. The stage at the producer position carries an `applicable_units` list naming a load unit rather
   than a memory-space attribute.
3. That chain's numerical result matches the same chain written with a global intermediate.

Until (1), no frontend work here can be tested. That is why this document specifies a surface and
does not propose implementing it.

## Where the address comes from

This is the substantive question. A pinned buffer needs a scratchpad offset; the author does not
supply one; so some other channel must. This section rules out the channel the off-chip path uses,
weighs the two that remain, and picks one.

### Not a kernel argument — and that is a simplification

The off-chip path gets its addresses through the function signature: pointer arguments become
`index` arguments, and either the launcher fills them or `MaterializeBaseAddresses` replaces them
with constants. It is tempting to extend that to LX by appending an argument per pinned buffer.

Nothing needs it. An off-chip buffer's address is in the signature because something *outside* the
compilation — the caller, the launcher — has to agree about where the buffer is. An LX buffer has
no such counterparty: it exists only between two schedules of one kernel, and nobody outside ever
addresses it. So the signature stays exactly as the author wrote it.

This is a real simplification, and it is downstream of the earlier decision. Because off-chip
round trips are author-written through descriptors, the signature already carries every address the
runtime needs, and **the LX pin adds nothing to the function signature at all** — no argument, no
metadata entry for the launcher, no interaction with the symbolic-address path.

### Channel 1 — `SpyreOptions`, surfaced as a pass option

There is a direct precedent, and it is the same mechanism twice. The KTDP transforms `Passes.td`
declares `ListOption<"baseAddresses", "base-addresses", "int64_t", ...>` on the base-address
materialization pass and `ListOption<"grid", "grid", "int64_t", ...>` on the work-distribution
pass. Both are fed
from `SpyreOptions` through the backend's pass-option table, which maps a pass name to the option
fields it declares and forwards them as keyword arguments to that pass's Python binding. So "a
list of integers reaching a pass from build configuration" is established plumbing here, not a new
mechanism, and adding one more entry to that table is the cheapest change in this document.

Two costs, both worth stating before relying on it.

**The precedent is on its way out — for a reason that does not transfer.** `base_addresses` is
annotated as soon to be deprecated, along with baked addresses generally, because a baked
off-chip address is only correct if the runtime binds buffers to those segments at launch; once it
does not, addresses have to arrive symbolically and be patched through a correction table. That
argument is specifically about memory a *runtime* owns. An LX offset has no runtime counterparty to
disagree with it — it is a compile-time fact about a compiler-managed pool — so the deprecation
does not reach it. What is being cited as precedent is therefore the *plumbing*, which survives,
not the field, which does not.

**It has to enter the cache key.** An option that changes the emitted artifact must be in
`options.hash()`, as `symbolic_args` is and for the same stated reason. An LX address field is
exactly such an option: two builds of one kernel at different offsets are different binaries.

### Channel 2 — alongside `work_slices`

The other candidate is the per-tile keyed structure that already exists. `tl.inter_tile` takes
`work_slices`, a `tl.constexpr` list of per-tile slice-index dicts; the frontend validates that
every entry carries the same keys, serializes them as op attributes on `tt.inter_tile_reduce`, and
`LowerInterTile` expands them. `tl.wk_slice_coord(work_slices, axis)` reads a per-tile column back
out. If addresses ever have to vary per tile, this is the structure that already knows which tile
is which, and hanging them off it avoids inventing a second per-tile index space. That instinct is
right, and it is why the channel deserved checking rather than dismissing.

**It fails on the combination check, decisively.** `work_slices` is not a standalone construct: it
is a keyword argument of `tl.inter_tile`, validated there and carried on the op that call produces.
There is nowhere to put a `work_slices` list except on a cross-tile reduction. And LX without any
cross-tile reduction is not a corner case — it is the common one. The example above is single-tile
with no grid; the chained kernels that motivate this whole document use no `tl.inter_tile`; exactly
one fixture in the tree does. Routing LX addresses through `work_slices` would therefore require
every kernel wanting a scratchpad intermediate to declare a cross-tile reduction it does not
perform, which is a worse coupling than the duplication it avoids.

So: **rejected as the channel, retained as the model.** If addresses ever need to vary per tile,
`work_slices` is the shape to copy — and `wk_slice_coord` already demonstrates the whole mechanism,
folding a compile-time per-tile column into a runtime scalar as a select chain on the program id.
Variation is deferred; see below.

### The decision

**`SpyreOptions`, surfaced as a pass option.** It is existing plumbing, it keeps LX placement
independent of the inter-tile surface, and it puts the number in the hands of the party building
the kernel rather than the party writing it — which, as the soundness section argues, is the only
party that can be right about it. Its two costs are a cache-key entry and a precedent whose
deprecation does not apply.

## More than one pin

A real kernel pins several values, and matching N addresses to N pin sites is where a bare list
gets dangerous. This section is about identification, because identification is what makes the
matching safe or unsafe.

**Why positional matching works for off-chip addresses and would not work here.** The
`baseAddresses` option is documented as positional over the entry block's `index` arguments in
scan order, and that is safe for a specific reason: after `ConvertFunctions`, the `index` arguments
are exactly the arguments that were `!tt.ptr`, so the list MLIR itself maintains *is* the list of
things that can receive an address, and a runtime scalar is never a candidate. The anchor is typed
and structural.

A pin site has no such anchor. Markers are not block arguments, there is no canonical list of them,
and their number is not a property of the signature — a `tl.constexpr` branch can change how many
pins a trace produces, so the count the author sees and the count the pass sees need not agree.
Positional matching between a configuration list and source-order markers is exactly the pairing
that silently patches the wrong buffer.

**So a pin carries a name, and the channel is keyed by name.** The `name=` argument in the surface
above is required, not decorative. The configuration is a mapping, and it reaches the pass as two
parallel lists — one of names, one of addresses — which is the representation this tree already
uses for keyed attributes elsewhere: the inter-tile op carries its work-slice dimensions and values
as parallel key and value arrays, for the same reason, that a `ListOption` of `int64_t` cannot
carry strings.

Matching is then a lookup, and every mismatch is diagnosable:

| situation | outcome |
|---|---|
| pin named `t`, address supplied for `t` | matched |
| pin named `t`, no address for `t` | error naming the pin and its source location |
| address supplied for `s`, no pin named `s` | error naming the key, listing the pins that exist |
| two pins named `t` | error at the second, naming both |

None of these can patch the wrong buffer, which is the property positional matching cannot offer.
The cost is that the author writes a name — one string per intermediate, and a kernel with
intermediates worth pinning has few.

One question this does not settle: whether a name should be *required* or default to something
derived, and if derived, from what. A source location is stable enough to diagnose with and too
unstable to configure against. Deriving from the Python variable name is available at trace time
and quietly couples build configuration to the kernel's local variable names. Requiring the name is
the conservative choice and is what is specified.

## First step: one address, the same on every core

Everything above assumes a pinned buffer sits at one offset, identical on every core. That is not a
simplifying fiction adopted for convenience — it is what the evidence supports, and this section
says why, because the varying case in the next section has to be argued against it.

The scheduler's address-assignment pass collects allocations, groups them by memory space, and
replaces each with a **literal constant offset in the local-schedule function body**. There is one
program text and every core executes it. So the offsets are identical on every core *by
construction*, not as an observed coincidence that a later change could break. Offsets seen in
real schedules are multiples like 512, 3072 and 98304.

That is also where per-core variation genuinely lives, for contrast: on the global side, where
addresses are handles the launcher patches and per-core offsets come from tile-id arithmetic inside
the kernel body. Nothing on the scratchpad side works that way today.

A figure that has circulated as evidence for an author-written address field — two scratchpad
allocations at offsets `0` and `256` — should be treated as retracted. It does not reproduce: no
artifact in the tree records scratchpad offsets at all, every job-preparation plan carries a single
allocation for the job binary, and `256` appears only in the register file, not the scratchpad.
Nothing in this document depends on it, and any argument that does should be re-derived.

## Whether addresses vary

The uniform case is the proposal. Because a richer form is easy to add and hard to remove, it is
worth being explicit about what each would mean and cost. Three forms, in increasing order of what
they demand:

**A constant.** The proposal. One offset per pinned buffer, materialized as an `index` constant in
each schedule that touches the buffer — twice in the two-schedule example, because nothing may be
shared between compute groups. Fully checkable at build time, which the soundness section relies
on.

**A list of constants.** The immediate question is *indexed by what*, and every answer is either
already covered or not yet meaningful. Per pin is the name-keyed mapping above, not a list. Per
core is the wrong shape: uniformity across cores is a property of how the schedule body is emitted,
so a per-core list would describe variation the mechanism does not have. Per tile or per partition
is the only reading with content, and there is no per-tile index space in the frontend outside
`work_slices`. A list also amends an invariant the cross-core design relies on — that partition
views differ only in their coordinate set and holder, with offsets identical across them. So a list
should not be admitted before a kernel needs one; adding it early buys a positional trap and
nothing else.

**A Triton variable — an address derived at runtime from the program id.** The interesting form,
and the hardest, though not for the reason one would guess.

The frontend part is nearly free. `wk_slice_coord` already folds a compile-time per-tile column
into a runtime `i32` scalar as a chain of selects on `program_id(0)`, so a per-tile address table
would be built the same way with no new machinery. The KTDP part is free too, and this is worth
stating precisely: `ktdp.construct_memory_view` takes its offset as an SSA `Index` **operand**, not
an attribute, and the op's own description separates memory *interpretation* from memory
*allocation* — it explicitly imposes no constraint on how the memory was created or managed. A
dynamic offset is therefore representable, not a violation.

What would have to be true is downstream, and there are two conditions of quite different weight.
The mild one: the scheduler must accept a non-constant offset on a scratchpad view. There is
circumstantial evidence it prefers constants — the base-address materialization pass exists
precisely to turn address arguments into constants before the scheduler sees them, and the
address-assignment pass writes literal constants — but a preference is not a requirement, and
nothing has tested a dynamic scratchpad offset. The severe one: **a runtime address cannot be
checked for overlap at build time**, and that check is the mitigation that makes pinning sound at
all. This form does not merely add machinery; it forfeits the safety argument the uniform case
depends on. That is the reason to defer it, and it will still be the reason after the
representational questions are answered.

## Is a pinned address coherent at all?

Two objections stand against pinning regardless of which channel carries the number, and both are
about the scheduler's own allocator rather than about the surface. They are why this section exists
rather than the document ending at the grammar. One is answerable; one is not yet.

Moving the number from kernel source to build configuration changes something real: the party
choosing it can know things about the scheduler's allocations that a kernel author cannot — it runs
the compiler, it can read what the scheduler produced, and it can iterate. That is what makes the
first objection tractable. It does nothing for the second, and it would be dishonest to let the
relocation stand in for an answer.

### Objection 1 — the pin can collide with a compiler placement

The scratchpad is not an empty space waiting to be pinned into. The address-assignment pass is
already handing offsets out of it, per schedule, and the offsets it produces are low — 512 is one
of them. A pin naming an offset in that region has nothing telling the allocator the region is
taken, so the failure mode is a silent overlap between a pinned buffer and a compiler-placed one:
no diagnostic, wrong numbers, and a dependence on how many allocations that particular schedule
happened to need.

**This is answerable, and the answer is a check rather than a convention.** The requirement is
weaker than it first looks: a pinned region does not need the allocator to *reserve* it, it needs
to be **disjoint from every schedule's allocation pool**. Disjointness is verifiable after the
fact, because the assigned offsets are literal constants in the schedule bodies — so a check can
read every allocation's offset and size out of the scheduled module, compare them against the
pinned ranges, and fail the build on an intersection.

That converts the failure from silent to loud, and it is a capability the build-configuration
channel has and a kernel author does not: the author cannot know any schedule's high-water mark,
while the party running the build can look. A convention on top — pins from the top of the
scratchpad, allocations from the bottom, which is why the example's offset is high rather than 512
— makes intersections rare, but the convention is not the answer. **The check is a precondition of
the design, not a hardening step.** Without it, pinning is unsound no matter who chose the number.

### Objection 2 — the allocator resets, so an offset may not denote the same storage twice

The harder one, and it does not dissolve when the number moves. Each schedule restarts the
allocator at zero, deliberately: the model is that a program owns the memories it allocates from,
and what one puts there is discarded before the next runs. A pinned buffer is written by one
schedule and read by the next, so it needs the opposite property.

The objection narrows under examination but does not vanish. The reset governs **the allocator's
own pool** — allocations it made, whose contents it is entitled to treat as dead once the schedule
ends. A pinned region is not in that pool: it is a view over an offset the allocator never handed
out. So the reset does not *discard* a pinned region; it declines to make any promise about it.
That is a weaker claim than the one the objection first appears to make, and it is the reason the
approach is not simply dead.

But declining to promise is not the same as promising, and the question underneath is a fact about
the machine that nobody here has established: **does scratchpad content outside the allocator's
pool survive from one local schedule to the next?** If it does, the pin works, and objection 1's
check is what makes it safe. If it does not — if the scratchpad is cleared, reused or
re-initialized between schedules — then no address channel helps and the approach is dead in this
shape, because the value would have to stay inside one schedule.

**This is stated as a precondition rather than answered, because it is not ours to assert.** It is
a question for whoever owns the scheduler and the hardware model, it is cheap to ask, and it should
be asked before any of the above is built. The honest position is that this document specifies a
surface conditional on a fact it does not have, and names the fact.

One alternative shape is worth putting on the record while that question is open. The scheduler
*does* have a scratchpad-handoff facility, and it is between the **sibling stages** of one pipeline
— its double-buffering pass takes a scratchpad memref written by one sibling stage and read by
others, allocates two physical copies and rotates them. That is one level below the boundary in
question: it handles handoff inside a schedule, not compute-to-compute across schedules. Reaching
it would mean both computes in one schedule, which one-compute-per-schedule forbids. So if the
answer to the persistence question is no, the next question is whether the one-compute rule is a
property of the current scheduler or of the hardware — because that decides whether scratchpad
intermediates are reachable a different way rather than not reachable at all. (That reading of the
double-buffering pass comes from its stated contract, not from a measurement.)

## What this does not solve

The surface is the small part. Ordered by how likely each is to be mistaken for something that
falls out of the above.

- **It does not unblock the path.** The `applicable_units` fatal is in the scheduler and no
  frontend change reaches it. Everything here is contingent on that clearing, and on the
  persistence question above.
- **It does not size the win.** No measurement here compares a scratchpad round trip against an
  off-chip one, because no scratchpad round trip runs. The argument for LX is the memory hierarchy,
  not a number.
- **It does not manage capacity.** The scratchpad is finite and per-core. A kernel whose pinned
  buffers do not fit will fail at whatever capacity diagnostic the scheduler happens to raise, and
  nothing here decides which intermediates go off-chip instead. On a real kernel this is the wall
  an author hits first.
- **It does not reuse regions.** Pins make live ranges visible — one buffer written by one compute
  and read by later ones — but nothing allocates against them, so two intermediates whose lifetimes
  do not overlap still occupy two offsets.
- **It does not let the author name a region.** Deliberately: author-written scratchpad descriptors
  are deferred, so a kernel cannot express "read this buffer at that offset". A cross-core relayout
  needs exactly that and is not served by this proposal.
- **It says nothing about loops**, where a pinned buffer's live range is per-iteration and the
  interaction with double buffering stops being incidental.

## Open questions

Ordered by how much they would change.

- **Does scratchpad content outside the allocator's pool survive a schedule boundary?** The whole
  proposal is conditional on this. Cheapest to answer by asking rather than measuring, and it
  should be answered first.
- **Is one-compute-per-schedule a scheduler property or a hardware one?** Only matters if the
  previous answer is no, and then it matters entirely — it decides whether the sibling-stage shape
  is available as an alternative.
- **Where does the disjointness check live?** It needs the scheduler's assigned offsets, which
  exist only after the scheduler has run, so it is a post-scheduling verification rather than a
  frontend verifier. Whether those offsets can be read back out of the produced artifact reliably
  enough to fail a build on is unexamined.
- **Should a pin name be required or derived?** Required is specified. Deriving from the Python
  variable name is available at trace time and couples build configuration to local variable names;
  deriving from source location gives something diagnosable but not configurable.
- **Is the producing compute's physical type the right type for the scratchpad buffer?** Stated
  above as the natural reading and unverified. If it is not, the pin needs a layout after all, and
  the surface grows.
- **Does the scheduler accept a dynamic offset on a scratchpad view?** KTDP permits it; downstream
  is untested. Only matters once the varying case does.
