# Intermediates in the scratchpad

## The problem

The scheduler admits one compute per local schedule, so a value handed from one
compute to the next cannot stay in registers — it has to go through memory. Today
that memory is always off-chip: every producer of `ktdp.construct_memory_view` in
the tree hardcodes the global space. So an intermediate that never needed to leave
the chip pays two off-chip DMAs.

The scratchpad is on-chip and much closer, and nothing can currently ask for it.

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
4. **One address, the same on every core**, as a first step. This is the natural
   case rather than a simplification: the scheduler's address-assignment pass
   replaces each allocation with a literal constant in the single local-schedule
   body that every core executes, so one program text means one set of offsets.
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
tl.spyre_pin(e, ...)                          # name and arguments TBD, see below
out_desc.store([0], tl.sqrt(e))
```

The pin states no layout — the buffer is the compiler's, so there is nothing for
the author to agree with, and the physical type is the producing compute's result
type.

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

## Open: what the pin is for

This decides whether the op exists at all, so it comes before the questions about
how it is spelled.

If the compiler already places every intermediate, an override needs a reason.
Two candidates:

- **Coordination** — something outside the kernel expects a value at a known
  offset.
- **Disagreement** — the author believes the compiler's placement is wrong.

If neither turns out to be real, the pass alone is the whole proposal and the rest
of this document is unnecessary. That is a live possibility and worth settling
first.

## Open: the name

`tl.spyre_pin` is a placeholder. Alternatives to weigh, once the purpose above is
settled, since the purpose is what a good name should express — a coordination
marker and a placement override want different words.

## Open: what an address is

Three forms, in increasing difficulty. Only the first is proposed.

```python
addr = 917504                    # 1. one constant, the same on every core
addr = [917504, 918528, ...]     # 2. one per core
addr = f(tl.program_id(0))       # 3. derived at run time
```

Form 3 is representable — `ktdp.construct_memory_view` takes its offset as an SSA
`index` operand rather than an attribute — so the obstacle is not the IR. It is
that a runtime offset forfeits the disjointness check above, which is the thing
that makes pinning safe at all.

Form 2 needs an index space, and the natural candidate already exists:
`tl.inter_tile`'s `work_slices` is a `tl.constexpr` list of per-tile dicts, and
`wk_slice_coord` already folds a compile-time per-tile column into a runtime value
via a select chain on the program id. That is the mechanism form 2 and form 3 would
both want. Note it is a keyword argument of `tl.inter_tile` rather than a standalone
construct, so reusing the *structure* is not the same as routing addresses through
that op.

## Open: where an address comes from

The channel probably differs by form, and saying so is better than pretending one
covers all three.

For form 1, build configuration works, and there is precedent for the plumbing:
`Passes.td` already declares a `ListOption` of addresses fed from `SpyreOptions`
through the backend's pass-option path. Two things it would need — entry into the
options hash, so the cache key distinguishes two builds that differ only in a
pinned address; and keying by something other than position, because a pin site is
not a block argument, has no canonical list, and a `tl.constexpr` branch can change
the count between what the author sees and what the pass sees.

For form 3 a build-time value cannot express the address at all, and it may have to
arrive as a kernel argument. That is unresolved.
