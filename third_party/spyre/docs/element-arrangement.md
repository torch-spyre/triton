# Element arrangement

Shape and dtype do not say where a value's elements sit inside a stick. Two independent
facts do, and conflating them is what makes the rules hard to state:

| axis | values | changed by | governs |
|---|---|---|---|
| **order** | `standard`, staggered-by-widening, staggered-by-narrowing | a precision conversion | ops that consult within-stick *position* |
| **multiplicity** | one value per stick, two (`EXX2`) | a fused partial reduction | what a consumer may *read* |

They are not two settings of one property. An order tag is an opaque permutation of a
stick's contents; a multiplicity says how many values the stick holds. "Arrangements must
match" means something different on each, which is why they are separated here even though
the underlying `ElementArrangement` enum holds both.

## The problem

Order is already wrong in a kernel that compiles and runs:

```python
tile     = in_desc.load([row_start, col_offset])
tile_f32 = tile.to(tl.float32)                      # within-stick order now scrambled
row_max  = tl.maximum(row_max, tl.max(tile_f32, axis=1, keep_dims=True))
denom    = denom * ... + tl.sum(tl.exp(tile_f32 - row_max), axis=1, keep_dims=True)
out_desc.store([...], softmax_out.to(tl.float16))   # order restored
```

Widening to fp32 does not redistribute elements across sticks — that costs a reshuffle the
hardware skips — so it leaves them out of order *within* the stick, and the closing
narrowing puts them back. Nothing records that `tile_f32` is in that state, so nothing
checks that the ops between tolerate it.

Where a value *lives* is `lx-placement.md`'s subject. This is how its elements sit.

## What the enum actually says

`staggered` is not an enum member. It is the [FP32 element-arrangement RFC][rfc]'s umbrella
term for the two conversion-produced orderings, which the enum separates by direction — a
widening and a narrowing scramble differently, and a narrowing undoes a widening. The enum
holds more, including quantization outputs the RFC puts out of scope.

Two properties of it drive everything below.

**The permutation is deliberately undescribed.** The RFC states only that it is *consistent*
per value and warns against depending on its shape. So an order tag is an opaque label, not
a layout that composes or inverts. A compiler can match tags; it cannot reason about the
underlying order, and therefore cannot synthesise a repair.

**The integer encoding is interleaved.** `EXX2` sits numerically between the two conversion
orderings, the later of which was appended. The encoding is explicit and append-only by
contract, so a frontend spelling must not renumber, and the umbrella term is a set
membership, never a range check.

## Assumptions

1. **A precision conversion is an op, and it changes order as a side effect.** There will
   be a `spyreop` for converting dtypes. That op is the only thing that changes a value's
   order mid-kernel, because changing precision is how the device restaggers. There is no
   order-only rearrangement primitive and none is proposed.
2. **Arrangement enters two ways and propagates one way.** A buffer the caller filled
   already has an order — that is a fact about memory, stated on its layout annotation. A
   fused reduction's `EXX2` output is a fact about an op, taken from the op's contract.
   Both then propagate forward as a property of the value, which is what lets the middle of
   a kernel be checked at all.
3. **An order requirement originates at a consuming op and is stated on the value.**
   Different ops impose different requirements and produce different results, so there is
   no global invariant like "all operands agree". Where the compiler models the op it knows
   the requirement; where it does not, the author states it on the value feeding it — which
   loses nothing, because a value cannot serve two consumers at two orders anyway. The only
   way to change order is a conversion, and that yields a different value.
4. **Applies to the innermost dimension only, for now.** Transpose a staggered value and
   the structure no longer aligns with the dimension it described. The narrow answer is
   taken: fix order to the innermost dimension and refuse a reshape or transpose of a
   non-`standard` value. By assumption 1 this is also the only implementable answer — such
   a value cannot be repaired without changing its dtype.
5. **A value has one arrangement per compute group.** The two readings of a fused pair —
   the pair, and one of its halves — happen in different groups, never at once.
6. **Blocked downstream: the dialect cannot record an arrangement.** The enum belongs to
   the tensor-layout type the surrounding stack uses. The dialect this fork lowers to has
   no arrangement attribute — nothing in its sources, the frontend submodule, or a built
   binding mentions one. A non-`standard` value reaching a memory view therefore gets a
   row-major view that misdescribes its element order, silently. The frontend checks below
   are worth having regardless, but nothing downstream can honour them yet.

## The proposal

Two pieces of surface and one policy.

### 1. Order enters on the layout annotation

A tensor arriving from global memory already has whatever order the caller wrote. That is a
property of the buffer, so it is stated where the buffer's other physical properties are:

```python
tl.spyre_tensor_layout(x_desc, TILE, arrangement="standard")     # default
```

A load from that descriptor yields a value carrying it. Nothing else needs a declaration:
every later order in the kernel is produced by a conversion, and every `EXX2` by the
reduction that emits it.

### 2. An order requirement is asserted on the value, reusing the pin

For ops the compiler models, the requirement is implicit — it is in the table below. Where
the compiler cannot know, because the consumer is an opaque intrinsic, the author asserts it
on the value, on the same marker `lx-placement.md` uses to say where a value lives:

```python
y = tl.exp(x)
tl.spyre_pin(y, requires_standard=True)          # assertion, not a request
z = tl.spyre_op(spyreop.something, y)            # the consumer that needs it
```

**It is an assertion, and that is what keeps §3's policy intact.** The compiler *checks*
that `y` is `standard` and refuses if it is not. It does not make it standard.

This is also why the hint does not ride on the conversion op, which is the other obvious
home for it. Those are two different kinds of thing. "Give me this in standard order" is a
*request*, and it is already spelled — it is the conversion of assumption 1, which the
author writes because it changes dtype and the compiler must not choose that for them.
"This value must be in standard order here" moves no data and changes only what is checked.
Tying an assertion to an operation would mean emitting a conversion in order to declare a
constraint.

Reusing the pin costs nothing that matters. A pin exists only where the author named the
value — but an author-written assertion is available only there either way, so the
restriction is not one. Note the pin carries the author's *requirement*, never the value's
own arrangement: that is in the type, propagated per assumption 2, and no annotation states
it.

### 3. The policy is reject, never insert

A mismatch is refused with a diagnostic naming the conversion that would avoid it. The
compiler does not insert one, and the reason is not that nothing exists — by assumption 1
something does. It is that the only available repair **also changes the dtype**, which is a
semantic change the author did not ask for. Silently narrowing a value to fix its order
would be worse than refusing.

### The per-op table

Order first. The only question is whether an op consults within-stick position:

| op class | requires of operands | order of result |
|---|---|---|
| unary pointwise | any | same as operand |
| binary pointwise | same order, **or** one operand size-1 at the stick dim | the non-`standard` one |
| reduction along the stick dim | any | `standard` |
| precision conversion | `standard`, or the staggering it undoes | set or cleared by direction |
| reshape, transpose | `standard` only (assumption 4) | `standard` |
| opaque intrinsic | as asserted | as asserted, else same as operand |

Two differing non-`standard` orders never combine. The size-1 exemption is what makes the
softmax fixture legal: it subtracts a `standard` `[BLOCK_M, 1]` from a staggered tile, and
the `standard` operand broadcasts, so there is no position to disagree about. A sameness
rule would reject a kernel that runs.

### Multiplicity: an op that returns a pair

The producing case is a fused partial reduction, and it is worth writing out because its
shape is easy to misread:

```python
pair = tl.spyre_op(spyreop.exx2_fused, x, axis=1)              # ONE value, two per stick
sc   = tl.spyre_op(spyreop.layernormscale_fused, pair)         # takes the pair whole

sq   = extract(pair, 1)                                        # one component, back to f16
out  = tl.spyre_op(spyreop.layernormnorm, x, sq, sc, w, b)     # takes the component
```

Both consumers are shown because the shape only makes sense with them: `layernormscale_fused`
takes the pair as it stands, while `layernormnorm`'s operand is an ordinary `f16`, so what it
receives is one of the two values rather than the pair. The extract exists to feed that
operand, not to decompose the pair for its own sake.

`pair` is **one result, not two.** Its element type is a fused pair and its multiplicity is
2 — which is precisely why multiplicity is an axis of arrangement rather than a question of
op arity. Nobody declares it; it comes from `exx2_fused`'s contract, per assumption 2.

The rule is correspondingly short. `EXX2` never participates in a pointwise op — it encodes
a reduction mode, not an ordering, so there is no "combining" case to specify. A consumer
either takes the pair whole or takes one of its two values, and nothing past the second.

Reading one component is an ordinary cast, value to value, so no aliasing declaration on a
buffer is needed. Assumption 5 is what permits that: the two readings sit in different
compute groups and are never live at once, so nothing has to be two things simultaneously.

## Where the checking lives

The generic layout pass, not the per-op pass it replaces — and it fits better. That older
pass resolves physical types in two halves, a forward type analysis and a backward
requirement analysis. Both are retired with it: they exist to answer *no* before any IR
moves, and every decline they can issue traces to a shape limitation of a *named* linalg
op. Once every compute is a `linalg.generic`, identity says nothing and the indexing maps
say everything.

Arrangement needs no new mechanism there. That pass's consistency predicate deliberately
tests **types**, never maps — maps are its own arithmetic, so testing them would be
circular. Assumption 2 makes arrangement part of a value's type. So an operand carrying an
order its rule does not admit is *already* inconsistent, the same way an op with one
physical and one logical operand is. The per-op requirements above live in the rebuild rule.

## The relation to `lx-placement.md`

The two designs share one op and nothing else, and the split between what the op carries
and what it does not is the whole of the relationship.

**A value's own arrangement is never on the pin.** It is in the value's type, entering from
a layout annotation or an op contract per assumption 2 and propagating from there. This is
not a stylistic choice: order matters everywhere, including where a value never reaches
memory — `tile_f32` above has no placement site and is the value whose order decides whether
the kernel is legal. A pin exists only where the author named the value, so most
intermediates could not carry one. Arrangement's domain strictly contains placement's.

**An author's *requirement* is on the pin**, per §2, and there the restriction is not one:
an assertion is available only where the author can point at the value anyway. So the pin
carries assertions about a named value — where it lives, and what order it must be in — and
those are two of a kind rather than two designs forced together.

**And placement must still record what it stores.** The intersection is the common case, not
an edge: one compute per local schedule means every compute-to-compute value goes through
memory, so a store has to record the order of what it wrote or the load on the far side
reads a stick it cannot interpret. It needs no new field for that — it already takes its
physical type from the producing compute's result type, and assumption 2 puts arrangement in
that type.

One consequence for the other document: its open question about what to call the pin now has
a wider subject than placement. The op is a marker for author assertions about a named
value, of which an address is one and an order requirement another.

Both designs remain blocked on the same region of the dialect — that one on the memory space
it can name, this one on the element order it can describe.

## Open: does anything non-`standard` reach memory the author owns

The layout annotation above admits a non-`standard` order on input, but it is not known
whether a real caller produces one. The RFC brackets fp32 between an upcast and a downcast
and lists a missing global closure check among its gaps, which suggests no for order.
`EXX2` is less clear: the verified layernorm target writes its fused pair to a global
buffer. The answer decides whether a store must agree on arrangement, or whether bracket
closure keeps every buffer `standard` by construction — and if the latter, the annotation
in §1 is a guard rather than a feature.

## Open: which dimension, when it is not the innermost

Is there a kernel that needs to transpose a non-`standard` value? None is known here. The
answer decides whether assumption 4 is a first step or the end state.

[rfc]: https://github.com/torch-spyre/RFCs/blob/main/2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md
