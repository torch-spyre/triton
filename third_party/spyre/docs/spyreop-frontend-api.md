# tl.spyre_op: A Generic Triton Frontend Interface for SpyreOps

This proposal introduces a single, string-dispatched Triton frontend entry
point,

```python
tl.spyre_op(op_name, *args, **attrs)
```

as the mechanism for exposing Spyre-only compute intrinsics — the SpyreOp
dialect ops that have no portable meaning in ordinary Triton — to kernel
authors. Operations that already have a natural Triton spelling (`tl.exp`,
`tl.sqrt`, arithmetic operators, ...) are unaffected: the compiler continues
to lower them to their SpyreOp equivalents internally, with no frontend
change. `tl.spyre_op` covers everything else: activation functions with no
existing Triton name, address-generation intrinsics, Element Arrangement
(EA) rearrangement, and composite fused operations such as `layernorm` that
must not leak their internal representation into kernel code.




## The problem

The SpyreOp dialect already has more ops than the frontend can name one at
a time without constant churn. Some of its ops overlap cleanly with
existing Triton semantics (`sqrt`, `exp`, `rsqrt`, integer add/mul,
division) and are already lowered to automatically. Others — activation
functions with no existing Triton spelling, address-generation intrinsics,
and fused multi-op sequences that depend on arrangement tricks — have
nothing to attach to. Minting a dedicated `@builtin` function for each of
these ties every new SpyreOp to a new frontend PR, and the set of these
ops is expected to keep growing as address-generation, data-layout, and
fused operations are added.

`tl.spyre_op` exists to give that growing set exactly one place to land,
without requiring a new Python entry point for each addition.

## What already works, unchanged

Where a portable, backend-agnostic Triton spelling already exists for the
underlying math, kernel authors keep using it, and the compiler decides
internally whether to fuse it into the corresponding SpyreOp. This
proposal does not touch this path.

| Triton source | TTIR | Spyre lowering |
|---|---|---|
| `tl.exp(x)` | `math.exp` | `spyreop.exp` (F16/F32 only) |
| `tl.sqrt(x)` | `math.sqrt` | `spyreop.sqrt` |
| `tl.rsqrt(x)` | `math.rsqrt` | `spyreop.rsqrt` |
| `x / y` | `arith.divf` | `spyreop.realdiv` |
| `x + y`, `x * y` (int32/int64, inside elementwise compute) | `arith.addi`/`arith.muli` | `spyreop.addi32toi32` / `addi64toi64` / `muli32toi32` |


The rule for what belongs in this table rather than behind `tl.spyre_op`:
**does this op already have, or naturally deserve, a spelling that makes
sense on every backend?** If yes, it stays here, regardless of how it
happens to be implemented on Spyre.

## The proposal

### 1. A single generic entry point

```python
@core.builtin
def spyre_op(op_name: tl.constexpr, *args, **attrs) -> tensor:
    ...
```

`op_name` is an ordinary Python string literal. Like any non-tensor
argument to a Triton function, it is a compile-time constant by
construction — Triton specializes on it the same way it specializes on a
`tl.constexpr` shape argument, so there is nothing to opt into and nothing
extra for the caller to annotate.

`*args` are the operation's tensor/scalar SSA operands. `**attrs` are its
compile-time parameters — the same role `base_address`/`stride` already
play on `Spyre_Idx32ToAddr`, or `beta`/`threshold` on `Spyre_Softplus`.

### 2. A registry, not a bare string match

`spyre_op` is not a raw dispatch on an unchecked string. Each supported
`op_name` has an entry in an internal table naming its operand arity and
dtypes, its required attributes and their types, and its result-type rule:

```python
_SPYRE_OP_REGISTRY: dict[str, SpyreOpSpec] = {
    "gelu":        SpyreOpSpec(arity=1, dtypes=(fp16)),
    "silu":        SpyreOpSpec(arity=1, dtypes=(fp16, fp32)),
    "reciprocal":  SpyreOpSpec(arity=1, dtypes=(fp16, fp32)),
    "softplus":    SpyreOpSpec(arity=1, dtypes=(fp16),
                                attrs={"beta": ScalarAttr(f32),
                                       "threshold": ScalarAttr(f32)}),
    "idx32toaddr": SpyreOpSpec(arity=1, dtypes=(i32,),
                                attrs={"base": ConstexprAttr(i32),
                                       "stride": ConstexprAttr(i32)}),
    "addi32toi32": SpyreOpSpec(arity=2, dtypes=(i32,)),
    "addi64toi64": SpyreOpSpec(arity=2, dtypes=(i64,)),
    "muli32toi32": SpyreOpSpec(arity=2, dtypes=(i32,)),
    "ea_reorder":  SpyreOpSpec(arity=1,
                                attrs={"target_ea": EnumAttr(
                                    "standard", "dl16_to_fp32",
                                    "fp32_to_dl16")}),
    "layernorm":   SpyreOpSpec(arity=3, dtypes=(fp16, fp32),
                                attrs={"eps": ScalarAttr(f32),
                                       "axis": ConstexprAttr(i32)}),
}
```

An unknown `op_name` raises at trace time with a suggestion against the
registry's keys, before any IR is emitted. A wrong arity, dtype, or missing
attribute raises with the same specificity a dedicated function's own
argument checking would give. Adding a new op is one registry row, plus the
dialect op itself if it doesn't already exist — never a new `@builtin`,
export, docstring location, or conversion pattern.

### 3. Lowering: direct emission, no TTIR staging op

A validated call traces straight to the `spyreop.*` op the registry names:

```mlir
%r = spyreop.gelu %x : tensor<...xf16>
```

No intermediate TTIR-level op is introduced. Every `spyre_op` entry already
has a concrete, fully-typed target — the `spyreop.*` op the SpyreOp dialect
defines for it, with its own ODS-generated arity, dtype constraints, and
attributes — and the registry's arity/dtype/attribute check *is* that
contract, checked once, at the one place (trace time) an author would see
a mistake soonest. A staging op such as `tt.spyre_intrinsic {kind = "..."}
(...)`, converted to `spyreop.*` by a table-driven conversion pattern, was
considered and rejected: it would restate the same contract a second time
without anything keeping the two copies in sync beyond code review, and add
a lowering pass whose only job is to unwrap something the frontend already
validated. Adding a new entry now touches only the Python registry and, if
it doesn't exist yet, the SpyreOp dialect — nothing on the TTIR/KTIR
conversion-pass side, because there is no conversion left to write.

This does mean a traced module can now contain `spyreop.*` ops before any
KTIR lowering pass has run — something no other path through the frontend
does today; `tl.sqrt` and friends only ever produce `math.sqrt`, and it's
`LowerSpyreOps.cpp`, a lowering pass, that later rewrites it to
`spyreop.sqrt`. That is acceptable here specifically because `tl.spyre_op`
is already Spyre-only — it does not exist on any other backend's build
(the same guard `tl.spyre_pin` and `tl.spyre_tensor_layout` already use),
so a module that calls it was never going to lower anywhere else. Direct
emission does not make an otherwise-portable module non-portable; it just
makes the non-portability visible one stage earlier, at the same point the
call itself already committed to it.

## Initial operation set

| `op_name` | Args | Attrs | Notes |
|---|---|---|---|
| `"gelu"` | `x` | — | F16/DF16 only |
| `"silu"` | `x` | — | F16/DF16/F32 |
| `"softplus"` | `x` | `beta`, `threshold` (`constexpr[f32]`) | F16/DF16 only |
| `"reciprocal"` | `x` | — | see below — deliberately never inferred from `1 / x` |
| `"idx32toaddr"` | `index` | `base`, `stride` (`constexpr[i32]`) | address-generation intrinsic |
| `"addi32toi32"` | `a, b` | — | explicit address-arithmetic add; see below |
| `"addi64toi64"` | `a, b` | — | 64-bit form |
| `"muli32toi32"` | `a, b` | — | explicit address-arithmetic multiply; see below |
| `"ea_reorder"` | `x` | `target_ea` (`constexpr` enum) | see *Element Arrangement operations* |
| `"layernorm"` | `x, weight, bias` | `eps`, `axis` | see *LayerNorm* |

**On `addi32toi32`/`muli32toi32` appearing here despite `+`/`*` already
being transparent above:** these are two different call sites for the same
underlying hardware op, not a duplication. Ordinary tensor `+`/`*` inside
elementwise compute is already covered by the transparent path — that path
deliberately fires only inside elementwise compute, precisely so it never
misclassifies hand-written index or address arithmetic elsewhere in a
kernel as something to fuse. That exclusion should stay. It does mean a
kernel author computing an address by hand (feeding `idx32toaddr`, for
example) has no transparent route to the hardware add/multiply there —
which is exactly the gap `tl.spyre_op("addi32toi32", a, b)` fills: an
explicit request for the intrinsic in a context the transparent path is
deliberately blind to.

## Reciprocal: explicit only, never inferred

`tl.spyre_op("reciprocal", x)` is the only way to reach the dedicated
hardware reciprocal instruction. `1 / x` continues to lower to
`spyreop.realdiv`, unconditionally, with **no** compiler pattern that
rewrites a `1.0`-numerator division into `reciprocal`.

This is a deliberate choice, not an oversight: `1 / x` and `reciprocal(x)`
are not guaranteed to be the same operation. A hardware reciprocal
instruction may trade numerical exactness for speed in a way ordinary
division does not. Silently substituting one for the other — inferring the
faster, less exact op from an author's use of the exact one, or vice versa
— removes a choice that belongs to whoever is writing the kernel. Requiring
an explicit call for one of the two options is the same shape as EA's own
"reject, never insert" policy: an operation with different guarantees is
never silently substituted for the one the author actually wrote.

## Element Arrangement: rearrangement is one generic operation

This proposal's role here is narrow: give EA's rearrangement concept
**a frontend surface**, not a design. The actual set of representation
states, which transitions between them are legal, and what op(s) a
transition ultimately lowers to are owned by `element-arrangement.md` and
whatever implementation work follows from it — this document does not
re-derive or override any of that. What it adds is the one thing
`element-arrangement.md` does not itself need to specify: how a kernel
author asks for a rearrangement from Triton source.

Whatever internal structure EA's own design eventually settles on,
changing a value's representation is, by that design's own policy,
explicit and kernel-author-driven — never something the compiler inserts
on its own. That is exactly the shape `tl.spyre_op` is for: a single,
parameterized, explicit request rather than one op name per
representation-changing transition, so that a new representation EA later
defines is a registry-enum addition, not a new frontend op:

```python
y = tl.spyre_op("ea_reorder", x, target_ea="standard")
```

Two different things are being named here, and they should not share a
word. `target_ea`'s values (`standard`, `dl16_to_fp32`, `fp32_to_dl16`,
...) name **representation states** — each one describes the physical
shape a value's representation currently has, borrowing the name of
whichever precision conversion produces that shape as a side effect. The
operation itself is not one of those states; it is the request to move a
value from whatever state it is currently typed as into the state named by
`target_ea`. Calling that operation `ea_convert` would collide with a
word EA already owns: **conversion**, in EA's own vocabulary, means the
dtype-changing precision-conversion op that changing representation rides
along with. This operation does not change dtype — it only changes how a
value's data is physically arranged — so naming it after what it does
rather than after "conversion" keeps the two apart: `ea_reorder`.

`target_ea` is validated at trace time against EA's own (append-only)
set of representation states. Lowering maps `("ea_reorder", target_ea)`
to whichever op the dialect eventually defines for that transition — this
proposal does not itself define that op, since EA's own design notes that
the target dialect has no arrangement attribute today; `ea_reorder` is the
frontend-facing name that op will attach to once it exists.

**What `ea_reorder` deliberately does not cover.** `EXX2` produces an
internal representation with no stable, frontend-facing name today — and
whether it ever gets one through `ea_reorder`'s `target_ea` enum, or is
handled by some other mechanism EA's own design settles on, is not this
proposal's call to make. What is this proposal's call: until such a name
exists and a kernel author has an actual reason to request it directly,
`EXX2`'s representation gets no `tl.spyre_op` entry of its own —
general-purpose (a bare `"ea_reorder"` target) or dedicated (a standalone
`"exx2_fused"`-style op). It stays entirely inside the `layernorm`
composite described next. A kernel author reasoning about `EXX2` directly,
rather than through a composite that hides it, is exactly the leak this
proposal exists to prevent. If a case for naming it directly ever appears,
that is the point to revisit this — not before.

## LayerNorm: one composite operation, not three primitives

```python
y = tl.spyre_op("layernorm", x, weight, bias, eps=1e-5, axis=-1)
```

`x` is the input tensor; `weight`/`bias` match the shape of the normalized
axis; `eps` is a scalar constant; `axis` defaults to the innermost
dimension, which is also the only dimension EA's arrangement machinery
currently covers. The result has the same shape and dtype as `x`. Nothing
about `EXX2` or an intermediate pair value appears in this signature.

### Why one operation, not `exx2`, `layernorm_scale`, `layernorm_norm`

The usual usability argument applies — three ops with a fixed required call
order and a pair-value hand-off between them are easy to misuse — but the
governing reason is narrower: **`EXX2`'s internal representation has no
stable, frontend-facing name.** Whatever `layernorm_scale` and
`layernorm_norm` pass between themselves today has nothing a kernel author
could name, assert, or convert on their own, and no `ea_reorder` target the
way an ordinary representation state does. If it cannot be named, it must
not be exposed — a kernel author with no way to reason about it in the
frontend cannot be handed a value that carries it.
`layernorm` as a single semantic operation is the consequence of that, not
a separate usability decision layered on top of it: the frontend presents
LayerNorm as one operation precisely *because* its internal pair
representation has nowhere else to go. Decomposition into `EXX2`,
`layernorm_scale`, `layernorm_norm`, or whatever future SpyreOps replace
them, happens entirely during lowering, behind a dedicated expansion pass —
not as a Python-level macro that would instantiate the pair value directly
in TTIR, where every generic pass downstream could see it.

### Limitation: this also hides data movement

Hiding the compute sequence has a cost the compute-only framing above does
not mention: it also hides *where the intermediate values live* between
the three internal steps. `lx-placement.md` gives kernel authors a way to
pin a value to on-chip scratchpad rather than pay two off-chip DMAs for an
intermediate that never needed to leave the chip. A hand-written
`exx2_fused` → `layernormscale_fused` → `layernormnorm` sequence could use
that mechanism directly on its own intermediates. An opaque `"layernorm"`
composite cannot — there is no intermediate value in the kernel author's
own IR to pin. Whatever placement decision gets made for the internal
`EXX2` pair and the scale intermediate is made entirely by the lowering
pass, with no lever exposed to the author who might know their kernel's
memory pressure better than a general-purpose pass would.

This is also, directly, EA's own second open question: whether the
verified layernorm target's fused pair genuinely needs to reach memory
packed, or whether "bracket closure" could keep every buffer standard (and,
by extension, on-chip) by construction. This proposal does not resolve
that question — it inherits it. Until it is resolved, the composite's
internal placement behavior is a lowering-pass implementation detail, and
should be documented as one rather than assumed to be optimal.

Two ways to soften this later, without giving up the opacity that makes the
composite safe to expose in the first place:

- A coarse placement **hint**, not a pin — e.g. `tl.spyre_op("layernorm",
  ..., scratch_hint=True)` — that asks the lowering pass to prefer on-chip
  placement for its internal intermediates without naming which value that
  applies to. This preserves `EXX2`'s invisibility while still giving the
  author one lever.
- Accepting the limitation for now, and revisiting only if a real
  performance problem is observed — the lowering pass can hardcode whatever
  placement the verified layernorm target already relies on, and this
  proposal takes no position on whether that placement is currently
  correct.

## Other composite candidates

A composite `tl.spyre_op` entry is warranted when there is something to
hide: a fixed multi-step protocol with an easy-to-violate call order, or an
internal representation (like `EXX2`) that has nowhere else to be exposed.
It is not warranted merely because several ops happen to run in sequence.

- **Softmax.** Its usual decomposition — row-max reduce, subtract, `exp`,
  sum-reduce, divide — is made entirely of ops already covered by the
  transparent path above. There is no hidden internal representation
  analogous to `EXX2` in ordinary softmax; the representation change a
  precision conversion introduces mid-kernel is already the exact case
  EA's own per-op table and pin-based assertions are built to handle. A
  composite `"softmax"` entry would add an abstraction with nothing behind
  it to hide. Any scheduling win across its sub-ops belongs in a backend
  fusion pass that recognizes the pattern structurally, not in a new
  registry entry.
- **Activation functions.** `gelu`, `silu`, `softplus` are each already a
  single dialect op — there is no sequence to hide, so each is a plain
  entry in the initial operation set above, not a composite.
- **Future EA-driven fused ops.** The same test applies to anything added
  later: does this operation need to hide an internal representation trick
  to keep the frontend consistent with what EA actually represents, or is
  it just several already-transparent ops placed next to each other?
  The former earns a composite, opaque-until-lowering entry, following the
  `layernorm` treatment above, including its placement-visibility
  limitation. The latter should stay decomposed.

## Open questions

1. **Placement visibility for composite ops.** `layernorm`'s internal
   `EXX2`/scale intermediates have no author-visible pin today. Whether a
   coarse `scratch_hint`-style lever is worth adding, or whether this
   should simply wait on EA's own open question about whether `EXX2`'s
   pair needs to reach memory packed at all, is unresolved here.
2. **`ea_reorder`'s eventual target op.** This proposal names the frontend
   entry point but does not define the KTIR-level op it should lower to,
   since none exists yet in the target dialect. The lowering table entry
   for `ea_reorder` is a placeholder until that op is designed.
3. **Whether a second consumer of an `EXX2`-like internal representation
   ever appears.** If one does, the position taken here — that such a
   representation gets no general-purpose `tl.spyre_op` entry — should be
   revisited rather than assumed to still hold.

## Related discussions

This proposal sits alongside several other in-flight designs rather than
inside any of them.

- **[`element-arrangement.md`](element-arrangement.md) — Element
  Arrangement.** Owns EA's representation-state model and its "reject,
  never insert" policy (both summarized, not restated, in *Background*
  below), and will own the KTIR-level op `ea_reorder` (*Element
  Arrangement: rearrangement is one generic operation*, above) eventually
  lowers to. This document adds only a frontend entry point for that op;
  it takes no position on EA's own open questions.
- **`LowerSpyreOps.cpp` — SpyreOp lowering.**
  The existing pass that rewrites transparent-path ops (`math.sqrt`, ...)
  to their `spyreop.*` equivalents after they trace to ordinary TTIR.
  `tl.spyre_op`'s direct-emission design (*Lowering*, below) deliberately
  bypasses this pass: a `spyre_op` call already names its `spyreop.*`
  target at trace time, so there is nothing left for a lowering pass to
  rewrite for it.