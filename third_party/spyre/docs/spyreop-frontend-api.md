# `@tl.spyre_intrinsic`: A Staged Frontend Interface for SpyreOps

This proposal introduces a frontend mechanism for exposing Spyre-only
compute intrinsics — the SpyreOp dialect ops that have no portable meaning
in ordinary Triton — to kernel authors, staged through a software-fallback
TTIR representation rather than emitted directly:

```python
spyre_gelu(x)       # a backend-authored wrapper, decorated with @tl.spyre_intrinsic
```

Operations that already have a natural Triton spelling (`tl.exp`,
`tl.sqrt`, arithmetic operators, ...) are unaffected: the compiler continues
to lower them to their SpyreOp equivalents internally, with no frontend
change. This mechanism covers everything else: activation functions with no
existing Triton name, address-generation intrinsics, and composite fused
operations such as `layernorm` that must not leak their internal
representation into kernel code.


## KTIR must stay backend-independent

KTIR is not a Spyre-specific representation. It is the common lowering
target `_make_ktir` produces for *any* backend built on it, and its
abstractions — `linalg`, `arith`, `math`, and friends — are defined with no
knowledge of Spyre at all. `spyreop.*` is exactly the opposite: a dialect
that exists only to name Spyre-hardware-specific operations, and it
belongs exclusively to the Spyre-specific half of the pipeline —
`_make_spyrecode` and the `LowerSpyreOps` pass that already runs there
today for `math.sqrt` and friends.

`_make_ktir`'s job, by construction, is to produce generic KTIR. Letting a
`spyreop.*` op appear in its output — the direct-emission design this
proposal previously recommended — breaks that separation: it puts a
Spyre-only op into a representation that is supposed to mean the same
thing regardless of backend, for any intrinsic that has no pre-existing
portable spelling. `tl.sqrt` happens not to raise this problem, because
`math.sqrt` is already a generic, backend-agnostic op in its own right.
`spyreop.gelu` does: there is no generic KTIR op that already means
"GELU," so a direct-emission design would have no choice but to put a
Spyre-specific op directly into `_make_ktir`'s otherwise-generic output.
That is an architectural layering violation, not a style preference, and
it is the reason this proposal now adopts a staged design: **every new
intrinsic needs a generic KTIR representation it can trace to that isn't
`spyreop.*` itself**, so that `_make_ktir`'s output stays Spyre-agnostic
regardless of which intrinsics a kernel happens to call.

A useful consequence of keeping `_make_ktir`'s output generic falls out of
this for free: because the artifact never contains `spyreop.*`, it also
happens to be exactly what `ktir_cpu` — the numerical interpreter the test
suite runs against, which has no `spyreop` support — needs in order to
execute it directly and verify the kernel's numerics on the CPU. That is a
real and valuable benefit of this design, but it is a consequence of
keeping KTIR backend-independent, not the reason for doing so: the
staging design would still be the right one even for a hypothetical
backend with no CPU interpreter at all, because the underlying problem —
a Spyre-only op leaking into a representation that is supposed to be
generic — has nothing to do with who else happens to consume that
representation.

## Staged intrinsics with a software fallback

```
                              ┌─ _make_ktir:       dissolve tts.spyre_op → plain fallback ops (cached, generic)
kernel API  →  tts.spyre_op  ─┤
                              └─ _make_spyrecode:  LowerSpyreOps (hint match) → spyreop.*
```

### 1. `@tl.spyre_intrinsic`: a decorator over a real fallback implementation

```python
@tl.spyre_intrinsic("gelu")
def spyre_gelu(x):
    return 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
```

The decorated function's body is not a stub or a docstring — it is a
working implementation of the op, expressed in ordinary Triton/TTIR
operations, and it is what actually runs when `ktir_cpu` executes the
op numerically. `tl.spyre_op(op_name, *args, **attrs)` still exists
underneath the decorator as the low-level builtin that records the
intrinsic's name and operands, but it is internal plumbing now, not a
user-facing entry point — a kernel author calls `spyre_gelu(x)`, never
`tl.spyre_op("gelu", x)` directly.

### 2. `tts.spyre_op`: the TTIR staging op

Tracing a call to a `@tl.spyre_intrinsic`-decorated function does not
inline the fallback body into the caller. It emits one TTIR op,
`tts.spyre_op`, carrying:

- a **hint** attribute naming the intrinsic (`"gelu"`), used later purely
  for matching, and
- a **nested region** holding the fallback body, traced under ordinary
  semantics — so the region by itself is just `linalg`/`arith`/`math`,
  with nothing Spyre-specific in it.

```mlir
%r = tts.spyre_op {hint = "gelu"} (%x) ({
  ^bb0(%arg: tensor<...xf32>):
    // fallback body: the traced form of spyre_gelu's Python source
    ...
    tts.yield %result : tensor<...xf32>
}) : (tensor<...xf32>) -> tensor<...xf32>
```

This single op is what makes the generic/Spyre-specific split work in
practice: the ops inside its region are plain `linalg`/`arith`/`math`,
nothing Spyre-specific, and `_make_spyrecode` has a stable, hint-tagged
anchor to pattern-match and replace with the real `spyreop.*` op. Making
`_make_ktir`'s own cached output equally generic takes one more step,
though — the staging op itself still needs to be gone from that artifact,
not just its contents — covered next. Nothing about an unknown or future
intrinsic needs its own TTIR op either way: one generic staging op serves
all of them, the same economy-of-surface-area the earlier direct-emission
design wanted, just moved one level later.

### 3. Dissolving `tts.spyre_op` before `_make_ktir` caches its output (open design item)

Keeping `_make_ktir`'s artifact free of `spyreop.*` is not enough on its
own: if `tts.spyre_op` itself survives into that cached artifact, the
artifact still contains a custom, newly-introduced op type that nothing
outside this proposal knows how to interpret — including `ktir_cpu`,
which would then need special-cased support for `tts.spyre_op`
specifically rather than genuinely consuming plain KTIR. That reintroduces,
one level up, the same problem staging exists to avoid.

So `_make_ktir`'s pipeline needs an explicit step — called
`DissolveSpyreOpStaging` here as a placeholder name, not a committed one
— that runs before the artifact is cached and replaces each remaining
`tts.spyre_op` with the contents of its own region: the region's ops are
spliced in directly where the staging op was, its `tts.yield` operand is
wired to the staging op's result, and the hint attribute and the
`tts.spyre_op`/`tts.yield` wrapper are both dropped entirely. After this
step, nothing in the cached artifact names `tts.spyre_op` at all — only
the plain `linalg`/`arith`/`math` ops the fallback body was written in,
which is what makes the artifact genuinely generic rather than merely
free of `spyreop.*`.

This step does not exist today, in any form, and this proposal does not
yet have a concrete design for it. The real `_make_ktir` pipeline
(`LowerDescriptorMemory`, `LowerScalarLoad`, `LowerTTSMarkers`,
`LowerComputeOps`, `LowerInterTile`, `ConvertFunctions`, `DistributeWork`,
`Canonicalizer`) has nothing that performs this dissolution today, so it
is recorded here as new, required work the staged design depends on — not
an implementation detail to fill in later without affecting the design.

It also raises an ordering question the staged design has to answer, not
just implement: `_make_spyrecode`'s `LowerSpyreOps` needs the hint to pick
the right `spyreop.*` op, but if `_make_ktir`'s own pipeline already
dissolves that hint away before caching, `_make_spyrecode` cannot simply
consume the cached artifact and expect the hint to still be there. This
proposal does not yet pick between the two ways to resolve that:

- `_make_spyrecode` operates on a form of the IR captured before
  dissolution runs, kept separate from the cached, `ktir_cpu`-facing
  artifact; or
- the dissolution step itself leaves behind a discardable, non-structural
  marker on the spliced-in ops that only `LowerSpyreOps` reads (and then
  removes), so the cached artifact stays clean for every other consumer
  while `LowerSpyreOps` can still recover which hint applied to which ops.

Either is workable, but the choice affects both passes' design, so it is
called out here as something this proposal still needs to settle, not
something implicit in "the artifact is generic and `LowerSpyreOps`
matches on the hint" as stated above.

### 4. `ExpandSpyreOps`: splitting composite intrinsics

A composite like `layernorm` doesn't lower to one `spyreop.*` op; it
expands, during lowering, into a short internal sequence
(`EXX2`, a scale op, a norm op — see *LayerNorm*, below). `ExpandSpyreOps`
runs between tracing and `LowerSpyreOps` and splits one
`tts.spyre_op {hint = "layernorm"}` into several smaller `tts.spyre_op`s,
each with its own hint and its own slice of the fallback body, so that
`LowerSpyreOps` can replace each sub-step independently rather than
needing a single pattern that understands the whole composite's internal
protocol at once.

### 5. `LowerSpyreOps`: hint-based matching, alongside its existing job

`LowerSpyreOps` already exists and already rewrites `math.sqrt` →
`spyreop.sqrt` by ordinary pattern matching, in `_make_spyrecode`. This
proposal extends it with a second matching mode: given a `tts.spyre_op`,
look up its hint, discard the fallback region, and emit the `spyreop.*`
op the hint names. The existing structural pattern matching
(`math.sqrt` → `spyreop.sqrt`, and friends) is unaffected and keeps
running in the same pass — this adds a second rule set, not a
replacement.

## Preserving the existing Triton path

`tl.exp`, `tl.sqrt`, `tl.rsqrt`, `tl.where`, and the arithmetic operators
keep working exactly as they do today, through their existing route,
unaffected by any of the above:

| Triton source | TTIR | Spyre lowering |
|---|---|---|
| `tl.exp(x)` | `math.exp` | `spyreop.exp` |
| `tl.sqrt(x)` | `math.sqrt` | `spyreop.sqrt` |
| `tl.rsqrt(x)` | `math.rsqrt` | `spyreop.rsqrt` |
| `x / y` | `arith.divf` | `spyreop.realdiv` (or `spyreop.reciprocal` when the numerator is a constant `1.0` — see *Reciprocal*, below) |
| `x + y`, `x * y` (int32/int64, inside elementwise compute) | `arith.addi`/`arith.muli` | `spyreop.addi32toi32` / `addi64toi64` / `muli32toi32` |
| `tl.where(cond, x, y)` | `arith.cmpf` + `arith.select` | `spyreop.compare` + `spyreop.select`, via pattern recognition (PR #215) — see below |

None of these ever produce a `tts.spyre_op` — they trace straight to
`math.*`/`arith.*`, which is already a valid, portable fallback in its own
right, and `LowerSpyreOps`'s existing structural matching handles them in
`_make_spyrecode` exactly as it does today. **`@tl.spyre_intrinsic` is not
a replacement for these APIs** and this proposal does not recommend
migrating them onto it. The staged path exists for the cases the
transparent path cannot cover:

- **Spyre-only operations** with no portable meaning at all (`addi32toi32`,
  `idx32toaddr`).
- **Operations with no standard Triton spelling**, even where the
  underlying math could in principle be written out by hand (`gelu`,
  `reciprocal`).
- **Composite operations** that must not leak an internal, unnamed
  representation into kernel code (`layernorm`).

The rule for which bucket an op falls into is that if it
already has, or naturally deserves, a spelling that makes sense on every
backend, it stays on the transparent path above, regardless of how it
happens to be implemented on Spyre.

## Who writes `@tl.spyre_intrinsic` functions?

**Spyre backend/compiler developers, not kernel authors.** A fallback
body is not an arbitrary convenience implementation — it is what
`ktir_cpu` treats as ground truth for the op during `_make_ktir`, so it
must be numerically faithful to the real `spyreop.*` op's defined
semantics. Getting that right requires exactly the dialect-level knowledge
(what the op computes, what its edge cases are) that only someone
implementing or maintaining the SpyreOp dialect reliably has. These
functions are shipped as part of Spyre's own frontend library, in the same
spirit the current registry's arity/dtype table was backend-maintained
rather than something each kernel wrote per-call:

```python
# illustrative — exact module path not fixed by this proposal
# triton/language/extra/spyre/intrinsics.py

import triton.language as tl

@tl.spyre_intrinsic("gelu")
def spyre_gelu(x):
    ...

@tl.spyre_intrinsic("layernorm")
def spyre_layernorm(x, weight, bias, eps: tl.constexpr, axis: tl.constexpr):
    ...
```

exported as, illustratively, `triton.language.extra.spyre`, providing
`spyre_gelu`, `spyre_layernorm`, and the rest of the initial operation set
below. Nothing in the decorator mechanism technically prevents a kernel
author from writing their own `@tl.spyre_intrinsic`-decorated function,
but doing so correctly requires knowing the exact hint string
`LowerSpyreOps` matches on and the real op's semantics — implementation-
internal knowledge, in the same way hand-constructing `spyreop.*` TTIR
today is possible but not intended. The expectation is a closed,
backend-maintained set, not an open one.

## What kernel authors actually write

A kernel author imports the pre-built wrappers and calls them like any
other `tl.*` function — no decorator, no hint string, no awareness that a
staging op exists underneath:

```python
import triton
import triton.language as tl
from triton.language.extra import spyre   # illustrative import

@triton.jit
def my_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N)

    y1 = tl.exp(x)             # existing Triton spelling — untouched
    y2 = spyre.spyre_gelu(x)   # Spyre-only op, no tl.gelu exists

    tl.store(out_ptr + offs, y1 + y2, mask=offs < N)
```

`tl.exp(x)` traces to `math.exp`, which `LowerSpyreOps`'s existing
structural matching turns into `spyreop.exp` in `_make_spyrecode` — no
`tts.spyre_op` involved at any point. `spyre.spyre_gelu(x)` traces to
`tts.spyre_op {hint = "gelu"}` wrapping the fallback body. `_make_ktir`'s
pipeline dissolves that staging op before caching its artifact (see
*Dissolving `tts.spyre_op`*, above), so what `ktir_cpu` actually runs is
the fallback body's own ops, spliced in directly with no `tts.spyre_op`
wrapper left behind. `_make_spyrecode`'s extended `LowerSpyreOps`
instead replaces the whole staging op — hint, region, and all — with
`spyreop.gelu`, ahead of whatever point dissolution would otherwise erase
the hint it needs. Both calls end up fully lowered by the time
`_make_spyrecode` is done; the difference is invisible from the kernel
author's side and only matters to how the compiler gets there.

## Initial operation set

| Frontend API | Fallback | Generated `tts.spyre_op` | Final lowering | Composite/Simple | Notes |
|---|---|---|---|---|---|
| `spyre_gelu(x)` | GELU approximation, standard Triton ops (`tanh`, arithmetic) | `tts.spyre_op<"gelu">` | `spyreop.gelu` | Simple | F16/DF16 only |
| `spyre_silu(x)` | `x * sigmoid(x)`, standard ops | `tts.spyre_op<"silu">` | `spyreop.silu` | Simple | F16/DF16/F32 |
| `spyre_softplus(x, beta, threshold)` | `log1p(exp(beta * x)) / beta`, with the linear fallback above `threshold`, standard ops | `tts.spyre_op<"softplus">` | `spyreop.softplus` | Simple | F16/DF16 only |
| `spyre_reciprocal(x)` | `1.0 / x`, ordinary division | `tts.spyre_op<"reciprocal">` | `spyreop.reciprocal` | Simple | fallback is **exact**; real op trades exactness for speed — see below |
| `spyre_idx32toaddr(index, base, stride)` | `base + stride * index`, ordinary integer arithmetic | `tts.spyre_op<"idx32toaddr">` | `spyreop.idx32toaddr` | Simple | address-generation intrinsic |
| `spyre_addi32toi32(a, b)` / `addi64toi64` | ordinary `a + b` | `tts.spyre_op<"addi32toi32">` (etc.) | `spyreop.addi32toi32` (etc.) | Simple | explicit address-arithmetic add; see below |
| `spyre_muli32toi32(a, b)` | ordinary `a * b` | `tts.spyre_op<"muli32toi32">` | `spyreop.muli32toi32` | Simple | explicit address-arithmetic multiply; see below |
| `spyre_layernorm(x, weight, bias, eps, axis)` | expands via `ExpandSpyreOps`; see *LayerNorm* | `tts.spyre_op<"layernorm">` → split into per-step `tts.spyre_op`s | `EXX2` + scale + norm sequence | Composite | see *LayerNorm* |

`tl.where` is deliberately absent from this table: compare/select is not a
`@tl.spyre_intrinsic` and produces no `tts.spyre_op` at all. It stays on
the standard Triton API surface — see *Preserving the existing Triton
path*, below, and the dedicated discussion after this table.

**On `spyre_addi32toi32`/`spyre_muli32toi32` appearing here despite `+`/`*`
already being transparent above:** these are two different call sites for
the same underlying hardware op, not a duplication. Ordinary tensor `+`/`*`
inside elementwise compute is already covered by the transparent path —
that path deliberately fires only inside elementwise compute, precisely so
it never misclassifies hand-written index or address arithmetic elsewhere
in a kernel as something to fuse. That exclusion should stay. It does mean
a kernel author computing an address by hand (feeding `idx32toaddr`, for
example) has no transparent route to the hardware add/multiply there —
which is exactly the gap `spyre_addi32toi32(a, b)` fills: an explicit
request for the intrinsic in a context the transparent path is
deliberately blind to.

**`tl.where`: standard API, Spyre-specific lowering underneath.** A kernel
author writes ordinary `tl.where(cond, x, y)` — no intrinsic, no hint, no
import beyond standard Triton — exactly as they would on any other
backend:

```
tl.where(cond, x, y)
  →  arith.cmpf + arith.select     (ordinary TTIR, unchanged)
  →  compare/select pattern recognition
  →  spyreop.compare + spyreop.select
```

A compiler pass structurally recognizes the resulting `cmpf`/`select` pair
and rewrites it directly to `spyreop.compare` + `spyreop.select`, as
described in PR #215. This decomposition is similar in spirit to
`layernorm`'s: one frontend-visible operation lowers to more than one
`spyreop.*` op. The difference is where that operation lives. `layernorm`
has no portable Triton spelling at all, so it needs a dedicated
`@tl.spyre_intrinsic` wrapper and the staged `tts.spyre_op`/
`ExpandSpyreOps` machinery described above. `tl.where` already has a
portable, backend-agnostic Triton spelling — it belongs on the transparent
path in *Preserving the existing Triton path*, not in the table above, and
the multi-op decomposition happens entirely inside pattern-based lowering,
with nothing staged and nothing added to the frontend surface.


## Reciprocal: an existing automatic rewrite, plus an explicit intrinsic

`LowerSpyreOps` already contains a rewrite, independent of this proposal,
that recognizes a constant-`1.0` numerator in `arith.divf` and rewrites it
to `spyreop.reciprocal` rather than `spyreop.realdiv`, unconditionally —
this is existing `_make_spyrecode` behavior today, not something this
proposal introduces or should describe as absent:

```cpp
// arith.divf has two targets rather than one: a numerator of constant 1
// becomes the unary spyreop.reciprocal and everything else the binary
// spyreop.realdiv.
```

So `1 / x`, written literally, already lowers to `spyreop.reciprocal`
today, with no `@tl.spyre_intrinsic` involved at all.

That existing rewrite covers the common case where a kernel author
happens to write division by the literal constant one. It is not a
substitute for keeping an explicit `spyre_reciprocal(x)` intrinsic in the
initial operation set, for two reasons:

- **Not every call site is reachable as a literal `1.0`-numerator
  division.** Wherever the numerator is an expression rather than the
  literal constant `1.0` — even one provably equal to one by other means
  — the existing pattern match doesn't fire, and the kernel falls back to
  `spyreop.realdiv` whether or not the author actually wanted the
  dedicated reciprocal instruction.
- **An explicit call states intent directly**, independent of how the
  division happens to be spelled, giving `LowerSpyreOps` an unambiguous
  signal rather than inferring intent from a specific numerator shape.

The two paths are expected to coexist: `1 / x` continues to be caught by
the existing `arith.divf` rewrite exactly as it is today, and
`spyre_reciprocal(x)` remains available as a direct, staged intrinsic
(fallback: ordinary division, for `ktir_cpu`) for a kernel author who
wants to request the dedicated instruction explicitly rather than relying
on how a division happens to be written. Both are expected to compile to
the same `spyreop.reciprocal` op; this proposal does not change the
existing `arith.divf` → `spyreop.reciprocal` rewrite, and does not propose
removing it in favor of the explicit intrinsic.

The numeric-divergence note from the original framing still applies to
the explicit intrinsic's fallback, unchanged by this revision: a hardware
reciprocal instruction may trade numerical exactness for speed in a way
ordinary division does not, so `spyre_reciprocal`'s own software fallback
(ordinary division, for `ktir_cpu`'s purposes) is **not** necessarily a
numerically exact stand-in for what `spyreop.reciprocal` actually computes
on real hardware — the two are expected to diverge slightly, by design.
What changes here is only the claim that `1 / x` never reaches
`reciprocal` without going through the intrinsic; it already does.

## LayerNorm: one composite operation, not three primitives

```python
y = spyre_layernorm(x, weight, bias, eps=1e-5, axis=-1)
```

`x` is the input tensor; `weight`/`bias` match the shape of the normalized
axis; `eps` is a scalar constant; `axis` defaults to the innermost
dimension. The result has the same shape and dtype as `x`. Nothing about
`EXX2` or an intermediate pair value appears in this signature.

Staged, this traces to one `tts.spyre_op {hint = "layernorm"}`.
`ExpandSpyreOps` then splits it into a short sequence of smaller
`tts.spyre_op`s — one hinted `exx2_fused`, one `layernormscale_fused`, one
`layernormnorm` — each carrying its own slice of the fallback body, so
`_make_ktir`'s artifact still contains nothing but portable KTIR across
all of them. `LowerSpyreOps` then replaces each of those independently
with its real op during `_make_spyrecode`:

```
spyre_layernorm(...)
  → tts.spyre_op<"layernorm">
  → ExpandSpyreOps
  → tts.spyre_op<"exx2_fused">, tts.spyre_op<"layernormscale_fused">, tts.spyre_op<"layernormnorm">
  → LowerSpyreOps
  → EXX2, layernorm_scale, layernorm_norm (spyreop.*)
```

### Why one operation, not `exx2`, `layernorm_scale`, `layernorm_norm`

The usual usability argument applies — three ops with a fixed required call
order and a pair-value hand-off between them are easy to misuse — but the
governing reason is narrower: **`EXX2`'s internal representation has no
stable, frontend-facing name.** Whatever `layernormscale_fused` and
`layernormnorm` pass between themselves has nothing a kernel author could
name, assert, or convert on their own. If it cannot be named, it must not
be exposed — a kernel author with no way to reason about it in the
frontend cannot be handed a value that carries it. `spyre_layernorm` as a
single frontend intrinsic is the consequence of that, not a separate
usability decision layered on top of it: the expansion into `EXX2` and its
companions happens entirely inside `ExpandSpyreOps` and `LowerSpyreOps`,
never as something the kernel author's own traced IR exposes.

### Limitation: this also hides data movement

Hiding the compute sequence has a cost the compute-only framing above does
not mention: it also hides *where the intermediate values live* between
the internal steps. `lx-placement.md` gives kernel authors a way to pin a
value to on-chip scratchpad rather than pay two off-chip DMAs for an
intermediate that never needed to leave the chip. A hand-written
`exx2_fused` → `layernormscale_fused` → `layernormnorm` sequence could use
that mechanism directly on its own intermediates — and under the staged
model, those intermediates do exist as real values between
`ExpandSpyreOps`'s output ops, just not ones the kernel author's own
source ever names. Whatever placement decision gets made for the internal
`EXX2` pair and the scale intermediate is made entirely by the lowering
passes, with no lever exposed to the author who might know their kernel's
memory pressure better than a general-purpose pass would.

This is also an open question in its own right: whether the fused pair
genuinely needs to reach memory packed, or whether some other
representational discipline could keep every buffer on-chip by
construction. This proposal does not resolve that question — it inherits
it. Until it is resolved, the composite's internal placement behavior is a
lowering-pass implementation detail, and should be documented as one
rather than assumed to be optimal.

Two ways to soften this later, without giving up the opacity that makes the
composite safe to expose in the first place:

- A coarse placement **hint**, not a pin — e.g. `spyre_layernorm(...,
  scratch_hint=True)` — that asks the lowering passes to prefer on-chip
  placement for their internal intermediates without naming which value
  that applies to. This preserves `EXX2`'s invisibility while still giving
  the author one lever.
- Accepting the limitation for now, and revisiting only if a real
  performance problem is observed — the lowering passes can hardcode
  whatever placement the verified layernorm target already relies on, and
  this proposal takes no position on whether that placement is currently
  correct.

## Other composite candidates

A composite intrinsic is warranted when there is something to hide: a
fixed multi-step protocol with an easy-to-violate call order, or an
internal representation (like `EXX2`) that has nowhere else to be exposed.
It is not warranted merely because several ops happen to run in sequence.

- **Softmax.** Its usual decomposition — row-max reduce, subtract, `exp`,
  sum-reduce, divide — is made entirely of ops already covered by the
  transparent path above. There is no hidden internal representation
  analogous to `EXX2` in ordinary softmax, so there is nothing a composite
  entry would need to hide. A composite `spyre_softmax` would add an
  abstraction with nothing behind it to hide. Any scheduling win across its
  sub-ops belongs in a backend fusion pass that recognizes the pattern
  structurally — the same kind of pass that already turns `tl.where` into
  `spyreop.compare` + `spyreop.select` — not in a new intrinsic.
- **Activation functions.** `gelu`, `silu`, `softplus` are each already a
  single dialect op — there is no sequence to hide, so each is a plain
  entry in the initial operation set above, not a composite.
- **Future fused ops.** The same test applies to anything added later:
  does this operation need to hide an internal representation trick to
  keep the frontend from leaking something backend-internal, or is it
  just several already-transparent ops placed next to each other? The
  former earns a composite, `ExpandSpyreOps`-driven entry, following the
  `layernorm` treatment above, including its placement-visibility
  limitation. The latter should stay decomposed.

## Open question: a frontend-visible DF16 type

Separately, Spyre is considering introducing a frontend-visible,
Spyre-native 16-bit datatype (`DF16`) rather than treating everything as
`FP16`. This proposal does not resolve how that interacts with the staged
intrinsic design, but it raises several questions worth naming now:

- Should kernel authors be able to pass a `DF16` tensor into a
  `@tl.spyre_intrinsic`-wrapped call directly, or only after an explicit
  conversion to a type the fallback already handles?
- A fallback body is ordinary Triton source. If `DF16` isn't a type most
  standard Triton ops (the ones a fallback body is built from) know how to
  operate on natively, does every fallback that needs to support `DF16`
  acquire a type-specific branch, or does `DF16` route through a
  conversion before reaching the fallback at all — and if so, is that
  conversion itself expressible without a `spyreop.*` op appearing in
  `_make_ktir`'s artifact?
- Does `ktir_cpu` need its own `DF16` numerics to execute a fallback body
  faithfully, or does the fallback's `DF16` handling necessarily degrade to
  an approximation (e.g. widen to `FP32`, compute, narrow back) purely for
  the software path, distinct from what the real hardware op does?
- How does `DF16` interact with the dtype checking a `@tl.spyre_intrinsic`
  function's own signature currently provides — does it need to be named
  explicitly in every relevant wrapper's type annotations, or does it need
  a broader frontend-level typing mechanism this proposal doesn't yet
  have?

None of these are resolved here; they are recorded as follow-up design
work the staged model will need to account for once `DF16` is real.

## Open question: long-term SpyreOp granularity

This proposal assumes today's SpyreOp dialect shape: a handful of large,
dedicated ops (`gelu`, `layernorm`, `softplus`, ...) each covering
substantial functionality. A future redesign might instead express more of
that functionality as compositions of smaller primitives (`exp`, `log`,
`add`, `mul`, `rsqrt`, ...), closer to how `softmax` is already handled on
the transparent path.

- **Potential advantage:** fewer dedicated dialect ops to design, verify,
  and maintain; more reuse of primitives that already have fallbacks and
  lowering.
- **Potential drawback:** some of today's dedicated ops exist specifically
  because their internal representation (`EXX2`) must not be expressed in
  terms of primitives a kernel author's own fallback body could end up
  constructing by accident — granularity and representation-hiding pull in
  opposite directions, and that tension doesn't disappear just because the
  primitives are smaller.
- **Impact on `@tl.spyre_intrinsic`:** a wrapper's fallback body is written
  against today's dialect shape implicitly, by being numerically faithful
  to today's op. If the backend later re-expresses `gelu` as a composition
  of primitives, the existing `spyre_gelu` wrapper and its hint keep
  working unchanged as long as `LowerSpyreOps` is updated to emit the new
  composition instead of the old single op — the frontend-facing contract
  (name, arity, fallback) does not need to change for this to happen
  underneath it.
- **Open question:** should that frontend-facing stability be treated as a
  guarantee of this design going forward, or only as something that
  happens to hold today? This proposal does not take a position, but notes
  that the hint-based matching in `LowerSpyreOps` is exactly what would
  need to be re-pointed, not anything in the frontend's own surface.

## Summary of open questions

- **Dissolving `tts.spyre_op` before `_make_ktir` caches its output.** No
  pass does this today, and this proposal does not yet have a concrete
  design for it, nor a resolution to the ordering question it raises
  against `_make_spyrecode`'s hint-based matching — see *Dissolving
  `tts.spyre_op` before `_make_ktir` caches its output*, above.
- **DF16 frontend support.** How a Spyre-native 16-bit type interacts with
  intrinsic signatures, fallback bodies, and `ktir_cpu`'s numerics —
  unresolved, see above.
- **Long-term SpyreOp granularity.** Whether today's large, dedicated
  SpyreOps remain the right shape, or get decomposed into smaller
  primitives later, and what that implies for frontend stability — see
  above.
- **Composite-op placement visibility.** `layernorm`'s internal `EXX2`/
  scale intermediates have no author-visible pin today. Whether a coarse
  `scratch_hint`-style lever is worth adding, or whether this should wait
  on a resolution to the packed-vs-on-chip question for the fused pair, is
  unresolved — see *LayerNorm*, above.
- **Ownership of frontend intrinsic libraries.** This proposal expects
  `@tl.spyre_intrinsic` wrappers to be backend-maintained and shipped as a
  library kernel authors import, not something they write themselves — but
  the exact module location, versioning, and review process for that
  library is not specified here and needs an owner.
- **Whether a second consumer of an `EXX2`-like internal representation
  ever appears.** If one does, the position taken here — that such a
  representation gets no general-purpose intrinsic of its own, only the
  `layernorm` composite — should be revisited rather than assumed to still
  hold.

## Related discussions

- **PR #212 — `@tl.spyre_intrinsic` and the staged design.** This document
  adopts the design direction from that PR's "Implementation Think-through"
  discussion in full: the KTIR-backend-independence motivation, the
  `tts.spyre_op` staging op, `ExpandSpyreOps`, and `LowerSpyreOps`'s
  hint-based matching are that proposal's content, organized here as a
  frontend-facing RFC.
- **PR #215 — `tl.where` compare/select pattern recognition.** Owns the
  pattern detection that turns `tl.where`'s `cmpf`/`select` pair into
  `spyreop.compare` + `spyreop.select` automatically, referenced above
  under *Preserving the existing Triton path* and *Other composite
  candidates*. This document depends on that pass existing for `tl.where`,
  but does not define it.
- **`LowerSpyreOps.cpp` — existing SpyreOp lowering.** The pass this
  proposal extends with hint-based matching, alongside its existing
  structural pattern matching (`math.sqrt` → `spyreop.sqrt`, and friends),
  which is unaffected.
