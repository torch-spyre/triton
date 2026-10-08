# `tl.spyre_op`: A Staged Frontend Interface for SpyreOps

This document specifies the frontend mechanism for exposing Spyre-only
compute intrinsics — the SpyreOp dialect ops that have no portable meaning
in ordinary Triton — to kernel authors, and the lowering path that
connects a kernel's call to the real `spyreop.*` hardware op.

```python
tl.spyre_op("gelu", x)
```

Operations that already have a natural Triton spelling (`tl.exp`,
`tl.sqrt`, arithmetic operators, `tl.where`, ...) keep that spelling as
the default: the compiler lowers them to their SpyreOp equivalents
internally, with no frontend change required. `tl.spyre_op` is still
reachable for these — `tl.spyre_op("exp", x)` is a valid, equivalent way
to ask for the same thing explicitly, for a kernel author who wants every
SpyreOp request spelled the same way — but it adds nothing `tl.exp(x)`
doesn't already give for free. Where `tl.spyre_op` earns its place is
everything with no existing Triton spelling at all: activation functions
with no existing Triton name, address-generation intrinsics, composite
operations such as LayerNorm's stages, whose internal computation must
not leak into kernel code, and operations like `topk` whose real
implementation depends on inter-core communication that has no
representation in a Triton kernel at all.

## Design principle: explicit requests are honored, generic code is optimized freely

One rule governs when a `spyreop.*` op may appear in place of what a
kernel author actually wrote, and it applies uniformly to everything in
this document:

- **An explicit request for a SpyreOp is always honored.** A call to
  `tl.spyre_op(name, ...)` compiles to exactly the op `name` identifies.
  The compiler does not second-guess it, substitute a different op it
  considers more efficient, or silently decline it. Asking for a specific
  instruction is a choice that belongs to whoever is writing the kernel,
  and lowering must respect it unconditionally.
- **Generic Triton code carries no such guarantee, in either direction.**
  If a kernel author writes ordinary code — arithmetic, `tl.exp`,
  `tl.where`, a plain `1 / x` — with no explicit SpyreOp request anywhere
  in it, the compiler is free to lower it to a SpyreOp whenever doing so
  is recognizably the efficient implementation on Spyre hardware, with no
  change required to what the author wrote. This is permission, not an
  obligation: it holds only where a lowering pass actually recognizes the
  pattern, and nothing here commits every generic op to a SpyreOp mapping
  or requires a kernel author to track which ones currently are.

Everything in this document is one side of that rule or the other.
`LowerSpyreOps`'s structural pattern matching (`math.sqrt` →
`spyreop.sqrt`, `1 / x` → `spyreop.reciprocal`, and friends) and the
compare/select pattern recognition that turns `tl.where` into
`spyreop.compare`/`spyreop.select` are the compiler exercising its
freedom over generic code — see *Preserving the existing Triton path* and
*Reciprocal*, below. `tl.spyre_op` is the other side: it exists for ops
with no generic Triton spelling at all for the compiler to
opportunistically recognize, where the only way to reach the SpyreOp is
to ask for it directly — and, per the first bullet above, that request is
never overridden once made.

## KTIR stays backend-independent

KTIR is not a Spyre-specific representation. It is the common lowering
target `_make_ktir` produces for *any* backend built on it, and its
abstractions — `linalg`, `arith`, `math`, and friends — are defined with
no knowledge of Spyre at all. `spyreop.*` is exactly the opposite: a
dialect that exists only to name Spyre-hardware-specific operations, and
it belongs exclusively to the Spyre-specific half of the pipeline —
`_make_spyrecode` and the `LowerSpyreOps` pass that runs there.

Letting a `spyreop.*` op appear in `_make_ktir`'s output would break that
separation: it would put a Spyre-only op into a representation that is
supposed to mean the same thing regardless of backend. `tl.sqrt` never
raises this problem, because `math.sqrt` is already a generic,
backend-agnostic op in its own right. `spyreop.gelu` does: there is no
generic KTIR op that already means "GELU." For any intrinsic with no
pre-existing portable spelling, `_make_ktir`'s output has to stay
Spyre-agnostic some other way — which is what `tl.spyre_op` and its
staging op, `tts.spyre_op`, exist to do: every such intrinsic traces to a
generic KTIR representation first, never to `spyreop.*` directly.

A useful consequence falls out of this for free: because the cached
`_make_ktir` artifact never contains `spyreop.*`, it is also exactly what
`ktir_cpu` — the numerical interpreter the test suite runs against, which
has no `spyreop` support — needs in order to execute it directly and
verify the kernel's numerics on the CPU. That is a real and valuable
benefit, but it is a consequence of keeping KTIR backend-independent, not
the reason for doing so: the staging design would still be the right one
even for a hypothetical backend with no CPU interpreter at all, because
the underlying problem — a Spyre-only op leaking into a representation
that is supposed to be generic — has nothing to do with who else happens
to consume that representation.

## `tl.spyre_op`: the frontend entry point

```python
tl.spyre_op(op_name, *args, **attrs)
```

This is the one builtin a kernel author calls for every Spyre-only
intrinsic — there is no per-op wrapper function. `op_name` is a
compile-time constant string naming the intrinsic (`"gelu"`,
`"reciprocal"`, `"layernorm"`, ...); `args` are its tensor operands;
`attrs` carries anything that has to be a compile-time constant (`beta`
and `threshold` for `"softplus"`, `eps` for LayerNorm's scale stage, and
so on).

Each intrinsic's fallback implementation — a real, numerically faithful
computation expressed in ordinary Triton operations, not a stub — is
registered under its name, separately from any call site:

```python
@tl.spyre_intrinsic("gelu")
def _gelu_fallback(x):
    return 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
```

`@tl.spyre_intrinsic(name)` is a registration decorator, not a wrapper
factory: it records `_gelu_fallback` under the key `"gelu"` in an
internal registry and returns the function unchanged. The registered
function is never called directly by kernel code — it exists purely to
be traced into the staging op's region (see *`tts.spyre_op`*, below) the
first time `tl.spyre_op("gelu", ...)` is traced for a given signature,
and to give `ktir_cpu` something numerically correct to execute before
`spyreop.gelu` is in the picture.

An op that already has a generic Triton spelling is registered the same
way, with no special case anywhere in `tl.spyre_op`'s own implementation:

```python
@tl.spyre_intrinsic("exp")
def _exp_fallback(x):
    return tl.exp(x)
```

`tl.spyre_op("exp", x)` traces through the identical mechanism as
`tl.spyre_op("gelu", x)` — the same staging op, the same dissolution and
tagging, the same matching in `LowerSpyreOps` — and happens to land on
exactly the `spyreop.exp` that plain `tl.exp(x)` would already reach
automatically (see *Preserving the existing Triton path*, below). Nothing
distinguishes this registry entry from any other; it exists purely so a
kernel author who wants every SpyreOp request spelled the same way can
write `tl.spyre_op("exp", x)` instead of `tl.exp(x)`, without that choice
changing what the call lowers to.

This design has a direct consequence worth stating plainly: `op_name` is
a bare string, and the registry is populated by whichever
`@tl.spyre_intrinsic`-decorated modules have been imported by trace time.
A kernel calling `tl.spyre_op("gelu", x)` without the registration having
run yet, or calling `tl.spyre_op("gleu", x)` by typo, fails at trace time
with a registry-lookup error rather than a `NameError` at the point the
call was written. This is a deliberate trade: one uniform entry point and
one op shape for every lowering pass to recognize, in exchange for giving
up the compile-time name-checking and discoverability a dedicated
per-op wrapper function would have provided.

## `tts.spyre_op`: the TTIR staging op

Tracing a call to `tl.spyre_op(name, *args)` does not inline the
registered fallback's body into the caller. It emits one TTIR op,
`tts.spyre_op`, carrying:

- a **hint** attribute naming the intrinsic (`"gelu"`), used later purely
  for matching;
- a **nested region** holding the fallback body, traced under ordinary
  semantics; and
- the **`IsolatedFromAbove`** trait, so the region cannot implicitly
  capture an SSA value from the enclosing function — every value the
  fallback body needs must be an explicit operand of `tts.spyre_op`
  itself. This is what makes "the region is a self-contained fallback
  body" a structural guarantee rather than a convention.

```mlir
%r = tts.spyre_op {hint = "gelu"} (%x) ({
  ^bb0(%arg: tensor<...xf32>):
    // fallback body: the traced form of the registered function
    ...
    tts.spyreop_yield %result : tensor<...xf32>
}) : (tensor<...xf32>) -> tensor<...xf32>
```

The terminator is `tts.spyreop_yield`, specific to this op's body, not a
generic yield shared across the `tts` dialect. The region's verifier
rejects any memory-effecting operation inside the body — no loads, no
stores, nothing with side effects — enforcing that a staged intrinsic's
fallback is pure compute, matching how it is documented.

Immediately after tracing, the region's contents are **not** uniformly
generic KTIR-ready ops. For a simple elementwise intrinsic (`gelu`,
`silu`, `softplus`), Triton already traces arithmetic straight to
`arith`/`math` operating on tensors — a Python literal like `0.5` becomes
a splatted `arith.constant dense<...>` after canonicalization, so the
body already looks close to its final generic form. For a composite
intrinsic (one of LayerNorm's stages, `topk`'s fallback), the region
instead contains genuine `tt.*` ops — `tt.reduce`, `tt.broadcast`,
`tt.expand_dims`, `tt.reshape`, and so on — identical in kind to what the
rest of the kernel traces to, because that is exactly what a reduction or
a reshape look like at this point in the pipeline.

The region only becomes uniformly plain `linalg`/`arith`/`math`/`tensor`
partway through `_make_ktir`'s own pipeline: `LowerComputeOps` walks into
the region the same way it walks the rest of the module, turning
`tt.reduce`/`tt.broadcast` into `linalg.reduce`/`linalg.broadcast`, and
that stage's canonicalizer folds what it can. Only once that has happened
— right before `LowerTTSMarkers` runs — is the region uniformly generic,
which is exactly the point at which it gets dissolved (see below).

One generic staging op serves every intrinsic in the initial operation
set: nothing about an unknown or future intrinsic needs its own TTIR op.

## Dissolving `tts.spyre_op`: `LowerTTSMarkers` and per-operation tagging

`tts.spyre_op` survives, as a stable, hint-tagged op, all the way to the
point where `LowerTTSMarkers` runs — the same existing pass that already
consumes the `tts` dialect's other marker op, `tts.tensor_layout`, at the
end of `_make_ktir`'s pipeline. `LowerTTSMarkers` is extended to also
handle `tts.spyre_op`: it inlines the region directly into the
surrounding function in place of the op, and tags **every operation that
came from the region** with a discardable `tts_hint = {hint, id}`
attribute (`id` distinguishes multiple staged intrinsics of the same kind
in one function). The one operation that merely *feeds* the intrinsic —
computed outside the region, passed in as an operand — is not part of
the body and stays untagged.

```mlir
%sum    = arith.addf %x, %y : tensor<128xf16>              // feeds the intrinsic; untagged
%half   = arith.constant dense<5.000000e-01> : tensor<128xf16>
%one    = arith.constant dense<1.000000e+00> : tensor<128xf16>
%t      = math.tanh %sum      {tts_hint = {hint = "gelu", id = 0 : i64}} : tensor<128xf16>
%u      = arith.addf %one, %t {tts_hint = {hint = "gelu", id = 0 : i64}} : tensor<128xf16>
%v      = arith.mulf %sum, %u {tts_hint = {hint = "gelu", id = 0 : i64}} : tensor<128xf16>
%result = arith.mulf %half, %v {tts_hint = {hint = "gelu", id = 0 : i64}} : tensor<128xf16>
```

This is what the cached `_make_ktir` artifact actually contains: ordinary
`arith`/`math` tensor ops, each carrying a discardable attribute.
Discardable attributes do not change execution semantics, so `ktir_cpu`
runs this exactly as it would any other kernel — no special-casing for
`tts.spyre_op`, or anything else, is needed anywhere in the numerical
interpreter. The artifact is genuinely generic, not merely free of
`spyreop.*`.

## `FuseComputeAndDataMovement`: regrouping tagged operations

`_make_spyrecode`'s `ConvertElementwiseToLinalg` scalarizes each tagged
op into its own `linalg.generic`, copying the tag onto the scalar op
inside each body — it has no idea the tagged ops are related, so a single
intrinsic's worth of tagged ops ends up spread across several separate
generics:

```mlir
%17 = linalg.generic ... { %a = math.tanh %in         {tts_hint = {hint = "gelu", id = 0}} : f16 ... }
%19 = linalg.generic ... { %a = arith.addf %in, %cst_0 {tts_hint = {hint = "gelu", id = 0}} : f16 ... }
%21 = linalg.generic ... { %a = arith.mulf %in, %in_1  {tts_hint = {hint = "gelu", id = 0}} : f16 ... }
%23 = linalg.generic ... { %a = arith.mulf %in, %cst   {tts_hint = {hint = "gelu", id = 0}} : f16 ... }
```

`FuseComputeAndDataMovement` — an existing pass, not a new one — gains one
additional fusion clause: also fuse a producer into its consumer when
both bodies carry the same `tts_hint.id`. That merges the fragmented
generics back into a single body, with every op still individually
tagged:

```mlir
%17 = linalg.generic ... ins(%15) {              // the whole tagged group, one body
^bb0(%in: f16, %out: f16):
  %a = math.tanh %in         {tts_hint = {hint = "gelu", id = 0 : i64}} : f16
  %b = arith.addf %a, %cst_0 {tts_hint = {hint = "gelu", id = 0 : i64}} : f16
  %c = arith.mulf %in, %b    {tts_hint = {hint = "gelu", id = 0 : i64}} : f16
  %d = arith.mulf %c, %cst   {tts_hint = {hint = "gelu", id = 0 : i64}} : f16
  linalg.yield %d : f16
}
```

## `LowerSpyreOps`: tag-based matching

`LowerSpyreOps` already rewrites scalar ops like `math.sqrt` →
`spyreop.sqrt` by ordinary structural pattern matching inside a
`linalg.generic` body — that matching is unaffected and keeps running in
the same pass. It gains one additional rule: given a `linalg.generic`
body where a group of operations shares one `tts_hint` tag, replace the
whole group with the single `spyreop.*` op the tag's `hint` names
(`spyreop.gelu`, in the example above), discarding the tagged ops
entirely. This is the point at which the real hardware op finally
appears — nowhere earlier in the pipeline, and never in anything
`_make_ktir` caches.

For an intrinsic like `"exp"`, whose fallback is just `tl.exp(x)`, the
tagged op is a single `math.exp`, already caught by `LowerSpyreOps`'s
existing structural rule regardless of the tag — the two rules simply
agree on the same answer.

## Preserving the existing Triton path

`tl.exp`, `tl.sqrt`, `tl.rsqrt`, `tl.where`, and the arithmetic operators
work exactly as they do on any other backend, through their existing
route. None of this is an explicit SpyreOp request, so none of it is
required to lower to one — the table below is the compiler exercising
the second half of the design principle above: recognizing an efficient
Spyre lowering for ordinary code without asking the kernel author to
change anything:

| Triton source | TTIR | Spyre lowering |
|---|---|---|
| `tl.exp(x)` | `math.exp` | `spyreop.exp` |
| `tl.sqrt(x)` | `math.sqrt` | `spyreop.sqrt` |
| `tl.rsqrt(x)` | `math.rsqrt` | `spyreop.rsqrt` |
| `x / y` | `arith.divf` | `spyreop.realdiv` (or `spyreop.reciprocal` when the numerator is a constant `1.0` — see *Reciprocal*, below) |
| `x + y`, `x * y` (int32/int64, inside elementwise compute) | `arith.addi`/`arith.muli` | `spyreop.addi32toi32` / `addi64toi64` / `muli32toi32` |
| `tl.where(cond, x, y)` | `arith.cmpf` + `arith.select` | `spyreop.compare` + `spyreop.select`, via pattern recognition — see below |

None of these ever *need* to produce a `tts.spyre_op` — they trace
straight to `math.*`/`arith.*`, which is already a valid, portable
fallback in its own right, and `LowerSpyreOps`'s structural matching
handles them in `_make_spyrecode` directly. The first three are also
reachable as `tl.spyre_op("exp", x)`/`"sqrt"`/`"rsqrt"`, which *does*
produce a `tts.spyre_op`, trivially, purely so every SpyreOp request can
be spelled the same way — see *`tl.spyre_op`: the frontend entry point*,
above. Either spelling reaches the same `spyreop.*` op. `tl.spyre_op` is
not a replacement for the plain spellings in this table, and this
document does not recommend migrating kernels onto it for ops that
already have one. The staged path earns its place for the cases the
transparent path cannot cover:

- **Spyre-only operations** with no portable meaning at all (`addi32toi32`,
  `idx32toaddr`).
- **Operations with no standard Triton spelling**, even where the
  underlying math could in principle be written out by hand (`gelu`,
  `reciprocal`).
- **Composite operations** that must not leak an internal, unnamed
  representation into kernel code.

The rule for which bucket an op falls into is that if it already has, or
naturally deserves, a spelling that makes sense on every backend, it
stays on the transparent path above — generic code the compiler is free
to lower efficiently. `tl.spyre_op` exists precisely where that is not
available: there is no generic spelling for the compiler to
opportunistically recognize in the first place, so reaching the SpyreOp
requires an explicit request — one that, once made, is never overridden.

## Who maintains fallback implementations

**Spyre backend/compiler developers, not kernel authors.** A fallback
body is not an arbitrary convenience implementation — it is what
`ktir_cpu` treats as ground truth for the op during `_make_ktir`, so it
must be numerically faithful to the real `spyreop.*` op's defined
semantics. Getting that right requires exactly the dialect-level
knowledge (what the op computes, what its edge cases are) that only
someone implementing or maintaining the SpyreOp dialect reliably has.
These registrations are shipped as part of Spyre's own frontend library:

```python
# illustrative — exact module path not fixed by this document
# triton/language/extra/spyre/_fallbacks.py

import triton.language as tl

@tl.spyre_intrinsic("gelu")
def _gelu_fallback(x):
    ...

@tl.spyre_intrinsic("exx2")
def _exx2_fallback(x):
    ...
```

None of these registered functions are exported — a kernel author never
imports `_gelu_fallback` and never sees its name. The only thing imported
is `tl` itself, since `tl.spyre_op` is already part of core Triton.

Nothing technically prevents a kernel author from registering their own
`@tl.spyre_intrinsic`-decorated function, but doing so correctly requires
knowing the exact hint string `LowerSpyreOps` matches on and the real
op's semantics — implementation-internal knowledge, in the same way
hand-constructing `spyreop.*` TTIR today is possible but not intended.
The expectation is a closed, backend-maintained registry, not an open
one.

## What kernel authors actually write

A kernel author calls `tl.spyre_op` directly, like any other `tl.*`
function — no decorator, no registry to think about, no awareness that a
staging op exists underneath:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N)

    y1 = tl.exp(x)                   # existing Triton spelling — untouched
    y2 = tl.spyre_op("gelu", x)      # Spyre-only op, no tl.gelu exists

    tl.store(out_ptr + offs, y1 + y2, mask=offs < N)
```

`tl.exp(x)` traces to `math.exp`, which `LowerSpyreOps`'s structural
matching turns into `spyreop.exp` in `_make_spyrecode` — no
`tts.spyre_op` involved at any point. `tl.spyre_op("gelu", x)` traces to
`tts.spyre_op {hint = "gelu"}` wrapping the registered fallback body;
`LowerTTSMarkers` inlines and tags it before `_make_ktir` caches its
artifact, and `_make_spyrecode`'s `FuseComputeAndDataMovement` plus
`LowerSpyreOps` turn the tagged group into `spyreop.gelu`. Both calls end
up fully lowered by the time `_make_spyrecode` is done; the difference is
invisible from the kernel author's side and only matters to how the
compiler gets there.

## Initial operation set

| Frontend API | Fallback | Final lowering | Notes |
|---|---|---|---|
| `tl.spyre_op("exp", x)` | identical to `tl.exp(x)` | `spyreop.exp` | also reachable directly via `tl.exp(x)` — see *Preserving the existing Triton path* |
| `tl.spyre_op("sqrt", x)` | identical to `tl.sqrt(x)` | `spyreop.sqrt` | also reachable directly via `tl.sqrt(x)` |
| `tl.spyre_op("rsqrt", x)` | identical to `tl.rsqrt(x)` | `spyreop.rsqrt` | also reachable directly via `tl.rsqrt(x)` |
| `tl.spyre_op("gelu", x)` | GELU approximation, standard Triton ops (`tanh`, arithmetic) | `spyreop.gelu` | F16/DF16 only |
| `tl.spyre_op("silu", x)` | `x * sigmoid(x)`, standard ops | `spyreop.silu` | F16/DF16/F32 |
| `tl.spyre_op("softplus", x, beta, threshold)` | `log1p(exp(beta * x)) / beta`, with the linear fallback above `threshold`, standard ops | `spyreop.softplus` | F16/DF16 only |
| `tl.spyre_op("reciprocal", x)` | `1.0 / x`, ordinary division | `spyreop.reciprocal` | fallback is **exact**; real op trades exactness for speed — see below |
| `tl.spyre_op("idx32toaddr", index, base, stride)` | `base + stride * index`, ordinary integer arithmetic | `spyreop.idx32toaddr` | address-generation intrinsic |
| `tl.spyre_op("addi32toi32", a, b)` / `"addi64toi64"` | ordinary `a + b` | `spyreop.addi32toi32` (etc.) | explicit address-arithmetic add; see below |
| `tl.spyre_op("muli32toi32", a, b)` | ordinary `a * b` | `spyreop.muli32toi32` | explicit address-arithmetic multiply; see below |
| `tl.spyre_op("exx2", x)` | mean and sum-of-squares, standard reduction | `spyreop.exx2` | LayerNorm stage 0 — see *LayerNorm*, below |
| `tl.spyre_op("layernormscale", x, mean, msq, eps)` | variance and scale from the stage-0 pair, standard ops | `spyreop.layernormscale` | LayerNorm stage 1 — see *LayerNorm*, below |
| `tl.spyre_op("layernormnorm", x, mean, scale, weight, bias)` | normalize and affine-transform, standard ops | `spyreop.layernormnorm` | LayerNorm stage 2 — see *LayerNorm*, below |
| `tl.spyre_op("topk", x, k=k, axis=axis)` | generic top-k along `axis` (e.g. sort-and-slice or repeated argmax-and-mask), standard ops, no inter-core communication | `spyreop.topk`, realized via SFP-ring inter-core communication at device-codegen time | hardware-constrained on `k`; see *TopK*, below |

The first three rows are registered exactly like every other row — see
*`tl.spyre_op`: the frontend entry point*, above — with no special
casing anywhere in the mechanism; they simply happen to land on the same
`spyreop.*` op the plain `tl.exp`/`tl.sqrt`/`tl.rsqrt` spelling already
reaches on its own.

`tl.where` is deliberately absent from this table: compare/select is not
a `tl.spyre_op` call and produces no `tts.spyre_op` at all. It stays on
the standard Triton API surface — see *Preserving the existing Triton
path*, above, and the dedicated discussion after this table.

**On `"addi32toi32"`/`"muli32toi32"` appearing here despite `+`/`*`
already being transparent above:** these are two different call sites
for the same underlying hardware op, not a duplication. Ordinary tensor
`+`/`*` inside elementwise compute is already covered by the transparent
path — that path deliberately fires only inside elementwise compute,
precisely so it never misclassifies hand-written index or address
arithmetic elsewhere in a kernel as something to fuse. That exclusion
should stay. It does mean a kernel author computing an address by hand
(feeding `idx32toaddr`, for example) has no transparent route to the
hardware add/multiply there — which is exactly the gap
`tl.spyre_op("addi32toi32", a, b)` fills: an explicit request for the
intrinsic in a context the transparent path is deliberately blind to.

## `tl.where`: standard API, Spyre-specific lowering underneath

A kernel author writes ordinary `tl.where(cond, x, y)` — no intrinsic, no
hint, no special import — exactly as they would on any other backend:

```
tl.where(cond, x, y)
  →  arith.cmpf + arith.select     (ordinary TTIR, unchanged)
  →  compare/select pattern recognition
  →  spyreop.compare + spyreop.select
```

A compiler pass structurally recognizes the resulting `cmpf`/`select`
pair and rewrites it directly to `spyreop.compare` + `spyreop.select`.
This decomposition is similar in spirit to LayerNorm's: one
frontend-visible operation lowers to more than one `spyreop.*` op. The
difference is where that operation lives. LayerNorm's stages have no
portable Triton spelling at all, so they need explicit
`tl.spyre_op(...)` calls and the staging machinery described above.
`tl.where` already has a portable, backend-agnostic Triton spelling — it
belongs on the transparent path in *Preserving the existing Triton path*,
not in the operation-set table, and its multi-op decomposition happens
entirely inside pattern-based lowering, with nothing staged and nothing
added to the frontend surface.

## Reciprocal: an automatic rewrite and an explicit request

`LowerSpyreOps` recognizes a constant-`1.0` numerator in `arith.divf` and
rewrites it to `spyreop.reciprocal` rather than `spyreop.realdiv`,
unconditionally. So `1 / x`, written literally, lowers to
`spyreop.reciprocal` with no explicit request involved at all.

That automatic rewrite covers the common case where a kernel author
happens to write division by the literal constant one. It is not a
substitute for also keeping `tl.spyre_op("reciprocal", x)` in the initial
operation set, for two reasons:

- **Not every call site is reachable as a literal `1.0`-numerator
  division.** Wherever the numerator is an expression rather than the
  literal constant `1.0` — even one provably equal to one by other means
  — the automatic pattern doesn't fire, and the kernel falls back to
  `spyreop.realdiv` whether or not the author actually wanted the
  dedicated reciprocal instruction.
- **An explicit call states intent directly**, independent of how the
  division happens to be spelled, giving `LowerSpyreOps` an unambiguous
  signal rather than inferring intent from a specific numerator shape.

The two paths coexist: `1 / x` is caught by the automatic `arith.divf`
rewrite, and `tl.spyre_op("reciprocal", x)` is available as a direct,
staged intrinsic (fallback: ordinary division, for `ktir_cpu`) for a
kernel author who wants to request the dedicated instruction explicitly
rather than relying on how a division happens to be written. Both compile
to the same `spyreop.reciprocal` op.

A hardware reciprocal instruction may trade numerical exactness for speed
in a way ordinary division does not, so the explicit intrinsic's software
fallback (ordinary division, for `ktir_cpu`'s purposes) is **not**
necessarily a numerically exact stand-in for what `spyreop.reciprocal`
actually computes on real hardware — the two are expected to diverge
slightly, by design. This applies equally to the automatic path: a kernel
author who writes `1 / x` and gets `spyreop.reciprocal` is accepting the
same divergence, whether or not they intended to invoke the dedicated
instruction.

## LayerNorm: a three-stage pattern

```python
mean, msq = tl.spyre_op("exx2", x)
scale     = tl.spyre_op("layernormscale", x, mean, msq, eps=1e-5)
y         = tl.spyre_op("layernormnorm", x, mean, scale, weight, bias)
```

LayerNorm is exposed as three separate, explicit `tl.spyre_op` calls, one
per hardware stage, rather than a single composite entry point. Each
stage is an ordinary row in the initial operation set — simple, not
composite, in exactly the sense `gelu` or `silu` are — because each
traces to exactly one `tts.spyre_op`, dissolves and tags independently,
and lowers to exactly one `spyreop.*` op.

Each stage's fallback is real, numerically faithful Triton code, no
different in kind from any other registered fallback:

```python
@tl.spyre_intrinsic("exx2")
def _exx2_fallback(x):
    # one reduction, two results: mean and mean-of-squares
    def _add2(a0, a1, b0, b1):
        return a0 + b0, a1 + b1
    N = x.shape[1]
    mean, msq = tl.reduce((x / N, x * x / N), axis=1, combine_fn=_add2)
    return mean, msq

@tl.spyre_intrinsic("layernormscale")
def _layernormscale_fallback(x, mean, msq, eps: tl.constexpr):
    var = msq - mean * mean
    return tl.rsqrt(var + eps)

@tl.spyre_intrinsic("layernormnorm")
def _layernormnorm_fallback(x, mean, scale, weight, bias):
    return (x - mean[:, None]) * scale[:, None] * weight[None, :] + bias[None, :]
```

This deliberately exposes `mean`/`msq` and `scale` as ordinary,
Python-visible tensor values between the three calls — a kernel author
can see them, name them, and place them. That is the point of the
three-stage pattern, not an accident of it: `lx-placement.md` lets a
kernel author pin a value to on-chip scratchpad rather than pay an
off-chip round trip for an intermediate that never needed to leave the
chip, and that lever only works on a value the kernel's own source
names. A single opaque `layernorm` entry point would hide exactly the
values this placement control needs, with no lever exposed to an author
who might know their kernel's memory pressure better than a
general-purpose lowering pass would.

The cost of this is equally direct: the intermediate values between
stages are no longer purely internal to the implementation — a kernel
author who chains the three calls incorrectly (wrong order, mismatched
shapes, reusing `mean` from a different call) gets whatever error that
produces, in exchange for the placement control above. There is no
dedicated verifier enforcing the three-stage call protocol beyond
ordinary Triton type/shape checking on each call's own arguments.

**A future, fully-integrated `tl.spyre_op("layernorm", ...)` single-call
entry point remains possible**, if the three-call pattern's visibility
into the intermediate values proves undesirable for some kernels — for
instance, if a kernel author should not be able to see or depend on the
exact `mean`/`msq` pair's shape or dtype, so the backend can change what
it computes between stages without a frontend-visible break. Such an
entry point would need a splitting pass (here called `ExpandSpyreOps`,
not yet built) that mechanically produces the same three tagged groups
from one traced call, trading away the placement control above for full
opacity. This document does not build that now: the three-call pattern
already reaches the real hardware ops directly, and nothing about it
needs to be revisited unless a concrete kernel actually needs the
opacity a unified call would provide.

## TopK: hidden SFP-ring inter-core communication

```python
values, indices = tl.spyre_op("topk", x, k=k, axis=axis)
```

TopK is different in kind from every other entry in the initial operation
set, not just in degree. Every other intrinsic — even LayerNorm's stages
— is a single-core computation: whatever each stage computes still lives
entirely inside one core's own data and instructions. Spyre's real TopK
implementation is not single-core at all: it is realized through
**SFP-ring inter-core communication**, in which participating cores
exchange partial top-k candidates around a ring to combine their local
results into a globally correct top-k.

Nothing about SFP-ring communication is representable in a Triton kernel
today — there is no `tl.*` construct, no TTIR op, and no KTIR abstraction
for "a value traveling between cores around a ring." Introducing one
would be a substantial undertaking in its own right (a new dialect-level
concept, new scheduling and placement implications, likely new KTDP
primitives), and this document deliberately leaves it **out of scope**.
`"topk"` is staged precisely so that this document doesn't have to
resolve that: the ring protocol is entirely a `_make_spyrecode`/
device-codegen-time concern, invisible above the single `spyreop.topk`
op itself. A kernel author calls `tl.spyre_op("topk", x, k=k, axis=axis)`
and never sees, or needs to see, that cores are communicating at all.

**This is a restriction on one specific communication mechanism, not on
inter-core communication in general.** Cores already communicate with
each other through LX/HBM today — reading and writing shared tensor data
across cores is an ordinary part of the existing programming model, not
something this document touches. What's specifically out of scope is the
**SFP-ring** mechanism `spyreop.topk` happens to use internally: that one
mechanism has no Triton-level abstraction and isn't getting one here. A
future operation built on a different inter-core mechanism — or even a
future, lower-level exposure of SFP-ring itself — is not precluded by
anything in this document; it simply isn't what `"topk"` does, and
`"topk"` doesn't need it to be exposed in order to work.

This also differs from LayerNorm's design in kind, not just reuse of the
same idea. LayerNorm's hiding (what little remains of it, given the
three-stage pattern above) happens at the TTIR decomposition layer. TopK
hides an internal *communication protocol* one level below even the
final `spyreop.topk` op: nothing about the ring shows up as a decomposed
`tts.spyre_op` at all, because there is nothing at the TTIR level to
decompose — the ring is realized entirely inside this one op's
device-side implementation. `"topk"` therefore needs the staging
treatment, but nothing resembling LayerNorm's multi-call pattern.

### Hardware constraints on `k`

TopK's real implementation is subject to hardware limits that have no
analogue in the fallback and no natural expression in today's frontend:

- Each core can perform up to **4 local sorts**.
- With **32 cores**, the maximum supported `k` is **128** — consistent
  with (though this document does not assert any more general formula
  than) 32 cores × 4 local sorts each.
- Valid `k` values depend on **both** the requested `k` and the number of
  participating cores, not on `k` alone — the same `k` that's valid with
  32 cores participating may not be valid with fewer.

Because these are hardware constraints, not soft tuning advice, a
`"topk"` call with an infeasible `(k, num_cores)` pairing must fail
loudly rather than silently generating code that computes a wrong result
or fails only at the device level.

### Validation strategy

Validation happens in two layers:

1. **`k` is required to be a compile-time constant.** `tl.spyre_op`'s
   `k` attribute for `"topk"` is a `tl.constexpr`, not a runtime value.
   Nothing about a hardware-feasibility check is possible otherwise — a
   `k` that can vary at runtime can't be checked once at compile time,
   and would push this entire question to a device-time failure, which
   is exactly what this design avoids.
2. **The authoritative check is a verifier with access to the resolved
   participating core count, not a frontend-only assertion.** `k` alone
   is not enough to validate — the constraint is on the pair `(k,
   num_cores)` — and the call site does not necessarily know the final
   core allocation (that may only be resolved once scheduling/placement
   has run). So the primary check belongs on `spyreop.topk` itself (or on
   `tts.spyre_op {hint = "topk"}`, before dissolution): a verifier that
   runs once core count is resolved, reads both `k` and the resolved
   `num_cores` off the op, and rejects any combination outside the
   documented limits (today: ≤4 local sorts/core, ≤128 total at 32 cores)
   with `mlir::emitError`, failing the pass pipeline rather than
   proceeding.

   A **frontend-side assertion in `tl.spyre_op`'s `"topk"` handling** is
   worth adding alongside the verifier, as an early, best-effort check:
   reject any `k` above the largest value feasible under *any* core
   count the target hardware supports (today, `k > 128`) immediately at
   trace time, before core allocation is even attempted. This catches
   the most common mistake — a `k` that's never going to work regardless
   of scheduling — with a kernel-level Python traceback instead of a
   pass-level MLIR diagnostic, without pretending the frontend can fully
   validate the pair on its own.

Either diagnostic names the actual numbers involved — the requested `k`,
the resolved or assumed `num_cores`, and the limit that was violated —
not a generic "unsupported `topk` configuration." A kernel author should
be able to fix the call from the error message alone, without needing to
understand the ring protocol that makes the limit exist.

This resolves *where* the check lives and *what* it checks; it does not
by itself answer whether a kernel author should ever need to state
`num_cores` explicitly, which is the next open question.

### Open questions this document does not resolve

- **Should core-count-dependent limits on `k` be part of `"topk"`'s own
  frontend API contract** — e.g. an explicit `num_cores` or
  `max_k`-style argument the kernel author must supply — or should the
  frontend stay silent about core count and rely entirely on the verifier
  above, which reads the resolved count rather than asking for it? The
  validation strategy above is written to work either way, but defaults
  toward the latter (no explicit argument) since core count is normally
  a scheduling decision, not something a kernel author chooses directly.
- **How much SFP-ring behavior should be observable at all, even just as
  documentation or diagnostic text, versus staying a pure implementation
  detail?** The verifier's diagnostic above is deliberately framed in
  terms of `k` and core count, not ring mechanics — but a kernel author
  who wants to understand *why* the limit is 4 local sorts per core, for
  instance, needs that explained somewhere. What the error message is
  allowed to say, versus what belongs only in this document or in
  backend-internal comments, is part of this question.

### Software fallback divergence

If `"topk"`'s software fallback (for `ktir_cpu`) implements top-k with an
ordinary sequential algorithm — sort-and-slice, or repeated
argmax-and-mask — with no ring communication at all, that fallback can be
a numerically valid top-k while still being structurally nothing like the
real ring-based hardware execution. This is the same class of
fallback/real-op divergence this document already accepts for
`reciprocal`, but it is worth naming on its own for TopK, for two reasons
specific to it:

- **Tie-breaking and ordering may not match.** Where the input has
  repeated values at the `k`-th-largest boundary, which particular
  elements (and indices, since `"topk"` returns indices — an
  ordering-sensitive result, unlike `reciprocal`'s single scalar) get
  selected can legitimately differ between an arbitrary sequential
  tie-break and whatever the ring protocol's own tie-break happens to
  produce.
- **The fallback cannot exercise the hardware constraints above at all.**
  A sequential software implementation has no notion of "32 cores" or "4
  local sorts," so it will happily compute a top-k for a `k` the real
  hardware op could never support. The fallback succeeding is therefore
  no signal that a given `(k, num_cores)` pairing is actually feasible —
  the validation discussed above has to be enforced independently of the
  fallback, not inferred from it.

## Other composite candidates

A multi-call pattern (in the sense LayerNorm uses one) is warranted when
there is something to hide or to place: a fixed multi-step protocol with
an easy-to-violate call order, or an intermediate value worth exposing for
placement control. It is not warranted merely because several ops happen
to run in sequence.

- **Softmax.** Its usual decomposition — row-max reduce, subtract, `exp`,
  sum-reduce, divide — is made entirely of ops already covered by the
  transparent path above. There is no hidden internal representation and
  no placement-worthy intermediate analogous to LayerNorm's, so there is
  nothing a staged entry would need to expose or hide. It is generic code
  with no explicit SpyreOp request anywhere in it, so under the design
  principle above, any scheduling win across its sub-ops is the
  compiler's to take if it can recognize the pattern — a backend fusion
  pass, the same kind of pass that already turns `tl.where` into
  `spyreop.compare` + `spyreop.select` — not a reason to add a new
  intrinsic.
- **Activation functions.** `gelu`, `silu`, `softplus` are each already a
  single dialect op — there is no sequence to hide, so each is a plain
  entry in the initial operation set above, not a multi-call pattern.
- **Future fused ops.** The same test applies to anything added later:
  does this operation have an internal representation or a
  placement-worthy intermediate that a kernel author should be able to
  see and control, or is it just several already-transparent ops placed
  next to each other? The former earns a multi-call entry following the
  LayerNorm treatment above; the latter should stay decomposed on the
  transparent path.

## Open question: a frontend-visible DF16 type

Separately, Spyre is considering introducing a frontend-visible,
Spyre-native 16-bit datatype (`DF16`) rather than treating everything as
`FP16`. This document does not resolve how that interacts with the
staged intrinsic design, but it raises several questions worth naming
now:

- Should kernel authors be able to pass a `DF16` tensor into a
  `tl.spyre_op` call directly, or only after an explicit conversion to a
  type the fallback already handles?
- A fallback body is ordinary Triton source. If `DF16` isn't a type most
  standard Triton ops (the ones a fallback body is built from) know how
  to operate on natively, does every fallback that needs to support
  `DF16` acquire a type-specific branch, or does `DF16` route through a
  conversion before reaching the fallback at all — and if so, is that
  conversion itself expressible without a `spyreop.*` op appearing in
  `_make_ktir`'s artifact?
- Does `ktir_cpu` need its own `DF16` numerics to execute a fallback body
  faithfully, or does the fallback's `DF16` handling necessarily degrade
  to an approximation (e.g. widen to `FP32`, compute, narrow back) purely
  for the software path, distinct from what the real hardware op does?
- How does `DF16` interact with the dtype checking a registered
  fallback's own signature currently provides — does it need to be named
  explicitly in every relevant registration's type annotations, or does
  it need a broader frontend-level typing mechanism this document
  doesn't yet have?

None of these are resolved here; they are recorded as follow-up design
work the staged model will need to account for once `DF16` is real.

## Open question: long-term SpyreOp granularity

This document assumes today's SpyreOp dialect shape: a handful of large,
dedicated ops (`gelu`, `layernormnorm`, `softplus`, ...) each covering
substantial functionality. A future redesign might instead express more
of that functionality as compositions of smaller primitives (`exp`,
`log`, `add`, `mul`, `rsqrt`, ...), closer to how `softmax` is already
handled on the transparent path.

- **Potential advantage:** fewer dedicated dialect ops to design, verify,
  and maintain; more reuse of primitives that already have fallbacks and
  lowering.
- **Potential drawback:** some of today's dedicated ops exist
  specifically because their internal computation benefits from staying
  opaque to a kernel author's own fallback body — granularity and
  representation-hiding pull in opposite directions, and that tension
  doesn't disappear just because the primitives are smaller.
- **Impact on `tl.spyre_op`:** a registered fallback is written against
  today's dialect shape implicitly, by being numerically faithful to
  today's op. If the backend later re-expresses `gelu` as a composition
  of primitives, the existing registration and its hint keep working
  unchanged as long as `LowerSpyreOps` is updated to emit the new
  composition instead of the old single op — the frontend-facing
  contract (name, arity, fallback) does not need to change for this to
  happen underneath it.
- **Open question:** should that frontend-facing stability be treated as
  a guarantee of this design going forward, or only as something that
  happens to hold today? This document does not take a position, but
  notes that the tag-based matching in `LowerSpyreOps` is exactly what
  would need to be re-pointed, not anything in the frontend's own
  surface.

## Summary of open questions

- **DF16 frontend support.** How a Spyre-native 16-bit type interacts
  with intrinsic signatures, fallback bodies, and `ktir_cpu`'s numerics —
  unresolved, see above.
- **Long-term SpyreOp granularity.** Whether today's large, dedicated
  SpyreOps remain the right shape, or get decomposed into smaller
  primitives later, and what that implies for frontend stability — see
  above.
- **Ownership of the fallback registry.** This document expects
  `tl.spyre_intrinsic` registrations to be backend-maintained and shipped
  as a library, not something kernel authors write themselves — but the
  exact module location, versioning, and review process for that
  registry is not specified here and needs an owner.
- **Whether a future, fully-integrated `layernorm` call is ever worth
  building.** The three-stage pattern already reaches the real hardware
  ops directly; a unified entry point would trade away the placement
  control it provides for full opacity, and is worth building only if a
  concrete kernel needs that trade — see *LayerNorm*, above.
- **Whether core-count-dependent limits on `k` belong in `"topk"`'s
  frontend API contract** (an explicit `num_cores`/`max_k` argument) or
  stay implicit, resolved by the verifier proposed in *TopK*, above,
  which that proposal defaults toward without closing off the
  alternative.
- **How much SFP-ring behavior should be observable, even just in
  diagnostic text**, without exposing the ring protocol itself — see
  *TopK*, above.
- **TopK's software fallback has no notion of hardware feasibility.** A
  fallback that succeeds is not evidence that a given `(k, num_cores)`
  pairing is valid on real hardware, and its tie-breaking on repeated
  values at the `k`-th-largest boundary is not guaranteed to match the
  ring protocol's — see *TopK*, above.
