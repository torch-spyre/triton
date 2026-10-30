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

## Lowering: `tl.spyre_op` to `spyreop.*`

```
tl.spyre_op(...)
  -> tts.spyre_op(...)
  -> linalg.generic { spyreop.* }
```

A call to `tl.spyre_op` traces to one TTIR op, `tts.spyre_op` (see
below), which `_make_ktir`'s TTIR→KTIR pipeline converts directly into
the real `spyreop.*` op the call named, wrapped in a `linalg.generic`
body. There is no intermediate decomposition into `arith`/`math`, and no
separate pass resolves anything later: by the time `_make_ktir` finishes,
the real hardware op is already in the cached artifact.

### Rationale

This is the simplest path that gets every intrinsic in the initial
operation set to a real, testable lowering now: one new conversion step,
run once, inside `_make_ktir`'s existing TTIR→KTIR pipeline — the same
stage that already converts `tt.reduce` into `linalg.reduce` for
ordinary Triton code. It keeps the design small enough to implement and
validate immediately, which is the priority while the broader question
of how a SpyreOp request should ultimately be represented at the KTIR
level is still being worked out — see *Future work*, at the end of this
document.

## `tl.spyre_op`: the frontend entry point

```python
tl.spyre_op(op_name, *args, **attrs)
```

This is the one builtin a kernel author calls for every Spyre-only
intrinsic — there is no per-op wrapper function. `op_name` is a
compile-time constant string naming the intrinsic (`"gelu"`,
`"reciprocal"`, `"exx2"`, ...); `args` are its tensor operands; `attrs`
carries anything that has to be a compile-time constant (`beta` and
`threshold` for `"softplus"`, `axis` for `"exx2"`, `eps` for LayerNorm's
scale stage, and so on).

`op_name` is validated against the `spyreop` dialect's own op
definitions at trace time: arity, attribute names, and result count are
read directly off the real `spyreop.*` op this call targets, not
maintained as a separate frontend-side registry. A kernel calling
`tl.spyre_op("gelu", x)` for an op that doesn't exist, or passing the
wrong number of operands or attributes for the one it names, fails at
trace time with a diagnostic naming the mismatch — a call to
`tl.spyre_op("gleu", x)` by typo fails the same way a call naming any
other unregistered op would.

This design has a direct consequence worth stating plainly: there is one
uniform entry point and one op shape for every lowering pass to
recognize, in exchange for giving up the compile-time name-checking and
discoverability a dedicated per-op wrapper function would have provided.

## `tts.spyre_op`: the TTIR representation

Tracing a call to `tl.spyre_op(op_name, *args, **attrs)` emits one TTIR
op, `tts.spyre_op`, carrying:

- a **name** attribute identifying the intrinsic (`"gelu"`);
- its **tensor operands**, in call order;
- any **compile-time attributes** the intrinsic needs (`axis`, `beta`,
  `threshold`, `eps`, ...), carried as ordinary MLIR attributes; and
- as many **results** as the named intrinsic produces — one, for most;
  two, for `"exx2"`.

```mlir
%r = tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> tensor<128xf16>
```

One generic staging op serves every intrinsic in the initial operation
set: nothing about an unknown or future intrinsic needs its own TTIR op.
The op's verifier checks `name`, operand count, and attributes against
the same `spyreop` dialect definitions `tl.spyre_op` validates against at
trace time, so a `tts.spyre_op` built any other way (directly in MLIR,
bypassing the frontend) is held to the identical contract.

## Lowering `tts.spyre_op` to `spyreop.*`

For an intrinsic whose real `spyreop.*` op is elementwise — same shape
in, same shape out, one operand through five (`gelu`, `silu`,
`softplus`, `reciprocal`, `layernormscale`, `layernormnorm`, the
address-arithmetic ops) — the TTIR→KTIR conversion that resolves
`tts.spyre_op` builds the equivalent `linalg.generic`, with the real
scalar `spyreop.*` op as its body, directly:

```mlir
// tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> tensor<128xf16>

%empty = tensor.empty() : tensor<128xf16>
%r = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%x : tensor<128xf16>) outs(%empty : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = spyreop.gelu %in : f16
    linalg.yield %0 : f16
  } -> tensor<128xf16>
```

This is a single conversion step, run once, inside `_make_ktir`'s own
TTIR→KTIR pipeline. No decomposition into `arith`/`math`, no attribute
tagging, and no later pass recognizes or regroups anything: the
`spyreop.*` op this produces is already final, and it is what
`_make_ktir` caches.

For `"exx2"`, whose real `spyreop.exx2`/`spyreop.exx2_fused` op computes a
mean and a mean-of-squares over a whole axis, the same wrapping strategy
applies, built on a different `linalg` recipe: the reduction-shaped one
this codebase already builds for ordinary `tt.reduce` (`ConvertTTReduce`
in `LowerComputeOps.cpp`, which `_make_spyrecode`'s own
`LinalgGeneralizeNamedOps` already turns into a `linalg.generic`), not
the elementwise one above. The TTIR→KTIR conversion for `"exx2"` builds a
`linalg.generic` with a `reduction`-typed iterator over the named axis,
a `tensor.empty`/`linalg.fill`-built init operand, and the real
`spyreop.exx2`/`spyreop.exx2_fused` op as the whole reduction body,
applied to the un-reduced row rather than modeled as a literal per-step
combiner:

```mlir
// tts.spyre_op {name = "exx2", axis = 1 : i64} (%x)
//     : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)

%mean_init = ... // tensor.empty + linalg.fill, neutral element
%sq_init   = ...
%mean, %squares = linalg.generic {
    indexing_maps = [#row_map, #col_map, #col_map],
    iterator_types = ["parallel", "reduction"]
  } ins(%x : tensor<4x8xf16>) outs(%mean_init, %sq_init : tensor<4xf16>, tensor<4xf16>) {
  ^bb0(%in: f16, %acc0: f16, %acc1: f16):
    %r0, %r1 = spyreop.exx2 %in : f16
    linalg.yield %r0, %r1 : f16, f16
  } -> (tensor<4xf16>, tensor<4xf16>)
```

This shape is not hypothetical: it is exactly what a real, working
device kernel does today (a `tl.spyre_intrinsic("exx2_fused", x)`
request, selected into `linalg.generic { spyreop.exx2_fused }` over a
`linalg.reduce`-shaped body, in an experiment branch implementing a
full three-stage LayerNorm on device). That selection does not treat
`spyreop.exx2_fused` as a literal per-step reduction combiner — it
matches the whole reduction body as one unit (discarding the fallback's
internal accumulator plumbing entirely) and rebuilds it as the target op
applied to the call's original, un-reduced operand. The conversion this
document specifies follows the same shape, one step earlier in the
pipeline: build the reduction skeleton `ConvertTTReduce` already knows
how to build, with the real op as its body from the start, rather than a
fallback computation to be matched and replaced later.

`"exx2"`'s scalar-looking ODS signature is this: `spyreop.exx2` is a
Spyre-specific implementation detail that packs multiple values into a
single stick-level hardware operation, not a literal per-element
combiner step — the `linalg.generic` body above calls it once per stick,
not once per logical element, the same way the real device kernel this
is modeled on does.

`"exx2"` is supported through `tl.spyre_op` as an initial
implementation, not as a long-term API. Exposing a stick-packed hardware
detail directly at the frontend is expected to be temporary: a future,
higher-level API is expected to replace this call shape once the
backend has a better answer for how LayerNorm's reduction stage should
be named at the Triton level. A kernel author relying on
`tl.spyre_op("exx2", ...)` today should expect that call to be
superseded, not treated as a stable part of the frontend surface.

## KTIR is not device-neutral under this design

`_make_ktir`'s cached artifact can contain `spyreop.*` directly — a
dialect that exists to name Spyre-hardware-specific operations. This is
a deliberate property of this design, not an oversight: the artifact a
kernel compiles to at the `ktir` stage is, under this design, specific
to Spyre, not a backend-agnostic intermediate representation. Any future
backend built on KTIR, or any tool that inspects `_make_ktir`'s output
expecting it to be portable, needs to know this.

## `ktir_cpu`: executing `spyreop.*` directly

`ktir_cpu`, the numerical interpreter the test suite runs against, is
extended with one native handler per SpyreOp intrinsic — registered the
same way its existing `arith.*`/`math.*`/`ktdp.*` handlers are (see
`ktir_cpu/dialects/registry.py`'s dispatch-by-op-name design) — so it can
execute the cached `_make_ktir` artifact directly, exactly as
`_make_spyrecode` does. Both consumers of that artifact read the same
`spyreop.*` ops; `ktir_cpu`'s handler, not a traced Triton
implementation, is what defines that backend's numerics for each
intrinsic.

## Known concerns

`_make_spyrecode`'s `FuseComputeAndDataMovement` pass fuses any
`linalg.generic`→`linalg.generic` producer/consumer pair with no
inspection of what either body contains. Under this design, a
`spyreop.*`-bearing generic exists from `_make_ktir` onward — well before
that pass runs — so it is eligible for the same fusion treatment as any
other generic, including a case that pass was never designed or tested
against: fusing two already-final, already-selected device ops together,
or fusing one into an adjacent data-movement generic. Whether this
produces a real correctness problem, and what (if anything)
`FuseComputeAndDataMovement` needs to guard against it, is tracked in
**issue #230**, not resolved by this document.

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
representation in its own right, and `LowerSpyreOps`'s structural
matching handles them in `_make_spyrecode` directly. The first three are
also reachable as `tl.spyre_op("exp", x)`/`"sqrt"`/`"rsqrt"`, which
*does* produce a `tts.spyre_op`, trivially, purely so every SpyreOp
request can be spelled the same way — see *`tl.spyre_op`: the frontend
entry point*, above. Either spelling reaches the same `spyreop.*` op.
`tl.spyre_op` is not a replacement for the plain spellings in this
table, and this document does not recommend migrating kernels onto it
for ops that already have one. The staged path earns its place for the
cases the transparent path cannot cover:

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

## Who maintains the SpyreOp intrinsic table

**Spyre backend/compiler developers, not kernel authors.** The
definitions `tl.spyre_op` and `tts.spyre_op` validate against — each
intrinsic's name, arity, attributes, and result count — come directly
from the `spyreop` dialect's own op definitions, not a parallel registry
a kernel author could extend. Adding a new intrinsic to `tl.spyre_op`'s
reachable set means defining the real `spyreop.*` op and its KTIR
lowering shape; nothing about doing so correctly is kernel-author-facing
work.

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
`tts.spyre_op {name = "gelu"}`, which `_make_ktir`'s TTIR→KTIR pipeline
converts directly into `linalg.generic { spyreop.gelu }`. Both calls end
up at a real `spyreop.*` op — one by the end of `_make_ktir`, the other
by the end of `_make_spyrecode` — but the difference is invisible from
the kernel author's side and only matters to how the compiler gets
there.

## Initial operation set

| Frontend API | Final lowering | KTIR lowering shape | Notes |
|---|---|---|---|
| `tl.spyre_op("exp", x)` | `spyreop.exp` | elementwise, `linalg.generic` | also reachable directly via `tl.exp(x)` — see *Preserving the existing Triton path* |
| `tl.spyre_op("sqrt", x)` | `spyreop.sqrt` | elementwise, `linalg.generic` | also reachable directly via `tl.sqrt(x)` |
| `tl.spyre_op("rsqrt", x)` | `spyreop.rsqrt` | elementwise, `linalg.generic` | also reachable directly via `tl.rsqrt(x)` |
| `tl.spyre_op("gelu", x)` | `spyreop.gelu` | elementwise, `linalg.generic` | F16/DF16 only |
| `tl.spyre_op("silu", x)` | `spyreop.silu` | elementwise, `linalg.generic` | F16/DF16/F32 |
| `tl.spyre_op("softplus", x, beta, threshold)` | `spyreop.softplus` | elementwise, `linalg.generic` | F16/DF16 only |
| `tl.spyre_op("reciprocal", x)` | `spyreop.reciprocal` | elementwise, `linalg.generic` | real op trades exactness for speed — see *Reciprocal*, below |
| `tl.spyre_op("idx32toaddr", index, base, stride)` | `spyreop.idx32toaddr` | elementwise, `linalg.generic` | address-generation intrinsic |
| `tl.spyre_op("addi32toi32", a, b)` / `"addi64toi64"` | `spyreop.addi32toi32` (etc.) | elementwise, `linalg.generic` | explicit address-arithmetic add; see below |
| `tl.spyre_op("muli32toi32", a, b)` | `spyreop.muli32toi32` | elementwise, `linalg.generic` | explicit address-arithmetic multiply; see below |
| `tl.spyre_op("exx2", x, axis=axis)` | `spyreop.exx2` | reduction-shaped, `linalg.generic` over the named axis — see *Lowering `tts.spyre_op` to `spyreop.*`*, above | LayerNorm stage 0 — see *LayerNorm*, below |
| `tl.spyre_op("layernormscale", x, mean, msq, eps=eps)` | `spyreop.layernormscale` | elementwise, `linalg.generic` | LayerNorm stage 1 — see *LayerNorm*, below |
| `tl.spyre_op("layernormnorm", x, mean, scale, weight, bias)` | `spyreop.layernormnorm` | elementwise, `linalg.generic` | LayerNorm stage 2 — see *LayerNorm*, below |
| `tl.spyre_op("topk", x, k=k, axis=axis)` | `spyreop.topk`, realized via SFP-ring inter-core communication at device-codegen time | not elementwise, not a simple reduction — see *TopK*, below | hardware-constrained on `k`; see *TopK*, below |

The first three rows reach `spyreop.*` through the identical mechanism as
every other row — see *`tl.spyre_op`: the frontend entry point*, above —
and simply happen to land on exactly the op the plain
`tl.exp`/`tl.sqrt`/`tl.rsqrt` spelling already reaches on its own (see
*Preserving the existing Triton path*).

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
`tl.spyre_op(...)` calls and the staged path described above. `tl.where`
already has a portable, backend-agnostic Triton spelling — it belongs on
the transparent path in *Preserving the existing Triton path*, not in
the operation-set table, and its multi-op decomposition happens entirely
inside pattern-based lowering, with nothing staged and nothing added to
the frontend surface.

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
staged intrinsic for a kernel author who wants to request the dedicated
instruction explicitly rather than relying on how a division happens to
be written. Both compile to the same `spyreop.reciprocal` op.

A hardware reciprocal instruction may trade numerical exactness for
speed in a way ordinary division does not. `ktir_cpu`'s own
`spyreop.reciprocal` handler is free to implement either behavior —
exact division, or the same approximation the real hardware op uses —
and whichever it chooses, a kernel author relying on `_make_ktir`'s
cached numerics should not assume they match real device output exactly.
This applies equally to the automatic path: a kernel author who writes
`1 / x` and gets `spyreop.reciprocal` is subject to the same
consideration, whether or not they intended to invoke the dedicated
instruction.

## LayerNorm: a three-stage pattern

```python
mean, msq = tl.spyre_op("exx2", x, axis=1)
scale     = tl.spyre_op("layernormscale", x, mean, msq, eps=1e-5)
y         = tl.spyre_op("layernormnorm", x, mean, scale, weight, bias)
```

LayerNorm is exposed as three separate, explicit `tl.spyre_op` calls, one
per hardware stage, rather than a single composite entry point. Each
stage is an ordinary row in the initial operation set — simple, not
composite, in exactly the sense `gelu` or `silu` are — because each
traces to exactly one `tts.spyre_op` and lowers to exactly one
`spyreop.*` op.

The three stages are not uniform in KTIR lowering shape. `"exx2"` is
reduction-shaped — a `linalg.generic` built on the same recipe
`ConvertTTReduce` already uses for ordinary `tt.reduce`, with
`spyreop.exx2`/`exx2_fused` as its body (see *Lowering `tts.spyre_op` to
`spyreop.*`*, above). `"layernormscale"` and `"layernormnorm"` are both
elementwise and lower the same way `"gelu"` does: a `linalg.generic`
wrapping the real scalar `spyreop.*` op.

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
entry point would need a splitting pass (not yet built) that mechanically
produces the same three lowered ops from one traced call, trading away
the placement control above for full opacity. This document does not
build that now: the three-call pattern already reaches the real hardware
ops directly, and nothing about it needs to be revisited unless a
concrete kernel actually needs the opacity a unified call would provide.

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
analogue in `ktir_cpu`'s handler and no natural expression in today's
frontend:

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
   `tts.spyre_op {name = "topk"}`, before the TTIR→KTIR conversion that
   produces it): a verifier that runs once core count is resolved, reads
   both `k` and the resolved `num_cores` off the op, and rejects any
   combination outside the documented limits (today: ≤4 local sorts/core,
   ≤128 total at 32 cores) with `mlir::emitError`, failing the pass
   pipeline rather than proceeding.

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

### `ktir_cpu`'s handler and ring-protocol divergence

If `"topk"`'s `ktir_cpu` handler implements top-k with an ordinary
sequential algorithm — sort-and-slice, or repeated argmax-and-mask — with
no ring communication at all, that handler can be a numerically valid
top-k while still being structurally nothing like the real ring-based
hardware execution. This is the same class of divergence this document
already accepts for `reciprocal`'s handler, but it is worth naming on its
own for TopK, for two reasons specific to it:

- **Tie-breaking and ordering may not match.** Where the input has
  repeated values at the `k`-th-largest boundary, which particular
  elements (and indices, since `"topk"` returns indices — an
  ordering-sensitive result, unlike `reciprocal`'s single scalar) get
  selected can legitimately differ between an arbitrary sequential
  tie-break and whatever the ring protocol's own tie-break happens to
  produce.
- **`ktir_cpu`'s handler cannot exercise the hardware constraints above
  at all.** A sequential software implementation has no notion of "32
  cores" or "4 local sorts," so it will happily compute a top-k for a `k`
  the real hardware op could never support. The handler succeeding is
  therefore no signal that a given `(k, num_cores)` pairing is actually
  feasible — the validation discussed above has to be enforced
  independently of `ktir_cpu`'s handler, not inferred from it.

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
direct-lowering design, but it raises several questions worth naming now:

- Should kernel authors be able to pass a `DF16` tensor into a
  `tl.spyre_op` call directly, or only after an explicit conversion to a
  type the target `spyreop.*` op already accepts?
- Does the TTIR→KTIR conversion that builds `linalg.generic { spyreop.* }`
  need a `DF16`-specific case for every intrinsic that supports it, or
  does `DF16` route through a conversion before reaching that conversion
  step at all?
- Does `ktir_cpu` need its own `DF16` numerics in its native `spyreop.*`
  handlers to execute faithfully, or does `DF16` handling in those
  handlers necessarily degrade to an approximation (e.g. widen to
  `FP32`, compute, narrow back), distinct from what the real hardware op
  does?
- How does `DF16` interact with the arity/type checking `tl.spyre_op`
  performs against the `spyreop` dialect's own op definitions — does it
  need to be named explicitly in every relevant op's type constraints, or
  does it need a broader frontend-level typing mechanism this document
  doesn't yet have?

None of these are resolved here; they are recorded as follow-up design
work this design will need to account for once `DF16` is real.

## Open question: long-term SpyreOp granularity

This document assumes today's SpyreOp dialect shape: a handful of large,
dedicated ops (`gelu`, `layernormnorm`, `softplus`, ...) each covering
substantial functionality. A future redesign might instead express more
of that functionality as compositions of smaller primitives (`exp`,
`log`, `add`, `mul`, `rsqrt`, ...), closer to how `softmax` is already
handled on the transparent path.

- **Potential advantage:** fewer dedicated dialect ops to design, verify,
  and maintain; more reuse of primitives that already have lowering and
  `ktir_cpu` handlers.
- **Potential drawback:** some of today's dedicated ops exist
  specifically because their internal computation benefits from staying
  opaque to anything inspecting the artifact — granularity and
  representation-hiding pull in opposite directions, and that tension
  doesn't disappear just because the primitives are smaller.
- **Impact on `tl.spyre_op`:** this design's lowering is only as stable
  as the `spyreop.*` op it targets. If the backend later re-expresses
  `gelu` as a composition of primitives, the TTIR→KTIR conversion that
  currently builds `linalg.generic { spyreop.gelu }` directly would need
  to build the new composition instead — a change to the lowering pass,
  not to `tl.spyre_op`'s own call shape (`op_name`, arity, attributes),
  which can stay the same as long as the new composition accepts the
  same operands and produces the same result.
- **Open question:** should that frontend-facing stability be treated as
  a guarantee of this design going forward, or only as something that
  happens to hold today? This document does not take a position, but
  notes that the TTIR→KTIR conversion is exactly what would need to
  change, not anything in the frontend's own surface.

## Future work

This is an interim implementation, adopted to keep the design simple and
unblock Spyre backend development now. It may later be replaced by a
more device-neutral KTIR-level representation — the approaches proposed
in PR220 and PR223 are both candidates — once backend requirements and
the compiler's internal interfaces are better understood. Revisiting
this is expected, not merely possible: the tradeoff accepted in *KTIR is
not device-neutral under this design*, above, is specifically the kind
of cost a device-neutral KTIR-level API would remove. Nothing in this
document's frontend-facing surface — `tl.spyre_op`'s call shape,
`tts.spyre_op`'s shape — would need to change to make that migration, if
and when it happens; only the TTIR→KTIR conversion step and `ktir_cpu`'s
execution path would.

## Summary of open questions

- **Whether and when `"exx2"` is superseded by a higher-level API.**
  `tl.spyre_op("exx2", ...)` exposes a stick-packed hardware detail
  directly at the frontend as an initial implementation, not a long-term
  one — see *Lowering `tts.spyre_op` to `spyreop.*`*, above. What that
  higher-level replacement should look like, and what happens to
  existing `tl.spyre_op("exx2", ...)` call sites when it lands, is not
  resolved here.
- **The `FuseComputeAndDataMovement` fusion-safety question tracked in
  issue #230** — whether a `spyreop.*`-bearing `linalg.generic`, now
  present from `_make_ktir` onward, can be incorrectly fused with an
  adjacent generic that pass was never designed to treat as already-final
  device IR.
- **Whether and when to migrate off this interim design** — see *Future
  work*, above.
- **DF16 frontend support.** How a Spyre-native 16-bit type interacts
  with intrinsic arity/type checking, the TTIR→KTIR conversion, and
  `ktir_cpu`'s native handlers — unresolved, see above.
- **Long-term SpyreOp granularity.** Whether today's large, dedicated
  SpyreOps remain the right shape, or get decomposed into smaller
  primitives later, and what that implies for the stability of
  `tl.spyre_op`'s own call shape — see above.
- **Ownership of the SpyreOp intrinsic table.** This document expects the
  definitions `tl.spyre_op`/`tts.spyre_op` validate against to come from
  the `spyreop` dialect's own op definitions and to be maintained by
  backend/compiler developers — but the exact mechanism (generated from
  the dialect's ODS, or a separate shared table) is not specified here
  and needs an owner.
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
- **`ktir_cpu`'s `"topk"` handler has no notion of hardware
  feasibility.** A handler that succeeds is not evidence that a given
  `(k, num_cores)` pairing is valid on real hardware, and its
  tie-breaking on repeated values at the `k`-th-largest boundary is not
  guaranteed to match the ring protocol's — see *TopK*, above.
