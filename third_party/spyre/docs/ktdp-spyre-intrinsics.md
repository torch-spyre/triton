# Device-Neutral Spyre Intrinsics via `ktdp.*` Compute Ops

*A standalone architecture proposal for representing Spyre-only compute
intrinsics (GELU, SiLU, LayerNorm, TopK, ...) as first-class KTIR ops, rather
than tracing a Triton-level fallback and reconstructing the intrinsic's
identity from decomposed, tagged generic code after the fact.*

## 1. Motivation and design principles

### 1.1 The problem this solves

Spyre has hardware instructions — `spyreop.gelu`, `spyreop.silu`, and
eventually LayerNorm's stages and TopK — with no natural Triton spelling.
A kernel author needs a way to request one by name. Whatever mechanism
answers that request has to satisfy one non-negotiable constraint:

**`_make_ktir`'s output must stay backend-independent.** KTIR is the common
lowering target Triton produces for any backend built on it; `spyreop.*` is
a dialect that exists only to name Spyre-hardware-specific operations, and
it belongs exclusively to the Spyre-specific half of the pipeline
(`_make_spyrecode`). Letting a `spyreop.*` op leak into `_make_ktir`'s cached
artifact would break that separation outright. For an op with no
pre-existing generic KTIR equivalent — there is no generic op that already
means "GELU" the way `math.sqrt` already means "square root" —
`_make_ktir`'s output has to represent the request some other way.

### 1.2 Design principles

1. **An explicit request for a SpyreOp is always honored.** A call naming an
   intrinsic compiles to exactly that op. The compiler does not substitute,
   decline, or silently drop it — an unfulfillable request is a compile
   error, never a quiet fallback.
2. **KTIR represents the request directly, not as decomposed-and-reconstructed
   generic code.** The intrinsic's identity — its name, its arity, its result
   shape — should survive as a real, inspectable thing in the artifact, not
   as metadata scattered across several unrelated-looking ops that a later
   pass has to recognize and regroup.
3. **Scalarization and grouping decisions should see the whole op, not
   reconstruct it after the fact.** This tree has already learned this lesson
   once, independently of this proposal: `LowerSpyreOps` originally matched
   both single-op and multi-op (compare+select) patterns in one pass, and the
   multi-op case was split out into its own pass that ran first, specifically
   because "whether a GROUP of ops has a single device op... depends on
   seeing the group whole, which the lowering has already destroyed by the
   time it is asked" (that split has since been folded back into one pass,
   now with both a one-op-to-one tier and a group-to-one tier, but the
   lesson — decide before you destroy the evidence — is the same one this
   proposal applies to named intrinsics). An intrinsic *request* is a cleaner
   case of the same problem: the request's identity shouldn't need to be
   reconstructed at all if nothing ever tears it apart in the first place.
4. **Every backend decides for itself how to execute an intrinsic it doesn't
   have hardware for.** A backend without `spyreop` support — `ktir_cpu`
   today, hypothetically others later — needs *a* way to run the
   computation, but that way is that backend's own business, not something
   Triton has to trace, carry, and babysit through its own pipeline.

### 1.3 The proposed answer

Give the intrinsic request a real, named op at the KTIR level: `ktdp.gelu(x)
-> y`. It is device-neutral (nothing about it says "Spyre"), it operates on
tensors (no scalarization yet), and it is structurally a 1:1 mirror of the
eventual `spyreop.gelu` — same arity, same element-type constraints, same
result-shape rule, just one level more abstract. `_make_ktir`'s cached
artifact contains this op directly. Two independent consumers then do two
independent things with it: Spyre's own pipeline lowers it to the real
hardware instruction; any backend without one (starting with `ktir_cpu`)
decomposes it into a generic reference implementation first.

This is not a novel pattern for this dialect. `ktdp.construct_distributed_memory_view`
is already a device-neutral KTDP op whose lowering is expected to differ by
target — resolving to "selection of the appropriate underlying memref and a
local coordinate computation, potentially producing explicit address
calculation **and communication** when required by the target architecture"
(see `inter-tile-lowering-to-mem-view.md`). This proposal applies the same
shape — one device-neutral KTIR op, more than one legitimate lowering,
chosen by the consumer — to compute intrinsics instead of memory views.

## 2. Architecture overview

### 2.1 Today's compilation flow

Spyre's `SpyreBackend` (`third_party/spyre/backend/compiler.py`) registers
three stages:

| Stage | In → out | Method | What runs |
|---|---|---|---|
| `ttir` | Triton IR → Triton IR | `_make_ttir` | Upstream Triton's standard optimization passes (inliner, canonicalizer, combine, ...) |
| `ktir` | Triton IR → KTIR | `_make_ktir` | `buildTTIRToKTIRPipeline` (`third_party/spyre/lib/Pipeline.cpp`) — crosses the `tt`→KTIR dialect boundary |
| `spyrecode` | KTIR → device binary | `_make_spyrecode` | `buildSpyrecodePipeline` — shapes the module for the device, then `dbo-opt` |

The first two stages are pure IR-to-IR and each artifact is cached under its
stage name; `_make_spyrecode` consumes `_make_ktir`'s actual output,
sequentially — whatever `_make_ktir` produces is exactly what both
`ktir_cpu` (the NumPy-based numerical interpreter — see
`torch-spyre-integration.md`) and `_make_spyrecode` see.

`ktdp` (Kernel Tile Data-Parallel) is the dialect KTIR itself is built from —
every KTIR consumer already loads it unconditionally to do anything at all.
Its current scope is deliberately narrow: memory-view construction,
access-tile description, explicit load/store, inter-tile communication. Its
own description states this directly: *"Compute and control-flow operations
are delegated to standard MLIR dialects (Arith, Math, LinAlg, SCF, Tensor,
MemRef, Func)."* This proposal is, honestly, an extension of that scope —
see §11.1 for the alternative that doesn't require it.

**What exists today, concretely, and what doesn't.** `third_party/spyre/lib/Transforms/LowerSpyreOps.cpp`
already does instruction selection for the *transparent* path — ordinary
Triton code that happens to have an efficient Spyre instruction, with no
explicit request involved. It has two tiers: a one-op-to-one tier
(`math.sqrt`→`spyreop.sqrt`, `arith.divf`→`spyreop.reciprocal`/`realdiv`
depending on its numerator, and so on) and a group-to-one tier (an
`arith.cmpf`+cast or `arith.cmpf`+`arith.select` pair → `spyreop.compare`/
`spyreop.select`, since neither half of that pair is selectable alone).
`FuseComputeAndDataMovement` exists specifically to make the group-to-one
tier possible at all: `ConvertElementwiseToLinalg` scalarizes every
tensor-level op into its own `linalg.generic`, so a pair that needs to be
seen together has to be fused back into one body first. **What does not
exist anywhere in this tree today is any mechanism for a kernel to request
a named intrinsic explicitly** — there is no `tl.spyre_op`, no TTIR-level
staging op, nothing analogous. This proposal is the first thing to
introduce that capability, not an extension of one that already exists.

### 2.2 The proposed flow

```
Triton:              tl.spyre_op("gelu", x)
                              │  trace
                              ▼
TTIR:                 tts.spyre_op "gelu" (x)
                              │  _make_ktir, new conversion pass
                              ▼
General KTIR:         ktdp.gelu(x) -> y                    ◄── cached _make_ktir artifact
                       (device-neutral, tensor-level)
                      ╱                            ╲
            ktir_cpu's own                   _make_spyrecode, new
            decomposition pass                  lowering pass
                    ╱                                    ╲
                   ▼                                      ▼
   tanh-approx arith/math              linalg.generic { spyreop.gelu }
   (tensor-level, directly                 (scalar-wrapped only here,
    interpretable)                          at the very last step)
```

This is a small, local, mechanical conversion at each arrow — exactly the
shape of translation this codebase already expects to work well. The
companion note `triton-ktir-lowering-thought-exercise.md` makes the broader
case directly: *"The Triton-to-KTIR lowering pass is a bounded, local
translation"* once kernel-shape decisions are absorbed upstream and
execution-level decisions are left to the KTIR scheduler downstream. Neither
new conversion this proposal adds (`tts.spyre_op`→`ktdp.*`, `ktdp.*`→
`spyreop.*`) makes any scheduling or shape decision of its own — each is a
rename-and-retype step, which is exactly the kind of step that hypothesis
says should stay cheap.

Three new pieces of code own the three new arrows: a Triton-side conversion
from `tts.spyre_op` to `ktdp.*` (§5), a Triton-side lowering from `ktdp.*` to
`spyreop.*` (§6), and a decomposition pass living in `ktir-mlir-frontend`
that any non-Spyre consumer — `ktir_cpu` first — runs for itself (§7).

## 3. Frontend API design: `tl.spyre_op`

One flat builtin, no per-intrinsic wrapper function:

```python
tl.spyre_op(op_name, *args, **attrs)
```

- `op_name` — a compile-time constant string naming the intrinsic (`"gelu"`,
  `"layernormscale"`, `"topk"`, ...).
- `args` — the tensor operands, positional, as many as the intrinsic takes.
- `attrs` — compile-time-constant, non-tensor parameters (`beta`/`threshold`
  for `"softplus"`, `eps` for LayerNorm's scale stage, `k`/`axis` for
  `"topk"`), passed as keyword arguments.

```python
y = tl.spyre_op("gelu", x)
mean, msq = tl.spyre_op("exx2", x, axis=1)
y = tl.spyre_op("softplus", x, beta=1.0, threshold=20.0)
values, indices = tl.spyre_op("topk", x, k=8, axis=-1)
```

`semantic.py`'s implementation is a lookup, a type/arity check, and a direct
op-builder call — nothing is traced. There is no fallback-registration
decorator and no registry of Python functions on the Triton side at all
(contrast with a design where each intrinsic's software behavior is
authored as a traced `@triton.jit` body — that authorship moves to
`ktir-mlir-frontend`; see §7 and §10). The signature's arity/dtype/attribute/
result-shape rule is read from a small table — the natural place for it is
the `ktdp` op's own ODS definition, queried through a pybind binding, so
there is exactly one place this contract is stated, not two kept in sync by
hand.

## 4. TTIR representation: `tts.spyre_op`

A flat, region-less staging op in the existing `tts` dialect (Triton-to-Spyre
authoring annotations — today home to `tts.tensor_layout` and `tts.pin`,
neither of which this proposal touches):

```tablegen
def TTS_SpyreOpOp : TTS_Op<"spyre_op", [Pure]> {
  let summary = "Request one spyreop-family intrinsic by name";
  let arguments = (ins StrAttr:$name, Variadic<AnyRankedTensor>:$inputs,
                       DictionaryAttr:$attrs);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $name `(` $inputs `)` $attrs attr-dict `:` functional-type($inputs, $results)
  }];
  let hasVerifier = 1;
}
```

Its only job is to exist on the `tt`-descended side of the dialect boundary
long enough for a `TritonToKTIR` conversion pass to have something to
convert *from* — that directory's own criterion is explicitly "a `tt`
source dialect," so a `tt`/`tts`-level op is the natural anchor point even
though, under this design, nothing about it is load-bearing beyond its
signature. It carries no region and traces nothing: `op_name`, the operand
tensors, and the `attrs` dictionary are the entire payload. Its verifier
checks name/arity/dtype/attribute shape against the same table the frontend
already consulted, so a malformed op can never reach `_make_ktir`.

## 5. KTIR representation: `ktdp.*`

One dedicated op per intrinsic, living directly in the `ktdp` dialect, each a
near-mechanical mirror of its eventual `spyreop.*` counterpart — same
element-type constraints, same arity, same result-shape rule — but
tensor-typed and carrying no notion of "this requires Spyre hardware."

```tablegen
// third_party/spyre/ktir-mlir-frontend/include/ktir/Dialect/KTDP/KTDP.td
def KTDP_GeLUOp : Ktdp_Op<"gelu", [Pure, SameOperandsAndResultType, Elementwise]> {
  let summary = "Device-neutral GELU request: x * CDF(x)";
  let arguments = (ins AnyRankedTensorOf<[F16, F32]>:$operand);
  let results = (outs AnyRankedTensorOf<[F16, F32]>:$result);
}

def KTDP_EXX2Op : Ktdp_Op<"exx2", [Pure]> {
  let summary = "Mean and mean-of-squares along an axis -- LayerNorm stage 0";
  let arguments = (ins AnyRankedTensorOf<[F16, F32]>:$operand, I64Attr:$axis);
  let results = (outs AnyRankedTensorOf<[F16, F32]>:$mean,
                       AnyRankedTensorOf<[F16, F32]>:$msq);
}

def KTDP_SoftplusOp : Ktdp_Op<"softplus", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyRankedTensorOf<[F16, F32]>:$operand,
                       F32Attr:$beta, F32Attr:$threshold);
  let results = (outs AnyRankedTensorOf<[F16, F32]>:$result);
}
```

**No region, no body, no fallback.** The op *is* the declaration of the
computation, not a wrapper around one. This is the structural reason
multi-result ops (`exx2`), multi-operand ops (LayerNorm's later stages), and
attributed ops (`softplus`, `topk`) are all equally representable here —
each is just an ordinary ODS signature, with nothing analogous to "the body
must be elementwise and shape-preserving" to fight. LayerNorm's `exx2`/
`scale` intermediates are also exactly the kind of value `lx-placement.md`'s
scratchpad-pinning mechanism (`tl.spyre_pin`/`tts.pin`) already targets:
because each stage is its own op with its own real result, a kernel author
can pin the intermediate between them without this proposal needing to
invent any placement mechanism of its own.

## 6. Lowering to Spyre

**No new pass.** The elementwise `ktdp.*` ops (`gelu`, `silu`, `softplus`,
...) are already tagged `Elementwise`/`SameOperandsAndResultType` in §5 —
and `ConvertElementwiseToLinalg` is a *generic* upstream pass: it scalarizes
any op carrying those traits into a `linalg.generic` body, not a fixed list
of known op types. It already runs in `buildSpyrecodePipeline`, already
before `LowerSpyreOps`. So `ktdp.gelu(tensor<...>) -> tensor<...>` arrives
at `LowerSpyreOps` pre-scalarized, for free:

```mlir
%y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
     ins(%x : tensor<128xf16>) outs(%empty : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %r = ktdp.gelu %in : f16
    linalg.yield %r : f16
} -> tensor<128xf16>
```

— exactly the shape `math.sqrt` arrives in today. The only remaining step is
"replace scalar `ktdp.gelu` with scalar `spyreop.gelu`," which is
structurally identical to `LowerSpyreOps`'s existing one-op-to-one rules
(`SelectUnaryFloat<math::SqrtOp, spyreop::Sqrt>` and friends), so it belongs
in that pass, as one more pattern — not in a separate one:

```cpp
template <typename KTDPOp, typename SpyreOp>
struct SelectKTDPOp : public OpRewritePattern<KTDPOp> {
  using OpRewritePattern<KTDPOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(KTDPOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<SpyreOp>(op, op.getType(), op.getOperand());
    return success();
  }
};
// patterns.add<SelectKTDPOp<ktdp::GeLUOp, spyreop::GeLU>>(ctx);
```

Keeping this as a separate pass was an earlier draft's mistake, worth naming
rather than quietly fixing: `LowerSpyreOps`'s own header already states the
lesson this would have repeated — *"Splitting these across two passes is
what an earlier shape did, and it bought a standing question — which pass
claims this op — for nothing."* Design principle 3 (§1.2) cites that same
history; a second new pass here would have ignored the lesson it cites.

Because `ConvertElementwiseToLinalg` scalarizes `ktdp.gelu` as one op into
one body, there is nothing for `FuseComputeAndDataMovement` to do here
either — no fragmentation occurs in the first place, so there is nothing to
re-fuse.

Multi-result and multi-operand *elementwise* ops (`softplus` with its
`beta`/`threshold` attributes, a hypothetical elementwise LayerNorm stage)
scalarize and select the same way — more operands/results on the same
pattern shape, attributes read directly off the op. **`exx2` and `topk` are
different**, and this mechanism does not reach them: neither is elementwise
(`exx2` is a reduction, its output shape differs from its input; `topk` is
a sort/select), so `ConvertElementwiseToLinalg` will not touch either one.
Something still has to materialize them into `linalg` form — closer in
spirit to what `DropReductionInitFill`/`LowerComputeOps` already do for
other reduction-shaped constructs in this same pipeline — before any
selection pattern has a body to replace. That materialization step is new,
real work this section does not solve, and it is tracked honestly as such
in §11.4 rather than folded into "scalar wrapping" as if it were the same
problem.

## 7. Decomposition for `ktir_cpu`

`ktir_cpu` has no notion of `spyreop` and must never be asked to. It needs
*a* correct way to execute `ktdp.gelu`, and that way is a decomposition into
ops it already knows how to interpret.

This logic is authored **once**, in `ktir-mlir-frontend` — not duplicated in
Triton and not duplicated in `ktir_cpu` — as a small, generic pass:

```cpp
// third_party/spyre/ktir-mlir-frontend/lib/Dialect/KTDP/Transforms/DecomposeKTDPCompute.cpp
struct DecomposeGeLU : OpRewritePattern<ktdp::GeLUOp> {
  LogicalResult matchAndRewrite(ktdp::GeLUOp op,
                                PatternRewriter &rewriter) const override {
    // widen to f32, compute 0.5 * x * (1 + erf(x / sqrt(2))), narrow back --
    // directly in arith/math on tensors; no linalg, nothing scalar yet.
    ...
  }
};
```

`ktir_cpu` runs this pass (exposed through `mlir_ktdp`'s pass-pipeline hooks)
as a mandatory pre-interpretation step over any module that still contains
`ktdp.*` compute ops — before its interpreter ever sees the artifact.
`torch-spyre-integration.md` already documents the entry point this hooks
into: setting `TORCH_SPYRE_KTIR_CPU=1` runs the lowered KTIR on `ktir_cpu`,
the NumPy-based interpreter, so that a kernel can be validated without
physical Spyre hardware — this proposal adds exactly one mandatory step
immediately before that interpretation begins. The result for GELU:

```mlir
%xf  = arith.extf %x : tensor<128xf16> to tensor<128xf32>
%c   = arith.constant dense<0.70710678> : tensor<128xf32>
%s   = math.erf %xf, %c
%one = arith.constant dense<1.0> : tensor<128xf32>
%sum = arith.addf %one, %s : tensor<128xf32>
%h   = arith.constant dense<0.5> : tensor<128xf32>
%m   = arith.mulf %h, %xf : tensor<128xf32>
%r32 = arith.mulf %m, %sum : tensor<128xf32>
%y   = arith.truncf %r32 : tensor<128xf32> to tensor<128xf16>
```

No `linalg.generic` is needed here at all — `ktir_cpu` interprets
tensor-level `arith`/`math` directly, the same as it does today for every
other op `_make_ktir` already produces.

## 8. Pass pipeline changes

| Stage | Pass | Status | Role |
|---|---|---|---|
| `ttir` | *(unchanged)* | — | Trace `tl.spyre_op` → `tts.spyre_op`, no region |
| `ktir` (`_make_ktir`) | **`ConvertSpyreOpToKTDP`** | new (Triton) | `tts.spyre_op "name"` → `ktdp.<name>` |
| `ktir` (`_make_ktir`) | *(all existing passes — `LowerDescriptorMemory`, `LowerScalarLoad`, `LowerComputeOps`, `ConvertFunctions`, `DistributeWork`, `Canonicalizer`, `LowerTTSMarkers`)* | unchanged | — |
| → `_make_ktir`'s cached artifact: general KTIR, contains `ktdp.*` directly | | | |
| `ktir_cpu` consumption | **`DecomposeKTDPCompute`** | new (`ktir-mlir-frontend`, run by `ktir_cpu`) | `ktdp.gelu` → tanh-approx `arith`/`math` |
| `spyrecode` (`_make_spyrecode`) | `ConvertElementwiseToLinalg` | unchanged, does new work for free | Scalarizes `ktdp.gelu` into a `linalg.generic` body, same as any other `Elementwise`-tagged op |
| `spyrecode` (`_make_spyrecode`) | `FuseComputeAndDataMovement` | unchanged | Nothing to do for `ktdp.*` — one op scalarizes into one body, nothing to re-fuse |
| `spyrecode` (`_make_spyrecode`) | `LowerSpyreOps` | **gains new patterns**, no new pass | Scalar `ktdp.gelu` → `spyreop.gelu`, same shape as its existing one-op-to-one rules |

## 9. Implementation plan, by repository

### 9.1 `ktir-mlir-frontend` (must land first)

Triton's build depends on this submodule's pin defining the new ops before
anything in Triton can compile against them.

- `include/ktir/Dialect/KTDP/KTDP.td`: add one `KTDP_XOp` per intrinsic,
  structurally mirroring the corresponding `SpyreOp.td` entry (same element
  types, same arity, same result-shape rule) — this file and `SpyreOp.td`
  should be kept parallel on purpose, so a future `spyreop` addition is a
  mechanical `ktdp` addition.
- New transforms directory `include/ktir/Dialect/KTDP/Transforms/` +
  `lib/.../Transforms/` for `DecomposeKTDPCompute.cpp` — one
  `OpRewritePattern` per intrinsic, each the reference implementation in
  `arith`/`math` on tensors.
- `python/mlir_ktdp/dialects/`: Python bindings for the new ops, so
  `ktir_cpu` can parse and manipulate them.
- A tagged release both Triton's submodule pin and `ktir_cpu`'s dependency
  pin can advance to together.

### 9.2 Triton (this repo), once the submodule pin advances

- `python/triton/language/core.py` / `semantic.py`: add
  `tl.spyre_op(op_name, *args, **attrs)` as a `@builtin` (plus
  `@requires_backend("spyre")`); `semantic.py`'s implementation does a table
  lookup and emits `tts.spyre_op` directly — no tracing, no region, no
  insertion-point management.
- `third_party/spyre/triton_spyre.cc`: `create_spyre_op` builder — operands,
  name, attrs dictionary, declared result types; no region-opening.
- `third_party/spyre/include/Dialect/TTS/IR/TTSOps.td` +
  `lib/Dialect/TTS/IR/Ops.cpp`: add `TTS_SpyreOpOp` as specified in §4.
- **New pass**: `third_party/spyre/lib/Conversion/TritonToKTIR/ConvertSpyreOpToKTDP.cpp`
  (+ `Passes.td`/`Passes.h` entries) — mechanical `tts.spyre_op` →
  `ktdp.<name>` rewrite, one `OpRewritePattern` keyed by `name`. Wired into
  `buildTTIRToKTIRPipeline` (`lib/Pipeline.cpp`), after `LowerComputeOps`.
- **No new pass for the Spyre lowering direction.** For the elementwise
  intrinsics, add one `SelectKTDPOp<KTDPOp, SpyreOp>` pattern per op directly
  to the existing `third_party/spyre/lib/Transforms/LowerSpyreOps.cpp` (§6)
  — no pipeline edit needed, since that pass already runs after
  `ConvertElementwiseToLinalg` has scalarized `ktdp.gelu` for free. `exx2`/
  `topk`'s separate materialization-into-`linalg` problem (§6, §11.4) is
  still open and will need its own pass when tackled.
- CMake: new source files added to the relevant `CMakeLists.txt`s (see
  `ttir_only_build.md` for how the TTIR-only build flag affects what gets
  compiled, if any of the new passes need guarding against a GPU-backend
  build — unlikely, since this is entirely Spyre-only code, but worth a
  check during implementation).

### 9.3 `ktir_cpu`

- A new, mandatory pre-interpretation step: run `DecomposeKTDPCompute` (via
  `mlir_ktdp`) over any parsed module before interpreting it.
- Dependency bump to track the `ktir-mlir-frontend` commit that defines the
  new ops and decompositions, kept in lockstep with Triton's own submodule
  pin.
- New fixtures exercising the decomposition pass's numerical correctness
  directly.

## 10. Verifier and ownership considerations

**Verification** is purely structural at every stage: `TTS_SpyreOpOp`'s
verifier checks name/arity/dtype/attrs against the shared table; each
`ktdp.*` op's verifier is ordinary ODS type-checking, since there is no body
to validate. Nothing in this design needs an elementwise-only or
shape-preserving constraint anywhere.

**Ownership shifts in a way worth stating plainly, not glossing over.**
Today's natural model — a Spyre-maintaining engineer writes a `@triton.jit`
reference implementation, in Triton, in the same language kernels are
written in — does not apply here. Under this design:

- The **signature contract** (arity, dtypes, attributes, result shape) is
  owned by `ktir-mlir-frontend`'s `KTDP.td`, read by Triton's frontend
  through a query rather than duplicated into a second Python-side table.
- The **reference semantics** (what GELU's software fallback actually
  computes) is also owned by `ktir-mlir-frontend`, authored as an
  MLIR-rewriting C++ pattern, not as Triton source. Triton no longer
  carries, traces, or checks any fallback implementation at all.
- `ktir_cpu` owns nothing about *what* GELU means — only the obligation to
  run the decomposition pass before interpreting.

This is a genuine loss of authoring ergonomics for whoever maintains the
reference semantics (C++ pattern-matching code is harder to write, read,
and numerically sanity-check than a `@triton.jit` function), and it is a
real new coordination cost across repository boundaries that did not
previously exist in this form. It is also, in exchange, the thing that
makes the signature contract exist in exactly one place instead of two.

## 11. Future work and open questions

### 11.1 A dedicated `ktcompute` dialect, instead of extending `ktdp`

`ktdp`'s own description states its scope excludes compute ops entirely.
Extending it, as this proposal does for its initial implementation, is a
pragmatic choice — one dialect to register, no new build target, directly
reuses `ktdp`'s existing position as the dialect every KTIR consumer
already loads — but it is a real charter change, not a cosmetic one.

A future alternative moves every `ktdp.*` compute op into a new sibling
dialect, `ktcompute`, with no change to any op's own signature — purely a
namespace move (`ktdp.gelu` → `ktcompute.gelu`). Arguments for doing this
eventually:

- **Decoupled evolution.** `ktdp`'s memory/scheduling ops should be stable;
  the intrinsic set will churn as Spyre's hardware does (see §11.2). Keeping
  them in one dialect version-couples two things that change for unrelated
  reasons.
- **Narrower dependency surface.** Compute ops will eventually need
  `math`/`linalg` as dependent dialects; `ktdp` today needs neither. A
  sibling dialect keeps that dependency local to the few passes that
  actually need it, rather than pulling it into every `ktdp` consumer,
  including ones with no interest in intrinsics at all.
- **Preserves a useful structural proxy.** `FuseComputeAndDataMovement`
  already draws an explicit line between data-movement ops and compute ops.
  A dedicated compute dialect keeps "is this a `ktdp` op" a true signal for
  "is this *not* compute," which several passes in this tree already lean
  on structurally.

Arguments for *not* doing this now: it's a second dialect every
registration call site, pipeline, and doc has to know about, for no
behavioral difference on day one; and the migration cost (once there are a
dozen `ktdp.*` compute ops, moving all of them) only grows the longer it's
deferred, so this should be revisited deliberately rather than indefinitely,
not treated as free to delay forever.

### 11.2 Long-term SpyreOp granularity

If Spyre's own `spyreop` dialect is ever re-granularized (today's large,
dedicated ops recomposed from smaller primitives), every `ktdp.*` mirror
needs to track that change to stay a near-1:1 mirror. This is a new, ongoing
maintenance obligation this design introduces that did not exist when the
intrinsic's identity wasn't represented as a distinct op at all.

### 11.3 Extensibility: why LayerNorm, TopK, and the harder shapes fit cleanly

This is the point worth stating most directly, since it's the proposal's
strongest practical argument. Because a `ktdp.*` op is a plain declared
signature with no body to validate:

- **Multi-result ops** (`ktdp.exx2(x, axis) -> (mean, msq)`, LayerNorm's
  stage 0) are an ordinary two-result ODS op — nothing about having two
  results is special-cased anywhere in this design.
- **Multi-operand ops** (`ktdp.layernormnorm(x, mean, scale, weight, bias) ->
  y`) are an ordinary five-operand ODS op, for the same reason.
- **Operations with compile-time attributes** (`ktdp.softplus(x,
  beta: F32Attr, threshold: F32Attr)`, `ktdp.topk(x, k: I64Attr, axis:
  I64Attr)`) carry those attributes as ordinary ODS fields, read directly
  off the op by whichever pass needs them — `tl.spyre_op`'s `**attrs`
  threads them through from the frontend call with no separate mechanism.
- **TopK specifically**: `ktdp.topk(x, k, axis) -> (values, indices)` is
  representable as a single op despite being a sort/selection, not an
  elementwise computation, because nothing about `ktdp.*`'s definition
  requires shape preservation or elementwise structure. Representability is
  not the whole problem, though: unlike the elementwise cases in §6,
  `ktdp.topk` (and `exx2`) won't be scalarized by `ConvertElementwiseToLinalg`
  for free, so they still need a dedicated materialization step before any
  selection pattern applies — and SFP-ring inter-core communication and the
  core-count-dependent validity of `k` are hardware constraints this
  architecture does not solve by itself either way; see §11.4.

None of this required inventing a splitting mechanism, a call-site-matching
rule, or a region-shape restriction. Each op simply declares what it is.

### 11.4 Still unresolved by this architecture alone

- **TopK's hardware constraints.** Making `ktdp.topk` *representable* does
  not solve the SFP-ring/core-count validation problem — that still needs
  its own verifier design, reading a resolved core count, independent of
  this proposal.
- **DF16.** Untouched by this design; the element-type question is exactly
  as open as it was before.
- **Cross-repo version skew.** Nothing in this design prevents Triton's
  submodule pin and `ktir_cpu`'s dependency pin from disagreeing about which
  `ktdp.*` ops exist or how they decompose — that risk is structural to the
  three-repo split and should be named as an operational concern for
  whoever owns the release process, not assumed away.
- **No automated check that `ktir_cpu`'s decomposition and Spyre's real
  hardware semantics agree.** With no Triton-traced reference to compare
  against, a numerical divergence between `DecomposeKTDPCompute`'s output
  and real `spyreop.gelu` behavior could only be caught by a dedicated
  cross-repo numerical test — worth building deliberately, not assumed to
  fall out of this design for free.

## Related documents

- `lx-placement.md` — the scratchpad-pinning design (`tl.spyre_pin`/
  `tts.pin`) that LayerNorm's named intermediates (§5) are meant to use.
- `inter-tile-lowering-to-mem-view.md` — the existing precedent for a
  device-neutral `ktdp` op whose lowering legitimately differs by target
  (§1.3).
- `triton-ktir-lowering-thought-exercise.md` — the "bounded, local
  translation" framing this proposal's two new conversions are designed to
  fit (§2.2).
- `torch-spyre-integration.md` — how `ktir_cpu` is actually driven in
  practice, and where the new decomposition step (§7) attaches.
