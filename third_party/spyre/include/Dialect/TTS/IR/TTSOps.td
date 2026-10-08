#ifndef TTS_OPS
#define TTS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Dialect/TTS/IR/TTSDialect.td"
// For TT_TensorDescType. This is the one direction the dependency may point:
// `tts` names a Triton type, Triton names nothing of ours. TritonTypes.td
// defines no op, so `-gen-op-decls` over this file still emits only ours.
include "triton/Dialect/Triton/IR/TritonTypes.td"

class TTS_Op<string mnemonic, list<Trait> traits = []>
    : Op<TTS_Dialect, mnemonic, traits>;

def TTS_TensorLayoutOp : TTS_Op<"tensor_layout", []> {
  let summary = "Annotate a tensor descriptor with its physical device layout";

  let description = [{
    `tts.tensor_layout` is the **authoring** spelling of the physical
    (stick-tiled) device layout for a `tt.make_tensor_descriptor`: three
    parallel i64 array attributes (`phys_src` / `phys_op` / `phys_arg`), one
    entry per physical dim, in the OpSpec `device_coordinates` form. See the
    `tts` dialect's `kTensorLayoutAttrName` note for what the three arrays mean.

    No result; the op is a marker. `LowerTTSMarkers` consumes it: the three
    arrays are written onto the `ktdp.construct_memory_view` built for the
    descriptor as the `tts.tensor_layout` **attribute**, and the op is erased
    along with the `unrealized_conversion_cast` that bridged the view's memref
    back to `!tt.tensordesc` for this operand's sake. The view and that cast are
    `LowerDescriptorMemory`'s, which is why this marker cannot be lowered before
    that pass has run.

    So the dialect spells the same layout twice, and the two spellings are a
    before/after pair rather than alternatives:

      op        — what a kernel author writes, on a `!tt.tensordesc`
      attribute — what the lowered IR carries, on the memory view

    The structural rules on the three arrays are the attribute's rules, checked
    by the same `tts::verifyTensorLayoutArrays`. The only difference is where the
    logical rank comes from: here it is the descriptor's block type, there it is
    the memory view's memref. Same rank, different source.
  }];

  let arguments = (ins
    TT_TensorDescType:$desc,
    DenseI64ArrayAttr:$phys_src,
    DenseI64ArrayAttr:$phys_op,
    DenseI64ArrayAttr:$phys_arg
  );

  let assemblyFormat = "$desc attr-dict `:` type($desc)";
  let hasVerifier = 1;
}

def TTS_PinOp : TTS_Op<"pin", []> {
  let summary = "Place a named value's buffer in a chosen memory space";

  let description = [{
    `tts.pin` is the **authoring** spelling of where an intermediate's buffer
    lives. The scheduler admits one compute per local schedule, so a value handed
    from one compute to the next has to go through memory; this op is how a
    kernel says *which* memory, and for the scratchpad, where in it.

    No result; the op is a marker, and the value it names keeps its own type and
    its own uses. `LowerTTSMarkers` consumes it, writing the `tts.pin`
    **attribute** onto the op PRODUCING the pinned value -- so like
    `tts.tensor_layout` this is the authoring half of a before/after pair, and
    what differs is only where the attribute lands.

    The pin names a **value** and not a `!tt.tensordesc`: a descriptor takes its
    element type from the `!tt.ptr` it is based on, so giving one a scratchpad
    location would mean supplying a dtype as well, and a value has a type already.

    `memory_space` is a `ktdp::MemorySpaceKind` NAME held as a string, and this is
    the one field whose spelling is forced rather than chosen. Building
    `#ktdp.memory_space` needs ktdp LOADED; ktdp declares `mlir::func::FuncDialect`
    dependent (KTDPDialect.td), and loading `func` makes it promise a
    `DialectInlinerInterface` nothing in this tree registers, so the `ttir` stage's
    Inliner aborts the compile. This op is built during TRACING, so the one place
    that attribute would be constructed is the one place ktdp must not be loaded.
    `MaterializePinnedBuffers` symbolizes the name in `spyrecode`, where the dialect
    is loaded anyway. A string also cannot carry a `ct_id`, which is the right shape:
    a pin means whichever core runs the kernel, so `ct_id = 7` is not expressible
    rather than refused.

    Only `ct_local` is admitted -- an intermediate in HBM is written as a
    `tl.make_tensor_descriptor` with an explicit store and load, and nothing here
    allocates an anonymous global buffer -- and the field stays a memory space so the
    surface does not change shape if `global` ever becomes placeable.

    `offset` is an **element index**, the unit `ktdp.construct_memory_view`'s
    `$offset` operand feeds, counted from the base of whatever allocation the
    scratchpad allocator gives this kernel and NOT from the start of the scratchpad.
    That decides which rules exist: alignment is the allocator's, since it aligns the
    base it hands out, and what survives is what holds for any base -- offset plus
    extent within one scratchpad, and no two pins of one kernel overlapping. Both are
    `MaterializePinnedBuffers`'.

    Optional in ODS and REQUIRED by the verifier, which is the honest statement of
    what exists: an offsetless pin is the design's baseline, since the compiler places
    every intermediate and a pin only overrides where, but nothing here can act on one
    -- no placement analysis, and nothing allocates a buffer that is not a kernel
    argument. Optional is what lets that refusal lift without the surface changing.

    An ATTRIBUTE and not an operand, which is what lets the marker become an
    attribute on the producing op -- an operand would have nowhere to go -- and a
    single `i32`, so the region a pin occupies is known here by arithmetic. ONE
    offset for every core, with no per-core spelling, because the scratchpad is
    per-core: an offset varying with the program id would move each core's own value
    within its own scratchpad rather than spread a tensor across cores, and every
    core would then reserve the hull of the set. A per-core form stays additive if
    something ever wants it. An affine offset could not come back either way, since
    `tl.program_id(0)` is a run-time read until DistributeWork folds it against the
    grid, so nothing at trace time can evaluate `base + pid*stride` into an attribute.

    Example:
    ```mlir
      tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<11x2049xf16>
    ```
  }];

  let arguments = (ins
    AnyStaticShapeTensor:$value,
    StrAttr:$memory_space,
    OptionalAttr<I32Attr>:$offset
  );

  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}

def TTS_SpyreOpOp : TTS_Op<"spyre_op", [
    IsolatedFromAbove, SingleBlockImplicitTerminator<"SpyreOpYieldOp">,
    RecursiveMemoryEffects]> {
  let summary = "Request one spyreop intrinsic, carrying its fallback body";

  let description = [{
    `tts.spyre_op` is the **authoring** spelling of `tl.spyre_op(name, *args)`:
    a request that the computation be done by the spyreop intrinsic `name`. The
    region holds that computation's FALLBACK -- the backend's registered
    `@triton.jit` body for `name`, traced at the operands' types -- so the op
    means the same thing whether or not the intrinsic is ever selected.

    The block arguments are the operands, one for one and of the same types, and
    the body ends in `tts.spyreop_yield`, whose operands are the results.

    RESULT TYPES ARE DECLARED, not inferred from the body. The op is created
    before its region is traced -- a region belongs to an op -- so the frontend
    takes the result types from the registry entry for `name` and refuses a
    fallback whose returned types differ, at the kernel line. The verifier states
    the same rule over the IR: the yielded types are the result types.

    `IsolatedFromAbove`, so the body reads nothing but its block arguments: a
    value the kernel computed outside reaches the fallback only as an operand,
    and a constant the canonicalizer folds stays inside. No memory op may appear
    in the body -- the region is a computation on values, and the intrinsic it
    stands for reads and writes no memory. The one op admitted without that
    check is a call: tracing reaches the registered fallback through a `tt.call`,
    which the `ttir` stage's inliner replaces with the callee's body.

    Like the marker ops this is consumed before any artifact is written:
    `LowerTTSMarkers` inlines the body in place and tags every op it inlined
    with the `tts.spyreop_hint` attribute, so the `ktir` artifact is the fallback alone,
    in dialects any consumer loads. See `kSpyreopHintAttrName` for what the tag
    carries and who reads it.

    Example:
    ```mlir
      %y = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
      ^bb0(%a: tensor<128xf32>):
        ...
        tts.spyreop_yield %r : tensor<128xf32>
      }
    ```
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyRankedTensor>:$inputs);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $name `(` $inputs `)` attr-dict `:` functional-type($inputs, $results)
    $body
  }];
  let hasRegionVerifier = 1;
}

def TTS_SpyreOpYieldOp : TTS_Op<"spyreop_yield", [
    Pure, Terminator, HasParent<"SpyreOpOp">]> {
  let summary = "Terminate a tts.spyre_op body, yielding its results";

  let arguments = (ins Variadic<AnyRankedTensor>:$values);

  // The no-operand builder SingleBlockImplicitTerminator requires.
  let builders = [OpBuilder<(ins), [{ /* no values */ }]>];

  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

#endif // TTS_OPS
