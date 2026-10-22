#ifndef TTS_DIALECT
#define TTS_DIALECT

include "mlir/IR/OpBase.td"

def TTS_Dialect : Dialect {
  let name = "tts";

  let cppNamespace = "::mlir::triton::tts";

  // A `tts.spyre_intrinsic` names the op its call site becomes, and the
  // verifiers check that the name is a registered op. Every target today is
  // spyreop's, so loading this dialect loads that one, wherever the request is
  // built or parsed.
  let dependentDialects = ["::mlir::spyreop::SpyreOpDialect"];

  let summary = "Triton-to-Spyre authoring annotations";

  let description = [{
    `tts` owns the annotations the Spyre backend authors on its way from Triton
    IR to KTIR — the things a kernel says about how it wants to be laid out,
    which are neither Triton's (they mean nothing to a GPU backend) nor KTIR's
    (they are consumed before any artifact is emitted).

    The dialect defines **no types and no attribute types**. What it owns is a
    *name* and the verifier that name gets: a discardable attribute whose prefix
    is a registered dialect is routed to that dialect's
    `verifyOperationAttribute` hook, on any op, at every verification point. An
    undialected name (`tensor_layout = ...`) gets no hook at all, and a name
    prefixed with a dialect that does not implement the hook (`ktdp.…`) is
    accepted unchecked. The prefix here is ours, so the checks are ours.

    Being only a name is also what lets the annotation ride in an artifact our
    consumers parse without loading this dialect. An attribute *name* is a
    string; anything that has to be **constructed** — an `AttrDef` value, a type
    parameter — needs its dialect loaded, and neither the KTIR scheduler nor
    `dbo-opt` loads ours. So the value of `tts.tensor_layout` is built out of
    builtin attributes only, and must stay that way.

    The two ops the dialect does define, `tts.tensor_layout` and `tts.pin`, are
    the *authoring* forms of two such annotations, and are bound by the same rule
    from the other side: both are fully consumed before any artifact is written,
    by `LowerTTSMarkers`, which turns each into its attribute. An op does have to
    be constructed, so one surviving into an artifact would fail at parse in a
    consumer that does not load us — which is why each op and its attribute are a
    before/after pair and never a choice of encoding for the same IR.

    Where the attribute lands is the marker's own business and differs between
    the two: a layout names a descriptor and lands on the memory view that
    descriptor became, a pin names a value and lands on the op producing it.
    What does not differ is that the value of either is built out of builtin
    attributes only.

    The third op, `tts.spyre_intrinsic`, is a request for one spyreop intrinsic
    and holds a body rather than naming a value, so it is not a marker; it is
    bound by the same rule all the same, and its attribute shares its name as
    theirs do. `LowerTTSMarkers` inlines the body and leaves the request as the
    `tts.spyre_intrinsic` attribute on every op it inlined.
  }];

  let extraClassDeclaration = [{
    /// The physical (stick-tiled) device layout of the tensor a
    /// `ktdp.construct_memory_view` addresses, as a `DictionaryAttr` of three
    /// parallel `DenseI64ArrayAttr` entries with one entry per PHYSICAL dim:
    ///
    ///   phys_src[k] : the logical dim physical dim k derives from
    ///   phys_op[k]  : 0 = identity, 1 = floordiv, 2 = mod, 3 = splat
    ///   phys_arg[k] : divisor (floordiv) / modulus (mod) / lane count (splat);
    ///                 ignored for identity
    ///
    /// e.g. [M, N] stick-on-N at width 64 is
    ///   phys_src = [1, 0, 1], phys_op = [1, 0, 2], phys_arg = [64, 0, 64]
    /// giving physical size [ceildiv(N, 64), M, 64]. A splat dim replicates its
    /// source logical dim across `phys_arg` lanes instead of partitioning it.
    ///
    /// Discardable, so nothing forces a pass that rebuilds the view to carry it
    /// — see the note on `RewriteDescriptorLayoutGeneric`'s owned-attribute set,
    /// which drops it deliberately and is the only pass allowed to.
    static constexpr ::llvm::StringLiteral kTensorLayoutAttrName =
        "tts.tensor_layout";

    /// The three entry names of that dictionary, in the order the arrays are
    /// read. Spelled here so the verifier and its readers cannot drift.
    static constexpr ::llvm::StringLiteral kPhysSrcName = "phys_src";
    static constexpr ::llvm::StringLiteral kPhysOpName = "phys_op";
    static constexpr ::llvm::StringLiteral kPhysArgName = "phys_arg";

    /// Where a pinned intermediate's buffer lives, as a `DictionaryAttr` on the
    /// op PRODUCING the value, with the two entries `tts.pin` carried as an op:
    ///
    ///   memory_space : a `ktdp::MemorySpaceKind` name, today only "ct_local",
    ///                  as a string for the consumer to symbolize
    ///   offset       : an element index from the base of this kernel's scratchpad
    ///                  allocation, as a single `i32` -- one offset, the same on
    ///                  every core, since the scratchpad is per-core and nothing
    ///                  wants a per-core spelling. Absent when the pin named no
    ///                  offset, which the op's verifier refuses today.
    ///
    /// `memory_space` is a NAME and not `#ktdp.memory_space`, and that is the
    /// dialect's builtin-value rule holding rather than an oversight -- see
    /// TTSOps.td for the harder constraint underneath it, which is that the op
    /// this dictionary is built from is created during tracing, where loading
    /// ktdp would load `func` and break the `ttir` stage's Inliner.
    ///
    /// So the consumer symbolizes the name. There is no `ct_id` to carry: a pin
    /// means the scratchpad of whichever core is running, never a named peer's, so
    /// a consumer builds a `#ktdp.memory_space` with `ct_id` unspecified (-1),
    /// which is what every site in this tree that constructs one passes --
    /// LowerDescriptorMemory and LowerScalarLoad both do.
    ///
    /// The producing op is the carrier because a pin names a VALUE, and a value's
    /// only op is the one defining it. So a value with no defining op has no
    /// attribute form, and `LowerTTSMarkers` is where that is reported -- not the
    /// op's verifier, which cannot tell a block argument the author named from one
    /// a fold put there. See the note in `PinOp::verify`.
    ///
    /// Discardable, and builtin-valued like the layout above, so a consumer that
    /// does not load this dialect still round-trips it.
    static constexpr ::llvm::StringLiteral kPinAttrName = "tts.pin";

    /// The two entry names of that dictionary.
    static constexpr ::llvm::StringLiteral kMemorySpaceName = "memory_space";
    static constexpr ::llvm::StringLiteral kOffsetName = "offset";

    /// Which `tts.spyre_intrinsic` call site an op was inlined from, as a
    /// `DictionaryAttr` on every op `LowerTTSMarkers` inlined out of that op's
    /// body:
    ///
    ///   target   : the op the call site names, the op's `target`, as the
    ///              string of a registered operation name
    ///   id       : an `i64` unique per `tts.spyre_intrinsic` in the module, so
    ///              two call sites of the same target stay distinguishable
    ///              after both are inlined into one function. It alone says
    ///              which call site an op is of: the ops of one call site carry
    ///              different dictionaries, since `operands` is per op
    ///   arity    : an `i64`, the call site's operand count, on every op of it
    ///   operands : on an op reading a call operand directly, and only there, a
    ///              `DenseI64ArrayAttr` with one entry per operand of the op it
    ///              sits on: the index of the call operand that operand IS, or
    ///              -1 when it is not one (a value computed in the body, or a
    ///              constant). E.g. `array<i64: 1, -1>` on the `arith.divf` of
    ///              `b / t`, `b` the call's operand 1
    ///
    /// `operands` is the call site's operand order, written when it is known
    /// exactly -- LowerTTSMarkers replacing block argument k with call operand
    /// k -- because nothing later can recover it: a generic's block arguments
    /// follow its `ins`, which ConvertElementwiseToLinalg orders by the first
    /// op's operands and fusion splices and deduplicates. It is per operand of
    /// an op, and what it says depends on the op:
    ///
    ///   - on a NON-COMMUTATIVE op, positions are exact. Every pass between
    ///     the tagging and selection keeps such an op's operand order:
    ///     ConvertElementwiseToLinalg gives operand j of the scalar op block
    ///     argument j, whose `ins` operand is the tensor op's operand j; fusion
    ///     and the layout rewrite clone body ops whole; and a rewrite replacing
    ///     a hinted op keeps lhs and rhs and copies its `tts` attributes.
    ///   - on a COMMUTATIVE op, the binding is a set: which call operands the
    ///     op reads, not at which positions. The one transformation that
    ///     permutes operands, the commutative trait fold any folding driver
    ///     applies, moves constants after the other operands, and fires only
    ///     on an op where operand order does not matter.
    ///
    /// LowerSpyreIntrinsicCalls reads it that way; see matchCallSite.
    ///
    /// On the ops and not on what they become: ConvertElementwiseToLinalg
    /// writes a tensor op's attributes onto the scalar op it puts in the body,
    /// and fusion and the layout rewrite clone body ops whole, while every
    /// pass that rebuilds a `linalg.generic` drops the generic's own
    /// attributes. FuseComputeAndDataMovement fuses two generics whose bodies
    /// carry one call site, and LowerSpyreIntrinsicCalls replaces the body the
    /// call site became with the op `target` names. What is checked about it,
    /// and where, is stated once, in `tts.spyre_intrinsic`'s description.
    ///
    /// Builtin-valued like the two above, so the `ktir` artifact carrying it
    /// parses in a consumer that does not load this dialect.
    static constexpr ::llvm::StringLiteral kSpyreIntrinsicAttrName =
        "tts.spyre_intrinsic";

    /// The entry names of that dictionary.
    static constexpr ::llvm::StringLiteral kSpyreIntrinsicTargetKey = "target";
    static constexpr ::llvm::StringLiteral kSpyreIntrinsicIdKey = "id";
    static constexpr ::llvm::StringLiteral kSpyreIntrinsicArityKey = "arity";
    static constexpr ::llvm::StringLiteral kSpyreIntrinsicOperandsKey =
        "operands";
  }];

  let hasOperationAttrVerify = 1;
}

#endif // TTS_DIALECT
