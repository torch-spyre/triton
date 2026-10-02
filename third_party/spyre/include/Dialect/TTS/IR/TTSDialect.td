#ifndef TTS_DIALECT
#define TTS_DIALECT

include "mlir/IR/OpBase.td"

def TTS_Dialect : Dialect {
  let name = "tts";

  let cppNamespace = "::mlir::triton::tts";

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
  }];

  let hasOperationAttrVerify = 1;
}

#endif // TTS_DIALECT
