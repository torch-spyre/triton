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
    ///   memory_space : a `ktdp::MemorySpaceKind` name, today only "ct_local"
    ///   address      : an element index -- one `i32` for an address that is the
    ///                  same on every core, or a `DenseI32ArrayAttr` holding one
    ///                  per program id. Absent when the pin named no address.
    ///
    /// `memory_space` is ktdp's own attribute and not a string naming its enum,
    /// which is allowed for a reason worth stating: the rule this dialect lives
    /// by is that nothing here needs THIS dialect loaded to parse, and ktdp is
    /// not this dialect. Every consumer that reads this annotation out of the
    /// `ktir` artifact loads ktdp already, because the artifact's own memref
    /// types carry `#ktdp.memory_space`. A `tts.`-defined attribute type would
    /// break the rule; borrowing one does not.
    ///
    /// Its `ct_id` must be unspecified, which the attribute defaults to (-1), and
    /// which is what every site in this tree that constructs one passes --
    /// LowerDescriptorMemory and LowerScalarLoad both do. A pin means the
    /// scratchpad of whichever core is running, never a named peer's, so a
    /// consumer can use this attribute as it stands rather than rebuilding it.
    ///
    /// The producing op is the carrier because a pin names a VALUE, and a value's
    /// only op is the one defining it. That is also why a pinned block argument
    /// has no attribute form: see `LowerTTSMarkers`, which refuses one.
    ///
    /// Discardable, and builtin-valued like the layout above, so a consumer that
    /// does not load this dialect still round-trips it.
    static constexpr ::llvm::StringLiteral kPinAttrName = "tts.pin";

    /// The two entry names of that dictionary.
    static constexpr ::llvm::StringLiteral kMemorySpaceName = "memory_space";
    static constexpr ::llvm::StringLiteral kAddressName = "address";
  }];

  let hasOperationAttrVerify = 1;
}

#endif // TTS_DIALECT
