#ifndef TTS_OPS
#define TTS_OPS

include "mlir/IR/OpBase.td"
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

    No result; the op is a marker. `LowerDescriptorMemory` consumes it: the
    three arrays are written onto the `ktdp.construct_memory_view` built for the
    descriptor as the `tts.tensor_layout` **attribute**, and the op is erased
    along with the `unrealized_conversion_cast` that bridged the view's memref
    back to `!tt.tensordesc` for this operand's sake.

    So the dialect spells the same layout twice, and the two spellings are a
    before/after pair rather than alternatives:

      op        — what a kernel author writes, on a `!tt.tensordesc`
      attribute — what the lowered IR carries, on the memory view

    The discriminator is structural: a descriptor annotated with this op is
    lowered to the attribute, while one annotated with `tt.spyre_tensor_layout`
    keeps the older shape (op and bridge cast both survive) and is consumed by
    the named `rewrite-descriptor-layout` pass instead. No flag selects between
    them.

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

  // Deliberately the same shape as tt.spyre_tensor_layout's, so the two are
  // legible side by side while both exist.
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
    its own uses. `PlacePinnedValues` consumes it, and unlike
    `tts.tensor_layout` it does not become an attribute -- see that pass for why.

    The pin names a **value** and not a `!tt.tensordesc`. A descriptor takes its
    element type from the `!tt.ptr` it is based on, so giving one a scratchpad
    address would mean supplying a dtype from somewhere as well; a value has a
    type already.

    Only a value the author can point at may be pinned -- `y = tl.exp(x)` can be
    pinned, `tl.exp(x)` inside a larger expression cannot -- and that is enforced
    by the SURFACE rather than here: an anonymous subexpression has no name to
    pass to `tl.spyre_pin`. So a block argument is pinnable like any other value.
    A `tensor` is a value whichever way it was defined, and receiving one is the
    same state a `ktdp.load` result is in; placing the intermediates nobody named
    is a separate job and not this op's.

    `memory_space` is a plain string, checked against `ktdp::MemorySpaceKind` by
    the verifier rather than typed as `#ktdp.memory_space`. The dialect defines no
    attribute type on purpose -- see the note in TTSDialect.td -- and this keeps
    that true for its second marker. Only `ct_local` is admitted: an intermediate
    in HBM is written as a `tl.make_tensor_descriptor` with an explicit store and
    load, which is lx-placement.md's first assumption, and nothing here allocates
    an anonymous global buffer. The parameter stays a memory space anyway, so the
    surface does not change shape if `global` ever becomes placeable.

    `address` is where in the scratchpad, as an **element index** and not a byte
    address, matching what `ktdp.construct_memory_view`'s `$offset` operand
    feeds. It is optional because the design's baseline is that the compiler
    places every intermediate and a pin is an override; no pass honours an
    unaddressed pin yet. It is an SSA operand rather than an attribute because that is the form
    the offset is consumed in, and `i32` rather than `index` because that is what
    `tl.program_id` yields, which keeps the verifier's structural walk free of
    casts. Two forms are admitted and nothing else:

      constant                 one address, the same on every core
      base + pid * stride      affine in the program id, constant coefficients

    The restriction is what keeps a pin's address SET finite and enumerable, and
    so keeps capacity, alignment and disjointness answerable about it. An address
    read from memory forfeits all three.

    Example:
    ```mlir
      %c4096 = arith.constant 4096 : i32
      tts.pin %e, address %c4096 {memory_space = "ct_local"} : tensor<2x11x2049xf16>
    ```
  }];

  let arguments = (ins
    AnyStaticShapeTensor:$value,
    Optional<I32>:$address,
    StrAttr:$memory_space
  );

  let assemblyFormat = "$value (`,` `address` $address^)? attr-dict `:` type($value)";
  let hasVerifier = 1;
}

#endif // TTS_OPS
