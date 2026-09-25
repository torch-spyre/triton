#ifndef TTS_OPS
#define TTS_OPS

include "mlir/IR/OpBase.td"
include "Dialect/TTS/IR/TTSDialect.td"
// For TT_TensorDescType. This is the one direction the dependency may point:
// `tts` names a Triton type, Triton names nothing of ours. TritonTypes.td
// defines no op, so `-gen-op-decls` over this file still emits only ours.
include "triton/Dialect/Triton/IR/TritonTypes.td"
// For Ktdp_MemorySpaceAttr, which `tts.pin` types its memory space as. Same
// direction as above -- we name theirs, they name nothing of ours -- and neither
// file defines an op, so `-gen-op-decls` still emits only ours.
include "ktir/Dialect/KTDP/KTDPAttrs.td"

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
    its own uses. `LowerTTSMarkers` consumes it, writing the `tts.pin`
    **attribute** onto the op PRODUCING the pinned value -- so like
    `tts.tensor_layout` this is the authoring half of a before/after pair, and
    what differs is only where the attribute lands.

    The pin names a **value** and not a `!tt.tensordesc`. A descriptor takes its
    element type from the `!tt.ptr` it is based on, so giving one a scratchpad
    address would mean supplying a dtype from somewhere as well; a value has a
    type already.

    A block argument verifies here and is refused by the lowering. It is a value
    like any other -- a `tensor` is a value whichever way it was defined -- so
    nothing about it is ill formed; what it lacks is a defining op for the
    attribute to land on. The two halves are split that way on purpose:
    well-formedness is this op's question, and whether a carrier exists is the
    lowering's.

    `memory_space` is ktdp's own `#ktdp.memory_space`, not a string checked against
    its enum. The dialect still defines no attribute type of its own -- using one
    another dialect defines is a different thing, and the consumers that read this
    annotation out of the `ktir` artifact all load ktdp already, since the
    artifact's own memref types carry that attribute. Typing it makes a misspelled
    space a PARSE error rather than a verifier error, and carries `ct_id` as data
    instead of as a convention.

    `ct_id` must stay unspecified, which the attribute's own default already makes
    the easy spelling. A pin means the scratchpad of whichever core is running the
    kernel; `#ktdp.memory_space<ct_local, ct_id = 7>` would mean core 7's, which
    is a different request and one nothing here honours.

    Only `ct_local` is admitted: an intermediate
    in HBM is written as a `tl.make_tensor_descriptor` with an explicit store and
    load, which is lx-placement.md's first assumption, and nothing here allocates
    an anonymous global buffer. The parameter stays a memory space anyway, so the
    surface does not change shape if `global` ever becomes placeable.

    `address` is where in the scratchpad, as an **element index** and not a byte
    address, matching what `ktdp.construct_memory_view`'s `$offset` operand
    feeds. It is optional because the design's baseline is that the compiler
    places every intermediate and a pin is an override; no pass honours an
    unaddressed pin yet.

    It is an ATTRIBUTE and not an SSA operand, which is what lets the marker
    become an attribute on the op producing the pinned value -- an operand would
    have nowhere to go. Two spellings, and the difference is per-core:

      `address = 4096 : i32`               one address, the same on every core
      `address = array<i32: 4096, 6144>`   one per program id, positionally

    Both are numbers rather than an expression, so the set of addresses a pin
    occupies is known here and needs no matcher to recover: capacity, alignment
    and disjointness are all answerable by arithmetic on the attribute. An
    address computed in the IR would forfeit that, which is why the operand form
    is gone.

    An author writing an affine address states the arm per core rather than the
    coefficients. The array is therefore what a `lambda pid: ...` surface would
    evaluate to, and a consumer wanting `base + ctid*stride` recovers the two
    coefficients by asking whether the array is an arithmetic progression --
    a question about integers, not about op structure.

    Example:
    ```mlir
      tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<2x11x2049xf16>
    ```
  }];

  let arguments = (ins
    AnyStaticShapeTensor:$value,
    Ktdp_MemorySpaceAttr:$memory_space,
    OptionalAttr<AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>>:$address
  );

  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}

#endif // TTS_OPS
