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

#endif // TTS_OPS
