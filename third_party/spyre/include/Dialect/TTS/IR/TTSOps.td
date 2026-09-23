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

def TTS_InterTileReduceOp : TTS_Op<"inter_tile_reduce", [
    AttrSizedOperandSegments
]> {
  let summary = "Cross-tile partial-combination metadata op";

  let description = [{
    `tts.inter_tile_reduce` names the intent of combining per-tile partial
    values across the cooperating tiles identified by `axis`. The
    `LowerInterTile` pass expands it into a `ktdp.inter_tile_produce` +
    delivery op pair. No result; uses are replaced by the delivery op results.

    Attributes:
    - `axis`              : work-slice dim name (key into `numWkSlicesPerDim`).
    - `mode`              : one of `all_reduce`, `reduce_to_one`,
                            `reduce_scatter`, `broadcast`.
    - `combiner`          : shorthand `"add"` / `"max"` / `"mul"`, or `""` for
                            a custom reducer region. `broadcast` uses `""`.
    - `scatter_dimension` : i64, required iff `mode = reduce_scatter`.

    Operands:
    - `partials`  : A per-tile partial tensors (already lowered to linalg/tensor
                    values by LowerComputeOps; A >= 1).
    - `identities`: A explicit identity tensors, required iff combiner is `""`
                    (region form). Empty for shorthand combiners.

    The op may carry a combiner region (2A block args → A yielded values via
    `tt.reduce.return`) when `combiner` is `""`.

    Like `tts.tensor_layout`, the op is fully consumed before any artifact is
    written, so it never has to be parsed by a consumer that does not load this
    dialect.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$partials,
    Variadic<AnyTensor>:$identities,
    StrAttr:$axis,
    StrAttr:$mode,
    StrAttr:$combiner,
    OptionalAttr<I64Attr>:$scatter_dimension,
    // Work-slice op attributes (from tl.inter_tile work_slices/dep_work_slices
    // constexprs). The frontend derives W from C and stores both on the op so
    // LowerInterTile can read them without consulting the enclosing tt.func.
    DictionaryAttr:$numWkSlicesPerDim,        // W: axis → count
    ArrayAttr:$coreIdToWkSlice,               // C: per-tile {axis → slice-idx}
    OptionalAttr<DictionaryAttr>:$depWkSlices // D: consumer-local → [prod-local]
  );
  let results = (outs Variadic<AnyTensor>:$results);
  let regions = (region AnyRegion:$combiner_region);

  let assemblyFormat = [{
    `partials` `(` $partials `:` type($partials) `)`
    `identities` `(` $identities `:` type($identities) `)`
    `axis` `=` $axis `mode` `=` $mode `combiner` `=` $combiner
    (`scatter_dimension` `=` $scatter_dimension^)?
    attr-dict `->` `(` type($results) `)`
    ($combiner_region^)?
  }];
}

#endif // TTS_OPS
