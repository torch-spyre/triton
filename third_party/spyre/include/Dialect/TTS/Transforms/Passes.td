#ifndef TTS_TRANSFORMS_PASSES
#define TTS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Transforms whose subject is the `tts` dialect's own abstraction: a marker op
// that names a value so that something can be said about it.  Not a conversion
// -- a marker's whole content is already an attribute, and what happens to it
// is that the attribute moves onto the op the named value resolves to.  For
// passes that translate `tt` into KTIR, see Conversion/TritonToKTIR/Passes.td;
// for passes about ktdp's memory-view model, Dialect/KTDP/Transforms/Passes.td.
//
// Each pass declares its contract below:
//   Input    - dialects/ops it expects to see
//   Lowering - ops it converts and what they become
//   Output   - what the IR looks like after this pass
//
// For the order the passes are actually run in, see buildTTIRToKTIRPipeline in
// third_party/spyre/lib/Pipeline.cpp.
// ============================================================================

def LowerTTSMarkers : Pass<"lower-tts-markers", "mlir::ModuleOp"> {
    let summary = "Move each tts marker op's annotation onto the op the marked "
                  "value resolves to, and erase the marker.";
    let description = [{
        Input: lowered KTIR (after LowerDescriptorMemory and LowerScalarLoad)
        still carrying `tts` marker ops. A marker is an op with no result whose
        content is entirely in its attributes; it exists to name a value.

        Lowering:
        - tts.tensor_layout (op) → the `tts.tensor_layout` ATTRIBUTE, a
          dictionary of the op's three phys_src / phys_op / phys_arg arrays, on
          the ktdp.construct_memory_view the marked descriptor resolved to. The
          marker is erased, and so is the builtin.unrealized_conversion_cast
          bridging that view's memref back to !tt.tensordesc, which the marker
          was the last user of.

        The pass is one generic driver over any number of marker kinds: resolve
        the marked value to an op, set the attribute, erase the marker, then
        erase a bridge cast the marker's erasure left dead. What each marker
        supplies is how its operand resolves and which resolutions it admits —
        for `tts.tensor_layout`, the lowered-descriptor bridge and nothing else.
        A marker whose operand does not resolve is an error, not a skip.

        Must run after LowerDescriptorMemory, which creates both the memory view
        the attribute lands on and the bridge cast the marker's operand resolves
        through; and before LowerComputeOps, which knows nothing of `tts` and
        would fail the marker as an unconverted op. LowerScalarLoad in between
        is indifferent — it only has to keep the marker legal.

        Output: no `tts` marker op remains anywhere in the module. The
        annotations they carried are attributes on ktdp ops, which is the form
        [RewriteDescriptorLayoutGeneric] reads. IR carrying no marker is
        returned unchanged.
    }];

    // No dependentDialects: this pass constructs no op and no dialect
    // attribute. It writes a dictionary of builtin attributes under a `tts.`
    // name, which is a string -- see the note in TTSDialect.td on why the
    // value must stay builtin. Loading `tts` is what makes that name VERIFIED
    // rather than accepted unchecked, and whoever parsed the marker op has
    // already done it.
}

#endif // TTS_TRANSFORMS_PASSES
