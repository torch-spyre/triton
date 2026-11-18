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
        Input: lowered KTIR (after LowerDescriptorMemory, LowerScalarLoad and
        LowerComputeOps) still carrying `tts` marker ops. A marker is an op with
        no result whose content is entirely in its attributes; it exists to name
        a value.

        Lowering:
        - tts.tensor_layout (op) → the `tts.tensor_layout` ATTRIBUTE, a
          dictionary of the op's three phys_src / phys_op / phys_arg arrays, on
          the ktdp.construct_memory_view the marked descriptor resolved to. The
          marker is erased, and so is the builtin.unrealized_conversion_cast
          bridging that view's memref back to !tt.tensordesc, which the marker
          was the last user of.
        - tts.pin (op) → the `tts.pin` ATTRIBUTE, a dictionary of the op's
          memory_space and its address when it stated one, on the op DEFINING
          the pinned value. The marker is erased.

        The pass is one generic driver over any number of marker kinds: resolve
        the marked value to an op, set the attribute, erase the marker, then
        erase a bridge cast the marker's erasure left dead. What each marker
        supplies is how its operand resolves and which resolutions it admits —
        for `tts.tensor_layout` the lowered-descriptor bridge and nothing else,
        for `tts.pin` any value with a defining op. A marker whose operand does
        not resolve is an error, not a skip; for a pin that is a block argument,
        which has no op the annotation could live on.

        Ordering, one lower bound per marker:
        - after LowerDescriptorMemory, which creates both the memory view a
          layout attribute lands on and the bridge cast its operand resolves
          through;
        - after LowerComputeOps, because a pin's carrier is the op producing the
          pinned value and that op has to be its final form. A pinned reduction
          is a tt.reduce before this pass and a linalg.reduce after, and the
          attribute would be dropped with the op it replaced.

        Both markers stay legal through the passes in between. LowerComputeOps
        converts against an explicit illegal-op list that names no `tts` op, so
        a marker reaching it passes through untouched; LowerScalarLoad is
        indifferent to markers.

        And one UPPER bound, which is why this runs at the very end of the `ktir`
        stage rather than as early as those lower bounds allow: after that
        stage's closing canonicalize, because the conversion this pass performs
        is what makes a pinned value deletable.

        A marker op is not memory-effect-free, so DCE leaves it — and while it
        survives it USES the value it marks, which is what keeps that value
        alive. An attribute does not: it rides on the producer, and a producer
        whose results are otherwise unused is trivially dead. So converting
        before the DCE can delete the very value the pin asked to place, taking
        the annotation with it. A relayout's share is exactly that case: its only
        consumer is a compose that consumes it by erasing the marker.

        Running last costs nothing, because nothing in this stage READS an
        annotation — every consumer is in `spyrecode`, and the first pass there
        is MaterializePinnedBuffers, so no DCE runs between the write and the
        honouring.

        Output: no `tts` marker op remains anywhere in the module, which is also
        what lets the stage's artifact be parsed by a consumer that does not load
        this dialect. The annotations they carried are attributes — the layout on
        a ktdp op, which is the form [RewriteDescriptorLayoutGeneric] reads, and
        the pin on a compute op, which MaterializePinnedBuffers honours. IR
        carrying no marker is returned unchanged.
    }];

    // No dependentDialects: this pass constructs no op and no dialect
    // attribute. It writes a dictionary of builtin attributes under a `tts.`
    // name, which is a string -- see the note in TTSDialect.td on why the
    // value must stay builtin. Loading `tts` is what makes that name VERIFIED
    // rather than accepted unchecked, and whoever parsed the marker op has
    // already done it.
}

#endif // TTS_TRANSFORMS_PASSES
