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

def PlacePinnedValues : Pass<"place-pinned-values", "mlir::ModuleOp"> {
    let summary = "Give each tts.pin'd value a buffer in the memory space its "
                  "pin names, and route its uses through it.";
    let description = [{
        Input: TTIR-with-ktdp, after LowerDescriptorMemory, carrying `tts.pin`
        markers. A pinned value is still a `tensor` produced by an arith/math op,
        a `tt.reduce` or a `ktdp.load`.

        Lowering: per `tts.pin`, one ktdp.construct_memory_view over the pinned
        value's own shape in the pin's memory space -- at the pinned address when
        there is one -- then a ktdp.store of the value through an access tile, and
        one access tile + ktdp.load per remaining use, each use rewritten to read
        the buffer. The marker is erased.

        Unlike `tts.tensor_layout`, this marker does NOT become an attribute, and
        that is forced rather than chosen: a value pin's only possible carrier is
        the op PRODUCING the value, and elementwise-to-linalg conversion replaces
        that op, which would drop a discardable attribute silently. So the buffer
        is built while the marker is still here.

        Position: after LowerDescriptorMemory, whose views this one is modelled on
        and whose `ktdp` ops must already be legal here; before LowerComputeOps,
        which knows nothing of `tts` and would fail the marker as unconverted. The
        same window LowerTTSMarkers runs in, and order against it does not matter:
        that pass reaches memory views built for DESCRIPTORS and this one builds
        views for INTERMEDIATES, which carry no layout annotation.

        Placing the intermediates nobody pinned is deliberately not here. That is
        every compute-to-compute edge, named or not, and it is a different pass's
        subject; this one does only what a marker asked for.

        For the same reason a pin with NO address is refused here although the op
        admits one: nothing in this tree allocates a buffer that is not a kernel
        argument, so an unaddressed pin names no location a view could be built
        at, and offset 0 would be a fabricated address rather than an allocation.
        Since an address is admitted only in `ct_local`, this is also why a
        `global` pin does not compile today.

        Output: no `tts.pin` remains. Every value one named is stored to, and read
        back from, a buffer in the named space.
    }];

    // dependentDialects, unlike LowerTTSMarkers: this pass constructs ktdp ops
    // (memory view, access tile, load, store) and an arith index cast, so both
    // dialects have to be LOADED and not merely loadable. `tts` itself is loaded
    // by whoever parsed the marker.
    let dependentDialects = [
        "::mlir::ktdp::KtdpDialect",
        "::mlir::arith::ArithDialect"
    ];

    // The grid is here for the same reason DistributeWork has it: a pinned
    // address may be affine in tl.program_id(0), and prod(grid) is what makes
    // that expression's address SET finite -- which is what capacity and overlap
    // are checked over.
    //
    // The overlap check is TEMPORARY, and the pass says so where it performs it:
    // reusing one scratchpad region for two values whose live ranges do not
    // overlap is legitimate, and refusing every intersection forbids it.  The two byte counts are hardware facts this tree does
    // not otherwise hold: `SpyreUtils.get_device_properties` states that the LX
    // scratchpad "is sized by the device description and allocated by the
    // scheduler", so they arrive as options rather than as literals in the pass,
    // and 0 means "do not check" rather than "zero bytes".
    let options = [
        ListOption<"grid", "grid", "int64_t",
                   "Per-axis partition of the Spyre hardware grid, as "
                   "DistributeWork takes it. Default [32].">,
        Option<"lxCapacityBytes", "lx-capacity-bytes", "int64_t",
               /*default=*/"2097152",
               "Bytes of per-core scratchpad a pinned range must fit in. The "
               "default is the capacity dbo-opt reports. 0 disables the check.">,
        Option<"alignmentBytes", "alignment-bytes", "int64_t",
               /*default=*/"128",
               "Byte alignment every pinned address must meet -- one stick. "
               "0 disables the check.">
    ];
}

#endif // TTS_TRANSFORMS_PASSES
