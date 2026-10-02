#ifndef KTDP_TRANSFORMS_PASSES
#define KTDP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Transforms whose subject is the KTDP dialect's own abstractions -- memory
// views and access tiles -- rather than a dialect boundary.  A pass here reads
// and rewrites ktdp ops in place; it does not translate one dialect into
// another.  For that, see Conversion/TritonToKTIR/Passes.td.
//
// Each pass declares its contract below:
//   Input    - dialects/ops it expects to see
//   Lowering - ops it converts and what they become
//   Output   - what the IR looks like after this pass
//
// For the order the passes are actually run in, see the two places that build
// the pipeline:
//   - add_convert_ttir_to_ktdp in third_party/spyre/triton_spyre.cc
//   - the `ktir` stage in third_party/spyre/backend/compiler.py
// ============================================================================

def RewriteDescriptorLayoutGeneric
    : Pass<"rewrite-descriptor-layout-generic", "mlir::ModuleOp"> {
  let summary = "Rewrite logical descriptors to physical (stick-tiled) layout "
                "for kernels whose compute ops are linalg.generic.";
  let description = [{
    Input: lowered KTDP IR (after LowerDescriptorMemory + LowerComputeOps +
    FoldDataMovementGenerics).  Each annotated descriptor has been lowered to:
      ktdp.construct_memory_view (logical shape / strides, wrapped in an
      UnrealizedConversionCast bridging back to !tt.tensordesc so the marker's
      desc operand stays valid) →
      ktdp.construct_access_tile (logical block shape) →
      ktdp.load / store.
    The marker holds the physical layout as the OpSpec device_coordinates form
    (phys_src / phys_op / phys_arg i64 arrays; op 0=identity, 1=floordiv,
    2=mod, 3=splat).  phys_src must name every logical dim at least once: a dim
    named by none would have its extent dropped from the physical type, and the
    pass declines such a marker rather than emitting a layout that cannot
    address the dim.

    Phase 1  Physicalize each annotated descriptor: rebuild
             construct_memory_view with physical sizes and with strides laid out
             row-major over them -- a physicalized view addresses stick-tiled
             device data, so the logical strides are not read -- rebuild each
             access tile with physical block shape and remapped subscripts,
             retype load results and record the physicalized view against the
             layout it was physicalized under.

    Phase 2  Rewrite the generics adjacent to those views.  For each recorded
             view, every generic that reads one of its loads or supplies one of
             its stores is restated over the physical loop domain: all its
             operands and results at their prescribed physical rank, and its
             indexing maps stated over that domain.  The domain refines the
             generic's iteration space — a stick split is a delinearization over
             the basis (ceildiv(extent, W), W), so a split logical dim becomes
             two loop dims, and an operand holding that dim whole carries the
             linearization `stick * W + lane` in its map.  A split adds loop dims
             and nothing else: no scf.for is emitted for it.  The domain splits
             any logical dim any operand splits; the recorded views are therefore
             the scope of the rewrite, and a generic on no physicalized chain is
             never examined.  Each generic is rewritten at most once; there is no
             fixpoint and no pattern driver.

    Phase 3  Erase the markers and their now-dead bridge casts.

    Unannotated kernels are left unchanged.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdp::KtdpDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def MaterializePinnedBuffers
    : Pass<"materialize-pinned-buffers", "mlir::ModuleOp"> {
  let summary = "Give each value carrying a tts.pin annotation a buffer in the "
                "memory space the annotation names, and route its uses through "
                "it.";
  let description = [{
    Input: KTIR at the head of the `spyrecode` stage, in which some ops carry the
    `tts.pin` attribute `LowerTTSMarkers` wrote there. A pinned value is a
    `tensor` produced by an arith/math op, a `linalg` op or a `ktdp.load`.

    Lowering: per annotated op, one `ktdp.construct_memory_view` over the pinned
    value's own shape at the annotation's offset and in its memory space, then a
    `ktdp.store` of the value through a whole-buffer access tile, and one access
    tile + `ktdp.load` per use, each use rewritten to read the buffer. The
    attribute is removed, so the pass is idempotent and nothing downstream sees a
    request that has already been honoured.

    Pinning therefore also SPLITS the producer from its consumers, which is what
    makes a pin the way to stop two pointwise ops fusing into one compute.

    Position: FIRST in `spyrecode`, ahead of NormalizeForDevice,
    DropReductionInitFill, ConvertElementwiseToLinalg and
    LinalgGeneralizeNamedOps. Each of those replaces an op that could be a pin's
    carrier, and a replaced op takes its discardable attributes with it silently.
    In this stage rather than in `ktir` for the reason Pipeline.h gives: a pin
    exists because the scheduler admits one compute per local schedule, which is
    a fact about dbo-opt and not about the IR, and a kernel that stops at `ktir`
    needs no buffer built at all.

    Placing the intermediates nobody pinned is deliberately not here. That is
    every compute-to-compute edge, named or not, and it is a different pass's
    subject; this one does only what an annotation asked for.

    An offset is counted from the base of whatever allocation the scratchpad
    allocator gives this kernel, and NOT from the start of the scratchpad. That is
    what decides which numeric rules exist here at all: only the ones that hold for
    any base, which are two. A range cannot be longer than a whole scratchpad, and
    two pins of one kernel share one base and so must not overlap.

    ALIGNMENT is therefore not among them, and its absence is a consequence of that
    reading rather than a relaxation. A stick-aligned offset is neither necessary
    nor sufficient for a stick-aligned buffer when the base is unknown here;
    aligning the base it hands out is the allocator's rule, and asserting a stick
    multiple would ask the author to guarantee something they cannot see either.

    The capacity rule is this pass's rather than the op's because its subject is a
    fact about the device that this tree does not hold --
    `SpyreUtils.get_device_properties` states that the LX scratchpad "is sized by
    the device description and allocated by the scheduler" -- so it arrives as an
    option. `lx-capacity-bytes` has NO default: a per-kernel budget is the caller's
    to state, and 0 means the capacity question is not asked rather than that it is
    answered with a number this pass invented.

    Output: no `tts.pin` attribute remains. Every value one named is stored to,
    and read back from, a buffer in the named space.
  }];

  let options = [
    Option<"lxCapacityBytes", "lx-capacity-bytes", "int64_t",
           /*default=*/"0",
           "Bytes of scratchpad a pin's range may reach, per core. 0 does not "
           "check.">
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdp::KtdpDialect"
  ];
}

#endif // KTDP_TRANSFORMS_PASSES
