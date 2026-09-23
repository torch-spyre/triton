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

#endif // KTDP_TRANSFORMS_PASSES
