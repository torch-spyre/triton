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

def RewriteDescriptorLayout : Pass<"rewrite-descriptor-layout", "mlir::ModuleOp"> {
    let summary = "Rewrite logical tensor descriptors to physical (stick-tiled) "
                  "layout from tt.spyre_tensor_layout annotations.";
    let description = [{
        Input: lowered KTDP IR (after LowerDescriptorMemory + LowerComputeOps).
        Each annotated descriptor has been lowered to:
          ktdp.construct_memory_view (logical shape/strides, wrapped in an
          UnrealizedConversionCast bridging back to !tt.tensordesc so the
          tt.spyre_tensor_layout marker's desc operand stays valid) →
          ktdp.construct_access_tile (logical block shape) →
          ktdp.load/store.
        The marker holds the physical layout as the OpSpec device_coordinates form
        (phys_src / phys_op / phys_arg i64 arrays, one entry per physical dim;
        op 0=identity, 1=floordiv, 2=mod).

        Runs in three phases (staged model):

        Phase 1 — physicalize each annotated descriptor:
        - rebuild ktdp.construct_memory_view with physical shape/strides/memref
          type: floordiv dims get extent ceil(N/div) (covering partial boundary
          sticks), mod dims get extent = modulus, identity dims are unchanged
          (e.g. [M,N] -> [ceil(N/64), M, 64]);
        - rebuild each ktdp.construct_access_tile with physical block shape +
          remapped index operands (identity / divsi / remsi);
        - retype ktdp.load result tensors and forward-retype the elementwise
          compute chain to the physical rank (stops at multi-tensor ops);
        - redirect ktdp.store's access tile operand to the physical tile.

        Phase 2 — decide each remaining op's OUTPUT, then rewrite it. Phase 2A
        resolves, for every value reachable from Phase 1's loads, the physical
        type it will carry; Phase 2B rewrites against those answers. An op whose
        output cannot be physicalized (matmul / batch_matmul, which contract
        exactly one K) keeps a logical result and gets a stick loop; one whose
        output can be is emitted at physical shape directly.

        For linalg.reduce that second case has a condition, since it decides
        WHICH physical shape: reducing a logical axis away deletes the physical
        dims sourced from it, so the operand's layout induces an output layout
        (surviving physical dims, in order, coord ops and args intact). When the
        descriptor the result is stored to declares exactly that layout, the
        reduce is emitted at it and a surviving stick-index dim becomes a batch
        dim of the one op — e.g. physical [2, 64, 64] reduced on the middle dim
        gives `outs tensor<2x64xf16> dimensions = [1]`, which ktdp.store consumes
        directly. Otherwise the result stays logical and the store's widen stage
        builds the physical form. See docs/spyre-tensor-layouts.md.

        Phase 3 — erase the tt.spyre_tensor_layout markers and the now-dead
        UnrealizedConversionCast bridges.

        Runs after LowerInterTile, so every inter-tile reduction is already in
        ktdp form; LowerInterTile rejects a kernel that combines a layout marker
        with an inter-tile reduction, since no propagation pattern covers it.

        Output: physical ktdp.construct_memory_view + access_tile + load/store;
        no tt.spyre_tensor_layout ops remain. Descriptors without an
        annotation are left untouched. A pure pointwise kernel exercises only
        Phase 1.
    }];

    let options = [
        Option<"dataLayout", "data-layout", "std::string",
               /*default=*/"\"device\"",
               "Format of tensors: \"device\" (stickified) or \"host\" (row-major)">
    ];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::ktdp::KtdpDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect"
    ];
}

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
