#ifndef KTDP_TRANSFORMS_PASSES
#define KTDP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Pass pipeline: TTIR → KTIR
//
//   LowerDescriptorMemory   [LowerPointerChainMemory]
//           └──────────────┬──────────────┘
//                          ↓
//                   LowerComputeOps
//                          |
//                   LowerInterTile    ← added for spyre (inter-tile collectives)
//                          |
//                   ConvertFunctions
//                          |
//                    DistributeWork
//                          |
//                   canonicalize + CSE
//                          |
//                        KTIR
//
// Each pass declares its contract below:
//   Input    — dialects/ops it expects to see
//   Lowering — ops it converts and what they become
//   Output   — what the IR looks like after this pass
// ============================================================================

def LowerDescriptorMemory : Pass<"lower-descriptor-memory", "mlir::ModuleOp"> {
    let summary = "Lower tt.descriptor_* ops to ktdp memory ops.";
    let description = [{
        Input: TTIR — triton, arith, scf dialects; !tt.ptr and !tt.tensordesc types.

        Lowering:
        - tt.descriptor_load/store → ktdp.construct_memory_view + ktdp.construct_access_tile + ktdp.load/store
        - tt.descriptor_gather/scatter → ktdp.construct_indirect_access_tile + ktdp.load/store
        - tt.make_tensor_descriptor — dead after uses are lowered, erased by cleanup

        Output: triton ops that this pass doesn't handle (tt.reduce, tt.broadcast,
        tt.expand_dims, tt.func, tt.return, tt.get_program_id) remain unchanged.
        New ktdp, memref, and arith ops are introduced.
    }];

    let dependentDialects = [
        "mlir::ktdp::KtdpDialect",
        "mlir::arith::ArithDialect",
        "mlir::memref::MemRefDialect"
    ];
}

def LowerComputeOps : Pass<"lower-compute-ops", "mlir::ModuleOp"> {
    let summary = "Lower tt compute/shape ops to linalg/tensor ops.";
    let description = [{
        Input: triton + ktdp + arith + memref + scf dialects.

        Lowering (Group A — shape manipulation):
        - tt.splat → linalg.fill
        - tt.reshape → tensor.reshape
        - tt.expand_dims → tensor.expand_shape
        - tt.broadcast → tensor.collapse_shape + linalg.broadcast
        - tt.trans → linalg.transpose
        - tt.join → tensor.expand_shape + tensor.concat
        - tt.split → tensor.extract_slice × 2

        Lowering (Group B — reduction):
        - tt.reduce (+ tt.reduce.return) → linalg.reduce

        Lowering (Group C — matrix multiply):
        - tt.dot → linalg.matmul

        Trivially dead ops left by prior passes are swept at the end.

        Output: remaining triton ops (tt.func, tt.return, tt.get_program_id)
        pass through unchanged.

    }];

    let dependentDialects = [
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect",
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect"
    ];
}

def LowerInterTile : Pass<"lower-inter-tile", "mlir::ModuleOp"> {
    let summary = "Lower tt.inter_tile_reduce to ktdp inter-tile produce/reduce pairs.";
    let description = [{
        Input: TTIR after LowerComputeOps — tt.inter_tile_reduce ops whose
        partial operands are already linalg/tensor values; enclosing func.func
        carries numWkSlicesPerDim and coreIdToWkSlice function attributes.

        Lowering:
        - tt.inter_tile_reduce (mode=all_reduce) →
            ktdp.inter_tile_produce (producer region + ktdp.yield_partial) +
            ktdp.inter_tile_reduce (full consumer set + ktdp.yield_reduced)
        - tt.inter_tile_reduce (mode=reduce_to_one) →
            ktdp.inter_tile_produce +
            ktdp.inter_tile_reduce (consumer = pick₀(g))
        - tt.inter_tile_reduce (mode=broadcast, reduce_scatter) →
            rejected with "not yet supported" diagnostic (P8/R8); delivery ops
            not yet available in ktir-mlir-frontend#25.
        - Axis with numWkSlicesPerDim[axis] == 1 → fold-away (erase op,
            forward partial unchanged).
        - depWkSlices present → producer_dependency_per_consumer affine set;
            absent → full-barrier (attribute omitted).

        Must run after LowerComputeOps (partials must be linalg/tensor) and
        before ConvertFunctions (reads function attributes ConvertFunctions
        would rewrite).

        Output: no tt.inter_tile_reduce ops remain; ktdp.inter_tile_produce /
        inter_tile_reduce pairs are introduced; the tile_future is single-use.
    }];

    let dependentDialects = [
        "mlir::ktdp::KtdpDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
    ];
}

def ConvertFunctions : Pass<"convert-ttir-functions", "mlir::ModuleOp"> {
    let summary = "Convert tt.func/tt.return to func.func/func.return and replace !tt.ptr args with index.";
    let description = [{
        Input: triton + ktdp + func + arith + linalg + tensor + memref + scf dialects.
        Must run after all memory passes — they consume !tt.ptr args via
        getBasePtrAsIndex before this pass replaces them.

        Lowering:
        - tt.func → func.func
        - tt.return → func.return
        - !tt.ptr function arguments → index

        Output: no triton dialect ops or types remain.
    }];

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::arith::ArithDialect"
    ];
}

def DistributeWork : Pass<"distribute-work-to-spyre-cores", "mlir::ModuleOp"> {
    let summary = "Replace tt.get_program_id with ktdp.get_compute_tile_id "
                  "and fold tt.get_num_programs against the grid.";
    let description = [{
        Rewrites each tt.get_program_id to a shared
        ktdp.get_compute_tile_id (variadic: one index per grid dim) +
        per-axis i32 casts, folds each tt.get_num_programs to an
        arith.constant grid[axis] : i32, and stamps a grid attribute on
        the enclosing function. Both rewrites share the same grid as a
        single source of truth.

        Enforces three invariants on the kernel's grid contract:
          (a) a function that reads tt.get_num_programs must also read
              tt.get_program_id (no grid-size-only kernels);
          (b) the set of axes read must be dense from 0 (no holes);
          (c) the caller-supplied grid rank must match the set of axes
              read.

        Trusts the kernel to distribute work itself via an explicit
        per-core loop. Does not synthesize a wrapping scf.for.
    }];

    let options = [
        ListOption<"grid", "grid", "int64_t",
                   "Per-axis partition of the Spyre hardware grid. One "
                   "entry per tl.program_id axis the kernel reads; "
                   "prod(grid) must equal the hardware core count. "
                   "Default [32] covers the common 1D case.">
    ];

    let dependentDialects = [
        "mlir::ktdp::KtdpDialect",
        "mlir::scf::SCFDialect",
        "mlir::arith::ArithDialect"
    ];
}


#endif // KTDP_TRANSFORMS_PASSES
