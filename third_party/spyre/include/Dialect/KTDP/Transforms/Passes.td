#ifndef KTDP_TRANSFORMS_PASSES
#define KTDP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Pass pipeline: TTIR → KTIR
//
//   LowerDescriptorMemory   [LowerPointerChainMemory]
//           └───────────┬───────────┘
//                       ↓
//                LowerScalarLoad
//                       ↓
//                LowerComputeOps
//                       ↓
//          RewriteDescriptorLayout   (spyre_tensor_layout-annotated
//                       ↓             descriptors only; otherwise a no-op)
//                ConvertFunctions
//                       ↓
//               DistributeWork
//                       ↓
//              canonicalize + CSE
//                       ↓
//                     KTIR
//
// RewriteDescriptorLayout runs LAST (before ConvertFunctions) so that tt.dot
// is already linalg.matmul before operands are physicalized. The marker
// (tt.spyre_tensor_layout) persists through LowerDescriptorMemory +
// LowerComputeOps via the UnrealizedConversionCast bridge left by Walk 1; both
// passes mark the marker legal so it is never flagged as unconverted.
//
// Each pass declares its contract below:
//   Input    — dialects/ops it expects to see
//   Lowering — ops it converts and what they become
//   Output   — what the IR looks like after this pass
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

        Phase 2 — synthesize loop nests for ops whose physical dims need their
        own iteration:
        - source (matmul): physicalize INPUTS, emit an scf.for nest extracting
          2-D slices + linalg.matmul (K-stick dims drive a reduction loop);
          the matmul OUTPUT stays logical;
        - sink (store, only when the output descriptor is annotated): consume the
          logical value, emit an scf.for nest scattering it into the physical
          output buffer via tensor.insert_slice.

        Phase 3 — erase the tt.spyre_tensor_layout markers and the now-dead
        UnrealizedConversionCast bridges.

        Output: physical ktdp.construct_memory_view + access_tile + load/store,
        with scf.for / linalg.matmul / tensor.insert_slice loop nests where
        required; no tt.spyre_tensor_layout ops remain. Descriptors without an
        annotation are left untouched. A pure pointwise kernel exercises only
        Phase 1.
    }];

    let options = [
        Option<"hwDataLayout", "hw-data-layout", "bool", "false",
               "When true, compute physical strides as row-major over the "
               "physical shape (hardware layout mode). When false (default), "
               "derive strides from logical strides via the coord map "
               "(logical layout mode).">
    ];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::ktdp::KtdpDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect"
    ];
}

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

def LowerScalarLoad : Pass<"lower-scalar-load", "mlir::ModuleOp"> {
    let summary = "Lower a scalar tt.load (pointer in, scalar result) to ktdp memory ops.";
    let description = [{
        Input: TTIR — triton, arith dialects; !tt.ptr types. Matches only a
        tt.load whose pointer operand is scalar (!tt.ptr<ElemT>, not a
        tensor of pointers) — the tensor-of-pointers form is out of scope
        and left legal for a future [LowerPointerChainMemory].

        Lowering:
        - scalar tt.addptr chain feeding the matched tt.load → folded into
          a single index offset (plain adds, no element-size scaling)
        - scalar tt.load → rank-0 ktdp.construct_memory_view +
          rank-0 ktdp.construct_access_tile + ktdp.load + tensor.extract
        - masked scalar tt.load → the mask must be a compile-time constant
          (a materialized arith.constant i1); constant-true drops straight
          through to the unconditional read, constant-false skips the read
          and yields `other` or a materialized zero of the element type.
          Spyre has no runtime control-flow divergence, so a mask that
          might depend on runtime data — including a comparison of two
          constants — is refused with a diagnostic and fails the pass; no
          scf.if or other runtime branch on the mask is ever emitted.
        - tt.addptr ops left dead by the fold are swept by cleanup

        Output: tensor-of-pointers tt.load, and all other triton ops this
        pass doesn't handle, remain unchanged. New ktdp, arith, and tensor
        ops are introduced.
    }];

    let dependentDialects = [
        "mlir::ktdp::KtdpDialect",
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect"
    ];
}

def LowerComputeOps : Pass<"lower-compute-ops", "mlir::ModuleOp"> {
    let summary = "Lower tt compute/shape ops to linalg/tensor ops.";
    let description = [{
        Input: triton + ktdp + arith + memref + tensor + scf dialects.

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
