#ifndef KTDP_TRANSFORMS_PASSES
#define KTDP_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Passes used to lower TTIR → KTIR.
//
// Each pass declares its contract below:
//   Input    — dialects/ops it expects to see
//   Lowering — ops it converts and what they become
//   Output   — what the IR looks like after this pass
//
// Ordering constraints are stated per-pass in the descriptions below, since
// they are properties of the individual pass. For the order the passes are
// actually run in, see the two places that build the pipeline:
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

        Phase 3 — erase the tt.spyre_tensor_layout markers and the now-dead
        UnrealizedConversionCast bridges.

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

def UnaliasLinalgOuts : Pass<"unalias-linalg-outs", "mlir::ModuleOp"> {
    let summary = "Replace a linalg 'outs' operand that aliases an 'ins' operand "
                  "with a fresh tensor.empty.";
    let description = [{
        Input: any IR containing linalg ops on ranked tensors. Matches the
        LinalgOp interface, so both named ops (linalg.add, linalg.mul, …) and
        linalg.generic are covered.

        Lowering:
        - linalg op whose outs[i] is the same SSA value as one of its ins
          operands, and whose body does not read that operand's initial value
          → outs[i] replaced with a fresh tensor.empty of the same type.
        - Same aliasing, but the body does read the initial value (an
          accumulating op such as linalg.reduce or linalg.matmul) → rejected
          with a diagnostic; the initial value is live, so no replacement
          preserves the result.
        - Same aliasing on a non-tensor or dynamically shaped outs → rejected
          with a diagnostic.

        A linalg op is in destination-passing style: its outs operands supply
        both the result shape and the initial value of each output element.
        tensor.empty preserves the shape and drops the initial value, which is
        why the rewrite is only applied where that value is provably unread.

        Motivation: upstream MLIR's ConvertElementwiseToLinalg reuses the first
        type-matching operand as outs rather than creating a fresh one, with no
        flag to suppress it. The Spyre dataflow-scheduler maps each operand of a
        compute op to a freshly loaded value, so a value appearing in both ins
        and outs is mapped twice and the second mapping overwrites the first,
        producing invalid SSA ordering.

        Ordering: must run after any pass that can introduce the aliasing —
        ConvertElementwiseToLinalg is the known source, so running after it
        covers the current pipeline. Independent of ConvertFunctions: this pass
        reads neither !tt.ptr values nor function attributes.

        Output: no linalg op has an outs operand that also appears in its ins.
        New tensor.empty ops are introduced.
    }];

    let dependentDialects = [
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
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
            rejected with "not yet supported" diagnostic.
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

def MaterializeBaseAddresses
    : Pass<"materialize-base-addresses", "mlir::ModuleOp"> {
    let summary = "Replace base-address function arguments with arith.constant "
                  "and drop them from the signature.";
    let description = [{
        Input: func.func after ConvertFunctions — base addresses are `index`
        arguments of the entry block. Must run after ConvertFunctions: before
        it the arguments are still !tt.ptr and the memory passes still see
        unrealized_conversion_cast placeholders.

        Lowering:
        - the i-th `index` argument of the entry block, for each entry i of
          baseAddresses, → arith.constant baseAddresses[i] : index, with the
          argument erased from the signature

        The address list is positional over the entry block's `index`
        arguments, in scan order — not over all arguments. ConvertFunctions
        retypes only !tt.ptr arguments to `index`, so after it runs the `index`
        arguments are exactly the base addresses. An interleaved signature such
        as `(%a: index, %n: i32, %b: index)` therefore assigns the two
        addresses to %a and %b, never to the i32 runtime scalar %n. Argument
        names and attributes are already gone by this point (ConvertFunctions
        clones from the FunctionType), so position is the only available key.

        Values are ELEMENT indices, not byte addresses: `$offset` on
        ktdp.construct_memory_view feeds MemRef.base_ptr, which ktir-cpu
        derives the byte position from as `base_ptr * bytes_per_elem(dtype)`
        (ktir_cpu/ir_types.py:84-87). No scaling is applied here — the integer
        supplied is emitted as-is. Scaling could not be done in this pass in
        any case, since element size is per-descriptor and only the function
        signature is visible here.

        Negative addresses, and a list longer than the `index`-argument count,
        are rejected with a diagnostic. A shorter list is legal — it
        materializes a prefix and leaves the rest as arguments.

        Applies to public functions only; a private helper is not a kernel
        boundary. With an empty list the pass is a no-op — it is opt-in, run
        only via required_fixes, and is in neither _CORE_PIPELINE_PASSES nor
        the fused add_convert_ttir_to_ktdp binding.

        Output: func.func whose base-address arguments are gone; every other
        argument survives, keeping relative order. Arity zero is not
        guaranteed — a kernel with a runtime scalar keeps it.
    }];

    let options = [
        ListOption<"baseAddresses", "base-addresses", "int64_t",
                   "Fixed HBM base addresses, as element indices, for the "
                   "kernel's base-address "
                   "arguments, positionally: entry i is the address for the "
                   "i-th `index` argument of the entry block. May be shorter "
                   "than the `index`-argument count (materializes a prefix); "
                   "longer is an error. Empty (default) makes the pass a "
                   "no-op.">
    ];

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::arith::ArithDialect"
    ];
}


#endif // KTDP_TRANSFORMS_PASSES
