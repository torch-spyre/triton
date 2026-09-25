#ifndef TRITON_TO_KTIR_PASSES
#define TRITON_TO_KTIR_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// TTIR -> KTIR conversions.
//
// Every pass here has a source dialect and a target dialect: it consumes `tt.*`
// and produces something else.  That is the criterion, not "drives
// applyPartialConversion" -- ConvertFunctions uses no conversion-framework
// symbol at all and still belongs here, because it rewrites tt.func to
// func.func.
//
// The target is KTIR, not KTDP: `ktdp` is one dialect, KTIR is the language it
// composes with linalg, arith, tensor, memref and scf.  Three of these passes
// emit no ktdp op at all -- LowerComputeOps gives linalg/tensor, LowerSpyreOps
// gives spyreop, ConvertFunctions gives func.
//
// Each pass declares its contract below:
//   Input    - dialects/ops it expects to see
//   Lowering - ops it converts and what they become
//   Output   - what the IR looks like after this pass
//
// Ordering constraints are stated per-pass in the descriptions below, since
// they are properties of the individual pass. For the order the passes are
// actually run in, see the two places that build the pipeline:
//   - add_convert_ttir_to_ktdp in third_party/spyre/triton_spyre.cc
//   - the `ktir` stage in third_party/spyre/backend/compiler.py
// ============================================================================

def LowerDescriptorMemory : Pass<"lower-descriptor-memory", "mlir::ModuleOp"> {
    let summary = "Lower tt.descriptor_* ops to ktdp memory ops.";
    let description = [{
        Input: TTIR — triton, arith, scf dialects; !tt.ptr and !tt.tensordesc types.

        Lowering:
        - tt.descriptor_load/store → ktdp.construct_memory_view + ktdp.construct_access_tile + ktdp.load/store
        - tt.descriptor_gather/scatter → ktdp.construct_indirect_access_tile + ktdp.load/store
        - tt.make_tensor_descriptor — dead after uses are lowered, erased by cleanup

        A layout marker on a descriptor — `tt.spyre_tensor_layout` or
        `tts.tensor_layout` — is left untouched, along with the
        builtin.unrealized_conversion_cast bridging the new view's memref back
        to !tt.tensordesc so the marker's operand keeps verifying. Consuming
        either is someone else's job: [RewriteDescriptorLayout] for the former,
        [LowerTTSMarkers] for the latter.

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
        - scalar tt.load → single-element 1-D ktdp.construct_memory_view +
          single-element 1-D ktdp.construct_access_tile + ktdp.load +
          tensor.extract
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
        ops are introduced. A layout marker left standing by
        [LowerDescriptorMemory] — `tt.spyre_tensor_layout` or
        `tts.tensor_layout` — passes through here too.
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

def LowerSpyreOps : Pass<"lower-spyre-ops", "mlir::ModuleOp"> {
    let summary = "Lower scalar math/arith ops to spyreop dialect intrinsics.";
    let description = [{
        Input: any IR containing scalar math/arith ops — typically the body
        of a linalg.generic after ConvertElementwiseToLinalg has scalarized
        a tensor-level op. spyreop's intrinsics are scalar-only
        (f16/df16/f32, or i32/i64 for the integer ops), so an op still on a
        tensor or vector is left alone rather than matched — this pass is
        not the one that will eventually convert it.

        Lowering:
        - math.sqrt  (operand f16 or f32)        → spyreop.sqrt
        - math.exp   (operand f16 or f32)        → spyreop.exp
        - math.rsqrt (operand f16 or f32)        → spyreop.rsqrt
        - arith.divf (operand f16 or f32)        → spyreop.realdiv,
          or spyreop.reciprocal when the numerator is a constant 1.0
        - arith.addi (i32, inside linalg.generic) → spyreop.addi32toi32
        - arith.addi (i64, inside linalg.generic) → spyreop.addi64toi64
        - arith.muli (i32, inside linalg.generic) → spyreop.muli32toi32

        math.sqrt/exp/rsqrt and arith.divf are matched unconditionally on
        scalar type: those ops only ever appear in real floating-point
        compute, never in address/index arithmetic, so a scalar occurrence
        still on an unsupported type (f64, bf16, ...) has no spyreop
        intrinsic to lower to and is left illegal, reported with a
        diagnostic rather than silently skipped. Still on a tensor/vector,
        they are left legal, quietly, since ConvertElementwiseToLinalg (not
        this pass) is what will eventually scalarize them.

        arith.divf splits on its NUMERATOR as well as its type: a numerator
        of one goes to the UNARY spyreop.reciprocal, so no float immediate
        reaches the device at all. Matched by value with m_OneFloat, which
        accepts a scalar float constant or a splat. No other value is
        special-cased.

        arith.addi/arith.muli are different: plain scalar integer add/mul is
        also used throughout a kernel for loop indices, offsets, and tile
        addressing, not just scalarized tensor compute. So these two are
        only matched inside a linalg.generic body (the structural signal
        that this is scalarized elementwise compute, not address math), and
        only at the specific bit-widths spyreop has an intrinsic for --
        anything else (other widths, or outside a linalg.generic) is left
        legal rather than reported.

        Ordering: must run after ConvertElementwiseToLinalg for a kernel
        that computes over a tensor -- anchored earlier, the op is still
        tensor-typed and left alone, never reaching its spyreop intrinsic.

        Output: no math.sqrt/exp/rsqrt or arith.divf remains on a supported
        scalar type (a divf by a constant-1.0 numerator leaving its constant
        behind, dead, for the pipeline's own cleanup); no arith.addi/muli at
        a supported bit-width remains
        inside a linalg.generic. The corresponding spyreop intrinsic is
        introduced in its place. Everything else -- unsupported types or
        widths, tensor/vector operands, or index arithmetic outside a
        linalg.generic -- passes through unchanged.
    }];

    let dependentDialects = [
        "mlir::spyreop::SpyreOpDialect",
        "mlir::math::MathDialect",
        "mlir::arith::ArithDialect"
    ];
}

def ConvertFunctions : Pass<"convert-ttir-functions", "mlir::ModuleOp"> {
    let summary = "Convert tt.func/tt.return to func.func/func.return and replace !tt.ptr args with index.";
    let description = [{
        Input: triton + ktdp + func + arith + linalg + tensor + memref + scf dialects.
        Must run after all memory passes — they consume !tt.ptr args via
        getBasePtrAsIndex before this pass replaces them.

        Precondition: every !tt.ptr function argument must be used only by
        unrealized_conversion_cast ops, and each such cast must give that
        argument at least one result of its own, every one of them index-typed.
        Replacing the argument with an index fixes up those casts, but a user of
        any other kind would be left with a wrongly typed operand, so the pass
        rejects that input with a diagnostic instead of producing IR that fails
        verification. Because the cast op is variadic on both sides, its arity
        decides which results belong to a given argument: a 1-to-N cast gives
        every result to its single operand, an N-to-N cast pairs result i with
        operand i, and an N-to-1 cast gives no result to any one operand — the
        last is rejected, as is a corresponding result typed anything but index.

        Lowering:
        - tt.func → func.func
        - tt.return → func.return
        - !tt.ptr function arguments → index (definitions only)

        Output: no tt.func or tt.return ops remain. Other triton ops are left
        untouched — tt.get_program_id and tt.get_num_programs in particular
        survive for DistributeWork, which runs after this pass.

        !tt.ptr does not disappear from the module. A function argument is
        retyped by rewriting the entry block's argument, so a declaration —
        which has no body, and whose parameters live only in its FunctionType —
        is skipped and keeps !tt.ptr in its signature. Retyping one would mean
        rewriting every caller too, which is out of scope here. A downstream
        pass must therefore not assume the type is gone.
    }];

    let dependentDialects = [
        "mlir::func::FuncDialect"
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

        Must run after LowerComputeOps (partials must be linalg/tensor),
        before RewriteDescriptorLayout (that pass has no physical-type
        propagation pattern for tt.inter_tile_reduce, so no inter-tile op may
        still be live when it runs), and before ConvertFunctions (reads function
        attributes ConvertFunctions would rewrite).

        A kernel that carries both a layout annotation and a
        tt.inter_tile_reduce is not supported: the layout pass cannot carry a
        physical type through the produce/reduce pair, whose two halves
        communicate through a !ktdp.tile_future rather than a RankedTensorType.
        That combination is not diagnosed — it fails downstream as a type
        mismatch.

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

#endif // TRITON_TO_KTIR_PASSES
