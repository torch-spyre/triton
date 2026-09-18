#ifndef SPYRE_TRANSFORMS_PASSES
#define SPYRE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Transforms whose subject is upstream structure or the whole program rather
// than any one dialect's abstractions.  UnaliasLinalgOuts and
// DropReductionInitFill act on linalg form; MaterializeBaseAddresses acts on
// the entry function's signature.  None of them translates a dialect, and none
// is about ktdp's own memory-view model, so they sit at the top level -- which
// does not stop them including dialect headers freely.
//
// Each pass declares its contract below:
//   Input    - dialects/ops it expects to see
//   Lowering - ops it converts and what they become
//   Output   - what the IR looks like after this pass
//
// These are all opt-in fix passes rather than members of the core pipeline;
// they are spliced in from Python via SpyreOptions.required_fixes.  See
// third_party/spyre/backend/compiler.py.
// ============================================================================

def UnaliasLinalgOuts : Pass<"unalias-linalg-outs", "mlir::ModuleOp"> {
    let summary = "Replace a linalg 'outs' operand that aliases an 'ins' operand "
                  "with a fresh tensor.empty.";
    let description = [{
        Input: any IR containing linalg ops on ranked tensors. Matches the
        LinalgOp interface, so both named ops (linalg.add, linalg.mul, …) and
        linalg.generic are covered.

        Ops outside the LinalgOp interface (linalg.yield, linalg.index,
        linalg.softmax, winograd_*) are not matched.

        Lowering:
        - linalg op whose outs[i] is the same SSA value as one of its ins
          operands, and whose body does not read that operand's initial value
          → outs[i] replaced with a fresh tensor.empty of the same type.
        - Same aliasing, but the body does read the initial value (an
          accumulating op such as linalg.reduce or linalg.matmul) → rejected
          with a diagnostic; the initial value is live, so no replacement
          preserves the result.
        - Same aliasing on a dynamically shaped outs → rewritten, with one
          tensor.dim per dynamic dimension measuring the extent off the aliased
          ins operand to size the tensor.empty.
        - Same aliasing on a non-tensor outs → rejected with a diagnostic.

        Every op matching the LinalgOp interface is in destination-passing
        style: its outs operands supply both the result shape and the initial
        value of each output element. tensor.empty preserves the shape and
        drops the initial value, which is why the rewrite is only applied where
        that value is provably unread.

        Motivation: the Spyre dataflow-scheduler maps each operand of a compute
        op to a freshly loaded value, so a value appearing in both ins and outs
        is mapped twice and the second mapping overwrites the first. The main
        known source is upstream MLIR's ConvertElementwiseToLinalg, which reuses
        the first type-matching operand as outs with no flag to suppress it.

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

def DropReductionInitFill : Pass<"drop-reduction-init-fill", "mlir::ModuleOp"> {
    let summary = "Drop a zero linalg.fill that initialises a reduction's outs.";
    let description = [{
        Input: any IR containing linalg ops on ranked tensors. Matches the
        LinalgOp interface but only ops with at least one reduction iterator, so
        linalg.reduce and a linalg.generic carrying a reduction iterator are both
        covered while an elementwise op's fill is out of scope — which is what
        keeps this pass away from the other producer of linalg.fill in this
        pipeline, tt.splat.

        Lowering:
        - reduction whose outs[i] is defined by a linalg.fill of a constant zero
          (float or integer, +0.0 and -0.0 alike) writing into a tensor.empty
          → outs[i] repointed at that tensor.empty, and the fill erased once
          dead.
        - Same shape, but the fill value is not zero → rejected with a
          diagnostic.
        - A fill writing into anything other than tensor.empty → left alone;
          repointing outs at it would substitute live contents for the stated
          initial value.

        A reduction's payload reads its outs operand, so dropping the init is
        sound only because the scheduler emits its own accumulator reset ahead of
        the reduction loop. That reset is hardcoded to zero, which is why the
        gate is the literal value zero rather than the combiner's neutral
        element: a mulf reduction initialised to 1.0 would be reset to zero by
        the scheduler whatever this pass does, so it is reported instead of
        silently lowered.

        That makes this a deliberate, narrow exception to the rule
        UnaliasLinalgOuts enforces. That pass refuses exactly this rewrite,
        because a payload reading its init makes tensor.empty observable; the
        exception is justified here by the reset guarantee, which the IR does not
        express.

        Motivation: LowerComputeOps gives every tt.reduce an outs of
        tensor.empty + linalg.fill, per upstream linalg semantics. The
        dataflow-scheduler's KTIRLegalityCheck allowlist is add/mul/sub/reduce
        and rejects the fill; widening that allowlist alone turns the rejection
        into an assertion failure in ConstructThreeStagePipeline, which admits
        one compute op per group and cannot fuse a producer feeding an init
        operand.

        Ordering: must run after LowerComputeOps, the pass that creates the fill;
        anchored earlier it is a silent no-op. Independent of
        RewriteDescriptorLayout — running before it simply means the layout pass
        physicalizes a tensor.empty rather than a fill.

        Output: reductions whose outs is a bare tensor.empty, as hand-written
        reference KTIR states directly.
    }];

    let dependentDialects = [
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
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

#endif // SPYRE_TRANSFORMS_PASSES
