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
        - arith.divf (operand f16 or f32)        → spyreop.realdiv
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
        scalar type; no arith.addi/muli at a supported bit-width remains
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
        still be live when it runs; a kernel carrying both a
        tt.spyre_tensor_layout marker and a tt.inter_tile_reduce is rejected
        here), and before ConvertFunctions (reads function attributes
        ConvertFunctions would rewrite).

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

def HbmRoundtrip : Pass<"hbm-roundtrip", "mlir::ModuleOp"> {
    let summary = "Break compute-to-compute tensor edges by storing the value "
                  "to HBM and loading it back.";
    let description = [{
        Input: KTIR after ConvertFunctions and DistributeWork — a `func.func`
        whose base addresses are `index` arguments, whose compute is `linalg`
        ops on tensors, and which carries a `grid` attribute.

        Why: the scheduler forms one local schedule per connected region of
        compute, and a compute occupies the whole local register file of the
        functional unit it lands on. One compute per schedule, so a compute's
        result reaches another compute through memory rather than as a value — a
        `ktdp.store` ending one schedule and a `ktdp.load` beginning the next, the
        way hand-written reference KTIR for a chained normalisation is built.

        Lowering: each tensor value produced by a compute `linalg` op in the entry
        block and consumed by another one gets a spill buffer of its own, added as
        a new `index` argument, with a `ktdp.store` after the producer and a
        `ktdp.load` before each consumer. One buffer per value, never shared or
        reused: a value that is both a kernel output and a later compute's input is
        written twice, which costs a store and buys a uniform rule.

        A buffer holds one slab per compute tile — the tile shape with dimension 0
        multiplied by the grid extent, dense row-major, the access tile anchored at
        `tile_id * tile_shape[0]`. Same rank as the tile, so no reshape either way;
        a grid of one tile gives a buffer that is exactly the tile, anchored at
        zero.

        Splitting the compute is half of it. Each group is extracted into a
        schedule of its own, so an operation two groups read belongs to neither,
        and the pass therefore gives each group its own copy of everything it
        reads: an access tile per store and per load even where two would be
        identical, a clone of the `linalg.fill` on `outs` and of any `ktdp.load` a
        second compute reads, and a rebuilt address cone per memory operation down
        to the base address. Index arithmetic is not exempt — a descriptor offset
        read by two access tiles is shared just as a tensor would be.

        Those copies are Pure, so **CSE must not run after this pass** or it merges
        them back together. In the spyrecode stage that holds by position: this
        pass is installed last on its branch, and that stage's only
        canonicalize/CSE is on the other arm of the same conditional.

        Scope: `linalg.generic`, `linalg.fill` and `linalg.reduce`, on a grid of
        rank 1. The op set follows the scheduler's own named-op allowlist for
        compute, `linalg.{add,mul,sub,max,min,reduce,generic,yield}`: `reduce` is
        in it and is a compute like any other, while `broadcast` (which
        LowerComputeOps emits for `tt.broadcast` / `tt.expand_dims`), `transpose`
        and `matmul` are not. The gate says nothing about what a compute does — a
        generic carrying reduction iterators is in scope. Rank 1 only because
        `ktdp.get_compute_tile_id` yields one index per grid dimension and reducing
        several to the single slab index a buffer is addressed by is work this
        temporary pass does not do.

        A function outside that scope is *reported* — but only once it has an edge
        to route. Without one there is nothing to do and the function is left
        exactly as it arrived, which is what keeps a single-compute kernel built
        from an out-of-scope named op compiling as it does today. Same for a tensor
        read by a compute that is none of `tensor.empty`, `linalg.fill` or
        `ktdp.load`: reported, since the pass cannot place it, and only reachable
        on a kernel that has an edge at all.

        A `linalg.fill` is a `linalg` op but not a compute, so it is neither a
        spilled producer nor a consumer whose operands get rewritten. It is cloned
        into each group that reads it, like the `tensor.empty` it writes into.

        Ordering: after ConvertFunctions (the arguments it adds are `index`, the
        type that pass leaves base addresses as) and after DistributeWork (it reads
        `grid` and the compute-tile id). The arguments it appends can only be
        filled in by a correction table, so the backend runs it in
        symbolic-argument mode — the default — and not where
        MaterializeBaseAddresses bakes addresses instead: there is no address to
        bake for a buffer the caller never declared.

        The buffers created are reported on the module as
        `ktdp.hbm_roundtrip_buffers`, an array of `{shape, element_type}`
        dictionaries in argument order, so the launcher can allocate one each and
        pass them after the kernel's own pointers. No element *width*: the backend
        already derives one from an element type's spelling for the kernel's own
        pointers. Only one public function per module may need buffers, since the
        list is per module and consumed positionally.

        Output: KTIR in which no compute `linalg` op reads a value produced by
        another, so each compute becomes its own local schedule. A no-op on any
        kernel with no compute-to-compute edge, whatever its `linalg` ops or grid
        rank. `--debug-only=hbm-roundtrip` traces the edges found, the buffers
        created and any reason for skipping.
    }];

    let dependentDialects = [
        "mlir::ktdp::KtdpDialect",
        "mlir::func::FuncDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect"
    ];
}


#endif // KTDP_TRANSFORMS_PASSES
