#ifndef SPYRE_TRANSFORMS_PASSES
#define SPYRE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Transforms whose subject is upstream structure or the whole program rather
// than any one dialect's abstractions.  UnaliasLinalgOuts,
// DropReductionInitFill, CanonicalizeGenericOperands and FuseGenerics act on
// linalg form -- generic bodies, indexing maps, reshape reassociations;
// NormalizeForDevice on upstream ops the toolchain downstream of this tree will
// not take, wherever they appear; MaterializeBaseAddresses acts on the entry
// function's signature.
// None of them translates a dialect, and none is about ktdp's own memory-view
// model, so they sit at the top level -- which does not stop them including
// dialect headers freely.
//
// Each pass declares its contract below:
//   Input    - dialects/ops it expects to see
//   Lowering - ops it converts and what they become
//   Output   - what the IR looks like after this pass
//
// All of them are members of a fixed stage pipeline in lib/Pipeline.cpp, not
// opt-in, and all run in `spyrecode`: NormalizeForDevice at that stage's head,
// DropReductionInitFill, UnaliasLinalgOuts, CanonicalizeGenericOperands and
// FuseGenerics in its shaping prefix, LowerSpyreOps after the layout rewrite,
// and MaterializeBaseAddresses at its tail and only when `bindBaseAddresses` is
// set.
// A pass's position there is chosen in C++ from a typed factory, so there is no
// caller hook for inserting one at a named point.
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

def NormalizeForDevice : Pass<"normalize-for-device", "mlir::ModuleOp"> {
    let summary = "Rewrite valid upstream IR the Spyre toolchain refuses or "
                  "mis-lowers into a spelling it takes. A PATTERN HOST: see the "
                  "admission rule. MAY CHANGE BEHAVIOUR, per pattern.";
    let description = [{
        Input: any IR. Each rewrite is an independent pattern, matched wherever
        its op appears — inside a linalg.generic body, inside a linalg.reduce
        combiner, or as plain scalar arithmetic — because nothing this pass
        rewrites depends on where the op sits. A greedy driver runs them to
        fixpoint.

        Lowering, one line per pattern:
        - arith.maxnumf → arith.maximumf
        - arith.minnumf → arith.minimumf
        Operands and the fastmath attribute are carried across unchanged.
        Nothing else is touched.

        WHAT MAY BE ADDED HERE. A pass that exists to host patterns becomes a
        junk drawer without a rule, so: a pattern belongs here only if all
        three hold.

        1. It rewrites ONE op into ONE op, both of them upstream. Those two
           together are the line, and each half excludes a neighbour in this
           same file: a rewrite whose target is a spyreop intrinsic belongs in
           [LowerSpyreOps], whether it takes one op or a group, and a rewrite
           that changes which generic an op sits in belongs in
           [FuseGenerics]. The reciprocal is the worked example:
           its motive is a device behaviour exactly like the patterns here, but
           its target is `spyreop.reciprocal`, so it is a selection rule and not
           a normalization.
        2. The input is VALID IR that some downstream consumer refuses or
           mis-lowers, not a defect being fixed. A defect is a bug in whatever
           produced it, and patching it here hides that.
        3. The rewrite names the downstream behaviour it is for — the tool, the
           component, what it does — so a reader can check whether it still
           holds.

        The min/max families differ on NaN, so this is a behaviour change and
        not only a spelling fix.

        Ordering: after LowerComputeOps, which is what turns a `tt.reduce` into
        a `linalg.reduce` carrying the combiner in its body — before that the
        combiner is not an arith op in a linalg body and there is nothing here
        to match. Otherwise free, so it runs at the head of the `spyrecode`
        stage as an invariant rather than to satisfy a later pass.

        In the `spyrecode` stage and not the `ttir`→`ktir` pipeline: what a
        pattern here claims is a claim about the SCHEDULER, and the `ktir`
        artifact keeps the spelling Triton's own semantics name.

        Output: no arith.maxnumf or arith.minnumf remains anywhere in the
        module.
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect"
    ];
}

def LowerSpyreOps : Pass<"lower-spyre-ops", "mlir::ModuleOp"> {
    let summary = "Select spyreop intrinsics for the arith/math ops in a compute "
                  "body.";
    let description = [{
        Input: KTIR whose computes are linalg.generic — so after
        ConvertElementwiseToLinalg and LinalgGeneralizeNamedOps, and after
        [FuseGenerics], on which see the contract below. Bodies hold
        arith/math ops. KTIR in, KTIR out: nothing here reads a `tt` op, which
        is why the pass is here and not in Conversion/TritonToKTIR/, whose
        criterion is a `tt` source dialect.

        A RULE HOST. Adding a case is one pattern plus one line in the pass — no
        change here, no CMake change, no pipeline change.

        Rules, one op to one:
        - math.sqrt  (f16 or f32)                 → spyreop.sqrt
        - math.exp   (f16 or f32)                 → spyreop.exp
        - math.rsqrt (f16 or f32)                 → spyreop.rsqrt
        - arith.divf (f16 or f32), numerator 1.0  → spyreop.reciprocal
        - arith.divf (f16 or f32), otherwise      → spyreop.realdiv
        - arith.addi (i32, inside linalg.generic) → spyreop.addi32toi32
        - arith.addi (i64, inside linalg.generic) → spyreop.addi64toi64
        - arith.muli (i32, inside linalg.generic) → spyreop.muli32toi32

        Rules, a GROUP to one:
        - arith.cmpf + arith.uitofp     → spyreop.compare, carrying the predicate
        - arith.cmpf + arith.select     → spyreop.select, fed by a spyreop.compare
                                          unless the cmpf is a zero test the
                                          select performs itself

        NO NaN REACHES A COMPUTE BODY. Every rule assumes it. Under it, each
        unordered cmpf predicate computes what its ordered counterpart does
        (`une` as `one`, `ueq` as `oeq`, ...), so both spellings select the same
        spyreop.compare — which is what lets Triton's `!=`, emitted as `une`,
        lower at all. A NaN in a kernel's inputs gives an unspecified result on
        the NaN lanes.

        ONE PASS FOR ALL SELECTION, and the two kinds are not two tiers. They
        share one greedy pattern set, and no two rules are rooted on the same
        op, so no order is declared anywhere.

        A GROUP IS WHAT IS SELECTABLE when no member of it could have been
        selected alone — the criterion, and not the op count. `arith.cmpf`
        produces an `i1`, a type no spyreop op produces, so what consumes that
        `i1` is what says which device op the pair is: a `uitofp` makes it one
        `spyreop.compare`, an `arith.select` would make it a compare feeding a
        `spyreop.select`, two ops and still one indivisible choice.

        EVERYTHING UNMATCHED FLOWS THROUGH, WITH ONE EXCEPTION. There is no
        conversion target: an op with no device form reaches the backend, which is
        the component that knows what it can take, and it refuses there. So this
        pass has no notion of an unsupported input — it selects what it can and
        leaves the rest as it found it. That is a change from when it was a
        dialect conversion, where an unsupported float width was reported here.

        The exception is an `i1`, and it is a different kind of thing rather than
        a carve-out. Flows-through is about a CAPABILITY gap: an f64 math.sqrt is
        valid IR that a future device or a future rule may do. An `i1` in a
        compute body is UNREPRESENTABLE — no spyreop intrinsic produces or
        consumes that type, so no rule anyone could add would ever remove it. That
        is a property of the dialect, not of a device generation, and it makes the
        program unlowerable rather than unsupported. It is reported HERE because
        this is the only place that still knows the PREDICATE; by the time the
        backend sees it, the failure names an op several lowerings below the one
        the author wrote.

        WHAT THE i1 CHECK REFUSES, precisely, because the scope is narrower than
        "any i1 in a body". It reads op RESULTS only and skips one that is
        yielded, so what it refuses is an i1 this pass produced and KEPT: made
        inside one body and read inside the same body. That is exactly a group it
        had the chance to select and did not.

        An i1 crossing a generic BOUNDARY is the other pass's finding. It is the
        tensor form, and [FuseGenerics]' legalization step reports it; before
        that pass has run every compare sits in a generic of its own yielding
        precisely that — so refusing it here would report IR a predecessor was
        meant to reshape, while naming the wrong cause. The two
        shapes are indistinguishable at this point, which is why the line is drawn
        by where the value lives rather than by how it got there.

        THE GENERIC BODY IS THE SCOPE for a group rule, and not a predicate.
        Every compute reaches this pass as a linalg.generic, so "inside a body"
        is not a guess about whether an op is compute — it is where the compute
        is. An `arith.addi` on an element type in a body is compute, and the addi
        of a loop index or a tile address is not in a body at all.

        The integer 1:1 rules use the same test as a per-op PREDICATE, which is a
        PROXY and is known to be one: plain scalar integer add and mul are also
        loop indices, offsets and tile addressing, and no exact test separates
        those from scalarized integer compute. It over-claims —
        `tl.arange(0, N) * stride` is an arith.muli that will be inside a body
        once scalarized — and holds today only because the tensor-of-pointers
        tt.load path is unimplemented.

        WHAT THIS PASS RELIES ON [FuseGenerics] FOR. A rule matches ops in ONE
        body, and ConvertElementwiseToLinalg gives every tensor-level op a body
        of its own — so a group spanning two tensor ops arrives spread over two
        generics unless something fused them. FuseGenerics' GroupComputes does,
        and only for a group this pass lowers: it asks `matchLowerableGroup`
        (include/Transforms/LowerSpyreOps.h), which SelectCompare and
        SelectWhere ask too, so the two passes cannot disagree on what a group
        is. Each group shape is one match function in kGroupShapes, and
        matchLowerableGroup tries each in turn. One shape covers every reader of
        the producer's result: a shape matches only when it covers every reader,
        and a result whose readers form different shapes is not grouped, so
        FuseGenerics reports the `i1` left between the generics. The
        reciprocal's constant is a scalar in the body because
        CanonicalizeGenericOperands' SplatOperandToScalar put it there. A
        contract between adjacent passes rather than an accident — and the
        reciprocal reads its numerator THROUGH the body, so for that rule the
        dependency is about whether it FIRES, never about whether it is correct.

        WHAT IS NOT SELECTED, and what each costs. None is a diagnostic, per
        EVERYTHING UNMATCHED FLOWS THROUGH above.

        A 1:1 rule does not fire and the op goes to the backend as arith:
        - a float width with no intrinsic (f64, bf16)
        - an integer width with no intrinsic, or integer add/mul outside a body
        - anything still tensor-typed, which only ConvertElementwiseToLinalg not
          having run can produce

        A group rule declines and its members stay as they were:
        - a divide whose numerator is not a constant one, or whose constant is
          the denominator: the realdiv rule takes it, which is the right answer
        - a numerator resolving to a NON-SPLAT constant tensor. Reading an
          operand through the body is sound only for a value uniform across it,
          so a mixed constant matches nothing — without that, element 0 being 1.0
          would be taken for the whole tensor
        - `ord` and `uno`, which ask about NaN-ness rather than an ordering, and
          the constant predicates, which ask nothing. No counterpart exists.
        - arith.sitofp where uitofp was wanted: an `i1` read as signed is 0 or
          -1, so the cast gives -1.0 where the predicate holds
        - a compare at one width whose result is cast to another.
          spyreop.compare has SameOperandsAndResultType and cannot do both

        Each of the compare declines leaves an `i1` INSIDE a body, so each is
        REFUSED by the check above rather than passed through — they are the
        exception, not the rule. The diagnostic names the predicate and says
        which kind of decline it was: one with no counterpart at all, or one whose
        predicate was fine and whose READER was the problem. That distinction is
        what an author can act on.

        Ordering: after the passes that make every compute a linalg.generic, and
        after [FuseGenerics] per the contract above.

        Output: the selected ops replaced by their intrinsics. Everything else
        unchanged.
    }];

    let dependentDialects = [
        "mlir::spyreop::SpyreOpDialect",
        "mlir::math::MathDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect"
    ];
}

def DropReductionInitFill : Pass<"drop-reduction-init-fill", "mlir::ModuleOp"> {
    let summary = "Drop the linalg.fill that initialises a reduction's outs.";
    let description = [{
        Input: any IR containing linalg ops on ranked tensors. Matches the
        LinalgOp interface but only ops with at least one reduction iterator, so
        linalg.reduce and a linalg.generic carrying a reduction iterator are both
        covered while an elementwise op's fill is out of scope — which is what
        keeps this pass away from the other producer of linalg.fill in this
        pipeline, tt.splat.

        Lowering:
        - single-ins, single-init reduction with a simple body (payload + yield,
          init read by the payload) whose outs[i] is defined by a linalg.fill
          writing into a tensor.empty → outs[i] repointed at that tensor.empty,
          and the fill erased once dead. Any fill value, any combiner.
        - Any other shape — matmul or another multi-operand contraction, a
          multi-result reduction, a compound body → left alone, silently. The
          fill is load-bearing there because nothing downstream re-establishes
          the accumulator.
        - A fill writing into anything other than tensor.empty → left alone;
          repointing outs at it would substitute live contents for the stated
          initial value.

        The pass never fails: it either rewrites or leaves the IR alone.

        A reduction's payload reads its outs operand, so dropping the init is
        sound only because the scheduler re-establishes the accumulator ahead of
        the reduction loop, with the neutral element of that reduction's own
        combiner (MapReductionPartials, via getNeutralAttr). That is a claim
        about the backend which this IR cannot express and this pass cannot
        verify, so the gate is shape only — a combiner the backend does not
        handle is the backend's diagnostic to emit, not a reason to refuse here.
        The residual unsoundness is a *seeded* reduction (a non-neutral fill,
        e.g. addf over 2.5), which no neutral element can recover; nothing above
        this pass can express one, since tt.reduce has no init operand.

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
        operand. The pass becomes unnecessary once downstream consumes that fill
        directly, and deletion is the intended end state rather than more gates.

        Ordering: must run after LowerComputeOps, the pass that creates the fill;
        anchored earlier it is a silent no-op. Independent of
        RewriteDescriptorLayoutGeneric — running before it simply means the layout
        pass physicalizes a tensor.empty rather than a fill.

        Output: reductions whose outs is a bare tensor.empty, as hand-written
        reference KTIR states directly.
    }];

    let dependentDialects = [
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
    ];
}


def CanonicalizeGenericOperands
    : Pass<"canonicalize-generic-operands", "mlir::ModuleOp"> {
    let summary = "State reshape and splat-constant operands of a "
                  "linalg.generic in the generic's own terms.";
    let description = [{
        Input: IR in which every compute op is already a `linalg.generic` — so
        after ConvertElementwiseToLinalg, LinalgGeneralizeNamedOps and
        UnaliasLinalgOuts.

        Lowering:
        - ReshapeToGeneric: a `tensor.collapse_shape` or `tensor.expand_shape`
          with static extents whose every user is an `ins` operand of a
          `linalg.generic`, and whose result → source map is projected (every
          result a bare dim or a constant) → one forwarding `linalg.generic`
          over the result's shape, all-parallel, whose body yields its input.
          The input map is that result → source map: `(d0) → (d0, 0)` for
          `[64,1] → [64]`, `(d0, d1) → (d0)` for `[4] → [4,1]`, and
          `() → (0, 0)` for a collapse to rank 0. The output map is the
          identity.
        - SplatOperandToScalar: an `ins` operand defined by an `arith.constant`
          of a splat int-or-float dense tensor → a scalar `arith.constant`
          above the generic, read in the body; the operand, its indexing map
          and its block argument are erased. A reduced port of upstream
          linalg's FoldScalarOrSplatConstant (foldSplatConstantOperand in
          Utils/GenericOperands).
        - Left alone:
          - a reshape with any user that is not a generic `ins`, e.g. one
            feeding a `ktdp.store`;
          - a linearizing reshape, one that merges or splits more than one
            non-unit dim, e.g. `[64,2] → [128]`, whose result → source map
            `(d0) → (d0 floordiv 2, d0 mod 2)` is not projected;
          - a reshape with a dynamic extent: none reaches this pass from the
            fixture kernels, so no `tensor.empty` with `tensor.dim` sizes is
            built for one;
          - `tensor.reshape`, `tensor.concat` and `tensor.extract_slice`;
          - a splat operand whose removal leaves some loop dim named by no
            map; a non-splat constant; a scalar `arith.constant` passed as an
            `ins`.

        Nothing is fused. Every forwarding generic this pass emits has
        projected maps, so FuseGenerics, which composes them through a
        permutation, only ever composes projected maps. A linearizing reshape
        stays a tensor op and reaches RewriteDescriptorLayoutGeneric as one,
        which refuses it on a physicalized path.

        HOW TO ADD A PATTERN: a pattern is rooted on, and matches, one kind of
        producer — the op that defines the operand — never the operand's type,
        nor anything downstream. A new pattern names a producer kind no
        existing pattern matches; one whose kind overlaps an existing pattern's
        is a new case of that pattern. Its lit file feeds it the other
        patterns' producers and checks that it does not fire. The two today
        are rooted on a reshape and on a generic with a splat-constant operand,
        so they never match the same op.

        `--debug-only=canonicalize-generic-operands` traces each rewrite and
        each reshape left alone.

        Ordering: after UnaliasLinalgOuts and before FuseGenerics, whose
        AbsorbForwardingGeneric absorbs the forwarding generics this emits.

        Output: no static `tensor.collapse_shape` / `tensor.expand_shape` whose
        only users are generic `ins`, and no splat dense constant `ins` that
        could be dropped without unnaming a loop dim.
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
    ];
}


def FuseGenerics : Pass<"fuse-generics", "mlir::ModuleOp"> {
    let summary = "Combine a producer linalg.generic into its consumer: absorb "
                  "a forwarding generic into an operand map, or group the ops "
                  "LowerSpyreOps lowers as one intrinsic into one body.";
    let description = [{
        Input: IR in which every compute op is a `linalg.generic` and every
        reshape or splat constant operand has been stated as a generic by
        CanonicalizeGenericOperands — so after ConvertElementwiseToLinalg,
        LinalgGeneralizeNamedOps, UnaliasLinalgOuts and
        CanonicalizeGenericOperands.

        Two patterns, both rooted on a consumer `linalg.generic` and looking at
        one `ins` operand at a time. They differ in what the operand's producer
        generic `P` yields for the result the consumer reads, so no operand
        matches both:

        - AbsorbForwardingGeneric: `P` yields one of its `ins` block arguments,
          `x`. When `P` is all-parallel and its result map `R` is a
          permutation → the operand is repointed at `x` under
          `A.compose(inversePermutation(R)).compose(C)`, `A` being `P`'s map
          for `x` and `C` the consumer's map for the operand. This covers a
          generalized `linalg.broadcast` / `linalg.transpose`, the forwarding
          generics CanonicalizeGenericOperands emits for a reshape, and one
          forwarded result of a multi-result generic. `P` is erased once
          dead.
        - GroupComputes: `P` yields an op result, the anchor. When `P` is
          all-parallel, `linalg::areElementwiseOpsFusable` holds, and one
          group shape LowerSpyreOps lowers as one intrinsic covers the anchor
          and every use of the matching block argument in the consumer, its
          readers (`matchLowerableGroup`, include/Transforms/LowerSpyreOps.h)
          → `P` is fused into the consumer by `linalg::fuseElementwiseOps`,
          and the fused op's dead results and duplicate operands are removed
          by `linalg::deduplicateOperandsAndRemoveDeadResults`. When `P` has a
          user other than the consumer, it is fused only if every
          non-terminator op in its body is a producer op the group consumes,
          so nothing outside the group is copied. The groups today are an
          `arith.cmpf` read by `arith.uitofp` (SelectCompare) or by
          `arith.select` as its condition (SelectWhere), at the compared
          width; the group shapes are owned by LowerSpyreOps, and a new one
          there is groupable here with no change. One shape covers every
          reader of the producer's result. Readers that form different shapes,
          e.g. one compare read by both an `arith.uitofp` and an
          `arith.select`, are not grouped: no shape covers every reader, and
          legalization reports the `i1`.

        Both patterns decline a producer that has a reduction loop. Both then
        check the consumer's operand list as it would be after the rewrite,
        before mutating anything (`checkOperandRewrite`,
        Utils/GenericOperands), and decline when:
          - a loop dim of the consumer is named by no operand map;
          - a value would be both an `ins` and an `outs` of the consumer.
        No map is checked for being projected (every result a bare dim or a
        constant). Every generic reaching this pass has projected maps, since
        CanonicalizeGenericOperands emits a forwarding generic only for a
        reshape with a projected map, and composing projected maps through a
        permutation keeps every result a bare dim or a constant.

        Anything else is left alone: two computes that form no group, a
        reduction producer, an `outs` operand, any tensor op.

        Legalization, a separate step after the patterns converge
        (`legalizeAfterFusion`). It rewrites nothing, and runs one check: an
        `ins` of element type `i1` defined by a `linalg.generic` is an error
        at the consumer, with notes naming the op that made it, each reader,
        and the groups LowerSpyreOps lowers an `i1` in. No spyreop op produces
        or consumes `i1`, so it has no device form on any path. LowerSpyreOps' rejectSurvivingBooleans reports the other
        form, an `i1` made and read inside one body.

        A general `where(x > y, p, q)` forms a group and is fused; LowerSpyreOps
        lowers it to `spyreop.compare` feeding `spyreop.select` in one body,
        which dbo-opt refuses, and it reports that itself. Kernels write the
        spilled form instead (`where__spilled*`): a compare to a float mask, a
        store, a load, then a zero-test select. Producing the two-group form in
        the compiler, with the intermediate held in the local scratchpad rather
        than in HBM, is a follow-up.

        HOW TO ADD A PATTERN: a pattern matches on its producer's kind — the op
        that defines the operand, or what the producer's body computes for it —
        never on the operand's type, nor on anything downstream. A new pattern
        names a producer kind no existing pattern matches; one whose kind
        overlaps an existing pattern's is a new case of that pattern. Its lit
        file feeds it the other patterns' producers and checks that it does not
        fire. A new group shape is a match function in LowerSpyreOps, not a
        pattern here.

        `--debug-only=fuse-generics` traces each rewrite and each decline, with
        the check that declined it.

        Ordering: after CanonicalizeGenericOperands, so the only producers are
        generics, and before RewriteDescriptorLayoutGeneric, which must see the
        absorbed maps so that no forwarding generic is left for it to refuse,
        and before LowerSpyreOps, whose groups must be in one body.

        Output: no forwarding generic whose absorption passes these checks
        feeds a generic `ins`; every group LowerSpyreOps lowers whose edge
        passes the checks is in one body; no `i1` tensor between generics, or the pass
        fails.

        Not checked here: a re-indexing tensor op this pass leaves in front of
        a generic, e.g. a `tensor.reshape` or `tensor.concat`.
        RewriteDescriptorLayoutGeneric refuses one that would need a
        linearizing map on a physicalized path; anything else reaches dbo-opt.
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
        boundary. With an empty list the pass is a no-op. It runs in the
        `spyrecode` stage only when `bindBaseAddresses` is set; the symbolic
        mode leaves the pointer arguments in place.

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
