#ifndef SPYRE_TRANSFORMS_PASSES
#define SPYRE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ============================================================================
// Transforms whose subject is upstream structure or the whole program rather
// than any one dialect's abstractions.  UnaliasLinalgOuts,
// DropReductionInitFill and FuseComputeAndDataMovement act on linalg form --
// generic bodies, indexing maps, reshape reassociations; NormalizeForDevice on
// upstream ops the toolchain downstream of this tree will not take, wherever
// they appear; MaterializeBaseAddresses acts on the entry function's signature.
// None of them translates a dialect, and none is about ktdp's own memory-view
// model, so they sit at the top level -- which does not stop them including
// dialect headers freely.
//
// Each pass declares its contract below:
//   Input    - dialects/ops it expects to see
//   Lowering - ops it converts and what they become
//   Output   - what the IR looks like after this pass
//
// All five are members of a fixed stage pipeline in lib/Pipeline.cpp, not
// opt-in, and all five run in `spyrecode`: NormalizeForDevice at that stage's
// head, DropReductionInitFill, UnaliasLinalgOuts and
// FuseComputeAndDataMovement in its shaping prefix, MaterializeBaseAddresses at its tail and only when
// `bindBaseAddresses` is set.
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
           [FuseComputeAndDataMovement]. The reciprocal is the worked example:
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
        [FuseComputeAndDataMovement], on which see the contract below. Bodies hold
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

        Rules, a REQUEST to one — a linalg.generic body whose ops, constants
        aside, all carry one `tts.spyreop_hint` tag, read one `ins` block argument and
        yield one value, i.e. the whole fallback of one `tl.spyre_op`:
        - name "gelu"    (f16)          → spyreop.gelu
        - name "silu"    (f16 or f32)   → spyreop.silu
        - name "sigmoid" (f16 or f32)   → spyreop.sigmoid
        The body is replaced whole, casts included, by the intrinsic applied to
        the block argument; the type it reads must be the type it yields.

        A TAGGED OP IS CLAIMED. Every 1:1 and group rule declines an op carrying
        `tts.spyreop_hint`, so a request's `math.exp` is never selected alone as
        spyreop.exp. That is what keeps the rules unordered: the request rule is
        rooted on the generic, the others never touch its members.

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

        EVERYTHING UNMATCHED FLOWS THROUGH, WITH TWO EXCEPTIONS. There is no
        conversion target: an op with no device form reaches the backend, which is
        the component that knows what it can take, and it refuses there. So this
        pass has no notion of an unsupported input — it selects what it can and
        leaves the rest as it found it. That is a change from when it was a
        dialect conversion, where an unsupported float width was reported here.

        The first exception is a REQUEST left unselected: any op still carrying
        `tts.spyreop_hint` after the rewrite is an error naming the intrinsic
        and why. The author asked for it by name, so passing the
        fallback through would ignore the request without saying so. The causes
        it names: the request split over two bodies (a fusion that did not
        happen), a body also holding ops outside the request (a fold that merged
        one in), a body reading two values or yielding two, a type the intrinsic
        does not take (gelu at f32), and a request that reads one type and
        yields another.

        The second exception is an `i1`, and it is a different kind of thing
        rather than a carve-out. Flows-through is about a CAPABILITY gap: an f64 math.sqrt is
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
        tensor form, [FuseComputeAndDataMovement] has the clause that removes it,
        and before that pass has run every compare sits in a generic of its own
        yielding precisely that — so refusing it here would report IR a
        predecessor was meant to reshape, while naming the wrong cause. The two
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

        WHAT THIS PASS RELIES ON [FuseComputeAndDataMovement] FOR. A rule
        matches ops in ONE body, and ConvertElementwiseToLinalg gives every tensor-level op a
        body of its own — so a group spanning two tensor ops arrives spread over
        two generics unless something fused them. That pass's `i1` clause exists
        for exactly this: the compare rule sees its pair only because that pass
        brought them together, and the reciprocal's constant is a scalar in the
        body for the same reason. A contract between adjacent passes rather than
        an accident — and the reciprocal reads its numerator THROUGH the body, so
        for that rule the dependency is about whether it FIRES, never about
        whether it is correct. Its `tts.spyreop_hint` clause is the same contract for requests: each
        tensor op of a fallback arrives in its own generic, and that pass fuses
        the ones of one request into one body.

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
        after [FuseComputeAndDataMovement] per the contract above.

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


def FuseComputeAndDataMovement
    : Pass<"fuse-compute-and-data-movement", "mlir::ModuleOp"> {
    let summary = "Fold a pure data-movement linalg.generic into the indexing "
                  "map of the op that consumes it.";
    let description = [{
        Input: IR in which every compute op is already a `linalg.generic` — so
        after ConvertElementwiseToLinalg, UnaliasLinalgOuts and
        LinalgGeneralizeNamedOps. A coordinate change (broadcast, transpose,
        inserted unit dim) arrives as its own generic whose body computes
        nothing; only its indexing maps differ.

        Lowering:
        - linalg.generic PRODUCER whose body is exactly one operation, a
          `linalg.yield` of an `ins` block argument, feeding a linalg.generic
          consumer → fused, its operand map composed into the consumer's.
          Upstream's `populateElementwiseOpsFusionPatterns` does the
          composition; this pass supplies the control function that decides
          which fusions may happen.
        - A COORDINATE RESTATEMENT on a generic's `ins` operand — currently
          `tensor.collapse_shape` or `tensor.expand_shape` — whose
          result-to-source map, COMPOSED with the consumer's operand map, has
          every result a bare loop dim or a constant → absorbed: the operand is
          repointed at the restatement's source and given the composed map. A
          unit-dim collapse `[64,1] → [64]` gives `(d0) → (d0, 0)`, the reshape
          `tt.broadcast`'s lowering puts in front of the broadcast generic;
          a unit-dim expand `[4] → [4,1]` gives `(d0, d1) → (d0)`, the one
          `tt.expand_dims` leaves behind where no collapse cancelled it.
        - A linalg.generic PRODUCER feeding a linalg.generic consumer, both of
          whose bodies carry the same `tts.spyreop_hint` tag → fused. The tag means both
          hold ops LowerTTSMarkers inlined from one `tts.spyre_op`, and
          [LowerSpyreOps] replaces a request only as ONE body. Constants carry
          no hint, and a body carrying two hints has none.
          Here rather than in a pass after RewriteDescriptorLayoutGeneric: that
          pass physicalizes only the generics one hop off an annotated load or
          store, leaving the request's first generic with a linearizing result
          map, which elementwise fusion cannot fuse through.
        - Anything else is left alone. Two computes of different requests, or of
          none, never merge, and a compute already carrying a shape change never
          merges with a downstream compute.
        - A coordinate restatement whose composed map LINEARIZES — a collapse
          fusing two non-unit dims, an expand splitting into two — is not
          absorbed. `tensor.reshape` and `tensor.concat` have no result-to-source
          map at all: a reshape takes its shape as a tensor operand, so there is
          no static structure to read one off, and a concat selects between
          operands per coordinate. All three are left alone SILENTLY where
          nothing physicalizes them, and REJECTED where something does — see the
          gate below.

        The predicate is STRUCTURAL, not an attribute: it is what gates every
        fusion decision, so a stale or missing label would not merely mislead a
        reader, it would silently change the emitted program. There is also
        nothing to read — the ops it keys on reach this pass from
        LinalgGeneralizeNamedOps rewriting a linalg.broadcast or
        linalg.transpose, neither of which records its provenance, and
        RewriteDescriptorLayoutGeneric rebuilds every generic on a physicalized
        chain, so a label would have to be maintained by a pass with no other
        reason to know it exists.

        The control function does not check `hasOneUse`, unlike the one
        upstream's own --linalg-fuse-elementwise-ops supplies. A data-movement
        generic feeding two consumers is folded into each and then dies, which is
        the intent: the goal is that none survives, and duplicating a coordinate
        change costs nothing because there is no body to duplicate — the second
        consumer gains an operand map, not an op.

        Two properties fall out rather than needing cases of their own. A
        generic that is both a compute and a shape change is declined as a
        producer and remains eligible as a consumer. And because the predicate
        keys on the producer, a REDUCTION consumer is fine: the shape change
        folds into its operand map while its `iterator_types` and single-compute
        body are untouched — which is also what keeps DropReductionInitFill's
        precondition (a reduce body of exactly two ops) true, since a pure
        data-movement producer contributes no body op.

        Motivation: a data-movement generic has no descriptor and so no layout
        marker, so RewriteDescriptorLayoutGeneric leaves its result logical while
        its consumer is physical. The consumer's operand map then has to bridge
        the two and a LINEARIZATION is folded into it
        (`(d0, d1, d2) -> (d1, d0 * 64 + d2)`), which the dataflow-scheduler
        cannot schedule: "could not locate matching linalg operand to project
        loop IVs and tile sizes through; the data_transfer rank would not match
        the underlying memref". Folded into the consumer, the same coordinate
        change is an operand map the layout pass restates at physical rank like
        any other — a statistic read at a constant lane comes out as
        `(d0, d1, d2) -> (d1, 0)`, which is what hand-written reference KTIR
        states.

        Why the projection check is on the COMPOSED map and not on the
        reassociation: a linearizing collapse is not INEXPRESSIBLE as an affine
        map — AffineExpr has FloorDiv and Mod, and this pass emits
        `(d0) → (d0 floordiv 2, d0 mod 2)` for `[64,2] → [128]`. The reason to
        decline it is that the scheduler cannot project loop IVs and tile sizes
        through floordiv or mod, which is a property of the map the CONSUMER ends
        up holding. The two can differ, because composition folds: the same
        collapse read at a constant coordinate composes to `(d0, d1) → (0, 0)`,
        which projects fine and is absorbed.

        Not used here: upstream's `populateFoldReshapeOpsBy{Expansion,
        Collapsing}Patterns`, which take the same control-function type. They
        answer a general reshape by changing the consumer's iteration space and
        pushing a reshape onto its other operands, which on a physicalized chain
        emits a `tensor.expand_shape` that RewriteDescriptorLayoutGeneric
        rejects ("this op reads a value the rewrite retyped, but the rewrite
        restates only linalg.generic"). The absorption above keeps the iteration
        space and moves the coordinate change into one operand's map, which is
        the whole of the difference.

        `tensor.extract_slice` is deliberately NOT a coordinate restatement, and
        not for the reason one would guess. Its coordinate map is derivable —
        static offsets and unit strides give `(d0, d1) → (d0 + o0, d1 + o1)`. What
        is not is its EXTENT change: linalg infers its loop bounds from the
        operand shapes through the indexing maps, so repointing an operand at the
        larger unsliced source under any map makes the inferred extents
        inconsistent and fails linalg's own verifier. A slice crops; an operand
        map re-indexes. It is also the one shape op the scheduler already looks
        through, and has zero occurrences across the fixture artifacts. Being
        "not a restatement" means it is neither absorbed nor rejected.

        THE GATE on rejection: an unabsorbable restatement is refused only on a
        path RewriteDescriptorLayoutGeneric will physicalize, asked with THAT
        PASS'S OWN seed traversals (shared from Dialect/KTDP/Utils, not copied):
        seed from every `tts.tensor_layout`-annotated
        `ktdp.construct_memory_view`, reach its loads and stores through its
        access tiles, take the generics one hop off those. One hop, not a
        transitive closure — that is the scope the layout pass rewrites.

        Why gated, and the reason is forward-looking rather than protective. As
        annotation coverage grows toward everything on a device path being
        physicalized, the gate fires more often and converges to the ungated
        behaviour by itself, so it never needs removing; shipping ungated would
        instead mean adding a gate later, the moment an unannotated path has to
        be legal. Kept pending the LX roundtrip, at which point the better move
        may be to invert it into an assertion that anything on a device path IS
        annotated.

        Three seeds, because the hazard has two sides. BACKWARD from each `ins`
        operand of such a generic, through every generic not itself in the rewrite
        set, stopping at anything that is not a coordinate restatement; this
        catches a re-indexing op whose own load is on an UNANNOTATED view, which
        nothing else looks at (retypeToPhysical is only called for an operand that
        HAS a layout). BACKWARD from the DATA of each store over an annotated
        view, the exact mirror: findLayoutForResult walks a generic's result users
        for a `ktdp.store` DIRECTLY, so a shape op between the two makes it return
        null and the linearization lands on the `outs` map instead — `gather__1d`
        carries an `8x1 → 8` `tt.reshape` into a store, so the shape is real.
        FORWARD one hop off each physicalized load, for a result neither backward
        walk reaches; where it reaches a generic this is redundant with
        checkConsumersAreRewritable and is kept only because it fires a stage
        earlier and says which property of the op is the problem.

        The store side is REJECTION ONLY, never absorption; the header of
        FuseComputeAndDataMovement.cpp says why, and what the eventual fix would
        have to choose between.

        Rejection is DERIVED, not a declared op list: the condition is exactly
        "absorption failed" and "on a physicalized path", and the first is asked
        of the same function the absorber asks. Teaching that function a static
        `tensor.reshape` stops the rejection for it with nothing else to update.

        `--debug-only=fuse-compute-and-data-movement` traces the decisions: each
        restatement asked for and the answer, each operand the absorber considered
        with the composition spelled out, the gate's seeds and scope, and each
        rejection with the path that reached it.

        Ordering: after the passes that make every compute op a linalg.generic
        (fusion matches generic → generic only, so a named producer or consumer
        blocks it whatever the control function says), and before
        RewriteDescriptorLayoutGeneric, which must see the folded maps so that no
        data-movement generic survives for it to linearize — and whose scope the
        gate above predicts.

        Output: no linalg.generic whose body is a lone yield of an input remains
        where a linalg.generic consumes its result, and no coordinate restatement
        with a projectable composed map feeds a generic. Every op is left with the
        same single compute in its body it had before. On a module with at least
        one annotated memory view, no unabsorbable coordinate restatement is left
        on a path that view physicalizes; the pass fails instead.
    }];

    // No ktdp and no tts, though the gate reads both: dependentDialects is about
    // ops a pass CREATES, and this one only inspects them.  The link edges those
    // reads need are in the CMakeLists, derived from the #includes.
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
