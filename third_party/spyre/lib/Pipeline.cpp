//===- Pipeline.cpp - The Spyre backend's compile stages as pipelines -----===//
//
// The pass lists, once. Each per-pass ordering constraint is stated by the pass
// itself, in the Passes.td of whichever of the three libraries it belongs to;
// what is recorded here is only what the *sequence* has to get right, which no
// single pass can say.
//
//===----------------------------------------------------------------------===//

#include "Pipeline.h"

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/TTS/Transforms/Passes.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

void mlir::triton::spyre::buildTTIRToKTIRPipeline(
    OpPassManager &pm, const TTIRToKTIRPipelineOptions &options) {
  // tt.descriptor_load/store/gather/scatter -> ktdp memory ops.
  pm.addPass(createLowerDescriptorMemoryPass());

  // A scalar tt.load (plus its addptr chain) -> a single-element 1-D read.
  // [LowerPointerChainMemory would sit here -- planned, not implemented; it
  // would handle the tensor-of-pointers tt.load this one leaves legal.]
  pm.addPass(createLowerScalarLoadPass());

  // tt.reduce/broadcast/expand_dims/dot -> linalg + tensor, and a dead-op sweep.
  pm.addPass(createLowerComputeOpsPass());

  // tts.inter_tile_reduce -> ktdp.inter_tile_produce + delivery. After
  // LowerComputeOps, because the partials it consumes have to be linalg/tensor
  // by then.
  pm.addPass(createLowerInterTilePass());

  // tt.func/tt.return -> func.func/func.return, !tt.ptr -> index. Last of the
  // conversions, because every memory pass above consumes !tt.ptr arguments
  // through getBasePtrAsIndex, and because LowerInterTile reads work-slice
  // function attributes this rewrites. metadata["name"] and the base-address
  // inference read the module before the pipeline for the same reason.
  pm.addPass(createConvertFunctionsPass());

  // tt.get_program_id -> ktdp.get_compute_tile_id, tt.get_num_programs folded
  // against the grid, which is stamped on the entry function.
  pm.addPass(createDistributeWorkPass(options.grid));

  // Folds muli x,1 and the cast chains the conversions leave behind. It used to
  // also fold the arith.divsi/remsi a layout pass emits per access-tile index;
  // physicalization is in the `spyrecode` stage now, so that arithmetic is
  // created after this runs and the canonicalize this stage ends with is not the
  // one that folds it.
  pm.addPass(mlir::createCanonicalizerPass());

  // LAST, and after that canonicalize rather than before it. Each tts marker op's
  // annotation -> an attribute on the op the marked value resolved to.
  //
  // Bounded below by two passes, one per marker. LowerDescriptorMemory, because a
  // tts.tensor_layout lands on the memory view that pass builds, reached through
  // the bridge cast it also builds. LowerComputeOps, because a tts.pin lands on
  // the op PRODUCING the pinned value, so that op has to be in its final form --
  // a pinned reduction is a tt.reduce until there and a linalg.reduce after, and
  // an attribute written on the tt.reduce would go with it.
  //
  // Bounded from above by the canonicalize, and that bound is the subtle one. A
  // marker op is not memory-effect-free, so DCE leaves it alone -- and while it
  // survives it USES the value it marks, which is what keeps that value alive. The
  // attribute does not: it rides on the producer, and a producer whose results are
  // otherwise unused is trivially dead. So converting a marker before the DCE can
  // delete the very value it asked to place, attribute and all.
  //
  // That is not hypothetical. A relayout's share is consumed by the compose, which
  // consumes it by erasing the marker; with the conversion done early, the share's
  // producer is dead by the time this stage's canonicalize runs, and the pin is gone
  // before anything could honour it. Running here instead means the marker holds the
  // value through the DCE and the attribute is written when nothing left in this
  // stage deletes anything.
  //
  // Nothing in this stage reads the attributes, which is what makes the move free:
  // every consumer is in `spyrecode` -- RewriteDescriptorLayoutGeneric and
  // FoldDataMovementGenerics today, and the pin's own consumer when it lands, at the
  // head of that stage -- so no DCE runs between the write and the honouring.
  //
  // Markers must not cross into the artifact, and do not: this is the last pass, so
  // what leaves the stage is attributes. An op from a dialect a consumer does not
  // load fails at parse, which is the whole reason the annotation is an attribute.
  pm.addPass(tts::createLowerTTSMarkersPass());
}

void mlir::triton::spyre::buildSpyrecodePipeline(
    OpPassManager &pm, const SpyrecodePipelineOptions &options) {
  // Ahead of everything else in the stage, and not part of what follows:
  // upstream ops this tree emits legally but the toolchain below will not take
  // become ones it will. A pattern host, so a future case is a pattern there
  // rather than a line here; Passes.td has its ordering constraint and why it
  // is in this stage rather than the one above.
  pm.addPass(createNormalizeForDevicePass());

  // Physicalization is this stage's job, and the four passes below are it. The
  // `ktir` artifact carries LOGICAL descriptors plus a `tts.tensor_layout`
  // attribute on each annotated memory view; stick-tiling them happens here, on
  // the way to dbo-opt, which is what requires a physical layout -- a kernel that
  // stops at `ktir` is better served by the logical form it was written as.
  //
  // The annotation being an attribute is what allows this placement: an op in a
  // Triton-side dialect could not cross the boundary, since no consumer of the
  // `ktir` artifact registers that dialect and an op from an unregistered
  // dialect fails at parse. A discardable attribute with builtin values needs no
  // dialect to round-trip.
  //
  // The three shaping passes cross the boundary with the layout pass because they
  // establish its input contract: it restates `linalg.generic` and diagnoses
  // anything else on a physicalized chain, while the `ktir` stage emits `arith` on
  // tensors plus named linalg ops.
  //
  // The order is forced at every step except one:
  //
  //   DropReductionInitFill       FIRST, and the non-obvious one. It matches by
  //                               class -- `getDefiningOp<linalg::FillOp>()` on a
  //                               reduction's `outs` -- so after generalization it
  //                               matches nothing and returns success silently,
  //                               leaving dbo-opt to reject the fill it should have
  //                               removed. The opposite order tempts, on the
  //                               argument that the layout pass must physicalize
  //                               that fill; that is false both ways. Dropped
  //                               first, the `outs` is a bare `tensor.empty`, which
  //                               retypeToPhysical handles explicitly. Dropped
  //                               second, it is a generalized fill, which
  //                               retypeToPhysical does not handle: it takes the
  //                               blanket early return for a `linalg.generic`
  //                               producer, whose premise -- that such a producer
  //                               is on the rewrite's worklist -- a fill reading
  //                               only a scalar never meets, so the rebuilt result
  //                               map gains a rank the operand lacks and the linalg
  //                               verifier rejects it. Measured: this order is the
  //                               difference between the reduce family reaching the
  //                               device and not.
  //
  //   ConvertElementwiseToLinalg  arith-on-tensors -> generic. Emits generics only,
  //                               never a named op, so it and generalization act on
  //                               disjoint sets: their relative order is the one
  //                               free choice here.
  //
  //   LinalgGeneralizeNamedOps    named linalg -> generic. Upstream's. Numbers a
  //                               reduce's loops in input order, which is the form
  //                               the scheduler wants.
  //
  //   UnaliasLinalgOuts           last. What it removes is what
  //                               ConvertElementwiseToLinalg creates -- an `ins`
  //                               reused as `outs`, which dbo-opt maps to the wrong
  //                               value -- so it cannot precede it, and after
  //                               generalization it has one spelling to match
  //                               rather than two.
  //
  // Why DropReductionInitFill exists at all: LowerComputeOps gives every reduction
  // a zero `linalg.fill` on its `outs` per upstream linalg semantics, and the
  // scheduler's allowlist is add/mul/sub/reduce, which rejects the fill.
  //
  // Device-only by the SECOND half of the stage's rule in Pipeline.h -- its output
  // is not standalone KTIR. A reduce stripped of its neutral element means what it
  // says only because a downstream pass writes the accumulator before it is read,
  // and that pass is MapReductionPartials' initializer, which ktir_cpu never runs.
  // No KTIR reader can see that.
  //
  // NOT by the first half, which used to be stated here and is false: the
  // scheduler does not reset an accumulator to zero whatever the combiner is.
  // MapReductionPartials' lowerIterArgInitializer asks getNeutralAttr and fills
  // with the answer -- 0.0 for addf/subf, 1.0 for mulf, -inf for maximumf, +inf
  // for minimumf, and the integer counterparts. So the combiner a reduction uses
  // is not by itself a reason to refuse it, which is why the pass no longer
  // gates on the combiner or on the fill value at all: its gate is shape only.
  // See the header of DropReductionInitFill.cpp.
  //
  // The conclusion is unchanged and the pass does not move. Recorded because the
  // wrong reason is the more memorable one, and it is the reason that would
  // justify moving the pass back.
  pm.addPass(createDropReductionInitFillPass());
  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createUnaliasLinalgOutsPass());

  // Every coordinate change becomes an `indexing_maps` entry on the generic that
  // consumes it, so nothing whose only effect is to re-index survives.
  //
  // Both neighbours fix the position. It must follow the three passes above,
  // which are what make every compute a linalg.generic: the fusion it drives
  // matches generic -> generic, so a named producer or consumer blocks it whatever
  // the control function says. And it must precede the layout pass below, whose
  // behaviour it changes -- left in place, a data-movement generic has no layout
  // marker, so that pass leaves its result logical and bridges the gap with a
  // linearizing operand map the scheduler cannot project loop IVs through.
  pm.addPass(createFuseComputeAndDataMovementPass());

  // Logical descriptors -> physical (stick-tiled) layout, rooted on the
  // `tts.tensor_layout` attribute LowerTTSMarkers wrote onto each annotated
  // memory view in the `ktir` stage. Unannotated descriptors are left alone.
  //
  // No option selects a stride mode, and there is nothing to select: a
  // physicalized view lays its strides out row-major over its PHYSICAL sizes,
  // because that is what stick-tiled device data is, and the logical strides are
  // not read. A caller wanting the logical form reads the `ktir` artifact, which
  // is logical.
  pm.addPass(ktdp::createRewriteDescriptorLayoutGeneric());

  // The cleanup belonging to the pass above, moved across the boundary with it.
  // Physicalization emits an arith.divsi/remsi per access-tile index and a cast
  // chain per retyped value; the `ktir` stage's tail canonicalize used to fold the
  // static ones and now runs before them. Nothing else here would -- the
  // canonicalize and CSE below are inside the bind-base-addresses branch, which a
  // device launch does not take -- so omitting this hands dbo-opt the unfolded form
  // on exactly the path that matters. Observed as a nondeterministic dbo-opt crash,
  // not a diagnostic.
  //
  // Canonicalize only, for the reason the `ktir` stage's header gives: on an
  // author-written HBM round trip a store's ktdp.construct_access_tile and the
  // matching load's are both Pure over the same block, so CSE merges them and
  // dbo-opt's compute-group extraction aborts. See issue #161.
  pm.addPass(mlir::createCanonicalizerPass());

  // Instruction selection: the arith and math ops in each compute body become the
  // spyreop intrinsics that do the same thing. One pass for all of it -- the
  // one-op-to-one rules and the group rules share a pattern set, each rooted on
  // a different op.
  //
  // Needs every compute to be a linalg.generic, which the passes above make it: a
  // group rule's SCOPE is the generic body, and the 1:1 rules leave a tensor-typed
  // op alone.
  //
  // And it needs FuseComputeAndDataMovement to have run, which it has, well above. A rule
  // matches ops in ONE body while ConvertElementwiseToLinalg gives every
  // tensor-level op a body of its own, so a group spanning two tensor ops is two
  // generics until something fuses them -- that pass's `i1` clause is what does,
  // and the compare rule fires only because of it. The reciprocal is the softer
  // case: it reads its numerator through the body, so it fires either way.
  //
  // Nothing is reported here. An op with no device form flows through to dbo-opt,
  // which is the component that knows what it can take.
  pm.addPass(createLowerSpyreOpsPass());

  if (options.bindBaseAddresses) {
    // The one genuine choice in this stage: symbolic and bound are real
    // argument-passing modes, not a repair. Needs ConvertFunctions to have run,
    // which it has -- that is in the `ktir` stage.
    pm.addPass(createMaterializeBaseAddressesPass(options.baseAddresses));
    // Part of the materialization rather than of the stage: they fold the
    // arith.constant addresses it just introduced into their users. Symbolic
    // mode has no constants to fold.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
}
