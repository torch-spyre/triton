//===- DropReductionInitFill.cpp - Drop a reduction's init fill -----------===//
//
// Removes the `linalg.fill` that supplies the `outs` of a reduction, repointing
// that `outs` at the `tensor.empty` the fill wrote into. The fill VALUE is not
// read: what replaces it downstream is derived from the combiner, not from what
// the fill states — see below.
//
// Why this is needed:
//   tt.reduce lowers (LowerComputeOps) to a linalg.reduce whose `outs` is
//   tensor.empty + linalg.fill of the combiner's neutral element, which is what
//   upstream linalg semantics call for. The Spyre dataflow-scheduler will not
//   take it: KTIRLegalityCheck's named-op allowlist is add/mul/sub/reduce, so
//   the fill is rejected outright, and with that check widened the fill is
//   generalized into a second linalg.generic and trips
//   ConstructThreeStagePipeline's one-compute-op-per-group assertion (the fill
//   feeds an init operand, so the existing elementwise fusion — which only walks
//   `ins` — never absorbs it). Hand-written reference KTIR states a bare
//   tensor.empty for exactly this reason.
//
// Why the rewrite is sound, and what the two gates are:
//   A reduction's payload READS its init operand — `linalg.reduce` names the
//   operand `$inits` and its own ODS example writes `arith.addf %out, %in` — and
//   `tensor.empty` has explicitly "unspecified" contents. So the rewritten IR is
//   only well-defined because something downstream re-establishes the
//   accumulator before it is read: MapReductionPartials' initializer, which asks
//   `getNeutralAttr` for the neutral element OF THIS COMBINER and fills with the
//   answer — 0.0 for addf/subf, 1.0 for mulf, -inf for maximumf, +inf for
//   minimumf, 0.0 for the absmax form, and the integer counterparts.
//
//   That is a claim about the BACKEND, not about this IR, and it is the whole
//   soundness argument: nothing here can verify it, and it can change without
//   this file noticing. What follows from it is narrow but real — the combiner
//   is not by itself a reason to refuse a reduction, because the reset is
//   derived from the combiner. So there is no combiner allowlist here. A
//   combiner the backend does not yet handle is the backend's diagnostic to
//   emit; mirroring its table here only rots toward refusing reductions that
//   have since started working.
//
//   What IS checked is shape, because the reset only happens to ops that pass
//   actually rewrites:
//
//     isMapReductionPartialsShape  one `ins`, one `init` — mirroring that pass's
//                                  own asserts. Excludes MATMUL above all: a
//                                  contraction is an addf-accumulate reduction
//                                  whose neutral genuinely IS zero, so a
//                                  fill-value-only gate would drop its init even
//                                  though MapReductionPartials never rewrites a
//                                  matmul and nothing would reset it. That
//                                  hazard is a shape question and survives any
//                                  widening of the combiner gate.
//     simpleReductionPayload       body is exactly payload + yield, init read by
//                                  the payload — the shape that pass clones and
//                                  LinalgLowering maps to one vectorchain op.
//
//   Failing either means the fill is load-bearing, so it is LEFT ALONE and no
//   diagnostic is emitted — this pass is not responsible for ops it cannot
//   reason about, and failing on them would break any pipeline that merely
//   contains a matmul. The pass therefore never fails.
//
//   The one thing no neutral table can recover is a SEEDED reduction: a fill of
//   2.5 under an addf combiner states `2.5 + sum`, and this pass rewrites it to
//   `sum`, silently. A bias is not a property of the combiner, so the backend
//   cannot restore it. Unreachable from anything above this pass — `tl.reduce` /
//   `tl.sum` / `tl.max` take no init parameter and `tt.reduce`'s ODS has no init
//   operand — so it is a hand-written-KTIR hazard only, pinned by a test rather
//   than gated here.
//
// This pass is a TEMPORARY fixup, and the intended end state is DELETION rather
// than more gates: it exists only because the scheduler rejects the
// `linalg.fill` that upstream linalg semantics require on a reduction's `outs`.
// Once downstream consumes that fill directly — honouring the init it states
// instead of re-deriving one — there is nothing left for this pass to do, and
// the seeded-reduction unsoundness above goes away with it.
//
// Algorithm:
//   1. Collect linalg ops that have at least one reduction iterator
//      (collect-then-rewrite, so erasing fills cannot invalidate the walk).
//      Ops with no reduction loop are left alone, which is what keeps the other
//      producer of linalg.fill in this pipeline — tt.splat — out of scope.
//   2. Skip any op that is not MapReductionPartials-shaped.
//   3. For each remaining `outs` operand defined by a linalg.fill whose own
//      output is a tensor.empty:
//      a. Skip    — the body is not a simple reduction.
//      b. Rewrite — point the operand at the tensor.empty, and erase the fill
//                   if nothing else uses it.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_DROPREDUCTIONINITFILL
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

/// True iff `op` has the shape MapReductionPartials actually handles, and will
/// therefore get its accumulator re-established by that pass's initializer.
///
/// With no combiner gate left this carries the whole soundness argument, so the
/// check mirrors that pass's own preconditions rather than approximating them: it
/// asserts a single `ins` and a single `init`.
///
/// The single-input condition is what excludes a **matmul**: a contraction is an
/// `addf`-accumulate reduction whose neutral genuinely is zero, so neither a
/// fill-value gate nor a combiner gate would stop this pass dropping its init —
/// it passes both, which is how it slipped through once. But MapReductionPartials
/// never rewrites a matmul, so nothing would reset that accumulator and the fill
/// is load-bearing. Same for any other multi-operand reduction (argmax and
/// friends, which carry an index lane).
bool isMapReductionPartialsShape(linalg::LinalgOp op) {
  return op.getNumDpsInputs() == 1 && op.getNumDpsInits() == 1;
}

/// The payload op of a *simple* reduction body: the one op computing the yielded
/// value, when the body is exactly that op plus the yield, and the init block
/// argument is one of its operands. Null otherwise.
///
/// This is `linalg.reduce`'s "shortened print form" shape, and it is what
/// MapReductionPartials + LinalgLowering handle — they clone the region wholesale
/// into a buffer-semantics generic and map the payload onto a single
/// `vectorchain` binary op. A body doing anything else is not our business.
Operation *simpleReductionPayload(linalg::LinalgOp op, OpOperand &init) {
  Block *body = op.getBlock();
  if (!body || body->getOperations().size() != 2)
    return nullptr;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield->getNumOperands() != 1)
    return nullptr;
  Operation *payload = yield->getOperand(0).getDefiningOp();
  if (!payload || payload->getBlock() != body)
    return nullptr;
  // The init must actually be read. If it is not, there is no init to discard
  // and this pass has nothing to do.
  Value initArg = op.getMatchingBlockArgument(&init);
  if (!llvm::is_contained(payload->getOperands(), initArg))
    return nullptr;
  return payload;
}

struct DropReductionInitFillPass
    : public mlir::triton::spyre::impl::DropReductionInitFillBase<
          DropReductionInitFillPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect first: the rewrite erases fills, which would invalidate a walk in
    // progress. Reductions only — see the header on tt.splat.
    SmallVector<linalg::LinalgOp> reductions;
    mod.walk([&](linalg::LinalgOp op) {
      if (op.getNumReductionLoops() > 0)
        reductions.push_back(op);
    });

    for (auto op : reductions)
      dropOne(op, rewriter);
  }

  /// Drops every `linalg.fill` feeding an `outs` operand of `op`.
  ///
  /// There is only one kind of non-rewrite left, and it is silent: `op` is not a
  /// reduction MapReductionPartials will ever touch (matmul or another
  /// multi-operand contraction, a multi-result reduction, a non-trivial body).
  /// The fill is load-bearing there because nothing downstream re-establishes
  /// the accumulator, so leaving it is the correct and conservative answer.
  /// Diagnosing it is not this pass's job either: whatever cannot lower it will
  /// say so, and failing here would make any pipeline that merely *contains* a
  /// matmul unable to run this fix.
  ///
  /// Nothing is rejected on the strength of the combiner or the fill VALUE. The
  /// reset downstream is derived per combiner (`getNeutralAttr`), so a combiner
  /// this pass has never heard of is for the backend to accept or refuse — see
  /// the header.
  void dropOne(linalg::LinalgOp op, IRRewriter &rewriter) {
    // Not a shape MapReductionPartials rewrites -> no reset downstream -> the
    // fill must stay. This is the matmul case.
    if (!isMapReductionPartialsShape(op))
      return;

    for (OpOperand &out : op.getDpsInitsMutable()) {
      auto fill = out.get().getDefiningOp<linalg::FillOp>();
      if (!fill)
        continue;

      // The fill must be writing into a fresh tensor, not over live data:
      // repointing `outs` at its output substitutes that output's contents for
      // the stated init, and only tensor.empty makes that a no-op.
      if (!fill.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
        continue;

      // A body this pass does not recognise as a simple reduction is left alone
      // for the same reason as the matmul case above.
      if (!simpleReductionPayload(op, out))
        continue;

      out.set(fill.getOutputs()[0]);
      // Only this reduction used it in the pipeline's own output, but a fill is
      // a normal value and something else may hold it.
      if (fill->use_empty())
        rewriter.eraseOp(fill);
    }
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createDropReductionInitFillPass() {
  return std::make_unique<DropReductionInitFillPass>();
}

} // namespace mlir::triton::spyre
