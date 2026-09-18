//===- DropReductionInitFill.cpp - Drop a reduction's init fill -----------===//
//
// Removes a `linalg.fill` of a combiner's neutral element that supplies the
// `outs` of a reduction, repointing that `outs` at the `tensor.empty` the fill
// wrote into.
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
// Why the rewrite is sound, and why the gate is what it is:
//   A reduction's payload READS its init operand — `linalg.reduce` names the
//   operand `$inits` and its own ODS example writes `arith.addf %out, %in` — and
//   `tensor.empty` has explicitly "unspecified" contents. So the rewritten IR is
//   only well-defined because something downstream re-establishes the
//   accumulator before it is read: MapReductionPartials, which on seeing a
//   reduction whose init is a bare `tensor.empty` emits its own `linalg.fill` of
//   the neutral element it derives FROM THE COMBINER — `getNeutralAttr` there,
//   whose float table is
//
//       addf / subf → 0.0    mulf → 1.0
//       maximumf    → the largest negative finite    minimumf → the largest
//                                                                positive finite
//
//   (`maxnumf` is in that table too and `minnumf` is not, which is a separate
//   reason NormalizeFloatMinMax runs before this pass: it settles the spelling
//   so both this gate and that table see one form.)
//
//   The rewrite is therefore sound when that pass will run on this op AND the
//   value it will write is a neutral element for this combiner. Both halves are
//   checked, because neither implies the other:
//
//     isMapReductionPartialsShape  one `ins`, one `init` — mirroring that pass's
//                                  own asserts. Excludes MATMUL above all: a
//                                  contraction is an addf-accumulate reduction
//                                  whose neutral genuinely IS zero, so a
//                                  fill-value-only gate would drop its init even
//                                  though MapReductionPartials never rewrites a
//                                  matmul and nothing would reset it.
//     simpleReductionPayload       body is exactly payload + yield, init read by
//                                  the payload — the shape that pass clones and
//                                  LinalgLowering maps to one vectorchain op.
//     neutralKind                  the combiner is one the scheduler derives a
//                                  neutral for: the five FLOAT entries of that
//                                  table. Integers are excluded — see below.
//     statesNeutral                the init this pass is about to throw away is
//                                  that same neutral, so throwing it away loses
//                                  nothing.
//
//   Note `maximumf` is admitted even though the two values are not identical: we
//   state -inf (what LowerComputeOps emits) and the scheduler writes the largest
//   negative FINITE float. Both are ≤ every finite element, so the answer agrees
//   for finite data; they differ only for an input that is entirely -inf, where
//   the scheduler's init would survive as the answer. `statesNeutral` accepts
//   either spelling for that reason — hand-written reference KTIR uses the
//   scheduler's.
//
//   Integer combiners are held out on purpose. The scheduler's table does cover
//   addi/subi/muli, so the rewrite would very likely be sound there too, but no
//   integer reduction in this tree has been carried to a binary, so there is
//   nothing that has shown it is. An early diagnostic is the honest answer until
//   one exists. (This is the one gate here that costs something: it also refuses
//   an integer reduce that only ever meant to run on ktir_cpu, which never sees
//   the scheduler at all — see the note on _SPYRECODE_STAGE_PASSES in
//   backend/compiler.py about this pass not belonging in a pipeline every compile
//   runs.)
//
//   Failing the first two means the fill is load-bearing, so it is LEFT ALONE and
//   no diagnostic is emitted — this pass is not responsible for ops it cannot
//   reason about, and failing on them would break any pipeline that merely
//   contains a matmul. Failing the last two means this IS our op but its init
//   cannot be honoured; both alternatives are wrong, so it is reported.
//
//   ALL OF THIS IS A CLAIM ABOUT THE SCHEDULER, not about linalg. It holds for a
//   dbo-opt whose MapReductionPartials derives the neutral per combiner. Against
//   an older one that reset every accumulator to a hardcoded zero, dropping a
//   `mulf` or `maximumf` init would silently return the wrong numbers rather than
//   failing — which is why the reasoning above names that function.
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
//      b. Reject  — the scheduler derives no neutral for this combiner, or the
//                   fill does not state that neutral.
//      c. Rewrite — point the operand at the tensor.empty, and erase the fill
//                   if nothing else uses it.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_DROPREDUCTIONINITFILL
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// The neutral element a combiner needs, as a *kind* rather than a value: the
/// float value differs by element type, and for min/max this pass accepts two
/// spellings of the same idea (see `statesNeutral`).
enum class Neutral { Zero, One, NegExtreme, PosExtreme };

/// The neutral MapReductionPartials will re-establish for `payload`'s combiner,
/// or nullopt when it derives none that this pass will act on.
///
/// The five float entries of that pass's `getNeutralAttr` table, and only those.
/// Integers are deliberately absent — the header says why.
std::optional<Neutral> neutralKind(Operation *payload) {
  if (isa<arith::AddFOp, arith::SubFOp>(payload))
    return Neutral::Zero;
  if (isa<arith::MulFOp>(payload))
    return Neutral::One;
  if (isa<arith::MaximumFOp>(payload))
    return Neutral::NegExtreme;
  if (isa<arith::MinimumFOp>(payload))
    return Neutral::PosExtreme;
  return std::nullopt;
}

/// True iff `v` is a constant stating the `kind` neutral.
///
/// Zero accepts -0.0 as well as +0.0, and an integer zero, via m_AnyZeroFloat /
/// m_Zero: all are additive identities.
///
/// The two extremes accept BOTH an infinity and the largest finite of that sign,
/// because the two producers of this IR spell it differently and both are neutral
/// over finite data: LowerComputeOps emits ±inf, and the scheduler's own reset —
/// which hand-written reference KTIR mirrors — uses the largest finite.
bool statesNeutral(Value v, Neutral kind) {
  if (kind == Neutral::Zero)
    return matchPattern(v, m_AnyZeroFloat()) || matchPattern(v, m_Zero());

  FloatAttr attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return false;
  const APFloat &f = attr.getValue();
  if (kind == Neutral::One)
    return f.isExactlyValue(1.0);

  bool wantNegative = kind == Neutral::NegExtreme;
  if (f.isNegative() != wantNegative)
    return false;
  return f.isInfinity() ||
         f.bitwiseIsEqual(APFloat::getLargest(f.getSemantics(), wantNegative));
}

/// True iff `op` has the shape MapReductionPartials actually handles, and will
/// therefore get its accumulator overwritten by that pass's zero reset.
///
/// This is the whole soundness argument, so the check mirrors that pass's own
/// preconditions rather than approximating them: it asserts a single `ins` and a
/// single `init`.
///
/// The single-input condition is what excludes a **matmul**: a contraction is an
/// `addf`-accumulate reduction whose neutral genuinely is zero, so a gate that
/// only looked at the fill value would drop its init — but MapReductionPartials
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
    : public mlir::triton::ktdp::impl::DropReductionInitFillBase<
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

    // Keep going after a rejection so one run reports every fill it cannot
    // handle, rather than costing the user one recompile per reduction.
    bool anyFailed = false;
    for (auto op : reductions) {
      if (failed(dropOne(op, rewriter)))
        anyFailed = true;
    }
    if (anyFailed)
      signalPassFailure();
  }

  /// Drops every zero `linalg.fill` feeding an `outs` operand of `op`.
  ///
  /// Two distinct kinds of non-rewrite, deliberately not conflated:
  ///
  ///  * **Skip, silently** — `op` is not a reduction MapReductionPartials will
  ///    ever touch (matmul or another multi-operand contraction, a
  ///    multi-result reduction, a non-trivial body). The fill is load-bearing
  ///    there because nothing downstream resets the accumulator, so leaving it
  ///    is the correct and conservative answer. Diagnosing it is not this pass's
  ///    job either: whatever cannot lower it will say so, and failing here would
  ///    make any pipeline that merely *contains* a matmul unable to run this fix.
  ///
  ///  * **Reject, with a diagnostic** — `op` IS a simple single-input reduction,
  ///    so this pass is responsible for it, but its init cannot be discarded
  ///    soundly (a combiner the scheduler derives no neutral for, or a fill that
  ///    is not that neutral). Both alternatives are wrong — dropping loses a
  ///    stated init, keeping it is refused downstream — so the error is the only
  ///    honest outcome.
  ///
  /// Operands that can be rewritten are rewritten even when a sibling is
  /// rejected, so the IR is left partially modified on failure.
  LogicalResult dropOne(linalg::LinalgOp op, IRRewriter &rewriter) {
    LogicalResult result = success();

    // Not a shape MapReductionPartials rewrites -> no zero reset downstream ->
    // the fill must stay. This is the matmul case.
    if (!isMapReductionPartialsShape(op))
      return success();

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
      Operation *payload = simpleReductionPayload(op, out);
      if (!payload)
        continue;

      std::optional<Neutral> kind = neutralKind(payload);
      if (!kind) {
        op->emitError("reduction 'outs' operand #")
            << out.getOperandNumber() << " is combined with '"
            << payload->getName().getStringRef()
            << "', which the dataflow-scheduler derives no neutral element for; "
               "it rejects linalg.fill, and re-establishes an accumulator only "
               "for the float combiners addf/subf/mulf/maximumf/minimumf, so "
               "this reduction cannot be lowered correctly at all";
        result = failure();
        continue;
      }

      if (!statesNeutral(fill.getInputs()[0], *kind)) {
        op->emitError("reduction 'outs' operand #")
            << out.getOperandNumber()
            << " is initialised by a linalg.fill of a value that is not '"
            << payload->getName().getStringRef()
            << "'s neutral element; the dataflow-scheduler rejects linalg.fill "
               "and re-establishes the accumulator at the neutral element "
               "whatever the stated init was, so this reduction cannot be "
               "lowered without discarding that stated initial value";
        result = failure();
        continue;
      }

      out.set(fill.getOutputs()[0]);
      // Only this reduction used it in the pipeline's own output, but a fill is
      // a normal value and something else may hold it.
      if (fill->use_empty())
        rewriter.eraseOp(fill);
    }
    return result;
  }
};

} // namespace

namespace mlir::triton::ktdp {

std::unique_ptr<OperationPass<ModuleOp>> createDropReductionInitFillPass() {
  return std::make_unique<DropReductionInitFillPass>();
}

} // namespace mlir::triton::ktdp
