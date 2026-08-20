//===- DropReductionInitFill.cpp - Drop a zero reduction init fill --------===//
//
// Removes a `linalg.fill` of zero that supplies the `outs` of a reduction,
// repointing that `outs` at the `tensor.empty` the fill wrote into.
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
// Why the rewrite is sound, and why the gate is narrow:
//   A reduction's payload READS its init operand — `linalg.reduce` names the
//   operand `$inits` and its own ODS example writes `arith.addf %out, %in` — and
//   `tensor.empty` has explicitly "unspecified" contents. So the rewritten IR is
//   only well-defined because something downstream overwrites the accumulator
//   before it is read: MapReductionPartials' "Step 2: zero-fill the accumulator".
//   That reset is a hardcoded 0.0. The rewrite is therefore sound in exactly one
//   situation — when that pass will run on this op, and the value it writes is
//   the value the fill states.
//
//   Both halves of that are checked, because neither implies the other:
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
//     isZeroNeutralCombiner        addf/subf only. An allowlist, not a
//                                  neutral-is-zero test: integer combiners have a
//                                  zero neutral but abort the scheduler, whose
//                                  reset is built as a float attribute, and
//                                  maxnumf mis-lowers to abs_max.
//     isConstantZero               the stated init matches what the reset writes.
//
//   Failing the first two means the fill is load-bearing, so it is LEFT ALONE and
//   no diagnostic is emitted — this pass is not responsible for ops it cannot
//   reason about, and failing on them would break any pipeline that merely
//   contains a matmul. Failing the last two means this IS our op but its init
//   cannot be honoured (mulf wants 1.0, max wants -inf); both alternatives are
//   wrong, so it is reported.
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
//      b. Reject  — the combiner's neutral is not zero, or the fill is non-zero.
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

/// True iff `v` is defined by a constant whose value is zero, of either a float
/// or an integer type. m_AnyZeroFloat accepts -0.0 as well as +0.0; both are
/// additive identities, and both are what the scheduler's reset writes.
bool isConstantZero(Value v) {
  return matchPattern(v, m_AnyZeroFloat()) || matchPattern(v, m_Zero());
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

/// True iff `payload` is a combiner whose neutral is zero AND which the
/// scheduler lowers correctly with a zero reset.
///
/// Deliberately an allowlist of two, not "does its neutral happen to be zero".
/// These are the only combiners that reach a correct answer today: `mul` needs
/// 1.0, `max`/`min` need -/+inf, and `maxnumf` additionally mis-lowers to
/// `abs_max`. Integer combiners (`addi`, `ori`) have a zero neutral but abort the
/// scheduler outright, because its reset is built as a float attribute — so
/// restricting to the float ops turns that abort into a diagnostic here.
bool isZeroNeutralCombiner(Operation *payload) {
  return isa<arith::AddFOp, arith::SubFOp>(payload);
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
  ///    soundly (non-zero fill, or a combiner whose neutral is not zero). Both
  ///    alternatives are wrong — dropping loses a stated init, keeping it is
  ///    refused downstream — so the error is the only honest outcome.
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

      if (!isZeroNeutralCombiner(payload)) {
        op->emitError("reduction 'outs' operand #")
            << out.getOperandNumber() << " is combined with '"
            << payload->getName().getStringRef()
            << "', whose neutral element is not zero; the dataflow-scheduler "
               "resets a reduction accumulator to zero regardless of the "
               "combiner, so this reduction cannot be lowered correctly at all";
        result = failure();
        continue;
      }

      if (!isConstantZero(fill.getInputs()[0])) {
        op->emitError("reduction 'outs' operand #")
            << out.getOperandNumber()
            << " is initialised by a linalg.fill of a non-zero value; the "
               "dataflow-scheduler rejects linalg.fill and resets a reduction "
               "accumulator to zero regardless of the combiner, so this "
               "reduction cannot be lowered without discarding its stated "
               "initial value";
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
