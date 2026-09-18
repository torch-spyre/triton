//===- NormalizeFloatMinMax.cpp - maxnumf/minnumf -> maximumf/minimumf ----===//
//
// Rewrites `arith.maxnumf` to `arith.maximumf` and `arith.minnumf` to
// `arith.minimumf`, everywhere in the module.
//
// WHAT THIS COSTS. The two families are not synonyms. They differ on NaN, and
// only on NaN:
//
//     maxnumf(NaN, x) = x        the non-NaN operand wins
//     maximumf(NaN, x) = NaN     NaN propagates
//
// So this pass CHANGES THE COMPUTED RESULT of a kernel whose input contains a
// NaN. That is deliberate, and it is the cheaper of the two available wrongs:
// leaving `maxnumf` alone is wrong for EVERY input on the device, not just for
// NaN ones.
//
// WHY IT IS NEEDED. Triton's frontend and the Spyre dataflow-scheduler disagree
// about which spelling exists. `tl.max`, `tl.min`, `tl.maximum` and `tl.minimum`
// all emit the `numf` form. The scheduler's KTDF->DFIR LinalgLowering then maps
//
//     arith.maximumf -> vectorchain `max`        correct
//     arith.maxnumf  -> vectorchain `abs_max`    a different function entirely:
//                                                it compares MAGNITUDES, so
//                                                abs_max(-5.0, 1.0) = -5.0
//
// and the scheduler's MapReductionPartials, which supplies a reduction's neutral
// element, has an entry for `maximumf`, `maxnumf` and `minimumf` but NONE for
// `minnumf` -- so a `tl.min` reduce does not mis-compute, it fails to compile.
// Between the two, every Triton float min/max reaching the device today is
// either silently wrong or refused.
//
// WHY A PASS AND NOT A TWEAK IN LowerComputeOps. The mis-lowering is a property
// of the op, not of the context it appears in. LowerComputeOps' `ConvertTTReduce`
// clones a `tt.reduce` combiner and is where a reduce's `maxnumf` comes from, so
// fixing it there would cover `tl.max`/`tl.min` -- but the identical `abs_max`
// mis-lowering hits an ELEMENTWISE `tl.maximum(a, b)`, which arrives through
// ConvertElementwiseToLinalg and never passes through that pattern. One
// module-wide rewrite covers both and puts the NaN trade in one place.
//
// ORDERING. After LowerComputeOps (a `tt.reduce` combiner is not an arith op in a
// linalg body until then) and before DropReductionInitFill, which reads a
// reduction's combiner op to decide whether its stated init may be discarded --
// `maxnumf` and `maximumf` are separate cases both there and in the scheduler's
// neutral-element table, so the spelling has to be settled before either looks.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_NORMALIZEFLOATMINMAX
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// Replaces `op` with `Target`, carrying its operands and fastmath flags over.
///
/// Both families are `arith`'s binary float ops with identical ODS shape --
/// two same-typed operands, one result, one optional fastmath attribute -- so the
/// rewrite is a substitution of the op name and nothing else. Templated because
/// the max and min cases would otherwise be the same six lines twice.
template <typename Target, typename Source>
void replaceWith(Source op, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  auto replacement = Target::create(rewriter, op.getLoc(), op.getLhs(),
                                    op.getRhs(), op.getFastmathAttr());
  rewriter.replaceOp(op, replacement.getResult());
}

struct NormalizeFloatMinMaxPass
    : public mlir::triton::ktdp::impl::NormalizeFloatMinMaxBase<
          NormalizeFloatMinMaxPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect before rewriting: replaceOp erases the op the walk is standing on.
    SmallVector<arith::MaxNumFOp> maxes;
    SmallVector<arith::MinNumFOp> mins;
    mod.walk([&](Operation *op) {
      if (auto max = dyn_cast<arith::MaxNumFOp>(op))
        maxes.push_back(max);
      else if (auto min = dyn_cast<arith::MinNumFOp>(op))
        mins.push_back(min);
    });

    for (auto max : maxes)
      replaceWith<arith::MaximumFOp>(max, rewriter);
    for (auto min : mins)
      replaceWith<arith::MinimumFOp>(min, rewriter);
  }
};

} // namespace

namespace mlir::triton::ktdp {

std::unique_ptr<OperationPass<ModuleOp>> createNormalizeFloatMinMaxPass() {
  return std::make_unique<NormalizeFloatMinMaxPass>();
}

} // namespace mlir::triton::ktdp
