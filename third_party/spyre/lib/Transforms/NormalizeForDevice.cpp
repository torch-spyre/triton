//===- NormalizeForDevice.cpp - Rewrite upstream IR for the device -------===//
//
// A PATTERN HOST, not one rewrite: each case is its own OpRewritePattern,
// rewriting valid upstream IR into an equivalent spelling the Spyre toolchain
// below this tree accepts. Adding a case is one pattern plus one lit case -- no
// Passes.td, CMake or pipeline change.
//
// WHAT MAY BE ADDED HERE, so the pass does not become a junk drawer. A pattern
// belongs here only if all three hold:
//
//   1. Source and target are both upstream ops (arith, linalg, tensor, scf,
//      ...). Crossing a dialect boundary is a conversion and belongs in
//      Conversion/ -- which is why the reciprocal peephole, arith.divf ->
//      spyreop.reciprocal, sits in LowerSpyreOps and not here.
//   2. The input is valid IR that a downstream consumer refuses or mis-lowers,
//      not a defect. A defect is a bug in whatever produced it, and patching it
//      here hides that.
//   3. The rewrite names the downstream behaviour it is for -- the tool and the
//      component -- so a reader can check whether it still holds.
//
// Runs at the head of the `spyrecode` stage, not in `ttir`->`ktir`. Passes.td
// states the ordering constraint; Pipeline.cpp has the placement.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_NORMALIZEFORDEVICE
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

//===----------------------------------------------------------------------===//
// arith.maxnumf -> arith.maximumf, arith.minnumf -> arith.minimumf
//===----------------------------------------------------------------------===//

/// Replaces `Source` with `Target`, carrying both operands and the fastmath
/// attribute across. Applies wherever the op appears -- a linalg.generic body, a
/// linalg.reduce combiner, or plain scalar arithmetic.
///
/// Usable for any pair of binary `arith` float ops: the only ODS shape it reads
/// is two same-typed operands, one result and an optional fastmath attribute.
template <typename Source, typename Target>
struct ReplaceBinaryFloatOp : public OpRewritePattern<Source> {
  using OpRewritePattern<Source>::OpRewritePattern;

  LogicalResult matchAndRewrite(Source op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Target>(op, op.getLhs(), op.getRhs(),
                                        op.getFastmathAttr());
    return success();
  }
};

struct NormalizeForDevicePass
    : public mlir::triton::spyre::impl::NormalizeForDeviceBase<
          NormalizeForDevicePass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // One line per rewrite: the only edit a new case needs outside its own
    // pattern.
    patterns.add<ReplaceBinaryFloatOp<arith::MaxNumFOp, arith::MaximumFOp>,
                 ReplaceBinaryFloatOp<arith::MinNumFOp, arith::MinimumFOp>>(ctx);

    // The greedy driver, and not a walk: rewriting during a walk erases the op
    // the walk is standing on.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createNormalizeForDevicePass() {
  return std::make_unique<NormalizeForDevicePass>();
}

} // namespace mlir::triton::spyre
