//===- ConvertSpyreOpToKTDP.cpp - tts.spyre_op -> ktdp.* -------------------===//
//
// Converts `tts.spyre_op "name"` to its device-neutral `ktdp.*` mirror. A 1:1
// rename, not a staging-and-recovery problem: `tts.spyre_op` carries no
// region and no fallback body (see TTSOps.td), so there is nothing to inline
// and nothing to tag before this pass can fire.
//
// Deliberately a greedy rewrite, not a dialect conversion -- the same driver
// style `LowerSpyreOps` uses, for the same reason: there is exactly one rule,
// rooted on `tts.spyre_op`, and nothing else in the module competes for it.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_CONVERTSPYREOPTOKTDP
#include "Conversion/TritonToKTIR/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

/// `tts.spyre_op "gelu" (%x) -> ktdp.gelu %x`. Hard-coded to the one
/// registered name today, matching `SpyreOpOp::verify()` -- generalizes into
/// a name-to-op-class table once a second intrinsic exists.
struct ConvertSpyreOpGeLU
    : public OpRewritePattern<mlir::triton::tts::SpyreOpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(mlir::triton::tts::SpyreOpOp op,
                  PatternRewriter &rewriter) const override {
    if (op.getName() != "gelu")
      return failure();
    rewriter.replaceOpWithNewOp<ktdp::GeLUOp>(op, op.getType(),
                                              op.getInput());
    return success();
  }
};

struct ConvertSpyreOpToKTDPPass
    : public mlir::triton::spyre::impl::ConvertSpyreOpToKTDPBase<
          ConvertSpyreOpToKTDPPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertSpyreOpGeLU>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createConvertSpyreOpToKTDPPass() {
  return std::make_unique<ConvertSpyreOpToKTDPPass>();
}

} // namespace mlir::triton::spyre
