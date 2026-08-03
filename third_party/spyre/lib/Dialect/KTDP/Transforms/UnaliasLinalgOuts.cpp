//===- UnaliasLinalgOuts.cpp - Un-alias linalg outs from ins --------------===//
//
// Replaces any `outs` operand of a linalg op that is the same SSA value as one
// of that op's `ins` operands with a fresh `tensor.empty()` of the same type.
//
// Why this is needed:
//   A destination-passing-style (DPS) linalg op takes its result buffers as
//   `outs` operands. Upstream MLIR's --convert-elementwise-to-linalg reuses the
//   first operand whose type matches the result as the `outs` buffer instead of
//   creating a fresh one, and offers no flag to suppress that. The resulting op
//   has the same value in both `ins` and `outs`. Downstream, the Spyre
//   dataflow-scheduler's three-stage-pipeline construction maps each operand to
//   a freshly loaded value; a value appearing twice is mapped twice, the second
//   mapping overwrites the first, and the result is invalid SSA ordering.
//
// Why the rewrite is sound:
//   An `outs` operand serves two roles: it supplies the shape of the result,
//   and it supplies the initial value of each output element. tensor.empty()
//   preserves the first role exactly. The second role only matters if the op
//   body actually reads that initial value — an accumulating op such as
//   linalg.reduce or linalg.matmul does, an elementwise op does not, because it
//   writes every output element before any read of it.
//   LinalgOp::payloadUsesValueFromOperand answers that question per operand, so
//   only operands whose initial value is provably dead are rewritten.
//
// Algorithm:
//   1. Collect all linalg ops (collect-then-rewrite, to keep the walk cursor
//      valid against the tensor.empty ops inserted during rewriting).
//   2. For each op, for each `outs` operand that also appears in `ins`:
//      a. Reject  — body reads the initial value, or the type is not a
//                   statically shaped tensor.
//      b. Rewrite — point the operand at a fresh tensor.empty of same type.
//
// An aliased `outs` that cannot be rewritten is reported as an error rather
// than skipped: leaving it in place crashes the scheduler much later, with
// nothing pointing back to here.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_UNALIASLINALGOUTS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

struct UnaliasLinalgOutsPass
    : public mlir::triton::ktdp::impl::UnaliasLinalgOutsBase<
          UnaliasLinalgOutsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect all linalg ops first (collect-then-rewrite).
    SmallVector<linalg::LinalgOp> ops;
    mod.walk([&](linalg::LinalgOp op) { ops.push_back(op); });

    // Keep going after a rejection so one run reports every op it cannot
    // handle. Bailing out on the first would cost the user one rebuild per
    // aliased op.
    bool anyFailed = false;
    for (auto op : ops) {
      if (failed(unaliasOne(op, rewriter)))
        anyFailed = true;
    }
    if (anyFailed)
      signalPassFailure();
  }

  /// Un-aliases every `outs` operand of `op` that shares its SSA value with an
  /// `ins` operand. Returns failure if any aliased operand cannot be rewritten,
  /// after emitting one diagnostic on `op` per such operand. Operands that can
  /// be rewritten are rewritten even when a sibling operand is rejected, so the
  /// IR is left partially modified on failure; callers must not rely on the
  /// module being unchanged when this returns failure.
  LogicalResult unaliasOne(linalg::LinalgOp op, IRRewriter &rewriter) {
    // The `ins` operands as a set, so alias testing is a lookup rather than a
    // scan per output. Built once: setting an `outs` operand below never
    // changes which values are in `ins`.
    llvm::SmallPtrSet<Value, 4> ins(op.getDpsInputs().begin(),
                                    op.getDpsInputs().end());
    if (ins.empty())
      return success();

    rewriter.setInsertionPoint(op);
    LogicalResult result = success();
    for (OpOperand &out : op.getDpsInitsMutable()) {
      Value outValue = out.get();
      if (!ins.contains(outValue))
        continue;

      if (op.payloadUsesValueFromOperand(&out)) {
        op->emitError("linalg 'outs' operand #")
            << out.getOperandNumber()
            << " is the same value as an 'ins' operand and the op body "
               "reads its initial value; replacing it with tensor.empty "
               "would change the result";
        result = failure();
        continue;
      }

      // linalg on tensors is the only form the Spyre pipeline produces — it
      // runs entirely before bufferization. A memref `outs` would need a
      // different replacement (an allocation), so reject rather than skip.
      auto tensorType = dyn_cast<RankedTensorType>(outValue.getType());
      if (!tensorType) {
        op->emitError("linalg 'outs' operand #")
            << out.getOperandNumber()
            << " aliases an 'ins' operand but has non-tensor type "
            << outValue.getType() << "; expected a ranked tensor";
        result = failure();
        continue;
      }
      // Each dynamic dimension would need its own size operand on the
      // tensor.empty. The pipeline is fully static, so name what is missing
      // instead of emitting an under-specified op.
      if (!tensorType.hasStaticShape()) {
        op->emitError("linalg 'outs' operand #")
            << out.getOperandNumber()
            << " aliases an 'ins' operand but has dynamic shape " << tensorType
            << "; dynamic sizes are not supported";
        result = failure();
        continue;
      }

      out.set(tensor::EmptyOp::create(rewriter, op.getLoc(),
                                      tensorType.getShape(),
                                      tensorType.getElementType()));
    }
    return result;
  }
};

} // namespace

namespace mlir::triton::ktdp {

std::unique_ptr<OperationPass<ModuleOp>> createUnaliasLinalgOutsPass() {
  return std::make_unique<UnaliasLinalgOutsPass>();
}

} // namespace mlir::triton::ktdp
