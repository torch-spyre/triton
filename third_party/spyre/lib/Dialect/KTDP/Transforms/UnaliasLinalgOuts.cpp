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
//   has the same value in both `ins` and `outs`.
//
//   Downstream, the dataflow-scheduler consumes that IR and cannot handle it:
//
//     repo: https://github.com/torch-spyre/dataflow-scheduler
//     pass: ConstructThreeStagePipeline
//     file: lib/Conversion/frontend/KTIRToScheduleIR/
//           ConstructThreeStagePipeline.cpp
//     func: ConstructThreeStagePipelinePass::createComputeOps
//
//   (Named by symbol rather than deep-linked by line: a blob URL with a line
//   anchor goes stale on the next edit to that file.)
//
//   createComputeOps builds one mlir::IRMapping for the compute op it is about
//   to clone. First it maps each `ins` operand to the ktdf.read_from_fifo result
//   that supplies the loaded tile. Then it maps the first `outs` operand to a
//   fresh tensor.empty. When `ins` and `outs` hold the same value, that second
//   map() call overwrites the first on the same key, so the cloned op reads the
//   uninitialized tensor.empty instead of the loaded data.
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
//   1. Walk the module and collect the linalg ops that have at least one `outs`
//      operand aliasing an `ins` operand. Collect first and rewrite after: the
//      tensor.empty ops inserted while rewriting would invalidate the walk
//      cursor.
//   2. For each collected op, for each `outs` operand that also appears in
//      `ins`:
//      a. Reject  — body reads the initial value, or the type is not a ranked
//                   tensor.
//      b. Rewrite — point the operand at a fresh tensor.empty of same type. A
//                   dynamic dimension is sized by a tensor.dim on the operand
//                   being replaced.
//
// An aliased `outs` that cannot be rewritten is reported as an error rather
// than skipped: leaving it in place crashes the scheduler much later, with
// nothing pointing back to here.
//
// Trace what the pass collected and rewrote with
// `spyre-triton-opt ... -debug-only=unalias-linalg-outs` (needs an LLVM built
// with assertions).
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "unalias-linalg-outs"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_UNALIASLINALGOUTS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// Stack slots in the `ins`-operand sets below before they spill to the heap.
/// An allocation hint only; any operand count behaves identically.
constexpr unsigned kInlineInsCapacity = 4;

struct UnaliasLinalgOutsPass
    : public mlir::triton::ktdp::impl::UnaliasLinalgOutsBase<
          UnaliasLinalgOutsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect the ops to rewrite before rewriting any of them: inserting a
    // tensor.empty during a walk invalidates the walk cursor.
    SmallVector<linalg::LinalgOp> ops;
    mod.walk([&](linalg::LinalgOp op) {
      if (hasAliasedInit(op))
        ops.push_back(op);
    });

    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "] collected " << ops.size()
                   << " linalg op(s) with an 'outs' operand aliasing an 'ins' "
                      "operand\n";
      for (linalg::LinalgOp op : ops)
        llvm::dbgs() << "  " << op->getName() << " at " << op->getLoc() << "\n";
    });

    // Keep going after a rejection so one run reports every op it cannot
    // handle. Bailing out on the first would cost the user one rebuild per
    // aliased op.
    bool anyFailed = false;
    for (auto op : ops) {
      if (failed(rewriteOrRejectAliasedInits(op, rewriter)))
        anyFailed = true;
    }
    if (anyFailed)
      signalPassFailure();
  }

  /// Returns true if `op` has at least one `outs` operand holding the same SSA
  /// value as one of its `ins` operands — the only situation this pass rewrites.
  /// Used to filter the collection walk so that every collected op is one
  /// rewriteOrRejectAliasedInits will actually modify or reject.
  static bool hasAliasedInit(linalg::LinalgOp op) {
    // getDpsInputs/getDpsInits return by value, so bind the vectors rather
    // than a ValueRange view into a temporary.
    SmallVector<Value> inputs = op.getDpsInputs();
    if (inputs.empty())
      return false;
    llvm::SmallPtrSet<Value, kInlineInsCapacity> ins(inputs.begin(),
                                                     inputs.end());
    return llvm::any_of(op.getDpsInits(),
                        [&](Value init) { return ins.contains(init); });
  }

  /// Visits every `outs` operand of `op` that shares its SSA value with an
  /// `ins` operand and either rewrites it to a fresh tensor.empty or rejects it.
  /// An operand is rejected when the rewrite would change the op's result — the
  /// body reads the initial value — or when its type is one this pass cannot
  /// build an empty for. Each rejection emits a diagnostic on `op` and makes the
  /// return value failure.
  ///
  /// Rejecting one operand does not stop the others from being rewritten, so the
  /// IR is left partially modified on failure; callers must not rely on the
  /// module being unchanged when this returns failure.
  ///
  /// Precondition: `hasAliasedInit(op)` holds. Callers filter with it so that
  /// this function is never invoked on an op it would leave untouched.
  LogicalResult rewriteOrRejectAliasedInits(linalg::LinalgOp op,
                                            IRRewriter &rewriter) {
    assert(hasAliasedInit(op) && "rewriteOrRejectAliasedInits requires an op "
                                 "with an aliased 'outs' operand");

    // The `ins` operands as a set, so alias testing is a lookup rather than a
    // scan per output. Built once: setting an `outs` operand below never
    // changes which values are in `ins`. getDpsInputs returns by value, so hold
    // the vector — calling .begin()/.end() on it directly would produce
    // iterators into two separate destroyed temporaries.
    SmallVector<Value> inputs = op.getDpsInputs();
    llvm::SmallPtrSet<Value, kInlineInsCapacity> ins(inputs.begin(),
                                                     inputs.end());

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
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] " << op->getName() << ": repointing 'outs' operand #"
                 << out.getOperandNumber() << " at a fresh tensor.empty of type "
                 << tensorType << "\n");

      // A dynamically shaped outs needs one size operand per `?` on the
      // replacement tensor.empty, measured off the value being replaced.
      // outValue is an `ins` operand of this same op, so it is defined before
      // the op and dominates the insertion point. A fully static type needs no
      // size operands, so it is built from the type alone.
      Value empty =
          tensorType.hasStaticShape()
              ? mlir::triton::ktdp::createEmptyTensor(rewriter, op.getLoc(),
                                                      tensorType)
              : mlir::triton::ktdp::createEmptyTensor(rewriter, op.getLoc(),
                                                      tensorType, outValue);
      out.set(empty);
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
