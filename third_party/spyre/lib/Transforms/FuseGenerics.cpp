//===- FuseGenerics.cpp - Combine linalg.generic producers into consumers -===//
//
// Combines a producer linalg.generic into the consumer that reads it, by two
// patterns. Both are rooted on the consumer and look at one `ins` at a time;
// they differ in what the producer computes for the value the consumer reads:
//
//   AbsorbForwardingGeneric  nothing: the producer yields one of its `ins`
//                            block arguments. The consumer's operand is
//                            repointed at that input, under the composed map.
//   GroupComputes            an op, the anchor: the producer is fused into the
//                            consumer when the anchor and every reader of it
//                            form a group LowerSpyreOps lowers as one intrinsic
//                            (formsLowerableGroup).
//
// After the greedy driver converges, a separate step, legalizeAfterFusion,
// checks what the patterns left. Today its one check,
// checkNoBooleanBetweenGenerics, reports every `i1` tensor still passed from
// one generic to another.
//
// Both patterns decline a producer that has a reduction loop. Both then check
// the consumer's operand list as it would be after the rewrite, before mutating
// anything (checkOperandRewrite):
//
//   - every loop dim of the consumer is named by some operand map;
//   - no value is both an `ins` and an `outs` of the consumer.
//
// Every producer map has every result a bare dim or a constant, and composing
// such maps through a permutation keeps that, so no map is checked for it.
//
// HOW TO ADD A PATTERN. A pattern matches on its PRODUCER'S KIND: the op that
// defines the operand, or what the producer's body computes for it. Never on
// the operand's type, nor on anything downstream. A new pattern names a
// producer kind no existing pattern matches; one whose kind overlaps an
// existing pattern's is a new case of that pattern. Its lit file feeds it the
// other patterns' producers and checks that it does not fire. A new group shape
// is not a pattern here: it is a match function in LowerSpyreOps, which
// formsLowerableGroup then answers for.
//
// `--debug-only=fuse-generics` traces, one line per decision, each operand
// skipped before any check, each decline with the check that declined it, each
// group formed, each rewrite, and each `i1` legalizeAfterFusion reports.
//
//===----------------------------------------------------------------------===//

#include "Transforms/LowerSpyreOps.h"
#include "Transforms/Passes.h"
#include "Utils/GenericOperands.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

#define DEBUG_TYPE "fuse-generics"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_FUSEGENERICS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

using OperandList = SmallVector<std::pair<Value, AffineMap>>;

/// Fills `ins` and `outs` with `op`'s ins and outs operands, in order, each
/// paired with its indexing map.
void operandLists(linalg::GenericOp op, OperandList &ins, OperandList &outs) {
  for (OpOperand *in : op.getDpsInputOperands())
    ins.emplace_back(in->get(), op.getMatchingIndexingMap(in));
  for (OpOperand &out : op.getDpsInitsMutable())
    outs.emplace_back(out.get(), op.getMatchingIndexingMap(&out));
}

/// Returns the value `producer`'s linalg.yield yields for its result `result`.
Value yieldedFor(linalg::GenericOp producer, OpResult result) {
  return producer.getBlock()->getTerminator()->getOperand(
      result.getResultNumber());
}

/// Returns the `ins` block argument of `producer`'s body that it yields for
/// `result`, or null when it yields an op result, an `outs` block argument or
/// a value defined outside the body.
BlockArgument forwardedInput(linalg::GenericOp producer, OpResult result) {
  auto arg = dyn_cast<BlockArgument>(yieldedFor(producer, result));
  if (!arg || arg.getOwner() != producer.getBlock() ||
      arg.getArgNumber() >= producer.getNumDpsInputs())
    return {};
  return arg;
}

/// Returns true when every loop of `op` is parallel.
bool isAllParallel(linalg::GenericOp op) {
  return op.getNumParallelLoops() == op.getNumLoops();
}

/// Traces, under `--debug-only=fuse-generics`, that `pattern` declined
/// `operand` because of `why`.
void traceDecline(StringRef pattern, OpOperand *operand, const llvm::Twine &why) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << pattern << ": operand "
                          << operand->getOperandNumber() << " of generic at "
                          << operand->getOwner()->getLoc() << ": declined, "
                          << why << "\n");
}

/// Traces, under `--debug-only=fuse-generics`, that `pattern` skipped `operand`
/// before any check, because of `why`.
void traceSkip(StringRef pattern, OpOperand *operand, const llvm::Twine &why) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << pattern << ": operand "
                          << operand->getOperandNumber() << " of generic at "
                          << operand->getOwner()->getLoc() << ": skipped, "
                          << why << "\n");
}

//===----------------------------------------------------------------------===//
// AbsorbForwardingGeneric
//===----------------------------------------------------------------------===//

/// Absorbs a producer generic that only forwards one of its inputs into the
/// operand map of the consumer that reads it.
///
/// Before:
///
///   %b = linalg.generic {indexing_maps = [(d0, d1) -> (d0),
///                                         (d0, d1) -> (d0, d1)]}
///          ins(%s : tensor<64xf16>) outs(%e : tensor<64x128xf16>) {
///     ^bb0(%in: f16, %out: f16): linalg.yield %in : f16 }
///   %r = linalg.generic {indexing_maps = [(d0, d1) -> (d0, d1), ...]}
///          ins(%b : tensor<64x128xf16>) ...
///
/// After:
///
///   %r = linalg.generic {indexing_maps = [(d0, d1) -> (d0), ...]}
///          ins(%s : tensor<64xf16>) ...
///
/// Conditions, for a consumer `ins` operand that is result `r` of producer `P`:
///
///   - `P` is a linalg.generic;
///   - `P` yields, for `r`, one of its own `ins` block arguments, `x`;
///   - `P` has no reduction loop;
///   - `P`'s indexing map for `r`, `R`, is a permutation;
///   - after the rewrite, every loop dim of the consumer is named by some
///     operand map;
///   - after the rewrite, no value is both an `ins` and an `outs` of the
///     consumer.
///
/// The rest of `P`'s body is not read.
///
/// Rewrite: the operand is set to `P`'s input for `x`, and its map to
/// `A.compose(inversePermutation(R)).compose(C)`, where `A` is `P`'s map for
/// `x` and `C` the consumer's map for the operand. The greedy driver erases `P`
/// once it is dead.
struct AbsorbForwardingGeneric : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!consumer.hasPureTensorSemantics())
      return failure();
    for (OpOperand *operand : consumer.getDpsInputOperands()) {
      auto result = dyn_cast<OpResult>(operand->get());
      auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
      if (!producer) {
        traceSkip("AbsorbForwardingGeneric", operand,
                  "the producer is not a linalg.generic");
        continue;
      }
      BlockArgument arg = forwardedInput(producer, result);
      if (!arg) {
        traceSkip("AbsorbForwardingGeneric", operand,
                  "the producer does not yield one of its ins block arguments");
        continue;
      }
      if (!isAllParallel(producer)) {
        traceDecline("AbsorbForwardingGeneric", operand,
                     "the producer has a reduction loop");
        continue;
      }
      AffineMap r = producer.getIndexingMapMatchingResult(result);
      if (!r.isPermutation()) {
        traceDecline("AbsorbForwardingGeneric", operand,
                     "the producer's result map is not a permutation");
        continue;
      }
      OpOperand *x = producer.getDpsInputOperand(arg.getArgNumber());
      AffineMap c = consumer.getMatchingIndexingMap(operand);
      AffineMap composed = producer.getMatchingIndexingMap(x)
                               .compose(inversePermutation(r))
                               .compose(c);

      OperandList ins, outs;
      operandLists(consumer, ins, outs);
      unsigned index = operand->getOperandNumber();
      ins[index] = {x->get(), composed};
      std::string why;
      if (failed(triton::spyre::checkOperandRewrite(
              ins, outs, consumer.getNumLoops(), why))) {
        traceDecline("AbsorbForwardingGeneric", operand, why);
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] AbsorbForwardingGeneric: operand "
                 << index << " of generic at " << consumer.getLoc()
                 << ": reads the input of the generic at " << producer.getLoc()
                 << " under " << composed << "\n");
      SmallVector<AffineMap> maps = consumer.getIndexingMapsArray();
      maps[index] = composed;
      rewriter.modifyOpInPlace(consumer, [&] {
        operand->set(x->get());
        consumer.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      });
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// GroupComputes
//===----------------------------------------------------------------------===//

/// Fuses a producer generic into its consumer when the op the producer yields
/// and the consumer ops that read it form a group LowerSpyreOps lowers as one
/// intrinsic.
///
/// Before:
///
///   %c = linalg.generic ... ins(%a, %b : ...) {
///     ^bb0(%x: f16, %y: f16, %o: i1):
///       %0 = arith.cmpf ogt, %x, %y : f16
///       linalg.yield %0 : i1 }
///   %m = linalg.generic ... ins(%c : tensor<64xi1>) {
///     ^bb0(%in: i1, %o: f16):
///       %1 = arith.uitofp %in : i1 to f16
///       linalg.yield %1 : f16 }
///
/// After:
///
///   %m = linalg.generic ... ins(%a, %b : ...) {
///     ^bb0(%x: f16, %y: f16, %o: f16):
///       %0 = arith.cmpf ogt, %x, %y : f16
///       %1 = arith.uitofp %0 : i1 to f16
///       linalg.yield %1 : f16 }
///
/// Conditions, for a consumer `ins` operand that is result `r` of producer `P`:
///
///   - `P` is a linalg.generic;
///   - `P` yields, for `r`, the result of an op in its body, the anchor;
///   - `P` has no reduction loop;
///   - linalg::areElementwiseOpsFusable holds for the operand;
///   - the consumer reads the operand's block argument at least once, and
///     every op reading it, with the anchor, forms a group
///     (formsLowerableGroup);
///   - when `P` has a user other than the consumer, the anchor is the only
///     non-terminator op in `P`'s body;
///   - after the rewrite, every loop dim of the consumer is named by some
///     operand map;
///   - after the rewrite, no value is both an `ins` and an `outs` of the
///     consumer.
///
/// Rewrite: linalg::fuseElementwiseOps, then
/// linalg::deduplicateOperandsAndRemoveDeadResults on the fused op, which
/// removes the result kept for `P`'s other users. Those users keep reading
/// `P`.
struct GroupComputes : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!consumer.hasPureTensorSemantics())
      return failure();
    for (OpOperand *operand : consumer.getDpsInputOperands())
      if (succeeded(groupOperand(consumer, operand, rewriter)))
        return success();
    return failure();
  }

private:
  /// Checks whether `producer`, whose body yields `anchor` for `operand`, can
  /// be fused into `consumer`: every condition of the pattern after the
  /// producer and the anchor are found. Rewrites nothing. Returns failure with
  /// `why` set to the first condition that does not hold.
  static LogicalResult checkGroup(linalg::GenericOp consumer,
                                  OpOperand *operand,
                                  linalg::GenericOp producer, Operation *anchor,
                                  std::string &why) {
    if (!isAllParallel(producer)) {
      why = "the producer has a reduction loop";
      return failure();
    }
    if (!linalg::areElementwiseOpsFusable(operand)) {
      why = "not fusable under linalg::areElementwiseOpsFusable";
      return failure();
    }

    BlockArgument arg = consumer.getMatchingBlockArgument(operand);
    if (arg.use_empty()) {
      why = "the consumer does not read it";
      return failure();
    }
    for (Operation *reader : arg.getUsers()) {
      std::string reason;
      if (!triton::spyre::formsLowerableGroup(producer, anchor, reader,
                                              &reason)) {
        why = "'" + anchor->getName().getStringRef().str() + "' and its reader '" +
              reader->getName().getStringRef().str() +
              "' form no group: " + reason;
        return failure();
      }
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] GroupComputes: operand "
                              << operand->getOperandNumber() << " of generic at "
                              << consumer.getLoc() << ": '" << anchor->getName()
                              << "' and its reader '" << reader->getName()
                              << "' form a group\n");
    }

    bool shared = llvm::any_of(producer->getUsers(), [&](Operation *user) {
      return user != consumer.getOperation();
    });
    if (shared && llvm::any_of(producer.getBlock()->without_terminator(),
                               [&](Operation &op) { return &op != anchor; })) {
      why = "the producer has another user and its body holds ops outside the "
            "group";
      return failure();
    }

    // The fused op's ins: the consumer's other ins, and each producer ins
    // under A o R^-1 o C, the composition fuseElementwiseOps writes.
    OperandList ins, outs;
    operandLists(consumer, ins, outs);
    ins.erase(ins.begin() + operand->getOperandNumber());
    AffineMap toProducerLoops =
        inversePermutation(producer.getIndexingMapMatchingResult(
                               cast<OpResult>(operand->get())))
            .compose(consumer.getMatchingIndexingMap(operand));
    for (OpOperand *in : producer.getDpsInputOperands())
      ins.emplace_back(in->get(), producer.getMatchingIndexingMap(in).compose(
                                      toProducerLoops));
    return triton::spyre::checkOperandRewrite(ins, outs,
                                              consumer.getNumLoops(), why);
  }

  /// Finds the producer and the anchor of `operand`, checks the group with
  /// checkGroup, and when it holds fuses the producer into `consumer` and
  /// erases `consumer`. Returns success when it rewrote.
  static LogicalResult groupOperand(linalg::GenericOp consumer,
                                    OpOperand *operand,
                                    PatternRewriter &rewriter) {
    auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
    if (!producer) {
      traceSkip("GroupComputes", operand, "the producer is not a linalg.generic");
      return failure();
    }
    Value yielded = yieldedFor(producer, cast<OpResult>(operand->get()));
    Operation *anchor = yielded.getDefiningOp();
    if (!anchor) {
      traceSkip("GroupComputes", operand,
                "the yielded value is a block argument, not an op result");
      return failure();
    }
    if (anchor->getBlock() != producer.getBlock()) {
      traceSkip("GroupComputes", operand,
                "the yielded value is defined outside the body");
      return failure();
    }

    std::string why;
    if (failed(checkGroup(consumer, operand, producer, anchor, why))) {
      traceDecline("GroupComputes", operand, why);
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] GroupComputes: operand "
                            << operand->getOperandNumber() << " of generic at "
                            << consumer.getLoc() << ": fusing the generic at "
                            << producer.getLoc() << ", anchor '"
                            << anchor->getName() << "'\n");
    FailureOr<linalg::ElementwiseOpFusionResult> fused =
        linalg::fuseElementwiseOps(rewriter, operand);
    if (failed(fused))
      return failure();
    // The consumer's results go to the fused op; the producer's other users
    // keep reading the producer.
    for (auto [orig, repl] : fused->replacements)
      rewriter.replaceUsesWithIf(orig, repl, [&](OpOperand &use) {
        return use.get().getDefiningOp() != producer;
      });
    rewriter.eraseOp(consumer);
    (void)linalg::deduplicateOperandsAndRemoveDeadResults(
        rewriter, cast<linalg::GenericOp>(fused->fusedOp),
        /*removeOutputs=*/true);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Legalization: an i1 tensor between generics
//===----------------------------------------------------------------------===//

/// Reports, as an error at the consumer, every `ins` of element type i1 that a
/// linalg.generic defines. The notes name the op that made the i1 in the
/// producer, or the operand the producer forwards; each op reading it in the
/// consumer; and the group shapes LowerSpyreOps lowers an i1 in. Returns
/// failure when it reported anything.
LogicalResult checkNoBooleanBetweenGenerics(ModuleOp mod) {
  LogicalResult result = success();
  mod.walk([&](linalg::GenericOp consumer) {
    for (OpOperand *operand : consumer.getDpsInputOperands()) {
      if (!getElementTypeOrSelf(operand->get().getType()).isInteger(1))
        continue;
      auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
      if (!producer)
        continue;
      result = failure();
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] legalizeAfterFusion: operand "
                              << operand->getOperandNumber() << " of generic at "
                              << consumer.getLoc()
                              << ": an i1 tensor from the generic at "
                              << producer.getLoc() << ", reported\n");

      Value yielded = yieldedFor(producer, cast<OpResult>(operand->get()));
      InFlightDiagnostic diag =
          consumer.emitError()
          << "fuse-generics: operand " << operand->getOperandNumber()
          << " is an i1 tensor passed from one linalg.generic to another; no "
             "spyreop op produces or consumes i1, so nothing downstream can "
             "lower it";
      if (Operation *maker = yielded.getDefiningOp())
        diag.attachNote(maker->getLoc())
            << "made here, by '" << maker->getName() << "'";
      else if (auto arg = dyn_cast<BlockArgument>(yielded);
               arg && arg.getOwner() == producer.getBlock())
        diag.attachNote(producer.getLoc())
            << "forwarded by this linalg.generic from its operand "
            << producer.getMatchingOpOperand(arg)->getOperandNumber();
      for (Operation *reader :
           consumer.getMatchingBlockArgument(operand).getUsers())
        diag.attachNote(reader->getLoc())
            << "read here, by '" << reader->getName() << "'";
      diag.attachNote()
          << "LowerSpyreOps lowers an i1 only within a group: an arith.cmpf "
             "read by arith.uitofp, or by arith.select, at the compared width "
             "(SelectCompare, SelectWhere)";
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// After the patterns: legalization
//===----------------------------------------------------------------------===//

/// The check step run once the fusion patterns have converged. Rewrites
/// nothing; fails when any check reports an error. Its one check is
/// checkNoBooleanBetweenGenerics.
LogicalResult legalizeAfterFusion(ModuleOp mod) {
  return checkNoBooleanBetweenGenerics(mod);
}

struct FuseGenericsPass
    : public mlir::triton::spyre::impl::FuseGenericsBase<FuseGenericsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<AbsorbForwardingGeneric, GroupComputes>(ctx);
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();

    if (failed(legalizeAfterFusion(mod)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createFuseGenericsPass() {
  return std::make_unique<FuseGenericsPass>();
}

} // namespace mlir::triton::spyre
