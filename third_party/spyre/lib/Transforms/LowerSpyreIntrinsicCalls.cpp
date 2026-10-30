//===- LowerSpyreIntrinsicCalls.cpp - Select what tl.spyre_intrinsic asks ===//
//
// REQUESTED SELECTION for the Spyre device: a compute body that is one
// `tl.spyre_intrinsic` call site's whole fallback becomes the op the call
// names, applied to the call operands in call order. The body arrives as the
// ops LowerTTSMarkers inlined out of the `tts.spyre_intrinsic` op, each
// carrying the `tts.spyre_intrinsic` attribute, and FuseComputeAndDataMovement
// has fused the ops of one call site into one linalg.generic.
//
// A pass of its own, apart from LowerSpyreOps, because the two have opposite
// failure policies. LowerSpyreOps selects IMPLICITLY, from what arith and math
// compute, and everything it does not match flows through to the device
// toolchain, which is what knows what it can take. This pass selects what an
// author asked for BY NAME, and anything it does not select is an error: the
// fallback is the request's meaning on a reader that runs KTIR, not a
// substitute for the op the author named on the device. So every call site is
// selected or reported, and nothing that carries the attribute leaves the pass.
//
// It runs before LowerSpyreOps, and the order is the pipeline's
// (Pipeline.cpp): a fallback's ops are arith and math, which LowerSpyreOps
// would otherwise select one by one, a `math.exp` as spyreop.exp. Nothing here
// depends on LowerSpyreOps running next: the inputs a replaced body stops
// reading are removed here.
//
// Three steps over the module:
//
//   1. selectCallSites -- per generic whose body getBodyCallSite names one call
//      site, matchCallSite reads which input is which call operand off the
//      members' `operands` bindings, and selectCallSite replaces the body with
//      the target, built by name and held to its own verifier;
//   2. rejectSurvivingHints -- a `tts.spyre_intrinsic` attribute left anywhere
//      is an error, once per call site, naming the target;
//   3. upstream's unused-operand cleanup, greedy, for the `ins` operands, block
//      arguments and indexing maps a selected body no longer reads.
//
// `--debug-only=lower-spyre-intrinsic-calls` prints one line per body selected.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "lower-spyre-intrinsic-calls"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERSPYREINTRINSICCALLS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

/// What the call-site rule needs from a body it replaces.
struct CallSiteMatch {
  DictionaryAttr hint;
  /// Call operand k -> the `ins` block argument holding it, in call order.
  SmallVector<BlockArgument> inputs;
  SmallVector<Operation *> members;
};

/// Whether block arguments `a` and `b` of `generic` read one element of one
/// value: the same `ins` operand through the same indexing map. Two ops of a
/// call site reading one call operand from two generics give the fused body two
/// such block arguments unless the fusion merges them, which it does for the
/// `ins` it sees are equal; a body that kept both still has one input there.
bool readSameInput(linalg::GenericOp generic, BlockArgument a,
                   BlockArgument b) {
  OpOperand *operandA = generic.getMatchingOpOperand(a);
  OpOperand *operandB = generic.getMatchingOpOperand(b);
  return operandA->get() == operandB->get() &&
         generic.getMatchingIndexingMap(operandA) ==
             generic.getMatchingIndexingMap(operandB);
}

/// What `generic`'s body, the call site `hint` names, reads for each of its
/// call operands, or failure, reported, when the body does not say.
///
/// FuseComputeAndDataMovement has already checked that the body holds one call
/// site, whole, and nothing else but constants, so what is left to find is the
/// inputs. Not by the block arguments' order, which after
/// ConvertElementwiseToLinalg and fusion is whatever order the `ins` ended up
/// in -- `b / a` converts to `ins(%b, %a)`, block argument 0 being call operand
/// 1 -- but from the binding LowerTTSMarkers wrote when it was exact: each
/// member's `operands`, entry j naming the call operand that member's operand j
/// is.
///
/// On a non-commutative member the binding is positional: operand j is call
/// operand `operands[j]`. On a commutative member it is read positionally when
/// every position it lists holds an `ins` block argument, and otherwise as a
/// set: the member's operands that are `ins` block arguments, whatever their
/// positions, are the call operands it lists, one each. A set reading assigns
/// only when the member lists one call operand, possibly more than once, since
/// then every assignment is the same one; listing two different ones, it is
/// refused rather than guessed. So a commutative member either binds by the
/// positions it was tagged with, or binds every input it reads to the one call
/// operand it names, and in both cases its bindings are checked against the
/// other members' like any.
///
/// Gathered over every member, and refused, naming the target and the call
/// operand, when it
///
///   - binds call operand k to an operand that is not an `ins` block argument
///     of this generic, which nothing between the two passes makes it;
///   - binds, on a commutative member read as a set, a number of call operands
///     other than the number of `ins` block arguments the member reads, or two
///     different call operands;
///   - binds call operand k to two block arguments reading different values,
///     two reading one value through one map being one input (readSameInput);
///   - binds no block argument to some call operand k below `arity`, which is
///     a fallback that does not read that operand, or whose reading folded
///     away: the target would have nothing to take it from.
///
/// A bound block argument whose `ins` operand is a constant is used like any
/// other: the binding says it is call operand k, so it is a call operand that
/// folded to a constant after LowerTTSMarkers refused the ones constant there,
/// and the target applied to it still computes what was asked. An unbound one
/// -- a constant the fusion did not fold into the body -- is not read for the
/// target at all, and goes with the members' other unused inputs.
FailureOr<CallSiteMatch> matchCallSite(linalg::GenericOp generic,
                                       DictionaryAttr hint) {
  StringRef target = triton::tts::getSpyreIntrinsicTarget(hint);
  int64_t arity = triton::tts::getSpyreIntrinsicArity(hint);
  Block *body = generic.getBlock();
  auto yield = cast<linalg::YieldOp>(body->getTerminator());
  auto refuse = [&](Location loc) {
    return mlir::emitError(loc)
           << "lower-spyre-intrinsic-calls: tl.spyre_intrinsic(\"" << target
           << "\"): ";
  };

  CallSiteMatch match;
  match.hint = hint;
  match.inputs.assign(arity, BlockArgument());
  for (Operation &op : body->without_terminator())
    if (!op.hasTrait<OpTrait::ConstantLike>())
      match.members.push_back(&op);

  auto isInput = [&](Value v) {
    auto arg = dyn_cast<BlockArgument>(v);
    return arg && arg.getOwner() == body &&
           generic.isDpsInput(generic.getMatchingOpOperand(arg));
  };

  for (Operation *op : match.members) {
    DictionaryAttr opHint = triton::tts::getSpyreIntrinsic(op);
    DenseI64ArrayAttr operands =
        opHint ? triton::tts::getSpyreIntrinsicOperands(opHint) : nullptr;
    if (!operands)
      continue;

    // (operand index, call operand) per bound operand of `op`: by position,
    // or, on a commutative op whose listed positions do not all hold inputs,
    // by set -- see the function comment.
    SmallVector<std::pair<unsigned, int64_t>> binding;
    SmallVector<int64_t> listed;
    bool positional = true;
    for (auto [j, k] : llvm::enumerate(operands.asArrayRef())) {
      if (k < 0)
        continue;
      binding.emplace_back(j, k);
      listed.push_back(k);
      positional &= isInput(op->getOperand(j));
    }
    if (!positional && op->hasTrait<OpTrait::IsCommutative>()) {
      SmallVector<unsigned> inputs;
      for (auto [j, v] : llvm::enumerate(op->getOperands()))
        if (isInput(v))
          inputs.push_back(j);
      if (inputs.size() != listed.size()) {
        refuse(op->getLoc())
            << "call operand " << listed.front() << " is bound on commutative '"
            << op->getName() << "', which binds " << listed.size()
            << " call operand(s) and reads " << inputs.size()
            << " input(s) of its compute body";
        return failure();
      }
      if (!llvm::all_equal(listed)) {
        refuse(op->getLoc())
            << "call operand " << listed.front() << " is bound on commutative '"
            << op->getName()
            << "', whose operands are not in the order its binding names, and "
               "which binds more than one call operand, so which input is "
               "which call operand is not known";
        return failure();
      }
      binding.clear();
      for (unsigned j : inputs)
        binding.emplace_back(j, listed.front());
    }

    for (auto [j, k] : binding) {
      Value v = op->getOperand(j);
      if (!isInput(v)) {
        refuse(op->getLoc())
            << "call operand " << k << " is operand " << j << " of '"
            << op->getName()
            << "', which is not an input of its compute body -- a compiler "
               "bug, since nothing between the call site's inlining and its "
               "selection makes it one";
        return failure();
      }
      auto arg = cast<BlockArgument>(v);
      BlockArgument &bound = match.inputs[k];
      if (!bound)
        bound = arg;
      else if (bound != arg && !readSameInput(generic, bound, arg)) {
        refuse(op->getLoc())
            << "call operand " << k
            << " is read from two different inputs of its compute body, block "
               "arguments "
            << bound.getArgNumber() << " and " << arg.getArgNumber();
        return failure();
      }
    }
  }

  for (auto [k, bound] : llvm::enumerate(match.inputs))
    if (!bound) {
      refuse(generic.getLoc())
          << "call operand " << k
          << " is read by no op left of the fallback, so there is no input to "
             "pass the target for it";
      return failure();
    }
  if (yield->getNumOperands() != 1) {
    refuse(generic.getLoc())
        << "the call site's body yields " << yield->getNumOperands()
        << " values, and a request has one result";
    return failure();
  }
  return match;
}

/// Replaces a body that is one call site with the op its hint targets.
///
/// Matched on the generic and not on a member, because the call site is the
/// BODY: which op the fallback happens to end in says nothing. What the members
/// computed -- the fallback, casts included -- is discarded whole, and the body
/// becomes the target applied to its inputs, in call order, at the type the
/// body yields. The inputs' unused siblings, a captured constant among them,
/// are removed by the unused-operand cleanup the pass runs after selection.
///
/// The target is built by name, so nothing here knows what it takes: its own
/// verifier is asked, and a target that refuses this operand or type is erased
/// again, its diagnostic emitted, and the body left for rejectSurvivingHints.
///
/// The `tts` attributes of the op whose value was yielded move to the target,
/// the hint aside: that op produced the call site's result, so a `tts.pin` on
/// the result is on it.
void selectCallSite(RewriterBase &rewriter, linalg::GenericOp generic,
                    const CallSiteMatch &match) {
  StringRef target = triton::tts::getSpyreIntrinsicTarget(match.hint);
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << generic->getName()
                          << " at " << generic->getLoc()
                          << ": body is call site -> " << target << "\n");

  auto yield = cast<linalg::YieldOp>(generic.getBlock()->getTerminator());
  Value result = yield->getOperand(0);
  rewriter.setInsertionPoint(yield);
  OperationState state(result.getLoc(), target);
  state.addOperands(ArrayRef<BlockArgument>(match.inputs));
  state.addTypes(result.getType());
  Operation *selected = rewriter.create(state);
  if (failed(mlir::verify(selected))) {
    rewriter.eraseOp(selected);
    return;
  }
  if (Operation *producer = result.getDefiningOp()) {
    triton::tts::copyTTSAttrs(producer, selected);
    selected->removeDiscardableAttr(
        triton::tts::TTSDialect::kSpyreIntrinsicAttrName);
  }
  rewriter.modifyOpInPlace(
      yield, [&] { yield->setOperand(0, selected->getResult(0)); });
  // Reverse program order, so each op's users are gone before it is.
  for (Operation *op : llvm::reverse(match.members))
    rewriter.eraseOp(op);
}

/// Selection: every body that is one call site becomes its target.
/// A walk and not a fixpoint, since a body holds at most one call site and
/// replacing it makes no other. A body matchCallSite refuses is reported there,
/// with the reason, and left hinted, so rejectSurvivingHints reports the
/// request as unselected as well.
void selectCallSites(ModuleOp module) {
  SmallVector<std::pair<linalg::GenericOp, CallSiteMatch>> matches;
  module.walk([&](linalg::GenericOp generic) {
    DictionaryAttr hint = triton::tts::getBodyCallSite(*generic.getBlock());
    if (!hint)
      return;
    FailureOr<CallSiteMatch> match = matchCallSite(generic, hint);
    if (succeeded(match))
      matches.emplace_back(generic, std::move(*match));
  });
  IRRewriter rewriter(module.getContext());
  for (auto &[generic, match] : matches)
    selectCallSite(rewriter, generic, match);
}

/// Reports every call site selection left in place, once per `id`, naming the
/// intrinsic: the author asked for it by name, so running its fallback instead
/// would ignore that request without saying so. After selection, so it is the
/// one check that no `tts.spyre_intrinsic` attribute reaches LowerSpyreOps.
LogicalResult rejectSurvivingHints(ModuleOp module) {
  llvm::SmallDenseSet<int64_t> reported;
  module.walk([&](Operation *op) {
    DictionaryAttr hint = triton::tts::getSpyreIntrinsic(op);
    if (!hint ||
        !reported.insert(triton::tts::getSpyreIntrinsicId(hint)).second)
      return;
    mlir::emitError(op->getLoc())
        << "lower-spyre-intrinsic-calls: tl.spyre_intrinsic(\""
        << triton::tts::getSpyreIntrinsicTarget(hint)
        << "\") was not selected. The request is explicit, so its fallback is "
           "not used instead";
  });
  return success(reported.empty());
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerSpyreIntrinsicCallsPass
    : public mlir::triton::spyre::impl::LowerSpyreIntrinsicCallsBase<
          LowerSpyreIntrinsicCallsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    selectCallSites(module);
    if (failed(rejectSurvivingHints(module))) {
      signalPassFailure();
      return;
    }

    // Upstream's: a selected body reads only the call operands' inputs, and
    // this removes every other `ins` operand, its block argument and its
    // indexing map, a captured constant among them.
    RewritePatternSet patterns(&getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreIntrinsicCallsPass() {
  return std::make_unique<LowerSpyreIntrinsicCallsPass>();
}

} // namespace mlir::triton::spyre
