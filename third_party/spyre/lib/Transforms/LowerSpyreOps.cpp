//===- LowerSpyreOps.cpp - Select spyreop intrinsics from compute bodies --===//
//
// INSTRUCTION SELECTION for the Spyre device: arith and math ops become the
// spyreop intrinsic that does the same thing. A RULE HOST -- every rule is one
// entry in kSpyreOpLoweringRules in SpyreOpLoweringRules.cpp, and this
// pass registers one LowerByRule pattern per entry, so adding a case is one
// entry and one lit case, with no new pattern.
//
// KTIR -> KTIR, which is why this sits in Transforms/ rather than in
// Conversion/TritonToKTIR/: that directory's criterion is a `tt` source
// dialect, and nothing here reads a tt op. The input is KTIR whose computes are
// linalg.generic, and the output is the same KTIR with intrinsics in the
// bodies.
//
// ONE PASS FOR ALL SELECTION, and the two kinds of rule are not two tiers:
//
//   one op to one   math.sqrt -> spyreop.sqrt, and so on. `arith.divf` has
//                   two rules: `1.0 / x` -> spyreop.reciprocal, tried first,
//                   and anything else -> spyreop.realdiv.
//   a GROUP to one  a compare and a cast -> spyreop.compare. The group is what
//                   is selectable: no member of it could have been selected
//                   alone.
//
// They share one greedy pattern set. The rules rooted on one op are tried in
// the order of their benefits, declared in the table, and no op is both a
// rule's reader and the anchor of a rule without readers.
// Splitting these across two passes is what an earlier shape did, and it
// bought a standing question -- which pass claims this op -- for nothing.
//
// EVERYTHING UNMATCHED FLOWS THROUGH, WITH ONE EXCEPTION. There is no
// conversion target: an op with no device form reaches the backend, which is
// the component that actually knows what it can take, and it refuses there. So
// this pass has no notion of an illegal input -- it selects what it can and
// leaves the rest exactly as it found it. A type or a predicate this file does
// not handle is a silent pass-through by design; see WHAT IS NOT SELECTED below
// for the list and what each one costs. The exception is an `i1` left inside a
// compute body, which rejectSurvivingBooleans reports.
//
// WHAT THIS PASS RELIES ON ITS PREDECESSOR FOR. A rule matches ops in ONE body,
// and ConvertElementwiseToLinalg gives every tensor-level op a body of its own
// -- so a group spanning two tensor ops arrives spread over two generics unless
// something fused them. FuseGenerics' GroupComputes is that something, and it
// fuses only when matchLowerableGroup -- in SpyreOpLoweringRules.h,
// which asks the rules with a reader in the same table this pass registers,
// on the same candidate: an anchor and every reader of it -- finds a rule
// covering the anchor and its one reader. An
// anchor read by more than one op is not grouped, and FuseGenerics reports the
// `i1` left between the generics.
//
// CONSTANTS ARRIVE AS SCALARS IN THE BODY. The reciprocal's 1.0 and a zero
// test's 0.0 are matched as scalar constants the body reads, never through a
// splat `ins` operand: CanonicalizeGenericOperands' SplatOperandToScalar, which
// runs first, is what puts a splat constant `ins` into the body as a scalar. A
// splat still passed as an `ins` reaches a rule as a block argument and matches
// nothing.
//
// THE GENERIC BODY IS THE SCOPE of every rule. Every compute reaches this pass
// as a linalg.generic, so "inside a body" is not a guess about whether an op is
// compute -- it is where the compute is. LowerByRule checks it once for every
// rule: the root's immediate parent is a linalg.generic, and a rule's reader is
// in the same body. An `arith.addi` on an element type in a body is compute,
// and the addi of a loop index or a tile address is not in a body at all; a
// scalar `math.sqrt` outside a body is left as it is.
//
// NO NaN REACHES A COMPUTE BODY. Every rule assumes it, which lets the pass:
//
//   - select an unordered predicate as its ordered one, e.g. `une` (Triton's
//     `!=`) as `spyreop.compare <notequal>`;
//   - drop a zero test in front of a select, e.g. `select(m != 0, p, q)` ->
//     `spyreop.select %m, %p, %q`.
//
// A NaN input gives an unspecified result on its lane.
//
// The mechanism is upstream's -- the greedy driver, and the unused-operand
// erasure -- and the POLICY is ours: which ops and which groups the device has
// a form for, and on what evidence.
//
// WHAT IS NOT SELECTED, and what each costs. None of these is a diagnostic, per
// EVERYTHING UNMATCHED FLOWS THROUGH above; Passes.td carries the same list
// with the reasoning and this is the short form.
//
//   A 1:1 rule does not fire and the op goes to the backend as arith:
//     - any op outside a generic body
//     - a float width with no intrinsic (f64, bf16)
//     - an integer width with no intrinsic
//     - anything still tensor-typed, which only ConvertElementwiseToLinalg not
//       having run can produce
//
//   A group rule declines and its members stay as they were:
//     - a divide whose numerator is not a constant one, or whose constant is
//       the denominator: the realdiv rule takes it, which is right
//     - a numerator that is a block argument, even one fed by a splat 1.0
//       constant `ins`: the rule reads the body, not the operands
//     - `ord`/`uno` and the constant predicates `false`/`true`: no
//       counterpart. They ask about NaN-ness or about nothing, not about an
//       ordering
//     - `arith.sitofp` where `uitofp` was wanted: an `i1` read as signed is 0
//       or -1, so the cast gives -1.0 where the predicate holds
//     - a compare at one width cast to another: spyreop.compare has
//       SameOperandsAndResultType and cannot do both
//
//   The compare declines are the ones to know: each leaves an `i1` inside a
//   body, so each is the exception above rather than a pass-through --
//   rejectSurvivingBooleans reports it here, naming the predicate.
//
// `--debug-only=lower-spyre-ops` traces the decisions of the rules with a
// reader rather than the control flow: one line per candidate looked at and
// what came of it, with each rule's reason when none matched. The rules
// without readers say nothing, save the reciprocal when it fires -- the output
// IR shows whether one did.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpyreOp/Utils/SpyreOpLoweringRules.h"
#include "Transforms/Passes.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#define DEBUG_TYPE "lower-spyre-ops"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERSPYREOPS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

using mlir::triton::spyre::GroupCandidate;
using mlir::triton::spyre::SpyreOpLoweringRule;
using mlir::triton::spyre::spyrePredicateFor;

namespace {

//===----------------------------------------------------------------------===//
// One pattern per rule
//===----------------------------------------------------------------------===//

/// Lowers by one rule of kSpyreOpLoweringRules, rooted on the rule's anchor op
/// at the rule's benefit, so the driver tries the rules on one op in benefit
/// order.
///
/// Every rule applies only in a compute body: the root's immediate parent is a
/// linalg.generic, and a rule's reader is in that same body. A root elsewhere
/// is left as it is, whatever its type.
///
/// The candidate is the root, that generic, and every op reading the root,
/// each once though it may read the root more than once. When the rule
/// matches, its build runs at the rule's last op -- its reader, or the root
/// when it lists none -- on the root's operands followed by the reader's, the
/// value it returns replaces that op, and a root left without users is erased.
///
/// A candidate in a body that no rule with a reader lowers is traced once, by
/// the last such rule on its anchor in the table, with every rule's reason
/// (matchLowerableGroup). The driver tries every pattern on an op before giving
/// up on it, so that rule runs whenever all of them decline. A rule without
/// readers says nothing -- the output IR shows whether it fired.
///
/// A declined compare read in the body is then reported by
/// rejectSurvivingBooleans.
struct LowerByRule : public RewritePattern {
  LowerByRule(const SpyreOpLoweringRule &rule, MLIRContext *ctx)
      : RewritePattern(rule.anchor, rule.benefit, ctx), rule(rule),
        tracesDeclines(isLastRuleWithReaders(rule)) {
    setDebugName(rule.name);
  }

  LogicalResult matchAndRewrite(Operation *root,
                                PatternRewriter &rewriter) const override {
    // Placement: the root's immediate parent is a linalg.generic. An op at
    // function level, or nested deeper inside a body, is left as it is.
    auto body = dyn_cast<linalg::GenericOp>(root->getParentOp());
    if (!body)
      return failure();
    // The ops reading the root, each once even if it reads the root twice.
    llvm::SetVector<Operation *> readers(root->user_begin(), root->user_end());
    // A rule with readers lowers the root and its reader together, so every
    // reader is in the same body: one in another generic, or a yield's user
    // outside the body, is not part of this group.
    if (!rule.readers.empty() &&
        llvm::any_of(readers, [&](Operation *reader) {
          return reader->getParentOp() != body.getOperation();
        }))
      return failure();
    GroupCandidate candidate{root, body, readers.takeVector()};
    // The rule's own checks: the anchor and reader op names, the reader count,
    // then its guard.
    std::string why;
    if (!triton::spyre::ruleMatches(rule, candidate, why)) {
      LLVM_DEBUG({
        if (tracesDeclines &&
            failed(triton::spyre::matchLowerableGroup(candidate, &why)))
          traceDecline(root, why);
      });
      return failure();
    }

    // The op whose result the group produces: the reader for a group, the
    // root for a one-op rule. The spyreop ops are built just before it, on the
    // root's operands followed by the reader's, and replace it.
    Operation *last = rule.readers.empty() ? root : candidate.readers.front();
    SmallVector<Value> operands(root->getOperands());
    if (!rule.readers.empty())
      llvm::append_range(operands, last->getOperands());
    rewriter.setInsertionPoint(last);
    Value replacement = rule.build(rewriter, candidate, operands);
    rewriter.replaceOp(last, replacement);
    // A group's root, its only reader now replaced, is dead.
    if (root != last && root->use_empty())
      rewriter.eraseOp(root);
    return success();
  }

private:
  /// Whether `rule` lists readers and no later rule in the table with readers
  /// has the same anchor.
  static bool isLastRuleWithReaders(const SpyreOpLoweringRule &rule) {
    if (rule.readers.empty())
      return false;
    ArrayRef<SpyreOpLoweringRule> rules = triton::spyre::spyreOpLoweringRules();
    auto later = rules.drop_front(&rule - rules.data() + 1);
    return llvm::none_of(later, [&](const SpyreOpLoweringRule &other) {
      return !other.readers.empty() && other.anchor == rule.anchor;
    });
  }

  static void traceDecline(Operation *root, const llvm::Twine &why) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                            << root->getLoc() << ": no group match (" << why
                            << ")\n");
  }

  const SpyreOpLoweringRule &rule;
  bool tracesDeclines;
};

//===----------------------------------------------------------------------===//
// After selection: an i1 left in a compute body
//===----------------------------------------------------------------------===//

/// Reports an `i1` that selection left in a compute body, because no spyreop
/// op takes that type, so the backend would refuse it without naming the
/// predicate. Runs after the fixpoint, since until then such an `i1` is what
/// the group rules match.
///
/// Reported, when made and read in the same body without being yielded:
///
///   - `cmpf ogt` read by an i32 `arith.select`;
///   - `cmpf ord` read by `arith.uitofp` (no counterpart);
///   - `cmpf oeq` read by `arith.sitofp` (the compare rule wants `uitofp`);
///   - `cmpf ogt` read by two ops (a group takes one reader).
///
/// Not reported, because each is the tensor form crossing a body boundary,
/// which FuseGenerics reports:
///
///   - an `i1` block argument, e.g. `^bb0(%cond: i1, ...)`;
///   - an `i1` that is yielded, e.g. `linalg.yield %c : i1`.
LogicalResult rejectSurvivingBooleans(ModuleOp mod) {
  LogicalResult result = success();
  mod.walk([&](linalg::GenericOp generic) {
    Block *body = generic.getBlock();
    if (!body)
      return;
    // Op results only: an i1 block argument is the tensor form crossing into
    // the body, which is not this pass's finding.
    SmallVector<Value> values;
    for (Operation &op : *body)
      values.append(op.getResults().begin(), op.getResults().end());

    for (Value v : values) {
      if (!getElementTypeOrSelf(v.getType()).isInteger(1))
        continue;
      // Nor one that is yielded: that is the tensor form leaving the body.
      if (llvm::any_of(v.getUsers(), [](Operation *user) {
            return isa<linalg::YieldOp>(user);
          }))
        continue;
      result = failure();

      InFlightDiagnostic diag = mlir::emitError(v.getLoc());
      diag << "lower-spyre-ops: an i1 value survives inside a compute body, "
              "which the Spyre device has no form for at all -- no spyreop "
              "intrinsic produces or consumes that type, so no selection rule "
              "can ever remove it";

      // Name the predicate: it is what an author can change.
      if (auto cmp = v.getDefiningOp<arith::CmpFOp>()) {
        StringRef pred = arith::stringifyCmpFPredicate(cmp.getPredicate());
        llvm::SetVector<Operation *> readers(v.user_begin(), v.user_end());
        if (spyrePredicateFor(cmp.getPredicate()) && readers.size() > 1)
          diag.attachNote(cmp.getLoc())
              << "the predicate '" << pred
              << "' does have a spyreop.compare counterpart, but the compare "
                 "is read by "
              << readers.size()
              << " ops, and a compare is lowered only together with exactly "
                 "one reader";
        else if (spyrePredicateFor(cmp.getPredicate()))
          diag.attachNote(cmp.getLoc())
              << "the predicate '" << pred
              << "' does have a spyreop.compare counterpart, so this compare "
                 "was selectable and something about its READER was not: the "
                 "reader must be an arith.uitofp to, or an arith.select of, "
                 "the width compared, and that width must be f16 or f32";
        else
          diag.attachNote(cmp.getLoc())
              << "the predicate '" << pred
              << "' has no spyreop.compare counterpart: `ord` and `uno` "
                 "ask whether an operand is NaN, and `false`/`true` ask "
                 "nothing, while that intrinsic only compares two values";
      }

      for (Operation *user : v.getUsers())
        diag.attachNote(user->getLoc())
            << "read here, by '" << user->getName() << "'";
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerSpyreOpsPass
    : public mlir::triton::spyre::impl::LowerSpyreOpsBase<LowerSpyreOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // One pattern per rule, rules with and without readers in one set. See
    // ONE PASS FOR ALL SELECTION in the header.
    for (const SpyreOpLoweringRule &rule :
         triton::spyre::spyreOpLoweringRules())
      patterns.add<LowerByRule>(rule, ctx);

    // Upstream's, and load-bearing rather than tidying: a group rule that stops
    // reading a block argument leaves an `ins` operand, that argument and its
    // indexing map behind, and this removes all three. In the same fixpoint, so
    // a rule never has to see the intermediate state.
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

    // The greedy driver, not a dialect conversion. Nothing here is illegal: an
    // op with no device form flows through to the backend, which is the
    // component that knows what it can take. A failure from the driver would
    // mean the rewrite diverged, not that an op went unhandled.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // The one thing this pass does report. Not a selection failure -- see
    // rejectSurvivingBooleans on why an unrepresentable TYPE is a different
    // kind of thing from an op the device happens not to do.
    if (failed(rejectSurvivingBooleans(module)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass() {
  return std::make_unique<LowerSpyreOpsPass>();
}

} // namespace mlir::triton::spyre
