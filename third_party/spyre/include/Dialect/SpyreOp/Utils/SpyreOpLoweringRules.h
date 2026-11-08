//===- SpyreOpLoweringRules.h - What lowers to a spyreop intrinsic --------===//
//
// The lowering rules LowerSpyreOps and FuseGenerics both read, so the two
// passes cannot disagree on what lowers.
//
// Every rule is one SpyreOpLoweringRule in one table, kSpyreOpLoweringRules in
// SpyreOpLoweringRules.cpp, read through spyreOpLoweringRules(). A rule names
// an anchor op and the ops reading it that it lowers together with it: none
// for a rule lowering one op, one for a group such as a compare and its cast.
// Several ways to lower one op are several rules, ordered by their pattern
// benefit.
//
// LowerSpyreOps registers one pattern per rule, rooted on the rule's anchor.
// FuseGenerics, before it puts two ops in one body, asks matchLowerableGroup,
// which tries the rules that list a reader.
//
// A group has exactly one reader. An anchor read by more than one op matches
// no rule with a reader, so FuseGenerics does not group it and LowerSpyreOps
// does not lower it, and the `i1` left behind is reported by whichever pass
// holds it.
//
// Whether a rule lowers a candidate is two questions, asked one after the other
// by every caller: ruleShapeMatches (the ops, their count, the guard), then
// probeSpyreOpRule (SpyreOpTypeProbe.h), which asks the spyreop ops the rule
// builds whether they take the candidate's types. No rule states a type itself.
// The probe takes the ShapeCheckedRule that ruleShapeMatches makes, so it
// cannot be asked about a candidate whose shape the rule does not lower.
//
// An op no rule lowers is left for the backend, unless a rule says that
// leaving it is illegal (SpyreOpLoweringRule::legal); LowerSpyreOps asks the
// rules once it converges.
//
// The passes' diagnostics about a group are made from the table: the rules
// with a reader, by name, anchor and readers (describeRulesWithReaders), and
// each rule's reason for declining a candidate (matchLowerableGroup).
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H
#define TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::triton::spyre {

/// The ops a rule would lower together: an anchor and every op that reads its
/// result.
struct GroupCandidate {
  /// The op whose result the group consumes: in the producer generic when the
  /// group spans two generics, in the one generic when it does not.
  Operation *anchor;
  /// The linalg.generic the group is in, or will be in once fused: the
  /// anchor's own generic when the group is in one, the consumer when it spans
  /// two. A rule with a reader needs every reader directly in its body
  /// (ruleShapeMatches). The anchor's own generic is `anchor->getParentOp()`.
  linalg::GenericOp groupGeneric;
  /// Every op that reads `anchor`'s result, each once (distinctReaders). There
  /// may be several: a rule takes exactly the readers it lists.
  SmallVector<Operation *> readers;
};

/// The ops that read a value, each once, in use order. An op that reads the
/// value in more than one operand is one reader, though it is in the value's
/// users once per operand. `users` is a value's or an op's getUsers().
template <typename UserRange>
SmallVector<Operation *> distinctReaders(UserRange &&users) {
  llvm::SetVector<Operation *> readers(users.begin(), users.end());
  return readers.takeVector();
}

/// One way to lower an anchor op, together with the readers it lists, to
/// spyreop ops.
///
/// A rule lowers a candidate when ruleShapeMatches holds -- the anchor is the
/// `anchor` op, the candidate's readers match `readers` one for one and are
/// directly in the group's generic (any readers at all when `readers` is
/// empty), and `guard` holds -- and then probeSpyreOpRule holds: every op
/// `build` makes passes its own verifier.
struct SpyreOpLoweringRule {
  /// The rule's name, e.g. "reciprocal" or "compare-select-zero-test": the
  /// pattern's debug name, and what a trace reports.
  StringLiteral name;
  /// The anchor's op name, taken from the op class.
  StringLiteral anchor;
  /// The readers' op names, taken from the op classes; empty for a rule that
  /// lowers the anchor alone, and otherwise one op, since LowerByRule and the
  /// builds read and replace only the first reader
  /// (verifySpyreOpLoweringRules).
  ArrayRef<StringLiteral> readers;
  /// The order among the rules on one anchor: a higher benefit is tried first.
  PatternBenefit benefit;
  /// The rule's own conditions on a candidate whose ops already match, about
  /// what no verifier sees, such as a constant operand; null when there are
  /// none. Never about types. Sets `why` when it returns false.
  bool (*guard)(const GroupCandidate &, std::string &why);
  /// Builds the spyreop ops at the builder's insertion point from `operands`,
  /// the anchor's operands followed by its readers', and returns the value
  /// that replaces the rule's last op: its reader, or the anchor when it lists
  /// none. Runs twice when a rule fires: once into probeSpyreOpRule's scratch
  /// block, whose arguments stand for `operands`, then at the real position,
  /// and only after `guard` holds. So it reads the candidate only for types and
  /// attributes, its operands only from `operands`, and returns a value of the
  /// replaced op's type. probeSpyreOpRule checks this.
  Value (*build)(OpBuilder &, const GroupCandidate &, ValueRange operands);
  /// Whether it is legal to leave the anchor as it is, asked once selection has
  /// converged and no rule lowered it; null when it always is. An op the
  /// rules do not lower is normally legal: it reaches the backend, which
  /// knows what the device can take. It is not when its result has a type no
  /// spyreop op produces or consumes, because then no rule anyone could add
  /// would ever remove it, and the program is unlowerable, not merely
  /// unsupported. Returns false for that, with `why` the reason. Never about
  /// whether some rule could lower the anchor: that is `guard` and the probe.
  bool (*legal)(const GroupCandidate &, std::string &why);
};

/// Every rule, kSpyreOpLoweringRules. The table is checked by
/// verifySpyreOpLoweringRules the first time any caller reads it, and the
/// process aborts, naming the rule, when it fails.
ArrayRef<SpyreOpLoweringRule> spyreOpLoweringRules();

/// Whether `rules` obeys what the passes rely on, with `why` set to the first
/// violation when it does not:
///
///   - a rule lists at most one reader;
///   - no op is both a rule's reader and the anchor of a rule without
///     readers, because benefit orders only patterns rooted on the same op.
bool verifySpyreOpLoweringRules(ArrayRef<SpyreOpLoweringRule> rules,
                                std::string &why);

/// A rule and a candidate that has the shape the rule lowers, its types not yet
/// asked. Only ruleShapeMatches makes one, so whatever takes a ShapeCheckedRule
/// may rely on the shape: the rule's build casts the anchor and readers to the
/// ops the rule names.
///
/// It is one rule's answer for one candidate, not the candidate's shape: a
/// candidate can pass the shape check of several rules, e.g. a compare read by
/// a select passes for the compare-select rules. The rule that lowers it is
/// found among them by the probe (GroupMatch).
///
/// It refers to the rule and the candidate, which must outlive it.
class ShapeCheckedRule {
public:
  const SpyreOpLoweringRule &rule() const { return *matchedRule; }
  const GroupCandidate &candidate() const { return *matchedCandidate; }

  /// The values `rule().build` takes: the anchor's operands, then its reader's
  /// when the rule lists one.
  SmallVector<Value> operands() const;

private:
  ShapeCheckedRule(const SpyreOpLoweringRule &rule,
                   const GroupCandidate &candidate)
      : matchedRule(&rule), matchedCandidate(&candidate) {}
  friend std::optional<ShapeCheckedRule>
  ruleShapeMatches(const SpyreOpLoweringRule &, const GroupCandidate &,
                   std::string &);

  const SpyreOpLoweringRule *matchedRule;
  const GroupCandidate *matchedCandidate;
};

/// The ShapeCheckedRule of `rule` and `candidate`, when the candidate has the
/// shape the rule lowers: its anchor is the rule's anchor op, its readers match
/// the rule's one for one and are directly in the group's generic, and the
/// guard holds. Says nothing of types: that is probeSpyreOpRule's. Sets `why`
/// when it does not match.
std::optional<ShapeCheckedRule>
ruleShapeMatches(const SpyreOpLoweringRule &rule,
                 const GroupCandidate &candidate, std::string &why);

/// What matchLowerableGroup found for one candidate: the rule that lowers it,
/// and which ops of the anchor's body the group consumes. It is the final
/// answer across the rules, where a ShapeCheckedRule is one rule's step toward
/// it.
struct GroupMatch {
  /// The rule's name, for the trace.
  StringRef shape;
  /// The anchor body's ops the group consumes.
  SmallVector<Operation *> producerOps;
};

/// Whether it is legal to leave `candidate`'s anchor as it is: every rule
/// anchored on its op that has a `legal` check agrees. Asked once selection has
/// converged. Sets `why` to the first such rule's reason when it is not.
bool isLegalToLeave(const GroupCandidate &candidate, std::string &why);

/// The rule with a reader that lowers `candidate`: the first in
/// kSpyreOpLoweringRules for which ruleShapeMatches and probeSpyreOpRule both
/// hold, tried in turn on this one candidate. Failure when none does, with
/// `why`, when non-null, set to the reasons of the rules with a reader on the
/// candidate's anchor, each reason once, after the names of the rules that gave
/// it, e.g. "compare-cast: the reader is 'arith.sitofp', not 'arith.uitofp'".
/// When some of those rules name the candidate's anchor and readers, only their
/// reasons are given.
FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why = nullptr);

/// Every rule with a reader, by its ops and then its name, rules on the same
/// ops together, in table order, e.g. "arith.cmpf read by arith.uitofp
/// (compare-cast), or arith.cmpf read by arith.select
/// (compare-select-zero-test, compare-select)".
std::string describeRulesWithReaders();

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H
