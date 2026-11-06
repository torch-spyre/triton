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
// The remaining exports are the questions the rules share with LowerSpyreOps'
// diagnostics: which scalar types the float intrinsics take, which spyreop
// predicate a cmpf predicate maps to, and what a body value names from outside
// the body.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H
#define TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H

#include "ktir/Dialect/SpyreOp/SpyreOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
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
  /// The generic whose body holds `anchor` directly, or null when `anchor`'s
  /// parent is not a generic. A compare operand that is one of its block
  /// arguments is looked up through it.
  linalg::GenericOp anchorBody;
  /// The ops that read `anchor`'s result, wherever they are: in `anchorBody`
  /// when the group is in one body, in the consumer's body when it spans two
  /// generics.
  SmallVector<Operation *> readers;
};

/// One way to lower an anchor op, together with the readers it lists, to
/// spyreop ops.
///
/// A rule matches a candidate (ruleMatches) when the anchor is the `anchor`
/// op, the candidate's readers match `readers` one for one -- any readers at
/// all when `readers` is empty -- and `guard` holds.
struct SpyreOpLoweringRule {
  /// The rule's name, e.g. "reciprocal" or "compare-select-zero-test": the
  /// pattern's debug name, and what a trace reports.
  StringLiteral name;
  /// The anchor's op name, taken from the op class.
  StringLiteral anchor;
  /// The readers' op names, taken from the op classes; empty for a rule that
  /// lowers the anchor alone.
  ArrayRef<StringLiteral> readers;
  /// The order among the rules on one anchor: a higher benefit is tried first.
  PatternBenefit benefit;
  /// The rule's own conditions on a candidate whose ops already match. Sets
  /// `why` when it returns false.
  bool (*guard)(const GroupCandidate &, std::string &why);
  /// Builds the spyreop ops at the builder's insertion point from `operands`,
  /// the anchor's operands followed by its readers', and returns the value
  /// that replaces the rule's last op: its reader, or the anchor when it lists
  /// none.
  Value (*build)(OpBuilder &, const GroupCandidate &, ValueRange operands);
};

/// Every rule, kSpyreOpLoweringRules.
ArrayRef<SpyreOpLoweringRule> spyreOpLoweringRules();

/// Whether `rule` lowers `candidate`: its anchor is the rule's anchor op, its
/// readers match the rule's one for one, and the guard holds. Sets `why` when
/// it does not.
bool ruleMatches(const SpyreOpLoweringRule &rule,
                 const GroupCandidate &candidate, std::string &why);

/// What matchLowerableGroup found: the rule, and which ops of the anchor's
/// body the group consumes.
struct GroupMatch {
  /// The rule's name, for the trace.
  StringRef shape;
  /// The anchor body's ops the group consumes.
  SmallVector<Operation *> producerOps;
};

/// The rule with a reader that lowers `candidate`: the first in
/// kSpyreOpLoweringRules that matches it. Failure when none does, with `why`,
/// when non-null, set to each such rule's reason.
FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why = nullptr);

/// Whether spyreop's scalar float intrinsics accept this operand type.
bool isSpyreOpScalarType(Type type);

/// The spyreop predicate computing the same thing as `p`, or nothing:
///
///   - an ordering maps from both spellings, e.g. `oeq` and `ueq` -> `equal`
///     (they differ only on NaN, and no NaN reaches a compute body);
///   - `ord`/`uno` ask about NaN and `false`/`true` about nothing, so they have
///     no counterpart.
std::optional<spyreop::ComparePredicate>
spyrePredicateFor(arith::CmpFPredicate p);

/// What `v` names from OUTSIDE `generic`'s body: the matching `ins` operand
/// when `v` is one of the body's input block arguments, and `v` itself
/// otherwise.
///
/// This is what lets a rule ask about an operand's DEFINITION without caring
/// how the value reached the body. The same `1.0` is a scalar `arith.constant`
/// above the generic once CanonicalizeGenericOperands has folded the splat in,
/// and a splat `ins` operand if that pass has not run; resolved, both answer
/// the same question.
///
/// SOUND ONLY BECAUSE OF WHAT CALLERS ASK. A body value corresponds to the same
/// tensor element only for a value uniform across the operand, so a caller may
/// ask a resolved value whether it is a SPLAT or a scalar constant -- which is
/// exactly what `m_OneFloat` tests -- and may not ask it anything about a
/// particular element. A non-splat constant tensor resolves here too and
/// correctly matches nothing.
///
/// The `outs` block arguments are deliberately not resolved: an `outs` of a
/// `tensor.empty` has no defined value to name, so forwarding it would invite a
/// rule to read one.
Value resolveThroughBody(linalg::GenericOp generic, Value v);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H
