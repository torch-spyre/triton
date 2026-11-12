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
// diagnostics: which scalar types the float intrinsics take, and which spyreop
// predicate a cmpf predicate maps to.
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
  /// The linalg.generic the group is in, or will be in once fused: the
  /// anchor's own generic when the group is in one, the consumer when it spans
  /// two. A rule with a reader needs every reader directly in its body
  /// (ruleMatches). The anchor's own generic is `anchor->getParentOp()`.
  linalg::GenericOp groupGeneric;
  /// Every op that reads `anchor`'s result, each once. There may be several: a
  /// rule takes exactly the readers it lists.
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
  /// lowers the anchor alone, and otherwise one op, since LowerByRule and the
  /// builds read and replace only the first reader
  /// (verifySpyreOpLoweringRules).
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

/// Whether `rule` lowers `candidate`: its anchor is the rule's anchor op, its
/// readers match the rule's one for one and are directly in the group's
/// generic, and the guard holds. Sets `why` when it does not.
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

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H
