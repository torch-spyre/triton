//===- SpyreOpLoweringRules.cpp - The spyreop lowering rules -------------===//
//
// kSpyreOpLoweringRules, every rule LowerSpyreOps lowers by and FuseGenerics
// fuses for, and the guards and builds its entries name. See
// SpyreOpLoweringRules.h.
//
// A new rule is one build function and one entry in kSpyreOpLoweringRules,
// plus a guard when it has a condition no verifier sees, with no new pattern
// in either pass.
//
// NO NaN REACHES A COMPUTE BODY. Every rule assumes it, which lets a group:
//
//   - select an unordered predicate as its ordered one, e.g. `une` (Triton's
//     `!=`) as `spyreop.compare <notequal>`;
//   - drop a zero test in front of a select, e.g. `select(m != 0, p, q)` ->
//     `spyreop.select %m, %p, %q`.
//
// NO RULE CHECKS A TYPE. Which types a rule lowers is what the spyreop ops it
// builds accept, asked of the ops by probeSpyreOpRule (SpyreOpTypeProbe.h),
// which says why. A rule whose ops do not verify does not lower, so two rules
// on one anchor that build ops of different types, such as addi32 and addi64,
// need no guard to tell them apart. A guard is left for what no verifier sees:
// a constant operand, or a predicate's meaning.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpyreOp/Utils/SpyreOpLoweringRules.h"
#include "Dialect/SpyreOp/Utils/SpyreOpTypeProbe.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace mlir;

namespace mlir::triton::spyre {

namespace {

template <typename OpT> constexpr StringLiteral opName() {
  return OpT::getOperationName();
}

/// The spyreop predicate computing the same thing as `p`, or nothing:
///
///   - an ordering maps from both spellings, e.g. `oeq` and `ueq` -> `equal`
///     (they differ only on NaN, and no NaN reaches a compute body);
///   - `ord`/`uno` ask about NaN and `false`/`true` about nothing, so they have
///     no counterpart.
std::optional<spyreop::ComparePredicate>
spyrePredicateFor(arith::CmpFPredicate p) {
  switch (p) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    return spyreop::ComparePredicate::Equal;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return spyreop::ComparePredicate::NotEqual;
  case arith::CmpFPredicate::OGT:
  case arith::CmpFPredicate::UGT:
    return spyreop::ComparePredicate::GreaterThan;
  case arith::CmpFPredicate::OGE:
  case arith::CmpFPredicate::UGE:
    return spyreop::ComparePredicate::GreaterEqual;
  case arith::CmpFPredicate::OLT:
  case arith::CmpFPredicate::ULT:
    return spyreop::ComparePredicate::LesserThan;
  case arith::CmpFPredicate::OLE:
  case arith::CmpFPredicate::ULE:
    return spyreop::ComparePredicate::LesserEqual;
  default:
    return std::nullopt;
  }
}

//===----------------------------------------------------------------------===//
// Guards
//===----------------------------------------------------------------------===//

/// The reciprocal: the numerator is a scalar constant 1.0.
bool guardReciprocal(const GroupCandidate &candidate, std::string &why) {
  if (matchPattern(cast<arith::DivFOp>(candidate.anchor).getLhs(),
                   m_OneFloat()))
    return true;
  why = "the numerator is not 1.0";
  return false;
}

std::string noCounterpart(arith::CmpFOp cmp) {
  return (llvm::Twine("the predicate '") +
          arith::stringifyCmpFPredicate(cmp.getPredicate()) +
          "' has no spyreop.compare counterpart (`ord` and `uno` ask whether "
          "an operand is NaN, and `false`/`true` ask nothing, while that "
          "intrinsic only compares two values)")
      .str();
}

/// compare-cast and compare-select: the predicate has a spyreop.compare
/// counterpart.
bool guardCompare(const GroupCandidate &candidate, std::string &why) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  if (spyrePredicateFor(cmp.getPredicate()))
    return true;
  why = noCounterpart(cmp);
  return false;
}

/// A zero test that `spyreop.select` performs itself, found by
/// selectFromZeroTest. `conditionOperand` is the index among the compare's
/// operands of the one that is not zero, the scalar already in the body.
struct ZeroTestSelect {
  unsigned conditionOperand;
  bool exchangeValues;
};

/// Finds a comparison the select can drop, because `spyreop.select` already
/// tests its condition against zero:
///
///   - `m != 0` (`one`/`une`) -> condition `m`, values in order;
///   - `m == 0` (`oeq`/`ueq`) -> condition `m`, values exchanged;
///   - zero, a scalar constant, on either side, and -0.0, e.g. `0 != m`.
///
/// Returns nothing for the rest, which keep their compare:
///
///   - an ordering, e.g. `m > 0` is false for a negative `m` that "not zero"
///     calls true;
///   - an operand that is not a scalar constant zero, e.g. a block argument.
std::optional<ZeroTestSelect> selectFromZeroTest(arith::CmpFOp cmp) {
  bool exchangeValues;
  switch (cmp.getPredicate()) {
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    exchangeValues = false;
    break;
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ:
    exchangeValues = true;
    break;
  default:
    return std::nullopt;
  }
  if (matchPattern(cmp.getRhs(), m_AnyZeroFloat()))
    return ZeroTestSelect{0, exchangeValues};
  if (matchPattern(cmp.getLhs(), m_AnyZeroFloat()))
    return ZeroTestSelect{1, exchangeValues};
  return std::nullopt;
}

/// compare-select-zero-test: the compare is a zero test spyreop.select
/// performs itself (selectFromZeroTest).
bool guardZeroTest(const GroupCandidate &candidate, std::string &why) {
  if (selectFromZeroTest(cast<arith::CmpFOp>(candidate.anchor)))
    return true;
  why = "the compare is not a zero test spyreop.select performs itself";
  return false;
}

//===----------------------------------------------------------------------===//
// Legality
//===----------------------------------------------------------------------===//

/// compare-cast, compare-select-zero-test and compare-select: a compare left
/// as it is, read in its own body and not yielded, is illegal. Its result is an
/// `i1`, and no spyreop intrinsic produces or consumes that type, so what is
/// left is a group the rules had the chance to select and did not, and no rule
/// anyone could add would remove it.
///
/// A compare whose `i1` is yielded is legal here: that is the tensor form
/// leaving the body, which FuseGenerics' legalization reports.
bool legalCompare(const GroupCandidate &candidate, std::string &why) {
  if (llvm::any_of(candidate.anchor->getUsers(), [](Operation *user) {
        return isa<linalg::YieldOp>(user);
      }))
    return true;
  why = "an i1 value survives inside a compute body, which the Spyre device "
        "has no form for at all -- no spyreop intrinsic produces or consumes "
        "that type, so no selection rule can ever remove it";
  return false;
}

//===----------------------------------------------------------------------===//
// Builds
//===----------------------------------------------------------------------===//

/// sqrt, exp, rsqrt: `Target`(a0).
template <typename Target>
Value buildUnary(OpBuilder &b, const GroupCandidate &candidate,
                 ValueRange operands) {
  Operation *op = candidate.anchor;
  return Target::create(b, op->getLoc(), op->getResult(0).getType(),
                        operands[0]);
}

/// `Target`(a0, a1), for realdiv and the integer rules.
template <typename Target>
Value buildBinary(OpBuilder &b, const GroupCandidate &candidate,
                  ValueRange operands) {
  Operation *op = candidate.anchor;
  return Target::create(b, op->getLoc(), op->getResult(0).getType(),
                        operands[0], operands[1]);
}

/// reciprocal: `spyreop.reciprocal`(a1).
Value buildReciprocal(OpBuilder &b, const GroupCandidate &candidate,
                      ValueRange operands) {
  Operation *op = candidate.anchor;
  return spyreop::Reciprocal::create(b, op->getLoc(),
                                     op->getResult(0).getType(), operands[1]);
}

spyreop::ComparePredicateAttr comparePredicate(arith::CmpFOp cmp) {
  return spyreop::ComparePredicateAttr::get(
      cmp.getContext(), *spyrePredicateFor(cmp.getPredicate()));
}

/// compare-cast: the reader, an arith.uitofp, becomes `spyreop.compare` of
/// the compare's operands under its predicate's counterpart, at the reader's
/// type.
Value buildCompareCast(OpBuilder &b, const GroupCandidate &candidate,
                       ValueRange operands) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  Operation *reader = candidate.readers.front();
  return spyreop::Compare::create(b, reader->getLoc(),
                                  reader->getResult(0).getType(), operands[0],
                                  operands[1], comparePredicate(cmp));
}

/// compare-select-zero-test: the reader, an arith.select, becomes a
/// `spyreop.select` on the compare's nonzero operand, with the values
/// exchanged when the test is `== 0`. Operands: a0, a1, then the select's
/// condition, r1 and r2.
Value buildZeroTestSelect(OpBuilder &b, const GroupCandidate &candidate,
                          ValueRange operands) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  Operation *select = candidate.readers.front();
  ZeroTestSelect zeroTest = *selectFromZeroTest(cmp);
  // An opposite zero test reaches the same condition value, so the exchange
  // is what keeps it meaning the same thing.
  Value trueValue = operands[3], falseValue = operands[4];
  if (zeroTest.exchangeValues)
    std::swap(trueValue, falseValue);
  return spyreop::Select::create(b, select->getLoc(),
                                 select->getResult(0).getType(),
                                 operands[zeroTest.conditionOperand], trueValue,
                                 falseValue);
}

/// compare-select: the reader, an arith.select, becomes a `spyreop.select`
/// whose condition is a `spyreop.compare` of the compare's operands, placed
/// before it.
Value buildCompareSelect(OpBuilder &b, const GroupCandidate &candidate,
                         ValueRange operands) {
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  Operation *select = candidate.readers.front();
  Type selected = select->getResult(0).getType();
  Value condition =
      spyreop::Compare::create(b, select->getLoc(), selected, operands[0],
                               operands[1], comparePredicate(cmp));
  return spyreop::Select::create(b, select->getLoc(), selected, condition,
                                 operands[3], operands[4]);
}

//===----------------------------------------------------------------------===//
// The table
//===----------------------------------------------------------------------===//

constexpr StringLiteral kUIToFPReader[] = {opName<arith::UIToFPOp>()};
constexpr StringLiteral kSelectReader[] = {opName<arith::SelectOp>()};

/// Every rule LowerSpyreOps lowers by, the rules lowering one op first, then
/// the rules with a reader. Among the rules on one anchor a higher benefit is
/// tried first, and rules of equal benefit on one anchor never both match one
/// candidate: a guard or the type probe tells them apart, as the probe does
/// addi32 and addi64. The rules with a reader are also tried in this order by
/// matchLowerableGroup, so rules with the same anchor and the same readers are
/// listed by descending benefit.
///
/// A rule lists at most one reader, and no op is both a rule's reader and the
/// anchor of a rule without readers: benefit orders only patterns rooted on the
/// same op, so the order the driver visits ops would decide whether the group
/// or the lone op is lowered. verifySpyreOpLoweringRules checks both.
const SpyreOpLoweringRule kSpyreOpLoweringRules[] = {
    {"sqrt", opName<math::SqrtOp>(), {}, 1, nullptr,
     buildUnary<spyreop::Sqrt>},
    {"exp", opName<math::ExpOp>(), {}, 1, nullptr,
     buildUnary<spyreop::Exp>},
    {"rsqrt", opName<math::RsqrtOp>(), {}, 1, nullptr,
     buildUnary<spyreop::RSqrt>},
    {"reciprocal", opName<arith::DivFOp>(), {}, 2, guardReciprocal,
     buildReciprocal},
    {"realdiv", opName<arith::DivFOp>(), {}, 1, nullptr,
     buildBinary<spyreop::RealDiv>},
    // Every i32 or i64 add or multiply in a body, whatever its result is used
    // for.
    {"addi32", opName<arith::AddIOp>(), {}, 1, nullptr,
     buildBinary<spyreop::AddI32ToI32>},
    {"addi64", opName<arith::AddIOp>(), {}, 1, nullptr,
     buildBinary<spyreop::AddI64ToI64>},
    {"muli32", opName<arith::MulIOp>(), {}, 1, nullptr,
     buildBinary<spyreop::MulI32ToI32>},
    {"compare-cast", opName<arith::CmpFOp>(), kUIToFPReader, 1, guardCompare,
     buildCompareCast, legalCompare},
    {"compare-select-zero-test", opName<arith::CmpFOp>(), kSelectReader, 2,
     guardZeroTest, buildZeroTestSelect, legalCompare},
    {"compare-select", opName<arith::CmpFOp>(), kSelectReader, 1, guardCompare,
     buildCompareSelect, legalCompare},
};

} // namespace

bool verifySpyreOpLoweringRules(ArrayRef<SpyreOpLoweringRule> rules,
                                std::string &why) {
  for (const SpyreOpLoweringRule &rule : rules)
    if (rule.readers.size() > 1) {
      why = ("the rule '" + rule.name + "' lists " +
             std::to_string(rule.readers.size()) +
             " readers, and a rule takes at most one")
                .str();
      return false;
    }
  for (const SpyreOpLoweringRule &rule : rules)
    for (StringLiteral reader : rule.readers)
      for (const SpyreOpLoweringRule &other : rules)
        if (other.readers.empty() && other.anchor == reader) {
          why = ("'" + reader + "' is the reader of the rule '" + rule.name +
                 "' and the anchor of the rule '" + other.name +
                 "', which has no readers")
                    .str();
          return false;
        }
  return true;
}

SmallVector<Value> ShapeCheckedRule::operands() const {
  SmallVector<Value> operands(matchedCandidate->anchor->getOperands());
  if (!matchedRule->readers.empty())
    llvm::append_range(operands,
                       matchedCandidate->readers.front()->getOperands());
  return operands;
}

ArrayRef<SpyreOpLoweringRule> spyreOpLoweringRules() {
  // The table is constant, so it is checked once.
  static const bool verified = [] {
    std::string why;
    if (!verifySpyreOpLoweringRules(kSpyreOpLoweringRules, why))
      llvm::report_fatal_error("SpyreOpLoweringRules: " + llvm::Twine(why));
    return true;
  }();
  (void)verified;
  return kSpyreOpLoweringRules;
}

namespace {

/// Whether `candidate`'s ops are the ones `rule` names: its anchor is the
/// rule's anchor op and, when the rule lists readers, its readers match them
/// one for one and are directly in the group's generic. Sets `why` when they
/// are not.
bool opsMatch(const SpyreOpLoweringRule &rule, const GroupCandidate &candidate,
              std::string &why) {
  StringRef anchorName = candidate.anchor->getName().getStringRef();
  if (anchorName != rule.anchor) {
    why = ("the anchor is '" + anchorName + "', not '" + rule.anchor + "'")
              .str();
    return false;
  }
  if (rule.readers.empty())
    return true;
  if (candidate.readers.size() != rule.readers.size()) {
    why = candidate.readers.empty()
              ? "nothing reads the '" + anchorName.str() + "'"
              : "the '" + anchorName.str() + "' is read by " +
                    std::to_string(candidate.readers.size()) +
                    " ops, and is lowered only together with exactly " +
                    (rule.readers.size() == 1
                         ? std::string("one reader")
                         : std::to_string(rule.readers.size()) + " readers");
    return false;
  }
  for (auto [reader, name] : llvm::zip(candidate.readers, rule.readers))
    if (reader->getName().getStringRef() != name) {
      why = ("the reader is '" + reader->getName().getStringRef() +
             "', not '" + name + "'")
                .str();
      return false;
    }
  Operation *group = candidate.groupGeneric;
  for (Operation *reader : candidate.readers)
    if (reader->getParentOp() != group) {
      why = ("the reader is '" + reader->getName().getStringRef() +
             "', which is not directly in the group's generic body")
                .str();
      return false;
    }
  return true;
}

} // namespace

std::optional<ShapeCheckedRule>
ruleShapeMatches(const SpyreOpLoweringRule &rule,
                 const GroupCandidate &candidate, std::string &why) {
  if (!opsMatch(rule, candidate, why) ||
      (rule.guard && !rule.guard(candidate, why)))
    return std::nullopt;
  return ShapeCheckedRule(rule, candidate);
}

namespace {

/// Rule names, each list under the one string its rules share, in the order
/// each string first appears in the table.
using NamesByKey = SmallVector<std::pair<std::string, SmallVector<StringRef>>>;

void addName(NamesByKey &names, std::string key, StringRef name) {
  auto it = llvm::find_if(names, [&](const auto &entry) {
    return entry.first == key;
  });
  if (it == names.end())
    names.push_back({std::move(key), {name}});
  else
    it->second.push_back(name);
}

} // namespace

bool isLegalToLeave(const GroupCandidate &candidate, std::string &why) {
  StringRef anchorName = candidate.anchor->getName().getStringRef();
  for (const SpyreOpLoweringRule &rule : kSpyreOpLoweringRules)
    if (rule.legal && rule.anchor == anchorName &&
        !rule.legal(candidate, why))
      return false;
  return true;
}

FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why) {
  StringRef anchorName = candidate.anchor->getName().getStringRef();
  // The reasons of the rules whose ops the candidate has, when there are
  // any; otherwise of every rule with a reader on the anchor.
  NamesByKey onOps, onOthers;
  for (const SpyreOpLoweringRule &rule : spyreOpLoweringRules()) {
    if (rule.readers.empty() || rule.anchor != anchorName)
      continue;
    std::string reason;
    if (!opsMatch(rule, candidate, reason)) {
      addName(onOthers, std::move(reason), rule.name);
      continue;
    }
    std::optional<ShapeCheckedRule> shape =
        ruleShapeMatches(rule, candidate, reason);
    if (shape && probeSpyreOpRule(*shape, reason))
      return GroupMatch{rule.name, {candidate.anchor}};
    addName(onOps, std::move(reason), rule.name);
  }
  if (!why)
    return failure();
  const NamesByKey &reasons = onOps.empty() ? onOthers : onOps;
  if (reasons.empty()) {
    *why = ("no rule with a reader is anchored on '" + anchorName + "'").str();
    return failure();
  }
  SmallVector<std::string> parts;
  for (const auto &[reason, names] : reasons)
    parts.push_back(llvm::join(names, ", ") + ": " + reason);
  *why = llvm::join(parts, "; ");
  return failure();
}

std::string describeRulesWithReaders() {
  NamesByKey byOps;
  for (const SpyreOpLoweringRule &rule : spyreOpLoweringRules())
    if (!rule.readers.empty())
      addName(byOps,
              (rule.anchor + " read by " + llvm::join(rule.readers, " and "))
                  .str(),
              rule.name);
  SmallVector<std::string> parts;
  for (const auto &[ops, names] : byOps)
    parts.push_back(ops + " (" + llvm::join(names, ", ") + ")");
  return llvm::join(parts, ", or ");
}

} // namespace mlir::triton::spyre
