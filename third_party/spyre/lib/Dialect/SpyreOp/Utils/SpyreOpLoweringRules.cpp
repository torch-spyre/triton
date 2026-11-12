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

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace mlir::triton::spyre {

//===----------------------------------------------------------------------===//
// Shared questions
//===----------------------------------------------------------------------===//

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

namespace {

template <typename OpT> constexpr StringLiteral opName() {
  return OpT::getOperationName();
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
  return (llvm::Twine("spyreop.compare has no counterpart for predicate '") +
          arith::stringifyCmpFPredicate(cmp.getPredicate()) + "'")
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
     buildCompareCast},
    {"compare-select-zero-test", opName<arith::CmpFOp>(), kSelectReader, 2,
     guardZeroTest, buildZeroTestSelect},
    {"compare-select", opName<arith::CmpFOp>(), kSelectReader, 1, guardCompare,
     buildCompareSelect},
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

SmallVector<Value> ShapeMatch::operands() const {
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

bool shapeHolds(const SpyreOpLoweringRule &rule,
                const GroupCandidate &candidate, std::string &why) {
  // A rule with readers asks for them one for one; a rule without asks
  // nothing of them.
  if (!rule.readers.empty() &&
      candidate.readers.size() != rule.readers.size()) {
    why = candidate.readers.empty()
              ? std::string("nothing reads the anchor")
              : "the anchor has " + std::to_string(candidate.readers.size()) +
                    " readers, and a group takes exactly " +
                    (rule.readers.size() == 1
                         ? std::string("one")
                         : std::to_string(rule.readers.size()));
    return false;
  }
  if (candidate.anchor->getName().getStringRef() != rule.anchor) {
    why = ("the anchor is not an " + rule.anchor).str();
    return false;
  }
  for (auto [reader, name] : llvm::zip(candidate.readers, rule.readers))
    if (reader->getName().getStringRef() != name) {
      why = "the reader, '" + reader->getName().getStringRef().str() +
            "', is not an " + name.str();
      return false;
    }
  Operation *group = candidate.groupGeneric;
  if (!rule.readers.empty())
    for (Operation *reader : candidate.readers)
      if (reader->getParentOp() != group) {
        why = "the reader, '" + reader->getName().getStringRef().str() +
              "', is not directly in the group's generic body";
        return false;
      }
  return !rule.guard || rule.guard(candidate, why);
}

} // namespace

std::optional<ShapeMatch> ruleShapeMatches(const SpyreOpLoweringRule &rule,
                                           const GroupCandidate &candidate,
                                           std::string &why) {
  if (!shapeHolds(rule, candidate, why))
    return std::nullopt;
  return ShapeMatch(rule, candidate);
}

FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why) {
  // Each distinct reason once: the rules on one reader count, or on one
  // anchor op, decline for the same one.
  SmallVector<std::string> reasons;
  for (const SpyreOpLoweringRule &rule : spyreOpLoweringRules()) {
    if (rule.readers.empty())
      continue;
    std::string reason;
    std::optional<ShapeMatch> shape = ruleShapeMatches(rule, candidate, reason);
    if (shape && probeSpyreOpRule(*shape, reason))
      return GroupMatch{rule.name, {candidate.anchor}};
    if (!llvm::is_contained(reasons, reason))
      reasons.push_back(reason);
  }
  if (why)
    *why = llvm::join(reasons, "; ");
  return failure();
}

} // namespace mlir::triton::spyre
