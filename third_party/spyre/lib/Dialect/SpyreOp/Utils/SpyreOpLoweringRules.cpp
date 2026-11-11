//===- SpyreOpLoweringRules.cpp - The spyreop lowering rules -------------===//
//
// kSpyreOpLoweringRules, every rule LowerSpyreOps lowers by and FuseGenerics
// fuses for, and the guards and builds its entries name. See
// SpyreOpLoweringRules.h.
//
// A new rule is one guard, one build function and one entry in
// kSpyreOpLoweringRules, with no new pattern in either pass.
//
// NO NaN REACHES A COMPUTE BODY. Every rule assumes it, which lets a group:
//
//   - select an unordered predicate as its ordered one, e.g. `une` (Triton's
//     `!=`) as `spyreop.compare <notequal>`;
//   - drop a zero test in front of a select, e.g. `select(m != 0, p, q)` ->
//     `spyreop.select %m, %p, %q`.
//
// The builds trace under `--debug-only=lower-spyre-ops`, the pass that runs
// them.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpyreOp/Utils/SpyreOpLoweringRules.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "lower-spyre-ops"

using namespace mlir;

namespace mlir::triton::spyre {

//===----------------------------------------------------------------------===//
// Shared questions
//===----------------------------------------------------------------------===//

bool isSpyreOpScalarType(Type type) {
  return isa<Float16Type, Float32Type>(type);
}

Value resolveThroughBody(linalg::GenericOp generic, Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || arg.getOwner() != generic.getBlock())
    return v;
  OpOperand *operand = generic.getMatchingOpOperand(arg);
  return generic.isDpsInput(operand) ? operand->get() : v;
}

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

void traceMatch(Operation *root, const llvm::Twine &what) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                          << root->getLoc() << ": " << what << "\n");
}

//===----------------------------------------------------------------------===//
// Guards
//===----------------------------------------------------------------------===//

/// The bit width of `type` if it is a scalar integer, or 0 otherwise -- e.g.
/// for a not-yet-scalarized tensor of integers.
unsigned getScalarIntBitWidth(Type type) {
  auto intTy = dyn_cast<IntegerType>(type);
  return intTy ? intTy.getWidth() : 0;
}

/// Whether this op is (transitively) inside a linalg.generic body.
///
/// A PROXY, used only by the integer rules, which have no better test: plain
/// scalar integer add and mul are also loop indices, offsets and tile
/// addressing, and there is no exact test separating those from scalarized
/// integer compute. The proxy over-claims -- `tl.arange(0, N) * stride` is an
/// `arith.muli` that will be inside a body once scalarized -- and it holds
/// today only because the tensor-of-pointers `tt.load` path is unimplemented.
/// The rules with a reader use the enclosing generic as a SCOPE instead,
/// requiring the generic to be the anchor's immediate parent.
bool isInsideLinalgGeneric(Operation *op) {
  return op->getParentOfType<linalg::GenericOp>() != nullptr;
}

/// The unary float rules and realdiv: the result type has a spyreop float
/// intrinsic. No placement check: these ops only ever appear in real
/// floating-point compute, never in address or index arithmetic, so every
/// scalar occurrence is one to lower, inside a generic body or not.
bool guardFloatResult(const GroupCandidate &candidate, std::string &why) {
  if (isSpyreOpScalarType(candidate.anchor->getResult(0).getType()))
    return true;
  why = "no spyreop intrinsic for this result type";
  return false;
}

/// The reciprocal: guardFloatResult, and the numerator is a 1.0, read through
/// `anchorBody` when there is one, so a one passed in as a splat `ins` operand
/// matches as well as a scalar constant.
bool guardReciprocal(const GroupCandidate &candidate, std::string &why) {
  if (!guardFloatResult(candidate, why))
    return false;
  Value numerator = cast<arith::DivFOp>(candidate.anchor).getLhs();
  if (candidate.anchorBody)
    numerator = resolveThroughBody(candidate.anchorBody, numerator);
  if (matchPattern(numerator, m_OneFloat()))
    return true;
  why = "the numerator is not 1.0";
  return false;
}

/// The integer rules: the anchor is inside a generic body (the proxy at
/// isInsideLinalgGeneric) and its result is an integer of `Width` bits.
template <unsigned Width>
bool guardIntegerInBody(const GroupCandidate &candidate, std::string &why) {
  if (!isInsideLinalgGeneric(candidate.anchor)) {
    why = "not inside a linalg.generic body";
    return false;
  }
  if (getScalarIntBitWidth(candidate.anchor->getResult(0).getType()) != Width) {
    why = "the result is not an i" + std::to_string(Width);
    return false;
  }
  return true;
}

/// The checks every compare rule makes: the anchor's immediate parent is a
/// generic, the reader's result type has a spyreop intrinsic, and the compared
/// type is that same type. Records the first that fails in `why`.
bool readsAtComparedWidth(const GroupCandidate &candidate, std::string &why) {
  if (!candidate.anchorBody) {
    why = "the anchor's parent is not a linalg.generic";
    return false;
  }
  auto cmp = cast<arith::CmpFOp>(candidate.anchor);
  Type resultType = candidate.readers.front()->getResult(0).getType();
  if (!isSpyreOpScalarType(resultType)) {
    why = "no spyreop intrinsic for this result type";
    return false;
  }
  // Device comparison and selection require identical input and result types.
  if (cmp.getLhs().getType() != resultType) {
    why = "the compared type and the consumer's result type differ, which one "
          "spyreop intrinsic cannot express";
    return false;
  }
  return true;
}

std::string noCounterpart(arith::CmpFOp cmp) {
  return (llvm::Twine("spyreop.compare has no counterpart for predicate '") +
          arith::stringifyCmpFPredicate(cmp.getPredicate()) + "'")
      .str();
}

/// compare-cast and compare-select: readsAtComparedWidth, and the predicate
/// has a spyreop.compare counterpart.
bool guardCompare(const GroupCandidate &candidate, std::string &why) {
  if (!readsAtComparedWidth(candidate, why))
    return false;
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
///   - zero on either side, as a scalar or a uniform input tensor, and -0.0,
///     e.g. `0 != m`, or `%z` fed by `dense<0.0>`.
///
/// Returns nothing for the rest, which keep their compare:
///
///   - an ordering, e.g. `m > 0` is false for a negative `m` that "not zero"
///     calls true;
///   - a non-uniform constant, e.g. `dense<[0.0, 1.0]>`.
std::optional<ZeroTestSelect> selectFromZeroTest(linalg::GenericOp generic,
                                                 arith::CmpFOp cmp) {
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
  if (matchPattern(resolveThroughBody(generic, cmp.getRhs()), m_AnyZeroFloat()))
    return ZeroTestSelect{0, exchangeValues};
  if (matchPattern(resolveThroughBody(generic, cmp.getLhs()), m_AnyZeroFloat()))
    return ZeroTestSelect{1, exchangeValues};
  return std::nullopt;
}

/// compare-select-zero-test: readsAtComparedWidth, and the compare is a zero
/// test spyreop.select performs itself (selectFromZeroTest, its operands
/// resolved through `anchorBody`).
bool guardZeroTest(const GroupCandidate &candidate, std::string &why) {
  if (!readsAtComparedWidth(candidate, why))
    return false;
  if (selectFromZeroTest(candidate.anchorBody,
                         cast<arith::CmpFOp>(candidate.anchor)))
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
  traceMatch(op, "numerator is 1.0 -> spyreop.reciprocal");
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
  traceMatch(cmp, llvm::Twine("arith.cmpf '") +
                      arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                      "' + arith.uitofp -> spyreop.compare");
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
  ZeroTestSelect zeroTest = *selectFromZeroTest(candidate.anchorBody, cmp);
  traceMatch(cmp, llvm::Twine("arith.cmpf '") +
                      arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                      "' tests against zero, which spyreop.select does to "
                      "its own condition -> spyreop.select alone");
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
  traceMatch(cmp, llvm::Twine("arith.cmpf '") +
                      arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                      "' + arith.select -> spyreop.compare feeding "
                      "spyreop.select");
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
/// candidate: a guard tells them apart, as guardIntegerInBody does addi32 and
/// addi64. The rules with a reader are also tried in this order by
/// matchLowerableGroup, so rules with the same anchor and the same readers are
/// listed by descending benefit.
///
/// A rule lists at most one reader, and no op is both a rule's reader and the
/// anchor of a rule without readers: benefit orders only patterns rooted on the
/// same op, so the order the driver visits ops would decide whether the group
/// or the lone op is lowered. verifySpyreOpLoweringRules checks both.
const SpyreOpLoweringRule kSpyreOpLoweringRules[] = {
    {"sqrt", opName<math::SqrtOp>(), {}, 1, guardFloatResult,
     buildUnary<spyreop::Sqrt>},
    {"exp", opName<math::ExpOp>(), {}, 1, guardFloatResult,
     buildUnary<spyreop::Exp>},
    {"rsqrt", opName<math::RsqrtOp>(), {}, 1, guardFloatResult,
     buildUnary<spyreop::RSqrt>},
    {"reciprocal", opName<arith::DivFOp>(), {}, 2, guardReciprocal,
     buildReciprocal},
    {"realdiv", opName<arith::DivFOp>(), {}, 1, guardFloatResult,
     buildBinary<spyreop::RealDiv>},
    // The integer rules over-claim: see isInsideLinalgGeneric.
    {"addi32", opName<arith::AddIOp>(), {}, 1, guardIntegerInBody<32>,
     buildBinary<spyreop::AddI32ToI32>},
    {"addi64", opName<arith::AddIOp>(), {}, 1, guardIntegerInBody<64>,
     buildBinary<spyreop::AddI64ToI64>},
    {"muli32", opName<arith::MulIOp>(), {}, 1, guardIntegerInBody<32>,
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

bool ruleMatches(const SpyreOpLoweringRule &rule,
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
  return !rule.guard || rule.guard(candidate, why);
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
    if (ruleMatches(rule, candidate, reason))
      return GroupMatch{rule.name, {candidate.anchor}};
    if (!llvm::is_contained(reasons, reason))
      reasons.push_back(reason);
  }
  if (why)
    *why = llvm::join(reasons, "; ");
  return failure();
}

} // namespace mlir::triton::spyre
