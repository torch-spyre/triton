//===- SpyreOpLoweringRules.cpp - The group shapes and shared questions ---===//
//
// The group shapes LowerSpyreOps lowers and FuseGenerics fuses for, and the
// questions they share with LowerSpyreOps' one-op patterns. See
// SpyreOpLoweringRules.h.
//
// A new group shape is one match function, one rewrite function and one entry
// in kGroupShapes, with no new pattern in either pass.
//
// NO NaN REACHES A COMPUTE BODY. Every rule assumes it, which lets a group:
//
//   - select an unordered predicate as its ordered one, e.g. `une` (Triton's
//     `!=`) as `spyreop.compare <notequal>`;
//   - drop a zero test in front of a select, e.g. `select(m != 0, p, q)` ->
//     `spyreop.select %m, %p, %q`.
//
// The rewrites trace under `--debug-only=lower-spyre-ops`, the pass that runs
// them.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpyreOp/Utils/SpyreOpLoweringRules.h"

#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"
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

void traceMatch(Operation *root, const llvm::Twine &what) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                          << root->getLoc() << ": " << what << "\n");
}

/// The checks every compare group makes of its reader: the reader's result
/// type has a spyreop intrinsic, and the compared type is that same type.
/// Records the first that fails in `why`.
bool readsAtComparedWidth(arith::CmpFOp cmp, Operation *reader,
                          std::string &why) {
  Type resultType = reader->getResult(0).getType();
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

//===----------------------------------------------------------------------===//
// Group shapes
//===----------------------------------------------------------------------===//

/// A zero test that `spyreop.select` performs itself, found by
/// selectFromZeroTest. `condition` is the scalar already in the body.
struct ZeroTestSelect {
  Value condition;
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
    return ZeroTestSelect{cmp.getLhs(), exchangeValues};
  if (matchPattern(resolveThroughBody(generic, cmp.getLhs()), m_AnyZeroFloat()))
    return ZeroTestSelect{cmp.getRhs(), exchangeValues};
  return std::nullopt;
}

constexpr StringLiteral kCompareCast = "compare-cast";
constexpr StringLiteral kCompareSelect = "compare-select";

/// The compare-cast shape, lowered by rewriteCompareCast to one
/// spyreop.compare:
///
///   %c = arith.cmpf ogt, %a, %b : f16
///   %m = arith.uitofp %c : i1 to f16
///
/// Matches when:
///
///   - the anchor is an arith.cmpf with a spyreop.compare counterpart;
///   - the reader is an arith.uitofp;
///   - the reader's result type has a spyreop intrinsic and is the compared
///     type.
///
/// It receives a candidate with exactly one reader (matchLowerableGroup).
/// Returns the match with `producerOps = {cmpf}`, or failure with `why` set.
FailureOr<GroupMatch> matchCompareCast(const GroupCandidate &candidate,
                                       std::string &why) {
  auto cmp = dyn_cast_or_null<arith::CmpFOp>(candidate.anchor);
  if (!cmp) {
    why = "the anchor is not an arith.cmpf";
    return failure();
  }
  Operation *reader = candidate.readers.front();
  if (!isa<arith::UIToFPOp>(reader)) {
    why = "the reader, '" + reader->getName().getStringRef().str() +
          "', is not an arith.uitofp";
    return failure();
  }
  if (!readsAtComparedWidth(cmp, reader, why))
    return failure();
  if (!spyrePredicateFor(cmp.getPredicate())) {
    why = noCounterpart(cmp);
    return failure();
  }
  return GroupMatch{kCompareCast, {cmp}};
}

/// The compare-select shape, lowered by rewriteCompareSelect to a
/// spyreop.select, with or without a spyreop.compare feeding it:
///
///   %c = arith.cmpf une, %m, %zero : f16
///   %r = arith.select %c, %p, %q : f16
///
/// Matches when:
///
///   - the anchor is an arith.cmpf;
///   - the reader is an arith.select;
///   - the reader's result type has a spyreop intrinsic and is the compared
///     type;
///   - the compare is a zero test spyreop.select performs itself
///     (selectFromZeroTest, its operands resolved through `anchorBody`), or
///     its predicate has a spyreop.compare counterpart.
///
/// It receives a candidate with exactly one reader (matchLowerableGroup).
/// Returns the match with `producerOps = {cmpf}`, or failure with `why` set.
FailureOr<GroupMatch> matchCompareSelect(const GroupCandidate &candidate,
                                         std::string &why) {
  auto cmp = dyn_cast_or_null<arith::CmpFOp>(candidate.anchor);
  if (!cmp) {
    why = "the anchor is not an arith.cmpf";
    return failure();
  }
  Operation *reader = candidate.readers.front();
  if (!isa<arith::SelectOp>(reader)) {
    why = "the reader, '" + reader->getName().getStringRef().str() +
          "', is not an arith.select";
    return failure();
  }
  if (!readsAtComparedWidth(cmp, reader, why))
    return failure();
  if (!selectFromZeroTest(candidate.anchorBody, cmp) &&
      !spyrePredicateFor(cmp.getPredicate())) {
    why = noCounterpart(cmp);
    return failure();
  }
  return GroupMatch{kCompareSelect, {cmp}};
}

/// Lowers the compare-cast group `match` found in `candidate`: the reader, an
/// arith.uitofp, becomes `spyreop.compare` of the compare's operands under its
/// predicate's counterpart.
void rewriteCompareCast(PatternRewriter &rewriter,
                        const GroupCandidate &candidate,
                        const GroupMatch &match) {
  auto cmp = cast<arith::CmpFOp>(match.producerOps.front());
  Operation *reader = candidate.readers.front();
  traceMatch(cmp, llvm::Twine("arith.cmpf '") +
                      arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                      "' + arith.uitofp -> spyreop.compare");
  rewriter.setInsertionPoint(reader);
  rewriter.replaceOpWithNewOp<spyreop::Compare>(
      reader, reader->getResult(0).getType(), cmp.getLhs(), cmp.getRhs(),
      spyreop::ComparePredicateAttr::get(
          cmp.getContext(), *spyrePredicateFor(cmp.getPredicate())));
}

/// Lowers the compare-select group `match` found in `candidate`: the reader,
/// an arith.select, becomes a `spyreop.select`. Its condition is the zero
/// test's operand when selectFromZeroTest finds one, with the values exchanged
/// when the test is `== 0`, and otherwise a `spyreop.compare` of the compare's
/// operands placed before the select.
void rewriteCompareSelect(PatternRewriter &rewriter,
                          const GroupCandidate &candidate,
                          const GroupMatch &match) {
  auto cmp = cast<arith::CmpFOp>(match.producerOps.front());
  auto select = cast<arith::SelectOp>(candidate.readers.front());
  Type selected = select.getType();
  std::optional<ZeroTestSelect> zeroTest =
      selectFromZeroTest(candidate.anchorBody, cmp);
  StringRef predicateName = arith::stringifyCmpFPredicate(cmp.getPredicate());
  rewriter.setInsertionPoint(select);

  Value condition;
  if (zeroTest) {
    traceMatch(cmp, llvm::Twine("arith.cmpf '") + predicateName +
                        "' tests against zero, which spyreop.select does to "
                        "its own condition -> spyreop.select alone");
    condition = zeroTest->condition;
  } else {
    traceMatch(cmp, llvm::Twine("arith.cmpf '") + predicateName +
                        "' + arith.select -> spyreop.compare feeding "
                        "spyreop.select");
    condition = spyreop::Compare::create(
        rewriter, select.getLoc(), selected, cmp.getLhs(), cmp.getRhs(),
        spyreop::ComparePredicateAttr::get(
            cmp.getContext(), *spyrePredicateFor(cmp.getPredicate())));
  }

  // An opposite zero test reaches the same condition value, so the exchange
  // is what keeps it meaning the same thing.
  Value trueValue = select.getTrueValue(), falseValue = select.getFalseValue();
  if (zeroTest && zeroTest->exchangeValues)
    std::swap(trueValue, falseValue);

  rewriter.replaceOpWithNewOp<spyreop::Select>(select, selected, condition,
                                               trueValue, falseValue);
}

/// One group shape: its name, the match function that decides whether it
/// covers a candidate, and the rewrite that lowers a candidate it matched.
struct GroupShape {
  StringLiteral name;
  FailureOr<GroupMatch> (*match)(const GroupCandidate &, std::string &);
  void (*rewrite)(PatternRewriter &, const GroupCandidate &,
                  const GroupMatch &);
};

/// Every group shape LowerSpyreOps lowers. The only place shapes are named: a
/// new shape is one match function, one rewrite function and one entry here.
constexpr GroupShape kGroupShapes[] = {
    {kCompareCast, matchCompareCast, rewriteCompareCast},
    {kCompareSelect, matchCompareSelect, rewriteCompareSelect},
};

/// The entry of kGroupShapes named `name`.
const GroupShape &groupShapeNamed(StringRef name) {
  for (const GroupShape &shape : kGroupShapes)
    if (shape.name == name)
      return shape;
  llvm_unreachable("a GroupMatch names an entry of kGroupShapes");
}

} // namespace

FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why) {
  if (candidate.readers.size() != 1) {
    if (why)
      *why = candidate.readers.empty()
                 ? std::string("nothing reads the anchor")
                 : "the anchor has " +
                       std::to_string(candidate.readers.size()) +
                       " readers, and a group takes exactly one";
    return failure();
  }
  std::string reasons;
  for (const GroupShape &shape : kGroupShapes) {
    std::string reason;
    FailureOr<GroupMatch> match = shape.match(candidate, reason);
    if (succeeded(match))
      return match;
    reasons += (reasons.empty() ? "" : "; ") + reason;
  }
  if (why)
    *why = reasons;
  return failure();
}

void rewriteLowerableGroup(PatternRewriter &rewriter,
                           const GroupCandidate &candidate,
                           const GroupMatch &match) {
  groupShapeNamed(match.shape).rewrite(rewriter, candidate, match);
}

} // namespace mlir::triton::spyre
