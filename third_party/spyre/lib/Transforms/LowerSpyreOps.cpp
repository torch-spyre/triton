//===- LowerSpyreOps.cpp - Select spyreop intrinsics from compute bodies --===//
//
// INSTRUCTION SELECTION for the Spyre device: arith and math ops become the
// spyreop intrinsic that does the same thing. A RULE HOST -- adding a case is
// one pattern plus one line in the pass and one lit case; a group shape is one
// entry in kGroupShapes instead, with no new pattern.
//
// KTIR -> KTIR, which is why this sits in Transforms/ rather than in
// Conversion/TritonToKTIR/: that directory's criterion is a `tt` source
// dialect, and nothing here reads a tt op. The input is KTIR whose computes are
// linalg.generic, and the output is the same KTIR with intrinsics in the
// bodies.
//
// ONE PASS FOR ALL SELECTION, and the two kinds of rule are not two tiers:
//
//   one op to one   math.sqrt -> spyreop.sqrt, and so on. `arith.divf`
//                   picks its target from the numerator: `1.0 / x` ->
//                   spyreop.reciprocal, anything else -> spyreop.realdiv.
//   a GROUP to one  a compare and a cast -> spyreop.compare. The group is what
//                   is selectable: no member of it could have been selected
//                   alone.
//
// They share one greedy pattern set, and no two rules are rooted on the same
// op, so no rule has to win over another and no order is declared anywhere.
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
// fuses only when matchLowerableGroup -- the export below, which the group
// pattern calls too, rooted on the compare and with every reader of it -- finds
// a shape covering the anchor and its one reader. An anchor read by more than
// one op is not grouped, and FuseGenerics reports the `i1` left between the
// generics. The reciprocal's constant is a scalar in the body because
// CanonicalizeGenericOperands folded the splat in. Stated as a
// contract between adjacent passes rather than left to be discovered -- and
// `resolveThroughBody` below keeps the reciprocal working on the unfused form
// too, so for that rule the dependency is about whether it FIRES, never about
// whether it is correct.
//
// THE GENERIC BODY IS THE SCOPE for a group rule, and not a predicate. Every
// compute reaches this pass as a linalg.generic, so "inside a body" is not a
// guess about whether an op is compute -- it is where the compute is. A rule
// rooted in a body never asks: an `arith.addi` on an element type in a body is
// compute, and the addi of a loop index or a tile address is not in a body at
// all. The integer 1:1 rules use the same test as a per-op PREDICATE, which is
// a proxy and is known to be one; see isInsideLinalgGeneric.
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
//     - a float width with no intrinsic (f64, bf16)
//     - an integer width with no intrinsic, or integer add/mul outside a body
//     - anything still tensor-typed, which only ConvertElementwiseToLinalg not
//       having run can produce
//
//   A group rule declines and its members stay as they were:
//     - a divide whose numerator is not a constant one, or whose constant is
//       the denominator: the realdiv rule takes it, which is right
//     - a numerator resolving to a NON-SPLAT constant tensor: reading an
//       operand through the body is sound only for a uniform value
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
// `--debug-only=lower-spyre-ops` traces the group rules' decisions rather than
// the control flow: one line per candidate looked at and what came of it, with
// the operand form named when a match was declined. The 1:1 rules say nothing
// -- there is one per op, and the output IR shows whether it fired.
//
//===----------------------------------------------------------------------===//

#include "Transforms/LowerSpyreOps.h"
#include "Transforms/Passes.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

#define DEBUG_TYPE "lower-spyre-ops"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERSPYREOPS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

using mlir::triton::spyre::GroupCandidate;
using mlir::triton::spyre::GroupMatch;

namespace {

//===----------------------------------------------------------------------===//
// Shared questions
//===----------------------------------------------------------------------===//

/// Whether spyreop's scalar float intrinsics accept this operand type.
bool isSpyreOpScalarType(Type type) {
  return isa<Float16Type, Float32Type>(type);
}

/// The bit width of `type` if it is a scalar integer, or 0 otherwise -- e.g.
/// for a not-yet-scalarized tensor of integers.
unsigned getScalarIntBitWidth(Type type) {
  auto intTy = dyn_cast<IntegerType>(type);
  return intTy ? intTy.getWidth() : 0;
}

/// Whether this op is (transitively) inside a linalg.generic body.
///
/// A PROXY, used only by the integer 1:1 rules, which have no better test:
/// plain scalar integer add and mul are also loop indices, offsets and tile
/// addressing, and there is no exact test separating those from scalarized
/// integer compute. The proxy over-claims -- `tl.arange(0, N) * stride` is an
/// `arith.muli` that will be inside a body once scalarized -- and it holds
/// today only because the tensor-of-pointers `tt.load` path is unimplemented.
/// Stated rather than glossed. The group rules use the enclosing generic as a
/// SCOPE instead, requiring the generic to be their immediate parent.
bool isInsideLinalgGeneric(Operation *op) {
  return op->getParentOfType<linalg::GenericOp>() != nullptr;
}

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
Value resolveThroughBody(linalg::GenericOp generic, Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || arg.getOwner() != generic.getBlock())
    return v;
  OpOperand *operand = generic.getMatchingOpOperand(arg);
  return generic.isDpsInput(operand) ? operand->get() : v;
}

/// The spyreop predicate computing the same thing as `p`, or nothing:
///
///   - an ordering maps from both spellings, e.g. `oeq` and `ueq` -> `equal`
///     (they differ only on NaN; see NO NaN REACHES A COMPUTE BODY);
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

void traceDecline(Operation *root, const llvm::Twine &why) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << root->getName() << " at "
                          << root->getLoc() << ": no group match (" << why
                          << ")\n");
}

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

/// Every group shape this pass lowers. The only place shapes are named: a new
/// shape is one match function, one rewrite function and one entry here.
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

//===----------------------------------------------------------------------===//
// One op to one: the unary float math ops
//===----------------------------------------------------------------------===//

/// math.sqrt/exp/rsqrt -> the matching spyreop intrinsic.
///
/// No body scope and no discriminator: these ops only ever appear in real
/// floating-point compute, never in address or index arithmetic, so every
/// scalar occurrence is one to select, inside a generic body or not. An operand
/// type with no intrinsic does not match and flows through.
template <typename Source, typename Target>
struct SelectUnaryFloat : public OpRewritePattern<Source> {
  using OpRewritePattern<Source>::OpRewritePattern;

  LogicalResult matchAndRewrite(Source op,
                                PatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    rewriter.template replaceOpWithNewOp<Target>(op, op.getType(),
                                                 op.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// One op to one: arith.divf -> spyreop.reciprocal or spyreop.realdiv
//===----------------------------------------------------------------------===//

/// `1.0 / x` -> `spyreop.reciprocal x`; any other `a / b` ->
/// `spyreop.realdiv a, b`.
///
/// The numerator is read through the enclosing body when there is one, so a
/// one passed in as a splat `ins` operand matches as well as a scalar
/// constant. Its unused block argument is then erased upstream.
struct SelectArithDivF : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();

    Value numerator = op.getLhs();
    if (linalg::GenericOp generic = dyn_cast<linalg::GenericOp>(op->getParentOp()))
      numerator = resolveThroughBody(generic, numerator);
    if (matchPattern(numerator, m_OneFloat())) {
      traceMatch(op, "numerator is 1.0 -> spyreop.reciprocal");
      rewriter.replaceOpWithNewOp<spyreop::Reciprocal>(op, op.getType(),
                                                       op.getRhs());
      return success();
    }

    rewriter.replaceOpWithNewOp<spyreop::RealDiv>(op, op.getType(), op.getLhs(),
                                                  op.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// A group to one: arith.cmpf and its readers -> spyreop.compare / select
//===----------------------------------------------------------------------===//

/// Lowers an `arith.cmpf` in one generic body together with the one op that
/// reads it, as the group shape that covers the pair:
///
///   - compare-cast: the reader is an `arith.uitofp`, e.g. `(m != 0)` used
///     multiplicatively -> `spyreop.compare`;
///   - compare-select: the reader is an `arith.select` taking it as its
///     condition, e.g. `tl.where(a > b, p, q)` -> `spyreop.select`, fed by a
///     `spyreop.compare` unless selectFromZeroTest finds a zero test the select
///     performs itself.
///
/// `arith.cmpf` alone produces an `i1`, a type no spyreop op produces, so the
/// compare is not selected on its own: its reader decides the device op
/// (matchLowerableGroup). `spyreop.compare` gives its answer "in the width
/// compared rather than as a boolean", which is compare-then-cast; `uitofp` and
/// not `sitofp`, since an `i1` read as signed is 0 or -1 and `sitofp` gives
/// -1.0 where the predicate holds.
///
/// Rooted on the compare, so the candidate holds every user of it, as
/// GroupComputes' does. The reader is rewritten, then the compare is erased.
/// Declines, leaving the arith ops, when:
///
///   - the compare has more than one user, e.g. two `arith.select`s, an
///     `arith.uitofp` and an `arith.select`, or a reader and the
///     `linalg.yield` of a compare that leaves the body;
///   - the reader is of no shape, e.g. `arith.sitofp`;
///   - the selected or cast type has no intrinsic, e.g. i32 or f64;
///   - the compared and result widths differ, e.g. f32 compared, f16 picked;
///   - a kept compare has no counterpart, e.g. `ord`.
///
/// A declined compare read in the body is then reported by
/// rejectSurvivingBooleans.
///
/// A compare with several readers is a possible extension, not done here: one
/// `spyreop.compare` at the compare, read by every reader -- a `uitofp` becomes
/// that mask, a `select` becomes `spyreop.select` on it, a zero test needs no
/// compare. A reader that leaves the body, a `linalg.yield`, still needs the
/// `i1` and would still decline.
struct SelectCompareGroup : public OpRewritePattern<arith::CmpFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpFOp cmp,
                                PatternRewriter &rewriter) const override {
    auto generic = dyn_cast<linalg::GenericOp>(cmp->getParentOp());
    if (!generic)
      return failure();
    // Each reader once, though it may read the compare more than once.
    llvm::SetVector<Operation *> readers(cmp->user_begin(), cmp->user_end());
    GroupCandidate candidate{cmp, generic, readers.takeVector()};
    std::string why;
    FailureOr<GroupMatch> match =
        triton::spyre::matchLowerableGroup(candidate, &why);
    if (failed(match)) {
      traceDecline(cmp, why);
      return failure();
    }
    groupShapeNamed(match->shape).rewrite(rewriter, candidate, *match);
    rewriter.eraseOp(cmp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// One op to one: the integer ops
//===----------------------------------------------------------------------===//

/// arith.addi -> spyreop.addi32toi32 / addi64toi64, inside a generic body only.
///
/// The body test here is the PROXY described at isInsideLinalgGeneric, not the
/// exact scope the group rules use. A width with no intrinsic flows through.
struct SelectArithAddI : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLinalgGeneric(op))
      return failure();
    unsigned width = getScalarIntBitWidth(op.getType());
    if (width == 32)
      rewriter.replaceOpWithNewOp<spyreop::AddI32ToI32>(
          op, op.getType(), op.getLhs(), op.getRhs());
    else if (width == 64)
      rewriter.replaceOpWithNewOp<spyreop::AddI64ToI64>(
          op, op.getType(), op.getLhs(), op.getRhs());
    else
      return failure();
    return success();
  }
};

/// arith.muli -> spyreop.muli32toi32, inside a generic body only. Same proxy,
/// and only one width has an intrinsic.
struct SelectArithMulI : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    if (!isInsideLinalgGeneric(op) || getScalarIntBitWidth(op.getType()) != 32)
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::MulI32ToI32>(op, op.getType(),
                                                      op.getLhs(), op.getRhs());
    return success();
  }
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
    // One line per rule, group rules and 1:1 rules in one set, each rooted on
    // a different op. See ONE PASS FOR ALL SELECTION in the header.
    patterns.add<SelectCompareGroup>(ctx);
    patterns.add<SelectArithDivF, SelectArithAddI, SelectArithMulI>(ctx);
    patterns.add<SelectUnaryFloat<math::SqrtOp, spyreop::Sqrt>,
                 SelectUnaryFloat<math::ExpOp, spyreop::Exp>,
                 SelectUnaryFloat<math::RsqrtOp, spyreop::RSqrt>>(ctx);

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

std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass() {
  return std::make_unique<LowerSpyreOpsPass>();
}

} // namespace mlir::triton::spyre
