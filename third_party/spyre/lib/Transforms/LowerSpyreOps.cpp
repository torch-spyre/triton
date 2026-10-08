//===- LowerSpyreOps.cpp - Select spyreop intrinsics from compute bodies --===//
//
// INSTRUCTION SELECTION for the Spyre device: arith and math ops become the
// spyreop intrinsic that does the same thing. A RULE HOST -- adding a case is
// one pattern plus one line in the pass and one lit case.
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
//   a REQUEST       a body whose ops all carry one `tts.spyreop_hint` -> the
//                   spyreop intrinsic the hint names, replacing the whole body.
//                   The author asked for that intrinsic with `tl.spyre_op`, and
//                   the body is its fallback; see SelectCallSite.
//
// They share one greedy pattern set, and no two rules are rooted on the same
// op, so no rule has to win over another and no order is declared anywhere.
// The request rule is rooted on the generic, and the other rules DECLINE an op
// carrying a hint: the hint says the op is already claimed, as part of its
// request. So a request's `math.exp` is never selected as `spyreop.exp` on its
// own, whichever rule the driver happens to try first.
// Splitting these across two passes is what an earlier shape did, and it
// bought a standing question -- which pass claims this op -- for nothing.
//
// EVERYTHING UNMATCHED FLOWS THROUGH, WITH TWO EXCEPTIONS. There is no
// conversion target: an op with no device form reaches the backend, which is
// the component that actually knows what it can take, and it refuses there. So
// this pass has no notion of an illegal input -- it selects what it can and
// leaves the rest exactly as it found it. A type or a predicate this file does
// not handle is a silent pass-through by design; see WHAT IS NOT SELECTED below
// for the list and what each one costs. The exceptions are an `i1` left inside
// a compute body, which rejectSurvivingBooleans reports, and an intrinsic
// request that was not selected, which rejectSurvivingHints reports: the
// author asked for the intrinsic by name, so running the fallback instead would
// ignore the request without saying so.
//
// WHAT THIS PASS RELIES ON ITS PREDECESSOR FOR. A rule matches ops in ONE body,
// and ConvertElementwiseToLinalg gives every tensor-level op a body of its own
// -- so a group spanning two tensor ops arrives spread over two generics unless
// something fused them. FuseComputeAndDataMovement is that something, and its
// `i1` clause exists for exactly this: the compare rule sees its pair only
// because that pass brought them together, and the reciprocal's constant is a
// scalar in the body for the same reason. Stated as a contract between adjacent
// passes rather than left to be discovered -- and `resolveThroughBody` below
// keeps the reciprocal working on the unfused form too, so for that rule the
// dependency is about whether it FIRES, never about whether it is correct.
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

#include "Transforms/Passes.h"

#include "Dialect/TTS/IR/Dialect.h"
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
#include "llvm/ADT/DenseMap.h"
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
/// above the generic once FuseComputeAndDataMovement has folded the splat in,
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

/// Whether `op` carries a `tts.spyreop_hint`, i.e. belongs to a `tl.spyre_op`
/// call site, which the call-site rule selects as a whole body. Every other rule
/// declines such an op.
bool hasSpyreopHint(Operation *op) {
  return static_cast<bool>(triton::tts::getSpyreopHint(op));
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

/// Matches a floating-point comparison feeding a single-result consumer
/// directly inside a linalg.generic body. The consumer and compared values
/// must have the same scalar f16 or f32 type. Returns null on a mismatch;
/// predicate support remains the caller's responsibility.
arith::CmpFOp matchComparedInput(Operation *consumer, Value input) {
  auto generic = dyn_cast<linalg::GenericOp>(consumer->getParentOp());
  if (!generic || hasSpyreopHint(consumer))
    return nullptr;
  Type resultType = consumer->getResult(0).getType();
  if (!isSpyreOpScalarType(resultType)) {
    traceDecline(consumer, "no spyreop intrinsic for this result type");
    return nullptr;
  }
  auto cmp = resolveThroughBody(generic, input).getDefiningOp<arith::CmpFOp>();
  if (!cmp) {
    traceDecline(consumer, "operand is not an arith.cmpf");
    return nullptr;
  }
  if (hasSpyreopHint(cmp)) {
    traceDecline(consumer, "the arith.cmpf belongs to an intrinsic request");
    return nullptr;
  }
  // Device comparison and selection require identical input and result types.
  if (cmp.getLhs().getType() != resultType) {
    traceDecline(consumer, "the compared type and the consumer's result type "
                           "differ, which one spyreop intrinsic cannot express");
    return nullptr;
  }
  return cmp;
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
    if (hasSpyreopHint(op) || !isSpyreOpScalarType(op.getType()))
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
    if (hasSpyreopHint(op) || !isSpyreOpScalarType(op.getType()))
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
// A group to one: arith.cmpf feeding arith.uitofp -> spyreop.compare
//===----------------------------------------------------------------------===//

/// A comparison whose answer is wanted as a NUMBER rather than as a flag -- `(m
/// != 0)` used multiplicatively, the shape a mask arrives in -- is two ops in
/// arith and one on the device.
///
/// The two are an indivisible choice. `arith.cmpf` alone produces an `i1`, a
/// type no spyreop op produces and the backend will not take in a compute body,
/// so the compare cannot be selected on its own: what consumes the `i1` is what
/// says which device op the pair is. Here the consumer is the widening cast,
/// and `spyreop.compare` is documented as giving its answer "in the width
/// compared rather than as a boolean" -- which is exactly compare-then-cast, so
/// the pair collapses with nothing left over.
///
/// ROOTED ON THE CAST, not on the compare. The consumer is the op that
/// identifies the rule, and rooting there means the match reads DOWN a def-use
/// edge it already holds rather than searching users. The compare is left to
/// dead-op elimination; when it has another reader it stays, and both readers
/// are correct, so nothing here asks about its use count.
///
/// `uitofp` and not `sitofp`: an `i1` read as unsigned is 0 or 1, which is the
/// mask wanted. Read as SIGNED it is 0 or -1, so `sitofp` gives -1.0 where the
/// predicate holds and is a different computation.
struct SelectCompare : public OpRewritePattern<arith::UIToFPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::UIToFPOp op,
                                PatternRewriter &rewriter) const override {
    auto cmp = matchComparedInput(op, op.getIn());
    if (!cmp)
      return failure();

    std::optional<spyreop::ComparePredicate> predicate =
        spyrePredicateFor(cmp.getPredicate());
    if (!predicate) {
      traceDecline(op, llvm::Twine("spyreop.compare has no counterpart for "
                                   "predicate '") +
                           arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                           "'");
      return failure();
    }

    traceMatch(op, llvm::Twine("arith.cmpf '") +
                       arith::stringifyCmpFPredicate(cmp.getPredicate()) +
                       "' + arith.uitofp -> spyreop.compare; the cmpf is left "
                       "to dead-op elimination");
    rewriter.replaceOpWithNewOp<spyreop::Compare>(
        op, op.getType(), cmp.getLhs(), cmp.getRhs(),
        spyreop::ComparePredicateAttr::get(op.getContext(), *predicate));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// A group to one: arith.cmpf feeding arith.select -> spyreop.select
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

/// Lowers `arith.cmpf` feeding `arith.select` (e.g. `tl.where(a > b, p, q)`)
/// in one of two ways, chosen by selectFromZeroTest:
///
///   - a zero test -> `spyreop.select %m, %p, %q` alone;
///   - otherwise -> `%c = spyreop.compare <greaterthan> %a, %b` then
///     `spyreop.select %c, %p, %q`.
///
/// Rooted on the select, because the cmpf's reader decides the device op. The
/// cmpf is left for dead-code removal. Declines, leaving the arith ops, when:
///
///   - the selected type has no `spyreop.select`, e.g. i32 or f64;
///   - the condition is not a cmpf, e.g. an `i1` function argument;
///   - the compared and selected widths differ, e.g. f32 compared, f16 picked;
///   - a kept compare has no counterpart, e.g. `ord`.
///
/// A declined cmpf is then reported by rejectSurvivingBooleans.
struct SelectWhere : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cmp = matchComparedInput(op, op.getCondition());
    if (!cmp)
      return failure();
    auto generic = cast<linalg::GenericOp>(op->getParentOp());
    Type selected = op.getType();

    std::optional<ZeroTestSelect> zeroTest = selectFromZeroTest(generic, cmp);
    StringRef predicateName = arith::stringifyCmpFPredicate(cmp.getPredicate());
    std::optional<spyreop::ComparePredicate> predicate =
        spyrePredicateFor(cmp.getPredicate());
    if (!zeroTest && !predicate) {
      traceDecline(op, llvm::Twine("spyreop.compare has no counterpart for "
                                   "predicate '") +
                           predicateName + "'");
      return failure();
    }

    // All rejection checks precede mutation: a failed match must not leave
    // partially constructed device operations in the body.
    Value condition = zeroTest ? zeroTest->condition : Value{};
    if (!zeroTest) {
      traceMatch(op, llvm::Twine("arith.cmpf '") + predicateName +
                         "' + arith.select -> spyreop.compare feeding "
                         "spyreop.select");
      condition = spyreop::Compare::create(
          rewriter, op.getLoc(), selected, cmp.getLhs(), cmp.getRhs(),
          spyreop::ComparePredicateAttr::get(op.getContext(), *predicate));
    } else {
      traceMatch(op,
                 llvm::Twine("arith.cmpf '") + predicateName +
                     "' tests against zero, which spyreop.select does to its "
                     "own condition -> spyreop.select alone");
    }

    // An opposite zero test reaches the same condition value, so the exchange
    // is what keeps it meaning the same thing.
    Value trueValue = op.getTrueValue(), falseValue = op.getFalseValue();
    if (zeroTest && zeroTest->exchangeValues)
      std::swap(trueValue, falseValue);

    rewriter.replaceOpWithNewOp<spyreop::Select>(op, selected, condition,
                                                 trueValue, falseValue);
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
    if (hasSpyreopHint(op) || !isInsideLinalgGeneric(op))
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
    if (hasSpyreopHint(op) || !isInsideLinalgGeneric(op) ||
        getScalarIntBitWidth(op.getType()) != 32)
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::MulI32ToI32>(op, op.getType(),
                                                      op.getLhs(), op.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// A request to one: a body hinted `tts.spyreop_hint` -> the intrinsic it names
//===----------------------------------------------------------------------===//

/// The intrinsics a request may name, and the element types each accepts --
/// SpyreOp.td's operand constraints, restricted to the builtin types this tree
/// produces (DF16 is spyreop's own type and nothing upstream of here makes one):
/// all take f16, and `takesF32` says whether f32 too. Every one is unary with
/// SameOperandsAndResultType, which is the shape the rule matches.
struct SpyreopIntrinsic {
  StringLiteral name;
  bool takesF32;
  Value (*build)(OpBuilder &, Location, Value);
};

template <typename Intrinsic>
Value buildUnaryIntrinsic(OpBuilder &builder, Location loc, Value operand) {
  return Intrinsic::create(builder, loc, operand.getType(), operand);
}

constexpr SpyreopIntrinsic kSpyreopIntrinsics[] = {
    {"gelu", /*takesF32=*/false, buildUnaryIntrinsic<spyreop::GeLU>},
    {"silu", /*takesF32=*/true, buildUnaryIntrinsic<spyreop::SiLU>},
    {"sigmoid", /*takesF32=*/true, buildUnaryIntrinsic<spyreop::Sigmoid>},
};

const SpyreopIntrinsic *lookupIntrinsic(StringRef name) {
  for (const SpyreopIntrinsic &entry : kSpyreopIntrinsics)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

bool takesType(const SpyreopIntrinsic &entry, Type type) {
  return isa<Float16Type>(type) || (entry.takesF32 && isa<Float32Type>(type));
}

/// Where each request's hinted ops are, counted once before the greedy driver
/// runs: how many generic bodies hold ops of the request, and whether any of
/// its ops is outside a body altogether.
///
/// This is what lets the rule tell a request that is one body from a request
/// SPLIT across two -- each half of which, seen alone, also reads one value and
/// yields one, and would be replaced by the whole intrinsic. Counted rather
/// than held as pointers, because the unused-operand cleanup in the same
/// fixpoint rebuilds a generic it trims, and a count of one survives that.
struct CallSiteSpread {
  unsigned bodies = 0;
  bool outsideBody = false;
};
using CallSiteSpreads = llvm::DenseMap<DictionaryAttr, CallSiteSpread>;

/// The ops of `generic`'s body a request rule counts: everything but the
/// terminator and constants. A constant is neutral -- the canonicalizer hoists
/// and merges constants without regard to hints, and fusion folds a splat
/// operand into a fresh unhinted scalar -- so whether one carries the hint says
/// nothing about the request.
SmallVector<Operation *> callSiteMembers(linalg::GenericOp generic) {
  SmallVector<Operation *> members;
  generic.getBlock()->walk([&](Operation *op) {
    if (!isa<linalg::YieldOp>(op) && !op->hasTrait<OpTrait::ConstantLike>())
      members.push_back(op);
  });
  return members;
}

CallSiteSpreads spreadOfCallSites(ModuleOp module) {
  CallSiteSpreads spreads;
  module.walk([&](linalg::GenericOp generic) {
    llvm::SmallSetVector<DictionaryAttr, 2> hints;
    for (Operation *op : callSiteMembers(generic))
      if (DictionaryAttr hint = triton::tts::getSpyreopHint(op))
        hints.insert(hint);
    for (DictionaryAttr hint : hints)
      ++spreads[hint].bodies;
  });
  module.walk([&](Operation *op) {
    DictionaryAttr hint = triton::tts::getSpyreopHint(op);
    if (hint && !op->hasTrait<OpTrait::ConstantLike>() &&
        !op->getParentOfType<linalg::GenericOp>())
      spreads[hint].outsideBody = true;
  });
  return spreads;
}

/// What the request rule needs from a body it can replace.
struct CallSiteMatch {
  DictionaryAttr hint;
  const SpyreopIntrinsic *intrinsic = nullptr;
  BlockArgument input;
  Value result;
  SmallVector<Operation *> members;
};

/// Whether `generic`'s body is exactly one request, whole. On a decline,
/// `why` says which condition failed, phrased for a diagnostic, and the result
/// is failure; a body with no hinted op at all is not a request, and fails
/// with `why` empty.
///
/// The conditions, in the order asked:
///   - every member carries the same hint: one request, and nothing else;
///   - no other body and no op outside a body carries it: the request is here
///     whole, not split by a fusion that did not happen;
///   - the members read exactly one block argument, an `ins` one, constants
///     aside, and the body yields one value a member produced: the intrinsic
///     is unary;
///   - an intrinsic of that name takes that type, and the type is the yielded
///     one, since every intrinsic here has the same operand and result type.
FailureOr<CallSiteMatch> matchCallSite(linalg::GenericOp generic,
                                     const CallSiteSpreads &spreads,
                                     std::string &why) {
  why.clear();
  CallSiteMatch match;
  match.members = callSiteMembers(generic);
  for (Operation *op : match.members)
    if ((match.hint = triton::tts::getSpyreopHint(op)))
      break;
  if (!match.hint)
    return failure();

  StringRef name = triton::tts::getSpyreopHintName(match.hint);
  for (Operation *op : match.members) {
    DictionaryAttr hint = triton::tts::getSpyreopHint(op);
    if (hint == match.hint)
      continue;
    why = hint ? "its body also holds ops of another request, " +
                    triton::tts::getSpyreopHintName(hint).str()
              : "its body also holds '" + op->getName().getStringRef().str() +
                    "', which is not part of the request";
    return failure();
  }

  auto spread = spreads.lookup(match.hint);
  if (spread.bodies != 1 || spread.outsideBody) {
    why = "the request's ops are spread over " +
          std::to_string(spread.bodies) + " compute bodies" +
          (spread.outsideBody ? " and outside any body" : "") +
          ", and the intrinsic can replace only one whole body";
    return failure();
  }

  // The values the request reads. A block argument whose `ins` operand is a
  // constant is a constant the fusion did not fold into the body, and is
  // neutral like one.
  llvm::SmallSetVector<BlockArgument, 2> read;
  for (Operation *op : match.members)
    for (Value v : op->getOperands()) {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg || arg.getOwner() != generic.getBlock())
        continue;
      Operation *def = resolveThroughBody(generic, arg).getDefiningOp();
      if (def && def->hasTrait<OpTrait::ConstantLike>())
        continue;
      read.insert(arg);
    }
  auto yield = cast<linalg::YieldOp>(generic.getBlock()->getTerminator());
  if (read.size() != 1 ||
      !generic.isDpsInput(generic.getMatchingOpOperand(read.front()))) {
    why = "the body reads " + std::to_string(read.size()) +
          " block arguments, and the intrinsic takes one input";
    return failure();
  }
  if (yield->getNumOperands() != 1 || !yield->getOperand(0).getDefiningOp() ||
      !llvm::is_contained(match.members,
                          yield->getOperand(0).getDefiningOp())) {
    why = "the body does not yield exactly one value the request computed";
    return failure();
  }
  match.input = read.front();
  match.result = yield->getOperand(0);

  if (match.input.getType() != match.result.getType()) {
    std::string types;
    llvm::raw_string_ostream os(types);
    os << match.input.getType() << " in, " << match.result.getType() << " out";
    why = "the request takes and yields different types (" + types +
          "), and spyreop." + name.str() + " has one type for both";
    return failure();
  }
  const SpyreopIntrinsic *intrinsic = lookupIntrinsic(name);
  if (!intrinsic) {
    why = "there is no spyreop intrinsic named '" + name.str() + "'";
    return failure();
  }
  if (!takesType(*intrinsic, match.input.getType())) {
    std::string type;
    llvm::raw_string_ostream os(type);
    os << match.input.getType();
    why = "spyreop." + name.str() + " does not take " + type;
    return failure();
  }
  match.intrinsic = intrinsic;
  return match;
}

/// Replaces a body that is one whole intrinsic request with that intrinsic.
///
/// Rooted on the generic and not on a member, because the request is the BODY:
/// which op the fallback happens to end in says nothing, and a rule rooted on
/// one member would have to search the rest. What the members computed --
/// the fallback, casts included -- is discarded whole, and the body becomes the
/// intrinsic applied to its input. The input's unused siblings, a captured
/// constant among them, then go to the unused-operand cleanup in the same
/// fixpoint.
///
/// Declines are not reported here: what is left hinted after the fixpoint is
/// reported by rejectSurvivingHints, which asks matchCallSite again for the
/// reason, so the diagnosis and the rule cannot disagree.
struct SelectCallSite : public OpRewritePattern<linalg::GenericOp> {
  SelectCallSite(MLIRContext *ctx, const CallSiteSpreads &spreads)
      : OpRewritePattern(ctx), spreads(spreads) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    std::string why;
    FailureOr<CallSiteMatch> match = matchCallSite(generic, spreads, why);
    if (failed(match)) {
      if (!why.empty())
        traceDecline(generic, why);
      return failure();
    }
    StringRef name = triton::tts::getSpyreopHintName(match->hint);
    traceMatch(generic, "body is intrinsic request " +
                            llvm::Twine(name) + " -> spyreop." + name);

    auto yield = cast<linalg::YieldOp>(generic.getBlock()->getTerminator());
    rewriter.setInsertionPoint(yield);
    Value selected =
        match->intrinsic->build(rewriter, match->result.getLoc(), match->input);
    rewriter.modifyOpInPlace(yield, [&] { yield->setOperand(0, selected); });
    // Reverse program order, so each op's users are gone before it is.
    for (Operation *op : llvm::reverse(match->members))
      rewriter.eraseOp(op);
    return success();
  }

  const CallSiteSpreads &spreads;
};

/// Reports every intrinsic request selection left in place, once per request,
/// naming the intrinsic and why. The author asked for it by name, so running
/// its fallback instead would ignore that request without saying so.
///
/// Whatever the cause, it is reported the same way, and two causes deserve
/// naming because nothing at the kernel line shows them: the request was split
/// across two bodies, because a fusion did not happen; or an op of it was
/// folded into an op outside it, which leaves a body holding both.
LogicalResult rejectSurvivingHints(ModuleOp module) {
  CallSiteSpreads spreads = spreadOfCallSites(module);
  llvm::DenseMap<DictionaryAttr, Operation *> firstSeen;
  SmallVector<DictionaryAttr> order;
  module.walk([&](Operation *op) {
    DictionaryAttr hint = triton::tts::getSpyreopHint(op);
    if (!hint)
      return;
    if (firstSeen.try_emplace(hint, op).second)
      order.push_back(hint);
  });

  for (DictionaryAttr hint : order) {
    Operation *op = firstSeen.lookup(hint);
    std::string why = "an op of it is outside any compute body";
    if (auto generic = op->getParentOfType<linalg::GenericOp>())
      if (succeeded(matchCallSite(generic, spreads, why)))
        why = "the rewrite did not reach a fixpoint";
    mlir::emitError(op->getLoc())
        << "lower-spyre-ops: tl.spyre_op(\"" << triton::tts::getSpyreopHintName(hint)
        << "\") was not selected: " << why
        << ". The request is explicit, so its fallback is not used instead";
  }
  return success(order.empty());
}

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
///   - `cmpf oeq` read by `arith.sitofp` (the compare rule wants `uitofp`).
///
/// Not reported, because each is the tensor form crossing a body boundary,
/// which FuseComputeAndDataMovement removes:
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
        if (spyrePredicateFor(cmp.getPredicate()))
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

    // Where each request is, before anything is rewritten; see
    // CallSiteSpread for why this is asked once rather than per match.
    CallSiteSpreads spreads = spreadOfCallSites(module);

    RewritePatternSet patterns(ctx);
    // One line per rule, request, group and 1:1 rules in one set, each rooted
    // on a different op. See ONE PASS FOR ALL SELECTION in the header.
    patterns.add<SelectCallSite>(ctx, spreads);
    patterns.add<SelectCompare, SelectWhere>(ctx);
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

    // The two things this pass does report. A request not selected is the
    // author's explicit ask going unanswered. An `i1` is not a selection
    // failure at all -- see rejectSurvivingBooleans on why an unrepresentable
    // TYPE is a different kind of thing from an op the device happens not to
    // do. Both are asked, so one run reports both.
    bool hintsSurvived = failed(rejectSurvivingHints(module));
    if (failed(rejectSurvivingBooleans(module)) || hintsSurvived)
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass() {
  return std::make_unique<LowerSpyreOpsPass>();
}

} // namespace mlir::triton::spyre
