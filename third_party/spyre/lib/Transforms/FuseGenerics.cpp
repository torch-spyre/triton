//===- FuseGenerics.cpp - Combine linalg.generic producers into consumers -===//
//
// Combines a producer linalg.generic into the consumer that reads it, by two
// patterns. Both are rooted on the consumer and look at one `ins` at a time;
// they differ in what the producer computes for the value the consumer reads:
//
//   AbsorbForwardingGeneric  nothing: the producer yields one of its `ins`
//                            block arguments. The consumer's operand is
//                            repointed at that input, under the composed map.
//   GroupComputes            an op, the anchor: the producer is fused into the
//                            consumer when the anchor and every reader of it
//                            form a group LowerSpyreOps lowers as one intrinsic
//                            (formsLowerableGroup).
//
// After the fixpoint, checkNoBooleanBetweenGenerics reports every `i1` tensor
// still passed from one generic to another.
//
// Both patterns check the consumer's operand list as it would be after the
// rewrite, before mutating anything (checkOperandRewrite): D1, the producer is
// all-parallel; D2, every map is projected; D3, every loop dim is named; D4, no
// value is both an `ins` and an `outs`.
//
// HOW TO ADD A PATTERN. A pattern matches on its PRODUCER'S KIND: the op that
// defines the operand, or what the producer's body computes for it. Never on
// the operand's type, nor on anything downstream. A new pattern names a
// producer kind no existing pattern matches; one whose kind overlaps an
// existing pattern's is a new case of that pattern. Its lit file feeds it the
// other patterns' producers and checks that it does not fire. A new group shape
// is not a pattern here: it is a match function in LowerSpyreOps, which
// formsLowerableGroup then answers for.
//
// `--debug-only=fuse-generics` traces each rewrite and each decline, with the
// check that declined it.
//
//===----------------------------------------------------------------------===//

#include "Transforms/LowerSpyreOps.h"
#include "Transforms/Passes.h"
#include "Utils/GenericOperands.h"

#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

#define DEBUG_TYPE "fuse-generics"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_FUSEGENERICS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

using OperandList = SmallVector<std::pair<Value, AffineMap>>;

/// `op`'s ins and outs, each with its indexing map.
void operandLists(linalg::GenericOp op, OperandList &ins, OperandList &outs) {
  for (OpOperand *in : op.getDpsInputOperands())
    ins.emplace_back(in->get(), op.getMatchingIndexingMap(in));
  for (OpOperand &out : op.getDpsInitsMutable())
    outs.emplace_back(out.get(), op.getMatchingIndexingMap(&out));
}

/// The value `producer` yields for its result `result`.
Value yieldedFor(linalg::GenericOp producer, OpResult result) {
  return producer.getBlock()->getTerminator()->getOperand(
      result.getResultNumber());
}

/// The `ins` block argument `producer` yields for `result`, or null.
BlockArgument forwardedInput(linalg::GenericOp producer, OpResult result) {
  auto arg = dyn_cast<BlockArgument>(yieldedFor(producer, result));
  if (!arg || arg.getOwner() != producer.getBlock() ||
      arg.getArgNumber() >= producer.getNumDpsInputs())
    return {};
  return arg;
}

bool isAllParallel(linalg::GenericOp op) {
  return op.getNumParallelLoops() == op.getNumLoops();
}

void traceDecline(StringRef pattern, OpOperand *operand, const llvm::Twine &why) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << pattern << ": operand "
                          << operand->getOperandNumber() << " of generic at "
                          << operand->getOwner()->getLoc() << ": declined, "
                          << why << "\n");
}

//===----------------------------------------------------------------------===//
// AbsorbForwardingGeneric
//===----------------------------------------------------------------------===//

/// A consumer `ins` that is result `r` of an all-parallel linalg.generic `P`
/// which yields for `r` one of its `ins` block arguments, `x`, under a
/// permutation result map `R` -> the operand repointed at `x`, under
/// `A.compose(inversePermutation(R)).compose(C)`, where `A` is `P`'s map for
/// `x` and `C` the consumer's map for the operand. The rest of `P`'s body is
/// not read. `P` is erased by the driver once dead.
struct AbsorbForwardingGeneric : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!consumer.hasPureTensorSemantics())
      return failure();
    for (OpOperand *operand : consumer.getDpsInputOperands()) {
      auto result = dyn_cast<OpResult>(operand->get());
      auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
      if (!producer)
        continue;
      BlockArgument arg = forwardedInput(producer, result);
      if (!arg)
        continue;
      if (!isAllParallel(producer)) {
        traceDecline("AbsorbForwardingGeneric", operand,
                     "D1 the producer has a reduction loop");
        continue;
      }
      AffineMap r = producer.getIndexingMapMatchingResult(result);
      if (!r.isPermutation()) {
        traceDecline("AbsorbForwardingGeneric", operand,
                     "the producer's result map is not a permutation");
        continue;
      }
      OpOperand *x = producer.getDpsInputOperand(arg.getArgNumber());
      AffineMap c = consumer.getMatchingIndexingMap(operand);
      AffineMap composed = producer.getMatchingIndexingMap(x)
                               .compose(inversePermutation(r))
                               .compose(c);

      OperandList ins, outs;
      operandLists(consumer, ins, outs);
      unsigned index = operand->getOperandNumber();
      ins[index] = {x->get(), composed};
      std::string why;
      if (failed(triton::spyre::checkOperandRewrite(
              ins, outs, consumer.getNumLoops(), why))) {
        traceDecline("AbsorbForwardingGeneric", operand, why);
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] AbsorbForwardingGeneric: operand "
                 << index << " of generic at " << consumer.getLoc()
                 << ": reads the input of the generic at " << producer.getLoc()
                 << " under " << composed << "\n");
      SmallVector<AffineMap> maps = consumer.getIndexingMapsArray();
      maps[index] = composed;
      rewriter.modifyOpInPlace(consumer, [&] {
        operand->set(x->get());
        consumer.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      });
      return success();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// GroupComputes
//===----------------------------------------------------------------------===//

/// A consumer `ins` that is result `r` of an all-parallel linalg.generic `P`
/// which yields for `r` an op result, the anchor -> `P` fused into the
/// consumer, when:
///
///   - linalg::areElementwiseOpsFusable holds;
///   - every use of the matching block argument in the consumer is a reader
///     that, with the anchor, forms a group (formsLowerableGroup), and there is
///     at least one;
///   - when `P` has a user other than the consumer, every non-terminator op in
///     `P`'s body is the anchor, so nothing outside the group is copied;
///   - D2-D4 hold on the fused operand list.
///
/// The rewrite is linalg::fuseElementwiseOps, then
/// linalg::deduplicateOperandsAndRemoveDeadResults on the fused op, which
/// removes the result kept for `P`'s other users that nothing reads.
struct GroupComputes : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!consumer.hasPureTensorSemantics())
      return failure();
    for (OpOperand *operand : consumer.getDpsInputOperands())
      if (succeeded(groupOperand(consumer, operand, rewriter)))
        return success();
    return failure();
  }

private:
  /// The candidate, group, duplication and operand checks, without rewriting.
  static LogicalResult checkGroup(linalg::GenericOp consumer,
                                  OpOperand *operand,
                                  linalg::GenericOp producer, Operation *anchor,
                                  std::string &why) {
    if (!isAllParallel(producer)) {
      why = "D1 the producer has a reduction loop";
      return failure();
    }
    if (!linalg::areElementwiseOpsFusable(operand)) {
      why = "not fusable under linalg::areElementwiseOpsFusable";
      return failure();
    }

    BlockArgument arg = consumer.getMatchingBlockArgument(operand);
    if (arg.use_empty()) {
      why = "the consumer does not read it";
      return failure();
    }
    for (Operation *reader : arg.getUsers()) {
      std::string reason;
      if (!triton::spyre::formsLowerableGroup(producer, anchor, reader,
                                              &reason)) {
        why = "'" + anchor->getName().getStringRef().str() + "' and its reader '" +
              reader->getName().getStringRef().str() +
              "' form no group: " + reason;
        return failure();
      }
    }

    bool shared = llvm::any_of(producer->getUsers(), [&](Operation *user) {
      return user != consumer.getOperation();
    });
    if (shared && llvm::any_of(producer.getBlock()->without_terminator(),
                               [&](Operation &op) { return &op != anchor; })) {
      why = "the producer has another user and its body holds ops outside the "
            "group";
      return failure();
    }

    // The fused op's ins: the consumer's other ins, and each producer ins
    // under A o R^-1 o C, the composition fuseElementwiseOps writes.
    OperandList ins, outs;
    operandLists(consumer, ins, outs);
    ins.erase(ins.begin() + operand->getOperandNumber());
    AffineMap toProducerLoops =
        inversePermutation(producer.getIndexingMapMatchingResult(
                               cast<OpResult>(operand->get())))
            .compose(consumer.getMatchingIndexingMap(operand));
    for (OpOperand *in : producer.getDpsInputOperands())
      ins.emplace_back(in->get(), producer.getMatchingIndexingMap(in).compose(
                                      toProducerLoops));
    return triton::spyre::checkOperandRewrite(ins, outs,
                                              consumer.getNumLoops(), why);
  }

  static LogicalResult groupOperand(linalg::GenericOp consumer,
                                    OpOperand *operand,
                                    PatternRewriter &rewriter) {
    auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
    if (!producer)
      return failure();
    Value yielded = yieldedFor(producer, cast<OpResult>(operand->get()));
    Operation *anchor = yielded.getDefiningOp();
    if (!anchor || anchor->getBlock() != producer.getBlock())
      return failure();

    std::string why;
    if (failed(checkGroup(consumer, operand, producer, anchor, why))) {
      traceDecline("GroupComputes", operand, why);
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] GroupComputes: operand "
                            << operand->getOperandNumber() << " of generic at "
                            << consumer.getLoc() << ": fusing the generic at "
                            << producer.getLoc() << ", anchor '"
                            << anchor->getName() << "'\n");
    FailureOr<linalg::ElementwiseOpFusionResult> fused =
        linalg::fuseElementwiseOps(rewriter, operand);
    if (failed(fused))
      return failure();
    // The consumer's results go to the fused op; the producer's other users
    // keep reading the producer.
    for (auto [orig, repl] : fused->replacements)
      rewriter.replaceUsesWithIf(orig, repl, [&](OpOperand &use) {
        return use.get().getDefiningOp() != producer;
      });
    rewriter.eraseOp(consumer);
    (void)linalg::deduplicateOperandsAndRemoveDeadResults(
        rewriter, cast<linalg::GenericOp>(fused->fusedOp),
        /*removeOutputs=*/true);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// After the fixpoint: an i1 tensor between generics
//===----------------------------------------------------------------------===//

/// Every `ins` of element type i1 defined by a linalg.generic is an error at
/// the consumer. Notes: the op that made the i1 in the producer (or the operand
/// it forwards), each reader in the consumer, and the group shapes
/// LowerSpyreOps lowers an i1 in.
LogicalResult checkNoBooleanBetweenGenerics(ModuleOp mod) {
  LogicalResult result = success();
  mod.walk([&](linalg::GenericOp consumer) {
    for (OpOperand *operand : consumer.getDpsInputOperands()) {
      if (!getElementTypeOrSelf(operand->get().getType()).isInteger(1))
        continue;
      auto producer = operand->get().getDefiningOp<linalg::GenericOp>();
      if (!producer)
        continue;
      result = failure();

      Value yielded = yieldedFor(producer, cast<OpResult>(operand->get()));
      InFlightDiagnostic diag =
          consumer.emitError()
          << "fuse-generics: operand " << operand->getOperandNumber()
          << " is an i1 tensor passed from one linalg.generic to another; no "
             "spyreop op produces or consumes i1, so nothing downstream can "
             "lower it";
      if (Operation *maker = yielded.getDefiningOp())
        diag.attachNote(maker->getLoc())
            << "made here, by '" << maker->getName() << "'";
      else if (auto arg = dyn_cast<BlockArgument>(yielded);
               arg && arg.getOwner() == producer.getBlock())
        diag.attachNote(producer.getLoc())
            << "forwarded by this linalg.generic from its operand "
            << producer.getMatchingOpOperand(arg)->getOperandNumber();
      for (Operation *reader :
           consumer.getMatchingBlockArgument(operand).getUsers())
        diag.attachNote(reader->getLoc())
            << "read here, by '" << reader->getName() << "'";
      diag.attachNote()
          << "LowerSpyreOps lowers an i1 only within a group: an arith.cmpf "
             "read by arith.uitofp, or by arith.select, at the compared width "
             "(SelectCompare, SelectWhere)";
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// The layout gate
//===----------------------------------------------------------------------===//
//
// After the fixpoint, a re-indexing tensor op left in front of a generic on a
// path RewriteDescriptorLayoutGeneric will physicalize is an error. The path is
// that pass's own scope: the generics one hop off the loads and stores of every
// `tts.tensor_layout`-annotated memory view. A module with no annotated view is
// not checked.

/// One op's coordinate restatement, as far as this pass can state it.
///
/// THREE answers, not two, and the third is what lets the gate's rejection be
/// derived rather than declared:
///
///   `source` null          -- not a coordinate restatement at all. A compute,
///   an
///                             scf result, a cast, a load. Nothing in this pass
///                             is about it, and it is neither absorbed nor
///                             refused.
///   `source` set, `why` empty
///                          -- a restatement whose map is derivable AND is a
///                             projected coordinate map. Absorbable.
///   `source` set, `why` set
///                          -- a restatement this pass cannot hand to a
///                             scheduler. `map` may still be present (a genuine
///                             linearization) or absent (no map exists at all);
///                             `why` says which, in the words the diagnostic
///                             uses.
///
/// `why` is filled HERE, by the code that read the reassociation, that being
/// the only place that knows how many non-unit dims a group fused.
struct CoordinateRestatement {
  /// The value a consumer should read instead of this op's result.
  Value source;
  /// This op's RESULT coordinates -> `source`'s coordinates. Null when no such
  /// map is derivable.
  AffineMap map;
  /// Why this restatement is not a projected coordinate map; empty when it is.
  std::string why;

  bool isReindex() const { return source != nullptr; }
};

/// Not a coordinate restatement.
CoordinateRestatement notReindex() { return CoordinateRestatement{}; }

/// A restatement with no derivable map.
CoordinateRestatement noMap(Value source, StringRef why) {
  return CoordinateRestatement{source, AffineMap(), why.str()};
}

/// The extents of `shape` at `dims`, multiplied -- the row-major stride of the
/// dim just outside them. Every entry must be static; the caller checks.
int64_t productOf(ArrayRef<int64_t> shape, ArrayRef<int64_t> dims) {
  int64_t p = 1;
  for (int64_t d : dims)
    p *= shape[d];
  return p;
}

/// The dims of `group` whose extent in `shape` is not 1. A unit dim carries no
/// coordinate: it can only ever be read at 0, and it contributes nothing to a
/// linearization either.
SmallVector<int64_t> nonUnitDims(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> group) {
  SmallVector<int64_t> out;
  for (int64_t d : group)
    if (shape[d] != 1)
      out.push_back(d);
  return out;
}

/// `tensor.collapse_shape`: RESULT coords -> SRC coords.
///
/// The reassociation has one group per RESULT dim, listing the SRC dims it
/// merged, so each result dim's index has to be DELINEARIZED back over the src
/// extents of its group. Three cases per group, and only the first two give a
/// map a loop IV can be projected through:
///
///   no non-unit dim   the group collapses to an extent-1 dim, so every src dim
///                     of it -- including the one the collapsed index addressed
///                     -- can only be read at 0.
///   one non-unit dim  that dim carries the collapsed index unchanged, every
///                     other dim of the group reads at 0. This is the
///                     `tt.broadcast` shape, `[64,1] -> [64]` giving
///                     `(d0) -> (d0, 0)`.
///   two or more       a genuine linearization, and the map says so:
///                     `[64,2] -> [128]` gives
///                     `(d0) -> (d0 floordiv 2, d0 mod 2)`. Emitted, not
///                     refused -- the projection check on the composed map is
///                     what decides, and composition can fold this away.
CoordinateRestatement collapseRestatement(tensor::CollapseShapeOp op) {
  MLIRContext *ctx = op.getContext();
  ArrayRef<int64_t> srcShape = op.getSrcType().getShape();
  unsigned srcRank = srcShape.size();
  unsigned resultRank = op.getResultType().getRank();
  // Bound to a local: getReassociationIndices returns by value, and enumerating
  // the temporary directly is a lifetime question nobody should have to ask.
  auto groups = op.getReassociationIndices();

  SmallVector<AffineExpr> results(srcRank);
  std::string why;
  for (auto [g, group] : llvm::enumerate(groups)) {
    AffineExpr collapsed = getAffineDimExpr(g, ctx);
    SmallVector<int64_t> nonUnit = nonUnitDims(srcShape, group);
    if (nonUnit.size() <= 1) {
      for (int64_t d : group)
        results[d] = (!nonUnit.empty() && d == nonUnit.front())
                         ? collapsed
                         : getAffineConstantExpr(0, ctx);
      continue;
    }
    // Linearizing. Every dim of the group must have a static extent for the
    // strides to exist at all; a unit dim is static by definition, so this only
    // asks about the non-unit ones.
    if (llvm::any_of(nonUnit, [&](int64_t d) {
          return ShapedType::isDynamic(srcShape[d]);
        }))
      return noMap(op.getSrc(),
                   "reassociation group " + std::to_string(g) +
                       " fuses a dynamic extent, whose "
                       "delinearization strides are not stateable");
    if (why.empty())
      why = "reassociation group " + std::to_string(g) + " fuses " +
            std::to_string(nonUnit.size()) +
            " non-unit dims, which only a linearizing map can state";
    for (auto [idx, d] : llvm::enumerate(group)) {
      if (srcShape[d] == 1) {
        results[d] = getAffineConstantExpr(0, ctx);
        continue;
      }
      int64_t stride =
          productOf(srcShape, ArrayRef<int64_t>(group).drop_front(idx + 1));
      AffineExpr e = stride == 1 ? collapsed : collapsed.floorDiv(stride);
      // The outermost non-unit dim needs no modulus: the collapsed index is
      // already bounded by the product of the group's extents.
      if (d != nonUnit.front())
        e = e % srcShape[d];
      results[d] = e;
    }
  }

  // A src dim in no group would leave its coordinate unstated. The op's own
  // verifier rules this out; be explicit rather than building a short map.
  if (llvm::any_of(results, [](AffineExpr e) { return !e; }))
    return noMap(op.getSrc(),
                 "its reassociation leaves a source dim unnamed, leaving no "
                 "map over all of them");

  return CoordinateRestatement{
      op.getSrc(), AffineMap::get(resultRank, /*symbolCount=*/0, results, ctx),
      why};
}

/// `tensor.expand_shape`: RESULT coords -> SRC coords.
///
/// The mirror image of the collapse, and the asymmetry is in the reassociation
/// rather than in the reasoning: here one group per SRC dim lists the RESULT
/// dims it was split into, so each src coordinate is a LINEARIZATION of its
/// group's result coordinates rather than a delinearization of one.
///
///   no non-unit result dim   the src dim has extent 1; its coordinate is 0.
///   one non-unit result dim  that dim carries the src coordinate and the
///                            inserted unit dims drop out. This is the
///                            `tt.expand_dims` shape, `[4] -> [4,1]` giving
///                            `(d0, d1) -> (d0)`.
///   two or more              a genuine linearization, `d_i * stride + d_j`,
///                            emitted and then declined by the projection
///                            check.
CoordinateRestatement expandRestatement(tensor::ExpandShapeOp op) {
  MLIRContext *ctx = op.getContext();
  ArrayRef<int64_t> resultShape = op.getResultType().getShape();
  unsigned resultRank = resultShape.size();
  auto groups = op.getReassociationIndices();

  SmallVector<AffineExpr> results;
  std::string why;
  for (auto [s, group] : llvm::enumerate(groups)) {
    SmallVector<int64_t> nonUnit = nonUnitDims(resultShape, group);
    if (nonUnit.empty()) {
      results.push_back(getAffineConstantExpr(0, ctx));
      continue;
    }
    if (nonUnit.size() == 1) {
      results.push_back(getAffineDimExpr(nonUnit.front(), ctx));
      continue;
    }
    if (llvm::any_of(nonUnit, [&](int64_t d) {
          return ShapedType::isDynamic(resultShape[d]);
        }))
      return noMap(op.getSrc(), "reassociation group " + std::to_string(s) +
                                    " expands into a dynamic extent, whose "
                                    "linearization strides are not stateable");
    if (why.empty())
      why = "reassociation group " + std::to_string(s) + " expands into " +
            std::to_string(nonUnit.size()) +
            " non-unit dims, which only a linearizing map can state";
    AffineExpr e = getAffineConstantExpr(0, ctx);
    for (auto [idx, d] : llvm::enumerate(group)) {
      if (resultShape[d] == 1)
        continue;
      int64_t stride =
          productOf(resultShape, ArrayRef<int64_t>(group).drop_front(idx + 1));
      AffineExpr term = getAffineDimExpr(d, ctx);
      e = e + (stride == 1 ? term : term * stride);
    }
    results.push_back(e);
  }

  return CoordinateRestatement{
      op.getSrc(), AffineMap::get(resultRank, /*symbolCount=*/0, results, ctx),
      why};
}

/// The map from `shapeOp`'s RESULT coordinates to its SOURCE coordinates, if
/// `shapeOp` is a coordinate restatement at all.
///
/// This function is the pass's whole inventory of what a coordinate restatement
/// IS, which is what makes both the absorption and the rejection follow from
/// one place. Teaching it a new op makes that op absorbable and stops it being
/// refused, with nothing else to edit.
///
/// The five tensor shape ops LowerComputeOps' Group A can emit:
///
///   A2  tt.reshape      tensor.reshape       no map: the shape is a tensor
///                                            OPERAND, so there is no static
///                                            structure to read one off
///   A3  tt.expand_dims  tensor.expand_shape  yes, expandRestatement
///   A4  tt.broadcast    collapse_shape +     yes, collapseRestatement
///                       linalg.broadcast
///   A6  tt.join         expand_shape x2 +    no map: a concat selects between
///                       tensor.concat        operands per coordinate
///   A7  tt.split        extract_slice x2 +   the collapse yes; the
///                       collapse_shape x2    extract_slice deliberately NOT a
///                                            restatement -- see below
///
/// WHY `tensor.extract_slice` IS NOT HERE. Its coordinate map is derivable --
/// static offsets and unit strides give `(d0, d1) -> (d0 + o0, d1 + o1)` -- but
/// its EXTENT change is not, and a linalg operand map cannot state one: linalg
/// infers its loop bounds from the operand shapes THROUGH the indexing maps, so
/// repointing an operand at the larger, unsliced source under any map makes the
/// inferred extents inconsistent and fails linalg's own verifier. A slice
/// crops; an operand map re-indexes.
///
/// The omission costs nothing. The offset part would be declined by the
/// projection check anyway (`d0 + o0` is neither a bare dim nor a constant), so
/// a sound version of this case would absorb only a zero-offset slice, which
/// the canonicalizer already removes. And the scheduler looks through the op
/// itself: `findIndexingMapForLoadResult` tries the direct operand and then one
/// alternative, an extract_slice user. So it is left as "not a coordinate
/// restatement" -- neither absorbed nor refused.
CoordinateRestatement resultToSourceMap(Operation *shapeOp) {
  CoordinateRestatement r = [&]() -> CoordinateRestatement {
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(shapeOp))
      return collapseRestatement(collapse);
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(shapeOp))
      return expandRestatement(expand);
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(shapeOp))
      return noMap(reshape.getSource(),
                   "no affine map from its result coordinates to its source "
                   "coordinates is derivable -- its shape is an operand rather "
                   "than a reassociation");
    if (auto concat = dyn_cast<tensor::ConcatOp>(shapeOp))
      return noMap(concat.getInputs().front(),
                   "it selects between operands per coordinate, which one "
                   "operand map cannot state");
    return notReindex();
  }();

  // One op, and the answer the rest of the file is derived from. A
  // non-restatement says nothing: every load, empty, constant and cast the
  // gate's walk touches goes through here.
  LLVM_DEBUG(if (r.isReindex()) {
    llvm::dbgs() << "[" DEBUG_TYPE "] restatement of " << shapeOp->getName()
                 << " at " << shapeOp->getLoc() << ": ";
    if (r.map)
      llvm::dbgs() << "result->source " << r.map;
    else
      llvm::dbgs() << "no map";
    if (!r.why.empty())
      llvm::dbgs() << " [not projectable: " << r.why << "]";
    llvm::dbgs() << "\n";
  });
  return r;
}

/// The diagnostic. `why` comes from `resultToSourceMap`, so the message is
/// derived from the same answer the absorption was.
void rejectUnabsorbable(Operation *op, StringRef why) {
  op->emitError(
      "fuse-generics: this op re-indexes a value on a path "
      "the "
      "layout pass will physicalize and it cannot be restated as an indexing "
      "map on its consumer (")
      << why
      << "), so the layout pass would bridge it with a linearizing map, which "
         "the scheduler cannot project loop IVs through";
}

/// Refuse every coordinate restatement that sits on a path RewriteDescriptorLayoutGeneric will physicalize. Silent when the module
/// has no annotated view at all, which is the gate.
LogicalResult rejectOnPhysicalizedPaths(ModuleOp mod) {
  // The seed: the layout pass's own roots.
  SmallVector<Operation *> annotatedViews;
  mod.walk([&](mlir::ktdp::ConstructMemoryViewOp view) {
    if (view->hasAttr(triton::tts::TTSDialect::kTensorLayoutAttrName))
      annotatedViews.push_back(view.getOperation());
  });
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] gate: " << annotatedViews.size()
                          << " tts.tensor_layout-annotated memory view(s)\n");
  if (annotatedViews.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] gate closed: nothing in this module is "
                  "physicalized, so nothing is refused\n");
    return success();
  }

  // The scope, from the shared traversals rather than from a copy of them.
  SmallVector<Operation *> adjacent;
  triton::ktdp::collectAdjacentGenerics(annotatedViews, adjacent);
  SmallPtrSet<Operation *, 8> inRewriteSet(adjacent.begin(), adjacent.end());

  SmallVector<Operation *> physLoads, physStores;
  for (Operation *view : annotatedViews)
    triton::ktdp::collectViewAccesses(view, physLoads, physStores);

  LLVM_DEBUG({
    llvm::dbgs() << "[" DEBUG_TYPE "] gate open: " << physLoads.size()
                 << " physicalized load(s), " << physStores.size()
                 << " physicalized store(s), " << adjacent.size()
                 << " generic(s) in the rewrite set\n";
    for (Operation *g : adjacent)
      llvm::dbgs() << "  rewrite set: generic at " << g->getLoc() << "\n";
  });

  LogicalResult result = success();
  SmallPtrSet<Operation *, 16> asked;
  // How each op was reached, so a rejection can print the path rather than
  // leaving the reader to re-derive why the op was on a physicalized path at
  // all. Maps an op to the op whose operand led the walk to it.
  llvm::DenseMap<Operation *, Operation *> cameFrom;

  // `why` empty means the restatement IS a projected coordinate map. Such an op
  // is still here only because it was never offered to the absorber -- nothing
  // between it and the memory access is a linalg.generic.
  // RewriteDescriptorLayout Generic names that case itself, and more precisely
  // than this could: its checkConsumersAreRewritable for the ins side, its
  // checkStoreDataIsRestatable for the store side. So say nothing.
  auto ask = [&](Operation *op, const CoordinateRestatement &restatement) {
    if (restatement.why.empty() || !asked.insert(op).second)
      return;
    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "] REJECT " << op->getName() << " at "
                   << op->getLoc() << "; reached by";
      // Capped: the chain follows reverse def-use edges and so cannot cycle,
      // but a debug print is not the place to bet the process on that.
      unsigned hops = 0;
      for (Operation *step = cameFrom.lookup(op); step && hops < 16;
           step = cameFrom.lookup(step), ++hops)
        llvm::dbgs() << " <- " << step->getName() << " at " << step->getLoc();
      llvm::dbgs() << "\n";
    });
    rejectUnabsorbable(op, restatement.why);
    result = failure();
  };

  // BACKWARD, from every value a generic the layout pass will rewrite reads,
  // and from the data of every store over an annotated view.
  //
  // The two seeds are the two sides of one hazard, and neither is optional:
  //   the generic's `ins` -- findLayoutForInput takes a ktdp.load and nothing
  //   else, so an operand behind a shape op has no layout and the layout pass
  //   refuses it as needing a linearizing map; the store's data --
  //   findLayoutForResult walks the generic's result users for a ktdp.store
  //   DIRECTLY (`dyn_cast<StoreOp>`, `continue` otherwise), so a shape op
  //   between the two makes it return null and the LINEARIZATION it would need
  //   is on the `outs` map instead. `gather__1d` carries an 8x1 -> 8 tt.reshape into a
  //   store, which is that shape in the tree -- unannotated, hence the gate.
  //
  // Nothing here absorbs on the store side; rejection only, per the header.
  SmallPtrSet<Operation *, 16> visited;
  SmallVector<std::pair<Value, Operation *>> worklist;
  auto seed = [&](Value v, Operation *user) { worklist.emplace_back(v, user); };
  for (Operation *g : adjacent)
    for (Value in : cast<linalg::GenericOp>(g).getDpsInputs())
      seed(in, g);
  for (Operation *st : physStores)
    seed(cast<mlir::ktdp::StoreOp>(st).getDataTile(), st);

  while (!worklist.empty()) {
    auto [v, user] = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    cameFrom.try_emplace(def, user);
    if (isa<linalg::GenericOp>(def)) {
      // A generic already in the rewrite set is its own entry point above; one
      // that is not is walked through, because the absorber is not gated on
      // physicalization and could have folded something above it into it.
      if (!inRewriteSet.contains(def))
        for (Value in : cast<linalg::GenericOp>(def).getDpsInputs())
          seed(in, def);
      continue;
    }
    CoordinateRestatement restatement = resultToSourceMap(def);
    if (!restatement.isReindex())
      continue; // a load, an empty, a constant, a fill, an scf result, a cast
    ask(def, restatement);
    seed(restatement.source, def);
  }

  // FORWARD, one hop off each load the layout pass will retype -- for a
  // `load -> reshape -> store` chain with no generic in it anywhere, which the
  // backward walk from the store's data also reaches, and a `load -> reshape`
  // whose result goes somewhere else entirely, which it does not.
  for (Operation *load : physLoads)
    for (Operation *user : load->getResult(0).getUsers()) {
      if (isa<linalg::GenericOp, mlir::ktdp::StoreOp>(user))
        continue;
      cameFrom.try_emplace(user, load);
      ask(user, resultToSourceMap(user));
    }

  return result;
}

struct FuseGenericsPass
    : public mlir::triton::spyre::impl::FuseGenericsBase<FuseGenericsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<AbsorbForwardingGeneric, GroupComputes>(ctx);
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();

    LogicalResult booleans = checkNoBooleanBetweenGenerics(mod);
    LogicalResult gate = rejectOnPhysicalizedPaths(mod);
    if (failed(booleans) || failed(gate))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createFuseGenericsPass() {
  return std::make_unique<FuseGenericsPass>();
}

} // namespace mlir::triton::spyre
