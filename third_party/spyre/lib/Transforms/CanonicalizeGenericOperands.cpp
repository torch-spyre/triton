//===- CanonicalizeGenericOperands.cpp - Generic operands in generic terms ===//
//
// States two kinds of linalg.generic operand in the generic's own terms, so
// that FuseGenerics, after it, only meets generics as producers. One generic in,
// one generic out, or a reshape replaced by a generic: nothing is fused here.
//
//   ReshapeToGeneric       a static tensor.collapse_shape / tensor.expand_shape
//                          whose every user is a generic `ins` becomes one
//                          forwarding generic, its input map the reshape's
//                          result -> source map
//   SplatOperandToScalar   a splat dense constant `ins` becomes a scalar
//                          arith.constant read in the body
//
// HOW TO ADD A PATTERN. A pattern is rooted on, and matches, one kind of
// PRODUCER: the op that defines the operand. Never the operand's type, nor
// anything downstream of the generic. A new pattern names a producer kind no
// existing pattern matches; one whose kind overlaps an existing pattern's is a
// new case of that pattern. Its lit file feeds it the other patterns'
// producers and checks that it does not fire.
//
// `--debug-only=canonicalize-generic-operands` traces each rewrite and each
// reshape left alone, with the reason.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"
#include "Utils/GenericOperands.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "canonicalize-generic-operands"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_CANONICALIZEGENERICOPERANDS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

//===----------------------------------------------------------------------===//
// ReshapeToGeneric
//===----------------------------------------------------------------------===//

/// The dims of `group` whose extent in `shape` is not 1.
SmallVector<int64_t> nonUnitDims(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> group) {
  SmallVector<int64_t> out;
  for (int64_t d : group)
    if (shape[d] != 1)
      out.push_back(d);
  return out;
}

/// The extents of `shape` at `dims`, multiplied. Every entry is static.
int64_t productOf(ArrayRef<int64_t> shape, ArrayRef<int64_t> dims) {
  int64_t p = 1;
  for (int64_t d : dims)
    p *= shape[d];
  return p;
}

/// collapse_shape, result coords -> source coords. One group per result dim,
/// listing the source dims it merged. A group with at most one non-unit dim
/// carries the collapsed index on that dim and 0 elsewhere; a group with
/// several delinearizes it with floordiv and mod, e.g. `[64,2] -> [128]` gives
/// `(d0) -> (d0 floordiv 2, d0 mod 2)`. A source dim in no group (a rank-0
/// result) reads 0. Every extent is static.
AffineMap collapseResultToSource(tensor::CollapseShapeOp op) {
  MLIRContext *ctx = op.getContext();
  ArrayRef<int64_t> srcShape = op.getSrcType().getShape();
  unsigned resultRank = op.getResultType().getRank();
  auto groups = op.getReassociationIndices();

  SmallVector<AffineExpr> results(srcShape.size(),
                                  getAffineConstantExpr(0, ctx));
  for (auto [g, group] : llvm::enumerate(groups)) {
    AffineExpr collapsed = getAffineDimExpr(g, ctx);
    SmallVector<int64_t> nonUnit = nonUnitDims(srcShape, group);
    if (nonUnit.size() <= 1) {
      if (!nonUnit.empty())
        results[nonUnit.front()] = collapsed;
      continue;
    }
    for (auto [idx, d] : llvm::enumerate(group)) {
      if (srcShape[d] == 1)
        continue;
      int64_t stride =
          productOf(srcShape, ArrayRef<int64_t>(group).drop_front(idx + 1));
      AffineExpr e = stride == 1 ? collapsed : collapsed.floorDiv(stride);
      // The collapsed index is bounded by the group's product, so the
      // outermost non-unit dim needs no modulus.
      if (d != nonUnit.front())
        e = e % srcShape[d];
      results[d] = e;
    }
  }
  return AffineMap::get(resultRank, /*symbolCount=*/0, results, ctx);
}

/// expand_shape, result coords -> source coords. One group per source dim,
/// listing the result dims it was split into. Each source dim reads its one
/// non-unit result dim, 0 when there is none, or a mul-add linearization of
/// several. Every extent is static.
AffineMap expandResultToSource(tensor::ExpandShapeOp op) {
  MLIRContext *ctx = op.getContext();
  ArrayRef<int64_t> resultShape = op.getResultType().getShape();
  auto groups = op.getReassociationIndices();

  SmallVector<AffineExpr> results;
  for (ArrayRef<int64_t> group : groups) {
    SmallVector<int64_t> nonUnit = nonUnitDims(resultShape, group);
    if (nonUnit.size() <= 1) {
      results.push_back(nonUnit.empty()
                            ? getAffineConstantExpr(0, ctx)
                            : getAffineDimExpr(nonUnit.front(), ctx));
      continue;
    }
    AffineExpr e = getAffineConstantExpr(0, ctx);
    for (auto [idx, d] : llvm::enumerate(group)) {
      if (resultShape[d] == 1)
        continue;
      int64_t stride = productOf(resultShape, group.drop_front(idx + 1));
      AffineExpr term = getAffineDimExpr(d, ctx);
      e = e + (stride == 1 ? term : term * stride);
    }
    results.push_back(e);
  }
  return AffineMap::get(resultShape.size(), /*symbolCount=*/0, results, ctx);
}

/// A static tensor.collapse_shape or tensor.expand_shape whose every user is an
/// `ins` operand of a linalg.generic -> one linalg.generic over the result's
/// shape, all-parallel, reading the source under the result -> source map and
/// yielding it. Left alone: a reshape with a dynamic extent, or with any other
/// user, e.g. a ktdp.store.
template <typename ReshapeOp>
struct ReshapeToGeneric : OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto leave = [&](const char *why) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] ReshapeToGeneric: left "
                              << op->getName() << " at " << op->getLoc()
                              << ": " << why << "\n");
      return failure();
    };
    RankedTensorType srcType = op.getSrcType();
    RankedTensorType resultType = op.getResultType();
    if (!srcType.hasStaticShape() || !resultType.hasStaticShape())
      return leave("a dynamic extent");
    if (op->use_empty())
      return failure();
    for (OpOperand &use : op->getUses()) {
      auto generic = dyn_cast<linalg::GenericOp>(use.getOwner());
      if (!generic || !generic.isDpsInput(&use))
        return leave("a user is not a linalg.generic ins");
    }

    AffineMap resultToSource;
    if constexpr (std::is_same_v<ReshapeOp, tensor::CollapseShapeOp>)
      resultToSource = collapseResultToSource(op);
    else
      resultToSource = expandResultToSource(op);
    unsigned rank = resultType.getRank();
    SmallVector<AffineMap> maps = {
        resultToSource,
        AffineMap::getMultiDimIdentityMap(rank, op.getContext())};
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Location loc = op.getLoc();
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType());
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{op.getSrc()}, ValueRange{init},
        maps, iterators, [](OpBuilder &b, Location l, ValueRange args) {
          linalg::YieldOp::create(b, l, args[0]);
        });
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] ReshapeToGeneric: "
                            << op->getName() << " at " << loc
                            << " -> forwarding generic, input map "
                            << resultToSource << "\n");
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SplatOperandToScalar
//===----------------------------------------------------------------------===//

/// A linalg.generic `ins` defined by a splat int-or-float dense
/// arith.constant -> a scalar arith.constant read in the body, through
/// foldSplatConstantOperand. Declines an operand whose removal would leave
/// some loop dim named by no map.
struct SplatOperandToScalar : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();
    for (OpOperand *operand : op.getDpsInputOperands()) {
      unsigned number = operand->getOperandNumber();
      if (succeeded(
              triton::spyre::foldSplatConstantOperand(rewriter, op, operand))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] SplatOperandToScalar: operand "
                   << number << " of generic at " << op.getLoc()
                   << " -> a scalar in the body\n");
        return success();
      }
    }
    return failure();
  }
};

struct CanonicalizeGenericOperandsPass
    : public mlir::triton::spyre::impl::CanonicalizeGenericOperandsBase<
          CanonicalizeGenericOperandsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ReshapeToGeneric<tensor::CollapseShapeOp>,
                 ReshapeToGeneric<tensor::ExpandShapeOp>, SplatOperandToScalar>(
        ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>>
createCanonicalizeGenericOperandsPass() {
  return std::make_unique<CanonicalizeGenericOperandsPass>();
}

} // namespace mlir::triton::spyre
