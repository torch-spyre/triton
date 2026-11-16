//===- LowerComputeOps.cpp - Lower tt compute ops to linalg/tensor --------===//
//
// Lowers Triton compute ops to linalg and tensor dialect.
// Patterns are organized into groups:
//
//   Group A — Shape manipulation (no compute, 1:1 structural rewrites)
//   Group B — Reduction (combiner region, identity element)
//   Group C — Matrix multiply
//   Group D — Spyre intrinsic requests (tts.spyre_op -> spyreop.* directly)
//
// Trivially dead ops left by prior passes are swept at the end of this pass.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToKTIR/Passes.h"
#include "ConversionUtils.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "Utils/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERCOMPUTEOPS
#include "Conversion/TritonToKTIR/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

using mlir::triton::spyre::cleanupDeadOps;
using mlir::triton::spyre::createEmptyTensor;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the identity attribute for a reduction combiner op.
/// Returns std::nullopt if the combiner op is not recognised.
static std::optional<TypedAttr>
getReductionNeutralAttr(Operation *combinerOp, Type elemType,
                        MLIRContext *ctx) {
  Builder b(ctx);
  if (isa<arith::MaxNumFOp>(combinerOp)) {
    auto ftype = cast<FloatType>(elemType);
    return TypedAttr(b.getFloatAttr(
        ftype, APFloat::getInf(ftype.getFloatSemantics(), /*neg=*/true)));
  }
  if (isa<arith::MinNumFOp>(combinerOp)) {
    auto ftype = cast<FloatType>(elemType);
    return TypedAttr(b.getFloatAttr(
        ftype, APFloat::getInf(ftype.getFloatSemantics(), /*neg=*/false)));
  }
  // arith.select has no algebraic neutral element, but is used as the index
  // lane in multi-operand reductions (e.g. argmax). The value lane's -inf
  // neutral guarantees every real element replaces the init, so the index
  // init is never the final answer. Use -1 as an obvious invalid-index
  // sentinel so incorrect results are detectable rather than silently 0.
  if (isa<arith::SelectOp>(combinerOp)) {
    if (auto itype = dyn_cast<IntegerType>(elemType))
      return TypedAttr(b.getIntegerAttr(itype, -1));
  }
  return arith::getNeutralElement(combinerOp);
}

//===----------------------------------------------------------------------===//
// Group A — Shape manipulation (pure tensor restructuring, no compute)
//
// Straightforward 1:1 structural rewrites.  No combiner region, no init
// value.  Each op targets a different output op so a shared template
// wouldn't reduce code (same as Triton's ViewOpToLLVM.cpp).
//
//   A1. tt.splat      → linalg.fill        scalar → tensor (fill all elems)
//   A2. tt.reshape    → tensor.reshape      same elems, new shape
//   A3. tt.expand_dims→ tensor.expand_shape insert size-1 dim
//   A4. tt.broadcast  → tensor.collapse_shape + linalg.broadcast
//                                            expand size-1 dims to match
//
//   A5. tt.trans      → linalg.transpose    permute dimensions
//   A6. tt.join       → tensor.expand_shape + tensor.concat
//                                            two tensors → new minor dim
//   A7. tt.split      → tensor.extract_slice × 2
//                                            last dim=2 → two tensors
//===----------------------------------------------------------------------===//

/// A1. tt.splat → linalg.fill
///   scalar 42.0  →  tensor<4x8xf32> filled with 42.0
///
/// Alternative: tt.splat → tensor.splat (single op, simpler).
/// linalg.fill is preferred when downstream passes fuse linalg ops.
struct ConvertTTSplat : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value empty = createEmptyTensor(rewriter, op.getLoc(), resultType);
    auto fill = linalg::FillOp::create(rewriter, op.getLoc(),
                                       adaptor.getSrc(), empty);
    rewriter.replaceOp(op, fill.getResult(0));
    return success();
  }
};

/// A2. tt.reshape → tensor.reshape
///   tensor<512xf32>  →  tensor<16x32xf32>  (same 512 elements)
struct ConvertTTReshape : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    SmallVector<Value> dims;
    for (int64_t d : resultType.getShape())
      dims.push_back(arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(d)));

    auto shapeTensor = tensor::FromElementsOp::create(rewriter, loc, dims);
    auto reshape = tensor::ReshapeOp::create(
        rewriter, loc, resultType, adaptor.getSrc(), shapeTensor);
    rewriter.replaceOp(op, reshape.getResult());
    return success();
  }
};

/// A3. tt.expand_dims → tensor.expand_shape
///   tensor<8xf32>  →  tensor<8x1xf32>  (insert size-1 dim at axis)
struct ConvertTTExpandDims
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getSrc();
    int32_t axis = op.getAxis();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();

    // Each input dim maps to one output dim, except the dim adjacent to
    // the inserted axis which maps to two (itself + the new size-1 dim).
    // axis == i:         input dim i → output dims [i, i+1]
    // axis == inputRank: input dim (inputRank-1) → output dims [inputRank-1, inputRank]
    SmallVector<ReassociationIndices> reassociation;
    for (int64_t i = 0; i < inputRank; ++i) {
      if (i == axis) {
        reassociation.push_back({i, i + 1});
      } else if (axis == inputRank && i == inputRank - 1) {
        reassociation.push_back({i, i + 1});
      } else {
        int64_t outIdx = (i < axis) ? i : i + 1;
        reassociation.push_back({outIdx});
      }
    }

    auto expandOp = tensor::ExpandShapeOp::create(
        rewriter, loc, resultType, input, reassociation);
    rewriter.replaceOp(op, expandOp.getResult());
    return success();
  }
};

/// A4. tt.broadcast → tensor.collapse_shape + linalg.broadcast
///   tensor<1x8xf32>  →  tensor<4x8xf32>  (expand size-1 dims)
struct ConvertTTBroadcast
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getSrc();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    SmallVector<int64_t> broadcastDims;
    SmallVector<int64_t> keptDims;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (inputType.getShape()[i] == 1 && resultType.getShape()[i] != 1)
        broadcastDims.push_back(i);
      else
        keptDims.push_back(i);
    }

    if (broadcastDims.empty()) {
      rewriter.replaceOp(op, input);
      return success();
    }

    SmallVector<int64_t> collapsedShape;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (!llvm::is_contained(broadcastDims, i))
        collapsedShape.push_back(inputType.getShape()[i]);
    }

    auto collapsedType = RankedTensorType::get(
        collapsedShape, inputType.getElementType());

    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices currentGroup;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      currentGroup.push_back(i);
      if (!llvm::is_contained(broadcastDims, i)) {
        reassociation.push_back(currentGroup);
        currentGroup.clear();
      }
    }
    if (!currentGroup.empty() && !reassociation.empty()) {
      for (auto idx : currentGroup)
        reassociation.back().push_back(idx);
    }

    Value collapsed = tensor::CollapseShapeOp::create(
        rewriter, loc, collapsedType, input, reassociation);

    auto emptyOp = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType());

    auto broadcastOp = linalg::BroadcastOp::create(
        rewriter, loc, collapsed, emptyOp.getResult(), broadcastDims);
    rewriter.replaceOp(op, broadcastOp->getResults());
    return success();
  }
};

/// A5. tt.trans → linalg.transpose
///   tensor<4x8xf32>  →  tensor<8x4xf32>  (permute dims via order attr)
struct ConvertTTTrans : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value empty = createEmptyTensor(rewriter, loc, resultType);

    SmallVector<int64_t> perm(op.getOrder().begin(), op.getOrder().end());
    auto transpose = linalg::TransposeOp::create(
        rewriter, loc, adaptor.getSrc(), empty, perm);
    rewriter.replaceOp(op, transpose->getResults());
    return success();
  }
};

/// A6. tt.join → tensor.expand_shape + tensor.concat
///   tensor<4x8xf32>, tensor<4x8xf32>  →  tensor<4x8x2xf32>
///   Result[..., 0] = lhs, Result[..., 1] = rhs.
struct ConvertTTJoin : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    auto inputType = cast<RankedTensorType>(adaptor.getLhs().getType());
    int64_t inputRank = inputType.getRank();

    // Expand lhs and rhs: <4x8xf32> → <4x8x1xf32>
    SmallVector<int64_t> expandedShape(inputType.getShape());
    expandedShape.push_back(1);
    auto expandedType = RankedTensorType::get(
        expandedShape, inputType.getElementType());

    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < inputRank - 1; ++i)
      reassoc.push_back({i});
    reassoc.push_back({inputRank - 1, inputRank});

    Value lhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, expandedType, adaptor.getLhs(), reassoc);
    Value rhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, expandedType, adaptor.getRhs(), reassoc);

    // Concatenate along the new last dim: <4x8x1> ++ <4x8x1> → <4x8x2>
    auto concat = tensor::ConcatOp::create(
        rewriter, loc, rank - 1,
        ValueRange{lhsExpanded, rhsExpanded});
    rewriter.replaceOp(op, concat.getResult());
    return success();
  }
};

/// A7. tt.split → tensor.extract_slice × 2
///   tensor<4x8x2xf32>  →  tensor<4x8xf32>, tensor<4x8xf32>
///   Extracts src[..., 0] and src[..., 1], then collapses the last dim.
struct ConvertTTSplit : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = adaptor.getSrc();
    auto srcType = cast<RankedTensorType>(src.getType());
    int64_t rank = srcType.getRank();

    // Extract slice for index 0 and 1 along the last dim.
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank - 1; ++i)
      sizes.push_back(rewriter.getIndexAttr(srcType.getShape()[i]));
    sizes.push_back(rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    // <4x8x1xelemtype>
    SmallVector<int64_t> sliceShape(srcType.getShape().drop_back());
    sliceShape.push_back(1);
    auto sliceType = RankedTensorType::get(
        sliceShape, srcType.getElementType());

    Value lhsSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, src, offsets, sizes, strides);

    offsets.back() = rewriter.getIndexAttr(1);
    Value rhsSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, src, offsets, sizes, strides);

    // Collapse <4x8x1> → <4x8>
    auto outType = cast<RankedTensorType>(op.getOutLHS().getType());
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0; i < rank - 2; ++i)
      reassoc.push_back({i});
    reassoc.push_back({rank - 2, rank - 1});

    Value lhs = tensor::CollapseShapeOp::create(
        rewriter, loc, outType, lhsSlice, reassoc);
    Value rhs = tensor::CollapseShapeOp::create(
        rewriter, loc, outType, rhsSlice, reassoc);

    rewriter.replaceOp(op, {lhs, rhs});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Group B — Reduction (compute along one axis, produces smaller tensor)
//
// These require extracting the combiner region, computing the identity
// element, and emitting the linalg op with the cloned combiner body.
//
//   B1. tt.reduce → linalg.reduce   reduce along axis with combiner
//
// Planned:
//   B2. tt.scan   → scf.for / linalg  prefix scan along axis
//===----------------------------------------------------------------------===//

/// B1. tt.reduce → linalg.reduce
///   tensor<4x8xf32> →reduce(axis=1)→ tensor<4xf32>  (e.g. row-wise sum)
///
/// Multi-operand (e.g. argmax): all N input tensors are reduced together
/// in a single linalg.reduce.  Block args in both tt.reduce and linalg.reduce
/// follow the same order: [lhs0..lhsN-1, rhs0..rhsN-1], so the combiner
/// region can be cloned with a direct 1:1 arg mapping.
struct ConvertTTReduce : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    int32_t axis = op.getAxis();
    unsigned numOperands = adaptor.getOperands().size();

    // Compute the result shape (same for all operands — they share input rank).
    auto firstInputType =
        cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    SmallVector<int64_t> resultShape;
    for (int64_t i = 0; i < firstInputType.getRank(); ++i) {
      if (i != axis)
        resultShape.push_back(firstInputType.getShape()[i]);
    }

    // Build one (empty, fill) init tensor per operand.
    // Neutral element: use the combiner op that produces the i-th yield value.
    Block &combinerBlock = op.getCombineOp().front();
    Operation *terminator = combinerBlock.getTerminator();

    SmallVector<Value> inputs(adaptor.getOperands().begin(),
                              adaptor.getOperands().end());
    SmallVector<Value> inits;
    for (unsigned i = 0; i < numOperands; ++i) {
      auto inputType = cast<RankedTensorType>(inputs[i].getType());
      Type elemType = inputType.getElementType();

      Value termYield = terminator->getOperand(i);
      Operation *combinerOp = termYield.getDefiningOp();
      auto neutralAttr = getReductionNeutralAttr(combinerOp, elemType, ctx);
      if (!neutralAttr)
        return failure();
      TypedAttr attr = *neutralAttr;
      Value fillVal = arith::ConstantOp::create(rewriter, loc, attr);
      auto emptyOp =
          tensor::EmptyOp::create(rewriter, loc, resultShape, elemType);
      inits.push_back(
          linalg::FillOp::create(rewriter, loc, fillVal, emptyOp.getResult())
              .getResult(0));
    }

    // Snapshot the combiner block contents before creating the reduce op.
    SmallVector<Operation *> combinerOps;
    for (auto &innerOp : combinerBlock.without_terminator())
      combinerOps.push_back(&innerOp);

    // Collect all yield values for the linalg.reduce body.
    SmallVector<Value> yieldOperands(terminator->operand_begin(),
                                     terminator->operand_end());

    SmallVector<int64_t> dimensions = {axis};
    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, inputs, inits, dimensions,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // tt.reduce and linalg.reduce share block arg order:
          //   [lhs0..lhsN-1, rhs0..rhsN-1]
          // Direct 1:1 mapping is correct for any N.
          IRMapping mapping;
          mapping.map(combinerBlock.getArguments(), args);
          for (auto *innerOp : combinerOps)
            b.clone(*innerOp, mapping);
          SmallVector<Value> mapped;
          for (Value v : yieldOperands)
            mapped.push_back(mapping.lookup(v));
          linalg::YieldOp::create(b, loc, mapped);
        });

    // If a result is a bare scalar (rank-0 reduce), extract from the
    // rank-0 tensor that linalg.reduce produced.
    SmallVector<Value> results;
    for (unsigned i = 0; i < numOperands; ++i) {
      Value r = reduceOp->getResult(i);
      if (resultShape.empty())
        r = tensor::ExtractOp::create(rewriter, loc, r, ValueRange{});
      results.push_back(r);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Group C — Matrix multiply
//
//   C1. tt.dot → linalg.matmul   d = matmul(a, b) + c
//===----------------------------------------------------------------------===//

/// C1. tt.dot → linalg.matmul / linalg.batch_matmul
///   2-D: tensor<M×K> @ tensor<K×N> + tensor<M×N> → tensor<M×N>
///   3-D: tensor<B×M×K> @ tensor<B×K×N> + tensor<B×M×N> → tensor<B×M×N>
struct ConvertTTDot : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    if (aType.getRank() == 3) {
      // Batched matmul: B×M×K @ B×K×N + B×M×N → B×M×N
      auto batchMatmul = linalg::BatchMatmulOp::create(
          rewriter, op.getLoc(),
          ValueRange{adaptor.getA(), adaptor.getB()},
          ValueRange{adaptor.getC()});
      rewriter.replaceOp(op, batchMatmul->getResults());
    } else {
      // Standard 2-D matmul: M×K @ K×N + M×N → M×N
      auto matmul = linalg::MatmulOp::create(
          rewriter, op.getLoc(),
          ValueRange{adaptor.getA(), adaptor.getB()},
          ValueRange{adaptor.getC()});
      rewriter.replaceOp(op, matmul->getResults());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Group D — Spyre intrinsic requests
//
// `tts.spyre_op` names one real `spyreop.*` op by a STRING attribute, not by
// C++ type -- its verifier (`verifySpyreIntrinsicCall`, Dialect.h/Ops.cpp)
// has already confirmed, by the time this pass runs, that the name is
// registered and that the call's operands/attrs/result types are exactly
// what that real op accepts. So building it here is "construct the op the
// verifier already checked," not a second independent check -- the same
// `RegisteredOperationName::lookup` + `OperationState` + `builder.create`
// idiom the verifier itself uses to build its throwaway probe op, except
// this one is kept rather than erased, and its operands are this pattern's
// real (scalar, block-argument) values rather than a scratch block's.
//
//   D1. tts.spyre_op (elementwise) → linalg.generic { spyreop.<name> }
//       Every intrinsic in scope except `exx2` — same shape in and out,
//       1-5 operands, 1 result (SameOperandsAndResultType).
//   D2. tts.spyre_op ("exx2")      → linalg.generic { spyreop.exx2 }
//       Reduction-shaped: modeled on ConvertTTReduce's own
//       (tensor.empty-init, reduction-iterator) skeleton, but with the real
//       op as the whole body applied to the UN-reduced input, not a
//       per-step combiner — see spyreop-frontend-api.md's own worked
//       example. Its init is bare `tensor.empty`, deliberately NOT
//       `linalg.fill`-initialized the way ConvertTTReduce's is: `exx2`'s
//       body never reads its accumulator block arguments at all, so there
//       is no uninitialized-read hazard, and a `linalg.fill`-built init is
//       exactly what DropReductionInitFill's own one-input/one-init gate
//       (DropReductionInitFill.cpp) would silently fail to remove for a
//       2-result op like this one, leaving a `linalg.fill` for the device
//       scheduler's named-op allowlist to then reject.
//===----------------------------------------------------------------------===//

/// Builds the real `spyreop.<name>` op inside the current insertion point,
/// applied to `operands` (already scalar-typed — the caller's block
/// arguments), and returns its results. `resultTypes`/`attrs` are exactly
/// what `tts::SpyreOpOp`'s own `extra_attrs` carried; `axis` is NOT among
/// them (see TTSOps.td) and is never forwarded here, matching
/// `verifySpyreIntrinsicCall`'s own treatment of it as lowering-only
/// metadata.
static Operation *buildSpyreIntrinsic(OpBuilder &builder, Location loc,
                                      StringRef name, ValueRange operands,
                                      TypeRange resultTypes,
                                      DictionaryAttr extraAttrs) {
  MLIRContext *ctx = builder.getContext();
  std::string qualified = ("spyreop." + name).str();
  std::optional<RegisteredOperationName> info =
      RegisteredOperationName::lookup(qualified, ctx);
  assert(info && "tts.spyre_op's own verifier already confirmed this name");

  SmallVector<NamedAttribute> attrList;
  if (extraAttrs)
    attrList.assign(extraAttrs.begin(), extraAttrs.end());
  OperationState state(loc, *info, operands, resultTypes, attrList);
  Operation *op = builder.create(state);
  assert(succeeded(verify(op, /*verifyRecursively=*/false)) &&
        "tts.spyre_op's own verifier already confirmed this call");
  return op;
}

/// D1. tts.spyre_op (elementwise) → linalg.generic { spyreop.<name> }
///   tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> tensor<128xf16>
///   becomes a linalg.generic over identity indexing maps, one scalar block
///   argument per operand, with `spyreop.gelu` as the body.
struct ConvertSpyreOpElementwise
    : public OpConversionPattern<triton::tts::SpyreOpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::tts::SpyreOpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getAxis())
      return failure(); // D2's case.

    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t rank = resultType.getRank();

    SmallVector<Value> inits;
    for (Type resultTy : op.getResultTypes())
      inits.push_back(
          createEmptyTensor(rewriter, loc, cast<RankedTensorType>(resultTy)));

    AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps(
        adaptor.getOperands().size() + inits.size(), identity);
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    StringRef name = op.getName();
    DictionaryAttr extraAttrs = op.getExtraAttrsAttr();
    SmallVector<Type> resultTypes(op.getResultTypes());

    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultTypes, adaptor.getOperands(), inits,
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
          ValueRange scalarOperands =
              blockArgs.take_front(adaptor.getOperands().size());
          Operation *intrinsic = buildSpyreIntrinsic(
              b, loc, name, scalarOperands, resultType.getElementType(),
              extraAttrs);
          linalg::YieldOp::create(b, loc, intrinsic->getResults());
        });
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

/// D2. tts.spyre_op ("exx2") → linalg.generic { spyreop.exx2 }
///   tts.spyre_op {name = "exx2", axis = 1} (%x)
///       : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)
///   becomes a reduction-shaped linalg.generic -- one `reduction` iterator
///   over `axis` -- with `spyreop.exx2` applied once per row of the
///   UN-reduced input as the whole body, not as a per-step combiner. The
///   body's accumulator block arguments are never read, matching the real
///   device kernel this is modeled on (spyreop-frontend-api.md).
struct ConvertSpyreOpReduction
    : public OpConversionPattern<triton::tts::SpyreOpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::tts::SpyreOpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<uint32_t> axis = op.getAxis();
    if (!axis)
      return failure(); // D1's case.

    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    // Bare tensor.empty inits -- see the Group D header comment on why no
    // linalg.fill is built here.
    SmallVector<Value> inits;
    SmallVector<Type> resultTypes(op.getResultTypes());
    for (Type resultTy : resultTypes)
      inits.push_back(
          createEmptyTensor(rewriter, loc, cast<RankedTensorType>(resultTy)));

    AffineMap inputMap = rewriter.getMultiDimIdentityMap(rank);
    AffineMap outputMap = inputMap.dropResult(*axis);
    SmallVector<AffineMap> indexingMaps(1, inputMap);
    indexingMaps.append(inits.size(), outputMap);

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    iteratorTypes[*axis] = utils::IteratorType::reduction;

    StringRef name = op.getName();
    DictionaryAttr extraAttrs = op.getExtraAttrsAttr();

    // One scalar result TYPE per tensor result -- `spyreop.exx2`'s own
    // results are scalar (SameOperandsAndResultType with the scalar
    // operand), not tensors, so this is each result tensor's element type,
    // not the tensor type itself. Same tensor/scalar distinction the
    // verifier's own `verifySpyreIntrinsicCall` draws.
    SmallVector<Type> scalarResultTypes;
    for (Type resultTy : resultTypes)
      scalarResultTypes.push_back(
          cast<RankedTensorType>(resultTy).getElementType());

    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultTypes, ValueRange{input}, inits, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
          // [input, acc0, acc1, ...] -- the accumulators are never read;
          // the real op is applied to the input alone.
          Value scalarInput = blockArgs[0];
          Operation *intrinsic =
              buildSpyreIntrinsic(b, loc, name, ValueRange{scalarInput},
                                  scalarResultTypes, extraAttrs);
          linalg::YieldOp::create(b, loc, intrinsic->getResults());
        });
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerComputeOpsPass
    : public mlir::triton::spyre::impl::LowerComputeOpsBase<
          LowerComputeOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    // Illegal: Triton compute ops that have conversion patterns below.
    //   Group A (shape): splat, reshape, expand_dims, broadcast, trans, join, split
    //   Group B (reduce): reduce + reduce.return (region replaced wholesale)
    //   Group C (matmul): dot
    //   Group D (spyre intrinsics): tts.spyre_op
    target.addIllegalOp<triton::SplatOp, triton::ReshapeOp,
                        triton::ExpandDimsOp, triton::BroadcastOp,
                        triton::TransOp, triton::JoinOp, triton::SplitOp,
                        triton::ReduceOp, triton::ReduceReturnOp,
                        triton::DotOp, triton::tts::SpyreOpOp>();
    // Legal: output dialects that conversion patterns lower into.
    //   linalg (fill, broadcast, transpose, reduce, matmul, generic),
    //   tensor (reshape, expand_shape, collapse_shape, extract_slice, concat, empty),
    //   arith/math (constants, index casts, cloned combiner body ops),
    //   spyreop (the real intrinsic op Group D's generic body calls)
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           arith::ArithDialect, math::MathDialect,
                           spyreop::SpyreOpDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertTTSplat, ConvertTTReshape,          // Group A
                 ConvertTTExpandDims, ConvertTTBroadcast,
                 ConvertTTTrans, ConvertTTJoin, ConvertTTSplit,
                 ConvertTTReduce,                          // Group B
                 ConvertTTDot,                             // Group C
                 ConvertSpyreOpElementwise,                // Group D
                 ConvertSpyreOpReduction>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError("LowerComputeOps: failed to convert compute ops");
      signalPassFailure();
      return;
    }

    cleanupDeadOps(module);
  }
};

} // namespace

namespace mlir::triton::spyre {
std::unique_ptr<OperationPass<ModuleOp>> createLowerComputeOpsPass() {
  return std::make_unique<LowerComputeOpsPass>();
}
} // namespace mlir::triton::spyre
