//===- ContractionSynthesis.cpp - Phase 2 contraction dispatch/emission ---===//
//
// Dispatches physicalized matmul / reduce / store ops to their stick-tiled
// emission routines, slicing operands according to per-operand OperandPlans
// derived from layout markers.
//
// Uses MLIR's greedy pattern rewrite driver instead of a manual fixpoint loop.
//
//===----------------------------------------------------------------------===//

#include "RewriteDescriptorLayout/ContractionSynthesis.h"
#include "RewriteDescriptorLayout/Classify.h"
#include "RewriteDescriptorLayout/PermutationUtils.h"
#include "RewriteDescriptorLayout/Types.h"

#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "rewrite-descriptor-layout"

using namespace mlir;

namespace {

using namespace mlir::triton::ktdp;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

// True iff op is a single-result elementwise op with exactly one
// RankedTensor operand.
bool isSingleTensorElementwiseOp(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumOperands() == 0)
    return false;
  int tensorOps = 0;
  for (auto operand : op->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      ++tensorOps;
  return tensorOps == 1;
}

// Emit linalg.transpose with the given permutation (input->output form).
// linalg.transpose uses "output<-input" form, so we invert here.
Value emitTranspose(OpBuilder &b, Location loc, Value src,
                    llvm::ArrayRef<int64_t> perm) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto mlirPerm = invertPerm(perm);
  llvm::SmallVector<int64_t> outShape(mlirPerm.size());
  for (unsigned i = 0; i < mlirPerm.size(); ++i)
    outShape[i] = srcTy.getDimSize(mlirPerm[i]);
  auto outTy = RankedTensorType::get(outShape, srcTy.getElementType());
  Value empty = tensor::EmptyOp::create(b, loc, outTy.getShape(),
                                        srcTy.getElementType());
  return linalg::TransposeOp::create(b, loc, src, empty,
      b.getDenseI64ArrayAttr(mlirPerm)).getResult()[0];
}

// Walk backward from `val` through single-tensor elementwise ops to the
// ktdp.load that produced it. Returns null if not found.
mlir::ktdp::LoadOp walkToLoad(Value val) {
  Value v = val;
  while (true) {
    auto *defOp = v.getDefiningOp();
    if (!defOp)
      return mlir::ktdp::LoadOp{};
    if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(defOp))
      return ld;
    if (!isSingleTensorElementwiseOp(defOp))
      return mlir::ktdp::LoadOp{};
    for (auto operand : defOp->getOperands())
      if (isa<RankedTensorType>(operand.getType())) { v = operand; break; }
  }
}

// Look up the marker for an access tile's base memView in the pass context.
triton::SpyreTensorLayoutOp
lookupMarkerFromTile(Value accessTile, const PassContext &ctx) {
  auto tileOp = accessTile.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
  if (!tileOp)
    return {};
  auto it = ctx.physMemViewToMarker.find(tileOp.getBase());
  return it != ctx.physMemViewToMarker.end() ? it->second
                                             : triton::SpyreTensorLayoutOp{};
}

// Walk back from an operand through the elementwise chain to the
// ktdp.load, then look up the physical memView -> marker map.
triton::SpyreTensorLayoutOp
findMarkerForOperand(Value operand, const PassContext &ctx) {
  auto ld = walkToLoad(operand);
  if (!ld)
    return {};
  return lookupMarkerFromTile(ld.getAccessTile(), ctx);
}

// Emit a stick loop (scf.for) or inline for trip <= 1.
Value emitStickLoop(OpBuilder &b, Location loc, int64_t tripCount,
                    Value acc,
                    llvm::function_ref<Value(OpBuilder &, Value, Value)> body) {
  if (tripCount <= 1) {
    Value s0 = arith::ConstantIndexOp::create(b, loc, 0);
    return body(b, s0, acc);
  }
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value ub = arith::ConstantIndexOp::create(b, loc, tripCount);
  auto forOp = scf::ForOp::create(b, loc, c0, ub, c1, ValueRange{acc});
  OpBuilder ib = OpBuilder::atBlockBegin(forOp.getBody());
  Value stepped =
      body(ib, forOp.getInductionVar(), forOp.getRegionIterArgs()[0]);
  scf::YieldOp::create(ib, loc, ValueRange{stepped});
  b.setInsertionPointAfter(forOp);
  return forOp.getResult(0);
}

// Outer parallel-scatter loop: iterates parallelIV = 0..tripCount, filling
// disjoint slabs of `container` via tensor.insert_slice returned by `body`.
// Inlines for tripCount <= 1 (passes a const-0 IV and returns the body result
// directly), so the extent-1 case emits exactly what the plain reduction path
// would.
Value emitParallelScatterLoop(
    OpBuilder &b, Location loc, int64_t tripCount, Value container,
    llvm::function_ref<Value(OpBuilder &, Value, Value)> body) {
  if (tripCount <= 1) {
    Value s0 = arith::ConstantIndexOp::create(b, loc, 0);
    return body(b, s0, container);
  }
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value ub = arith::ConstantIndexOp::create(b, loc, tripCount);
  auto forOp = scf::ForOp::create(b, loc, c0, ub, c1, ValueRange{container});
  OpBuilder ib = OpBuilder::atBlockBegin(forOp.getBody());
  Value updated =
      body(ib, forOp.getInductionVar(), forOp.getRegionIterArgs()[0]);
  scf::YieldOp::create(ib, loc, ValueRange{updated});
  b.setInsertionPointAfter(forOp);
  return forOp.getResult(0);
}

// Extract an op-tile stick slice from `plan`.
Value extractOpSlice(OpBuilder &b, Location loc,
                     const OperandPlan &plan,
                     RankedTensorType resultTy, Value stickIV,
                     Value parallelIV = nullptr) {
  auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };
  llvm::ArrayRef<int64_t> physBlock = plan.coords.physBlock;
  int rank = (int)physBlock.size();
  llvm::SmallVector<OpFoldResult> offsets(rank), sizes(rank), strides(rank, idx(1));
  for (int p = 0; p < rank; ++p) {
    switch (plan.dims.sliceKind[p]) {
    case SliceKind::StickIndex: {
      Value selectedIV = (plan.dimRoles[p] >= 0 && parallelIV) ? parallelIV : stickIV;
      Value iv = (physBlock[p] > 1) ? selectedIV : Value{};
      if (!iv) {
        offsets[p] = idx(0);
      } else if (iv.getType().isIndex()) {
        offsets[p] = iv;
      } else {
        offsets[p] = arith::IndexCastOp::create(b, loc,
                         b.getIndexType(), iv).getResult();
      }
      sizes[p] = idx(1);
      break;
    }
    case SliceKind::StickifiedBlock: {
      Value sIdx = stickIV.getType().isIndex()
                       ? stickIV
                       : arith::IndexCastOp::create(b, loc,
                             b.getIndexType(), stickIV).getResult();
      Value stickSz = arith::ConstantIndexOp::create(b, loc, plan.dims.stickSize);
      offsets[p] = arith::MulIOp::create(b, loc, sIdx, stickSz).getResult();
      sizes[p]   = idx(plan.dims.stickSize);
      break;
    }
    case SliceKind::WholeBlock:
      offsets[p] = idx(0);
      sizes[p]   = idx(physBlock[p]);
      break;
    }
  }
  return tensor::ExtractSliceOp::create(
      b, loc, resultTy, plan.value, offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// Source stage (matmul / reduce operands)
//===----------------------------------------------------------------------===//

// Source stage emission: extract slices, optional transpose, call emitOp.
// Returns the replacement value for the original op's result.
Value emitSourceStage(
    linalg::LinalgOp op,
    OpBuilder &b,
    llvm::function_ref<Value(OpBuilder &, Location, llvm::ArrayRef<Value>, Value,
                             RankedTensorType)>
        emitOp,
    llvm::ArrayRef<OperandPlan> plans,
    const OperandSetTripCounts &tripCounts) {
  Location loc = op.getLoc();

  Value cVal = op.getDpsInits()[0];
  auto accElemTy = cast<RankedTensorType>(cVal.getType()).getElementType();

  // Per-operand op-tile slice types.
  llvm::SmallVector<RankedTensorType> sliceTys;
  for (unsigned i = 0; i < plans.size(); ++i) {
    const OperandPlan &plan = plans[i];
    auto elemTy = cast<RankedTensorType>(plan.value.getType()).getElementType();
    sliceTys.push_back(RankedTensorType::get(plan.opExtents, elemTy));
  }

  // Derive acc shape from the union of all (outputAxis, extent) pairs.
  int64_t maxAxis = -1;
  for (auto &plan : plans)
    for (unsigned j = 0; j < plan.dims.opTileDims.size(); ++j) {
      int p = plan.dims.opTileDims[j];
      int64_t role = plan.dimRoles[p];
      if (role >= 0 && role > maxAxis)
        maxAxis = role;
    }
  llvm::SmallVector<int64_t> accDims(maxAxis + 1, 0);
  for (auto &plan : plans)
    for (unsigned j = 0; j < plan.dims.opTileDims.size(); ++j) {
      int p = plan.dims.opTileDims[j];
      int64_t role = plan.dimRoles[p];
      if (role >= 0)
        accDims[role] = plan.opExtents[j];
    }
  auto accTy = RankedTensorType::get(accDims, accElemTy);

  // Transpose helper (delegates to free function).
  auto doTranspose = [&](Value src, llvm::ArrayRef<int64_t> perm) -> Value {
    return emitTranspose(b, loc, src, perm);
  };

  // Trip counts are derived cross-operand by reconcileOperandSet(); a
  // disagreement there is signaled via tripCounts.parallelAgrees, which
  // dispatchSource checks before calling this function.
  int64_t stickFactor = tripCounts.stickFactor;
  int64_t parallelFactor = tripCounts.parallelFactor;
  int64_t parallelAccAxis = tripCounts.parallelAccAxis;
  int64_t parallelStickSize = tripCounts.parallelStickSize;

  // Emit the reduction stick loop for one parallel slab. `acc` is the
  // per-slab accumulator (shape `accTy`); `pIV` selects the parallel stick.
  Value stickIV;
  auto emitReductionLoop = [&](OpBuilder &bb, Value acc, Value pIV) -> Value {
    return emitStickLoop(bb, loc, stickFactor, acc,
        [&](OpBuilder &bbb, Value s, Value innerAcc) {
      stickIV = s;
      OpBuilder saved = b;
      b = bbb;
      llvm::SmallVector<Value> slices;
      for (unsigned i = 0; i < plans.size(); ++i) {
        Value slicePhys =
            extractOpSlice(b, loc, plans[i], sliceTys[i], stickIV, pIV);
        slices.push_back(!plans[i].transposePerm.empty()
                             ? doTranspose(slicePhys, plans[i].transposePerm)
                             : slicePhys);
      }
      Value r = emitOp(b, loc, slices, innerAcc, accTy);
      b = saved;
      return r;
    });
  };

  if (parallelFactor <= 1) {
    // Reduction-only path: no parallel scatter, accumulate straight into cVal.
    // extractOpSlice gets a null parallelIV, so every StickIndex dim uses the
    // reduction IV exactly as before.
    return emitReductionLoop(b, cVal, /*pIV=*/Value{});
  }

  // Parallel-scatter path. emitSourceStage must always return a canonical
  // LOGICAL value (the sink stage physicalizes it), so scatter into `cVal`
  // itself: it already has the full logical output shape and carries the
  // incoming accumulator. Each iteration extracts the pIV'th slab (shape
  // accTy) at offset pIV*stickSize on `parallelAccAxis`, runs the reduction
  // loop over it, and inserts the result back.
  llvm::SmallVector<OpFoldResult> slabSizes;
  for (int64_t d : accDims)
    slabSizes.push_back(b.getIndexAttr(d));
  llvm::SmallVector<OpFoldResult> slabStrides(accDims.size(),
                                              b.getIndexAttr(1));

  return emitParallelScatterLoop(b, loc, parallelFactor, cVal,
      [&](OpBuilder &bb, Value pIV, Value container) -> Value {
        Value stickSizeV =
            arith::ConstantIndexOp::create(bb, loc, parallelStickSize);
        Value off = arith::MulIOp::create(bb, loc, pIV, stickSizeV);
        llvm::SmallVector<OpFoldResult> slabOffsets(accDims.size(),
                                                    bb.getIndexAttr(0));
        slabOffsets[parallelAccAxis] = off;

        Value slab = tensor::ExtractSliceOp::create(
            bb, loc, accTy, container, slabOffsets, slabSizes, slabStrides);
        Value computed = emitReductionLoop(bb, slab, pIV);
        return tensor::InsertSliceOp::create(bb, loc, computed, container,
                                             slabOffsets, slabSizes,
                                             slabStrides);
      });
}

// Helper: emit an error and set the fatal error flag.
static LogicalResult emitFatalError(Operation *op, const PassContext &ctx,
                                    const llvm::Twine &msg) {
  ctx.hadError = true;
  return op->emitError(msg);
}

// Classify one operand and populate plans[i].
LogicalResult dispatchSource(linalg::LinalgOp op, const SourceOpSpec &spec,
                             const PassContext &ctx, PatternRewriter &rewriter) {
  unsigned nOps = spec.operands.size();
  llvm::SmallVector<OperandPlan, 2> plans(nOps);

  for (unsigned i = 0; i < nOps; ++i) {
    Value operand = op.getDpsInputs()[i];
    auto ld = walkToLoad(operand);

    if (ld) {
      auto marker = findMarkerForOperand(operand, ctx);
      if (!marker) {
        auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorTy ||
            tensorTy.getRank() != (int64_t)spec.operands[i].canonicalAxes.size())
          return emitFatalError(op, ctx,
              "spyre_tensor_layout: physical operand load has no layout marker");
        plans[i] = classifyScratchpad(operand, spec.operands[i]);
        continue;
      }
      auto physShape = cast<RankedTensorType>(operand.getType()).getShape();
      OperandCoords coords = OperandCoords::fromMarker(marker, spec.logicalRank,
                                                       physShape);
      // Compose erased transpose perm into canonicalAxes.
      llvm::SmallVector<int64_t> effectiveCanonicalAxes = spec.operands[i].canonicalAxes;
      {
        auto it = ctx.physicalLoadToTransposePerm.find(ld.getResult());
        if (it != ctx.physicalLoadToTransposePerm.end()) {
          const auto &tau = it->second;
          assert(tau.size() == effectiveCanonicalAxes.size() &&
                 "transpose perm size must match canonicalAxes size");
          llvm::SmallVector<int64_t> reordered(effectiveCanonicalAxes.size());
          for (unsigned j = 0; j < tau.size(); ++j)
            reordered[j] = effectiveCanonicalAxes[tau[j]];
          effectiveCanonicalAxes = std::move(reordered);
        }
      }
      llvm::SmallVector<int64_t> dimRoles;
      buildDimRoles(coords, effectiveCanonicalAxes, dimRoles);
      plans[i] = classify(operand, coords, dimRoles);
      LLVM_DEBUG({
        llvm::dbgs() << "  operand " << i << ": opTileDims=[";
        llvm::interleaveComma(plans[i].dims.opTileDims, llvm::dbgs());
        llvm::dbgs() << "] loopDims=[";
        llvm::interleaveComma(plans[i].dims.loopDims, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });
    } else {
      auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorTy ||
          tensorTy.getRank() != (int64_t)spec.operands[i].canonicalAxes.size())
        return emitFatalError(op, ctx,
            "spyre_tensor_layout: source op operand is neither a physical "
            "load nor a logical (scratchpad) tensor of the expected rank");
      plans[i] = classifyScratchpad(operand, spec.operands[i]);
    }
  }

  OperandSetTripCounts tripCounts = reconcileOperandSet(plans);
  for (unsigned i = 0; i < nOps; ++i)
    resolveOperand(plans[i], spec.operands[i].targetOrder,
                   TransposeDirection::Narrow);

  // Reject a scratchpad operand paired with a multi-stick reduction axis.
  for (unsigned i = 0; i < nOps; ++i) {
    if (plans[i].coords.src.empty())
      continue;
    bool multiStickReduction = false;
    for (int p : plans[i].dims.loopDims)
      if (plans[i].coords.physBlock[p] > 1) { multiStickReduction = true; break; }
    if (!multiStickReduction)
      continue;
    for (unsigned j = 0; j < nOps; ++j) {
      if (i == j) continue;
      if (plans[j].coords.src.empty())
        return emitFatalError(op, ctx,
            "spyre_tensor_layout: operands share a stickified contraction "
            "axis but not all are annotated — any two operands sharing a "
            "stickified contraction axis must both carry a "
            "tt.spyre_tensor_layout marker with the same stick size on that "
            "axis");
    }
  }

  if (!tripCounts.parallelAgrees)
    return emitFatalError(op, ctx,
        "spyre_tensor_layout: operands disagree on the parallel multi-stick "
        "scatter — two annotated operands carry parallel floor dims with "
        "different trip counts or on different output axes, which would need "
        "two independent scatter loops (not supported)");

  OpBuilder b(op.getOperation());
  Value result = emitSourceStage(op, b, spec.emitOp, plans, tripCounts);
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// Sink stage (store scatter)
//===----------------------------------------------------------------------===//

// Walk the forward use chain from value through elementwise ops to a
// ktdp.store, then look up the store's access tile base in physMemViewToMarker.
triton::SpyreTensorLayoutOp findMarkerForStore(Value value,
                                               const PassContext &ctx) {
  llvm::SmallVector<Value> worklist = {value};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (auto *user : v.getUsers()) {
      if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        auto marker = lookupMarkerFromTile(st.getAccessTile(), ctx);
        if (marker)
          return marker;
      }
      if (!isSingleTensorElementwiseOp(user))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return {};
}

// Sink stage: scatter a logical data tile into the physical D tensor shape.
// Preconditions (non-empty floorDims, empty loopDims) are checked by the
// caller before `dPlan` is even built for resolution; see
// RewriteStorePattern.
LogicalResult emitSinkStage(mlir::ktdp::StoreOp st,
                            const OperandPlan &dPlan,
                            const PassContext &ctx,
                            PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "  sink stage: stickSize=" << dPlan.dims.stickSize
                          << ", floorDims=" << dPlan.dims.floorDims.size() << "\n");
  Value inputTile = st.getDataTile();
  OpBuilder b(st.getOperation());
  Location loc = st.getLoc();

  Type elemTy = cast<RankedTensorType>(inputTile.getType()).getElementType();

  llvm::ArrayRef<int64_t> physBlock = dPlan.coords.physBlock;
  int physRank = (int)physBlock.size();
  int64_t stickSize = physBlock[dPlan.dims.lane];

  unsigned logRank = dPlan.coords.logicalRank;
  llvm::ArrayRef<int64_t> sinkPerm = dPlan.transposePerm;

  auto logDimToPos = [&](int64_t d) -> unsigned {
    return sinkPerm.empty() ? (unsigned)d : (unsigned)sinkPerm[d];
  };

  if (!sinkPerm.empty())
    inputTile = emitTranspose(b, loc, inputTile, sinkPerm);

  auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

  llvm::SmallVector<int64_t> sinkShape(physBlock.begin(), physBlock.end());
  Value physicalSink = tensor::EmptyOp::create(b, loc, sinkShape, elemTy);

  llvm::SmallVector<OpFoldResult> inputOffsetsBase(logRank, idx(0));
  llvm::SmallVector<OpFoldResult> inputSizesBase(logRank);
  llvm::SmallVector<OpFoldResult> inputStrides(logRank, idx(1));
  for (int p : dPlan.dims.opTileDims) {
    int64_t logDim = dPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      inputSizesBase[logDimToPos(logDim)] = idx(physBlock[p]);
  }
  for (int p : dPlan.dims.floorDims) {
    int64_t logDim = dPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      inputSizesBase[logDimToPos(logDim)] = idx(stickSize);
  }

  llvm::SmallVector<OpFoldResult> sinkOffsetsBase(physRank, idx(0));
  llvm::SmallVector<OpFoldResult> sinkSizes(physRank);
  llvm::SmallVector<OpFoldResult> sinkStrides(physRank, idx(1));
  for (int p = 0; p < physRank; ++p)
    sinkSizes[p] = llvm::is_contained(dPlan.dims.floorDims, p)
                       ? idx(1) : idx(physBlock[p]);

  Value acc = physicalSink;
  for (int p : dPlan.dims.floorDims) {
    int64_t logDim = dPlan.dimRoles[p];
    if (logDim < 0 || (unsigned)logDim >= logRank) continue;

    unsigned tileDim = logDimToPos(logDim);
    int64_t tripCount = physBlock[p];
    Value stickSizeVal = arith::ConstantIndexOp::create(b, loc, stickSize);

    llvm::SmallVector<int64_t> slShape(logRank);
    for (int p2 : dPlan.dims.opTileDims) {
      int64_t ld = dPlan.dimRoles[p2];
      if (ld >= 0 && (unsigned)ld < logRank)
        slShape[logDimToPos(ld)] = physBlock[p2];
    }
    slShape[tileDim] = stickSize;
    auto slTy = RankedTensorType::get(slShape, elemTy);

    acc = emitStickLoop(b, loc, tripCount, acc,
        [&](OpBuilder &bb, Value s, Value sinkAccumulator) -> Value {
          llvm::SmallVector<OpFoldResult> inOff = inputOffsetsBase;
          inOff[tileDim] =
              arith::MulIOp::create(bb, loc, s, stickSizeVal).getResult();
          Value inputSlice = tensor::ExtractSliceOp::create(
              bb, loc, slTy, inputTile, inOff, inputSizesBase, inputStrides);

          llvm::SmallVector<OpFoldResult> sinkOff = sinkOffsetsBase;
          sinkOff[p] = s;
          return tensor::InsertSliceOp::create(
              bb, loc, inputSlice, sinkAccumulator, sinkOff, sinkSizes, sinkStrides);
        });
  }

  // Mutate the store's data tile operand in-place.
  rewriter.modifyOpInPlace(st, [&]() {
    st.getDataTileMutable().set(acc);
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Shared matmul-like pattern helper
//===----------------------------------------------------------------------===//

// Shared implementation for matmul-like contractions (matmul, batch_matmul).
// Parameterized by the op type, logical rank, canonical axes, and op emitter.
template <typename OpTy>
static LogicalResult rewriteMatmulLike(
    OpTy op, PatternRewriter &rewriter, const PassContext &ctx,
    unsigned logicalRank,
    llvm::ArrayRef<SourceOperandSpec> operandSpecs,
    llvm::function_ref<Value(OpBuilder &, Location, llvm::ArrayRef<Value>, Value,
                             RankedTensorType)>
        emitOp) {
  // Only match when at least one input is physicalized.
  bool needsDispatch = llvm::any_of(
      cast<linalg::LinalgOp>(op.getOperation()).getDpsInputOperands(),
      [&](OpOperand *operand) {
        auto t = dyn_cast<RankedTensorType>(operand->get().getType());
        if (!t || t.getRank() <= (int)logicalRank)
          return false;
        return static_cast<bool>(findMarkerForOperand(operand->get(), ctx));
      });
  if (!needsDispatch)
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "  dispatching " << OpTy::getOperationName()
                          << " at " << op.getLoc() << "\n");

  SourceOpSpec spec;
  spec.operands.assign(operandSpecs.begin(), operandSpecs.end());
  spec.logicalRank = logicalRank;
  spec.emitOp = emitOp;
  return dispatchSource(op, spec, ctx, rewriter);
}

//===----------------------------------------------------------------------===//
// Pattern: linalg.matmul
//===----------------------------------------------------------------------===//

struct RewriteMatmulPattern : OpRewritePattern<linalg::MatmulOp> {
  const PassContext &ctx;
  RewriteMatmulPattern(MLIRContext *mlirCtx, const PassContext &layoutCtx)
      : OpRewritePattern(mlirCtx, /*benefit=*/2), ctx(layoutCtx) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp mm,
                                PatternRewriter &rewriter) const override {
    return rewriteMatmulLike<linalg::MatmulOp>(
        mm, rewriter, ctx, /*logicalRank=*/2,
        // canonicalAxes and targetOrder coincide: K sits at its operand
        // position in the emitted linalg.matmul (trailing in A, leading in B).
        {SourceOperandSpec{{0, -1}, {0, -1}},   // A=(m,k)
         SourceOperandSpec{{-1, 1}, {-1, 1}}},  // B=(k,n)
        [](OpBuilder &b, Location loc, llvm::ArrayRef<Value> slices, Value acc,
           RankedTensorType accTy) -> Value {
          return linalg::MatmulOp::create(b, loc, accTy,
              ValueRange{slices[0], slices[1]}, ValueRange{acc}).getResult(0);
        });
  }
};

//===----------------------------------------------------------------------===//
// Pattern: linalg.batch_matmul
//===----------------------------------------------------------------------===//

struct RewriteBatchMatmulPattern : OpRewritePattern<linalg::BatchMatmulOp> {
  const PassContext &ctx;
  RewriteBatchMatmulPattern(MLIRContext *mlirCtx, const PassContext &layoutCtx)
      : OpRewritePattern(mlirCtx, /*benefit=*/2), ctx(layoutCtx) {}

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp bmm,
                                PatternRewriter &rewriter) const override {
    return rewriteMatmulLike<linalg::BatchMatmulOp>(
        bmm, rewriter, ctx, /*logicalRank=*/3,
        {SourceOperandSpec{{0, 1, -1}, {0, 1, -1}},   // A=(b,m,k)
         SourceOperandSpec{{0, -1, 2}, {0, -1, 2}}},  // B=(b,k,n)
        [](OpBuilder &b, Location loc, llvm::ArrayRef<Value> slices, Value acc,
           RankedTensorType accTy) -> Value {
          return linalg::BatchMatmulOp::create(b, loc, accTy,
              ValueRange{slices[0], slices[1]}, ValueRange{acc}).getResult(0);
        });
  }
};

//===----------------------------------------------------------------------===//
// Pattern: linalg.reduce
//===----------------------------------------------------------------------===//

struct RewriteReducePattern : OpRewritePattern<linalg::ReduceOp> {
  const PassContext &ctx;
  RewriteReducePattern(MLIRContext *mlirCtx, const PassContext &layoutCtx)
      : OpRewritePattern(mlirCtx, /*benefit=*/2), ctx(layoutCtx) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp rd,
                                PatternRewriter &rewriter) const override {
    auto rdMarker = findMarkerForOperand(rd.getInputs()[0], ctx);
    unsigned logicalInputRank = 2;
    if (rdMarker) {
      for (int64_t src : rdMarker.getPhysSrc())
        if ((unsigned)(src + 1) > logicalInputRank)
          logicalInputRank = (unsigned)(src + 1);
    }

    bool needsDispatch = llvm::any_of(
        cast<linalg::LinalgOp>(rd.getOperation()).getDpsInputOperands(),
        [&](OpOperand *operand) {
          auto t = dyn_cast<RankedTensorType>(operand->get().getType());
          if (!t || t.getRank() <= (int)logicalInputRank)
            return false;
          return static_cast<bool>(findMarkerForOperand(operand->get(), ctx));
        });
    if (!needsDispatch)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "  dispatching reduce at " << rd.getLoc() << "\n");

    auto marker = findMarkerForOperand(rd.getInputs()[0], ctx);
    if (!marker)
      return emitFatalError(rd, ctx,
          "spyre_tensor_layout: dispatchReduce called but no marker on input");
    unsigned logicalRank = 0;
    for (int64_t src : marker.getPhysSrc())
      if ((unsigned)(src + 1) > logicalRank)
        logicalRank = (unsigned)(src + 1);

    llvm::SmallVector<int64_t> canonicalAxes(logicalRank, -1);
    unsigned outAxis = 0;
    for (unsigned d = 0; d < logicalRank; ++d)
      if (!llvm::is_contained(rd.getDimensions(), (int64_t)d))
        canonicalAxes[d] = outAxis++;

    // emitReduceOp always reduces the trailing axes (dims = [outputRank,
    // rank)), so the surviving roles must come first, in order, and the
    // reduced slots last — regardless of where the reduced axis sits
    // logically.
    llvm::SmallVector<int64_t> targetOrder;
    for (unsigned r = 0; r < outAxis; ++r)
      targetOrder.push_back((int64_t)r);
    targetOrder.append(logicalRank - outAxis, -1);

    Block &combinerBlock = rd.getOperation()->getRegion(0).front();
    llvm::SmallVector<Operation *> combinerOps;
    for (Operation &op : combinerBlock.without_terminator())
      combinerOps.push_back(&op);
    auto combinerYield = cast<linalg::YieldOp>(combinerBlock.getTerminator());
    llvm::SmallVector<Value> yieldVals(combinerYield.getValues().begin(),
                                       combinerYield.getValues().end());
    llvm::SmallVector<Value> origBlockArgs(combinerBlock.getArguments().begin(),
                                           combinerBlock.getArguments().end());

    unsigned outputRank = logicalRank - (unsigned)rd.getDimensions().size();
    auto emitReduceOp = [outputRank,
                         combinerOps = std::move(combinerOps),
                         yieldVals = std::move(yieldVals),
                         origBlockArgs = std::move(origBlockArgs)](
                            OpBuilder &b, Location loc,
                            llvm::ArrayRef<Value> slices, Value acc,
                            RankedTensorType accTy) -> Value {
      auto sliceTy = cast<RankedTensorType>(slices[0].getType());
      llvm::SmallVector<int64_t> dims;
      for (unsigned d = outputRank; d < (unsigned)sliceTy.getRank(); ++d)
        dims.push_back((int64_t)d);
      return linalg::ReduceOp::create(
          b, loc, ValueRange{slices[0]}, ValueRange{acc}, dims,
          [&](OpBuilder &inner, Location iloc, ValueRange args) {
            IRMapping mapping;
            for (unsigned i = 0; i < origBlockArgs.size(); ++i)
              mapping.map(origBlockArgs[i], args[i]);
            for (Operation *op : combinerOps)
              inner.clone(*op, mapping);
            llvm::SmallVector<Value> mapped;
            for (Value v : yieldVals)
              mapped.push_back(mapping.lookupOrDefault(v));
            linalg::YieldOp::create(inner, iloc, mapped);
          }).getResult(0);
    };
    SourceOpSpec spec;
    spec.operands = {SourceOperandSpec{canonicalAxes, targetOrder}};
    spec.logicalRank = logicalRank;
    spec.emitOp = emitReduceOp;
    return dispatchSource(rd, spec, ctx, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Pattern: ktdp.store (sink)
//===----------------------------------------------------------------------===//

struct RewriteStorePattern : OpRewritePattern<mlir::ktdp::StoreOp> {
  const PassContext &ctx;
  RewriteStorePattern(MLIRContext *mlirCtx, const PassContext &layoutCtx)
      : OpRewritePattern(mlirCtx, /*benefit=*/1), ctx(layoutCtx) {}

  LogicalResult matchAndRewrite(mlir::ktdp::StoreOp st,
                                PatternRewriter &rewriter) const override {
    auto dataTy = dyn_cast<RankedTensorType>(st.getDataTile().getType());
    auto tileTy = dyn_cast<mlir::ktdp::AccessTileType>(
        st.getAccessTile().getType());
    if (!dataTy || !tileTy ||
        dataTy.getRank() == (int)tileTy.getShape().size())
      return failure();
    auto marker = findMarkerForStore(st.getDataTile(), ctx);
    if (!marker)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "  dispatching store sink at " << st.getLoc() << "\n");

    llvm::ArrayRef<int64_t> physBlock = tileTy.getShape();
    unsigned logRank = dataTy.getRank();
    OperandCoords dC = OperandCoords::fromMarker(marker, logRank, physBlock);

    int physRank = (int)physBlock.size();
    llvm::SmallVector<int64_t> dimRoleD(physRank);
    for (int p = 0; p < physRank; ++p)
      dimRoleD[p] = marker.getPhysSrc()[p];

    OperandPlan dPlan = classify(st.getDataTile(), dC, dimRoleD);

    // The store's two preconditions, expressed as resolution outcomes: an
    // empty floorDims means nothing to scatter, and a non-empty loopDims
    // means a reduction the store cannot express. Checked here rather than
    // inside the emitter. Do not call reconcileOperandSet on this path: it
    // is a no-op for a store (dimRoles are phys_src, always >= 0, so
    // loopDims is always empty already) and would assert a cross-operand
    // dependency that does not exist for a single data tile.
    if (dPlan.dims.floorDims.empty()) {
      ctx.hadError = true;
      return st.emitError(
          "spyre_tensor_layout: store sink stage requires at least one "
          "parallel floor dim in the output layout");
    }
    if (!dPlan.dims.loopDims.empty()) {
      ctx.hadError = true;
      return st.emitError(
          "spyre_tensor_layout: store sink stage: unexpected reduction dim");
    }

    // A store contracts nothing, so its target order is the identity — the
    // logical dim order itself — and the inverse direction (op-tile ->
    // physical) is what scatters into the physical D tensor shape.
    llvm::SmallVector<int64_t> targetOrderD(logRank);
    std::iota(targetOrderD.begin(), targetOrderD.end(), 0);
    resolveOperand(dPlan, targetOrderD, TransposeDirection::Widen);

    return emitSinkStage(st, dPlan, ctx, rewriter);
  }
};

} // namespace

namespace mlir::triton::ktdp {

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

void populateContractionPatterns(RewritePatternSet &patterns,
                                 const PassContext &ctx) {
  MLIRContext *mlirCtx = patterns.getContext();
  patterns.add<RewriteMatmulPattern, RewriteBatchMatmulPattern,
               RewriteReducePattern>(mlirCtx, ctx);
  patterns.add<RewriteStorePattern>(mlirCtx, ctx);
}

} // namespace mlir::triton::ktdp
