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

namespace mlir::triton::ktdp {

// True iff op is a single-result elementwise op with exactly one
// RankedTensor operand.
//
// This single-tensor-operand restriction is specific to the BACKWARD walk in
// chainBlockedByPendingTranspose: with two or more tensor operands there is
// no single producer to follow, and the restriction is exactly what keeps
// that walk from crossing a matmul (3 tensor operands) or reduce (2). It is
// not a general elementwise predicate. Forward retyping (deciding an op's
// own result type in Phase 2) must NOT reuse this — see
// RewriteElementwisePattern, which uses a separate, purely local shape rule
// that also covers multi-tensor-operand elementwise ops (arith.addf, select,
// ...).
bool isSingleTensorElementwiseOp(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumOperands() == 0)
    return false;
  int tensorOps = 0;
  for (auto operand : op->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      ++tensorOps;
  return tensorOps == 1;
}

} // namespace mlir::triton::ktdp

namespace {

using namespace mlir::triton::ktdp;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

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

// True if a not-yet-erased linalg.transpose sits between `val` and a
// ktdp.load, i.e. a lookup of `val` in ctx.physicalValues missing is because
// a transpose pattern (RewriteTransposePattern) has not fired here yet,
// rather than `val` genuinely not deriving from a physical load. This is an
// IR-structural question -- "is there a live linalg.TransposeOp on this
// chain" -- not a "is this value already known-physical" question, so it
// cannot be answered by a map lookup: a transpose's own input is not itself
// recorded in ctx.physicalValues until after RewriteTransposePattern erases
// the transpose (see that pattern). Matmul/reduce/store dispatch must
// distinguish these two "not found" cases: the former means "not ready,
// defer to a later greedy iteration" (return failure()), the latter means
// "genuinely logical, treat as a scratchpad operand."
bool chainBlockedByPendingTranspose(Value val) {
  Value v = val;
  while (true) {
    auto *defOp = v.getDefiningOp();
    if (!defOp)
      return false;
    if (isa<mlir::ktdp::LoadOp>(defOp))
      return false;
    if (isa<linalg::TransposeOp>(defOp))
      return true;
    if (!isSingleTensorElementwiseOp(defOp))
      return false;
    Value next;
    for (auto operand : defOp->getOperands())
      if (isa<RankedTensorType>(operand.getType())) { next = operand; break; }
    if (!next)
      return false;
    v = next;
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

// Look up the originating marker for an operand directly from
// ctx.physicalValues: every physical value (a ktdp.load result, or a value
// Phase 2 has already retyped) carries its marker forward in the map, so no
// backward walk is needed.
triton::SpyreTensorLayoutOp
findMarkerForOperand(Value operand, const PassContext &ctx) {
  auto it = ctx.physicalValues.find(operand);
  return it != ctx.physicalValues.end() ? it->second.marker
                                       : triton::SpyreTensorLayoutOp{};
}

// True if `val`'s defining op is an elementwise op RewriteElementwisePattern
// has not yet retyped to its final shape: at least one tensor operand, a
// single tensor result, the operand/result shapes not yet uniform, and (the
// same reachability scoping RewriteElementwisePattern itself requires -- see
// its comment) at least one tensor operand already in ctx.physicalValues. A
// consumer (e.g. RewriteStorePattern) seeing such a value must defer --
// return failure() -- rather than read dataTy.getRank() now, since that rank
// is about to change out from under it once the elementwise op is retyped.
// Without this, a store whose data tile is produced by e.g. arith.addf can
// fire while addf is still logically shaped, misclassify the store as
// needing a sink/bridge stage, and lock in a decision building on a value
// that is not in its final form.
bool pendingElementwiseRetype(Value val, const PassContext &ctx) {
  auto *defOp = val.getDefiningOp();
  if (!defOp || defOp->getNumResults() != 1)
    return false;
  auto resTy = dyn_cast<RankedTensorType>(defOp->getResult(0).getType());
  if (!resTy)
    return false;
  ArrayRef<int64_t> commonShape;
  bool sawTensorOperand = false;
  for (Value o : defOp->getOperands()) {
    auto opndTy = dyn_cast<RankedTensorType>(o.getType());
    if (!opndTy)
      continue;
    if (!sawTensorOperand) {
      commonShape = opndTy.getShape();
      sawTensorOperand = true;
    } else if (opndTy.getShape() != commonShape) {
      // Operands themselves disagree -- not (yet) a case
      // RewriteElementwisePattern will touch; do not report pending.
      return false;
    }
  }
  if (!sawTensorOperand || resTy.getShape() == commonShape)
    return false;
  bool reachesPhysicalLoad = llvm::any_of(defOp->getOperands(), [&](Value o) {
    return ctx.physicalValues.contains(o);
  });
  return reachesPhysicalLoad;
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
    auto physIt = ctx.physicalValues.find(operand);

    if (physIt != ctx.physicalValues.end()) {
      auto marker = physIt->second.marker;
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
        const auto &tau = physIt->second.transposePerm;
        if (!tau.empty()) {
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
      // Not (yet) in ctx.physicalValues: either this operand is genuinely
      // logical (scratchpad), or a linalg.transpose sitting on its chain has
      // not been erased yet by RewriteTransposePattern. The latter is not a
      // fatal condition — defer to a later greedy iteration once the
      // transpose fires.
      if (chainBlockedByPendingTranspose(operand))
        return failure();
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
// Pattern: single-tensor elementwise op (retype in place)
//===----------------------------------------------------------------------===//

// An elementwise op's output can always be physicalized: it is rank-agnostic,
// so the physical type of its tensor operand(s) propagates unchanged (the
// work retypeChain used to do forward, in passing, over the whole chain).
//
// Deliberately local, and deliberately not isSingleTensorElementwiseOp (see
// the comment there): by the time Phase 2 runs, Phase 1 has already made
// every load's result physical, so the decision for an elementwise op is
// local to that op's own operand/result shapes, not a backward walk. This
// also means the rule covers multi-tensor-operand elementwise ops (addf,
// mulf, select, ...) for free -- it just requires every tensor operand to
// already agree on a shape before retyping the result to match.
//
// The shape-mismatch test alone is not sufficient to scope this pattern: it
// also matches ops that are not elementwise at all but happen to have one
// tensor operand and a differently-shaped result, e.g. tt.expand_dims
// (tensor<1xf32> -> tensor<1x1xf32> in softmax's row-max reduction). Base
// retypeChain never touched such ops because it only ever walked values
// reachable from a physicalized load's users -- an unannotated function
// (softmax has zero tt.spyre_tensor_layout markers) never seeded that walk
// at all.
//
// ctx.physicalValues restores that same reachability scoping, as a seeded
// worklist rather than a backward traceback: Phase 1 seeds it with every
// physical ktdp.load result (the root of every chain Phase 2 will retype),
// and this pattern grows it with the result of each op it retypes -- so
// membership IS the "reachable from a physicalized load" answer, propagated
// forward exactly the way base retypeChain did, just now living in Phase 2
// where the per-op decision belongs. An op on an unannotated path is simply
// never in the set, so tt.expand_dims in softmax is never retyped.
struct RewriteElementwisePattern : RewritePattern {
  const PassContext &ctx;
  RewriteElementwisePattern(MLIRContext *mlirCtx, const PassContext &layoutCtx)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, mlirCtx),
        ctx(layoutCtx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resTy)
      return failure();

    // Every tensor operand must be a RankedTensorType, and they must all
    // agree on shape -- that agreement is the safety condition: if one
    // operand is still logical and another already physical, decline rather
    // than retype to a guess.
    ArrayRef<int64_t> commonShape;
    bool sawTensorOperand = false;
    for (Value o : op->getOperands()) {
      auto opndTy = dyn_cast<RankedTensorType>(o.getType());
      if (!isa<RankedTensorType>(o.getType()))
        continue;
      if (!opndTy)
        return failure();
      if (!sawTensorOperand) {
        commonShape = opndTy.getShape();
        sawTensorOperand = true;
      } else if (opndTy.getShape() != commonShape) {
        return failure();
      }
    }
    if (!sawTensorOperand)
      return failure();

    // Idempotence: once the result already matches, stop matching, so the
    // greedy driver's repeated re-enqueue cannot re-fire this.
    if (resTy.getShape() == commonShape)
      return failure();

    // Reachability scoping (see comment above): require at least one tensor
    // operand to already be in the physicalized-value map. Without this, the
    // pattern would retype ops on paths that were never physicalized at all
    // (e.g. tt.expand_dims in an unannotated function).
    auto physOperandIt = llvm::find_if(op->getOperands(), [&](Value o) {
      return ctx.physicalValues.contains(o);
    });
    if (physOperandIt == op->getOperands().end())
      return failure();

    // Carry the originating operand's info (marker + any transpose perm)
    // forward to this op's result -- if several operands are already
    // physical (e.g. both sides of arith.addf), they were reconciled onto
    // the same physical shape by the operand-shape-agreement check above, so
    // any one of their PhysicalValueInfo entries is representative.
    PhysicalValueInfo info = ctx.physicalValues.find(*physOperandIt)->second;

    rewriter.modifyOpInPlace(op, [&]() {
      op->getResult(0).setType(
          RankedTensorType::get(commonShape, resTy.getElementType()));
    });
    // Propagate forward: this op's result is now itself a physicalized
    // value, so a downstream elementwise op reading it must see it as such.
    ctx.physicalValues[op->getResult(0)] = info;
    // The greedy driver re-enqueues an in-place-modified op itself, but not
    // its users — yet a user (e.g. a downstream linalg.transpose or another
    // elementwise op) may only become dispatchable now that this op's result
    // type has changed. Force those users back onto the worklist.
    for (Operation *user : llvm::make_early_inc_range(op->getResult(0).getUsers()))
      rewriter.modifyOpInPlace(user, [] {});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: linalg.transpose (erase and record permutation)
//===----------------------------------------------------------------------===//

// The transpose permutation is recorded in ctx.physicalValues (against
// `input`, which must already be an entry -- see PhysicalValueInfo) because
// this erase happens before dispatchSource can see the transpose: erasing it
// here (Phase 2, per-op, same as before but no longer forced by Phase 1's
// forward walk) still requires the downstream matmul/reduce to recover the
// permutation from somewhere once the transpose is gone.
struct RewriteTransposePattern : OpRewritePattern<linalg::TransposeOp> {
  const PassContext &ctx;
  RewriteTransposePattern(MLIRContext *mlirCtx, const PassContext &layoutCtx)
      : OpRewritePattern(mlirCtx, /*benefit=*/1), ctx(layoutCtx) {}

  LogicalResult matchAndRewrite(linalg::TransposeOp tr,
                                PatternRewriter &rewriter) const override {
    Value input = tr.getInput();
    auto it = ctx.physicalValues.find(input);
    if (it == ctx.physicalValues.end())
      return failure();

    auto perm = SmallVector<int64_t>(tr.getPermutation());
    LLVM_DEBUG({
      llvm::dbgs() << "  erasing transpose, perm: ";
      llvm::interleaveComma(perm, llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    auto &slot = it->second.transposePerm;
    slot = slot.empty() ? perm : composePerm(slot, perm);
    // Capture users before replaceOp() moves them onto `input`'s use-list:
    // the greedy driver does not automatically re-enqueue users of a
    // replaced value, but a downstream elementwise/transpose/contraction op
    // may only become dispatchable now that this transpose is gone.
    SmallVector<Operation *> formerUsers(tr.getResult()[0].getUsers().begin(),
                                         tr.getResult()[0].getUsers().end());
    rewriter.replaceOp(tr, input);
    for (Operation *user : formerUsers)
      rewriter.modifyOpInPlace(user, [] {});
    return success();
  }
};

// Bridging stage: the mirror image of emitSinkStage. The store's access tile
// is logical (no marker to physicalize into), but its data tile is physical
// (the source-side marker, from the load that produced it, is known). Gather
// the physical stick slices of the data tile into a logical-shaped
// accumulator via tensor.insert_slice, one stick loop per floor dim, then
// swap the store's data tile operand to that logical value — the access tile
// is left untouched, exactly as emitSinkStage leaves it untouched on the
// other side.
LogicalResult emitBridgeToLogical(mlir::ktdp::StoreOp st,
                                  const OperandPlan &sPlan,
                                  PatternRewriter &rewriter) {
  Value physTile = st.getDataTile();
  OpBuilder b(st.getOperation());
  Location loc = st.getLoc();

  Type elemTy = cast<RankedTensorType>(physTile.getType()).getElementType();
  llvm::ArrayRef<int64_t> physBlock = sPlan.coords.physBlock;
  int physRank = (int)physBlock.size();
  int64_t stickSize = physBlock[sPlan.dims.lane];
  unsigned logRank = sPlan.coords.logicalRank;

  auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

  llvm::SmallVector<int64_t> logShape(logRank, 0);
  for (int p : sPlan.dims.opTileDims) {
    int64_t logDim = sPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      logShape[logDim] = physBlock[p];
  }
  for (int p : sPlan.dims.floorDims) {
    int64_t logDim = sPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      logShape[logDim] = stickSize * physBlock[p];
  }
  Value logicalAcc = tensor::EmptyOp::create(b, loc, logShape, elemTy);

  llvm::SmallVector<OpFoldResult> physOffsetsBase(physRank, idx(0));
  llvm::SmallVector<OpFoldResult> physSizes(physRank);
  llvm::SmallVector<OpFoldResult> physStrides(physRank, idx(1));
  for (int p = 0; p < physRank; ++p)
    physSizes[p] = llvm::is_contained(sPlan.dims.floorDims, p)
                       ? idx(1) : idx(physBlock[p]);

  llvm::SmallVector<OpFoldResult> logOffsetsBase(logRank, idx(0));
  llvm::SmallVector<OpFoldResult> logSizesBase(logRank);
  llvm::SmallVector<OpFoldResult> logStrides(logRank, idx(1));
  for (int p : sPlan.dims.opTileDims) {
    int64_t logDim = sPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      logSizesBase[logDim] = idx(physBlock[p]);
  }
  for (int p : sPlan.dims.floorDims) {
    int64_t logDim = sPlan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      logSizesBase[logDim] = idx(stickSize);
  }

  Value acc = logicalAcc;
  for (int p : sPlan.dims.floorDims) {
    int64_t logDim = sPlan.dimRoles[p];
    if (logDim < 0 || (unsigned)logDim >= logRank) continue;

    int64_t tripCount = physBlock[p];
    Value stickSizeVal = arith::ConstantIndexOp::create(b, loc, stickSize);

    // The extracted slice must be logical-rank shaped (matching logAccumulator,
    // which insert_slice requires for a non-rank-reducing insert): the floor
    // dim being iterated collapses away entirely (it is not a logical axis of
    // its own — it and the lane dim share logDim), while every other physical
    // dim maps 1:1 onto a logical dim per dimRoles.
    llvm::SmallVector<int64_t> slShape(logRank);
    for (int p2 : sPlan.dims.opTileDims) {
      int64_t ld = sPlan.dimRoles[p2];
      if (ld >= 0 && (unsigned)ld < logRank)
        slShape[ld] = physBlock[p2];
    }
    slShape[logDim] = stickSize;
    auto slTy = RankedTensorType::get(slShape, elemTy);

    acc = emitStickLoop(b, loc, tripCount, acc,
        [&](OpBuilder &bb, Value s, Value logAccumulator) -> Value {
          llvm::SmallVector<OpFoldResult> physOff = physOffsetsBase;
          physOff[p] = s;
          Value physSlice = tensor::ExtractSliceOp::create(
              bb, loc, slTy, physTile, physOff, physSizes, physStrides);

          llvm::SmallVector<OpFoldResult> logOff = logOffsetsBase;
          logOff[logDim] =
              arith::MulIOp::create(bb, loc, s, stickSizeVal).getResult();
          return tensor::InsertSliceOp::create(
              bb, loc, physSlice, logAccumulator, logOff, logSizesBase, logStrides);
        });
  }

  rewriter.modifyOpInPlace(st, [&]() {
    st.getDataTileMutable().set(acc);
  });
  return success();
}

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
    if (!dataTy || !tileTy)
      return failure();
    // Defer if the data tile's producer is an elementwise op that has not
    // yet been retyped: dataTy.getRank() below is about to change once it
    // is, and deciding sink-vs-bridge on the stale rank would lock in the
    // wrong answer (see pendingElementwiseRetype).
    if (pendingElementwiseRetype(st.getDataTile(), ctx))
      return failure();
    if (dataTy.getRank() == (int)tileTy.getShape().size())
      return failure();
    auto marker = findMarkerForStore(st.getDataTile(), ctx);
    if (!marker) {
      // No destination marker to physicalize into: the access tile stays
      // logical. If the data tile is physical (source-side marker known),
      // this is the case the store output-decision table calls "the output
      // cannot be physicalized" — bridge with a loop that reassembles the
      // logical shape from the physical data tile's stick slices.
      auto srcMarker = findMarkerForOperand(st.getDataTile(), ctx);
      if (!srcMarker)
        return failure();

      LLVM_DEBUG(llvm::dbgs() << "  dispatching store bridge-to-logical at "
                              << st.getLoc() << "\n");

      auto physShape = dataTy.getShape();
      OperandCoords sC = OperandCoords::fromMarker(srcMarker,
                                                    tileTy.getShape().size(),
                                                    physShape);
      int physRank = (int)physShape.size();
      llvm::SmallVector<int64_t> dimRoleS(physRank);
      for (int p = 0; p < physRank; ++p)
        dimRoleS[p] = srcMarker.getPhysSrc()[p];
      OperandPlan sPlan = classify(st.getDataTile(), sC, dimRoleS);

      if (sPlan.dims.floorDims.empty())
        return failure();
      if (!sPlan.dims.loopDims.empty()) {
        ctx.hadError = true;
        return st.emitError(
            "spyre_tensor_layout: store bridge stage: unexpected reduction "
            "dim");
      }

      return emitBridgeToLogical(st, sPlan, rewriter);
    }

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
  patterns.add<RewriteTransposePattern>(mlirCtx, ctx);
  patterns.add<RewriteElementwisePattern>(mlirCtx, ctx);
  patterns.add<RewriteStorePattern>(mlirCtx, ctx);
}

} // namespace mlir::triton::ktdp
