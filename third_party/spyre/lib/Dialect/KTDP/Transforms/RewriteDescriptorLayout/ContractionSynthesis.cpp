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
#include "RewriteDescriptorLayout/PhysicalTypeAnalysis.h"
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
#include "llvm/ADT/STLExtras.h"
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
// Not a general elementwise predicate. It is scoped to findMarkerForStore's
// chain walk, where the single-tensor-operand restriction is load-bearing:
// rejecting two or more tensor operands is what stops that walk crossing a
// matmul (3 tensor operands) or a reduce (2) into an unrelated chain.
// Widening it requires re-proving that walk cannot leave the chain it started
// on.
//
// Forward retyping (deciding an op's own result type) must NOT reuse this --
// it would exclude multi-tensor-operand elementwise ops like arith.addf and
// select. See RewriteElementwisePattern, which uses its own local shape rule.
bool isSingleTensorElementwiseOp(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumOperands() == 0)
    return false;
  int tensorOps = 0;
  for (auto operand : op->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      ++tensorOps;
  return tensorOps == 1;
}

// Rebuild a DPS init at `accTy`, or null when its producer is one this cannot
// reproduce.
//
// Needed only in the Physical output-axis space, and there it is unavoidable:
// the accumulator is a *different physicalization of the same logical tensor*
// than the init the op arrived with (tensor<2x64xf16> against tensor<128xf16>),
// not a slab of it, so it can be neither reused nor sliced out — it has to be
// built again at the physical shape.
//
// Exactly two producers can be, and they are the two LowerComputeOps emits for
// a reduction's `outs`: a tensor.empty, and a linalg.fill of the combiner's
// neutral element over one (ConvertTTReduce). The fill is re-emitted rather than
// dropped: it is the neutral element the reduce's semantics need, and removing
// it is a device-specific fixup that belongs to DropReductionInitFill, which
// runs later and recognizes exactly this shape.
//
// Anything else — a loaded tensor carrying incoming data, say — holds values a
// rebuild would silently drop, so it is refused rather than guessed at.
static Value rebuildPhysicalInit(OpBuilder &b, Location loc, Value init,
                                 RankedTensorType accTy) {
  if (init.getType() == accTy)
    return init;
  auto empty = [&] {
    return tensor::EmptyOp::create(b, loc, accTy.getShape(),
                                   accTy.getElementType())
        .getResult();
  };
  if (auto fill = init.getDefiningOp<linalg::FillOp>())
    return linalg::FillOp::create(b, loc, fill.getInputs()[0], empty())
        .getResult(0);
  if (init.getDefiningOp<tensor::EmptyOp>())
    return empty();
  return {};
}

bool canRebuildPhysicalInit(Value init) {
  return init.getDefiningOp<linalg::FillOp>() ||
         init.getDefiningOp<tensor::EmptyOp>();
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

// Look up the marker for an access tile's base memView in the pass context.
triton::SpyreTensorLayoutOp
lookupMarkerFromTile(Value accessTile, const MarkerByMemView &markers) {
  auto tileOp = accessTile.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
  if (!tileOp)
    return {};
  auto it = markers.find(tileOp.getBase());
  return it != markers.end() ? it->second
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

// Emit a counted loop (scf.for) over a single iter_arg, or inline for trip <=
// 1. Used both for a reduction's stick loop (iter_arg is the accumulator) and
// for a parallel-scatter loop (iter_arg is the container being filled via
// tensor.insert_slice) -- the two calling contexts differ only in what `body`
// does with its iter_arg, not in the loop skeleton itself.
Value emitCountedLoop(OpBuilder &b, Location loc, int64_t tripCount,
                     Value iterArg,
                     llvm::function_ref<Value(OpBuilder &, Value, Value)> body) {
  if (tripCount <= 1) {
    Value s0 = arith::ConstantIndexOp::create(b, loc, 0);
    return body(b, s0, iterArg);
  }
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value ub = arith::ConstantIndexOp::create(b, loc, tripCount);
  auto forOp = scf::ForOp::create(b, loc, c0, ub, c1, ValueRange{iterArg});
  OpBuilder ib = OpBuilder::atBlockBegin(forOp.getBody());
  Value stepped =
      body(ib, forOp.getInductionVar(), forOp.getRegionIterArgs()[0]);
  scf::YieldOp::create(ib, loc, ValueRange{stepped});
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
  // Nothing to slice: every dim starts at 0 and spans the operand's whole
  // extent at unit stride, so the slice would be the identity. (Strides are
  // unit by construction above.) Emitting it costs an op the canonicalizer
  // then folds away.
  auto operandTy = cast<RankedTensorType>(plan.value.getType());
  bool wholeTensor = operandTy.getRank() == rank;
  for (int p = 0; wholeTensor && p < rank; ++p) {
    auto constOf = [](OpFoldResult r) -> std::optional<int64_t> {
      if (auto attr = dyn_cast<Attribute>(r))
        if (auto i = dyn_cast<IntegerAttr>(attr))
          return i.getInt();
      return std::nullopt;
    };
    auto off = constOf(offsets[p]);
    auto sz = constOf(sizes[p]);
    if (!off || *off != 0 || !sz || *sz != operandTy.getDimSize(p))
      wholeTensor = false;
  }
  if (wholeTensor && operandTy == resultTy)
    return plan.value;

  return tensor::ExtractSliceOp::create(
      b, loc, resultTy, plan.value, offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// Conversion stage: narrow N physical operands into one op-tile (a source op
// feeding matmul/reduce), or widen one op-tile into physical (a store's data
// tile). One direction parameter, not two emitters: both transpose, drive a
// counted loop, slice, and insert, differing only in which side of the loop is
// the physical container and whether an op fires inside it (narrow) or the loop
// body is a plain gather/scatter (widen).
//===----------------------------------------------------------------------===//

// Narrow stage emission: extract slices, optional transpose, call emitOp.
// Returns the replacement value for the original op's result.
Value emitNarrowStage(
    linalg::LinalgOp op,
    OpBuilder &b,
    const SourceOpSpec &spec,
    llvm::ArrayRef<OperandPlan> plans,
    const OperandSetTripCounts &tripCounts) {
  Location loc = op.getLoc();
  auto emitOp = spec.emitOp;

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

  // In the Physical output-axis space the accumulator is rebuilt at accTy
  // rather than reused (see rebuildPhysicalInit). Not gated on a type mismatch
  // alone: in the Logical space a mismatch is the parallel-scatter slab
  // relation below, where cVal is deliberately the wider container and accTy
  // one slab of it.
  //
  // Phase 2A only puts a reduce in this space when canRebuildPhysicalInit holds
  // for the same init, so a null here means those two have drifted apart.
  if (spec.outputAxes == OutputAxisSpace::Physical) {
    cVal = rebuildPhysicalInit(b, loc, cVal, accTy);
    if (!cVal)
      return {};
  }

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
    return emitCountedLoop(bb, loc, stickFactor, acc,
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

  return emitCountedLoop(b, loc, parallelFactor, cVal,
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

// Classify one operand and populate plans[i]. What the op tile may absorb --
// its reduce axis set, its surviving stick-index dims -- is declared by the
// calling pattern on the spec; see SourceOpSpec.
LogicalResult dispatchSource(linalg::LinalgOp op, const SourceOpSpec &spec,
                             const PassContext &ctx,
                             PatternRewriter &rewriter) {
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
      buildDimRoles(coords, effectiveCanonicalAxes, spec.outputAxes, dimRoles);
      plans[i] = classify(operand, coords, dimRoles, spec.outputAxes);
      LLVM_DEBUG({
        llvm::dbgs() << "  operand " << i << ": opTileDims=[";
        llvm::interleaveComma(plans[i].dims.opTileDims, llvm::dbgs());
        llvm::dbgs() << "] reduceLoopDims=[";
        llvm::interleaveComma(plans[i].dims.reduceLoopDims, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });
    } else {
      // Not in ctx.physicalValues. Two separate questions, asked in this
      // order, because the first is not about physicality at all.
      //
      // 1. LEGALITY. A reshape/broadcast producing this operand has an
      //    element-to-index mapping that matches neither a physical tensor's
      //    nor a plain logical one's, so no conversion is defined for it.
      //    Physicality has three answers -- physical, not yet, logical -- and
      //    "no defined conversion exists" is a fourth, so it gets its own
      //    check rather than being folded into the lookup below. It must come
      //    FIRST: the analysis correctly declines to give these ops a physical
      //    type (see ReshapePropagation / BroadcastPropagation), so such an
      //    operand is absent from the analysis map, and absence means
      //    "genuinely logical" -- committing it as a scratchpad, which is
      //    exactly the silent wrong slice this rejects. The two checks are
      //    complementary, not redundant.
      if (auto *defOp = operand.getDefiningOp()) {
        if (isa<tensor::ReshapeOp, tensor::ExpandShapeOp,
                tensor::CollapseShapeOp, linalg::BroadcastOp>(defOp))
          return emitFatalError(op, ctx,
              "spyre_tensor_layout: source op operand is produced by a "
              "reshape/broadcast, which cannot be treated as a physical "
              "load or a plain logical scratchpad");
      }
      // 2. PHYSICALITY. Phase 2A decided every value's final physical type up
      //    front, from Phase 1's roots alone. So an operand present in the
      //    analysis map WILL be physical; it just is not yet, because the
      //    rewrite that makes it so (a transpose erasure, an upstream retype)
      //    has not landed in this greedy iteration. Defer.
      if (ctx.physicalTypeAnalysis &&
          ctx.physicalTypeAnalysis->contains(operand))
        return failure();
      // Absent from the analysis map: genuinely logical. Commit.
      auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorTy ||
          tensorTy.getRank() != (int64_t)spec.operands[i].canonicalAxes.size())
        return emitFatalError(op, ctx,
            "spyre_tensor_layout: source op operand is neither a physical "
            "load nor a logical (scratchpad) tensor of the expected rank");
      plans[i] = classifyScratchpad(operand, spec.operands[i]);
    }
  }

  // When the op can absorb its whole reduce axis set directly, fold
  // reduceLoopDims into opTileDims (as WholeBlock slices spanning the full
  // physical extent) before reconcileOperandSet folds them into a trip count.
  // Only the accumulating bucket moves: scatterDims stay outside the op tile,
  // since absorbing a reduce axis says nothing about where the output axes are.
  if (spec.absorbReduceLoopDims) {
    for (auto &plan : plans) {
      for (int p : plan.dims.reduceLoopDims) {
        plan.dims.opTileDims.push_back(p);
        plan.dims.sliceKind[p] = SliceKind::WholeBlock;
      }
      plan.dims.reduceLoopDims.clear();
      llvm::sort(plan.dims.opTileDims);
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
    for (int p : plans[i].dims.reduceLoopDims)
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
        "scatter — two annotated operands carry multi-stick scatter dims with "
        "different trip counts or on different output axes, which would need "
        "two independent scatter loops (not supported)");

  OpBuilder b(op.getOperation());
  Value result = emitNarrowStage(op, b, spec, plans, tripCounts);
  if (!result)
    return emitFatalError(op, ctx,
        "spyre_tensor_layout: the op's output is to be physicalized but its "
        "init operand cannot be rebuilt at the physical shape — only a "
        "tensor.empty or a linalg.fill over one can be");

  // A Physical output-axis space means this op's result IS physical, under the
  // layout Phase 2A paired it with. Record it, so a consumer (an elementwise op
  // downstream, or a store deciding sink vs. nothing-to-do) sees a physical
  // value rather than deciding against a shape it cannot account for.
  //
  // `result` is a value the analysis never saw, so it also inherits the entry
  // the analysis made for the result it replaces -- see
  // PhysicalTypeCarryForward. Every pattern that mints a replacement wants this
  // pair of lines, and nothing else.
  if (spec.outputAxes == OutputAxisSpace::Physical) {
    ctx.physicalValues[result] = PhysicalValueInfo{spec.outputMarker, {}};
    ctx.physicalTypes.carryForward(op->getResult(0), result);
  }
  rewriter.replaceOp(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// Sink stage (store scatter)
//===----------------------------------------------------------------------===//

// The marker half of findStoreDestination: the layout of the store this value
// ultimately feeds, ignoring that store's tile shape. Named separately because
// the store pattern only ever asks "is the output annotated", where a shape it
// already has in hand would be noise.
triton::SpyreTensorLayoutOp findMarkerForStore(Value value,
                                               const PassContext &ctx) {
  return findStoreDestination(value, ctx.physMemViewToMarker).marker;
}

// Which side of a store's widening conversion is the physical one. The other
// side is always logical -- one op-tile, one physical container -- so naming
// the physical side's role also names the logical side's by elimination.
enum class WidenTarget {
  Physical, // op-tile (logical) -> physical D tensor shape: the sink case.
  Logical,  // physical data tile -> logical (no output marker): the bridge
            // case, gathering stick slices back into a logical accumulator.
};

// Widen stage: the shared conversion path's Widen direction. One op-tile
// side and one physical-block side; the loop scatters (Physical target) or
// gathers (Logical target) stick slices between them, one scatter dim at a
// time, then repoints the store's data tile operand to the assembled result.
//
// Whichever side is logical always carries the multiply-by-stickSize offset
// (it is the wide side being addressed one stick at a time); whichever side
// is physical always carries the raw stick index `p` (it already has one
// slot per stick). That invariant holds regardless of which side plays
// source vs. destination, which is what lets one emitter serve both, selected
// by `target`.
//
// Only the Physical target ever transposes: `plan.transposePerm` is filled by
// resolveOperand(), which the sink caller runs and the bridge caller does
// not (see RewriteStorePattern) -- so for a Logical target `plan.transposePerm`
// is always empty and the guard below is simply never taken.
LogicalResult emitWidenStage(mlir::ktdp::StoreOp st,
                             const OperandPlan &plan,
                             WidenTarget target,
                             PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "  widen stage: stickSize=" << plan.dims.stickSize
                          << ", scatterDims=" << plan.dims.scatterDims.size()
                          << ", target=" << (target == WidenTarget::Physical
                                                  ? "physical" : "logical")
                          << "\n");
  Value opTileSide = st.getDataTile();
  OpBuilder b(st.getOperation());
  Location loc = st.getLoc();

  Type elemTy = cast<RankedTensorType>(opTileSide.getType()).getElementType();

  llvm::ArrayRef<int64_t> physBlock = plan.coords.physBlock;
  int physRank = (int)physBlock.size();
  int64_t stickSize = physBlock[plan.dims.lane];

  unsigned logRank = plan.coords.logicalRank;
  llvm::ArrayRef<int64_t> perm = plan.transposePerm;

  auto logDimToPos = [&](int64_t d) -> unsigned {
    return perm.empty() ? (unsigned)d : (unsigned)perm[d];
  };

  if (target == WidenTarget::Physical && !perm.empty())
    opTileSide = emitTranspose(b, loc, opTileSide, perm);

  auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

  // The op-tile-side offsets/sizes (rank logRank) and the physical-side
  // offsets/sizes (rank physRank) are the same two arrays regardless of
  // target; only which one is the extract_slice source vs. the insert_slice
  // destination (and which one is the freshly created container) flips.
  llvm::SmallVector<OpFoldResult> logOffsetsBase(logRank, idx(0));
  llvm::SmallVector<OpFoldResult> logSizesBase(logRank);
  llvm::SmallVector<OpFoldResult> logStrides(logRank, idx(1));
  for (int p : plan.dims.opTileDims) {
    int64_t logDim = plan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      logSizesBase[logDimToPos(logDim)] = idx(physBlock[p]);
  }
  for (int p : plan.dims.scatterDims) {
    int64_t logDim = plan.dimRoles[p];
    if (logDim >= 0 && (unsigned)logDim < logRank)
      logSizesBase[logDimToPos(logDim)] = idx(stickSize);
  }

  llvm::SmallVector<OpFoldResult> physOffsetsBase(physRank, idx(0));
  llvm::SmallVector<OpFoldResult> physSizes(physRank);
  llvm::SmallVector<OpFoldResult> physStrides(physRank, idx(1));
  for (int p = 0; p < physRank; ++p)
    physSizes[p] = llvm::is_contained(plan.dims.scatterDims, p)
                       ? idx(1) : idx(physBlock[p]);

  // The container is the physical block for a Physical target, or the
  // logical (opTileDims/scatterDims-derived) shape for a Logical target.
  Value container;
  if (target == WidenTarget::Physical) {
    llvm::SmallVector<int64_t> physShape(physBlock.begin(), physBlock.end());
    container = tensor::EmptyOp::create(b, loc, physShape, elemTy);
  } else {
    llvm::SmallVector<int64_t> logShape(logRank, 0);
    for (int p : plan.dims.opTileDims) {
      int64_t logDim = plan.dimRoles[p];
      if (logDim >= 0 && (unsigned)logDim < logRank)
        logShape[logDim] = physBlock[p];
    }
    for (int p : plan.dims.scatterDims) {
      int64_t logDim = plan.dimRoles[p];
      if (logDim >= 0 && (unsigned)logDim < logRank)
        logShape[logDim] = stickSize * physBlock[p];
    }
    container = tensor::EmptyOp::create(b, loc, logShape, elemTy);
  }

  Value acc = container;
  for (int p : plan.dims.scatterDims) {
    int64_t logDim = plan.dimRoles[p];
    if (logDim < 0 || (unsigned)logDim >= logRank) continue;

    unsigned tileDim = logDimToPos(logDim);
    int64_t tripCount = physBlock[p];
    Value stickSizeVal = arith::ConstantIndexOp::create(b, loc, stickSize);

    // The op-tile-rank slice type: for a Physical target this is the input
    // slice cut from the logical op-tile side; for a Logical target this is
    // the (rank-reducing) result type of the slice cut from the physical
    // side, forced to logical rank so it matches the logical accumulator for
    // insert_slice.
    llvm::SmallVector<int64_t> slShape(logRank);
    for (int p2 : plan.dims.opTileDims) {
      int64_t ld = plan.dimRoles[p2];
      if (ld >= 0 && (unsigned)ld < logRank)
        slShape[logDimToPos(ld)] = physBlock[p2];
    }
    slShape[tileDim] = stickSize;
    auto slTy = RankedTensorType::get(slShape, elemTy);

    acc = emitCountedLoop(b, loc, tripCount, acc,
        [&](OpBuilder &bb, Value s, Value iterAcc) -> Value {
          if (target == WidenTarget::Physical) {
            llvm::SmallVector<OpFoldResult> inOff = logOffsetsBase;
            inOff[tileDim] =
                arith::MulIOp::create(bb, loc, s, stickSizeVal).getResult();
            Value inputSlice = tensor::ExtractSliceOp::create(
                bb, loc, slTy, opTileSide, inOff, logSizesBase, logStrides);

            llvm::SmallVector<OpFoldResult> containerOff = physOffsetsBase;
            containerOff[p] = s;
            return tensor::InsertSliceOp::create(
                bb, loc, inputSlice, iterAcc, containerOff, physSizes,
                physStrides);
          }

          llvm::SmallVector<OpFoldResult> physOff = physOffsetsBase;
          physOff[p] = s;
          Value physSlice = tensor::ExtractSliceOp::create(
              bb, loc, slTy, opTileSide, physOff, physSizes, physStrides);

          llvm::SmallVector<OpFoldResult> logOff = logOffsetsBase;
          logOff[tileDim] =
              arith::MulIOp::create(bb, loc, s, stickSizeVal).getResult();
          return tensor::InsertSliceOp::create(
              bb, loc, physSlice, iterAcc, logOff, logSizesBase, logStrides);
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
  // Both absorption flags keep their defaults, and both defaults are the answer
  // for a matmul-like op rather than a fallback. It contracts exactly one K, so
  // a second reduce axis must be a real accumulation loop (absorbReduceLoopDims =
  // false); and its accumulator extents must match the A/B slice extents, so a
  // surviving multi-stick parallel dim is scattered by an outer loop rather than
  // carried as an output axis (outputAxes = Logical). MatmulPropagation states
  // the second of those as its own rule.
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
    // Idempotence, first half. A reduce this pattern already rewrote into the
    // Physical output-axis space has its own result registered as physical (see
    // dispatchSource), and that registration is the direct statement of "this
    // op is already final". It is needed because such a reduce keeps consuming
    // the physical ktdp.load unsliced, so every other match condition below
    // still holds -- and its `dimensions` now hold physical positions that are
    // *within* logical range (a single surviving reduce axis at physical
    // position 1, say), so the second half of the guard further down cannot see
    // it either.
    if (ctx.physicalValues.contains(rd.getResult(0)))
      return failure();

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

    // Idempotence, second half: an already-rewritten case-1 reduce, i.e. one
    // left in the Logical output-axis space. Such a
    // reduce keeps its physical extract_slice as input; when every dim on
    // that slice is a WholeBlock (no narrowing at all -- e.g. a fully
    // absorbed reduce with no residual stick loop), RewriteElementwisePattern
    // sees identical operand/result shapes and copies the *same* marker
    // forward onto the slice, making it look like a fresh physical operand to
    // needsDispatch above. Once rewritten, rd.getDimensions() holds
    // slice-local physical positions (e.g. [0, 2]), not logical dims -- so if
    // any entry is already >= logicalRank, this op's `dimensions` cannot be
    // logical dims of the *current* marker and this is a re-visit of an
    // already-final reduce: reinterpreting it as logical again would silently
    // compute nonsense (see emitReduceOp/targetOrder below, which assume
    // rd.getDimensions() are logical). Fail rather than risk that.
    if (llvm::any_of(rd.getDimensions(),
                     [&](int64_t d) { return d >= (int64_t)logicalRank; }))
      return failure();

    llvm::SmallVector<int64_t> canonicalAxes(logicalRank, -1);
    unsigned outAxis = 0;
    for (unsigned d = 0; d < logicalRank; ++d)
      if (!llvm::is_contained(rd.getDimensions(), (int64_t)d))
        canonicalAxes[d] = outAxis++;

    // Where this reduce's output axes live is Phase 2A's answer, not a local
    // guess. ReducePropagation decided whether the result is physical -- i.e.
    // whether the descriptor it is stored to carries exactly the layout the
    // operand's surviving stick structure induces -- and recorded the type and
    // the marker when it is. Present means Physical; absent means Logical, the
    // shape every reduce had before, where a surviving stick-index dim is a
    // scatter dim and any physical form is built afterwards by the store.
    OutputAxisSpace outputAxes = OutputAxisSpace::Logical;
    triton::SpyreTensorLayoutOp outputMarker;
    if (ctx.physicalTypeAnalysis) {
      auto it = ctx.physicalTypeAnalysis->find(rd.getResult(0));
      if (it != ctx.physicalTypeAnalysis->end()) {
        outputAxes = OutputAxisSpace::Physical;
        outputMarker = it->second.marker;
      }
    }

    // targetOrder holds one entry per non-floor physical dim, in physical
    // order, role or -1 for a reduced dim. linalg.reduce's `dimensions` is
    // DenseArrayStrictlySorted with no trailing-position requirement, so
    // reduced dims need not be moved to the end.
    //
    // Derived from buildDimRoles' own output rather than from a second copy of
    // the role rule, and filtered by the same predicate in the same
    // OutputAxisSpace that classify() will use. That is what re-establishes the
    // 1:1 correspondence with opTileDims in physical order now that a surviving
    // stick-index dim can be an op-tile dim: the two sides cannot disagree about which
    // dims are excluded, because they ask one predicate with one argument.
    auto physShape = cast<RankedTensorType>(rd.getInputs()[0].getType())
                         .getShape();
    OperandCoords coords =
        OperandCoords::fromMarker(marker, logicalRank, physShape);
    llvm::SmallVector<int64_t> physDimRoles;
    buildDimRoles(coords, canonicalAxes, outputAxes, physDimRoles);
    llvm::SmallVector<int64_t> targetOrder;
    for (unsigned p = 0; p < physDimRoles.size(); ++p)
      if (!isScatterDim(physDimRoles[p],
                                  static_cast<CoordOp>(coords.op[p]), outputAxes))
        targetOrder.push_back(physDimRoles[p]);

    // The reduced slice-local axis positions are exactly the -1 slots of
    // targetOrder: resolveOperand's transpose (if any) reorders the operand
    // into targetOrder's order, so slot t ends up holding a reduced axis iff
    // targetOrder[t] == -1, regardless of where it sat before.
    llvm::SmallVector<int64_t> reducedSlicePositions;
    for (unsigned t = 0; t < targetOrder.size(); ++t)
      if (targetOrder[t] == -1)
        reducedSlicePositions.push_back((int64_t)t);

    Block &combinerBlock = rd.getOperation()->getRegion(0).front();
    llvm::SmallVector<Operation *> combinerOps;
    for (Operation &op : combinerBlock.without_terminator())
      combinerOps.push_back(&op);
    auto combinerYield = cast<linalg::YieldOp>(combinerBlock.getTerminator());
    llvm::SmallVector<Value> yieldVals(combinerYield.getValues().begin(),
                                       combinerYield.getValues().end());
    llvm::SmallVector<Value> origBlockArgs(combinerBlock.getArguments().begin(),
                                           combinerBlock.getArguments().end());

    auto emitReduceOp = [dims = reducedSlicePositions,
                         combinerOps = std::move(combinerOps),
                         yieldVals = std::move(yieldVals),
                         origBlockArgs = std::move(origBlockArgs)](
                            OpBuilder &b, Location loc,
                            llvm::ArrayRef<Value> slices, Value acc,
                            RankedTensorType accTy) -> Value {
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
    // linalg.reduce can absorb its whole reduce axis set directly via
    // `dimensions`, unlike matmul, which can only ever contract one axis.
    spec.absorbReduceLoopDims = true;
    spec.outputAxes = outputAxes;
    spec.outputMarker = outputMarker;
    spec.emitOp = emitReduceOp;
    return dispatchSource(rd, spec, ctx, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Pattern: single-tensor elementwise op (retype in place)
//===----------------------------------------------------------------------===//

// An elementwise op's output can always be physicalized: it is rank-agnostic,
// so the physical type of its tensor operand(s) propagates unchanged.
//
// Deliberately local, and deliberately not isSingleTensorElementwiseOp (see
// the comment there): by the time Phase 2 runs, Phase 1 has already made
// every load's result physical, so the decision for an elementwise op is
// local to that op's own operand/result shapes, not a backward walk. That
// also covers multi-tensor-operand elementwise ops (addf, mulf, select, ...)
// for free -- it just requires every tensor operand to already agree on a
// shape before retyping the result to match.
//
// The shape-mismatch test alone is not sufficient to scope this pattern: it
// also matches ops that are not elementwise at all but happen to have one
// tensor operand and a differently-shaped result, e.g. tt.expand_dims
// (tensor<1xf32> -> tensor<1x1xf32> in softmax's row-max reduction).
// ctx.physicalValues supplies the missing reachability scoping: Phase 1 seeds
// it with every physical ktdp.load result (the root of every chain Phase 2
// will retype) and this pattern grows it with the result of each op it
// retypes, so membership IS the "reachable from a physicalized load" answer.
// An op on an unannotated path is simply never in the set, so tt.expand_dims
// in softmax (zero tt.spyre_tensor_layout markers) is never retyped.
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
    // REQUIRED, unlike the transpose pattern's counterpart. setType above
    // changes this op's result type in place; the greedy driver re-enqueues the
    // modified op but not its users, and a user's own match condition reads
    // that operand's type. A downstream linalg.transpose whose init still has
    // the logical rank is the case that bites: left un-revisited it keeps a
    // rank-2 init against a now rank-3 input and fails its verifier.
    //
    // The analysis cannot substitute here. It predicts what a type WILL be, not
    // when a neighbour's in-place retype has landed, and this is the latter.
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
  // Benefit 1, like every other pattern here: dispatchSource asks the Phase 2A
  // analysis, which knows the transpose's permutation up front, so erasing the
  // transpose ahead of it buys nothing.
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
    // No re-enqueue of former users needed. Erasing this transpose does not
    // change any consumer's dispatchability: dispatchSource reads the analysis,
    // which already accounts for this permutation.
    rewriter.replaceOp(tr, input);
    return success();
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
    if (!dataTy || !tileTy)
      return failure();
    // Defer if the data tile's producer is an elementwise op that has not
    // yet been retyped: dataTy.getRank() below is about to change once it
    // is, and deciding sink-vs-bridge on the stale rank would lock in the
    // wrong answer (see pendingElementwiseRetype).
    if (pendingElementwiseRetype(st.getDataTile(), ctx))
      return failure();
    // The same hazard for a producer Phase 2A has predicted physical whose
    // rewrite has not landed yet, asked as the analysis question it is rather
    // than by inspecting the producer. A source op (a reduce whose result is
    // physicalized) REPLACES its op, so until it does, the data tile is still
    // the logical value and dataTy.getRank() below would send this store down
    // the sink path -- building a widen loop whose whole job is to produce the
    // shape the replacement will already have, around a value about to be
    // swapped out from under it. Deferral is safe rather than a stall for the
    // reason dispatchSource's own deferral is: presence in the analysis means
    // the value WILL be physical, and whatever makes it so registers it in
    // ctx.physicalValues, which re-enqueues this store.
    if (ctx.physicalTypeAnalysis &&
        ctx.physicalTypeAnalysis->contains(st.getDataTile()) &&
        !ctx.physicalValues.contains(st.getDataTile()))
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
      // A store's own axes are the logical dims of the tensor it writes; it has
      // no output-axis space of its own to choose, so the widen stage always
      // classifies in the Logical one -- the scatter dims it finds are exactly
      // the sticks it must gather or scatter one at a time.
      OperandPlan sPlan =
          classify(st.getDataTile(), sC, dimRoleS, OutputAxisSpace::Logical);

      if (sPlan.dims.scatterDims.empty())
        return failure();
      if (!sPlan.dims.reduceLoopDims.empty()) {
        ctx.hadError = true;
        return st.emitError(
            "spyre_tensor_layout: store bridge stage: unexpected reduction "
            "dim");
      }

      return emitWidenStage(st, sPlan, WidenTarget::Logical, rewriter);
    }

    LLVM_DEBUG(llvm::dbgs() << "  dispatching store sink at " << st.getLoc() << "\n");

    llvm::ArrayRef<int64_t> physBlock = tileTy.getShape();
    unsigned logRank = dataTy.getRank();
    OperandCoords dC = OperandCoords::fromMarker(marker, logRank, physBlock);

    // A store contracts nothing, so its canonicalAxes is the dense identity
    // (logical dim d carries role d) -- buildDimRoles against that identity
    // reduces to dimRoles[p] = coords.src[p] = marker.getPhysSrc()[p], the
    // same classification dispatchSource runs for a source operand.
    llvm::SmallVector<int64_t> identityAxes(logRank);
    std::iota(identityAxes.begin(), identityAxes.end(), 0);
    llvm::SmallVector<int64_t> dimRoleD;
    buildDimRoles(dC, identityAxes, OutputAxisSpace::Logical, dimRoleD);

    OperandPlan dPlan =
        classify(st.getDataTile(), dC, dimRoleD, OutputAxisSpace::Logical);

    // The store's two preconditions, expressed as resolution outcomes: an
    // empty scatterDims means nothing to scatter, and a non-empty
    // reduceLoopDims means a reduction the store cannot express. Checked here
    // rather than inside the emitter. Do not call reconcileOperandSet on this
    // path: it is a no-op for a store (dimRoles are phys_src, always >= 0, so
    // reduceLoopDims is always empty already) and would assert a cross-operand
    // dependency that does not exist for a single data tile.
    if (dPlan.dims.scatterDims.empty()) {
      ctx.hadError = true;
      return st.emitError(
          "spyre_tensor_layout: store sink stage requires at least one "
          "scatter dim in the output layout");
    }
    if (!dPlan.dims.reduceLoopDims.empty()) {
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

    return emitWidenStage(st, dPlan, WidenTarget::Physical, rewriter);
  }
};

} // namespace

namespace mlir::triton::ktdp {

//===----------------------------------------------------------------------===//
// Store destination lookup
//===----------------------------------------------------------------------===//

StoreDestination findStoreDestination(Value value,
                                     const MarkerByMemView &markers) {
  llvm::SmallVector<Value> worklist = {value};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (auto *user : v.getUsers()) {
      if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        auto marker = lookupMarkerFromTile(st.getAccessTile(), markers);
        if (!marker)
          continue;
        auto tileTy =
            dyn_cast<mlir::ktdp::AccessTileType>(st.getAccessTile().getType());
        if (!tileTy)
          continue;
        return {marker, llvm::SmallVector<int64_t>(tileTy.getShape())};
      }
      if (!isSingleTensorElementwiseOp(user))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return {};
}

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
