//===- Classify.cpp - Operand classification for contraction synthesis ----===//
//
// Derives per-operand slicing/transpose plans from physical layout coordinates
// and einsum dim-role assignments.
//
//===----------------------------------------------------------------------===//

#include "RewriteDescriptorLayout/Classify.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#define DEBUG_TYPE "rewrite-descriptor-layout"

namespace mlir::triton::ktdp {

void buildDimRoles(const OperandCoords &coords,
                   llvm::ArrayRef<int64_t> canonicalAxes,
                   llvm::SmallVectorImpl<int64_t> &roles) {
  int n = (int)coords.src.size();
  roles.resize(n);
  for (int p = 0; p < n; ++p) {
    int64_t logDim = coords.src[p];
    roles[p] = (logDim < (int64_t)canonicalAxes.size())
                   ? canonicalAxes[logDim]
                   : -1;
  }
}

OperandPlan classify(Value val, const OperandCoords &coords,
                     llvm::ArrayRef<int64_t> dimRoles) {
  int rank = (int)dimRoles.size();
  OperandPlan plan;
  plan.value     = val;
  plan.coords    = coords;
  plan.dimRoles  = llvm::SmallVector<int64_t>(dimRoles.begin(), dimRoles.end());

  ClassifiedDims &d = plan.dims;
  // Find the lane (mod) dimension — the physical dim carrying CoordOp::Mod.
  d.lane = rank - 1;  // fallback
  for (unsigned k = 0; k < rank; ++k) {
    if (static_cast<CoordOp>(coords.op[k]) == CoordOp::Mod) {
      d.lane = k;
      break;
    }
  }
  d.stickSize = coords.physBlock[d.lane];
  d.opInnerDim = -1;

  for (int p = rank - 1; p >= 0; --p) {
    int64_t role = dimRoles[p];
    bool isFloor = isFloorDim(role, static_cast<CoordOp>(coords.op[p]));
    if (role == -1) {
      d.reduceDims.push_back(p);
      if (d.opInnerDim == -1) {
        d.opInnerDim = p;
        d.opTileDims.push_back(p);
      } else {
        d.loopDims.push_back(p);
      }
    } else if (isFloor) {
      d.floorDims.push_back(p);
    } else {
      d.opTileDims.push_back(p);
    }
  }
  std::reverse(d.floorDims.begin(), d.floorDims.end());
  std::reverse(d.loopDims.begin(), d.loopDims.end());
  std::reverse(d.opTileDims.begin(), d.opTileDims.end());
  std::reverse(d.reduceDims.begin(), d.reduceDims.end());

  d.sliceKind.assign(rank, SliceKind::WholeBlock);
  auto markList = [&](llvm::ArrayRef<int> dims) {
    for (int p : dims)
      d.sliceKind[p] = SliceKind::StickIndex;
  };
  markList(d.floorDims);
  markList(d.loopDims);
  if (d.opInnerDim != -1 &&
      coords.physBlock[d.opInnerDim] > d.stickSize)
    d.sliceKind[d.opInnerDim] = SliceKind::StickifiedBlock;

  LLVM_DEBUG({
    llvm::dbgs() << "    classify: lane=" << d.lane
                 << " stickSize=" << d.stickSize
                 << " opInner=" << d.opInnerDim
                 << " floorDims=" << d.floorDims.size()
                 << " loopDims=" << d.loopDims.size()
                 << " reduceDims=" << d.reduceDims.size() << "\n";
  });

  return plan;
}

OperandPlan classifyScratchpad(Value val, const SourceOperandSpec &opSpec) {
  auto tensorTy = mlir::cast<RankedTensorType>(val.getType());
  int rank = (int)tensorTy.getRank();
  OperandPlan plan;
  plan.value = val;
  plan.coords.src        = {};
  plan.coords.op         = {};
  plan.coords.arg        = {};
  plan.coords.logicalRank = (unsigned)rank;
  plan.coords.physBlock   = tensorTy.getShape();

  ClassifiedDims &d = plan.dims;
  d.lane       = rank - 1;
  d.stickSize  = tensorTy.getDimSize(rank - 1);
  d.opInnerDim = -1;
  for (int p = 0; p < rank; ++p) {
    d.opTileDims.push_back(p);
    plan.dimRoles.push_back(opSpec.canonicalAxes[p]);
  }
  d.sliceKind.assign(rank, SliceKind::WholeBlock);

  plan.transposePerm      = {};
  for (int p : d.opTileDims)
    plan.opExtents.push_back(tensorTy.getDimSize(p));
  return plan;
}

int64_t opSliceExtent(const OperandPlan &plan, int p) {
  return plan.dims.sliceKind[p] == SliceKind::StickifiedBlock
             ? plan.dims.stickSize
             : plan.coords.physBlock[p];
}

OperandSetTripCounts
reconcileOperandSet(llvm::SmallVectorImpl<OperandPlan> &plans) {
  // StickifiedBlock demotion (before the trip-count fold below, since that
  // fold reads sliceKind).
  bool anyLoop = llvm::any_of(plans, [](const OperandPlan &p) {
    return !p.dims.loopDims.empty();
  });
  LLVM_DEBUG(llvm::dbgs() << "    reconcile: anyLoop=" << anyLoop
                          << (anyLoop ? " (no demotion)" : " (demoting StickifiedBlock)") << "\n");
  if (!anyLoop) {
    for (auto &plan : plans)
      for (auto &sk : plan.dims.sliceKind)
        if (sk == SliceKind::StickifiedBlock)
          sk = SliceKind::WholeBlock;
  }

  OperandSetTripCounts result;

  // Determine the stick loop trip count (stickFactor).
  for (auto &plan : plans) {
    for (int p : plan.dims.loopDims) {
      if (static_cast<CoordOp>(plan.coords.op[p]) != CoordOp::FloorDiv)
        continue;
      int64_t logDim = plan.dimRoles[p];
      if (logDim >= 0)
        continue;
      int64_t f;
      if (plan.dims.sliceKind[p] == SliceKind::StickifiedBlock)
        f = plan.coords.physBlock[p] / plan.dims.stickSize;
      else
        f = plan.coords.physBlock[p];
      if (f <= 1)
        continue;
      if (result.stickFactor != 1 && result.stickFactor != f)
        llvm_unreachable("reconcileOperandSet: plans disagree on stickFactor");
      result.stickFactor = f;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "    reconcile: stickFactor=" << result.stickFactor
                          << ", " << plans.size() << " operand plans\n");

  // Determine the parallel-scatter trip count (parallelFactor): a *parallel*
  // (role >= 0) floor dim spanning more than one stick. Such a dim cannot be
  // folded into the op tile — `accDims` holds the per-stick extent because
  // linalg.matmul requires its init/result extents to match the A/B slice
  // extents — so the full parallel extent is assembled by tiling the
  // accumulator across an outer loop.
  //
  // `parallelAccAxis` is the accumulator axis the parallel floor dim maps to
  // (`dimRoles[p]`, which indexes `accDims` directly); `parallelStickSize` is
  // that plan's lane extent, i.e. the width of one scattered slab.
  for (auto &plan : plans) {
    for (int p : plan.dims.floorDims) {
      int64_t role = plan.dimRoles[p];
      if (role < 0 || plan.coords.physBlock[p] <= 1)
        continue;
      int64_t f = plan.coords.physBlock[p];
      if (result.parallelFactor != 1 &&
          (result.parallelFactor != f || result.parallelAccAxis != role)) {
        result.parallelAgrees = false;
        return result;
      }
      result.parallelFactor = f;
      result.parallelAccAxis = role;
      result.parallelStickSize = plan.coords.physBlock[plan.dims.lane];
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "    reconcile: parallelFactor=" << result.parallelFactor
                          << " accAxis=" << result.parallelAccAxis
                          << " stickSize=" << result.parallelStickSize << "\n");

  return result;
}

void resolveOperand(OperandPlan &plan, llvm::ArrayRef<int64_t> targetOrder,
                    TransposeDirection direction) {
  // `targetOrder` is a property of the *target op*, not of this operand's
  // physical layout, so it is never reordered by an erased transpose perm.
  auto perm = computeTransposePerm(plan.dims.opTileDims, plan.dimRoles,
                                   targetOrder);
  plan.transposePerm = (direction == TransposeDirection::Widen && !perm.empty())
                            ? invertPerm(perm)
                            : perm;

  plan.opExtents.clear();
  for (int p : plan.dims.opTileDims)
    plan.opExtents.push_back(opSliceExtent(plan, p));
}

} // namespace mlir::triton::ktdp
