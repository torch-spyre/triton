#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CLASSIFY_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CLASSIFY_H

#include "RewriteDescriptorLayout/PermutationUtils.h"
#include "RewriteDescriptorLayout/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::ktdp {

/// Assign a role to each physical dim of an operand.
///   >= 0  : parallel dim, maps to output axis [value]
///   -1    : reduction dim
void buildDimRoles(const OperandCoords &coords,
                   llvm::ArrayRef<int64_t> canonicalAxes,
                   llvm::SmallVectorImpl<int64_t> &roles);

/// Classify one operand's physical dims into OperandPlan fields.
OperandPlan classify(Value val, const OperandCoords &coords,
                     llvm::ArrayRef<int64_t> dimRoles);

/// Build a plan for a scratchpad operand (no marker, logical shape).
OperandPlan classifyScratchpad(Value val, const SourceOperandSpec &opSpec);

/// Direction of `resolveOperand`'s transpose: which side of the permutation
/// the op-tile axis order names.
enum class TransposeDirection {
  Narrow, // physical -> op-tile order (source operands feeding an op).
  Widen,  // op-tile -> physical order (a store's data tile).
};

/// Trip counts derived from an operand set's cross-operand agreement, plus
/// whether the operands agree at all.
struct OperandSetTripCounts {
  int64_t stickFactor = 1;
  int64_t parallelFactor = 1;
  int64_t parallelAccAxis = -1;
  int64_t parallelStickSize = 1;
  /// False iff two annotated operands carry parallel floor dims with
  /// different trip counts or on different output axes.
  bool parallelAgrees = true;
};

/// Cross-operand fold over the whole operand set: demotes StickifiedBlock to
/// WholeBlock when no plan has a loop dim, and derives the stick/parallel
/// trip counts from the same loopDims/floorDims fold. Not applicable to a
/// store (its dimRoles are never reduced, so this would be a no-op).
OperandSetTripCounts reconcileOperandSet(llvm::SmallVectorImpl<OperandPlan> &plans);

/// Per-operand resolution: fills `plan.transposePerm` and `plan.opExtents`
/// from `plan.dims.opTileDims` against `targetOrder`, independent of any
/// sibling operand. `direction` selects which side of the permutation
/// `targetOrder` names: Narrow for a source operand feeding an op, Widen for
/// a store's data tile.
void resolveOperand(OperandPlan &plan, llvm::ArrayRef<int64_t> targetOrder,
                    TransposeDirection direction);

/// Per-dim op-tile slice extent.
int64_t opSliceExtent(const OperandPlan &plan, int p);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CLASSIFY_H
