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
///
/// `space` says what an output axis IS (see OutputAxisSpace), and therefore how
/// the roles are numbered. `canonicalAxes` is what says *whether* a dim survives
/// in either space; only the numbering changes.
///
///   Logical  — role = `canonicalAxes[coords.src[p]]`, the output's logical
///     axis. Many-to-one: the two physical dims of a stick-split logical axis
///     share one role, and the split is expressed elsewhere (a scatter dim plus
///     a loop, or the store's widen stage).
///   Physical — role = the position of `p` among the operand's surviving
///     physical dims, ascending. One-to-one, so a stick-split surviving axis
///     gets two roles — its stick index and its lane — which is what lets the
///     accumulator carry both.
///
/// Roles stay unique within an operand in both spaces, which is what
/// computeTransposePerm relies on.
void buildDimRoles(const OperandCoords &coords,
                   llvm::ArrayRef<int64_t> canonicalAxes,
                   OutputAxisSpace space,
                   llvm::SmallVectorImpl<int64_t> &roles);

/// Classify one operand's physical dims into OperandPlan fields. `space` must
/// be the one `dimRoles` was built in — it is what decides whether a surviving
/// stick-index dim is a scatter dim (Logical) or an op-tile dim (Physical).
OperandPlan classify(Value val, const OperandCoords &coords,
                     llvm::ArrayRef<int64_t> dimRoles,
                     OutputAxisSpace space);

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
  /// False iff two annotated operands carry multi-stick scatter dims with
  /// different trip counts or on different output axes.
  bool parallelAgrees = true;
};

/// Cross-operand fold over the whole operand set: demotes StickifiedBlock to
/// WholeBlock when no plan has a reduce loop dim, and derives the
/// stick/parallel trip counts from the same reduceLoopDims/scatterDims fold.
/// Not applicable to a store (its dimRoles are never reduced, so this would be
/// a no-op).
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
