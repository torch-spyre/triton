#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H

#include "RewriteDescriptorLayout/Types.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::ktdp {

/// True iff op has exactly one result and exactly one RankedTensorType
/// operand among all its operands. linalg.transpose has two tensor operands
/// (input + init/outs), so this is false for it — transpose is handled as a
/// separate, explicit candidate rather than folded into "elementwise".
bool isSingleTensorElementwiseOp(mlir::Operation *op);

/// Where a value's chain ends up in memory: the layout marker of the
/// ktdp.store it feeds (through single-tensor elementwise ops), together with
/// that store's access-tile shape — already physical, since Phase 1 repointed
/// every annotated store at its physical tile. Both empty when the walk finds
/// no store, or finds one whose access tile has no marker (an unannotated
/// output, which is what leaves a result logical).
///
/// Exported because Phase 2A needs it: whether a linalg.reduce's result is
/// physical is a question about the layout it is *stored under*, so the
/// propagation rule has to look forward to the store, exactly as
/// RewriteStorePattern looks backward from it.
struct StoreDestination {
  triton::SpyreTensorLayoutOp marker;
  llvm::SmallVector<int64_t> tileShape;
};
/// Takes Phase 1's marker map rather than the PassContext, so a Phase 2A caller
/// cannot reach Phase 2B state through it: `physicalValues` is a non-const
/// reference member, so a `const PassContext &` would still permit writing the
/// map Phase 2B owns.
StoreDestination findStoreDestination(mlir::Value value,
                                      const MarkerByMemView &markers);

/// True iff a DPS init operand can be rebuilt at a different (physical) shape
/// without losing information — see rebuildPhysicalInit for what that means and
/// which producers qualify. Asked by Phase 2A before it commits a reduce to the
/// Physical output-axis space, so the decision and the emission cannot disagree.
bool canRebuildPhysicalInit(mlir::Value init);

/// Phase 2: populate greedy rewrite patterns for contraction synthesis.
/// Source patterns (matmul, reduce) run at benefit 2; sink patterns (store)
/// run at benefit 1.
void populateContractionPatterns(mlir::RewritePatternSet &patterns,
                                 const PassContext &ctx);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
