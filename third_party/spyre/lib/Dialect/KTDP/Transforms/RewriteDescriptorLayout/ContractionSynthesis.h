#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H

#include "RewriteDescriptorLayout/Types.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::ktdp {

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
