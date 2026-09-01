#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H

#include "RewriteDescriptorLayout/Types.h"
#include "mlir/IR/PatternMatch.h"

#include <string>

namespace mlir::triton::ktdp {

/// Phase 2: populate greedy rewrite patterns for contraction synthesis.
/// Source patterns (matmul, reduce) run at benefit 2; sink patterns (store)
/// run at benefit 1.
void populateContractionPatterns(mlir::RewritePatternSet &patterns,
                                 const PassContext &ctx);

/// True if `op` is a consumer that Phase 2 has a pattern for.
///
/// Use this to pick the ops handed to the greedy driver, rather than writing a
/// second `isa<>` list: both this and `populateContractionPatterns` are derived
/// from one declaration of the supported set, so they cannot fall out of step
/// when a pattern is added.
///
/// This is the set Phase 2 can REPAIR, which is narrower than the set Phase 1
/// DEFERS (see `isContractionOp` in RewriteDescriptorLayout.cpp). The two need
/// not be kept in sync: Phase 3 finds unrepaired ops by re-checking the final
/// IR, not by differencing the two predicates.
bool isSupportedConsumer(mlir::Operation *op);

/// Comma-separated names of the ops Phase 2 can repair, for diagnostics.
/// Derived from the same declaration as `isSupportedConsumer`, so a diagnostic
/// never advertises a stale list.
std::string supportedConsumerNames();

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
