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

/// Phase 2: populate greedy rewrite patterns for contraction synthesis.
/// Source patterns (matmul, reduce) run at benefit 2; sink patterns (store)
/// run at benefit 1.
void populateContractionPatterns(mlir::RewritePatternSet &patterns,
                                 const PassContext &ctx);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_CONTRACTIONSYNTHESIS_H
