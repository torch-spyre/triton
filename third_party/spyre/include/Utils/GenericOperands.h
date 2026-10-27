//===- GenericOperands.h - Checks and rewrites on linalg.generic operands -===//
//
// Plain functions shared by the passes that rewrite a linalg.generic's operand
// list: CanonicalizeGenericOperands and FuseGenerics. Upstream linalg only.

#ifndef TRITON_SPYRE_UTILS_GENERICOPERANDS_H
#define TRITON_SPYRE_UTILS_GENERICOPERANDS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#include <string>
#include <utility>

namespace mlir::triton::spyre {

/// D3-D4 on a consumer's operand list as it would be after a rewrite: every
/// loop dim named by some map as a bare dim, and no value in both ins and
/// outs. Fails with `why` set to the first check that failed.
LogicalResult checkOperandRewrite(ArrayRef<std::pair<Value, AffineMap>> ins,
                                  ArrayRef<std::pair<Value, AffineMap>> outs,
                                  unsigned numLoops, std::string &why);

/// Ported from mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp,
/// FoldScalarOrSplatConstant: reduced to the splat dense tensor case and
/// rewritten in place.
///
/// When `operand`, an ins of `generic`, is defined by an arith.constant of a
/// splat dense int-or-float tensor, replaces its block argument by a scalar
/// arith.constant above `generic` and erases the operand, its indexing map and
/// its block argument. Fails, changing nothing, when `operand` is not such a
/// constant or when the remaining maps leave some loop dim unnamed.
LogicalResult foldSplatConstantOperand(RewriterBase &rewriter,
                                       linalg::GenericOp generic,
                                       OpOperand *operand);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_UTILS_GENERICOPERANDS_H
