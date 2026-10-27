//===- LowerSpyreOps.h - The group shapes LowerSpyreOps lowers ------------===//
//
// LowerSpyreOps' one exported query. Each group shape -- a set of ops that the
// pass lowers to one intrinsic only when they share a body -- is defined once,
// in LowerSpyreOps.cpp, and is used both by the selection pattern that lowers it
// and by this function, which FuseGenerics asks before putting the ops in one
// body.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H
#define TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <string>

namespace mlir::triton::spyre {

/// True iff `anchor`, the op whose result `anchorBody` yields, and `reader`,
/// an op that reads that result, would form a group LowerSpyreOps lowers as one
/// intrinsic if they were in one body. `reader` may be in another generic,
/// reading the result through a block argument. When false and `why` is
/// non-null, sets it to the reason.
bool formsLowerableGroup(linalg::GenericOp anchorBody, Operation *anchor,
                         Operation *reader, std::string *why = nullptr);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H
