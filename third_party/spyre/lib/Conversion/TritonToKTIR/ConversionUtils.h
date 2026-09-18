//===- ConversionUtils.h - Helpers private to the TTIR -> KTIR passes -----===//
//
// Not under include/: nothing outside this library uses either helper, and both
// exist because these passes convert in place -- one leaves the bridge cast a
// later pass in the same library consumes, the other sweeps what the rewrite
// orphaned.  Deliberately unpublished, so the surface stays what other
// libraries actually need.

#ifndef TRITON_SPYRE_CONVERSION_TRITONTOKTIR_CONVERSIONUTILS_H
#define TRITON_SPYRE_CONVERSION_TRITONTOKTIR_CONVERSIONUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::triton::spyre {

/// Erase trivially dead ops in reverse walk order.
/// An op is erased only if BOTH conditions hold:
///   1. predicate(op) is true (or predicate is null — matches all ops)
///   2. isOpTriviallyDead(op) — terminators, symbols, and side-effecting ops
///      are never considered dead regardless of the predicate.
void cleanupDeadOps(ModuleOp module,
                    llvm::function_ref<bool(Operation *)> predicate = nullptr);

/// Cast a `!tt.ptr` value to `index` via an `unrealized_conversion_cast`.
/// The cast survives the memory-lowering passes and is consumed by the
/// later `ConvertFunctions` pass, which rewrites `!tt.ptr` function
/// arguments to `index` and erases the matching casts. A no-op if `basePtr`
/// is already `index`-typed.
Value getBasePtrAsIndex(OpBuilder &builder, Location loc, Value basePtr);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_CONVERSION_TRITONTOKTIR_CONVERSIONUTILS_H
