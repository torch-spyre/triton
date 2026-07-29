//===- Utility.h - Shared transform utilities for KTDP passes -------------===//

#ifndef KTDP_TRANSFORMS_UTILITY_H
#define KTDP_TRANSFORMS_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>

namespace mlir::triton::ktdp {

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

/// Try to extract a compile-time int64 from an SSA value produced by
/// `arith.constant`. Returns std::nullopt if the value is not a
/// materialized constant.
std::optional<int64_t> getConstantInt(Value v);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_UTILITY_H
