//===- Utility.h - Shared transform utilities for KTDP passes -------------===//

#ifndef KTDP_TRANSFORMS_UTILITY_H
#define KTDP_TRANSFORMS_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>

namespace mlir::triton::ktdp {

/// True iff `desc` is a memref-backed lowered descriptor (the
/// UnrealizedConversionCast bridge left by LowerDescriptorMemory).
bool isLoweredDescriptor(Value desc);

/// Unwrap the bridge cast to recover the ktdp.construct_memory_view result.
Value getDescriptorMemView(Value desc);

/// Build a range-set constraint for an N-D coordinate space.
/// Static dims use arith constants; dynamic dims use IntegerSet symbols.
IntegerSet buildRangeSetND(MLIRContext *ctx, ArrayRef<int64_t> shape);

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

/// Build a `tensor.empty` of `type` — an uninitialized tensor used as the
/// `outs` operand of a destination-passing-style op, where it supplies the
/// result shape but no initial element values.
///
/// `type` must be statically shaped. For a type with dynamic dimensions, use
/// the overload below, which takes the value to measure them from.
Value createEmptyTensor(OpBuilder &builder, Location loc,
                        RankedTensorType type);

/// Build a `tensor.empty` of `type`, taking the extent of each dynamic
/// dimension from the corresponding dimension of `shapeSource` via
/// `tensor.dim`.
///
/// `tensor.empty` needs one size operand per dynamic dimension, so a type such
/// as `tensor<?x4xf32>` cannot be built from the type alone. `shapeSource` must
/// have the same rank as `type` and must dominate the insertion point; passing
/// a value the op being rewritten already uses satisfies both by construction.
/// Statically sized dimensions are taken from `type` and no `tensor.dim` is
/// emitted for them, so a fully static `type` produces the same op as the
/// overload above.
Value createEmptyTensor(OpBuilder &builder, Location loc,
                        RankedTensorType type, Value shapeSource);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_UTILITY_H
