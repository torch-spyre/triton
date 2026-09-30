//===- Utility.h - Dialect-neutral helpers for the Spyre passes -----------===//
//
// Helpers whose subject is no dialect of ours: they read and build upstream
// arith/tensor ops only.  Hence namespace mlir::triton::spyre, and hence a
// top-level Utils/ rather than a home under Dialect/.  Shared, so they live
// here instead of being private to a pass library -- each has a consumer in
// two different libraries, and duplicating createEmptyTensor is exactly the
// drift a shared home prevents.

#ifndef TRITON_SPYRE_UTILS_UTILITY_H
#define TRITON_SPYRE_UTILS_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <optional>

namespace mlir::triton::spyre {

/// Try to extract a compile-time int64 from an SSA value produced by
/// `arith.constant`. Returns std::nullopt if the value is not a
/// materialized constant.
std::optional<int64_t> getConstantInt(Value v);

/// The logical (host) sizes and strides a `tt.make_tensor_descriptor` describes,
/// in elements, with `ShapedType::kDynamic` wherever the operand is not a
/// materialized constant.
///
/// This is `LowerDescriptorMemory`'s rule and it is published rather than
/// restated because a second reader has appeared: `SpyreBackend` records each
/// annotated descriptor's device footprint in the compiled metadata, and that
/// footprint is a function of these same logical extents. Derived twice, the
/// footprint a launcher bounds-checks against and the memory view the IR is
/// built with could disagree -- and the disagreement would be silent in the
/// dangerous direction, because an under-claim is an unchecked overrun.
///
/// Two fallbacks come with the rule, and both are why this cannot be replaced by
/// "read the descriptor's block type" or "assume row-major":
///   - no `shape` operands at all: the block type's shape is the tensor's;
///   - no `strides` operands at all: row-major over `sizes`, which stops
///     multiplying at the first dynamic extent.
void getDescriptorLogicalLayout(triton::MakeTensorDescOp descOp,
                                SmallVectorImpl<int64_t> &sizes,
                                SmallVectorImpl<int64_t> &strides);

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
Value createEmptyTensor(OpBuilder &builder, Location loc, RankedTensorType type,
                        Value shapeSource);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_UTILS_UTILITY_H
