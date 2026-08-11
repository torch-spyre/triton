//===- Utility.cpp - Shared transform utilities for KTDP passes -----------===//

#include "Dialect/KTDP/Transforms/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::triton::ktdp {

void cleanupDeadOps(ModuleOp module,
                    llvm::function_ref<bool(Operation *)> predicate) {
  module.walk([&](Block *block) {
    for (auto it = block->rbegin(); it != block->rend();) {
      Operation &op = *it++;
      if ((!predicate || predicate(&op)) && isOpTriviallyDead(&op))
        op.erase();
    }
  });
}

Value getBasePtrAsIndex(OpBuilder &builder, Location loc, Value basePtr) {
  if (basePtr.getType().isIndex())
    return basePtr;
  return UnrealizedConversionCastOp::create(builder, loc,
                                            builder.getIndexType(), basePtr)
      .getResult(0);
}

std::optional<int64_t> getConstantInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt();
  return std::nullopt;
}

Value createEmptyTensor(OpBuilder &builder, Location loc,
                        RankedTensorType type) {
  assert(type.hasStaticShape() &&
         "createEmptyTensor requires a statically shaped type; use the "
         "shapeSource overload for a type with dynamic dimensions");
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType());
}

Value createEmptyTensor(OpBuilder &builder, Location loc,
                        RankedTensorType type, Value shapeSource) {
  auto sourceType = cast<RankedTensorType>(shapeSource.getType());
  assert(sourceType.getRank() == type.getRank() &&
         "shapeSource must have the same rank as the type being built");
  (void)sourceType;

  // One size operand per dynamic dimension, in dimension order — that is the
  // order tensor.empty matches them to the `?`s in its result type.
  SmallVector<Value> dynSizes;
  for (int64_t dim = 0, rank = type.getRank(); dim < rank; ++dim)
    if (type.isDynamicDim(dim))
      dynSizes.push_back(tensor::DimOp::create(builder, loc, shapeSource, dim));

  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType(), dynSizes);
}

/// True iff `desc` is a memref-backed lowered descriptor (the
/// UnrealizedConversionCast bridge left by LowerDescriptorMemory Walk 1).
bool isLoweredDescriptor(Value desc) {
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp && !castOp.getInputs().empty() &&
         isa<MemRefType>(castOp.getInputs()[0].getType());
}

/// Unwrap the bridge cast to recover the ktdp.construct_memory_view result.
Value getDescriptorMemView(Value desc) {
  assert(isLoweredDescriptor(desc) &&
         "descriptor operand was not lowered — "
         "precondition check should have caught this");
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp.getInputs()[0];
}

/// Build a range-set constraint for an N-D coordinate space.
/// Static dims use arith constants; dynamic dims use IntegerSet symbols.
IntegerSet buildRangeSetND(MLIRContext *ctx, ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  unsigned symCount = 0;
  for (auto s : shape)
    if (s == ShapedType::kDynamic)
      ++symCount;

  SmallVector<AffineExpr> constraints;
  SmallVector<bool> eqFlags;
  unsigned symIdx = 0;
  for (unsigned i = 0; i < rank; ++i) {
    auto di = getAffineDimExpr(i, ctx);
    AffineExpr upper;
    if (shape[i] == ShapedType::kDynamic)
      upper = getAffineSymbolExpr(symIdx++, ctx) - 1;
    else
      upper = getAffineConstantExpr(shape[i] - 1, ctx);
    constraints.push_back(di);
    eqFlags.push_back(false);
    constraints.push_back(upper - di);
    eqFlags.push_back(false);
  }
  return IntegerSet::get(rank, symCount, constraints, eqFlags);
}

} // namespace mlir::triton::ktdp
