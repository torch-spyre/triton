//===- Utility.cpp - Dialect-neutral helpers for the Spyre passes ---------===//

#include "Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::triton::spyre {

std::optional<int64_t> getConstantInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt();
  return std::nullopt;
}

void getDescriptorLogicalLayout(triton::MakeTensorDescOp descOp,
                                SmallVectorImpl<int64_t> &sizes,
                                SmallVectorImpl<int64_t> &strides) {
  sizes.clear();
  strides.clear();

  for (Value s : descOp.getShape())
    sizes.push_back(getConstantInt(s).value_or(ShapedType::kDynamic));
  if (sizes.empty()) {
    auto blockType =
        cast<triton::TensorDescType>(descOp.getResult().getType())
            .getBlockType();
    sizes.assign(blockType.getShape().begin(), blockType.getShape().end());
  }

  for (Value s : descOp.getStrides())
    strides.push_back(getConstantInt(s).value_or(ShapedType::kDynamic));
  if (strides.empty()) {
    int64_t stride = 1;
    strides.resize(sizes.size());
    for (int i = sizes.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      // A dynamic extent stops the running product rather than poisoning it: the
      // strides outside it are still the ones the view is built with.
      if (sizes[i] != ShapedType::kDynamic)
        stride *= sizes[i];
    }
  }
}

Value createEmptyTensor(OpBuilder &builder, Location loc,
                        RankedTensorType type) {
  assert(type.hasStaticShape() &&
         "createEmptyTensor requires a statically shaped type; use the "
         "shapeSource overload for a type with dynamic dimensions");
  return tensor::EmptyOp::create(builder, loc, type.getShape(),
                                 type.getElementType());
}

Value createEmptyTensor(OpBuilder &builder, Location loc, RankedTensorType type,
                        Value shapeSource) {
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

} // namespace mlir::triton::spyre
