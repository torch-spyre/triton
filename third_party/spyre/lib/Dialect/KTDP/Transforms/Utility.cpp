//===- Utility.cpp - Shared transform utilities for KTDP passes -----------===//

#include "Dialect/KTDP/Transforms/Utility.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
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
  if (constraints.empty()) {
    // Rank 0: there are no dims to constrain, but a genuinely empty
    // constraint list is not constructible — IntegerSet::get derives its
    // owning context from constraints[0], which would index past the end
    // of an empty array. Use a single always-true `0 >= 0` constraint,
    // matching the single point of a rank-0 coordinate space (this is the
    // same constraint LowerScalarLoad.cpp's now-removed `trivialIntegerSet`
    // built by hand, before it was the only rank-0 caller of this logic).
    constraints.push_back(getAffineConstantExpr(0, ctx));
    eqFlags.push_back(false);
  }
  return IntegerSet::get(rank, symCount, constraints, eqFlags);
}

Value buildMemoryView(OpBuilder &builder, Location loc, Value baseIndex,
                      ArrayRef<int64_t> staticSizes,
                      ArrayRef<int64_t> staticStrides, ValueRange dynSizes,
                      ValueRange dynStrides, Type elemType,
                      mlir::ktdp::MemorySpaceAttr memorySpace) {
  MLIRContext *ctx = builder.getContext();
  auto memrefType = MemRefType::get(staticSizes, elemType);
  auto memView = mlir::ktdp::ConstructMemoryViewOp::create(
      builder, loc, memrefType, baseIndex, dynSizes, dynStrides, staticSizes,
      staticStrides, memorySpace,
      IntegerSetAttr::get(buildRangeSetND(ctx, staticSizes)));
  return memView.getResult();
}

Value buildAccessTile(OpBuilder &builder, Location loc, Value memView,
                      ArrayRef<int64_t> blockShape, ValueRange indices) {
  MLIRContext *ctx = builder.getContext();
  auto indexType = builder.getIndexType();
  auto accessTileType = mlir::ktdp::AccessTileType::get(blockShape, indexType);
  unsigned rank = blockShape.size();
  auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

  // Cast index operands to index type — they arrive as i32 from Triton in
  // the descriptor paths, but are already index-typed in some callers (e.g.
  // a rescaled loop IV), so only cast when actually needed.
  SmallVector<Value> indexOperands;
  for (auto idx : indices) {
    if (idx.getType() != indexType)
      idx = arith::IndexCastOp::create(builder, loc, indexType, idx);
    indexOperands.push_back(idx);
  }

  auto accessTile = mlir::ktdp::ConstructAccessTilesOp::create(
      builder, loc, accessTileType, memView, identityMap, indexOperands,
      /*symbol_operands=*/ValueRange{}, buildRangeSetND(ctx, blockShape),
      identityMap);
  return accessTile.getResult();
}

} // namespace mlir::triton::ktdp
