//===- Utility.cpp - Shared ktdp-building helpers -------------------------===//

#include "Dialect/KTDP/Utils/Utility.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir::triton::ktdp {

bool isLoweredDescriptor(Value desc) {
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp && !castOp.getInputs().empty() &&
         isa<MemRefType>(castOp.getInputs()[0].getType());
}

Value getDescriptorMemView(Value desc) {
  assert(isLoweredDescriptor(desc) &&
         "descriptor operand was not lowered — "
         "precondition check should have caught this");
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp.getInputs()[0];
}

/// Build a range-set constraint for an N-D coordinate space.
/// Static dims use arith constants; dynamic dims use IntegerSet symbols.
/// Declared in the header: the two builders below are its callers here, and a
/// pass that rebuilds views and tiles of its own builds the same set.
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

//===----------------------------------------------------------------------===//
// The layout rewrite's seed traversals
//===----------------------------------------------------------------------===//

void collectViewAccesses(Operation *memView,
                         SmallVectorImpl<Operation *> &loads,
                         SmallVectorImpl<Operation *> &stores) {
  if (!memView || memView->getNumResults() != 1)
    return;
  for (Operation *tile : memView->getResult(0).getUsers()) {
    if (!isa<mlir::ktdp::ConstructAccessTilesOp,
             mlir::ktdp::ConstructIndirectAccessTilesOp>(tile))
      continue;
    for (Operation *user : tile->getResult(0).getUsers()) {
      if (isa<mlir::ktdp::LoadOp>(user))
        loads.push_back(user);
      else if (isa<mlir::ktdp::StoreOp>(user))
        stores.push_back(user);
    }
  }
}

void collectAdjacentGenerics(ArrayRef<Operation *> memViews,
                             SmallVectorImpl<Operation *> &out) {
  SmallPtrSet<Operation *, 8> seen;
  auto note = [&](Operation *op) {
    if (isa_and_nonnull<linalg::GenericOp>(op) && seen.insert(op).second)
      out.push_back(op);
  };
  SmallVector<Operation *> loads, stores;
  for (Operation *view : memViews) {
    loads.clear();
    stores.clear();
    collectViewAccesses(view, loads, stores);
    for (Operation *load : loads)
      for (Operation *consumer : load->getResult(0).getUsers())
        note(consumer);
    for (Operation *store : stores)
      note(cast<mlir::ktdp::StoreOp>(store).getDataTile().getDefiningOp());
  }
}

} // namespace mlir::triton::ktdp
