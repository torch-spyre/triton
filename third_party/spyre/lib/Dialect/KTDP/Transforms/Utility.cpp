//===- Utility.cpp - Shared transform utilities for KTDP passes -----------===//

#include "Dialect/KTDP/Transforms/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
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

} // namespace mlir::triton::ktdp
