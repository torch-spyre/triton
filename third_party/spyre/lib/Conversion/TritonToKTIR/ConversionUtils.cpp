//===- ConversionUtils.cpp - Helpers private to the TTIR -> KTIR passes ---===//

#include "ConversionUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::triton::spyre {

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

} // namespace mlir::triton::spyre
