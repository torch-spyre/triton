#ifndef TRITON_SPYRE_DIALECT_KTDP_TRANSFORMS_PASSES_H
#define TRITON_SPYRE_DIALECT_KTDP_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace mlir::triton::ktdp {

#define GEN_PASS_DECL
#include "Dialect/KTDP/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Dialect/KTDP/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLowerInterTilePass(); // --- added for spyre
std::unique_ptr<OperationPass<ModuleOp>> createConvertFunctionsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerDescriptorMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerScalarLoadPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerComputeOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createDistributeWorkPass(
    llvm::ArrayRef<int64_t> grid = {});

} // namespace mlir::triton::ktdp

#endif // TRITON_SPYRE_DIALECT_KTDP_TRANSFORMS_PASSES_H
