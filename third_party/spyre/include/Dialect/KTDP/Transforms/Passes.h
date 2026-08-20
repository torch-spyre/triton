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

std::unique_ptr<OperationPass<ModuleOp>> createLowerInterTilePass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertFunctionsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerDescriptorMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerScalarLoadPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerComputeOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createUnaliasLinalgOutsPass();
std::unique_ptr<OperationPass<ModuleOp>> createDropReductionInitFillPass();
std::unique_ptr<OperationPass<ModuleOp>> createDistributeWorkPass(
    llvm::ArrayRef<int64_t> grid = {});
std::unique_ptr<OperationPass<ModuleOp>> createMaterializeBaseAddressesPass(
    llvm::ArrayRef<int64_t> baseAddresses = {});

} // namespace mlir::triton::ktdp

#endif // TRITON_SPYRE_DIALECT_KTDP_TRANSFORMS_PASSES_H
