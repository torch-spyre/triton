// Declarations for the TTIR -> KTIR conversions.  One of three Passes.h under
// third_party/spyre, all in namespace mlir::triton::ktdp: this one for passes
// that cross a dialect boundary, Dialect/KTDP/Transforms/ for those whose
// subject is ktdp's own abstractions, Transforms/ for the rest.  The criterion
// and the per-pass contracts are in the Passes.td beside this file.

#ifndef TRITON_SPYRE_CONVERSION_TRITONTOKTIR_PASSES_H
#define TRITON_SPYRE_CONVERSION_TRITONTOKTIR_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace mlir::triton::ktdp {

#define GEN_PASS_DECL
#include "Conversion/TritonToKTIR/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/TritonToKTIR/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLowerDescriptorMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerScalarLoadPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerComputeOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertFunctionsPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerInterTilePass();
std::unique_ptr<OperationPass<ModuleOp>> createDistributeWorkPass(
    llvm::ArrayRef<int64_t> grid = {});

} // namespace mlir::triton::ktdp

#endif // TRITON_SPYRE_CONVERSION_TRITONTOKTIR_PASSES_H
