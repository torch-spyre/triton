// Declarations for transforms on upstream structure or the whole program.  One
// of three Passes.h under third_party/spyre, all in namespace
// mlir::triton::ktdp: this one for passes that are about neither a dialect
// boundary nor ktdp's own abstractions, Conversion/TritonToKTIR/ for the
// former, Dialect/KTDP/Transforms/ for the latter.  The criterion and the
// per-pass contracts are in the Passes.td beside this file.

#ifndef TRITON_SPYRE_TRANSFORMS_PASSES_H
#define TRITON_SPYRE_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace mlir::triton::ktdp {

#define GEN_PASS_DECL
#include "Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createUnaliasLinalgOutsPass();
std::unique_ptr<OperationPass<ModuleOp>> createDropReductionInitFillPass();
std::unique_ptr<OperationPass<ModuleOp>> createMaterializeBaseAddressesPass(
    llvm::ArrayRef<int64_t> baseAddresses = {});

} // namespace mlir::triton::ktdp

#endif // TRITON_SPYRE_TRANSFORMS_PASSES_H
