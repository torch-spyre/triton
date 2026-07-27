#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "Ktdp/KtdpDialect.hpp"
#include "Dialect/KTDP/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::ktdp::registerKTDPPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect>();
  registry.insert<mlir::ktdp::KtdpDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Spyre+Triton optimizer driver\n", registry));
}
