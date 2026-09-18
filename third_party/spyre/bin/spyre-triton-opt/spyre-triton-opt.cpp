#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"
#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Transforms/Passes.h"
#include "Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::triton::registerTritonPasses();
  // One call per tablegen'd pass group. All three must be here: a pass left
  // unregistered has no CLI flag, so it disappears from --help and every lit
  // test driving it fails with an unknown-option error rather than anything
  // that points at the cause.
  mlir::triton::ktdp::registerTritonToKTIRPasses();
  mlir::triton::ktdp::registerKTDPPasses();
  mlir::triton::ktdp::registerSpyreTransformsPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect>();
  registry.insert<mlir::ktdp::KtdpDialect>();
  registry.insert<mlir::spyreop::SpyreOpDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Spyre+Triton optimizer driver\n", registry));
}
