//===- RegisterEverything.cpp - Spyre backend registration ---------------===//
//
// Whatever is named here must also be a LINK_LIBS entry of this library; the
// build will not tell you otherwise, since these are OBJECT libraries.
//
//===----------------------------------------------------------------------===//

#include "RegisterEverything.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"

#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Transforms/Passes.h"
#include "Transforms/Passes.h"

void mlir::triton::spyre::registerPasses() {
  // One call per tablegen'd pass group. A pass left unregistered has no CLI
  // flag, so it vanishes from --help and lit tests driving it fail with an
  // unknown-option error rather than anything that points at the cause.
  ktdp::registerTritonToKTIRPasses();
  ktdp::registerKTDPPasses();
  ktdp::registerSpyreTransformsPasses();
}

void mlir::triton::spyre::registerDialects(DialectRegistry &registry) {
  // The lowering targets: ktdp and spyreop come from ktir-mlir-frontend, the
  // upstream three are what LowerComputeOps and friends emit. Not the Triton
  // dialect -- that is the host project's, registered by whoever parses tt.
  registry.insert<mlir::ktdp::KtdpDialect, mlir::spyreop::SpyreOpDialect,
                  linalg::LinalgDialect, tensor::TensorDialect,
                  math::MathDialect>();
}
