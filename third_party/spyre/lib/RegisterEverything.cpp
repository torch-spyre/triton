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

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Transforms/Passes.h"
#include "Pipeline.h"
#include "Transforms/Passes.h"

void mlir::triton::spyre::registerPasses() {
  // One call per tablegen'd pass group. A pass left unregistered has no CLI
  // flag, so it vanishes from --help and lit tests driving it fail with an
  // unknown-option error rather than anything that points at the cause.
  // Unqualified names are ours; ktdp:: is the one group whose subject is the
  // dialect, and which keeps its name.
  registerTritonToKTIRPasses();
  ktdp::registerKTDPPasses();
  registerSpyreTransformsPasses();
}

namespace {

// The CLI face of TTIRToKTIRPipelineOptions. Two structs rather than one because
// the members are different kinds of thing: these are llvm::cl objects that parse
// themselves out of a string, and a builder taking them could not be called with
// values.
struct TTIRToKTIRCLIOptions
    : public mlir::PassPipelineOptions<TTIRToKTIRCLIOptions> {
  Option<std::string> dataLayout{
      *this, "data-layout",
      llvm::cl::desc("HBM data layout: 'device' for stickified row-major "
                     "physical strides, 'host' for strides derived from the "
                     "logical ones through the coordinate map"),
      llvm::cl::init("device")};
  ListOption<int64_t> grid{
      *this, "grid",
      llvm::cl::desc("Per-axis partition of the Spyre hardware grid, one entry "
                     "per tl.program_id axis the kernel reads. Empty leaves "
                     "DistributeWork's own default")};
};

struct SpyrecodeCLIOptions
    : public mlir::PassPipelineOptions<SpyrecodeCLIOptions> {
  Option<bool> bindBaseAddresses{
      *this, "bind-base-addresses",
      llvm::cl::desc("Replace the entry function's base-address arguments with "
                     "arith.constant and drop them from the signature, instead "
                     "of leaving them symbolic for the runtime to patch"),
      llvm::cl::init(false)};
  ListOption<int64_t> baseAddresses{
      *this, "base-addresses",
      llvm::cl::desc("The addresses to bind, as element indices, positionally. "
                     "Read only with bind-base-addresses")};
};

} // namespace

void mlir::triton::spyre::registerPipelines() {
  // Named after the compile stages they implement rather than after what they
  // currently contain, because the contents are what moves: passes are expected
  // to cross this boundary, and a name like "logical" or "physical" would have
  // to be re-earned after each move while "the ktir stage" stays true.
  PassPipelineRegistration<TTIRToKTIRCLIOptions>(
      "spyre-ttir-to-ktir",
      "The backend's `ktir` compile stage, whole: Triton IR in, KTIR out.",
      [](OpPassManager &pm, const TTIRToKTIRCLIOptions &cli) {
        TTIRToKTIRPipelineOptions options;
        options.dataLayout = cli.dataLayout;
        options.grid.assign(cli.grid.begin(), cli.grid.end());
        buildTTIRToKTIRPipeline(pm, options);
      });

  // "prepare" and not "to-spyrecode": the pipeline stops at the module dbo-opt
  // is handed, and dbo-opt is what produces SpyreCode.
  PassPipelineRegistration<SpyrecodeCLIOptions>(
      "spyre-prepare-spyrecode",
      "The IR half of the backend's `spyrecode` stage: KTIR in, the module "
      "dbo-opt receives out.",
      [](OpPassManager &pm, const SpyrecodeCLIOptions &cli) {
        SpyrecodePipelineOptions options;
        options.bindBaseAddresses = cli.bindBaseAddresses;
        options.baseAddresses.assign(cli.baseAddresses.begin(),
                                     cli.baseAddresses.end());
        buildSpyrecodePipeline(pm, options);
      });
}

void mlir::triton::spyre::registerDialects(DialectRegistry &registry) {
  // The lowering targets: ktdp and spyreop come from ktir-mlir-frontend, the
  // upstream three are what LowerComputeOps and friends emit. Not the Triton
  // dialect -- that is the host project's, registered by whoever parses tt.
  registry.insert<mlir::ktdp::KtdpDialect, mlir::spyreop::SpyreOpDialect,
                  linalg::LinalgDialect, tensor::TensorDialect,
                  math::MathDialect>();
}
