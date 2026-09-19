//===- Pipeline.cpp - The Spyre backend's compile stages as pipelines -----===//
//
// The pass lists, once. Each per-pass ordering constraint is stated by the pass
// itself, in the Passes.td of whichever of the three libraries it belongs to;
// what is recorded here is only what the *sequence* has to get right, which no
// single pass can say.
//
//===----------------------------------------------------------------------===//

#include "Pipeline.h"

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Transforms/Passes.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

void mlir::triton::spyre::buildTTIRToKTIRPipeline(
    OpPassManager &pm, const TTIRToKTIRPipelineOptions &options) {
  // tt.descriptor_load/store/gather/scatter -> ktdp memory ops.
  pm.addPass(createLowerDescriptorMemoryPass());

  // A scalar tt.load (plus its addptr chain) -> a single-element 1-D read.
  // [LowerPointerChainMemory would sit here -- planned, not implemented; it
  // would handle the tensor-of-pointers tt.load this one leaves legal.]
  pm.addPass(createLowerScalarLoadPass());

  // tt.reduce/broadcast/expand_dims/dot -> linalg + tensor, and a dead-op sweep.
  pm.addPass(createLowerComputeOpsPass());

  // tt.inter_tile_reduce -> ktdp.inter_tile_produce + delivery. After
  // LowerComputeOps, because the partials it consumes have to be linalg/tensor
  // by then; before the layout pass, which has no propagation pattern for a
  // !ktdp.tile_future and so must not be reached with one live.
  pm.addPass(createLowerInterTilePass());

  // Logical tensor descriptors -> physical (stick-tiled) layout, from the
  // tt.spyre_tensor_layout annotations. After LowerComputeOps so a tt.dot is
  // already a linalg.matmul before its operands are physicalized.
  pm.addPass(ktdp::createRewriteDescriptorLayout(
      ktdp::RewriteDescriptorLayoutOptions{options.dataLayout}));

  // dbo-opt's scheduler requires the compute to be a linalg op with an
  // unaliased `outs`; without these two it rejects the ktdp.load operand,
  // because the memref keeps a dynamic strided<..., offset: ?> layout until a
  // linalg consumer pins it. They run after the layout pass and not after
  // LowerComputeOps: a linalg.generic built before the descriptor is
  // physicalized carries logical types, the two then disagree, and the pipeline
  // aborts. UnaliasLinalgOuts runs second because what it removes is what
  // ConvertElementwiseToLinalg creates.
  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  pm.addPass(createUnaliasLinalgOutsPass());

  // tt.func/tt.return -> func.func/func.return, !tt.ptr -> index. Last of the
  // conversions, because every memory pass above consumes !tt.ptr arguments
  // through getBasePtrAsIndex, and because LowerInterTile reads work-slice
  // function attributes this rewrites. metadata["name"] and the base-address
  // inference read the module before the pipeline for the same reason.
  pm.addPass(createConvertFunctionsPass());

  // tt.get_program_id -> ktdp.get_compute_tile_id, tt.get_num_programs folded
  // against the grid, which is stamped on the entry function.
  pm.addPass(createDistributeWorkPass(options.grid));

  // Folds muli x,1 and the cast chains the conversions leave behind, and the
  // arith.divsi/remsi the layout pass emits per access-tile index -- the static
  // ones; in a loop-distributed kernel they derive from the tile id and survive,
  // which is correct rather than a leak.
  pm.addPass(mlir::createCanonicalizerPass());
}

void mlir::triton::spyre::buildSpyrecodePipeline(
    OpPassManager &pm, const SpyrecodePipelineOptions &options) {
  // LowerComputeOps gives every reduction a zero linalg.fill on its `outs` per
  // upstream linalg semantics, and the scheduler's allowlist is
  // add/mul/sub/reduce. Device-only in both senses the stage's rule names: it
  // admits addf/subf alone -- the scheduler resets an accumulator to zero
  // whatever the combiner is, so mul and max/min would get the wrong answer and
  // are refused rather than silently lowered -- so in the `ktir` stage it would
  // error on a max reduce that otherwise lowers and runs on ktir_cpu. And a
  // reduce stripped of its neutral element is correct only given that same
  // zero-reset guarantee, which no KTIR reader can see.
  pm.addPass(createDropReductionInitFillPass());

  // Scalar math/arith (math.sqrt/exp/rsqrt, arith.divf, arith.addi/muli inside
  // a linalg.generic body) -> the spyreop spelling the scheduler expects. After
  // the elementwise-to-linalg conversion in the `ktir` stage, so the op it
  // matches is already inside a linalg.generic body.
  pm.addPass(createLowerSpyreOpsPass());

  if (options.bindBaseAddresses) {
    // The one genuine choice in this stage: symbolic and bound are real
    // argument-passing modes, not a repair. Needs ConvertFunctions to have run,
    // which it has -- that is in the `ktir` stage.
    pm.addPass(createMaterializeBaseAddressesPass(options.baseAddresses));
    // Part of the materialization rather than of the stage: they fold the
    // arith.constant addresses it just introduced into their users. Symbolic
    // mode has no constants to fold.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
}
