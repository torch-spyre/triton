// Declarations for transforms on the `tts` dialect's own abstraction -- its
// marker ops.  One of four Passes.h under third_party/spyre, and the second in a
// dialect's own namespace (mlir::triton::tts, as Dialect/KTDP/Transforms/ is in
// mlir::triton::ktdp) -- it earns the dialect's name because the dialect is its
// subject.  Conversion/TritonToKTIR/ holds the passes that cross a dialect
// boundary and Transforms/ the rest, both in mlir::triton::spyre.  The criterion
// and the per-pass contracts are in the Passes.td beside this file.

#ifndef TRITON_SPYRE_DIALECT_TTS_TRANSFORMS_PASSES_H
#define TRITON_SPYRE_DIALECT_TTS_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace mlir::triton::tts {

#define GEN_PASS_DECL
#include "Dialect/TTS/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Dialect/TTS/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLowerTTSMarkersPass();

/// `grid` is the pass option of the same name; the two byte counts keep their
/// tablegen defaults when not given, so a caller that has only the grid -- which
/// is every caller in the pipeline -- passes only that.
std::unique_ptr<OperationPass<ModuleOp>> createPlacePinnedValuesPass(
    llvm::ArrayRef<int64_t> grid = {});

} // namespace mlir::triton::tts

#endif // TRITON_SPYRE_DIALECT_TTS_TRANSFORMS_PASSES_H
