//===- RegisterEverything.h - Spyre backend registration -----------------===//
//
// The single place the Spyre backend's dialects and passes are named. Adding a
// pass library means editing RegisterEverything.cpp and its LINK_LIBS, not
// every consumer.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_REGISTEREVERYTHING_H
#define TRITON_SPYRE_REGISTEREVERYTHING_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::triton::spyre {

/// Registers every Spyre pass with the pass registry, giving each a CLI flag.
/// For `spyre-triton-opt`; the pybind plugin builds pipelines from create*
/// factories instead and does not call this.
void registerPasses();

/// Registers the dialects the Spyre passes read and write. Enough on its own
/// for a context that only runs the Spyre pipeline; `spyre-triton-opt` adds
/// upstream MLIR and the Triton dialect on top.
void registerDialects(DialectRegistry &registry);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_REGISTEREVERYTHING_H
