//===- LowerSpyreOps.h - The group shapes LowerSpyreOps lowers ------------===//
//
// LowerSpyreOps' one exported query. Each group shape -- a set of ops that the
// pass lowers to one intrinsic only when they share a body -- is one match
// function in LowerSpyreOps.cpp, listed once in kGroupShapes there. The
// selection patterns that lower a group and FuseGenerics, before it puts the
// ops in one body, both ask matchLowerableGroup, which tries each shape.
//
// One shape covers every reader of the producer's result. A shape receives
// every reader in the candidate and matches only when it covers every one of
// them; a result whose readers form different shapes matches no shape, so
// FuseGenerics does not group it and reports the `i1` left between the
// generics.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H
#define TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::triton::spyre {

/// The ops a group would put in one body: the producer's ops the group
/// consumes, and the consumer's ops that read the producer's result.
///
/// Two forms:
///
///   - across two generics, as FuseGenerics builds it: `producerResult` is a
///     result of `producer`, the anchor is the op `producer` yields for it,
///     and `readers` are every op in `consumer`'s body reading the matching
///     block argument;
///   - in one body, as the selection patterns build it: `producer` and
///     `consumer` are the same generic, `producerResult` is the anchor's own
///     result, and `readers` is the one op the pattern is rooted on.
struct GroupCandidate {
  linalg::GenericOp producer;
  OpResult producerResult;              // the producer result the consumer reads
  linalg::GenericOp consumer;
  SmallVector<Operation *> readers;     // the consumer ops reading it
};

/// What a group shape found: which of the producer's ops are in the group.
struct GroupMatch {
  StringRef shape;                      // e.g. "compare-cast", for the trace
  SmallVector<Operation *> producerOps; // the producer ops the group consumes
};

/// The group shape LowerSpyreOps lowers `candidate` as: the first shape in
/// kGroupShapes that covers every reader in `candidate`. Failure when none
/// does, with `why`, when non-null, set to the reason; when every reader forms
/// some shape alone, the reason says the readers form different shapes and no
/// single shape covers every reader.
FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why = nullptr);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H
