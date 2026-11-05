//===- LowerSpyreOps.h - The group shapes LowerSpyreOps lowers ------------===//
//
// LowerSpyreOps' one exported query. Each group shape -- an anchor and the op
// reading it, which the pass lowers together only when they share a body -- is
// one entry in kGroupShapes in LowerSpyreOps.cpp: a match function and the
// rewrite that lowers what it matched. LowerSpyreOps' group pattern, rooted on
// the anchor in a body, and FuseGenerics, before it puts the ops in one body,
// both ask matchLowerableGroup, which tries each shape.
//
// A group has exactly one reader. An anchor read by more than one op matches
// no shape, so FuseGenerics does not group it and LowerSpyreOps does not lower
// it, and the `i1` left behind is reported by whichever pass holds it.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H
#define TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::triton::spyre {

/// The ops a group would lower together: an anchor and every op that reads its
/// result.
struct GroupCandidate {
  /// The op whose result the group consumes: in the producer generic when the
  /// group spans two generics, in the one generic when it does not.
  Operation *anchor;
  /// The generic whose body holds `anchor`. A compare operand that is one of
  /// its block arguments is looked up through it.
  linalg::GenericOp anchorBody;
  /// The ops that read `anchor`'s result, wherever they are: in `anchorBody`
  /// when the group is in one body, in the consumer's body when it spans two
  /// generics.
  SmallVector<Operation *> readers;
};

/// What a group shape found: which ops of the anchor's body are in the group.
struct GroupMatch {
  StringRef shape;                      // e.g. "compare-cast", for the trace
  SmallVector<Operation *> producerOps; // the anchor body's ops the group consumes
};

/// The group shape LowerSpyreOps lowers `candidate` as: the first shape in
/// kGroupShapes that matches its anchor and reader. Failure when `candidate`
/// has other than exactly one reader, or when no shape matches, with `why`,
/// when non-null, set to the reason.
FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why = nullptr);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_TRANSFORMS_LOWERSPYREOPS_H
