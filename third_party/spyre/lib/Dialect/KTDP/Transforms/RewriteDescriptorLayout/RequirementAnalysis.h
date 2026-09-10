#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_REQUIREMENTANALYSIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_REQUIREMENTANALYSIS_H

#include "RewriteDescriptorLayout/Types.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <memory>

namespace mlir::triton::ktdp {

//===----------------------------------------------------------------------===//
// The backward half of Phase 2A: for every value it can reach, the layout some
// ktdp.store WANTS of it. The forward analysis (PhysicalTypeAnalysis.h) answers
// "what layout does this value have"; this one answers "what layout is wanted
// of it". Design, rule table and open questions: "a backward requirement
// analysis" in docs/spyre-tensor-layouts.md.
//
// This runs BEFORE the forward analysis, which consumes it: ReducePropagation
// compares what its operand's layout induces against the requirement at its
// result. Nothing here reads a forward fact, so the order is an ordering and not
// a cycle. Mutates no IR.
//===----------------------------------------------------------------------===//

/// The layout wanted of one value. Total, never partial: a requirement is
/// consumed at the op that can satisfy it rather than crossed through it, so no
/// dim slot is ever free (see "A requirement is total" in the doc).
///
/// The four arrays are what a marker plus its store's access-tile shape already
/// carry, held as data rather than read off the marker op because
/// `linalg.transpose` reorders them relative to the marker. That is why
/// ReducePropagation's comparison is arrays-against-arrays.
struct LayoutRequirement {
  llvm::SmallVector<int64_t> physSrc;
  llvm::SmallVector<int64_t> physOp;
  llvm::SmallVector<int64_t> physArg;
  /// Physical extents: the store's access-tile shape, already physical because
  /// Phase 1 repointed every annotated store at its physical tile.
  llvm::SmallVector<int64_t> physExtents;
  /// The store's marker this requirement descends from. Always set.
  triton::SpyreTensorLayoutOp marker;

  bool operator==(const LayoutRequirement &o) const {
    return marker == o.marker && physSrc == o.physSrc && physOp == o.physOp &&
           physArg == o.physArg && physExtents == o.physExtents;
  }
};

using RequirementMap = llvm::DenseMap<mlir::Value, LayoutRequirement>;

/// One op's rule for what a requirement on its result requires of its operands.
///
/// The contract: given `req` at `result`, produce the requirement `operand`
/// must satisfy. Failure means nothing crosses to that operand -- either the
/// rule consumes the requirement (reduce, matmul) or the op cannot carry one
/// across at all (reshape, broadcast). Not an error either way.
///
/// As on the forward side, an op no pattern matches is reported rather than
/// defaulted: teaching this pass a new op means adding a pattern here, never
/// editing a central dispatch.
class RequirementBackwardPattern {
public:
  virtual ~RequirementBackwardPattern() = default;

  /// True iff this pattern is the rule for `op`.
  virtual bool match(Operation *op) const = 0;

  /// The requirement induced on `operand`, one of `op`'s tensor operands.
  virtual llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const = 0;
};

using RequirementBackwardPatternSet =
    llvm::SmallVector<std::unique_ptr<RequirementBackwardPattern>>;

/// Register the backward rule for every op this pass has been taught, mirroring
/// populatePhysicalPropagationPatterns' shape. Pure functions of the op and the
/// incoming requirement -- unlike the forward side there is no marker map to
/// hand in, which is the point of getting the direction right.
void populateRequirementBackwardPatterns(
    RequirementBackwardPatternSet &patterns);

/// The analysis result.
struct RequirementAnalysis {
  /// One entry per value a requirement reaches. Absence means no store wants a
  /// particular layout of the value.
  RequirementMap requirements;
  /// Values two requirements reached that do not agree. Detected and reported,
  /// never resolved: `requirements` keeps the first and the second is dropped,
  /// so nothing downstream may read an entry for a conflicted value. The
  /// eventual answer is logical plus bridges -- see the doc's owed answers.
  llvm::DenseSet<mlir::Value> conflicts;
  /// Ops a requirement reached that no pattern is the rule for. The requirement
  /// stops there; kept so the gap is countable rather than invisible.
  llvm::SmallVector<Operation *> opsWithNoRule;
};

/// Push `req` onto `value` and, if it is new there, onto its producer's operands
/// through that producer's backward rule. Mutates no IR.
///
/// `visitStack` holds every value currently being propagated through, the same
/// contract getPhysicalizedType's invocationStack states: recording into
/// `requirements` before recursing already cuts a revisit, and this makes a
/// cycle in the value graph (an scf.for iter_arg closes one) fail visibly
/// instead of resting on that.
void propagateRequirement(Value value, const LayoutRequirement &req,
                          const RequirementBackwardPatternSet &patterns,
                          RequirementAnalysis &result,
                          llvm::SmallVector<Value> &visitStack);

/// Run the backward analysis over `module`, seeded from every `ktdp.store`
/// whose access tile Phase 1 physicalized. Mutates no IR.
RequirementAnalysis runRequirementAnalysis(ModuleOp module,
                                           const PassContext &ctx);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_REQUIREMENTANALYSIS_H
