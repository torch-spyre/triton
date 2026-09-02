#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PHYSICALTYPEANALYSIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PHYSICALTYPEANALYSIS_H

#include "RewriteDescriptorLayout/Types.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <memory>

namespace mlir::triton::ktdp {

//===----------------------------------------------------------------------===//
// Phase 2A: decide the final physical type of every value reachable from
// Phase 1's roots, before any IR is rewritten.
//
// Analysis only: it mutates no IR, creates no ops, and does not touch
// ctx.physicalValues. Its output is a map from value to the physical type that
// value ends up carrying once the rewrite has run.
//
// Structure follows One-Shot Bufferization: an analysis over SSA use-def
// chains first, an IR rewrite second. See
// BufferizableOpInterface::getBufferType for the invocationStack contract this
// mirrors.
//===----------------------------------------------------------------------===//

/// What one value's physicalization resolves to.
struct PhysicalTypeInfo {
  /// The final physical type of the value. Always a RankedTensorType.
  Type type;
  /// The layout marker the value's chain originates from. Always set: every
  /// entry descends from a Phase-1 root, which is seeded from a marker.
  triton::SpyreTensorLayoutOp marker;
  /// Accumulated permutation of every linalg.transpose on this value's chain,
  /// composed in chain order; empty if none. Mirrors
  /// PhysicalValueInfo::transposePerm, which Phase 2 fills incrementally as
  /// it erases transposes -- here it is known up front instead.
  llvm::SmallVector<int64_t> transposePerm;
};

/// One op's rule for how physicality flows through it.
///
/// The contract: given `op` and the resolved info of one of its physical
/// tensor operands, produce the info the op's own result carries. Returning
/// failure means "no physical result" -- either the pattern does not handle
/// this op (`match` said no) or the op genuinely ends the chain.
///
/// Each pattern computes its own result type; there is no shared category a
/// caller could switch on. An op no pattern matches is reported and treated as
/// producing nothing physical, so teaching this pass a new op means adding a
/// pattern here -- never editing a central dispatch.
class PhysicalPropagationPattern {
public:
  virtual ~PhysicalPropagationPattern() = default;

  /// True iff this pattern is the rule for `op`.
  virtual bool match(Operation *op) const = 0;

  /// The physical info `result` carries, given `src`/`srcInfo` -- one of the
  /// op's tensor operands already resolved to a physical type. Failure means
  /// the op produces no physical result.
  virtual llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const = 0;
};

using PhysicalPropagationPatternSet =
    llvm::SmallVector<std::unique_ptr<PhysicalPropagationPattern>>;

/// Register the propagation rule for every op this pass has been taught,
/// mirroring populateContractionPatterns' shape.
void populatePhysicalPropagationPatterns(PhysicalPropagationPatternSet &patterns);

// PhysicalTypeMap -- one entry per value reachable from a Phase-1 root whose
// physicalization resolves; a value absent from the map is logical, either
// never reachable from a root or produced by an op that ends the chain -- is
// declared in Types.h so PassContext can hold a pointer to it without
// including this header.

/// Compute the final physical type of `value`, resolving its producers
/// transitively. Mutates no IR.
///
/// `invocationStack` holds every value whose type computation is currently in
/// progress, exactly as BufferizableOpInterface::getBufferType's stack does:
/// the wrapper pushes `value` before dispatching and pops it after, and a
/// value already on the stack is a cycle (a tensor carried through scf.for
/// iter_args closes one in the value graph even where the op graph is a DAG).
/// Recursion on a cycle returns failure rather than diverging.
///
/// Returns failure when the value does not resolve to a physical type: it is
/// not reachable from a Phase-1 root, no pattern gives its producer a physical
/// result, or a cycle was detected.
llvm::FailureOr<PhysicalTypeInfo>
getPhysicalizedType(Value value, const PhysicalTypeMap &roots,
                    PhysicalTypeMap &resolved,
                    const PhysicalPropagationPatternSet &patterns,
                    llvm::SmallVector<Value> &invocationStack);

/// Run Phase 2A over `module`, seeded from `ctx.physicalValues` (Phase 1's
/// roots). Returns the resolved map. Mutates no IR.
PhysicalTypeMap runPhysicalTypeAnalysis(ModuleOp module,
                                        const PassContext &ctx);

/// Agreement checks between the analysis and the Phase-2 guards it will
/// eventually replace. Active in assertion builds only; see the definition for
/// the invariants checked and what a failure means.
///
/// `when` names the point in the pass the check runs at, for diagnostics.
void verifyPhysicalTypeAgreement(ModuleOp module, const PassContext &ctx,
                                 const PhysicalTypeMap &analysis,
                                 llvm::StringRef when);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PHYSICALTYPEANALYSIS_H
