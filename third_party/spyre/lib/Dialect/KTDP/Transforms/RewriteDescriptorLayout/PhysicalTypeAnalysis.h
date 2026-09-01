#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PHYSICALTYPEANALYSIS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PHYSICALTYPEANALYSIS_H

#include "RewriteDescriptorLayout/Types.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::triton::ktdp {

//===----------------------------------------------------------------------===//
// Phase 2A: decide the final physical type of every value reachable from
// Phase 1's roots, before any IR is rewritten.
//
// This is analysis only. It mutates no IR, creates no ops, and does not touch
// ctx.physicalValues. Its output is a map from value to the physical type that
// value will end up carrying once Phase 2B has run. Today (step 4c-A) nothing
// consumes that map except the agreement assertions in
// verifyPhysicalTypeAgreement() -- the map's whole purpose at this step is to
// be *measured* against what Phase 2's in-flight guards conclude, which is
// what makes 4c-B a verified change rather than a rewrite on faith.
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

/// How one op propagates physicality from a physical operand to its result.
///
/// This enum -- not an isa<> list of op types -- is what the analysis
/// dispatches on. Adding an op to this pass means naming its behaviour here
/// (see classifyPropagation in PhysicalTypeAnalysis.cpp): the classifier has
/// no silent default, so a new op either matches a stated behaviour or lands
/// in `Unknown`, which the analysis treats as "stops, and say so" rather than
/// guessing.
enum class PhysicalPropagation {
  /// The op's result carries its operand's physical type unchanged: it is
  /// rank-agnostic, so physicalization passes straight through. Elementwise
  /// ops (arith.*, and any single-result op whose tensor operands agree on a
  /// shape), plus the whole-tensor slice a rewritten chain leaves behind.
  PassThrough,
  /// The op consumes the physical value and contributes only a permutation:
  /// no new physical value appears, but the permutation must be composed into
  /// whatever downstream op finally reads the chain. linalg.transpose.
  Absorb,
  /// The op consumes a physical operand and produces a value that is NOT
  /// physical. Its own result is logical, and the chain ends here.
  /// linalg.matmul / linalg.batch_matmul (they contract exactly one K axis,
  /// so a split K needs real cross-stick accumulation, which the emitted
  /// accumulator holds at logical shape); linalg.reduce (its emitted result
  /// is the accumulator, at op-tile/logical shape -- see the reduce note in
  /// PhysicalTypeAnalysis.cpp); ktdp.store (no result at all).
  Stop,
  /// Not a behaviour this analysis has been taught. Treated as Stop, and
  /// reported, so a new op on a physicalized chain is a visible obligation
  /// rather than a silent default.
  Unknown,
};

// PhysicalTypeMap -- one entry per value reachable from a Phase-1 root whose
// physicalization resolves; a value absent from the map is logical, either
// never reachable from a root or ended by a Stop op -- is declared in Types.h
// so PassContext can hold a pointer to it without including this header.

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
/// not reachable from a Phase-1 root, its chain ends at a Stop op, or a cycle
/// was detected.
llvm::FailureOr<PhysicalTypeInfo>
getPhysicalizedType(Value value, const PhysicalTypeMap &roots,
                    PhysicalTypeMap &resolved,
                    llvm::SmallVector<Value> &invocationStack);

/// Run Phase 2A over `module`, seeded from `ctx.physicalValues` (Phase 1's
/// roots). Returns the resolved map. Mutates no IR.
PhysicalTypeMap runPhysicalTypeAnalysis(ModuleOp module,
                                        const PassContext &ctx);

/// Agreement checks between the analysis and the Phase-2 guards it will
/// eventually replace. Active in assertion builds only; see the definition for
/// the three invariants checked and what a failure means.
///
/// `when` names the point in the pass the check runs at, for diagnostics.
void verifyPhysicalTypeAgreement(ModuleOp module, const PassContext &ctx,
                                 const PhysicalTypeMap &analysis,
                                 llvm::StringRef when);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PHYSICALTYPEANALYSIS_H
