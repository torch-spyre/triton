#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ktdp {

struct PassContext {
  const llvm::DenseMap<mlir::Value, triton::SpyreTensorLayoutOp> &physMemViewToMarker;
  const llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> &physicalLoadToTransposePerm;
  /// Set by patterns to indicate a fatal error that should abort the pass.
  mutable bool hadError = false;
  /// Mirrors the emit-generic pass option: when true, a stick-split walk is
  /// expressed as a single linalg.generic rather than an scf.for nest. Read by
  /// the emission helpers in ContractionSynthesis.cpp.
  bool emitGenericBody = false;
};

/// Per-operand coord-map info read from a still-live marker.
struct OperandCoords {
  llvm::ArrayRef<int64_t> src; // phys_src
  llvm::ArrayRef<int64_t> op;  // phys_op  (0=Identity,1=FloorDiv,2=Mod)
  llvm::ArrayRef<int64_t> arg; // phys_arg
  unsigned logicalRank;
  llvm::ArrayRef<int64_t> physBlock;

  static OperandCoords fromMarker(triton::SpyreTensorLayoutOp marker,
                                  unsigned logRank,
                                  llvm::ArrayRef<int64_t> physBlock) {
    return {marker.getPhysSrc(), marker.getPhysOp(), marker.getPhysArg(),
            logRank, physBlock};
  }
};

/// How a single physical dim is sliced when extracting the per-iteration tile.
enum class SliceKind {
  StickIndex,       // floor/loopDims dim: offset = this operand's own loop IV,
                    // size = 1 (selects one stick along a stick-index dim).
  StickifiedBlock,  // opInnerDim spanning >1 stick (B's K-flat): offset =
                    // reduction IV * stickSize, size = stickSize (one stick).
  WholeBlock,       // lane / opSlice / single-stick opInnerDim: offset = 0,
                    // size = physBlock[p] (taken whole as part of the 2D tile).
};

/// Pure output of classify(): per-physical-dim role assignments.
struct ClassifiedDims {
  int                lane;        // innermost phys dim = rank-1
  int64_t            stickSize;   // stick/lane width = physBlock[lane]
  llvm::SmallVector<int>   floorDims;   // parallel stick-index dims
  llvm::SmallVector<int>   reduceDims;  // all -1 dims in right-to-left order
  int                opInnerDim;  // rightmost reduceDim; -1 if none
  llvm::SmallVector<int>   loopDims;    // reduceDims minus opInnerDim
  llvm::SmallVector<int>   opTileDims;  // residual >= 0 non-floor dims
  llvm::SmallVector<SliceKind> sliceKind; // per-phys-dim slice behavior
};

/// One operand's full plan: classification + resolution results.
struct OperandPlan {
  Value               value;      // SSA tensor (physical on memory side)
  OperandCoords       coords;     // coord map + shape
  llvm::SmallVector<int64_t> dimRoles;  // per-phys-dim role (>= 0 | -1)
  ClassifiedDims      dims;       // output of classify()

  // Resolved fields — filled by resolveAndReconcile() after classify().
  llvm::SmallVector<int64_t> transposePerm;
  llvm::SmallVector<int64_t> opExtents;
};

/// Per-operand descriptor for a source contraction op.
///
/// The two arrays answer two *different* questions and are indexed
/// differently:
///   - `canonicalAxes[d]` — set membership, indexed by the operand's logical
///     dim `d`: the output-axis role that logical dim carries, or -1 when it
///     is not an output axis (i.e. it is contracted/reduced away, and is
///     therefore also excluded from the accumulator shape).
///   - `targetOrder[t]` — position, indexed by the target op's axis `t`: the
///     role id (or -1 for the contracted/reduced slot) that must end up at
///     axis position `t` of the emitted op's operand.
///
/// They coincide whenever `canonicalAxes` is dense (every role id equals its
/// own index), which is the case for every matmul-like operand:
///
///   matmul B=(k,n):   canonicalAxes {-1, 1}    targetOrder {-1, 1}
///     logical 0 (k) is contracted; logical 1 carries role 1.
///     linalg.matmul wants (k, n), so role 1 sits at position 1 and the
///     contracted slot at position 0 — same array.
///
/// They diverge when `canonicalAxes` is *compacted*, i.e. built by numbering
/// only the surviving axes and skipping the reduced ones. A rank-3 reduce over
/// the middle axis:
///
///   canonicalAxes {0, -1, 1}   targetOrder {0, 1, -1}
///     logical 0 -> role 0, logical 1 reduced, logical 2 -> role 1.
///     Note slot 2 holds value 1: index != value, because numbering skipped
///     the reduced axis. Reading position off the *index* would send role 1
///     to position 2, but the output is only rank 2.
///     linalg.reduce additionally requires the reduced dims trailing, so the
///     target order is (role 0, role 1, reduced).
///
/// The store sink path needs no `SourceOperandSpec`: a store contracts
/// nothing, so there is no -1 and no compaction, and its target order is
/// always the identity (logical dim `d` -> position `d`). It therefore builds
/// a dense iota inline rather than carrying a spec. See `emitSinkStage`.
struct SourceOperandSpec {
  llvm::SmallVector<int64_t> canonicalAxes;
  llvm::SmallVector<int64_t> targetOrder;
};

/// Descriptor for one source contraction op (e.g. linalg.matmul).
struct SourceOpSpec {
  llvm::SmallVector<SourceOperandSpec> operands;
  unsigned logicalRank;

  /// Per-slice emitter. Receives operands already sliced for one iteration of
  /// the synthesized loop nest, plus the incoming accumulator and its type, and
  /// returns the value to accumulate. Drives the scf.for path, which is the
  /// default and the only form consumed downstream today. Always populated.
  llvm::function_ref<Value(OpBuilder &, Location, llvm::ArrayRef<Value>,
                           Value, RankedTensorType)>
      emitOp;

  /// Region-body builder for the linalg.generic form, used when the
  /// emit-generic pass option is on. Builds the *entire* body: every arithmetic
  /// op plus the closing linalg.yield.
  ///
  /// The ValueRange holds block arguments the linalg.generic builder has
  /// already created — one scalar per `ins` operand, then one per `outs`
  /// accumulator — so this callback reads them rather than declaring them. A
  /// two-input, one-output generic therefore sees args[0], args[1], args[2].
  ///
  /// Null when the op kind has no generic form, in which case the caller falls
  /// back to `emitOp`. Both emitters describe the same arithmetic at different
  /// granularity: per-slice versus per-element.
  llvm::function_ref<void(OpBuilder &, Location, mlir::ValueRange)>
      emitGenericBody = nullptr;
};

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H
