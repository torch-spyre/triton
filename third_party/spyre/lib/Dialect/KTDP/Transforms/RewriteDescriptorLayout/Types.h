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

/// What is known about one physical value (a physicalized ktdp.load result,
/// or a value retyped to physical by Phase 2's elementwise pattern).
struct PhysicalValueInfo {
  /// The layout marker for the descriptor this value ultimately derives
  /// from. Always set when a PhysicalValueInfo exists at all: Phase 1 seeds
  /// every entry from a ktdp.load whose access tile resolves to a marker via
  /// physMemViewToMarker, and Phase 2 only ever grows the map by copying an
  /// existing entry from an operand to that operand's op result.
  triton::SpyreTensorLayoutOp marker;
  /// The logical permutation of a linalg.transpose erased on this value's
  /// chain, empty if none. Populated by Phase 2's transpose pattern
  /// (RewriteTransposePattern) as it erases transposes, and read by
  /// dispatchSource to fold the erased permutation into canonicalAxes.
  llvm::SmallVector<int64_t> transposePerm;
};

struct PassContext {
  const llvm::DenseMap<mlir::Value, triton::SpyreTensorLayoutOp> &physMemViewToMarker;
  /// Maps a physical value (a ktdp.load result, or a value Phase 2 has
  /// retyped to physical) to what is known about it -- see PhysicalValueInfo.
  /// Seeded by Phase 1 with every physical ktdp.load result (the root of
  /// every chain Phase 2 will retype), and grown by Phase 2's
  /// RewriteElementwisePattern with the result of each op it retypes (a
  /// value it just made physical, carrying forward its operand's info) and
  /// by RewriteTransposePattern as it erases transposes (recording the
  /// permutation against the transpose's input, which is already an entry).
  /// Presence answers "is this value reachable from a physicalized load"
  /// without re-walking the chain: an op on a path that was never
  /// physicalized (no marker anywhere upstream) is simply never in this map.
  /// This is what keeps the elementwise pattern's local shape rule from
  /// mis-firing on ops like tt.expand_dims that happen to have one tensor
  /// operand and a differently-shaped result but are not reachable from any
  /// physicalized load.
  llvm::DenseMap<mlir::Value, PhysicalValueInfo> &physicalValues;
  /// Set by patterns to indicate a fatal error that should abort the pass.
  mutable bool hadError = false;
};

/// Per-operand coord-map info read from a still-live marker.
struct OperandCoords {
  llvm::SmallVector<int64_t> src; // phys_src
  llvm::SmallVector<int64_t> op;  // phys_op  (0=Identity,1=FloorDiv,2=Mod)
  llvm::SmallVector<int64_t> arg; // phys_arg
  unsigned logicalRank;
  // Aliases an MLIR type's shape (RankedTensorType::getShape() or
  // AccessTileType::getShape()), which is uniqued and immortal, so this one
  // stays a non-owning ArrayRef.
  llvm::ArrayRef<int64_t> physBlock;

  static OperandCoords fromMarker(triton::SpyreTensorLayoutOp marker,
                                  unsigned logRank,
                                  llvm::ArrayRef<int64_t> physBlock) {
    return {llvm::SmallVector<int64_t>(marker.getPhysSrc()),
            llvm::SmallVector<int64_t>(marker.getPhysOp()),
            llvm::SmallVector<int64_t>(marker.getPhysArg()),
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
  // All reduceDims, sorted ascending -- the full in-op reduce axis set a
  // linalg.reduce can express directly via `dimensions` (which requires
  // DenseArrayStrictlySorted, not contiguity or a trailing position). Added
  // alongside opInnerDim/loopDims rather than replacing them: matmul still
  // consumes those two fields exactly as before (it can only ever absorb one
  // reduce axis per operand; a second becomes a real cross-stick
  // accumulation loop). Only the reduce dispatch path reads this field.
  llvm::SmallVector<int>   opInnerReduceDims;
};

/// One operand's full plan: classification + resolution results.
struct OperandPlan {
  Value               value;      // SSA tensor (physical on memory side)
  OperandCoords       coords;     // coord map + shape
  llvm::SmallVector<int64_t> dimRoles;  // per-phys-dim role (>= 0 | -1)
  ClassifiedDims      dims;       // output of classify()

  // Resolved fields — filled by resolveOperand() after classify().
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
  llvm::function_ref<Value(OpBuilder &, Location, llvm::ArrayRef<Value>,
                           Value, RankedTensorType)>
      emitOp;
  /// True only for linalg.reduce: unlike matmul, a reduce op can absorb its
  /// entire reduce axis set (ClassifiedDims::opInnerReduceDims) directly via
  /// `dimensions`, so dispatchSource folds every plan's loopDims into
  /// opTileDims before resolving/slicing rather than driving a stick loop
  /// over them (see Step 3 of the layout refactor plan).
  bool absorbLoopDimsIntoReduce = false;
};

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H
