#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H

#include "RewriteDescriptorLayout/PermutationUtils.h"

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

/// Forward declaration: Phase 2A's result map. Defined in
/// PhysicalTypeAnalysis.h, which PassContext deliberately does not include --
/// the analysis depends on Types.h, not the other way round.
struct PhysicalTypeInfo;
using PhysicalTypeMap = llvm::DenseMap<mlir::Value, PhysicalTypeInfo>;

/// Phase 2B's ONLY write access to Phase 2A's PhysicalTypeMap, and it grants
/// exactly one operation.
///
/// Phase 2B reads the analysis through `PassContext::physicalTypeAnalysis`,
/// which is a pointer to const on purpose: a rewrite must not be able to invent
/// a physical-type decision, or revise one, behind the analysis's back. The one
/// thing a rewrite legitimately needs is this: when a source pattern REPLACES an
/// op whose result the analysis decided is physical, the replacement is a
/// brand-new value the analysis never saw, carrying the decision made for the
/// value it replaced. That is not a new decision, and without recording it
/// `verifyPhysicalTypeAgreement`'s containment check (every value Phase 2 found
/// physical, the analysis predicted) would fail on the replacement, leaving only
/// the option of exempting it -- a hole in the check. With it, containment holds
/// BY CONSTRUCTION for minted values.
///
/// The map is held privately and the class is defined here while
/// `PhysicalTypeInfo` is still incomplete, so `carryForward` cannot be inlined
/// and no Phase 2B translation unit can reach the map through this handle by
/// any other route. Widening what 2B may write therefore means adding a method
/// here, named and reviewed, rather than a `const` quietly going missing inside
/// a pattern.
class PhysicalTypeCarryForward {
public:
  /// Inert: no analysis to carry anything forward in. Calling `carryForward` on
  /// one of these is a bug, and asserts.
  PhysicalTypeCarryForward() = default;
  /// Out-of-line, like `carryForward` and for the same reason: taking the
  /// address of the map needs `PhysicalTypeInfo` complete, which it is not here.
  explicit PhysicalTypeCarryForward(PhysicalTypeMap &analysis);

  /// Record that `replacement` carries the physical-type decision the analysis
  /// already made for `replaced`, by copying `replaced`'s entry onto it.
  ///
  /// Precondition: `replaced` HAS an entry. That is not a hope -- it is the very
  /// thing that put the pattern on the path where it mints a replacement (see
  /// RewriteReducePattern, where the presence of the result's entry is what
  /// selects OutputAxisSpace::Physical). Absence would mean those two have
  /// drifted apart, so it asserts rather than skipping: carrying nothing forward
  /// would silently re-open exactly the hole this exists to close.
  ///
  /// Const because it is the map that is mutated, not the handle -- patterns
  /// hold `const PassContext &`, the same way they already mutate
  /// `ctx.physicalValues` through it.
  void carryForward(mlir::Value replaced, mlir::Value replacement) const;

private:
  PhysicalTypeMap *analysis = nullptr;
};

/// Phase 1's output: which marker each physicalized memory view came from.
/// Immutable for everything downstream, and named because Phase 2A's propagation
/// rules take *this* rather than the whole PassContext -- see the note there.
using MarkerByMemView =
    llvm::DenseMap<mlir::Value, triton::SpyreTensorLayoutOp>;

struct PassContext {
  const MarkerByMemView &physMemViewToMarker;
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
  /// physicalized (no marker anywhere upstream) is simply never in this map,
  /// which is what keeps the elementwise pattern's local shape rule from
  /// mis-firing on ops like tt.expand_dims (see RewriteElementwisePattern).
  llvm::DenseMap<mlir::Value, PhysicalValueInfo> &physicalValues;
  /// Phase 2A's answer: the final physical type of every value reachable from
  /// Phase 1's roots, computed before Phase 2 rewrites anything (see
  /// PhysicalTypeAnalysis.h). Null when the analysis was not run.
  const PhysicalTypeMap *physicalTypeAnalysis = nullptr;
  /// The one write path back into that map -- see PhysicalTypeCarryForward for
  /// why it is a handle with a single method rather than dropping the `const`
  /// above. Inert when the analysis was not run, which is consistent: nothing
  /// then decides a value is physical, so nothing mints a replacement either.
  PhysicalTypeCarryForward physicalTypes;
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
  StickIndex,       // scatterDims/reduceLoopDims dim: offset = this operand's
                    // own loop IV, size = 1 (one stick along a stick-index dim).
  StickifiedBlock,  // opInnerDim spanning >1 stick (B's K-flat): offset =
                    // reduction IV * stickSize, size = stickSize (one stick).
  WholeBlock,       // lane / opSlice / single-stick opInnerDim: offset = 0,
                    // size = physBlock[p] (taken whole as part of the 2D tile).
};

/// Pure output of classify(): per-physical-dim role assignments.
///
/// Two of these buckets, `scatterDims` and `reduceLoopDims`, hold dims that are
/// SLICED IDENTICALLY — both get `SliceKind::StickIndex`, one stick per
/// iteration of their own loop — and are named for the only thing that
/// distinguishes them, which is what their loop DOES with the slice:
///
///   scatterDims     surviving (role >= 0) stick indices. The loop SCATTERS:
///                   each iteration writes a different slice of the output.
///   reduceLoopDims  reduced (role == -1) dims beyond the first (`opInnerDim`
///                   takes that one). The loop ACCUMULATES: every iteration
///                   folds into the same accumulator.
///
/// The store sink path reads exactly that distinction as its two preconditions
/// (see RewriteStorePattern): an empty `scatterDims` means nothing to scatter,
/// and a non-empty `reduceLoopDims` means a reduction the store cannot express.
struct ClassifiedDims {
  int                lane;        // innermost phys dim = rank-1
  int64_t            stickSize;   // stick/lane width = physBlock[lane]
  llvm::SmallVector<int>   scatterDims; // surviving stick indices: loop scatters
  llvm::SmallVector<int>   reduceDims;  // all -1 dims, ascending
  int                opInnerDim;  // rightmost reduceDim; -1 if none
  // reduceDims minus opInnerDim: loop accumulates.
  llvm::SmallVector<int>   reduceLoopDims;
  llvm::SmallVector<int>   opTileDims;  // residual >= 0 non-scatter dims
  llvm::SmallVector<SliceKind> sliceKind; // per-phys-dim slice behavior
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
/// a dense iota inline rather than carrying a spec. See `emitWidenStage`.
struct SourceOperandSpec {
  llvm::SmallVector<int64_t> canonicalAxes;
  llvm::SmallVector<int64_t> targetOrder;
};

/// Descriptor for one source contraction op (e.g. linalg.matmul).
///
/// The two flags below are the synthesis's two absorption questions, declared
/// by the calling pattern because only it knows the answers. They are separate
/// because they are about different dim sets and are answered from different
/// evidence:
///
///   `absorbReduceLoopDims` — the op's REDUCE axis set. True iff the emitted op
///     can name every reduced axis at once (linalg.reduce's `dimensions`), false
///     when a second reduce axis must become a real cross-stick accumulation
///     loop (matmul contracts exactly one K). A property of the op kind.
///
///   `outputAxes` — the op's OUTPUT axes, and hence the fate of a *surviving*
///     stick-index dim. A property of this op instance, decided by Phase 2A
///     against the layout the result is stored under (see OutputAxisSpace).
struct SourceOpSpec {
  llvm::SmallVector<SourceOperandSpec> operands;
  unsigned logicalRank;
  bool absorbReduceLoopDims = false;
  OutputAxisSpace outputAxes = OutputAxisSpace::Logical;
  /// The layout the emitted op's output carries, when `outputAxes` is
  /// Physical: the marker of the descriptor the result is stored to, as Phase
  /// 2A resolved it. Null in the Logical space, where the result carries no
  /// layout of its own and any physical form is built afterwards by the
  /// store's widen stage.
  triton::SpyreTensorLayoutOp outputMarker;
  llvm::function_ref<Value(OpBuilder &, Location, llvm::ArrayRef<Value>,
                           Value, RankedTensorType)>
      emitOp;
};

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_TYPES_H
