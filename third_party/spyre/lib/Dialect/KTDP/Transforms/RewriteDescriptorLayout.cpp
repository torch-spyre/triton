//===- RewriteDescriptorLayout.cpp ----------------------------------------===//
//
// Rewrites logical tensor descriptors to their physical (stick-tiled) layout,
// driven by tt.spyre_tensor_layout markers. Runs LAST in the TTIR→KTDP
// pipeline (after LowerDescriptorMemory + LowerComputeOps), so that tt.dot
// is already lowered to linalg.matmul before operands are physicalized.
//
// The physical layout is the OpSpec `device_coordinates` form, carried on the
// marker as three i64 arrays, one entry per physical dim:
//   phys_src[k] : logical dim k derives from
//   phys_op[k]  : 0 = identity, 1 = floordiv, 2 = mod
//   phys_arg[k] : divisor (floordiv) / modulus (mod); ignored for identity
// e.g. [M,N] stick-on-N -> phys_src=[1,0,1] phys_op=[1,0,2] phys_arg=[64,0,64]
//   => device_size [N//64, M, N%64].
//
// The marker persists through LowerDescriptorMemory because Walk 1 of that
// pass RAUWs each tt.make_tensor_descriptor with an UnrealizedConversionCast
// wrapping a ktdp.construct_memory_view; since the cast preserves the
// !tt.tensordesc type, the marker's `desc` operand auto-re-points at it and
// remains valid. Both intervening conversion passes mark SpyreTensorLayoutOp
// legal so it is never flagged as unconverted.
//
// Staged model: each op-stage independently physicalizes the side that touches
// memory, reusing the enclosing scf.for nest that Phase 1 wired to stick
// granularity. The source stage (matmul) physicalizes its INPUTS and leaves the
// output LOGICAL; the sink stage (store) consumes the LOGICAL value and
// physicalizes its OUTPUT. Stages share only a logical SSA value + the placement
// rule (emit at the value's block, after its uses); they share the classify /
// Loop / classify utilities.
//
// Call graph:
//   runOnOperation
//     Phase 1 — for each marker:
//       rewriteOnePhysicalize(marker)
//         isLoweredDescriptor / getDescriptorMemView
//         buildPhysicalMemoryView   -> physical ConstructMemoryViewOp
//           applyStatic / applyCoordMap  derive physical static extents
//         rewriteAccessTile(tile, coords) -> physical ConstructAccessTilesOp
//           rescaleEnclosingLoop    rescale enclosing scf.for to stick granularity
//         retypeLoad                -> physical ktdp.load + retypeChain
//           retypeChain             propagate type (stops at isContractionOp)
//         redirectStoreAccessTile   redirect access tile operand (output stays
//                                   logical-typed; sink stage fixes it in Phase 2)
//         (marker kept alive for Phase 2)
//     Phase 2 — synthesizeContractions (loop-until-stable):
//       single walk → dispatchOne per op type:
//         linalg.MatmulOp → dispatchMatmul → dispatchSource (classify, emitSourceStage)
//         linalg.ReduceOp → dispatchReduce → dispatchSource (classify, emitSourceStage)
//         ktdp.StoreOp    → dispatchSink   (classify, emitSinkStage)
//       add new contraction/sink types to dispatchOne + a dispatchXxx helper
//       emitSourceStage uses getDpsInits()[0] (linalg DPS interface) so it
//       works generically for MatmulOp, ReduceOp, and future source ops
//     Phase 3 — eraseMarker (marker + dead bridge cast)
//
//   Coord helpers (free functions):
//     applyStatic     : compile-time extent for one dim
//     applyCoordMap   : compile-time extents for all physical dims
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Ktdp/KtdpAttrs.hpp"
#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "Ktdp/KtdpTypes.hpp"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <numeric>
#include <optional>

namespace mlir::triton::ktdp {

#define GEN_PASS_DEF_REWRITEDESCRIPTORLAYOUT
#include "Dialect/KTDP/Transforms/Passes.h.inc"

namespace {

// Coordinate op codes, matching the Python builtin (semantic.py).
enum class CoordOp : int64_t { Identity = 0, FloorDiv = 1, Mod = 2 };

/// Try to extract a compile-time int64 from an arith.constant SSA value.
static std::optional<int64_t> getConstantInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt();
  return std::nullopt;
}

/// Apply one coordinate op to a static (compile-time) logical extent.
/// Returns a non-kDynamic int64 on success, or std::nullopt when the result
/// is dynamic (i.e. needs a runtime SSA value). kDynamic is NOT returned as
/// a "valid" result — it would be silently propagated into static_sizes attrs
/// without a matching dynamic SSA operand, causing OOB at op-build time.
static std::optional<int64_t> applyStatic(int64_t logical, CoordOp op,
                                           int64_t arg) {
  switch (op) {
  case CoordOp::Identity:
    // kDynamic propagates as dynamic — must supply an SSA value.
    if (logical == ShapedType::kDynamic)
      return std::nullopt;
    return logical;
  case CoordOp::FloorDiv:
    if (logical == ShapedType::kDynamic)
      return std::nullopt; // dynamic; emit SSA ceildivsi at build time
    return arg == 0 ? std::optional<int64_t>(std::nullopt)
                    : std::optional<int64_t>((logical + arg - 1) / arg);
  case CoordOp::Mod:
    // Physical extent of a mod (lane) dim is the modulus itself — always static.
    return arg;
  }
  return std::nullopt;
}

// Compute physical static extents from logical static extents via a coord map.
// Returns false if any physical dim is dynamic; caller handles the dynamic path.
static bool applyCoordMap(ArrayRef<int64_t> logSizes,
                          ArrayRef<int64_t> physSrc,
                          ArrayRef<int64_t> physOp,
                          ArrayRef<int64_t> physArg,
                          SmallVectorImpl<int64_t> &out) {
  unsigned physRank = physSrc.size();
  out.resize(physRank);
  for (unsigned k = 0; k < physRank; ++k) {
    auto sz = applyStatic(logSizes[physSrc[k]],
                          static_cast<CoordOp>(physOp[k]), physArg[k]);
    if (!sz)
      return false;
    out[k] = *sz;
  }
  return true;
}

/// Walk a def-chain of index arithmetic back to the single BlockArgument it
/// derives from.  Handles index_cast, muli-by-constant, divsi/remsi-by-constant,
/// and addi-with-constant.  Returns the BlockArgument if the chain leads to
/// exactly one, or nullptr otherwise.
static BlockArgument traceToMLIRBlockArg(Value v) {
  while (true) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return ba;
    auto *op = v.getDefiningOp();
    if (!op)
      return nullptr;
    // index_cast / trunci / extsi / extui: single operand chains
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp,
            arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(op)) {
      v = op->getOperand(0);
      continue;
    }
    // muli / divsi / remsi / addi with a constant second operand
    if (isa<arith::MulIOp, arith::DivSIOp, arith::RemSIOp, arith::AddIOp>(op)) {
      if (op->getNumOperands() == 2 && getConstantInt(op->getOperand(1)))
        { v = op->getOperand(0); continue; }
    }
    return nullptr;
  }
}

// ---- helpers matching LowerDescriptorMemory's predicate / accessor ----

/// True iff desc is a memref-backed lowered descriptor (the
/// UnrealizedConversionCast bridge left by LowerDescriptorMemory Walk 1).
static bool isLoweredDescriptor(Value desc) {
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp && !castOp.getInputs().empty() &&
         isa<MemRefType>(castOp.getInputs()[0].getType());
}

/// Unwrap the bridge cast to recover the ktdp.construct_memory_view result.
static Value getDescriptorMemView(Value desc) {
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp.getInputs()[0];
}

/// Build the range-set constraint for an N-D coordinate space (mirrors the
/// helper in LowerDescriptorMemory). Static dims use constants; dynamic dims
/// introduce IntegerSet symbols.
static IntegerSet buildRangeSetND(MLIRContext *ctx, ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  unsigned symCount = 0;
  for (auto s : shape)
    if (s == ShapedType::kDynamic)
      ++symCount;

  SmallVector<AffineExpr> constraints;
  SmallVector<bool> eqFlags;
  unsigned symIdx = 0;
  for (unsigned i = 0; i < rank; ++i) {
    auto di = getAffineDimExpr(i, ctx);
    AffineExpr upper;
    if (shape[i] == ShapedType::kDynamic)
      upper = getAffineSymbolExpr(symIdx++, ctx) - 1;
    else
      upper = getAffineConstantExpr(shape[i] - 1, ctx);
    constraints.push_back(di);
    eqFlags.push_back(false);
    constraints.push_back(upper - di);
    eqFlags.push_back(false);
  }
  return IntegerSet::get(rank, symCount, constraints, eqFlags);
}

struct RewriteDescriptorLayoutPass
    : public mlir::triton::ktdp::impl::RewriteDescriptorLayoutBase<
          RewriteDescriptorLayoutPass> {

  // Maps each physical ConstructMemoryViewOp result -> its source marker.
  // Populated during Phase 1; read during Phase 2 (markers still live).
  DenseMap<Value, triton::SpyreTensorLayoutOp> physMemViewToMarker;

  // Loops already rescaled to stick granularity. Pass-level (not per-descriptor)
  // because multiple descriptors can share the same loop IV — the first
  // descriptor that physicalizes a FloorDiv dim rescales the loop and fixes the
  // muli multipliers; subsequent descriptors on the same loop must skip.
  DenseSet<scf::ForOp> rescaledLoops;


  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect markers up front; mutating while walking invalidates the cursor.
    SmallVector<triton::SpyreTensorLayoutOp> markers;
    module.walk([&](triton::SpyreTensorLayoutOp op) { markers.push_back(op); });

    // Phase 1: physicalize each annotated descriptor (memView + access tiles +
    // loads). retypeChain propagates physical types elementwise but stops at
    // multi-tensor ops (linalg.matmul etc.). Markers are NOT erased yet —
    // Phase 2 needs to read their coord maps to drive matmul synthesis.
    for (auto marker : markers)
      if (failed(rewriteOnePhysicalize(marker)))
        return signalPassFailure();

    // Phase 2: fixup linalg.matmul ops whose operands were retyped to rank-3
    // by Phase 1. Walks back from each mismatched operand to its marker to
    // read the coord map. Loop until stable (onion-peeling).
    if (failed(synthesizeContractions(module)))
      return signalPassFailure();

    // Phase 3: erase all markers (and their now-dead bridge casts).
    for (auto marker : markers)
      eraseMarker(marker);
  }

  // Generic stick-loop helper: emits scf.for s=0..tripCount iter_args(acc) and
  // calls body(builder, stickIV, acc). tripCount is the total number of stick
  // iterations at this nesting level. For tripCount<=1, calls body with the
  // outer builder and a constant-0 IV (no loop emitted — 0 or 1 trip inlines).
  // Used by both the matmul source stage and the store sink scatter.
  Value emitStickLoop(OpBuilder &b, Location loc, int64_t tripCount, Value acc,
                      function_ref<Value(OpBuilder &, Value, Value)> body) {
    if (tripCount <= 1) {
      Value s0 = arith::ConstantIndexOp::create(b, loc, 0);
      return body(b, s0, acc);
    }
    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value ub = arith::ConstantIndexOp::create(b, loc, tripCount);
    auto forOp = scf::ForOp::create(b, loc, c0, ub, c1, ValueRange{acc});
    OpBuilder ib = OpBuilder::atBlockBegin(forOp.getBody());
    Value stepped =
        body(ib, forOp.getInductionVar(), forOp.getRegionIterArgs()[0]);
    scf::YieldOp::create(ib, loc, ValueRange{stepped});
    b.setInsertionPointAfter(forOp);
    return forOp.getResult(0);
  }

  // Return true and dispatch if `op` needs Phase 2 processing, false if not.
  // Source ops (matmul, …): rank-mismatched inputs → dispatch, erase, RAUW.
  // Sink ops (store): data/tile rank mismatch with a marker → redirect data_tile.
  // Add new op types here; synthesizeContractions drives the loop generically.
  LogicalResult dispatchOne(Operation *op, bool &changed) {
    // Shared predicate for all linalg source ops: true if any input operand
    // is rank-increased (Phase 1 added a stick dim) AND traces back to a
    // physicalized memory view with a layout marker.  The rank pre-check
    // avoids re-dispatching canonical (already-lowered) ops whose inputs
    // happen to trace back through the same physical load.
    auto sourceNeedsDispatch = [&](linalg::LinalgOp op, unsigned logicalRank) {
      return llvm::any_of(op.getDpsInputOperands(), [&](OpOperand *operand) {
        auto t = dyn_cast<RankedTensorType>(operand->get().getType());
        if (!t || t.getRank() <= (int)logicalRank)
          return false;
        return static_cast<bool>(findMarkerForOperand(operand->get()));
      });
    };

    if (auto mm = dyn_cast<linalg::MatmulOp>(op))
      return sourceNeedsDispatch(mm, 2) ? (changed = true, dispatchMatmul(mm)) : success();

    if (auto bmm = dyn_cast<linalg::BatchMatmulOp>(op))
      return sourceNeedsDispatch(bmm, 3) ? (changed = true, dispatchBatchMatmul(bmm)) : success();

    if (auto rd = dyn_cast<linalg::ReduceOp>(op)) {
      // Derive the logical input rank from the marker (phys_src max+1) so the
      // rank pre-check in sourceNeedsDispatch is correct even when the init has
      // already been physicalized by Phase 1.
      auto rdMarker = findMarkerForOperand(rd.getInputs()[0]);
      unsigned logicalInputRank = 2; // fallback: minimum plausible rank
      if (rdMarker) {
        for (int64_t src : rdMarker.getPhysSrc())
          if ((unsigned)(src + 1) > logicalInputRank)
            logicalInputRank = (unsigned)(src + 1);
      }
      return sourceNeedsDispatch(rd, logicalInputRank)
                 ? (changed = true, dispatchReduce(rd))
                 : success();
    }

    if (auto st = dyn_cast<mlir::ktdp::StoreOp>(op)) {
      auto dataTy = dyn_cast<RankedTensorType>(st.getDataTile().getType());
      auto tileTy = dyn_cast<mlir::ktdp::AccessTileType>(
          st.getAccessTile().getType());
      if (!dataTy || !tileTy ||
          dataTy.getRank() == (int)tileTy.getShape().size())
        return success();
      auto marker = findMarkerForStore(st.getDataTile());
      if (!marker)
        return success();
      changed = true;
      return dispatchSink(st, marker);
    }
    return success();
  }

  // Phase 2 driver. Single walk collects all candidate ops; dispatchOne
  // handles each by type. Repeats until no op triggers a dispatch (onion-peel
  // for chained contractions). Terminates because each iteration strictly
  // reduces the number of mismatched ops and the IR is finite.
  LogicalResult synthesizeContractions(ModuleOp module) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> candidates;
      module.walk([&](Operation *op) {
        if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ReduceOp, mlir::ktdp::StoreOp>(op))
          candidates.push_back(op);
      });
      for (auto *op : candidates)
        if (failed(dispatchOne(op, changed)))
          return failure();
    }
    return success();
  }

  // True iff op is a single-result elementwise op with exactly one
  // RankedTensor operand. Used by all three chain-walk helpers.
  static bool isSingleTensorElementwiseOp(Operation *op) {
    if (op->getNumResults() != 1 || op->getNumOperands() == 0)
      return false;
    int tensorOps = 0;
    for (auto operand : op->getOperands())
      if (isa<RankedTensorType>(operand.getType()))
        ++tensorOps;
    return tensorOps == 1;
  }

  // Walk backward from `val` through single-tensor elementwise ops and
  // linalg.transpose ops to the ktdp.load that produced it.
  // Returns {load, accumulatedPermutation}. The permutation is nullopt if no
  // transpose was encountered (identity), or the composed logical permutation
  // of all traversed transposes. Returns {null, nullopt} if not found.
  static std::pair<mlir::ktdp::LoadOp, std::optional<SmallVector<int64_t>>>
  walkToLoad(Value val) {
    Value v = val;
    std::optional<SmallVector<int64_t>> accumPerm;
    while (true) {
      auto *defOp = v.getDefiningOp();
      if (!defOp)
        return {mlir::ktdp::LoadOp{}, std::nullopt};
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(defOp))
        return {ld, accumPerm};
      // Step through linalg.transpose: follow data operand(0), accumulate perm.
      if (auto tr = dyn_cast<linalg::TransposeOp>(defOp)) {
        auto perm = SmallVector<int64_t>(tr.getPermutation());
        if (!accumPerm.has_value()) {
          accumPerm = perm;
        } else {
          // Compose: new[i] = old[perm[i]]
          SmallVector<int64_t> composed(perm.size());
          for (unsigned i = 0; i < perm.size(); ++i)
            composed[i] = (*accumPerm)[perm[i]];
          accumPerm = composed;
        }
        v = defOp->getOperand(0); // data input of transpose
        continue;
      }
      if (!isSingleTensorElementwiseOp(defOp))
        return {mlir::ktdp::LoadOp{}, std::nullopt};
      for (auto operand : defOp->getOperands())
        if (isa<RankedTensorType>(operand.getType())) { v = operand; break; }
    }
  }

  // Walk back from a matmul operand through the elementwise chain to the
  // ktdp.load, then look up the physical memView -> marker map populated
  // during Phase 1.
  triton::SpyreTensorLayoutOp findMarkerForOperand(Value operand) {
    auto [ld, perm] = walkToLoad(operand);
    if (!ld)
      return {};
    auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
        ld.getAccessTile().getDefiningOp());
    if (!tile)
      return {};
    auto it = physMemViewToMarker.find(tile.getBase());
    return it != physMemViewToMarker.end() ? it->second
                                           : triton::SpyreTensorLayoutOp{};
  }

  // Per-operand coord-map info read from a still-live marker.
  struct OperandCoords {
    ArrayRef<int64_t> src; // phys_src
    ArrayRef<int64_t> op;  // phys_op  (0=Identity,1=FloorDiv,2=Mod)
    ArrayRef<int64_t> arg; // phys_arg
    // Logical rank of the descriptor (= number of logical dims, e.g. 2 for MxK).
    unsigned logicalRank;
    // Physical shape of the load result tensor.
    ArrayRef<int64_t> physBlock;

    static OperandCoords fromMarker(triton::SpyreTensorLayoutOp marker,
                                    unsigned logRank,
                                    ArrayRef<int64_t> physBlock) {
      return {marker.getPhysSrc(), marker.getPhysOp(), marker.getPhysArg(),
              logRank, physBlock};
    }
  };

  // Assign a role to each physical dim of an operand:
  //   >= 0  : parallel dim, maps to output axis [value] (0=M, 1=N, etc.)
  //   -1    : reduction dim — consumed by Op (inner) or looped (outer)
  //
  // All physical dims whose logical source has canonicalAxes[logDim] == -1
  // get role -1. classify() then splits them right-to-left: the rightmost is
  // opInnerDim (consumed by Op); the rest are loopDims (scf.for reduction loops).
  // Build per-physical-dim roles from canonicalAxes.
  // canonicalAxes[logDim] gives the output axis index (>= 0) or -1 (reduction).
  // Each physical dim p whose src is logDim gets roles[p] = canonicalAxes[logDim].
  static void buildDimRoles(const OperandCoords &coords,
                             ArrayRef<int64_t> canonicalAxes,
                             SmallVectorImpl<int64_t> &roles) {
    int n = (int)coords.src.size();
    roles.resize(n);
    for (int p = 0; p < n; ++p) {
      int64_t logDim = coords.src[p];
      roles[p] = (logDim < (int64_t)canonicalAxes.size())
                     ? canonicalAxes[logDim]
                     : -1;
    }
  }

  // ---- Shared utilities (staged model) ----

  // How a single physical dim is sliced when extracting the per-iteration tile.
  // Decided per-dim in classify() from local geometry; consumed by extractOpSlice
  // (and the slice-type computation) as a pure lookup — no loop-set inspection
  // or cross-operand reasoning at slice time.
  enum class SliceKind {
    StickIndex,       // floor/loopDims dim: offset = this operand's own loop IV,
                      // size = 1 (selects one stick along a stick-index dim).
    StickifiedBlock,  // opInnerDim spanning >1 stick (B's K-flat): offset =
                      // reduction IV * stickSize, size = stickSize (one stick, R-b).
    WholeBlock,       // lane / opSlice / single-stick opInnerDim: offset = 0,
                      // size = physBlock[p] (taken whole as part of the 2D tile).
  };

  // One operand classified by its Map(Op, X) roles and physical layout.
  // `classify` fills all derived fields from `coords` + `dimRoles`.
  // `resolveAndReconcile` fills the resolved fields after classify().
  //
  // Field semantics (right-to-left traversal of the physical dims, R-c):
  //   lane        = innermost phys dim (rank-1); always the stick lane — full slice
  //   floorDims   = parallel dims that are FloorDiv (stick-index dims) → loops
  //   reduceDims  = all -1 dims, right-to-left order
  //   opInnerDim = rightmost reduceDim → consumed by Op (inner dot)
  //   loopDims  = remaining reduceDims (all but rightmost) → reduction loops
  //   opTileDims = residual >= 0 dims that are not floorDims (fed to Op as-is)
  struct OperandPlan {
    Value               value;      // SSA tensor (physical on memory side)
    OperandCoords       coords;     // coord map + shape (kept for op/arg lookups)
    SmallVector<int64_t> dimRoles;  // per-phys-dim role (>= 0 | -1)
    // Owned storage for coords.physBlock when there is no marker (scratchpad).
    SmallVector<int64_t> physBlockStorage;

    int                lane;        // innermost phys dim = rank-1
    int64_t            stickSize;   // stick/lane width = physBlock[lane] (e.g. 64).
                                    // The slice extent for any stick-width dim
                                    // (the lane, and a sliced opInnerDim). Distinct
                                    // from physBlock[dim], which is the full dim
                                    // extent (= n_sticks * stickSize when BLOCK > stick).
    SmallVector<int>   floorDims;   // parallel stick-index dims → loops (role>=0, FloorDiv)
    SmallVector<int>   reduceDims;  // all -1 dims in right-to-left order
    int                opInnerDim;  // rightmost reduceDim → consumed by Op; -1 if none
    SmallVector<int>   loopDims;    // reduceDims minus opInnerDim → reduction loops
    SmallVector<int>   opTileDims;  // residual >= 0 non-floor dims (the 2D slice for matmul)
    SmallVector<SliceKind> sliceKind; // per-phys-dim slice behavior (see SliceKind)

    // Resolved fields — filled by resolveAndReconcile() after classify().
    // emitSourceStage reads these only; it does not re-derive from spec/dimRoles.
    SmallVector<int64_t> transposePerm;    // permutation from physical to canonical axis order; empty = identity (no transpose)
    SmallVector<int64_t> opExtents;        // extent of each opTileDim after slicing (pre-transpose)

  };

  // Classify one operand's physical dims into OperandPlan fields (R-c).
  // `coords` carries phys_src/op/arg and physBlock. `dimRoles` comes from
  // buildDimRoles (values: >= 0 parallel, -1 reduction). The resulting plan
  // owns copies of the role vector.
  //
  // Right-to-left pass:
  //   - innermost dim (rank-1) → lane (always full inner slice)
  //   - -1 dims → reduceDims (right-to-left); rightmost is opInnerDim
  //     (consumed by Op), the rest are loopDims (become scf.for loops)
  //   - >= 0 FloorDiv dims → floorDims (parallel stick-index loops)
  //   - residual >= 0 dims → opTileDims (fed to Op as-is)
  static OperandPlan classify(Value val, const OperandCoords &coords,
                               ArrayRef<int64_t> dimRoles) {
    int rank = (int)dimRoles.size();
    OperandPlan plan;
    plan.value     = val;
    plan.coords    = coords;
    plan.dimRoles  = SmallVector<int64_t>(dimRoles.begin(), dimRoles.end());
    plan.lane      = rank - 1;
    plan.stickSize = coords.physBlock[rank - 1]; // lane extent = stick width
    plan.opInnerDim = -1;

    // Walk right-to-left (innermost first) per R-c.
    // -1 dims: track reduceDims and split into opInnerDim (rightmost, consumed
    // by Op) and loopDims (outer, each becomes an scf.for reduction loop).
    // opInnerDim is also added to opTileDims during the walk so the existing
    // extraction machinery (which builds the 2D Op tile from opTileDims) keeps
    // working unchanged. S3-S5 will switch extraction to use opInnerDim directly.
    for (int p = rank - 1; p >= 0; --p) {
      int64_t role = dimRoles[p];
      bool isFloor = (role >= 0 &&
                      static_cast<CoordOp>(coords.op[p]) == CoordOp::FloorDiv);
      if (role == -1) {
        plan.reduceDims.push_back(p);
        if (plan.opInnerDim == -1) {
          plan.opInnerDim = p; // first -1 seen right-to-left = rightmost
          plan.opTileDims.push_back(p); // consumed by Op — part of the 2D tile
        } else {
          plan.loopDims.push_back(p); // outer -1 dims → loops
        }
      } else if (isFloor) {
        plan.floorDims.push_back(p);
      } else {
        // role >= 0, non-floor (identity parallel or lane)
        plan.opTileDims.push_back(p);
      }
    }
    // Reverse so dims appear in ascending physical order (left-to-right),
    // matching the expectations of buildExtract and collectLoops callers.
    std::reverse(plan.floorDims.begin(), plan.floorDims.end());
    std::reverse(plan.loopDims.begin(), plan.loopDims.end());
    std::reverse(plan.opTileDims.begin(), plan.opTileDims.end());

    // Assign the per-dim slice behavior from local geometry (R-c + R-b):
    //   floor / loopDims dim       → StickIndex (one stick by its own IV)
    //   opInnerDim spanning >1 stick → StickifiedBlock (B's K-flat; one stick per
    //     reduction iter). Detected purely by extent > stickSize, so a single-
    //     stick opInnerDim (incl. the case where opInnerDim IS the lane) is WholeBlock.
    //   everything else (lane, opSlice) → WholeBlock (taken whole in the 2D tile)
    plan.sliceKind.assign(rank, SliceKind::WholeBlock);
    auto markList = [&](ArrayRef<int> dims) {
      for (int p : dims)
        plan.sliceKind[p] = SliceKind::StickIndex;
    };
    markList(plan.floorDims);
    markList(plan.loopDims);
    if (plan.opInnerDim != -1 &&
        coords.physBlock[plan.opInnerDim] > plan.stickSize)
      plan.sliceKind[plan.opInnerDim] = SliceKind::StickifiedBlock;
    return plan;
  }

  // Trace a logical value forward to its ktdp.store, then walk back through
  // the store's access tile to the physical memView -> marker map.
  //
  // This is the mirror of findMarkerForOperand (which traces backward through
  // loads). The forward walk finds any ktdp.store that consumes `value`
  // (possibly through an elementwise chain). If the store's access tile
  // was physicalized by Phase 1, its base is in physMemViewToMarker.
  //
  // Returns the marker, or a null SpyreTensorLayoutOp if not found.
  triton::SpyreTensorLayoutOp findMarkerForStore(Value value) {
    // Walk the forward use chain: value → elementwise ops → ktdp.store.
    // We stop at any op that is not a single-tensor-result elementwise op.
    SmallVector<Value> worklist = {value};
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      for (auto *user : v.getUsers()) {
        if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
          // Found the store. Trace its access tile to the marker.
          auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
              st.getAccessTile().getDefiningOp());
          if (!tile)
            continue;
          auto it = physMemViewToMarker.find(tile.getBase());
          if (it != physMemViewToMarker.end())
            return it->second;
        }
        // Continue through single-tensor-result elementwise ops (e.g. truncf).
        if (!isSingleTensorElementwiseOp(user))
          continue;
        worklist.push_back(user->getResult(0));
      }
    }
    return {};
  }

  // Per-operand descriptor for a source contraction op.
  // canonicalAxes[i] = output-axis index for logical dim i, or -1 for reduction.
  // e.g. matmul A=(m,k): {0,-1};  B=(k,n): {-1,1}.
  struct SourceOperandSpec {
    SmallVector<int64_t> canonicalAxes; // one entry per logical dim of this operand
  };



  // Descriptor for one source contraction op (e.g. linalg.matmul).
  // `emitOp` receives all (possibly transposed) input slices + accumulator
  // and returns the updated accumulator. Acc shape is derived from each
  // operand's resolved opExtents + transposePerm fields by emitSourceStage.
  struct SourceOpSpec {
    SmallVector<SourceOperandSpec> operands;  // one per input
    unsigned logicalRank;
    function_ref<Value(OpBuilder &, Location, ArrayRef<Value> /*slices*/,
                       Value /*acc*/, RankedTensorType /*accTy*/)>
        emitOp;
  };

  // Generic source-stage entry point for N-input contraction ops. Iterates
  // spec.operands to trace each input to its marker, build OperandPlans, and
  // call emitSourceStage. Op-specific wiring comes from the spec; add a new
  // dispatchXxx (batch_matmul, conv) by building a different SourceOpSpec.
  // Build a plan for a scratchpad operand — a logical rank-2 tensor produced
  // by a prior contraction (e.g. tl.dot(B, C)) that has no descriptor and no
  // marker. The value is already in canonical op orientation (row-major [K, N])
  // so no transpose is needed and every dim is WholeBlock.
  //
  // Dispatch rule (called from dispatchSource when walkToLoad returns null):
  //   no load found + value is a contraction result → scratchpad (this path)
  //   no load found + value is something else       → error
  static OperandPlan classifyScratchpad(Value val,
                                        const SourceOperandSpec &opSpec) {
    auto tensorTy = cast<RankedTensorType>(val.getType());
    int rank = (int)tensorTy.getRank();
    OperandPlan plan;
    plan.value = val;
    plan.physBlockStorage.assign(tensorTy.getShape().begin(),
                                 tensorTy.getShape().end());
    // coords: src/op/arg unused (no marker); physBlock points at owned storage.
    plan.coords.src        = {};
    plan.coords.op         = {};
    plan.coords.arg        = {};
    plan.coords.logicalRank = (unsigned)rank;
    plan.coords.physBlock   = plan.physBlockStorage;

    plan.lane       = rank - 1;
    plan.stickSize  = tensorTy.getDimSize(rank - 1);
    plan.opInnerDim = -1;
    // All dims are opTileDims (dims 0 … rank-1, ascending), all WholeBlock.
    // dimRoles mirrors canonicalAxes so transposePerm derives correctly:
    //   canonical reduction dim → role -1, canonical parallel dim → its axis index.
    for (int p = 0; p < rank; ++p) {
      plan.opTileDims.push_back(p);
      plan.dimRoles.push_back(opSpec.canonicalAxes[p]);
    }
    plan.sliceKind.assign(rank, SliceKind::WholeBlock);

    // Resolved fields: no transpose (identity perm = empty); extents straight from shape.
    plan.transposePerm      = {};
    for (int p : plan.opTileDims)
      plan.opExtents.push_back(tensorTy.getDimSize(p));
    return plan;
  }

  // Classify one operand and populate plans[i]. Three outcomes:
  //   physical  — walkToLoad finds a load with a marker → classify()
  //   scratchpad — walkToLoad finds no load, value is a contraction →
  //                classifyScratchpad() (no marker, pass value whole)
  //   error     — load found but no marker, or no load and not a contraction
  template <typename OpT>
  LogicalResult dispatchSource(OpT op, const SourceOpSpec &spec) {
    unsigned nOps = spec.operands.size();
    SmallVector<OperandPlan, 2> plans(nOps);

    for (unsigned i = 0; i < nOps; ++i) {
      Value operand = op.getInputs()[i];
      auto [ld, transposePerm] = walkToLoad(operand);

      if (ld) {
        // Physical operand: load found — must have a marker.
        auto marker = findMarkerForOperand(operand);
        if (!marker)
          return op.emitError(
              "spyre_tensor_layout: physical operand load has no layout marker");
        auto physShape = cast<RankedTensorType>(operand.getType()).getShape();
        OperandCoords coords = OperandCoords::fromMarker(marker, spec.logicalRank,
                                                         physShape);
        // If a linalg.transpose sits between the load and this op, its perm τ
        // reorders the logical axes.  Compose τ into canonicalAxes so that
        // buildDimRoles sees the correct role for each physical opTileDim:
        //   newCanonicalAxes[i] = originalCanonicalAxes[τ[i]]
        SmallVector<int64_t> effectiveCanonicalAxes = spec.operands[i].canonicalAxes;
        if (transposePerm.has_value()) {
          const auto &tau = *transposePerm;
          assert(tau.size() == effectiveCanonicalAxes.size() &&
                 "transpose perm size must match canonicalAxes size");
          SmallVector<int64_t> reordered(effectiveCanonicalAxes.size());
          for (unsigned j = 0; j < tau.size(); ++j)
            reordered[j] = effectiveCanonicalAxes[tau[j]];
          effectiveCanonicalAxes = std::move(reordered);
        }
        SmallVector<int64_t> dimRoles;
        buildDimRoles(coords, effectiveCanonicalAxes, dimRoles);
        plans[i] = classify(operand, coords, dimRoles);
      } else {
        // No load found → scratchpad: a logical value flowing in from elsewhere
        // (a contraction result, or one carried out of an scf.for via iter_args,
        // etc.) that we consume whole. The real invariant is not "produced by a
        // contraction" but "already logical": a ranked tensor whose rank matches
        // this operand's logical rank (== canonicalAxes size). We process ops
        // inside-out, so whatever produced this operand has already been lowered
        // to its logical shape — exactly what this op expects. classifyScratchpad
        // indexes canonicalAxes per dim, so a rank mismatch would be UB — guard
        // it here with a clean error instead.
        auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorTy ||
            tensorTy.getRank() != (int64_t)spec.operands[i].canonicalAxes.size())
          return op.emitError(
              "spyre_tensor_layout: source op operand is neither a physical "
              "load nor a logical (scratchpad) tensor of the expected rank");
        plans[i] = classifyScratchpad(operand, spec.operands[i]);
      }
    }

    resolveAndReconcile(plans, spec);

    // D3 guard: reject parallel floor dims with physBlock > 1.
    // extractOpSlice uses stickIV (the reduction stick loop IV) as the offset
    // for StickIndex dims, which is only correct for reduction floors.  Parallel
    // floor dims with physBlock > 1 would need their own loop; until that is
    // implemented, fail early with a clear diagnostic.
    for (unsigned i = 0; i < nOps; ++i) {
      for (int p : plans[i].floorDims) {
        if (plans[i].dimRoles[p] >= 0 && plans[i].coords.physBlock[p] > 1)
          return op.emitError(
              "spyre_tensor_layout: source stage does not yet support "
              "multi-stick parallel floor dims (physBlock > 1 on a parallel "
              "axis)");
      }
    }

    return emitSourceStage(op, spec.emitOp, plans);
  }

  // Per-dim op-tile slice extent: stickSize for StickifiedBlock, full physBlock
  // otherwise. This is the extent of the op-tile dim before any transpose.
  static int64_t opSliceExtent(const OperandPlan &plan, int p) {
    return plan.sliceKind[p] == SliceKind::StickifiedBlock
               ? plan.stickSize
               : plan.coords.physBlock[p];
  }

  // Extract an op-tile stick slice from `plan`. Offsets/sizes follow sliceKind:
  //   StickIndex      → offset = stickIV (within-tile), size = 1
  //                     (only when physBlock[p] > 1; else offset = 0)
  //   StickifiedBlock → offset = stickIV * stickSize, size = stickSize
  //   WholeBlock      → offset = 0, size = physBlock[p]
  // `stickIV` is the current stick iteration variable (may be nullptr for
  // the trivial stickFactor=1 case where emitStickLoop passes a const-0).
  static Value extractOpSlice(OpBuilder &b, Location loc,
                               const OperandPlan &plan,
                               RankedTensorType resultTy, Value stickIV) {
    auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };
    ArrayRef<int64_t> physBlock = plan.coords.physBlock;
    int rank = (int)physBlock.size();
    SmallVector<OpFoldResult> offsets(rank), sizes(rank), strides(rank, idx(1));
    for (int p = 0; p < rank; ++p) {
      switch (plan.sliceKind[p]) {
      case SliceKind::StickIndex: {
        Value iv = (physBlock[p] > 1) ? stickIV : Value{};
        if (!iv) {
          offsets[p] = idx(0);
        } else if (iv.getType().isIndex()) {
          offsets[p] = iv;
        } else {
          offsets[p] = arith::IndexCastOp::create(b, loc,
                           b.getIndexType(), iv).getResult();
        }
        sizes[p] = idx(1);
        break;
      }
      case SliceKind::StickifiedBlock: {
        Value sIdx = stickIV.getType().isIndex()
                         ? stickIV
                         : arith::IndexCastOp::create(b, loc,
                               b.getIndexType(), stickIV).getResult();
        Value stickSz = arith::ConstantIndexOp::create(b, loc, plan.stickSize);
        offsets[p] = arith::MulIOp::create(b, loc, sIdx, stickSz).getResult();
        sizes[p]   = idx(plan.stickSize);
        break;
      }
      case SliceKind::WholeBlock:
        offsets[p] = idx(0);
        sizes[p]   = idx(physBlock[p]);
        break;
      }
    }
    return tensor::ExtractSliceOp::create(
        b, loc, resultTy, plan.value, offsets, sizes, strides);
  }

  // Resolution + cross-operand fix-up for N operand plans.
  //
  // 1. Resolve per-operand transpose + op-tile extents from the classified plan
  //    and the einsum spec, storing results into the plan's resolved fields so
  //    emitSourceStage reads them without re-consulting the spec or dimRoles.
  //
  // 2. StickifiedBlock demotion: StickifiedBlock is only valid when some plan
  //    has loopDims. When none do, demote all StickifiedBlock to WholeBlock so
  //    extractOpSlice never uses a null IV. (Must run after resolution because
  //    opExtents must reflect the demoted kind.)
  // Compute the permutation that reorders opTileDims from physical to canonical
  // axis order. Returns empty vector if already identity (no transpose needed).
  //
  // canonicalAxes defines the expected role at each logical position.
  // opTileDims are in physical order; each has role dimRoles[opTileDims[j]].
  // We reorder the physical dims so their roles match canonicalAxes.
  //
  // Algorithm: for each canonical position c, find which physical opTileDim
  // index j has the matching role. perm[j] = c (physical j goes to output c).
  // For reduction dims (role == -1), match left-to-right among unmatched.
  static SmallVector<int64_t> computeTransposePerm(
      ArrayRef<int> opTileDims,
      ArrayRef<int64_t> dimRoles,
      ArrayRef<int64_t> canonicalAxes) {
    unsigned nTile = opTileDims.size();
    SmallVector<int64_t> perm(nTile, -1);
    SmallVector<bool> used(nTile, false);

    // First pass: match parallel dims (role >= 0) — unique role values.
    for (unsigned c = 0; c < nTile; ++c) {
      int64_t canonRole = canonicalAxes[c];
      if (canonRole == -1) continue;
      for (unsigned j = 0; j < nTile; ++j) {
        if (!used[j] && dimRoles[opTileDims[j]] == canonRole) {
          perm[j] = (int64_t)c;
          used[j] = true;
          break;
        }
      }
    }
    // Second pass: match reduction dims (role == -1) left-to-right.
    for (unsigned c = 0; c < nTile; ++c) {
      if (canonicalAxes[c] != -1) continue;
      for (unsigned j = 0; j < nTile; ++j) {
        if (!used[j] && dimRoles[opTileDims[j]] == -1) {
          perm[j] = (int64_t)c;
          used[j] = true;
          break;
        }
      }
    }
    // Check if identity.
    bool isIdentity = true;
    for (unsigned j = 0; j < nTile; ++j)
      if (perm[j] != (int64_t)j) { isIdentity = false; break; }
    return isIdentity ? SmallVector<int64_t>{} : perm;
  }

  // Invert a permutation vector.
  static SmallVector<int64_t> invertPerm(ArrayRef<int64_t> perm) {
    SmallVector<int64_t> inv(perm.size());
    for (unsigned i = 0; i < perm.size(); ++i)
      inv[perm[i]] = i;
    return inv;
  }

  // Resolve per-operand transpose, extents, and output-axis contribution. All
  // einsum knowledge (canonicalAxes) is consumed here; emitSourceStage works
  // from resolved fields only and never touches the spec again.
  static void resolveAndReconcile(SmallVectorImpl<OperandPlan> &plans,
                                   const SourceOpSpec &spec) {
    // Step 1 — resolve per-operand fields.
    for (unsigned i = 0; i < plans.size(); ++i) {
      OperandPlan &plan = plans[i];
      const SourceOperandSpec &opSpec = spec.operands[i];

      plan.transposePerm = computeTransposePerm(
          plan.opTileDims, plan.dimRoles, opSpec.canonicalAxes);

      plan.opExtents.clear();
      for (int p : plan.opTileDims)
        plan.opExtents.push_back(opSliceExtent(plan, p));
    }

    // Step 2 — StickifiedBlock demotion.
    bool anyLoop = false;
    for (auto &p : plans)
      if (!p.loopDims.empty()) { anyLoop = true; break; }
    if (!anyLoop) {
      for (auto &plan : plans)
        for (auto &sk : plan.sliceKind)
          if (sk == SliceKind::StickifiedBlock)
            sk = SliceKind::WholeBlock;
      // Re-derive extents now that sliceKind has changed.
      for (unsigned i = 0; i < plans.size(); ++i) {
        OperandPlan &plan = plans[i];
        plan.opExtents.clear();
        for (int p : plan.opTileDims)
          plan.opExtents.push_back(opSliceExtent(plan, p));
      }
    }
  }

  // linalg.matmul instantiation:
  //   A=(m,k): dim0=M (output 0), dim1=K (reduction).
  //   B=(k,n): dim0=K (reduction), dim1=N (output 1).
  LogicalResult dispatchMatmul(linalg::MatmulOp mm) {
    SourceOpSpec spec;
    spec.operands = {SourceOperandSpec{{0, -1}},   // A=(m,k)
                     SourceOperandSpec{{-1, 1}}};  // B=(k,n)
    spec.logicalRank = 2;
    spec.emitOp = [](OpBuilder &b, Location loc,
                     ArrayRef<Value> slices, Value acc,
                     RankedTensorType accTy) -> Value {
      return linalg::MatmulOp::create(b, loc, accTy,
          ValueRange{slices[0], slices[1]}, ValueRange{acc}).getResult(0);
    };
    return dispatchSource(mm, spec);
  }

  // linalg.batch_matmul instantiation:
  //   A=(b,m,k): dim0=B (output 0), dim1=M (output 1), dim2=K (reduction).
  //   B=(b,k,n): dim0=B (output 0), dim1=K (reduction), dim2=N (output 2).
  LogicalResult dispatchBatchMatmul(linalg::BatchMatmulOp bmm) {
    SourceOpSpec spec;
    spec.operands = {SourceOperandSpec{{0, 1, -1}},   // A=(b,m,k)
                     SourceOperandSpec{{0, -1, 2}}};  // B=(b,k,n)
    spec.logicalRank = 3;
    spec.emitOp = [](OpBuilder &b, Location loc,
                     ArrayRef<Value> slices, Value acc,
                     RankedTensorType accTy) -> Value {
      return linalg::BatchMatmulOp::create(b, loc, accTy,
          ValueRange{slices[0], slices[1]}, ValueRange{acc}).getResult(0);
    };
    return dispatchSource(bmm, spec);
  }

  // linalg.reduce instantiation: generalized to any rank and any combiner.
  //   canonicalAxes maps each logical input dim to an output axis (or -1 for
  //   reduction dims).  After computeTransposePerm the slice has parallel dims
  //   first (0..outputRank-1) and reduction dims last (outputRank..sliceRank-1).
  LogicalResult dispatchReduce(linalg::ReduceOp rd) {
    // Derive logicalRank from the input operand's marker (phys_src encodes
    // which logical dim each physical dim derives from; logical rank =
    // max(phys_src) + 1).  The DPS init may already be physicalized by Phase 1
    // so its rank is the physical output rank, not the logical output rank.
    auto marker = findMarkerForOperand(rd.getInputs()[0]);
    if (!marker)
      return rd.emitError(
          "spyre_tensor_layout: dispatchReduce called but no marker on input");
    unsigned logicalRank = 0;
    for (int64_t src : marker.getPhysSrc())
      if ((unsigned)(src + 1) > logicalRank)
        logicalRank = (unsigned)(src + 1);
    auto reductionDims = rd.getDimensions();
    unsigned outputRank = logicalRank - (unsigned)reductionDims.size();

    // canonicalAxes: non-reduction dims get consecutive output axes; reduction → -1
    SmallVector<int64_t> canonicalAxes(logicalRank, -1);
    unsigned outAxis = 0;
    for (unsigned d = 0; d < logicalRank; ++d)
      if (!llvm::is_contained(reductionDims, (int64_t)d))
        canonicalAxes[d] = outAxis++;

    SourceOpSpec spec;
    spec.operands = {SourceOperandSpec{canonicalAxes}};
    spec.logicalRank = logicalRank;
    spec.emitOp = [outputRank](OpBuilder &b, Location loc,
                               ArrayRef<Value> slices, Value acc,
                               RankedTensorType accTy) -> Value {
      auto sliceTy = cast<RankedTensorType>(slices[0].getType());
      SmallVector<int64_t> dims;
      for (unsigned d = outputRank; d < (unsigned)sliceTy.getRank(); ++d)
        dims.push_back((int64_t)d);
      // TODO(S2): clone combiner region from original op — hardcoded addf for now
      return linalg::ReduceOp::create(
          b, loc, ValueRange{slices[0]}, ValueRange{acc}, dims,
          [&](OpBuilder &inner, Location iloc, ValueRange args) {
            Value sum = arith::AddFOp::create(inner, iloc, args[0], args[1]);
            linalg::YieldOp::create(inner, iloc, sum);
          }).getResult(0);
    };
    return dispatchSource(rd, spec);
  }

  // Source stage: for each operand in `plans`, extract its op slice from the
  // already-physicalized load, emit the contraction op (via spec.emitOp) at
  // op's position, RAUW the result, erase op. Phase 1 already rescaled and
  // wired the enclosing scf.for loops; the source stage emits only the slices
  // + op using those existing loops — no new loops, no IV recovery needed.
  template <typename OpT>
  LogicalResult emitSourceStage(
      OpT op,
      function_ref<Value(OpBuilder &, Location, ArrayRef<Value>, Value,
                         RankedTensorType)>
          emitOp,
      ArrayRef<OperandPlan> plans) {
    OpBuilder b(op);
    Location loc = op.getLoc();

    // getDpsInits() is the linalg DPS interface shared by all source ops
    // (MatmulOp, ReduceOp, …). It supersedes the op-specific getOutputs() /
    // getInits() accessors so emitSourceStage stays generic across op types.
    Value cVal = op.getDpsInits()[0];
    auto accElemTy = cast<RankedTensorType>(cVal.getType()).getElementType();

    // Per-operand op-tile slice types — derived from resolved extents.
    SmallVector<RankedTensorType> sliceTys;
    for (unsigned i = 0; i < plans.size(); ++i) {
      const OperandPlan &plan = plans[i];
      auto elemTy = cast<RankedTensorType>(plan.value.getType()).getElementType();
      sliceTys.push_back(RankedTensorType::get(plan.opExtents, elemTy));
    }

    // Derive acc shape from the union of all (outputAxis, extent) pairs.
    // For each plan, iterate opTileDims: dimRoles[p] >= 0 means parallel,
    // and its value is the output axis index. opSliceExtent gives the extent.
    int64_t maxAxis = -1;
    for (auto &plan : plans)
      for (unsigned j = 0; j < plan.opTileDims.size(); ++j) {
        int p = plan.opTileDims[j];
        int64_t role = plan.dimRoles[p];
        if (role >= 0 && role > maxAxis)
          maxAxis = role;
      }
    SmallVector<int64_t> accDims(maxAxis + 1, 0);
    for (auto &plan : plans)
      for (unsigned j = 0; j < plan.opTileDims.size(); ++j) {
        int p = plan.opTileDims[j];
        int64_t role = plan.dimRoles[p];
        if (role >= 0)
          accDims[role] = plan.opExtents[j];
      }
    auto accTy = RankedTensorType::get(accDims, accElemTy);

    // Transpose helper — emit linalg.transpose with the given permutation.
    // The pass stores perms in "input→output" form (perm[i] = output dim for
    // input dim i), but linalg.transpose uses "output←input" form (perm[i] =
    // input dim that feeds output dim i). Invert here so call sites stay simple.
    auto emitTranspose = [&](Value src, ArrayRef<int64_t> perm) -> Value {
      auto srcTy = cast<RankedTensorType>(src.getType());
      auto mlirPerm = invertPerm(perm);
      SmallVector<int64_t> outShape(mlirPerm.size());
      for (unsigned i = 0; i < mlirPerm.size(); ++i)
        outShape[i] = srcTy.getDimSize(mlirPerm[i]);
      auto outTy = RankedTensorType::get(outShape, srcTy.getElementType());
      Value empty = tensor::EmptyOp::create(b, loc, outTy.getShape(),
                                            srcTy.getElementType());
      return linalg::TransposeOp::create(b, loc, src, empty,
          b.getDenseI64ArrayAttr(mlirPerm)).getResult()[0];
    };

    // Determine the stick loop trip count (stickFactor).
    //
    // All operands of one op share the same logical reduction axis. A physical
    // dim p is the stickified floor of that axis iff:
    //   (a) dimRoles[p] == some logical reduction dim (canonicalAxes[logDim]==-1)
    //   (b) coords.op[p] == CoordOp::FloorDiv  (floor half of the stick pair)
    //   (c) physBlock[p] > 1                   (actually spans multiple sticks)
    //
    // For each plan, scan loopDims for such a dim and compute its trip count.
    // All plans that have a stickified floor dim must agree on the same factor
    // (they partition the same logical reduction axis). Assert at most one
    // stickified logical reduction axis today; to support multiple, emit one
    // nested stick loop per axis and move this derivation to resolveAndReconcile.
    //
    // NOTE: if stickFactor derivation is needed in more than one place, move it
    // into resolveAndReconcile and store it as a resolved field on SourceOpSpec.
    int64_t stickFactor = 1;
    for (auto &plan : plans) {
      for (int p : plan.loopDims) {
        // Only floor dims (FloorDiv) of the stick pair drive the loop.
        if (static_cast<CoordOp>(plan.coords.op[p]) != CoordOp::FloorDiv)
          continue;
        // dimRoles[p] is the logical dim index; must be a reduction dim.
        int64_t logDim = plan.dimRoles[p];
        if (logDim >= 0) // parallel dim in loopDims — shouldn't happen, skip
          continue;
        int64_t f;
        if (plan.sliceKind[p] == SliceKind::StickifiedBlock)
          f = plan.coords.physBlock[p] / plan.stickSize;
        else
          f = plan.coords.physBlock[p]; // StickIndex: physBlock = n_sticks
        if (f <= 1)
          continue; // single-stick, no loop needed
        if (stickFactor != 1 && stickFactor != f)
          llvm_unreachable("emitSourceStage: plans disagree on stickFactor — "
                           "mixed reduction stick counts not yet supported");
        stickFactor = f;
      }
    }

    // stickIV: within-tile stick index used by StickIndex / StickifiedBlock.
    Value stickIV;

    Value result = emitStickLoop(b, loc, stickFactor, cVal,
        [&](OpBuilder &bb, Value s, Value acc) {
      stickIV = s;
      OpBuilder saved = b;
      b = bb;
      SmallVector<Value> slices;
      for (unsigned i = 0; i < plans.size(); ++i) {
        auto elemTy =
            cast<RankedTensorType>(plans[i].value.getType()).getElementType();
        Value slicePhys = extractOpSlice(b, loc, plans[i], sliceTys[i], stickIV);
        slices.push_back(!plans[i].transposePerm.empty()
                             ? emitTranspose(slicePhys, plans[i].transposePerm)
                             : slicePhys);
      }
      Value r = emitOp(b, loc, slices, acc, accTy);
      b = saved;
      return r;
    });

    op.getResult(0).replaceAllUsesWith(result);
    op.erase();
    return success();
  }

  // Per-store entry point for the sink stage: read the D coord map from the
  // marker, build the OperandPlan for D (all dims parallel), and call
  // emitSinkStage. logRank is derived from the data tile's actual rank so
  // this handles both 2D outputs (matmul) and 1D outputs (reduce).
  LogicalResult dispatchSink(mlir::ktdp::StoreOp st,
                              triton::SpyreTensorLayoutOp marker) {
    // The access tile shape IS the physical block shape of D.
    auto tileTy = cast<mlir::ktdp::AccessTileType>(st.getAccessTile().getType());
    ArrayRef<int64_t> physBlock = tileTy.getShape();

    // Derive logical rank from the data tile (the value being stored).
    unsigned logRank =
        cast<RankedTensorType>(st.getDataTile().getType()).getRank();
    OperandCoords dC = OperandCoords::fromMarker(marker, logRank, physBlock);

    // D has no reduction dim: every physical dim's logical src is a parallel dim.
    // Roles[p] = phys_src[p] — the logical dim index it derives from.
    int physRank = (int)physBlock.size();
    SmallVector<int64_t> dimRoleD(physRank);
    for (int p = 0; p < physRank; ++p)
      dimRoleD[p] = marker.getPhysSrc()[p]; // logical dim index; all parallel

    OperandPlan dPlan = classify(st.getDataTile(), dC, dimRoleD);
    return emitSinkStage(st, dPlan);
  }

  // Sink stage: consume the LOGICAL data_tile from the store, synthesize a
  // scf.for nest that scatters it into a PHYSICAL D tensor matching the
  // (already physical) access tile, then redirect the store's data_tile to
  // the physical result.
  //
  // The store has NO reduction loop — every loop is parallel. The iter_arg
  // carries the accumulating physical D tensor. At the leaf body:
  //   1. Extract the logical C sub-slice for the current loop indices.
  //   2. Insert it into the physical D accumulator via tensor.insert_slice.
  //
  // Placement (R-e): emitted at the block of the logical data_tile, after its
  // last use. Since inputTile lives at the function body level (produced by the
  // matmul stage or by retypeChain from a load), the builder is set there by
  // OpBuilder(st) which positions before the store.
  LogicalResult emitSinkStage(mlir::ktdp::StoreOp st,
                               const OperandPlan &dPlan) {
    Value inputTile = st.getDataTile(); // logical input (rank-2)
    OpBuilder b(st);
    Location loc = st.getLoc();

    Type elemTy = cast<RankedTensorType>(inputTile.getType()).getElementType();

    // All physical accounting comes from the plan — no logShape lookups.
    ArrayRef<int64_t> physBlock = dPlan.coords.physBlock; // e.g. [Nstick, M, lane]
    int physRank = (int)physBlock.size();
    int64_t stickSize = physBlock[dPlan.lane]; // lane = innermost phys dim

    if (dPlan.floorDims.empty())
      return st.emitError(
          "spyre_tensor_layout: store sink stage requires at least one "
          "parallel floor dim in the output layout");
    if (!dPlan.loopDims.empty())
      return st.emitError(
          "spyre_tensor_layout: store sink stage: unexpected reduction dim");

    // Compute a full-rank permutation to reorder the logical input tile from
    // canonical (ascending logical-dim) order to the physical dim appearance
    // order. This ensures extracted slices match the physical sink layout.
    //
    // Build the permutation by scanning all physical dims (opTile + floor) in
    // ascending physical order and recording the logical dim they map to. The
    // order of first appearance of each logical dim gives the target position.
    unsigned logRank = dPlan.coords.logicalRank;
    // Compute sinkPerm as invertPerm(computeTransposePerm(opTileDims, dimRoles, iota(logRank))).
    // computeTransposePerm maps physical opTileDim order → canonical (ascending logical-dim)
    // order. Its inverse maps canonical → physical appearance order, which is what the sink
    // needs: source applies π (physical→canonical), sink applies π^-1 (canonical→physical).
    SmallVector<int64_t> sinkPerm;
    {
      SmallVector<int64_t> canonicalAxesD(logRank);
      std::iota(canonicalAxesD.begin(), canonicalAxesD.end(), 0);
      auto fwdPerm = computeTransposePerm(dPlan.opTileDims, dPlan.dimRoles,
                                          canonicalAxesD);
      if (!fwdPerm.empty()) {
        auto inv = invertPerm(fwdPerm);
        bool isIdentity = true;
        for (unsigned d = 0; d < logRank; ++d)
          if (inv[d] != (int64_t)d) { isIdentity = false; break; }
        if (!isIdentity)
          sinkPerm = std::move(inv);
      }
    }

    // logDimToPos[d] = position of logical dim d in the (possibly transposed)
    // input tile. Without transpose it is identity; with transpose it is sinkPerm.
    auto logDimToPos = [&](int64_t d) -> unsigned {
      return sinkPerm.empty() ? (unsigned)d : (unsigned)sinkPerm[d];
    };

    // Transpose helper — emit linalg.transpose with the given permutation.
    // The pass stores perms in "input→output" form (perm[i] = output dim for
    // input dim i), but linalg.transpose uses "output←input" form (perm[i] =
    // input dim that feeds output dim i). Invert here so call sites stay simple.
    auto emitTranspose = [&](Value src, ArrayRef<int64_t> perm) -> Value {
      auto srcTy = cast<RankedTensorType>(src.getType());
      auto mlirPerm = invertPerm(perm);
      SmallVector<int64_t> outShape(mlirPerm.size());
      for (unsigned i = 0; i < mlirPerm.size(); ++i)
        outShape[i] = srcTy.getDimSize(mlirPerm[i]);
      auto outTy = RankedTensorType::get(outShape, srcTy.getElementType());
      Value empty = tensor::EmptyOp::create(b, loc, outTy.getShape(),
                                            srcTy.getElementType());
      return linalg::TransposeOp::create(b, loc, src, empty,
          b.getDenseI64ArrayAttr(mlirPerm)).getResult()[0];
    };

    // If non-identity perm, transpose the input tile from canonical (ascending
    // logical-dim) order to physical dim appearance order before slicing.
    if (!sinkPerm.empty())
      inputTile = emitTranspose(inputTile, sinkPerm);

    auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

    // Build the initial physical sink accumulator (tensor.empty over physBlock).
    SmallVector<int64_t> sinkShape(physBlock.begin(), physBlock.end());
    Value physicalSink = tensor::EmptyOp::create(b, loc, sinkShape, elemTy);

    // Build extract/insert parameter base arrays from the physical plan.
    //
    // After the optional transpose, inputTile's dims are in physical appearance
    // order. We index offsets/sizes by logDimToPos(logDim) — the position of
    // each logical dim in the transposed tile.
    SmallVector<OpFoldResult> inputOffsetsBase(logRank, idx(0));
    SmallVector<OpFoldResult> inputSizesBase(logRank);
    SmallVector<OpFoldResult> inputStrides(logRank, idx(1));
    // Default: fill from opTileDims (physBlock[p] for each logical src).
    for (int p : dPlan.opTileDims) {
      int64_t logDim = dPlan.dimRoles[p];
      if (logDim >= 0 && (unsigned)logDim < logRank)
        inputSizesBase[logDimToPos(logDim)] = idx(physBlock[p]);
    }
    // Floor dims: size = stickSize on that logical dim (overridden in loop body).
    for (int p : dPlan.floorDims) {
      int64_t logDim = dPlan.dimRoles[p];
      if (logDim >= 0 && (unsigned)logDim < logRank)
        inputSizesBase[logDimToPos(logDim)] = idx(stickSize);
    }

    // Sink insert: offset 0 / full physBlock size for non-floor dims.
    SmallVector<OpFoldResult> sinkOffsetsBase(physRank, idx(0));
    SmallVector<OpFoldResult> sinkSizes(physRank);
    SmallVector<OpFoldResult> sinkStrides(physRank, idx(1));
    for (int p = 0; p < physRank; ++p)
      sinkSizes[p] = llvm::is_contained(dPlan.floorDims, p)
                         ? idx(1) : idx(physBlock[p]);

    // For each floor dim p: scatter inputTile's sticks into physicalSink.
    // tripCount = physBlock[p] (number of sticks along this dim — computed
    // from the coord map during Phase 1, no logShape division needed).
    Value acc = physicalSink;
    for (int p : dPlan.floorDims) {
      int64_t logDim = dPlan.dimRoles[p];
      if (logDim < 0 || (unsigned)logDim >= logRank) continue;

      unsigned tileDim = logDimToPos(logDim); // position in transposed tile
      int64_t tripCount = physBlock[p]; // sticks per tile along dim p
      Value stickSizeVal = arith::ConstantIndexOp::create(b, loc, stickSize);

      // Extract-result type: dimensions match the transposed tile layout.
      SmallVector<int64_t> slShape(logRank);
      for (int p2 : dPlan.opTileDims) {
        int64_t ld = dPlan.dimRoles[p2];
        if (ld >= 0 && (unsigned)ld < logRank)
          slShape[logDimToPos(ld)] = physBlock[p2];
      }
      slShape[tileDim] = stickSize;
      auto slTy = RankedTensorType::get(slShape, elemTy);

      acc = emitStickLoop(b, loc, tripCount, acc,
          [&](OpBuilder &bb, Value s, Value sinkAccumulator) -> Value {
            // Input-extract: s-th stick-slice of inputTile on the floor's dim.
            SmallVector<OpFoldResult> inOff = inputOffsetsBase;
            inOff[tileDim] =
                arith::MulIOp::create(bb, loc, s, stickSizeVal).getResult();
            Value inputSlice = tensor::ExtractSliceOp::create(
                bb, loc, slTy, inputTile, inOff, inputSizesBase, inputStrides);

            // Sink-insert: position s along floor dim p.
            SmallVector<OpFoldResult> sinkOff = sinkOffsetsBase;
            sinkOff[p] = s;
            return tensor::InsertSliceOp::create(
                bb, loc, inputSlice, sinkAccumulator, sinkOff, sinkSizes, sinkStrides);
          });
    }
    Value result = acc;

    // Redirect the store's data_tile to the physical result.
    st.getDataTileMutable().set(result);
    return success();
  }

  // Erase a marker and its now-dead bridge cast. Called in Phase 3 after
  // synthesizeContractions has finished reading the coord maps.
  void eraseMarker(triton::SpyreTensorLayoutOp marker) {
    if (!marker->getBlock())
      return;
    Value desc = marker.getDesc();
    auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
    marker.erase();
    if (castOp && castOp.use_empty())
      castOp.erase();
  }

  // Phase 1: physicalize one annotated descriptor (memView + access tiles +
  // loads). Does NOT erase the marker — Phase 2 still needs its coord map.
  //
  // What changes:
  //   1. ktdp.construct_memory_view  : logical -> physical shape/strides/type
  //   2. ktdp.construct_access_tile  : logical -> physical block + indices
  //   3. ktdp.load result tensor     : retyped to physical rank
  //   4. ktdp.store src tensor       : access tile redirected to physical tile
  //   5. elementwise ops downstream  : result tensors retyped (retypeChain)
  //      (stops at contraction ops; see isContractionOp)
  LogicalResult rewriteOnePhysicalize(triton::SpyreTensorLayoutOp marker) {
    Value desc = marker.getDesc();

    // The marker's desc operand must be a lowered descriptor (the bridge cast
    // left by LowerDescriptorMemory Walk 1). This holds as long as this pass
    // runs after LowerDescriptorMemory.
    if (!isLoweredDescriptor(desc))
      return marker.emitError(
          "spyre_tensor_layout: desc operand is not a lowered descriptor — "
          "pass must run after LowerDescriptorMemory");

    Value memView = getDescriptorMemView(desc);
    auto memViewOp = memView.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
    if (!memViewOp)
      return marker.emitError(
          "spyre_tensor_layout: cannot locate construct_memory_view behind cast");

    // Build OperandCoords from the marker's coord map. physBlock is
    // tile-specific and filled inside rewriteAccessTile from logBlock.
    OperandCoords coords;
    coords.src        = marker.getPhysSrc();
    coords.op         = marker.getPhysOp();
    coords.arg        = marker.getPhysArg();
    coords.logicalRank = cast<MemRefType>(memView.getType()).getRank();
    // physBlock is tile-specific; rewriteAccessTile computes it from logBlock.

    // Convenience aliases so the code below is unchanged.
    ArrayRef<int64_t> physSrc = coords.src;
    ArrayRef<int64_t> physOp  = coords.op;
    ArrayRef<int64_t> physArg = coords.arg;
    unsigned physRank = physSrc.size();

    // --- 1. Build physical construct_memory_view ---
    // The "coord map" (physSrc/physOp/physArg) defines the logical→physical
    // transformation per physical dim; see OperandPlan::coords for the same
    // triplet consumed by Phase 2.
    // Derive the coord map using the static sizes on the existing logical view.
    // Dynamic logical dims produce dynamic physical dims (floordiv) or are
    // resolved to the constant modulus (mod).
    Value newMemView;
    {
      OpBuilder b(memViewOp);
      Location loc = memViewOp.getLoc();
      MLIRContext *ctx = b.getContext();

      ArrayRef<int64_t> logStaticSizes   = memViewOp.getStaticSizes();
      ArrayRef<int64_t> logStaticStrides = memViewOp.getStaticStrides();
      // Capture dynamic size/stride SSA values into owning vectors before any
      // builder operations — ValueRange is a non-owning view and can be
      // invalidated if new ops cause MLIR to reallocate operand storage.
      SmallVector<Value> logDynSizes(memViewOp.getSizes().begin(),
                                     memViewOp.getSizes().end());
      SmallVector<Value> logDynStrides(memViewOp.getStrides().begin(),
                                       memViewOp.getStrides().end());

      // Build the logical -> dynamic-operand index maps so we can pick the
      // right SSA value when a physical dim needs a dynamic logical dim.
      // logDynIdx[i] = position in logDynSizes for logical dim i, or -1.
      SmallVector<int> logDynIdx(logStaticSizes.size(), -1);
      {
        int dynPos = 0;
        for (unsigned i = 0; i < logStaticSizes.size(); ++i)
          if (logStaticSizes[i] == ShapedType::kDynamic)
            logDynIdx[i] = dynPos++;
      }
      SmallVector<int> logDynStrideIdx(logStaticStrides.size(), -1);
      {
        int dynPos = 0;
        for (unsigned i = 0; i < logStaticStrides.size(); ++i)
          if (logStaticStrides[i] == ShapedType::kDynamic)
            logDynStrideIdx[i] = dynPos++;
      }

      SmallVector<int64_t> physStaticSizes;
      SmallVector<Value>   physDynSizes;
      SmallVector<int64_t> physStaticStrides;
      SmallVector<Value>   physDynStrides;

      for (unsigned k = 0; k < physRank; ++k) {
        int64_t src = physSrc[k];
        auto op = static_cast<CoordOp>(physOp[k]);
        int64_t arg = physArg[k];
        if (src < 0 || src >= (int64_t)logStaticSizes.size())
          return marker.emitError("spyre_tensor_layout: phys_src out of range");

        int64_t logSz = logStaticSizes[src];
        auto physSz = applyStatic(logSz, op, arg);
        if (physSz) {
          physStaticSizes.push_back(*physSz);
        } else {
          // Dynamic: must emit a runtime computation.
          physStaticSizes.push_back(ShapedType::kDynamic);
          if (op == CoordOp::FloorDiv) {
            // physDyn = logDynSizes[logDynIdx[src]] / arg
            if (logDynIdx[src] < 0)
              return marker.emitError(
                  "spyre_tensor_layout: expected dynamic size for floordiv dim");
            Value logDynSize = logDynSizes[logDynIdx[src]];
            // logDynSize is index type from arith.index_cast in buildBaseMemoryView
            // convert to i32 for divsi, or use index arithmetic
            Value argIdx = arith::ConstantOp::create(
                b, loc, b.getIndexAttr(arg));
            physDynSizes.push_back(
                arith::CeilDivSIOp::create(b, loc, logDynSize, argIdx).getResult());
          } else {
            // Identity with dynamic extent
            if (logDynIdx[src] < 0)
              return marker.emitError(
                  "spyre_tensor_layout: expected dynamic size for identity dim");
            physDynSizes.push_back(logDynSizes[logDynIdx[src]]);
          }
        }

        // Physical strides: row-major over the physical shape.
        // We derive them symbolically below after computing all sizes;
        // push placeholder for now.
        (void)logStaticStrides; (void)logDynStrides;
        (void)logDynStrideIdx;
      }

      // Compute physical strides from the logical strides via the coord map.
      //
      // For each physical dim p with phys_src[p]=s and op:
      //   Identity  → physStride[p] = logStride[s]
      //   FloorDiv(arg) → physStride[p] = logStride[s] * arg
      //                   (one stick-step = arg consecutive logical rows)
      //   Mod(arg)  → physStride[p] = logStride[s]
      //                   (one lane-step = 1 logical row)
      //
      // This is correct for any physical dim ordering and avoids the
      // row-major-over-physBlock assumption that breaks for stick-on-M.
      physStaticStrides.resize(physRank);
      physDynStrides.clear();
      {
        bool hasAnyDynStride = false;
        for (unsigned k = 0; k < physRank; ++k) {
          int64_t s = physSrc[k];
          auto op = static_cast<CoordOp>(physOp[k]);
          int64_t arg = physArg[k];
          int64_t logSt = logStaticStrides[s];

          if (logSt == ShapedType::kDynamic) {
            physStaticStrides[k] = ShapedType::kDynamic;
            hasAnyDynStride = true;
          } else if (op == CoordOp::FloorDiv) {
            physStaticStrides[k] = logSt * arg;
          } else {
            // Identity or Mod: stride equals the logical stride of the source dim.
            physStaticStrides[k] = logSt;
          }
        }
        if (hasAnyDynStride) {
          // Build SSA stride values for dynamic dims.
          int dynPos = 0;
          for (unsigned k = 0; k < physRank; ++k) {
            if (physStaticStrides[k] != ShapedType::kDynamic)
              continue;
            int64_t s = physSrc[k];
            auto op = static_cast<CoordOp>(physOp[k]);
            int64_t arg = physArg[k];
            if (logDynStrideIdx[s] < 0)
              return marker.emitError(
                  "spyre_tensor_layout: expected dynamic stride for dim");
            Value logDynSt = logDynStrides[logDynStrideIdx[s]];
            if (op == CoordOp::FloorDiv) {
              Value argVal = arith::ConstantOp::create(
                  b, loc, b.getIndexAttr(arg));
              physDynStrides.push_back(
                  arith::MulIOp::create(b, loc, logDynSt, argVal).getResult());
            } else {
              physDynStrides.push_back(logDynSt);
            }
            (void)dynPos++;
          }
        }
      }

      // Physical memref type.
      auto logMemrefType = cast<MemRefType>(memViewOp.getResult().getType());
      auto physMemrefType = MemRefType::get(physStaticSizes,
                                            logMemrefType.getElementType());
      auto memSpaceAttr = memViewOp.getMemorySpace();
      auto coordSet = IntegerSetAttr::get(buildRangeSetND(ctx, physStaticSizes));

      newMemView = mlir::ktdp::ConstructMemoryViewOp::create(
                      b, loc, physMemrefType,
                      memViewOp.getOffset(),
                      physDynSizes, physDynStrides,
                      physStaticSizes, physStaticStrides,
                      memSpaceAttr, coordSet)
                      .getResult();
    }

    // Record the physical memView -> marker mapping for Phase 2.
    physMemViewToMarker[newMemView] = marker;

    // --- 2. Rebuild each access tile that uses the old memView ---
    SmallVector<mlir::ktdp::ConstructAccessTilesOp> tiles;
    SmallVector<mlir::ktdp::ConstructIndirectAccessTilesOp> indirectTiles;
    for (auto *user : memView.getUsers()) {
      if (auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(user))
        tiles.push_back(tile);
      else if (auto tile =
                   dyn_cast<mlir::ktdp::ConstructIndirectAccessTilesOp>(user))
        indirectTiles.push_back(tile);
    }

    // Direct tiles only own the loop rescaling:
    //   - FloorDiv dim: traces index → scf.for IV; rescales loop to stick
    //     granularity + divides muli(iv, C) constants by the same factor.
    //     e.g. `for i in range(2)` + `offset = i*128` →
    //          `for i in range(0,4,2)` + `offset = i*64`
    //   - rescaledLoops guards against double-rescaling if multiple direct
    //     tiles in the same descriptor share the loop.
    //   - if the trace fails (trip-1 / non-loop index): falls back to divsi.
    for (auto tileOp : tiles) {
      if (failed(rewriteAccessTile(tileOp, newMemView, coords)))
        return failure();
    }
    // Indirect tiles (gather) rewrite only the affine subscript maps; the
    // stick math is baked into the map expressions. Loop rescaling is left to
    // the direct tiles above — the gather loop must also drive at least one
    // direct tile (e.g. the output store) so the loop is already rescaled here.
    for (auto tileOp : indirectTiles) {
      if (failed(rewriteIndirectAccessTile(tileOp, newMemView, coords)))
        return failure();
    }

    // --- 3/4/5. Retype loads/stores on the marker's descriptor ---
    //   After retileAccessTile, the old access tiles are gone and their users
    //   (ktdp.load / ktdp.store) have been updated. Any remaining ktdp.loads
    //   that still use the bridge cast's memView for other purposes stay as-is.

    // Marker and bridge cast are NOT erased here — Phase 2 (synthesizeContractions)
    // still needs to read the marker's coord map. Phase 3 calls eraseMarker.
    return success();
  }

  // After rescaleEnclosingLoop(iv, factor), the IV runs in stick-index space
  // (0, factor, 2*factor, ...). Any muli(iv, C) in the loop body that computes
  // a byte/element offset was written assuming the IV was a block index, so C
  // must be divided by factor to remain correct. This helper finds all direct
  // muli(iv, C) uses of iv where C is a constant multiple of factor and rewrites
  // the constant in place.
  void scaleDownIVMuls(BlockArgument iv, int64_t factor) {
    if (factor <= 1)
      return;
    for (Operation *user : llvm::make_early_inc_range(iv.getUsers())) {
      auto muli = dyn_cast<arith::MulIOp>(user);
      if (!muli || muli.getLhs() != iv)
        continue;
      auto cst = getConstantInt(muli.getRhs());
      if (!cst || (*cst % factor) != 0)
        continue;
      OpBuilder b(muli);
      Value newCst = arith::ConstantOp::create(
          b, muli.getLoc(),
          b.getIntegerAttr(muli.getRhs().getType(), *cst / factor));
      muli.getRhs().replaceAllUsesWith(newCst);
    }
  }

  // Rescale an scf.for loop to stick granularity.
  // Sets ub = old_ub * factor and step = factor (or step = 1 when factor == 1).
  // IV values become 0, factor, 2*factor, ... — tile-granular stick indices.
  // Phase 2 emits an inner loop over sticks within each tile.
  void rescaleEnclosingLoop(scf::ForOp forOp, int64_t factor) {
    Type ivTy = forOp.getInductionVar().getType();
    OpBuilder b(forOp);
    Location loc = forOp.getLoc();
    if (factor > 1) {
      Value factorV = arith::ConstantOp::create(b, loc,
                          b.getIntegerAttr(ivTy, factor));
      Value newUb = arith::MulIOp::create(b, loc,
                        forOp.getUpperBound(), factorV).getResult();
      forOp.setUpperBound(newUb);
      forOp.setStep(factorV);
    } else {
      Value c1v = arith::ConstantOp::create(b, loc,
                      b.getIntegerAttr(ivTy, 1));
      forOp.setStep(c1v);
    }
  }

  // Rebuild ConstructAccessTilesOp with the physical memView + block shape.
  // Erases the old tile; users (ktdp.load/store) are retargeted to the new tile.
  LogicalResult rewriteAccessTile(mlir::ktdp::ConstructAccessTilesOp tileOp,
                                  Value newMemView,
                                  const OperandCoords &coords) {
    ArrayRef<int64_t> physSrc = coords.src;
    ArrayRef<int64_t> physOp  = coords.op;
    ArrayRef<int64_t> physArg = coords.arg;

    OpBuilder b(tileOp);
    Location loc = tileOp.getLoc();
    MLIRContext *ctx = b.getContext();

    // The logical block shape is the existing tile's AccessTileType shape.
    auto logTileType = tileOp.getResult().getType();
    ArrayRef<int64_t> logBlock = logTileType.getShape();
    unsigned logRank = logBlock.size();
    unsigned physRank = physSrc.size();
    (void)logRank; // physRank may differ (stick dim expands 2->3)

    // Compute physical block shape via applyCoordMap.
    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, physSrc, physOp, physArg, physBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: cannot derive static block_shape");
    // Validate phys_src bounds.
    for (unsigned k = 0; k < physRank; ++k)
      if (physSrc[k] < 0 || physSrc[k] >= (int64_t)logRank)
        return tileOp.emitError("spyre_tensor_layout: phys_src out of range");

    // Validate stick width: a Mod (lane) dim with modulus `arg` requires its
    // source logical block extent to be at least `arg`. A sub-stick block
    // (logBlock[src] < arg) would pad the data into a full-width lane, and a
    // contraction over that dim would read garbage padding. Reject it here
    // rather than silently mislay (see Step 8b chained-matmul fixture).
    for (unsigned k = 0; k < physRank; ++k) {
      if (static_cast<CoordOp>(physOp[k]) != CoordOp::Mod)
        continue;
      int64_t logExtent = logBlock[physSrc[k]];
      if (logExtent != ShapedType::kDynamic && logExtent < physArg[k])
        return tileOp.emitError(
                   "spyre_tensor_layout: block extent of stick dim (")
               << logExtent << ") is smaller than the stick size ("
               << physArg[k] << "); a stick dim cannot be sub-stick";
    }

    // Map the logical index operands to physical index operands.
    // The existing tile's index operands are already index-typed (cast from i32
    // by buildDirectAccessTile in LowerDescriptorMemory).
    SmallVector<Value> logIdx(tileOp.getIndices().begin(),
                               tileOp.getIndices().end());
    // For FloorDiv (stick-index) dims: trace the logical index back to its
    // enclosing scf.for IV. If found, rescale the loop's trip by
    // factor = physBlock[k] (sticks per tile along this dim) and wire the IV
    // directly. This discards the muli(%iv, BLOCK_d) + divsi chain.
    // For Mod (lane) dims: always emit remsi (lane is never iterated).
    // For Identity: pass through.
    SmallVector<Value> physIdx;
    for (unsigned k = 0; k < physRank; ++k) {
      int64_t src = physSrc[k];
      auto op = static_cast<CoordOp>(physOp[k]);
      int64_t arg = physArg[k];
      Value logI = logIdx[src];
      switch (op) {
      case CoordOp::Identity:
        physIdx.push_back(logI);
        break;
      case CoordOp::FloorDiv: {
        // Attempt to trace logI back to an enclosing scf.for IV and rescale
        // that loop to stick granularity.
        BlockArgument iv = traceToMLIRBlockArg(logI);
        scf::ForOp forOp = iv ? dyn_cast_or_null<scf::ForOp>(
                                    iv.getOwner()->getParentOp())
                              : nullptr;
        if (forOp && forOp.getInductionVar() == iv) {
          if (rescaledLoops.insert(forOp).second) {
            rescaleEnclosingLoop(forOp, physBlock[k]);
            scaleDownIVMuls(iv, physBlock[k]);
          }
          // Wire the IV directly as the stick-index coord (tile-granular).
          // construct_access_tiles requires index-typed operands; cast if needed.
          Value ivIdx = iv.getType().isIndex()
                            ? iv
                            : arith::IndexCastOp::create(b, loc,
                                  b.getIndexType(), iv).getResult();
          physIdx.push_back(ivIdx);
        } else {
          // No enclosing loop found (trip-1 / inline dim): fall back to divsi.
          Value c = arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
          physIdx.push_back(
              arith::DivSIOp::create(b, loc, logI, c).getResult());
        }
        break;
      }
      case CoordOp::Mod: {
        Value c = arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
        physIdx.push_back(
            arith::RemSIOp::create(b, loc, logI, c).getResult());
        break;
      }
      }
    }

    auto physTileType = mlir::ktdp::AccessTileType::get(physBlock,
                                                         b.getIndexType());
    auto identityMap = AffineMap::getMultiDimIdentityMap(physRank, ctx);
    auto coordSet = buildRangeSetND(ctx, physBlock);

    auto newTile = mlir::ktdp::ConstructAccessTilesOp::create(
        b, loc, physTileType, newMemView,
        identityMap, physIdx, /*symbol_operands=*/ValueRange{},
        coordSet, identityMap);

    // Update consumers (ktdp.load / ktdp.store).
    for (auto *user : llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        retypeLoad(ld, newTile.getResult(), physBlock);
      } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        redirectStoreAccessTile(st, newTile.getResult());
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  // Capability gate for indirect-tile physicalization. This is the single
  // place that encodes WHICH gather layouts the rewrite below can express; the
  // mechanism in rewriteIndirectAccessTile is otherwise layout-agnostic (it
  // reconstructs any stick-split logical dim from its physical pieces). To
  // widen support, relax a clause here and add the corresponding fixture — the
  // emission code should not need to change for cases that remain affine.
  //
  // Currently supported envelope — matching `gather_kernel_spyre` (stick-on-N):
  //   * rank-2 gather `out[i,j] = in[idx[i], y+j]`;
  //   * logical dim 0 is INDIRECT (the gather-indexed row), dim 1 is DIRECT;
  //   * only the DIRECT dim may be stick-split.
  //
  // Why the indirect dim cannot be stick-split (a real limitation, not just
  // unimplemented): its coordinate is a *loaded* index value `idx[...]`, not an
  // affine function of the iteration variables, so floordiv/mod over the
  // iteration space cannot reconstruct it. Lifting that needs a different
  // mechanism (splitting the loaded value), so it stays rejected even as other
  // clauses relax.
  LogicalResult verifyIndirectPhysicalizable(
      mlir::ktdp::ConstructIndirectAccessTilesOp tileOp,
      const OperandCoords &coords, unsigned logRank, ArrayAttr oldKinds) const {
    if (logRank != 2)
      return tileOp.emitError(
          "spyre_tensor_layout: physicalizing an indirect access tile is only "
          "supported for a rank-2 gather (got logical rank ")
          << logRank << ")";
    if (!cast<BoolAttr>(oldKinds[0]).getValue() ||
        cast<BoolAttr>(oldKinds[1]).getValue())
      return tileOp.emitError(
          "spyre_tensor_layout: physicalizing an indirect access tile assumes "
          "logical dim 0 is indirect (gather) and logical dim 1 is direct; "
          "got a different subscript-kind layout");
    for (unsigned p = 0, e = coords.src.size(); p < e; ++p)
      if (coords.src[p] == 0 &&
          static_cast<CoordOp>(coords.op[p]) != CoordOp::Identity)
        return tileOp.emitError(
            "spyre_tensor_layout: stick-splitting the indirect (gather) row "
            "dim is not supported");
    return success();
  }

  // Rebuild ConstructIndirectAccessTilesOp over the physical (stick-tiled)
  // memView.
  //
  // Structure: a capability gate (verifyIndirectPhysicalizable) decides whether
  // the layout is one we can express; the rest is a layout-agnostic transform.
  // Each logical dim's index is reconstructed from whichever physical dims it
  // was split into, that reconstruction is substituted into the original
  // subscript, and the per-physical-dim CoordOp (FloorDiv/Mod/Identity) is
  // composed on top. Adding a new supported layout means relaxing the gate, not
  // rewriting the mechanism — provided the new case stays affine over the
  // iteration variables.
  //
  // The intermediate-variable space is rebuilt with one iteration variable per
  // PHYSICAL dim, so variables_space_set.numDims == result rank == physical
  // base rank (the op verifier requires this). A stick-split logical dim L
  // recombines as  j_L = stick_iter*arg + lane_iter; Identity dims keep their
  // single variable.
  LogicalResult rewriteIndirectAccessTile(
      mlir::ktdp::ConstructIndirectAccessTilesOp tileOp, Value newMemView,
      const OperandCoords &coords) {
    ArrayRef<int64_t> physSrc = coords.src;
    ArrayRef<int64_t> physOp  = coords.op;
    ArrayRef<int64_t> physArg = coords.arg;

    OpBuilder b(tileOp);
    Location loc = tileOp.getLoc();
    MLIRContext *ctx = b.getContext();

    unsigned physRank = physSrc.size();

    auto logTileType = tileOp.getResult().getType();
    ArrayRef<int64_t> logBlock = logTileType.getShape();
    unsigned logRank = logBlock.size();

    auto oldKinds = tileOp.getPerDimSubscriptKinds();
    auto oldMaps  = tileOp.getPerDimSubscriptMaps();
    unsigned numCaptured = tileOp.getCapturedVariables().size();

    if (failed(verifyIndirectPhysicalizable(tileOp, coords, logRank, oldKinds)))
      return failure();

    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, physSrc, physOp, physArg, physBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: cannot derive static block_shape for "
          "indirect access tile");

    // The old subscript maps are written over this affine domain:
    //   (captured_0 .. captured_{numCaptured-1},  i_0 .. i_{logRank-1})
    // where i_L is the iteration variable for logical dim L (e.g. the column
    // subscript "c_y + i_1"). After physicalization there is no single i_L per
    // logical dim anymore — a stick-tiled dim is split into two physical dims,
    // each with its own iteration variable. The new domain has one iteration
    // variable per PHYSICAL dim:
    //   (captured_0 .. captured_{numCaptured-1},  v_0 .. v_{physRank-1})
    // where v_p is the iteration variable for physical dim p (slot numCaptured+p).
    unsigned newDimCount = numCaptured + physRank;
    auto newVar = [&](unsigned slot) { return getAffineDimExpr(slot, ctx); };

    // To reuse the old subscripts we must re-express each old logical iteration
    // variable i_L as a function of the new physical iteration variables. That
    // function is `logicalFromPhysical[L]`: it recovers the logical index of
    // dim L from the physical dims that L was split into. A logical dim maps to
    // either:
    //   * one Identity physical dim          -> i_L = v_p
    //   * a (FloorDiv stick, Mod lane) pair   -> i_L = v_stick*arg + v_lane
    // We accumulate the pieces as we visit each contributing physical dim; the
    // stick piece adds v_stick*arg, the lane piece adds v_lane, so order doesn't
    // matter. `contributed[L]` tracks whether L already has a piece (so we add
    // rather than overwrite).
    SmallVector<AffineExpr> logicalFromPhysical(logRank);
    SmallVector<bool> contributed(logRank, false);
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t L = physSrc[p];
      if (L < 0 || L >= (int64_t)logRank)
        return tileOp.emitError(
            "spyre_tensor_layout: phys_src out of range for indirect tile");
      auto op = static_cast<CoordOp>(physOp[p]);
      int64_t arg = physArg[p];
      AffineExpr v = newVar(numCaptured + p); // this physical dim's iteration var

      // The piece this physical dim contributes to logical dim L's index.
      AffineExpr piece;
      switch (op) {
      case CoordOp::Identity: piece = v;       break; // whole dim, not split
      case CoordOp::FloorDiv: piece = v * arg; break; // stick index * stick size
      case CoordOp::Mod:      piece = v;       break; // lane offset within stick
      }

      logicalFromPhysical[L] =
          contributed[L] ? logicalFromPhysical[L] + piece : piece;
      contributed[L] = true;
    }

    // Substitution from the OLD affine domain into the NEW one:
    //   old captured slot c (0..numCaptured-1)  -> new captured slot c (unchanged)
    //   old iteration slot numCaptured + L       -> logicalFromPhysical[L]
    SmallVector<AffineExpr> oldToNew(numCaptured + logRank);
    for (unsigned c = 0; c < numCaptured; ++c)
      oldToNew[c] = newVar(c);
    for (unsigned L = 0; L < logRank; ++L)
      oldToNew[numCaptured + L] = logicalFromPhysical[L];

    // Build per-physical-dim kinds + maps.
    SmallVector<Attribute> newKinds, newMaps;
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t L = physSrc[p];
      auto op  = static_cast<CoordOp>(physOp[p]);
      int64_t arg = physArg[p];

      // The upfront scope guards guarantee logical dim 0 (indirect) is only
      // ever Identity-mapped here, so the kind carries over per physical dim.
      auto oldKindAttr = cast<BoolAttr>(oldKinds[L]);
      auto oldMapAttr  = cast<AffineMapAttr>(oldMaps[L]);

      // Re-express the old subscript in the new domain. The old map has no
      // symbols (gather subscripts are dim-only), so replaceDims suffices; the
      // resulting expression references new-domain dim slots, and the enclosing
      // AffineMap::get below sets the domain size to newDimCount.
      AffineExpr oldExpr = oldMapAttr.getValue().getResult(0);
      AffineExpr reExpr = oldExpr.replaceDims(oldToNew);

      // Compose the CoordOp for this physical dim.
      AffineExpr physExpr;
      switch (op) {
      case CoordOp::Identity: physExpr = reExpr; break;
      case CoordOp::FloorDiv: physExpr = reExpr.floorDiv(arg); break;
      case CoordOp::Mod:      physExpr = reExpr % arg; break;
      }

      newKinds.push_back(oldKindAttr);
      newMaps.push_back(AffineMapAttr::get(
          AffineMap::get(newDimCount, /*symbolCount=*/0, physExpr, ctx)));
    }

    // Build the new intermediate-variable space: one dim per physical dim,
    // each constrained to [0, physBlock[p]).
    SmallVector<AffineExpr> setConstraints;
    SmallVector<bool> setEqFlags;
    for (unsigned p = 0; p < physRank; ++p) {
      AffineExpr v = getAffineDimExpr(p, ctx);
      // 0 <= v
      setConstraints.push_back(v);
      setEqFlags.push_back(false);
      // v <= physBlock[p] - 1   <=>   (physBlock[p]-1) - v >= 0
      setConstraints.push_back(
          getAffineConstantExpr(physBlock[p] - 1, ctx) - v);
      setEqFlags.push_back(false);
    }
    auto newSpaceSet = IntegerSet::get(/*dimCount=*/physRank,
                                       /*symbolCount=*/0, setConstraints,
                                       setEqFlags);
    auto newSpaceOrder = AffineMap::getMultiDimIdentityMap(physRank, ctx);

    auto physTileType = mlir::ktdp::AccessTileType::get(physBlock,
                                                         b.getIndexType());

    auto newTile = mlir::ktdp::ConstructIndirectAccessTilesOp::create(
        b, loc, physTileType, newMemView,
        ArrayAttr::get(ctx, newKinds),
        ArrayAttr::get(ctx, newMaps),
        tileOp.getIndirectMemrefs(),
        tileOp.getCapturedVariables(),
        tileOp.getSymbolOperands(),
        newSpaceSet, newSpaceOrder);

    // Update consumers (ktdp.load only — indirect tiles are not stored to).
    for (auto *user : llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        retypeLoad(ld, newTile.getResult(), physBlock);
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of indirect access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  // Retype ktdp.load: replace with a new load of the physical tensor type,
  // then forward-retype the elementwise chain.
  void retypeLoad(mlir::ktdp::LoadOp ld, Value newTile,
                  ArrayRef<int64_t> physBlock) {
    OpBuilder b(ld);
    auto elemTy = cast<RankedTensorType>(ld.getResult().getType())
                      .getElementType();
    auto physResTy = RankedTensorType::get(physBlock, elemTy);
    auto newLd = mlir::ktdp::LoadOp::create(b, ld.getLoc(), physResTy, newTile);
    retypeChain(ld.getResult(), newLd.getResult());
    ld.erase();
  }

  // Redirect ktdp.store's access tile operand to the new physical tile.
  // Does NOT retype the store's data tensor — that is Phase 2's job (store sink
  // stage). The src tensor remains logical-rank until synthesizeContractions runs.
  void redirectStoreAccessTile(mlir::ktdp::StoreOp st, Value newTile) {
    st.getAccessTileMutable().set(newTile);
  }

  // True if the op is a contraction whose result shape is determined by its own
  // semantics rather than being inherited from a single physical input.
  // retypeChain stops here and does not propagate through.
  static bool isContractionOp(Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    return linalgOp && linalg::isaContractionOpInterface(linalgOp);
  }

  // Forward-retype the elementwise op chain: replace oldVal with newVal
  // everywhere and update result types of single-result ops that still carry
  // the old (logical-rank) type. Stops at contraction ops.
  void retypeChain(Value oldVal, Value newVal) {
    oldVal.replaceAllUsesWith(newVal);
    SmallVector<Operation *> worklist(newVal.getUsers().begin(),
                                      newVal.getUsers().end());
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (op->getNumResults() != 1)
        continue;
      if (isContractionOp(op))
        continue;
      if (isa<linalg::TransposeOp>(op))
        continue;
      auto resTy  = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      auto opndTy = op->getNumOperands() > 0
                        ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                        : nullptr;
      if (!resTy || !opndTy || resTy.getShape() == opndTy.getShape())
        continue;
      op->getResult(0).setType(
          RankedTensorType::get(opndTy.getShape(), resTy.getElementType()));
      worklist.append(op->getResult(0).getUsers().begin(),
                      op->getResult(0).getUsers().end());
    }
  }
};

} // namespace

} // namespace mlir::triton::ktdp
