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
// Loop / collectExistingLoopIVs utilities.
//
// Call graph:
//   runOnOperation
//     Phase 1 — for each marker:
//       rewriteOnePhysicalize(marker)
//         isLoweredDescriptor / getDescriptorMemView
//         buildPhysicalMemoryView   -> physical ConstructMemoryViewOp
//           applyStatic / applyCoordMap  derive physical static extents
//         rewriteAccessTile(tile, coords) -> physical ConstructAccessTilesOp
//           applyIndex              SSA index mapping per coord-op
//           rescaleEnclosingLoop    rescale enclosing scf.for to stick granularity
//         retypeLoad                -> physical ktdp.load + retypeChain
//           retypeChain             propagate type (stops at isContractionOp)
//         redirectStoreAccessTile   redirect access tile operand (output stays
//                                   logical-typed; sink stage fixes it in Phase 2)
//         (marker kept alive for Phase 2)
//     Phase 2 — synthesizeContractions (loop-until-stable):
//       source: dispatchMatmul(mm) → dispatchSource<MatmulOp>(mm, K-wiring)
//         findMarkerForOperand / walkToLoad  trace load chain -> marker
//         isSingleTensorElementwiseOp        shared predicate for all walks
//         classify                  build OperandPlan (dimRoles, floorDims, loopDims…)
//         emitSourceStage           walk existing loops (collectExistingLoopIVs),
//                                   extract slices + linalg.matmul; inner stick
//                                   loop (emitStickLoop) for StickifiedBlock
//       sink:   dispatchSink(st)   (only when output descriptor is annotated)
//         findMarkerForStore        forward worklist walk -> marker
//         classify / emitSinkStage  walk existing loops, scatter inputTile into
//                                   physicalSink via tensor.insert_slice
//     Phase 3 — eraseMarker (marker + dead bridge cast)
//
//   Coord helpers (free functions):
//     applyStatic     : compile-time extent for one dim
//     applyCoordMap   : compile-time extents for all physical dims
//     applyIndex      : SSA load/store offset (identity | divsi | remsi)
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

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

/// Like traceToMLIRBlockArg but also returns the product of all muli-by-constant
/// multipliers encountered on the way.  Other ops (cast, divsi, remsi, addi) are
/// treated as pass-through with multiplier 1.  Returns {nullptr, 1} if the chain
/// does not lead to a single BlockArgument.
static std::pair<BlockArgument, int64_t> traceToMLIRBlockArgWithStride(Value v) {
  int64_t stride = 1;
  while (true) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return {ba, stride};
    auto *op = v.getDefiningOp();
    if (!op)
      return {nullptr, 1};
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp,
            arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(op)) {
      v = op->getOperand(0);
      continue;
    }
    if (isa<arith::MulIOp>(op)) {
      if (auto c = getConstantInt(op->getOperand(1))) {
        stride *= *c;
        v = op->getOperand(0);
        continue;
      }
    }
    if (isa<arith::DivSIOp, arith::RemSIOp, arith::AddIOp>(op)) {
      if (op->getNumOperands() == 2 && getConstantInt(op->getOperand(1)))
        { v = op->getOperand(0); continue; }
    }
    return {nullptr, 1};
  }
}

/// Apply one coordinate op to an SSA index value.
///   identity -> the value unchanged
///   floordiv -> arith.divsi(value, arg)
///   mod      -> arith.remsi(value, arg)
static Value applyIndex(OpBuilder &builder, Location loc, Value logicalIdx,
                        CoordOp op, int64_t arg) {
  switch (op) {
  case CoordOp::Identity:
    return logicalIdx;
  case CoordOp::FloorDiv: {
    Value c = arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(static_cast<int32_t>(arg)));
    return arith::DivSIOp::create(builder, loc, logicalIdx, c).getResult();
  }
  case CoordOp::Mod: {
    Value c = arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(static_cast<int32_t>(arg)));
    return arith::RemSIOp::create(builder, loc, logicalIdx, c).getResult();
  }
  }
  llvm_unreachable("unhandled CoordOp");
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

  // Walk all linalg.matmul ops; fix any whose operands are rank-3 (physical).
  // Phase 2 driver: walk contraction ops whose operands have been physicalized
  // to rank > 2 and synthesize the scf.for loop nest that produces a 2D result.
  // Also walks ktdp.store ops whose output descriptor is annotated (data_tile
  // is logical but access_tile is already physical from Phase 1) and calls
  // dispatchSink to synthesize the scatter nest. Repeats until stable.
  // Currently handles linalg.matmul + ktdp.store; extend by adding a dispatch
  // branch for other contraction / sink ops as needed.
  LogicalResult synthesizeContractions(ModuleOp module) {
    // Onion-peeling: each iteration physicalizes one layer of the def chain.
    // Terminates because each iteration strictly reduces the number of
    // rank-mismatched ops (Phase 1 already handles single-tensor chains;
    // Phase 2 handles multi-tensor ops whose inputs were retyped by Phase 1
    // or by a prior Phase 2 iteration). The IR is finite so the loop converges.
    bool changed = true;
    while (changed) {
      changed = false;

      // --- matmul source stage ---
      SmallVector<linalg::MatmulOp> matmuls;
      module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
      for (auto mm : matmuls) {
        auto aType = dyn_cast<RankedTensorType>(mm.getInputs()[0].getType());
        auto bType = dyn_cast<RankedTensorType>(mm.getInputs()[1].getType());
        if (!aType || !bType)
          continue;
        if (aType.getRank() == 2 && bType.getRank() == 2)
          continue;
        if (failed(dispatchMatmul(mm)))
          return failure();
        changed = true;
      }

      // --- store sink stage ---
      SmallVector<mlir::ktdp::StoreOp> stores;
      module.walk([&](mlir::ktdp::StoreOp op) { stores.push_back(op); });
      for (auto st : stores) {
        // Check for rank mismatch: data_tile is logical (rank 2) but access_tile
        // is physical (rank 3). This mismatch is left by Phase 1 when the output
        // descriptor carries a tt.spyre_tensor_layout marker.
        auto dataTy = dyn_cast<RankedTensorType>(st.getDataTile().getType());
        auto tileTy = dyn_cast<mlir::ktdp::AccessTileType>(
            st.getAccessTile().getType());
        if (!dataTy || !tileTy)
          continue;
        int dataRank = dataTy.getRank();
        int tileRank = (int)tileTy.getShape().size();
        if (dataRank == tileRank)
          continue; // already consistent — unannotated path or already lowered
        // Only proceed when there is a layout marker for the output descriptor.
        auto marker = findMarkerForStore(st.getDataTile());
        if (!marker)
          continue;
        if (failed(dispatchSink(st, marker)))
          return failure();
        changed = true;
      }
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

  // Walk backward from `val` through single-tensor elementwise ops to the
  // ktdp.load that produced it. Returns the load, or null if not found.
  static mlir::ktdp::LoadOp walkToLoad(Value val) {
    Value v = val;
    while (true) {
      auto *defOp = v.getDefiningOp();
      if (!defOp)
        return {};
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(defOp))
        return ld;
      if (!isSingleTensorElementwiseOp(defOp))
        return {};
      for (auto operand : defOp->getOperands())
        if (isa<RankedTensorType>(operand.getType())) { v = operand; break; }
    }
  }

  // Walk back from a matmul operand through the elementwise chain to the
  // ktdp.load, then look up the physical memView -> marker map populated
  // during Phase 1.
  triton::SpyreTensorLayoutOp findMarkerForOperand(Value operand) {
    auto ld = walkToLoad(operand);
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
  };

  // Assign a role to each physical dim of an operand:
  //   >= 0  : parallel dim, maps to C dim [value]
  //   -1    : reduction dim — consumed by Op (inner) or looped (outer)
  //
  // All physical dims whose logical source is in `consumedLogicalDims` get
  // role -1. classify() then splits them right-to-left: the rightmost is
  // opInnerDim (consumed by Op); the rest are loopDims (scf.for reduction loops).
  //
  // `parallelRole` is the role assigned to non-consumed dims — typically the
  // logical C dim index they contribute to (0=M, 1=N, etc.).
  static void buildDimRoles(const OperandCoords &coords,
                             ArrayRef<int64_t> consumedLogicalDims,
                             int64_t parallelRole,
                             SmallVectorImpl<int64_t> &roles) {
    int n = (int)coords.src.size();
    roles.resize(n);
    for (int p = 0; p < n; ++p) {
      bool isConsumed = llvm::is_contained(consumedLogicalDims, coords.src[p]);
      roles[p] = isConsumed ? -1 : parallelRole;
    }
  }

  // Single-dim convenience overload for the common case (matmul: one K dim).
  static void buildDimRoles(const OperandCoords &coords, int64_t kLogicalSrc,
                             int64_t parallelRole,
                             SmallVectorImpl<int64_t> &roles) {
    buildDimRoles(coords, ArrayRef<int64_t>{kLogicalSrc}, parallelRole, roles);
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

    // Threaded from the physical construct_memory_view (R-a). Populated by
    // dispatchMatmul / dispatchSink after classify(); consumed by S5.
    SmallVector<OpFoldResult> strides; // physical stride per dim (attr=static, Value=dynamic)
    SmallVector<OpFoldResult> sizes;   // physical extent per dim (attr=static, Value=dynamic)
  };

  // Same walk as findMarkerForOperand, but returns the physical
  // ConstructMemoryViewOp rather than the marker. Used by dispatchMatmul to
  // thread strides/sizes into the OperandPlan (S4).
  mlir::ktdp::ConstructMemoryViewOp getMemViewForOperand(Value operand) {
    auto ld = walkToLoad(operand);
    if (!ld)
      return {};
    auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
        ld.getAccessTile().getDefiningOp());
    if (!tile)
      return {};
    return tile.getBase().getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
  }

  // Populate plan.strides and plan.sizes from the physical ConstructMemoryViewOp
  // that backs the operand's load. Static dims become IndexAttr; dynamic dims
  // become the SSA Value from the memViewOp's dynamic operand lists.
  // Also asserts static consistency: sizes[p] == coords.physBlock[p] for every
  // statically-known dim (catches mismatches between the tensor type and the
  // memView before S5 flips consumers to the threaded values).
  static LogicalResult threadStridesAndSizes(
      OperandPlan &plan, mlir::ktdp::ConstructMemoryViewOp memViewOp,
      MLIRContext *ctx) {
    ArrayRef<int64_t> staticSizes   = memViewOp.getStaticSizes();
    ArrayRef<int64_t> staticStrides = memViewOp.getStaticStrides();
    auto dynSizes   = memViewOp.getSizes();
    auto dynStrides = memViewOp.getStrides();

    int physRank = (int)staticSizes.size();
    plan.sizes.resize(physRank);
    plan.strides.resize(physRank);

    // TODO(S5): Revisit whether a consistency check between plan.sizes and
    // coords.physBlock is appropriate here. They serve different purposes:
    // - plan.sizes[p] (from memViewOp) = full descriptor extent = trip count
    //   for floor dims, or full physical extent for opSlice/lane dims.
    // - coords.physBlock[p] (from the load result tensor type) = block extent
    //   = 1 for floor dims (one stick per block), physBlock[p] for opSlice.
    // They are NOT equal in general (e.g. floor dim: 1 vs N_sticks; M-row dim:
    // BM vs M_total). A check needs to verify each dim category independently.
    int dynSzPos = 0, dynStPos = 0;
    for (int p = 0; p < physRank; ++p) {
      if (staticSizes[p] != ShapedType::kDynamic) {
        plan.sizes[p] = IntegerAttr::get(IndexType::get(ctx), staticSizes[p]);
      } else {
        plan.sizes[p] = dynSizes[dynSzPos++];
      }
      if (staticStrides[p] != ShapedType::kDynamic) {
        plan.strides[p] = IntegerAttr::get(IndexType::get(ctx), staticStrides[p]);
      } else {
        plan.strides[p] = dynStrides[dynStPos++];
      }
    }
    return success();
  }

  // One loop dim whose IV is needed by extractOpSlice / the store insert.
  // Collected from the existing scf.for nest by collectExistingLoopIVs.
  struct Loop {
    enum Kind { Parallel, Reduction } kind;
    int   owner;    // which operand (index into plans vector)
    int   physDim;  // phys dim the IV indexes
    Value iv;       // induction variable
  };

  // Walk enclosing scf.for ops of `op` and populate `loops` with one entry per
  // floor/loopDims dim in aPlan/bPlan whose IV was wired by Phase 1.
  // Phase 1 stored the (index-cast) loop IV directly as the coord operand of
  // the construct_access_tile for each FloorDiv dim; we recover it by tracing
  // each coord operand back through index_cast to its BlockArgument.
  // Trace an index Value through index_cast to a BlockArgument that is the
  // induction variable (arg #0) of an enclosing scf.for.
  static Value traceToForIV(Value v) {
    while (true) {
      if (auto ba = dyn_cast<BlockArgument>(v))
        return (isa<scf::ForOp>(ba.getOwner()->getParentOp()) &&
                ba.getArgNumber() == 0)
                   ? ba
                   : Value{};
      auto *op = v.getDefiningOp();
      if (!op) return {};
      if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
        { v = op->getOperand(0); continue; }
      return {};
    }
  }

  // Trace a plan's tensor value back through single-tensor ops to the
  // ktdp.load that produced it, then return the ConstructAccessTilesOp
  // feeding that load.
  static mlir::ktdp::ConstructAccessTilesOp
  tileForPlan(const OperandPlan &plan) {
    if (!plan.value) return {};
    auto ld = walkToLoad(plan.value);
    if (!ld) return {};
    return dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
        ld.getAccessTile().getDefiningOp());
  }

  // Scan one access tile's coord operands and record which enclosing scf.for
  // IVs drive each physical dim. Populates ivToPhysDim for use by
  // collectLoopIVsForTile.
  static void scanTileIVs(
      mlir::ktdp::ConstructAccessTilesOp tile,
      DenseMap<Value, int /*physDim*/> &ivToPhysDim) {
    if (!tile) return;
    auto indices = tile.getIndices();
    for (unsigned k = 0; k < indices.size(); ++k) {
      Value forIV = traceToForIV(indices[k]);
      if (forIV && !ivToPhysDim.count(forIV))
        ivToPhysDim[forIV] = (int)k;
    }
  }

  // Walk the enclosing scf.for nest of `op` and collect Loop entries for
  // every IV that drives a physical dim of `tile`. `owner` is stored on
  // each Loop for the caller's use. `loopDims` dims (if any) are marked
  // Loop::Reduction; all others are Loop::Parallel.
  static void collectLoopIVsForTile(
      Operation *op, mlir::ktdp::ConstructAccessTilesOp tile, int owner,
      ArrayRef<int> loopDims, SmallVectorImpl<Loop> &loops) {
    DenseMap<Value, int> ivToPhysDim;
    scanTileIVs(tile, ivToPhysDim);

    Operation *cur = op->getParentOp();
    while (cur) {
      if (auto forOp = dyn_cast<scf::ForOp>(cur)) {
        Value iv = forOp.getInductionVar();
        auto it = ivToPhysDim.find(iv);
        if (it != ivToPhysDim.end()) {
          int physDim = it->second;
          Loop::Kind kind = Loop::Parallel;
          for (int p : loopDims)
            if (p == physDim) { kind = Loop::Reduction; break; }
          loops.push_back({kind, owner, physDim, iv});
        }
      }
      cur = cur->getParentOp();
    }
  }

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
  struct SourceOperandSpec {
    SmallVector<int64_t> consumedLogicalDims; // reduction dims (≥1 for conv-ready)
    int64_t              parallelRole;        // logical C dim index this input maps to
    // Whether to transpose the physical slice before passing to the op.
    // `reductionFirst` = true  → transpose when dimRoles[dimLo] == -1 (A-like)
    // `reductionFirst` = false → transpose when dimRoles[dimLo] != -1 (B-like)
    bool reductionFirst = false;
    bool needsTranspose(int64_t dimLoRole) const {
      return reductionFirst ? (dimLoRole == -1) : (dimLoRole != -1);
    }
  };

  // Descriptor for one source contraction op (e.g. linalg.matmul).
  // `emitOp` receives all (possibly transposed) input slices + accumulator
  // and returns the updated accumulator. `computeAccShape` derives the 2D
  // accumulator type from the plan vector (called after classify()).
  struct SourceOpSpec {
    SmallVector<SourceOperandSpec> operands;  // one per input
    unsigned logicalRank;
    function_ref<Value(OpBuilder &, Location, ArrayRef<Value> /*slices*/,
                       Value /*acc*/, RankedTensorType /*accTy*/)>
        emitOp;
    function_ref<RankedTensorType(ArrayRef<OperandPlan> /*plans*/,
                                  Type /*accElemTy*/)>
        computeAccShape;
  };

  // Generic source-stage entry point for N-input contraction ops. Iterates
  // spec.operands to trace each input to its marker, build OperandPlans, and
  // call emitSourceStage. Op-specific wiring comes from the spec; add a new
  // dispatchXxx (batch_matmul, conv) by building a different SourceOpSpec.
  template <typename OpT>
  LogicalResult dispatchSource(OpT op, const SourceOpSpec &spec) {
    unsigned nOps = spec.operands.size();
    SmallVector<OperandPlan, 2> plans(nOps);
    MLIRContext *ctx = op.getContext();

    for (unsigned i = 0; i < nOps; ++i) {
      auto marker = findMarkerForOperand(op.getInputs()[i]);
      if (!marker)
        return op.emitError(
            "spyre_tensor_layout: cannot find layout marker for source op operand");
      auto physShape =
          cast<RankedTensorType>(op.getInputs()[i].getType()).getShape();
      OperandCoords coords{marker.getPhysSrc(), marker.getPhysOp(),
                           marker.getPhysArg(), spec.logicalRank, physShape};
      SmallVector<int64_t> dimRoles;
      buildDimRoles(coords, spec.operands[i].consumedLogicalDims,
                    spec.operands[i].parallelRole, dimRoles);
      plans[i] = classify(op.getInputs()[i], coords, dimRoles);
    }

    reconcilePlans(plans);

    // S4: thread physical strides/sizes from the construct_memory_view into
    // the plans. Not yet consumed by emitSourceStage (S5 flips the switch).
    for (unsigned i = 0; i < nOps; ++i) {
      auto memView = getMemViewForOperand(op.getInputs()[i]);
      if (!memView)
        return op.emitError(
            "spyre_tensor_layout: cannot locate physical memory view for source op operand");
      if (failed(threadStridesAndSizes(plans[i], memView, ctx)))
        return failure();
    }

    return emitSourceStage(op, spec, plans);
  }

  // Cross-operand SliceKind fix-up (N-ary): StickifiedBlock is only valid when
  // any plan has a loopDims. When none do, demote all StickifiedBlock to
  // WholeBlock so extractOpSlice never calls getReduceIV() on a null IV.
  static void reconcilePlans(SmallVectorImpl<OperandPlan> &plans) {
    bool anyLoop = false;
    for (auto &p : plans)
      if (!p.loopDims.empty()) { anyLoop = true; break; }
    if (anyLoop) return;
    for (auto &plan : plans)
      for (auto &sk : plan.sliceKind)
        if (sk == SliceKind::StickifiedBlock)
          sk = SliceKind::WholeBlock;
  }

  // linalg.matmul instantiation:
  //   A logical: dim0=M (src 0 → parallel=0), dim1=K (src 1 → reduction=-1).
  //   B logical: dim0=K (src 0 → reduction=-1), dim1=N (src 1 → parallel=1).
  LogicalResult dispatchMatmul(linalg::MatmulOp mm) {
    SourceOpSpec spec;
    // A: K is src 1, M is parallel=0, reductionFirst=true (transpose when K is low dim)
    // B: K is src 0, N is parallel=1, reductionFirst=false (transpose when N is low dim)
    spec.operands = {{{1}, 0, /*reductionFirst=*/true},
                     {{0}, 1, /*reductionFirst=*/false}};
    spec.logicalRank = 2;
    spec.emitOp = [](OpBuilder &b, Location loc,
                     ArrayRef<Value> slices, Value acc,
                     RankedTensorType accTy) -> Value {
      return linalg::MatmulOp::create(b, loc, accTy,
          ValueRange{slices[0], slices[1]}, ValueRange{acc}).getResult(0);
    };
    spec.computeAccShape = [](ArrayRef<OperandPlan> plans,
                               Type accElemTy) -> RankedTensorType {
      const OperandPlan &aPlan = plans[0], &bPlan = plans[1];
      ArrayRef<int64_t> aPhys = aPlan.coords.physBlock;
      ArrayRef<int64_t> bPhys = bPlan.coords.physBlock;
      int aDimLo = aPlan.opTileDims[0], aDimHi = aPlan.opTileDims[1];
      int bDimLo = bPlan.opTileDims[0], bDimHi = bPlan.opTileDims[1];
      // A: transpose when K is the low dim (reductionFirst=true)
      bool transposeA = (aPlan.dimRoles[aDimLo] == -1);
      // B: transpose when N is the low dim (reductionFirst=false)
      bool transposeB = (bPlan.dimRoles[bDimLo] != -1);
      int64_t M = transposeA ? aPhys[aDimHi] : aPhys[aDimLo];
      int64_t N = transposeB ? bPhys[bDimLo] : bPhys[bDimHi];
      return RankedTensorType::get({M, N}, accElemTy);
    };
    return dispatchSource(mm, spec);
  }

  // Source stage: for each operand in `plans`, extract its op slice from the
  // already-physicalized load, emit the contraction op (via spec.emitOp) at
  // op's position, RAUW the result, erase op. Phase 1 already rescaled and
  // wired the enclosing scf.for loops; Phase 2 reads their IVs via
  // collectExistingLoopIVs and emits only the slices + op — no new loops.
  template <typename OpT>
  LogicalResult emitSourceStage(OpT op, const SourceOpSpec &spec,
                                ArrayRef<OperandPlan> plans) {
    OpBuilder b(op);
    Location loc = op.getLoc();

    Value cVal = op.getOutputs()[0];
    auto accElemTy = cast<RankedTensorType>(cVal.getType()).getElementType();

    // Each operand must have exactly 2 opSlice dims (the op tile).
    for (unsigned i = 0; i < plans.size(); ++i)
      if (plans[i].opTileDims.size() != 2)
        return op.emitError(
            "spyre_tensor_layout: expected exactly 2 opSlice dims per operand");

    // Per-operand op-tile slice types and transpose flags.
    // Transpose rule (per SourceOperandSpec.reductionFirst):
    //   reductionFirst=true  (A-like): transpose when dimRoles[dimLo] == -1
    //                                  (reduction dim is physically first)
    //   reductionFirst=false (B-like): transpose when dimRoles[dimLo] != -1
    //                                  (parallel dim is physically first)
    auto opSliceExtent = [](const OperandPlan &plan, int p) -> int64_t {
      return plan.sliceKind[p] == SliceKind::StickifiedBlock
                 ? plan.stickSize
                 : plan.coords.physBlock[p];
    };
    SmallVector<RankedTensorType> sliceTys;
    SmallVector<bool> transposeFlags;
    for (unsigned i = 0; i < plans.size(); ++i) {
      const OperandPlan &plan = plans[i];
      int dimLo = plan.opTileDims[0], dimHi = plan.opTileDims[1];
      auto elemTy = cast<RankedTensorType>(plan.value.getType()).getElementType();
      sliceTys.push_back(RankedTensorType::get(
          {opSliceExtent(plan, dimLo), opSliceExtent(plan, dimHi)}, elemTy));
      // Use the spec's per-operand transpose hint.
      transposeFlags.push_back(spec.operands[i].needsTranspose(plan.dimRoles[dimLo]));
    }

    auto accTy = spec.computeAccShape(plans, accElemTy);

    auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

    auto emitTranspose2D = [&](Value src, Type elemT) -> Value {
      auto srcTy = cast<RankedTensorType>(src.getType());
      auto outTy = RankedTensorType::get(
          {srcTy.getDimSize(1), srcTy.getDimSize(0)}, elemT);
      Value empty = tensor::EmptyOp::create(b, loc, outTy.getShape(), elemT);
      return linalg::TransposeOp::create(b, loc, src, empty,
          b.getDenseI64ArrayAttr({1, 0})).getResult()[0];
    };

    // Collect IVs from the existing enclosing scf.for loops (wired by Phase 1).
    // Use the first plan's loopDims as the reduction-dim reference (all plans
    // share the same reduction loop structure for a given contraction op).
    SmallVector<Loop> loops;
    for (unsigned i = 0; i < plans.size(); ++i)
      collectLoopIVsForTile(op, tileForPlan(plans[i]), (int)i,
                            plans[0].loopDims, loops);

    // Determine if a StickifiedBlock dim exists and what factor it has.
    int64_t stickFactor = 1;
    for (auto &plan : plans)
      for (int p = 0; p < (int)plan.sliceKind.size(); ++p)
        if (plan.sliceKind[p] == SliceKind::StickifiedBlock) {
          int64_t f = plan.coords.physBlock[p] / plan.stickSize;
          if (f > stickFactor) stickFactor = f;
        }

    // stickIV: within-tile stick index used by StickIndex / StickifiedBlock.
    Value stickIV;

    // Extract an op-tile stick slice from `plan`. Offsets/sizes follow sliceKind:
    //   StickIndex      → offset = stickIV (within-tile), size = 1
    //                     (only when physBlock[p] > 1; else offset = 0)
    //   StickifiedBlock → offset = stickIV * stickSize, size = stickSize
    //   WholeBlock      → offset = 0, size = physBlock[p]
    auto extractOpSlice = [&](const OperandPlan &plan,
                               RankedTensorType resultTy) -> Value {
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
    };

    Value result = emitStickLoop(b, loc, stickFactor, cVal,
        [&](OpBuilder &bb, Value s, Value acc) {
      stickIV = s;
      OpBuilder saved = b;
      b = bb;
      SmallVector<Value> slices;
      for (unsigned i = 0; i < plans.size(); ++i) {
        auto elemTy =
            cast<RankedTensorType>(plans[i].value.getType()).getElementType();
        Value slicePhys = extractOpSlice(plans[i], sliceTys[i]);
        slices.push_back(transposeFlags[i]
                             ? emitTranspose2D(slicePhys, elemTy)
                             : slicePhys);
      }
      Value r = spec.emitOp(b, loc, slices, acc, accTy);
      b = saved;
      return r;
    });

    op.getResult(0).replaceAllUsesWith(result);
    op.erase();
    return success();
  }

  // Per-store entry point for the sink stage: read the D coord map from the
  // marker, build the OperandPlan for D (all dims parallel), and call
  // emitSinkStage.
  LogicalResult dispatchSink(mlir::ktdp::StoreOp st,
                              triton::SpyreTensorLayoutOp marker) {
    // The access tile shape IS the physical block shape of D.
    auto tileTy = cast<mlir::ktdp::AccessTileType>(st.getAccessTile().getType());
    ArrayRef<int64_t> physBlock = tileTy.getShape();

    OperandCoords dC{marker.getPhysSrc(), marker.getPhysOp(),
                     marker.getPhysArg(), 2, physBlock};

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

    // Guard: opTile dims must appear in ascending logical-dim order so that
    // the logical extract below matches physical insertion without a transpose.
    {
      int64_t prevLogDim = -1;
      for (int p : dPlan.opTileDims) {
        int64_t logDim = dPlan.dimRoles[p];
        if (logDim < prevLogDim)
          return st.emitError(
              "spyre_tensor_layout: store sink stage: output opTile dims are "
              "not in ascending logical-dim order (transpose not yet supported)");
        prevLogDim = logDim;
      }
    }

    auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

    // Build the initial physical sink accumulator (tensor.empty over physBlock).
    SmallVector<int64_t> sinkShape(physBlock.begin(), physBlock.end());
    Value physicalSink = tensor::EmptyOp::create(b, loc, sinkShape, elemTy);

    // Collect IVs from existing enclosing loops (wired by Phase 1).
    SmallVector<Loop> loops;
    auto storeTile = dyn_cast_or_null<mlir::ktdp::ConstructAccessTilesOp>(
        st.getAccessTile().getDefiningOp());
    collectLoopIVsForTile(st, storeTile, 0, /*loopDims=*/{}, loops);

    // Build floorIVs[physDim] → IV or c0 for trip-1 floor dims.
    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    SmallVector<Value> floorIVs(physRank, c0);
    for (auto &l : loops) {
      if (l.kind == Loop::Parallel) {
        Value iv = l.iv;
        if (!iv.getType().isIndex())
          iv = arith::IndexCastOp::create(b, loc, b.getIndexType(), iv)
                   .getResult();
        floorIVs[l.physDim] = iv;
      }
    }

    // Build extract/insert parameter base arrays from the physical plan.
    //
    // inputTile is logical (rank = dPlan.coords.logicalRank). For each logical
    // dim d, the base offset is 0 and the base size is:
    //   - stickSize            if d is the logical src of a floor dim (will be
    //                          overridden per-iteration inside the stick loop)
    //   - physBlock[p]         for opTile dims (taken whole)
    // We derive sizes by scanning dimRoles rather than reading logShape.
    unsigned logRank = dPlan.coords.logicalRank;
    SmallVector<OpFoldResult> inputOffsetsBase(logRank, idx(0));
    SmallVector<OpFoldResult> inputSizesBase(logRank);
    SmallVector<OpFoldResult> inputStrides(logRank, idx(1));
    // Default: fill from opTileDims (physBlock[p] for each logical src).
    for (int p : dPlan.opTileDims) {
      int64_t logDim = dPlan.dimRoles[p];
      if (logDim >= 0 && (unsigned)logDim < logRank)
        inputSizesBase[logDim] = idx(physBlock[p]);
    }
    // Floor dims: size = stickSize on that logical dim (overridden in loop body).
    for (int p : dPlan.floorDims) {
      int64_t logDim = dPlan.dimRoles[p];
      if (logDim >= 0 && (unsigned)logDim < logRank)
        inputSizesBase[logDim] = idx(stickSize);
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

      int64_t tripCount = physBlock[p]; // sticks per tile along dim p
      Value stickSizeVal = arith::ConstantIndexOp::create(b, loc, stickSize);

      // Extract-result type: stickSize on the floor's logical dim, physBlock
      // for every opTile logical dim.
      SmallVector<int64_t> slShape(logRank);
      for (int p2 : dPlan.opTileDims) {
        int64_t ld = dPlan.dimRoles[p2];
        if (ld >= 0 && (unsigned)ld < logRank) slShape[ld] = physBlock[p2];
      }
      slShape[logDim] = stickSize;
      auto slTy = RankedTensorType::get(slShape, elemTy);

      acc = emitStickLoop(b, loc, tripCount, acc,
          [&](OpBuilder &bb, Value s, Value sinkAccumulator) -> Value {
            // Input-extract: s-th stick-slice of inputTile on the floor's logDim.
            SmallVector<OpFoldResult> inOff = inputOffsetsBase;
            inOff[logDim] =
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

    // --- 2. Rebuild each ConstructAccessTilesOp that uses the old memView ---
    //   The access tile's base is the (logical) memView. We find access tiles
    //   that use the old memView, replace their base with the new physical view,
    //   update block shape + indices.
    SmallVector<mlir::ktdp::ConstructAccessTilesOp> tiles;
    for (auto *user : memView.getUsers())
      if (auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(user))
        tiles.push_back(tile);

    for (auto tileOp : tiles) {
      if (failed(rewriteAccessTile(tileOp, newMemView, coords)))
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

  // Remap logical indices -> physical indices via the coordinate map.
  SmallVector<Value> mapIndices(OpBuilder &builder, Location loc,
                                ValueRange logIdx, ArrayRef<int64_t> physSrc,
                                ArrayRef<int64_t> physOp,
                                ArrayRef<int64_t> physArg) {
    SmallVector<Value> physIdx;
    for (unsigned k = 0; k < physSrc.size(); ++k)
      physIdx.push_back(applyIndex(builder, loc, logIdx[physSrc[k]],
                                   static_cast<CoordOp>(physOp[k]), physArg[k]));
    return physIdx;
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
          rescaleEnclosingLoop(forOp, physBlock[k]);
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

  // True if the op is a contraction (matmul, reduce, etc.) whose result shape
  // is determined by its own semantics rather than being inherited from a single
  // physical input. retypeChain stops here and does not propagate through.
  // NOTE: uses a tensor-operand-count heuristic (>1 tensor operand). This
  // misclassifies elementwise ops with two tensor inputs (e.g. add) as
  // contractions; replacing with an explicit op-type check is tracked as P1-2.
  static bool isContractionOp(Operation *op) {
    int tensorOperandCount = 0;
    for (auto operand : op->getOperands())
      if (isa<RankedTensorType>(operand.getType()))
        ++tensorOperandCount;
    return tensorOperandCount > 1;
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
