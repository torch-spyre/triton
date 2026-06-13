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
//           applyStatic             derive physical static extents
//         rewriteAccessTile(tile)   -> physical ConstructAccessTilesOp
//           mapIndices / applyIndex
//         retypeLoad                -> physical ktdp.load + retypeChain
//           retypeChain             propagate type (stops at multi-tensor ops)
//         retypeStore               redirect access tile operand (output stays
//                                   logical-typed; sink stage fixes it in Phase 2)
//         (marker kept alive for Phase 2)
//     Phase 2 — synthesizeContractions (loop-until-stable):
//       source: dispatchMatmul(mm)
//         findMarkerForOperand      trace load -> access tile -> memView -> marker
//         classify                  build OperandPlan (dimRoles, floorDims, reduceLoop…)
//         emitMatmulStage           walk existing loops (collectExistingLoopIVs),
//                                   extract slices + linalg.matmul; inner stick
//                                   loop (emitStickLoop) for StickifiedBlock
//       sink:   dispatchStore(st)   (only when output descriptor is annotated)
//         findMarkerForStore        trace store -> access tile -> memView -> marker
//         classify / emitStoreStage walk existing loops, scatter logical C into
//                                   physical D via tensor.insert_slice
//     Phase 3 — eraseMarker (marker + dead bridge cast)
//
//   Coord helpers (free functions):
//     applyStatic  : compile-time extent (shape / block_shape)
//     applyIndex   : SSA load/store offset (identity | divsi | remsi)
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

  // Walk all linalg.matmul ops; fix any whose operands are rank-3 (physical).
  // Phase 2 driver: walk contraction ops whose operands have been physicalized
  // to rank > 2 and synthesize the scf.for loop nest that produces a 2D result.
  // Also walks ktdp.store ops whose output descriptor is annotated (data_tile
  // is logical but access_tile is already physical from Phase 1) and calls
  // dispatchStore to synthesize the scatter nest. Repeats until stable.
  // Currently handles linalg.matmul + ktdp.store; extend by adding a dispatch
  // branch for other contraction / sink ops as needed.
  LogicalResult synthesizeContractions(ModuleOp module) {
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
        if (failed(dispatchStore(st, marker)))
          return failure();
        changed = true;
      }
    }
    return success();
  }

  // Walk back from a matmul operand through the elementwise chain to the
  // ktdp.load, then look up the physical memView -> marker map populated
  // during Phase 1.
  triton::SpyreTensorLayoutOp findMarkerForOperand(Value operand) {
    Value v = operand;
    while (true) {
      auto *defOp = v.getDefiningOp();
      if (!defOp)
        return {};
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(defOp)) {
        auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
            ld.getAccessTile().getDefiningOp());
        if (!tile)
          return {};
        auto it = physMemViewToMarker.find(tile.getBase());
        return it != physMemViewToMarker.end() ? it->second
                                               : triton::SpyreTensorLayoutOp{};
      }
      // Step back through single-tensor-operand elementwise ops.
      if (defOp->getNumResults() != 1 || defOp->getNumOperands() == 0)
        return {};
      int tensorOps = 0;
      for (auto op : defOp->getOperands())
        if (isa<RankedTensorType>(op.getType()))
          ++tensorOps;
      if (tensorOps != 1)
        return {};
      for (auto op : defOp->getOperands())
        if (isa<RankedTensorType>(op.getType())) { v = op; break; }
    }
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
  //   -1    : reduction dim — K-stick (loop), K-flat, or K-lane (inner dot)
  //
  // All dims whose logical source is the K dimension get role -1 regardless of
  // whether they are FloorDiv (K-stick) or not. classify() then splits them
  // right-to-left: the rightmost -1 dim is reduceInner (consumed by the Op);
  // the rest are reduceLoop (each becomes an scf.for reduction loop).
  static void buildDimRoles(const OperandCoords &coords, int64_t kLogicalSrc,
                             int64_t parallelRole,
                             SmallVectorImpl<int64_t> &roles) {
    int n = (int)coords.src.size();
    roles.resize(n);
    for (int p = 0; p < n; ++p) {
      if (coords.src[p] != kLogicalSrc)
        roles[p] = parallelRole;
      else
        roles[p] = -1; // all K dims: reduction (inner or loop, split by classify)
    }
  }

  // ---- Shared utilities (staged model) ----

  // How a single physical dim is sliced when extracting the per-iteration tile.
  // Decided per-dim in classify() from local geometry; consumed by extractOpSlice
  // (and the slice-type computation) as a pure lookup — no loop-set inspection
  // or cross-operand reasoning at slice time.
  enum class SliceKind {
    StickIndex,       // floor/reduceLoop dim: offset = this operand's own loop IV,
                      // size = 1 (selects one stick along a stick-index dim).
    StickifiedBlock,  // reduceInner spanning >1 stick (B's K-flat): offset =
                      // reduction IV * stickSize, size = stickSize (one stick, R-b).
    WholeBlock,       // lane / opSlice / single-stick reduceInner: offset = 0,
                      // size = physBlock[p] (taken whole as part of the 2D tile).
  };

  // One operand classified by its Map(Op, X) roles and physical layout.
  // `classify` fills all derived fields from `coords` + `dimRoles`.
  //
  // Field semantics (right-to-left traversal of the physical dims, R-c):
  //   lane        = innermost phys dim (rank-1); always the stick lane — full slice
  //   floorDims   = parallel dims that are FloorDiv (stick-index dims) → loops
  //   reduceDims  = all -1 dims, right-to-left order
  //   reduceInner = rightmost reduceDim → consumed by Op (inner dot)
  //   reduceLoop  = remaining reduceDims (all but rightmost) → reduction loops
  //   opSliceDims = residual >= 0 dims that are not floorDims (fed to Op as-is)
  struct OperandPlan {
    Value               value;      // SSA tensor (physical on memory side)
    OperandCoords       coords;     // coord map + shape (kept for op/arg lookups)
    SmallVector<int64_t> dimRoles;  // per-phys-dim role (>= 0 | -1)

    int                lane;        // innermost phys dim = rank-1
    int64_t            stickSize;   // stick/lane width = physBlock[lane] (e.g. 64).
                                    // The slice extent for any stick-width dim
                                    // (the lane, and a sliced reduceInner). Distinct
                                    // from physBlock[dim], which is the full dim
                                    // extent (= n_sticks * stickSize when BLOCK > stick).
    SmallVector<int>   floorDims;   // parallel stick-index dims → loops (role>=0, FloorDiv)
    SmallVector<int>   reduceDims;  // all -1 dims in right-to-left order
    int                reduceInner; // rightmost reduceDim → consumed by Op; -1 if none
    SmallVector<int>   reduceLoop;  // reduceDims minus reduceInner → reduction loops
    SmallVector<int>   opSliceDims; // residual >= 0 non-floor dims (the 2D slice for matmul)
    SmallVector<SliceKind> sliceKind; // per-phys-dim slice behavior (see SliceKind)

    // Threaded from the physical construct_memory_view (R-a). Populated by
    // dispatchMatmul / dispatchStore after classify(); consumed by S5.
    SmallVector<OpFoldResult> strides; // physical stride per dim (attr=static, Value=dynamic)
    SmallVector<OpFoldResult> sizes;   // physical extent per dim (attr=static, Value=dynamic)
  };

  // Same walk as findMarkerForOperand, but returns the physical
  // ConstructMemoryViewOp rather than the marker. Used by dispatchMatmul to
  // thread strides/sizes into the OperandPlan (S4).
  mlir::ktdp::ConstructMemoryViewOp getMemViewForOperand(Value operand) {
    Value v = operand;
    while (true) {
      auto *defOp = v.getDefiningOp();
      if (!defOp)
        return {};
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(defOp)) {
        auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
            ld.getAccessTile().getDefiningOp());
        if (!tile)
          return {};
        return tile.getBase().getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
      }
      if (defOp->getNumResults() != 1 || defOp->getNumOperands() == 0)
        return {};
      int tensorOps = 0;
      for (auto op : defOp->getOperands())
        if (isa<RankedTensorType>(op.getType()))
          ++tensorOps;
      if (tensorOps != 1)
        return {};
      for (auto op : defOp->getOperands())
        if (isa<RankedTensorType>(op.getType())) { v = op; break; }
    }
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
    int   owner;    // which operand: 0=A, 1=B
    int   physDim;  // phys dim the IV indexes
    Value iv;       // induction variable
  };

  // Walk enclosing scf.for ops of `op` and populate `loops` with one entry per
  // floor/reduceLoop dim in aPlan/bPlan whose IV was wired by Phase 1.
  // Phase 1 stored the (index-cast) loop IV directly as the coord operand of
  // the construct_access_tile for each FloorDiv dim; we recover it by tracing
  // each coord operand back through index_cast to its BlockArgument.
  static void collectExistingLoopIVs(Operation *op,
                                     const OperandPlan &aPlan,
                                     const OperandPlan &bPlan,
                                     SmallVectorImpl<Loop> &loops) {
    // Helper: get the construct_access_tile backing a plan's load chain.
    auto getTile = [](const OperandPlan &plan)
        -> mlir::ktdp::ConstructAccessTilesOp {
      Value v = plan.value;
      if (!v) return {};
      while (true) {
        auto *defOp = v.getDefiningOp();
        if (!defOp) return {};
        if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(defOp))
          return dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(
              ld.getAccessTile().getDefiningOp());
        if (defOp->getNumResults() != 1 || defOp->getNumOperands() == 0)
          return {};
        int cnt = 0;
        for (auto o : defOp->getOperands())
          if (isa<RankedTensorType>(o.getType())) ++cnt;
        if (cnt != 1) return {};
        for (auto o : defOp->getOperands())
          if (isa<RankedTensorType>(o.getType())) { v = o; break; }
      }
    };

    // Helper: trace an index Value through index_cast to a BlockArgument; check
    // it is the induction var of an enclosing scf.for (arg #0 of body block).
    auto traceToForIV = [](Value v) -> Value {
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
    };

    auto aTile = getTile(aPlan);
    auto bTile = getTile(bPlan);

    // For the store stage, also scan the store's own access tile if present.
    mlir::ktdp::ConstructAccessTilesOp storeTile;
    if (auto st = dyn_cast<mlir::ktdp::StoreOp>(op))
      storeTile = dyn_cast_or_null<mlir::ktdp::ConstructAccessTilesOp>(
          st.getAccessTile().getDefiningOp());

    // For each tile, build a map physDim -> forIV from its coord operands.
    DenseMap<Value, std::pair<int /*owner*/, int /*physDim*/>> ivToOwnerDim;
    auto scanTile = [&](mlir::ktdp::ConstructAccessTilesOp tile, int owner) {
      if (!tile) return;
      auto indices = tile.getIndices();
      for (unsigned k = 0; k < indices.size(); ++k) {
        Value forIV = traceToForIV(indices[k]);
        if (forIV && !ivToOwnerDim.count(forIV))
          ivToOwnerDim[forIV] = {owner, (int)k};
      }
    };
    scanTile(aTile, 0);
    scanTile(bTile, 1);
    scanTile(storeTile, 0); // store: owner=0, physDim=coord index

    // Walk enclosing scf.for ops and emit Loop entries for matched IVs.
    Operation *cur = op->getParentOp();
    while (cur) {
      if (auto forOp = dyn_cast<scf::ForOp>(cur)) {
        Value iv = forOp.getInductionVar();
        auto it = ivToOwnerDim.find(iv);
        if (it != ivToOwnerDim.end()) {
          auto [owner, physDim] = it->second;
          // Determine kind: reduction if physDim is in aPlan.reduceLoop.
          Loop::Kind kind = Loop::Parallel;
          for (int p : aPlan.reduceLoop)
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
  //   - -1 dims → reduceDims (right-to-left); rightmost is reduceInner
  //     (consumed by Op), the rest are reduceLoop (become scf.for loops)
  //   - >= 0 FloorDiv dims → floorDims (parallel stick-index loops)
  //   - residual >= 0 dims → opSliceDims (fed to Op as-is)
  static OperandPlan classify(Value val, const OperandCoords &coords,
                               ArrayRef<int64_t> dimRoles) {
    int rank = (int)dimRoles.size();
    OperandPlan plan;
    plan.value     = val;
    plan.coords    = coords;
    plan.dimRoles  = SmallVector<int64_t>(dimRoles.begin(), dimRoles.end());
    plan.lane      = rank - 1;
    plan.stickSize = coords.physBlock[rank - 1]; // lane extent = stick width
    plan.reduceInner = -1;

    // Walk right-to-left (innermost first) per R-c.
    // -1 dims: track reduceDims and split into reduceInner (rightmost, consumed
    // by Op) and reduceLoop (outer, each becomes an scf.for reduction loop).
    // reduceInner is also added to opSliceDims during the walk so the existing
    // extraction machinery (which builds the 2D Op tile from opSliceDims) keeps
    // working unchanged. S3-S5 will switch extraction to use reduceInner directly.
    for (int p = rank - 1; p >= 0; --p) {
      int64_t role = dimRoles[p];
      bool isFloor = (role >= 0 &&
                      static_cast<CoordOp>(coords.op[p]) == CoordOp::FloorDiv);
      if (role == -1) {
        plan.reduceDims.push_back(p);
        if (plan.reduceInner == -1) {
          plan.reduceInner = p; // first -1 seen right-to-left = rightmost
          plan.opSliceDims.push_back(p); // consumed by Op — part of the 2D tile
        } else {
          plan.reduceLoop.push_back(p); // outer -1 dims → loops
        }
      } else if (isFloor) {
        plan.floorDims.push_back(p);
      } else {
        // role >= 0, non-floor (identity parallel or lane)
        plan.opSliceDims.push_back(p);
      }
    }
    // Reverse so dims appear in ascending physical order (left-to-right),
    // matching the expectations of buildExtract and collectLoops callers.
    std::reverse(plan.floorDims.begin(), plan.floorDims.end());
    std::reverse(plan.reduceLoop.begin(), plan.reduceLoop.end());
    std::reverse(plan.opSliceDims.begin(), plan.opSliceDims.end());

    // Assign the per-dim slice behavior from local geometry (R-c + R-b):
    //   floor / reduceLoop dim       → StickIndex (one stick by its own IV)
    //   reduceInner spanning >1 stick → StickifiedBlock (B's K-flat; one stick per
    //     reduction iter). Detected purely by extent > stickSize, so a single-
    //     stick reduceInner (incl. the case where reduceInner IS the lane) is WholeBlock.
    //   everything else (lane, opSlice) → WholeBlock (taken whole in the 2D tile)
    plan.sliceKind.assign(rank, SliceKind::WholeBlock);
    auto markList = [&](ArrayRef<int> dims) {
      for (int p : dims)
        plan.sliceKind[p] = SliceKind::StickIndex;
    };
    markList(plan.floorDims);
    markList(plan.reduceLoop);
    if (plan.reduceInner != -1 &&
        coords.physBlock[plan.reduceInner] > plan.stickSize)
      plan.sliceKind[plan.reduceInner] = SliceKind::StickifiedBlock;
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
        if (user->getNumResults() != 1)
          continue;
        int tensorOps = 0;
        for (auto op : user->getOperands())
          if (isa<RankedTensorType>(op.getType()))
            ++tensorOps;
        if (tensorOps != 1)
          continue;
        worklist.push_back(user->getResult(0));
      }
    }
    return {};
  }

  // Cross-operand SliceKind fix-up: StickifiedBlock is only valid when a
  // reduction loop exists (aPlan.reduceLoop non-empty). When neither operand
  // has a reduceLoop, K is contracted whole in one tile — demote any
  // StickifiedBlock to WholeBlock so extractOpSlice never calls getReduceIV()
  // on a null IV.
  static void reconcilePlans(OperandPlan &aPlan, OperandPlan &bPlan) {
    if (!aPlan.reduceLoop.empty() || !bPlan.reduceLoop.empty())
      return;
    for (auto *plan : {&aPlan, &bPlan})
      for (auto &sk : plan->sliceKind)
        if (sk == SliceKind::StickifiedBlock)
          sk = SliceKind::WholeBlock;
  }

  // Per-matmul entry point for Phase 2: trace operands to markers, build
  // OperandPlans, and call emitMatmulStage.
  LogicalResult dispatchMatmul(linalg::MatmulOp mm) {
    auto aMarker = findMarkerForOperand(mm.getInputs()[0]);
    auto bMarker = findMarkerForOperand(mm.getInputs()[1]);
    if (!aMarker || !bMarker)
      return mm.emitError(
          "spyre_tensor_layout: cannot find layout markers for matmul operands");

    auto aPhysShape = cast<RankedTensorType>(mm.getInputs()[0].getType()).getShape();
    auto bPhysShape = cast<RankedTensorType>(mm.getInputs()[1].getType()).getShape();

    OperandCoords aC{aMarker.getPhysSrc(), aMarker.getPhysOp(),
                     aMarker.getPhysArg(), 2, aPhysShape};
    OperandCoords bC{bMarker.getPhysSrc(), bMarker.getPhysOp(),
                     bMarker.getPhysArg(), 2, bPhysShape};

    // A logical: dim0=M (src 0 → parallel), dim1=K (src 1 → reduction).
    // B logical: dim0=K (src 0 → reduction), dim1=N (src 1 → parallel).
    SmallVector<int64_t> dimRoleA, dimRoleB;
    buildDimRoles(aC, /*kLogicalSrc=*/1, /*parallelRole=*/0, dimRoleA);
    buildDimRoles(bC, /*kLogicalSrc=*/0, /*parallelRole=*/1, dimRoleB);

    OperandPlan aPlan = classify(mm.getInputs()[0], aC, dimRoleA);
    OperandPlan bPlan = classify(mm.getInputs()[1], bC, dimRoleB);

    reconcilePlans(aPlan, bPlan);

    // S4: thread physical strides/sizes from the construct_memory_view into
    // the plans. Not yet consumed by emitMatmulStage (S5 flips the switch).
    auto aMemView = getMemViewForOperand(mm.getInputs()[0]);
    auto bMemView = getMemViewForOperand(mm.getInputs()[1]);
    if (!aMemView || !bMemView)
      return mm.emitError(
          "spyre_tensor_layout: cannot locate physical memory view for matmul operands");
    MLIRContext *ctx = mm.getContext();
    if (failed(threadStridesAndSizes(aPlan, aMemView, ctx)))
      return failure();
    if (failed(threadStridesAndSizes(bPlan, bMemView, ctx)))
      return failure();

    return emitMatmulStage(mm, aPlan, bPlan);
  }

  // Source stage: extract 2D A/B slices from the already-physicalized loads,
  // emit linalg.matmul(outs=cVal) at mm's position, RAUW mm's result, erase mm.
  // Phase 1 already rescaled and wired the enclosing scf.for loops; Phase 2
  // reads their IVs via collectExistingLoopIVs and emits only the slices +
  // matmul — no new loops.
  LogicalResult emitMatmulStage(linalg::MatmulOp mm,
                                const OperandPlan &aPlan,
                                const OperandPlan &bPlan) {
    OpBuilder b(mm);
    Location loc = mm.getLoc();

    Value cVal = mm.getOutputs()[0];

    auto aElemTy = cast<RankedTensorType>(aPlan.value.getType()).getElementType();
    auto bElemTy = cast<RankedTensorType>(bPlan.value.getType()).getElementType();
    auto accTy   = cast<RankedTensorType>(cVal.getType()).getElementType();

    ArrayRef<int64_t> aPhysShape = aPlan.coords.physBlock;
    ArrayRef<int64_t> bPhysShape = bPlan.coords.physBlock;

    // Each operand must have exactly 2 opSlice dims (the 2D matmul tile).
    if (aPlan.opSliceDims.size() != 2 || bPlan.opSliceDims.size() != 2)
      return mm.emitError(
          "spyre_tensor_layout: expected exactly 2 opSlice dims per operand");

    int aDimLo = aPlan.opSliceDims[0], aDimHi = aPlan.opSliceDims[1];
    int bDimLo = bPlan.opSliceDims[0], bDimHi = bPlan.opSliceDims[1];

    // 2D slice result types (physical order).
    auto opSliceExtent = [](const OperandPlan &plan, int p) -> int64_t {
      return plan.sliceKind[p] == SliceKind::StickifiedBlock
                 ? plan.stickSize
                 : plan.coords.physBlock[p];
    };
    auto sliceAPhysTy = RankedTensorType::get(
        {opSliceExtent(aPlan, aDimLo), opSliceExtent(aPlan, aDimHi)}, aElemTy);
    auto sliceBPhysTy = RankedTensorType::get(
        {opSliceExtent(bPlan, bDimLo), opSliceExtent(bPlan, bDimHi)}, bElemTy);

    bool transposeA = (aPlan.dimRoles[aDimLo] == -1);
    bool transposeB = (bPlan.dimRoles[bDimLo] != -1);

    int64_t M  = transposeA ? aPhysShape[aDimHi] : aPhysShape[aDimLo];
    int64_t KA = transposeA ? aPhysShape[aDimLo] : aPhysShape[aDimHi];
    int64_t N  = transposeB ? bPhysShape[bDimLo] : bPhysShape[bDimHi];
    auto accTy2D = RankedTensorType::get({M, N}, accTy);

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
    SmallVector<Loop> loops;
    collectExistingLoopIVs(mm, aPlan, bPlan, loops);

    // Determine if a StickifiedBlock dim exists and what factor it has.
    // factor > 1 means each loaded tile spans factor sticks; Phase 2 emits
    // an inner stick loop 0..factor to iterate within the tile.
    int64_t stickFactor = 1;
    for (auto *plan : {&aPlan, &bPlan}) {
      for (int p = 0; p < (int)plan->sliceKind.size(); ++p) {
        if (plan->sliceKind[p] == SliceKind::StickifiedBlock) {
          int64_t f = plan->coords.physBlock[p] / plan->stickSize;
          if (f > stickFactor) stickFactor = f;
        }
      }
    }

    // stickIV: the within-tile stick index used by StickIndex (when
    // physBlock[p] > 1) and StickifiedBlock. For factor==1 it stays null
    // (set below before extractOpSlice is called).
    Value stickIV;

    // Extract a 2D stick slice from `plan`. Offsets/sizes follow sliceKind:
    //   StickIndex      → offset = stickIV (within-tile), size = 1
    //                     (only when physBlock[p] > 1; else offset = 0)
    //   StickifiedBlock → offset = stickIV * stickSize, size = stickSize
    //   WholeBlock      → offset = 0, size = physBlock[p]
    auto extractOpSlice = [&](const OperandPlan &plan, int planOwner,
                               ArrayRef<int64_t> physBlock,
                               RankedTensorType resultTy) -> Value {
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

    // emitStickLoop: generic helper — emits scf.for s=0..factor iter_args(acc)
    // and calls body(bodyBuilder, stickIV, acc) to get the next accumulator.
    // For factor==1, calls body with the outer builder and a constant 0 IV.
    // After the call, `b` is positioned after the for op (or unchanged for 1).
    auto emitStickLoop = [&](int64_t factor, Value acc,
                             function_ref<Value(OpBuilder &, Value, Value)> body)
        -> Value {
      if (factor <= 1) {
        Value s0 = arith::ConstantIndexOp::create(b, loc, 0);
        return body(b, s0, acc);
      }
      Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
      Value ub = arith::ConstantIndexOp::create(b, loc, factor);
      auto forOp = scf::ForOp::create(b, loc, c0, ub, c1, ValueRange{acc});
      OpBuilder ib = OpBuilder::atBlockBegin(forOp.getBody());
      Value stepped = body(ib, forOp.getInductionVar(),
                           forOp.getRegionIterArgs()[0]);
      scf::YieldOp::create(ib, loc, ValueRange{stepped});
      b.setInsertionPointAfter(forOp);
      return forOp.getResult(0);
    };

    Value result = emitStickLoop(stickFactor, cVal,
        [&](OpBuilder &bb, Value s, Value acc) {
      stickIV = s;
      // Temporarily redirect b so extractOpSlice / emitTranspose2D use bb.
      OpBuilder saved = b;
      b = bb;
      Value aSlicePhys = extractOpSlice(aPlan, 0, aPhysShape, sliceAPhysTy);
      Value bSlicePhys = extractOpSlice(bPlan, 1, bPhysShape, sliceBPhysTy);
      Value aSlice = transposeA ? emitTranspose2D(aSlicePhys, aElemTy) : aSlicePhys;
      Value bSlice = transposeB ? emitTranspose2D(bSlicePhys, bElemTy) : bSlicePhys;
      Value r = linalg::MatmulOp::create(b, loc, accTy2D,
          ValueRange{aSlice, bSlice}, ValueRange{acc}).getResult(0);
      b = saved;
      return r;
    });

    mm.getResult(0).replaceAllUsesWith(result);
    mm.erase();
    return success();
  }

  // Per-store entry point for the sink stage: read the D coord map from the
  // marker, build the OperandPlan for D (all dims parallel), and call
  // emitStoreStage.
  LogicalResult dispatchStore(mlir::ktdp::StoreOp st,
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
    return emitStoreStage(st, dPlan);
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
  // last use. Since logicalC lives at the function body level (produced by the
  // matmul stage or by retypeChain from a load), the builder is set there by
  // OpBuilder(st) which positions before the store.
  LogicalResult emitStoreStage(mlir::ktdp::StoreOp st,
                               const OperandPlan &dPlan) {
    Value logicalC = st.getDataTile(); // the LOGICAL input (rank-2)
    OpBuilder b(st);
    Location loc = st.getLoc();

    auto logTy = cast<RankedTensorType>(logicalC.getType());
    Type elemTy = logTy.getElementType();
    ArrayRef<int64_t> logShape = logTy.getShape(); // [M, N_total]

    ArrayRef<int64_t> physBlock = dPlan.coords.physBlock; // [Nstick, M, lane]
    int physRank = (int)physBlock.size();

    // Verify assumptions: exactly one floor dim, rest are opSlice.
    // The lane dim (innermost) and any identity parallel dims are opSliceDims.
    if (dPlan.floorDims.empty())
      return st.emitError(
          "spyre_tensor_layout: store sink stage requires at least one "
          "parallel floor dim in the output layout");
    // reduceLoop must be empty for a store.
    if (!dPlan.reduceLoop.empty())
      return st.emitError(
          "spyre_tensor_layout: store sink stage: unexpected reduction dim");

    // The logical C slice is extracted in logical-dim order and inserted into
    // D's opSlice dims directly (no transpose). This is only correct when D's
    // opSlice dims are already in ascending logical-dim order. Guard against
    // layouts that would need a transpose (the source stage handles that case
    // via transposeA/transposeB; the sink stage does not yet).
    {
      int64_t prevLogDim = -1;
      for (int p : dPlan.opSliceDims) {
        int64_t logDim = dPlan.dimRoles[p];
        if (logDim < prevLogDim)
          return st.emitError(
              "spyre_tensor_layout: store sink stage: output opSlice dims are "
              "not in ascending logical-dim order (transpose not yet supported)");
        prevLogDim = logDim;
      }
    }

    // Logical C shape and the N-lane size from the physical plan.
    // physBlock[lane] is the stick-lane extent (e.g. 64); it equals the
    // block-N of the logical C tile for a single-N-stick case. For multi-
    // stick, logShape[1] = physBlock[0] * physBlock[lane].
    int laneDim = dPlan.lane; // innermost phys dim of D (= physRank-1)
    int64_t laneSize = physBlock[laneDim]; // e.g. 64

    // helpers
    auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

    // Build the initial physical D accumulator (tensor.empty over physBlock).
    SmallVector<int64_t> dPhysShape(physBlock.begin(), physBlock.end());
    Value dEmpty = tensor::EmptyOp::create(b, loc, dPhysShape, elemTy);

    // Collect IVs from existing enclosing loops (wired by Phase 1).
    // For floor dims with physBlock[p]==1 the IV is always 0 (trip-1, no loop).
    OperandPlan dummy; // bPlan unused for store; pass dummy
    SmallVector<Loop> loops;
    collectExistingLoopIVs(st, dPlan, dummy, loops);

    // Build floorIVs[physDim] → IV or c0 for trip-1 dims.
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

    // --- Extract sub-slice from logical C ---
    int logRank = (int)logShape.size();
    SmallVector<OpFoldResult> cOffsets(logRank), cSizes(logRank),
        cStrides(logRank, idx(1));
    SmallVector<int64_t> cSliceShape(logShape.begin(), logShape.end());
    for (int d = 0; d < logRank; ++d) {
      cOffsets[d] = idx(0);
      cSizes[d]   = idx(logShape[d]);
    }
    for (int p : dPlan.floorDims) {
      int64_t logDim = dPlan.dimRoles[p];
      if (logDim < 0 || logDim >= logRank) continue;
      Value iv = floorIVs[p];
      Value laneSizeVal = arith::ConstantIndexOp::create(b, loc, laneSize);
      Value offset = arith::MulIOp::create(b, loc, iv, laneSizeVal).getResult();
      cOffsets[logDim] = offset;
      cSizes[logDim]   = idx(laneSize);
      cSliceShape[logDim] = laneSize;
    }
    auto cSliceTy = RankedTensorType::get(cSliceShape, elemTy);
    Value cSlice = tensor::ExtractSliceOp::create(
        b, loc, cSliceTy, logicalC, cOffsets, cSizes, cStrides);

    // --- Insert cSlice into the physical D accumulator ---
    SmallVector<OpFoldResult> dOffsets(physRank), dSizes(physRank),
        dStrides(physRank, idx(1));
    for (int p = 0; p < physRank; ++p) {
      bool isFloor = llvm::is_contained(dPlan.floorDims, p);
      if (isFloor) {
        Value iv = floorIVs[p];
        dOffsets[p] = iv.getType().isIndex()
                          ? OpFoldResult(iv)
                          : OpFoldResult(arith::IndexCastOp::create(
                                b, loc, b.getIndexType(), iv).getResult());
        dSizes[p] = idx(1);
      } else {
        dOffsets[p] = idx(0);
        dSizes[p]   = idx(physBlock[p]);
      }
    }
    Value result = tensor::InsertSliceOp::create(
        b, loc, cSlice, dEmpty, dOffsets, dSizes, dStrides);

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
  //      (stops at multi-tensor ops like linalg.matmul)
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

    ArrayRef<int64_t> physSrc = marker.getPhysSrc();
    ArrayRef<int64_t> physOp  = marker.getPhysOp();
    ArrayRef<int64_t> physArg = marker.getPhysArg();
    unsigned physRank = physSrc.size();

    // --- 1. Build physical construct_memory_view ---
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
      if (failed(rewriteAccessTile(tileOp, newMemView,
                                   physSrc, physOp, physArg)))
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

  // Rebuild ConstructAccessTilesOp with the physical memView + block shape.
  // Erases the old tile; users (ktdp.load/store) are retargeted to the new tile.
  LogicalResult rewriteAccessTile(mlir::ktdp::ConstructAccessTilesOp tileOp,
                                  Value newMemView,
                                  ArrayRef<int64_t> physSrc,
                                  ArrayRef<int64_t> physOp,
                                  ArrayRef<int64_t> physArg) {
    OpBuilder b(tileOp);
    Location loc = tileOp.getLoc();
    MLIRContext *ctx = b.getContext();

    // The logical block shape is the existing tile's AccessTileType shape.
    auto logTileType = tileOp.getResult().getType();
    ArrayRef<int64_t> logBlock = logTileType.getShape();
    unsigned logRank = logBlock.size();
    unsigned physRank = physSrc.size();
    (void)logRank; // physRank may differ (stick dim expands 2->3)

    // Compute physical block shape.
    SmallVector<int64_t> physBlock(physRank);
    for (unsigned k = 0; k < physRank; ++k) {
      int64_t src = physSrc[k];
      auto op = static_cast<CoordOp>(physOp[k]);
      int64_t arg = physArg[k];
      if (src < 0 || src >= (int64_t)logRank)
        return tileOp.emitError("spyre_tensor_layout: phys_src out of range");
      auto pb = applyStatic(logBlock[src], op, arg);
      if (!pb)
        return tileOp.emitError(
            "spyre_tensor_layout: cannot derive static block_shape");
      physBlock[k] = *pb;
    }

    // Map the logical index operands to physical index operands.
    // The existing tile's index operands are already index-typed (cast from i32
    // by buildDirectAccessTile in LowerDescriptorMemory).
    SmallVector<Value> logIdx(tileOp.getIndices().begin(),
                               tileOp.getIndices().end());
    // For FloorDiv (stick-index) dims: trace the logical index back to its
    // enclosing scf.for IV. If found, rescale the loop's trip by
    // factor = BLOCK_d // stickSize (= arg for FloorDiv / arg for the Mod lane
    // dim sharing the same logical src) and wire the IV directly. This discards
    // the muli(%iv, BLOCK_d) + divsi chain that Phase 1 previously emitted.
    // For Mod (lane) dims: always emit remsi (lane is never iterated).
    // For Identity: pass through.
    //
    // stickSize for each logical src dim = physArg[lane_k] where lane_k is the
    // Mod dim sharing that src. We collect it per logical src below.
    SmallVector<int64_t> stickSizeForLogSrc(logBlock.size(), 0);
    for (unsigned k = 0; k < physRank; ++k)
      if (static_cast<CoordOp>(physOp[k]) == CoordOp::Mod)
        stickSizeForLogSrc[physSrc[k]] = physArg[k];

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
        // S6 Phase 1: attempt to trace logI back to an enclosing scf.for IV
        // and rescale that loop to stick granularity.
        auto [iv, logStride] = traceToMLIRBlockArgWithStride(logI);
        scf::ForOp forOp = iv ? dyn_cast_or_null<scf::ForOp>(
                                    iv.getOwner()->getParentOp())
                              : nullptr;
        if (forOp && forOp.getInductionVar() == iv) {
          // Rescale: new_trip = old_trip * factor, step = 1.
          // factor = logStride / stickSize: how many sticks the loop IV advances
          // per iteration.
          int64_t stickSize = stickSizeForLogSrc[src];
          if (stickSize > 0 && logStride < stickSize)
            return tileOp.emitError(
                "spyre_tensor_layout: block size (")
                << logStride << ") is smaller than stick size (" << stickSize
                << ") on logical dim " << src
                << " — BLOCK must be >= stickSize for stickified dims";
          int64_t factor    = (stickSize > 0) ? (logStride / stickSize) : 1;
          Type ivTy = iv.getType();
          // Rescale: ub *= factor, step = factor. IV values are 0, factor,
          // 2*factor, ... — tile-granular, valid as access-tile coordinates.
          // Phase 2 emits an inner loop 0..factor to iterate sticks within
          // each tile (StickifiedBlock on B's K-flat, StickIndex on A).
          if (factor > 1) {
            OpBuilder bBefore(forOp);
            Location forLoc = forOp.getLoc();
            Value oldUb = forOp.getUpperBound();
            Value factorV = arith::ConstantOp::create(
                bBefore, forLoc, bBefore.getIntegerAttr(ivTy, factor));
            Value newUb = arith::MulIOp::create(bBefore, forLoc, oldUb, factorV)
                              .getResult();
            forOp.setUpperBound(newUb);
            forOp.setStep(factorV);
          } else {
            // factor == 1: ensure step is 1.
            OpBuilder bBefore(forOp);
            Value c1v = arith::ConstantOp::create(
                bBefore, forOp.getLoc(), bBefore.getIntegerAttr(ivTy, 1));
            forOp.setStep(c1v);
          }
          // Wire the IV directly as the stick-index coord (tile-granular).
          // construct_access_tiles requires index-typed operands; cast if needed.
          Value ivIdx = ivTy.isIndex()
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
        retypeStore(st, newTile.getResult());
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
  // The src tensor is already physical-typed via retypeChain from retypeLoad.
  void retypeStore(mlir::ktdp::StoreOp st, Value newTile) {
    st.getAccessTileMutable().set(newTile);
  }

  // Forward-retype the elementwise op chain: replace oldVal with newVal
  // everywhere and update result types of single-result ops that still carry
  // the old (logical-rank) type.
  // Stops at multi-tensor-operand ops (contractions: linalg.matmul, reduce,
  // etc.) whose result shape is determined by their own semantics, not by the
  // physical shape of one input.
  void retypeChain(Value oldVal, Value newVal) {
    oldVal.replaceAllUsesWith(newVal);
    SmallVector<Operation *> worklist(newVal.getUsers().begin(),
                                      newVal.getUsers().end());
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (op->getNumResults() != 1)
        continue;
      // Stop at contraction ops: more than one tensor-typed operand means the
      // result shape is not simply inherited from one input.
      int tensorOperandCount = 0;
      for (auto operand : op->getOperands())
        if (isa<RankedTensorType>(operand.getType()))
          ++tensorOperandCount;
      if (tensorOperandCount > 1)
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
