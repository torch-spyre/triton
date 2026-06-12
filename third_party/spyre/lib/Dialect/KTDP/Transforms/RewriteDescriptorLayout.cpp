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
// v1 scope (pointwise): pure stick-tiling. Does NOT synthesize scf.for loops
// for matmul contraction -- deferred to Increment 2/3.
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
//         retypeStore               redirect access tile operand
//         (marker kept alive for Phase 2)
//     Phase 2 — fixupMatmulOps (loop-until-stable):
//       fixupMatmul(mm)
//         findMarkerForOperand      trace load -> access tile -> memView -> marker
//         analyzeMatmulCoords       read phys_src/op/arg -> MatmulDims
//         synthesizeMatmulLoop      emit scf.for nest + linalg.matmul slices
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
    if (failed(fixupMatmulOps(module)))
      return signalPassFailure();

    // Phase 3: erase all markers (and their now-dead bridge casts).
    for (auto marker : markers)
      eraseMarker(marker);
  }

  // Walk all linalg.matmul ops; fix any whose operands are rank-3 (physical).
  // For each, traces operands back to their tt.spyre_tensor_layout markers
  // (still live during Phase 2) to read the coord maps. Repeats until stable.
  LogicalResult fixupMatmulOps(ModuleOp module) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<linalg::MatmulOp> matmuls;
      module.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
      for (auto mm : matmuls) {
        auto aType = dyn_cast<RankedTensorType>(mm.getInputs()[0].getType());
        auto bType = dyn_cast<RankedTensorType>(mm.getInputs()[1].getType());
        if (!aType || !bType)
          continue;
        if (aType.getRank() == 2 && bType.getRank() == 2)
          continue;
        if (failed(fixupMatmul(mm)))
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
    ArrayRef<int64_t> physShape;
  };

  // Assign a role to each physical dim of an operand:
  //   >= 0  : parallel dim, maps to C dim [value]
  //   -1    : reduction (inner dot) dim — K-flat or K-lane
  //   -2    : reduction loop dim — K-stick (not yet supported, Inc 3)
  //
  // For A (logical dim0=M → C.dim0, dim1=K → reduction):
  //   phys_src[p]==0 → M → role 0
  //   phys_src[p]==1 AND op==FloorDiv → K-stick → role -2
  //   phys_src[p]==1 otherwise → K-flat or K-lane → role -1
  //
  // For B (logical dim0=K → reduction, dim1=N → C.dim1):
  //   phys_src[p]==1 → N → role 1
  //   phys_src[p]==0 AND op==FloorDiv → K-stick → role -2
  //   phys_src[p]==0 otherwise → K-flat or K-lane → role -1
  static void buildDimRoles(const OperandCoords &coords, int64_t kLogicalSrc,
                             int64_t parallelRole,
                             SmallVectorImpl<int64_t> &roles) {
    int n = (int)coords.src.size();
    roles.resize(n);
    for (int p = 0; p < n; ++p) {
      if (coords.src[p] != kLogicalSrc) {
        roles[p] = parallelRole;
      } else if (static_cast<CoordOp>(coords.op[p]) == CoordOp::FloorDiv) {
        roles[p] = -2; // K-stick: reduction loop (Inc 3)
      } else {
        roles[p] = -1; // K-flat or K-lane: inner dot
      }
    }
  }

  // Strategy dispatcher: trace operands to markers, build dim roles, synthesize.
  LogicalResult fixupMatmul(linalg::MatmulOp mm) {
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

    // Inc 3: K-stick loop (role -2) not yet supported.
    bool hasKStickLoopA = llvm::any_of(dimRoleA, [](int64_t r) { return r == -2; });
    bool hasKStickLoopB = llvm::any_of(dimRoleB, [](int64_t r) { return r == -2; });
    if (hasKStickLoopA || hasKStickLoopB)
      return mm.emitError(
          "spyre_tensor_layout: K-stick reduction loop not yet supported (Inc 3)");

    return synthesizeCase1(mm, dimRoleA, dimRoleB, aPhysShape, bPhysShape);
  }

  // Case 1: sticks are on parallel dims only (no K-stick reduction loop).
  // Emits one scf.for per parallel stick dim, with a single inner linalg.matmul.
  //
  // Slice extraction produces tensors in physical dim order; a linalg.transpose
  // is emitted when the reduction dim precedes the parallel dim in physical order
  // (since linalg.matmul requires [M,K] and [K,N]).
  LogicalResult synthesizeCase1(linalg::MatmulOp mm,
                                ArrayRef<int64_t> dimRoleA,
                                ArrayRef<int64_t> dimRoleB,
                                ArrayRef<int64_t> aPhysShape,
                                ArrayRef<int64_t> bPhysShape) {
    OpBuilder b(mm);
    Location loc = mm.getLoc();

    Value aVal = mm.getInputs()[0];
    Value bVal = mm.getInputs()[1];
    Value cVal = mm.getOutputs()[0];

    auto aElemTy = cast<RankedTensorType>(aVal.getType()).getElementType();
    auto bElemTy = cast<RankedTensorType>(bVal.getType()).getElementType();
    auto accTy   = cast<RankedTensorType>(cVal.getType()).getElementType();

    int aRank = (int)dimRoleA.size();
    int bRank = (int)dimRoleB.size();

    // Identify outer (stick) dims and inner (lane/flat) dims for each operand.
    // Outer = parallel (role>=0) AND FloorDiv. Inner = everything else.
    auto isOuterDim = [](const OperandCoords &c, ArrayRef<int64_t> roles, int p) {
      return roles[p] >= 0 &&
             static_cast<CoordOp>(c.op[p]) == CoordOp::FloorDiv;
    };

    // We need the OperandCoords to check phys_op for isOuterDim.
    // Re-create them from the markers (already checked non-null in fixupMatmul).
    auto aMarker = findMarkerForOperand(aVal);
    auto bMarker = findMarkerForOperand(bVal);
    OperandCoords aC{aMarker.getPhysSrc(), aMarker.getPhysOp(),
                     aMarker.getPhysArg(), 2, aPhysShape};
    OperandCoords bC{bMarker.getPhysSrc(), bMarker.getPhysOp(),
                     bMarker.getPhysArg(), 2, bPhysShape};

    // Collect outer dims for A and B in physical order.
    SmallVector<int> aOuterDims, bOuterDims;
    for (int p = 0; p < aRank; ++p)
      if (isOuterDim(aC, dimRoleA, p)) aOuterDims.push_back(p);
    for (int p = 0; p < bRank; ++p)
      if (isOuterDim(bC, dimRoleB, p)) bOuterDims.push_back(p);

    // Inner dims of A and B (non-outer), in ascending physical order.
    SmallVector<int> aInnerDims, bInnerDims;
    for (int p = 0; p < aRank; ++p)
      if (!isOuterDim(aC, dimRoleA, p)) aInnerDims.push_back(p);
    for (int p = 0; p < bRank; ++p)
      if (!isOuterDim(bC, dimRoleB, p)) bInnerDims.push_back(p);

    // Each operand must have exactly 2 inner dims (the 2D slice).
    if (aInnerDims.size() != 2 || bInnerDims.size() != 2)
      return mm.emitError(
          "spyre_tensor_layout: expected exactly 2 inner dims per operand");

    // Inner dims in ascending physical order: [lo, hi].
    int aDimLo = aInnerDims[0], aDimHi = aInnerDims[1];
    int bDimLo = bInnerDims[0], bDimHi = bInnerDims[1];

    // Physical-order 2D slice types (what extract_slice actually produces).
    auto sliceAPhysTy = RankedTensorType::get(
        {aPhysShape[aDimLo], aPhysShape[aDimHi]}, aElemTy);
    auto sliceBPhysTy = RankedTensorType::get(
        {bPhysShape[bDimLo], bPhysShape[bDimHi]}, bElemTy);

    // Determine if A needs transpose: reduction dim (-1) at lo position means
    // physical order is [K, M], but matmul needs [M, K].
    bool transposeA = (dimRoleA[aDimLo] == -1);
    bool transposeB = (dimRoleB[bDimLo] != -1); // B needs [K,N]; if lo is N, transpose

    // After transpose the 2D slice types are [M, K_A] and [K_B, N].
    int64_t M   = transposeA ? aPhysShape[aDimHi] : aPhysShape[aDimLo];
    int64_t KA  = transposeA ? aPhysShape[aDimLo] : aPhysShape[aDimHi];
    int64_t KB  = transposeB ? aPhysShape[aDimHi] : bPhysShape[bDimLo];
    int64_t N   = transposeB ? bPhysShape[bDimLo] : bPhysShape[bDimHi];
    (void)KB; // KB == KA for a valid matmul

    auto sliceATy  = RankedTensorType::get({M, KA}, aElemTy);
    auto sliceBTy  = RankedTensorType::get({KA, N}, bElemTy);
    auto accTy2D   = RankedTensorType::get({M, N}, accTy);

    // Helpers.
    auto idx = [&](int64_t v) -> OpFoldResult { return b.getIndexAttr(v); };

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);

    Value zeroAcc = arith::ConstantOp::create(
        b, loc, DenseElementsAttr::get(accTy2D, b.getFloatAttr(accTy, 0.0)));

    // Emit a 2D linalg.transpose (permutation [1,0]) on src.
    auto emitTranspose2D = [&](Value src, Type elemT) -> Value {
      auto srcTy = cast<RankedTensorType>(src.getType());
      auto outTy = RankedTensorType::get(
          {srcTy.getDimSize(1), srcTy.getDimSize(0)}, elemT);
      Value empty = tensor::EmptyOp::create(b, loc, outTy.getShape(), elemT);
      return linalg::TransposeOp::create(b, loc, src, empty,
          b.getDenseI64ArrayAttr({1, 0})).getResult()[0];
    };

    // Build extract_slice for an operand given outer IVs (one per outer dim).
    // Produces a rank-reduced tensor in physical-dim order.
    auto buildExtract = [&](Value operand, ArrayRef<int> outerDims,
                            ArrayRef<int> innerDims,
                            ArrayRef<int64_t> physShape,
                            ArrayRef<Value> outerIVs,
                            RankedTensorType resultTy) -> Value {
      int rank = (int)physShape.size();
      SmallVector<OpFoldResult> offsets(rank), sizes(rank), strides(rank, idx(1));
      for (int i = 0; i < (int)outerDims.size(); ++i) {
        offsets[outerDims[i]] = outerIVs[i];
        sizes[outerDims[i]]   = idx(1);
      }
      for (int p : innerDims) {
        offsets[p] = idx(0);
        sizes[p]   = idx(physShape[p]);
      }
      return tensor::ExtractSliceOp::create(
          b, loc, resultTy, operand, offsets, sizes, strides);
    };

    // Build the loop nest over all outer (stick) dims.
    // Outer dims of A are M-sticks; outer dims of B are N-sticks.
    // We loop over all of them, with acc as the single iter_arg.
    // For the common single-stick case (Ks=Ns=1) these are trip-1 loops.

    // Collect all outer loops: B's outer dims drive the N-stick loop;
    // A's outer dims drive M-stick loops (if any).
    // We emit B's outer loops outermost, then A's, then the inner matmul.
    // All share the same acc iter_arg accumulating into accTy2D.

    // Recursive lambda to emit nested scf.for loops for outer dims.
    // outerDimsAll: concatenation [bOuterDims..., aOuterDims...]
    // operands: [bVal, aVal] for indexing
    // physShapes: [bPhysShape, aPhysShape]
    // returns the final acc value.
    SmallVector<int> allOuterDimsB(bOuterDims), allOuterDimsA(aOuterDims);

    // Collect outer IVs as we descend into the loop nest.
    SmallVector<Value> bOuterIVs, aOuterIVs;

    // Helper: emit nested loops and return the innermost acc.
    // We use a std::function to allow recursion.
    std::function<Value(int, int, Value)> emitLoops =
        [&](int bIdx, int aIdx, Value acc) -> Value {
      // Still have B outer loops to emit.
      if (bIdx < (int)allOuterDimsB.size()) {
        int dim = allOuterDimsB[bIdx];
        int64_t trip = bPhysShape[dim];
        Value cTrip = arith::ConstantIndexOp::create(b, loc, trip);
        auto loop = scf::ForOp::create(b, loc, c0, cTrip, c1,
                                       ValueRange{acc});
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(loop.getBody());
        bOuterIVs.push_back(loop.getInductionVar());
        Value innerAcc = emitLoops(bIdx + 1, aIdx,
                                   loop.getRegionIterArgs()[0]);
        bOuterIVs.pop_back();
        scf::YieldOp::create(b, loc, ValueRange{innerAcc});
        return loop.getResult(0);
      }
      // Still have A outer loops to emit.
      if (aIdx < (int)allOuterDimsA.size()) {
        int dim = allOuterDimsA[aIdx];
        int64_t trip = aPhysShape[dim];
        Value cTrip = arith::ConstantIndexOp::create(b, loc, trip);
        auto loop = scf::ForOp::create(b, loc, c0, cTrip, c1,
                                       ValueRange{acc});
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(loop.getBody());
        aOuterIVs.push_back(loop.getInductionVar());
        Value innerAcc = emitLoops(bIdx, aIdx + 1,
                                   loop.getRegionIterArgs()[0]);
        aOuterIVs.pop_back();
        scf::YieldOp::create(b, loc, ValueRange{innerAcc});
        return loop.getResult(0);
      }
      // Innermost body: extract slices, transpose if needed, matmul.
      Value aSlicePhys = buildExtract(aVal, aOuterDims, aInnerDims,
                                      aPhysShape, aOuterIVs, sliceAPhysTy);
      Value bSlicePhys = buildExtract(bVal, bOuterDims, bInnerDims,
                                      bPhysShape, bOuterIVs, sliceBPhysTy);
      Value aSlice = transposeA ? emitTranspose2D(aSlicePhys, aElemTy) : aSlicePhys;
      Value bSlice = transposeB ? emitTranspose2D(bSlicePhys, bElemTy) : bSlicePhys;
      // acc += a_slice @ b_slice  ([M,K] x [K,N] -> [M,N])
      return linalg::MatmulOp::create(b, loc, accTy2D,
          ValueRange{aSlice, bSlice}, ValueRange{acc}).getResult(0);
    };

    Value result = emitLoops(0, 0, zeroAcc);
    mm.getResult(0).replaceAllUsesWith(result);
    mm.erase();
    return success();
  }

  // Erase a marker and its now-dead bridge cast. Called in Phase 3 after
  // fixupMatmulOps has finished reading the coord maps.
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

      // Compute row-major strides over physStaticSizes.
      // For fully static shapes this is compile-time; mixed/dynamic falls back.
      physStaticStrides.resize(physRank);
      physDynStrides.clear();
      {
        // Walk right-to-left accumulating product.
        // If we hit a kDynamic dim we need SSA multiplication.
        bool hasAnyDynStride = false;
        for (int k = (int)physRank - 1; k >= 0; --k) {
          if (k == (int)physRank - 1) {
            physStaticStrides[k] = 1;
          } else {
            int64_t prevSz = physStaticSizes[k + 1];
            int64_t prevSt = physStaticStrides[k + 1];
            if (prevSz != ShapedType::kDynamic && prevSt != ShapedType::kDynamic) {
              physStaticStrides[k] = prevSt * prevSz;
            } else {
              physStaticStrides[k] = ShapedType::kDynamic;
              hasAnyDynStride = true;
            }
          }
        }
        if (hasAnyDynStride) {
          // Build SSA stride values for dynamic dims: running product.
          SmallVector<Value> strideVals(physRank);
          Value running = arith::ConstantOp::create(
              b, loc, b.getIndexAttr(1));
          for (int k = (int)physRank - 1; k >= 0; --k) {
            strideVals[k] = running;
            // running *= physSize[k]
            Value dimVal;
            if (physStaticSizes[k] != ShapedType::kDynamic) {
              dimVal = arith::ConstantOp::create(
                  b, loc, b.getIndexAttr(physStaticSizes[k]));
            } else {
              // Find the matching physDynSize (positional).
              int dynPos = 0;
              for (unsigned j = 0; j < (unsigned)k; ++j)
                if (physStaticSizes[j] == ShapedType::kDynamic)
                  ++dynPos;
              dimVal = physDynSizes[dynPos];
            }
            running = arith::MulIOp::create(b, loc, running, dimVal).getResult();
          }
          for (unsigned k = 0; k < physRank; ++k) {
            if (physStaticStrides[k] == ShapedType::kDynamic)
              physDynStrides.push_back(strideVals[k]);
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

    // Marker and bridge cast are NOT erased here — Phase 2 (fixupMatmulOps)
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
    // applyIndex expects i32 inputs; these are index-typed. Rebuild using index
    // arithmetic directly (all the same: identity / divsi / const-0).
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
        Value c = arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
        physIdx.push_back(
            arith::DivSIOp::create(b, loc, logI, c).getResult());
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
