//===- RewriteDescriptorLayout.cpp ----------------------------------------===//
//
// Rewrites logical tensor descriptors to their physical (stick-tiled) layout,
// driven by tt.spyre_tensor_layout markers. Runs after LowerComputeOps and
// before LowerInterTile in the TTIR->KTDP pipeline, so that tt.dot is already
// lowered to linalg.matmul before operands are physicalized.
//
// The physical layout is the OpSpec `device_coordinates` form, carried on the
// marker as three i64 arrays, one entry per physical dim:
//   phys_src[k] : logical dim k derives from
//   phys_op[k]  : 0 = identity, 1 = floordiv, 2 = mod
//   phys_arg[k] : divisor (floordiv) / modulus (mod); ignored for identity
// e.g. [M,N] stick-on-N -> phys_src=[1,0,1] phys_op=[1,0,2] phys_arg=[64,0,64]
//   => device_size [N//64, M, N%64].
//
// Staged model:
//   Phase 1 — physicalize each annotated descriptor (memView + access tiles +
//             loads + stores)
//   Phase 3 — erase all markers (and their now-dead bridge casts)
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "RewriteDescriptorLayout/PermutationUtils.h"
#include "RewriteDescriptorLayout/Types.h"
#include "RewriteDescriptorLayout/ContractionSynthesis.h"
#include "RewriteDescriptorLayout/IndexDomain.h"
#include "RewriteDescriptorLayout/PhysicalTypeAnalysis.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <numeric>
#include <optional>

#define DEBUG_TYPE "rewrite-descriptor-layout"

namespace mlir::triton::ktdp {

#define GEN_PASS_DEF_REWRITEDESCRIPTORLAYOUT
#include "Dialect/KTDP/Transforms/Passes.h.inc"

} // namespace mlir::triton::ktdp

namespace {

using namespace mlir;
using namespace mlir::triton::ktdp;

/// Apply a single coordinate-map operation to an SSA index value.
/// Returns the transformed index (identity / divsi / remsi).
Value applyCoordOp(OpBuilder &b, Location loc, Value logicalIdx,
                   CoordOp op, int64_t arg) {
  switch (op) {
  case CoordOp::Identity:
    return logicalIdx;
  case CoordOp::FloorDiv: {
    Value c = arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
    return arith::DivSIOp::create(b, loc, logicalIdx, c).getResult();
  }
  case CoordOp::Mod: {
    Value c = arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
    return arith::RemSIOp::create(b, loc, logicalIdx, c).getResult();
  }
  }
  llvm_unreachable("invalid CoordOp");
}

/// Apply a coordinate-map operation to an AffineExpr (forward application).
AffineExpr applyCoordOpExpr(AffineExpr expr, CoordOp op, int64_t arg) {
  switch (op) {
  case CoordOp::Identity: return expr;
  case CoordOp::FloorDiv: return expr.floorDiv(arg);
  case CoordOp::Mod:      return expr % arg;
  }
  llvm_unreachable("invalid CoordOp");
}

struct RewriteDescriptorLayoutPass
    : public mlir::triton::ktdp::impl::RewriteDescriptorLayoutBase<
          RewriteDescriptorLayoutPass> {

  using RewriteDescriptorLayoutBase::RewriteDescriptorLayoutBase;

  // Maps each physical ConstructMemoryViewOp result -> its source marker.
  DenseMap<Value, triton::SpyreTensorLayoutOp> physMemViewToMarker;

  // Logical construct_memory_view ops replaced by a physical one in Phase 1.
  // They cannot be erased there: the marker's bridge cast still holds them as
  // an operand, and that cast is only erased in Phase 3 (eraseMarker). So
  // Phase 3 erases them right after, once nothing references them.
  SmallVector<mlir::ktdp::ConstructMemoryViewOp> deadLogicalMemViews;

  // Loops already rescaled to stick granularity.
  DenseSet<scf::ForOp> rescaledLoops;

  // Maps a physical value (a ktdp.load result, or a value Phase 2 has
  // retyped to physical) to what is known about it -- see PassContext's
  // physicalValues in Types.h. Seeded here in Phase 1 (retypeLoad) and grown
  // by Phase 2 as it retypes elementwise ops and erases transposes.
  llvm::DenseMap<mlir::Value, mlir::triton::ktdp::PhysicalValueInfo> physicalValues;

  // Resolved from the pass option: true = "device" (physical row-major strides),
  // false = "host" (derive strides from logical strides via coord map).
  bool hwDataLayout = false;

  // --- Stride computation ---

  // Compute physical strides as row-major of the physical shape.
  static std::pair<SmallVector<int64_t>, SmallVector<Value>>
  buildPhysicalStrides(unsigned physRank, ArrayRef<int64_t> physStaticSizes,
                       ArrayRef<Value> physDynSizes, OpBuilder &b,
                       Location loc) {
    SmallVector<int64_t> physStaticStrides(physRank);
    SmallVector<Value> physDynStrides;

    bool hasAnyDynStride = false;
    for (unsigned k = 0; k < physRank; ++k) {
      if (physStaticSizes[k] == ShapedType::kDynamic) {
        physStaticStrides[k] = ShapedType::kDynamic;
        hasAnyDynStride = true;
      }
    }
    if (!hasAnyDynStride) {
      physStaticStrides[physRank - 1] = 1;
      for (int k = (int)physRank - 2; k >= 0; --k)
        physStaticStrides[k] =
            physStaticStrides[k + 1] * physStaticSizes[k + 1];
    } else {
      SmallVector<Value> strideSsaVals(physRank);
      Value one =
          arith::ConstantOp::create(b, loc, b.getIndexAttr(1)).getResult();
      strideSsaVals[physRank - 1] = one;
      auto getSizeVal = [&](unsigned dim) -> Value {
        if (physStaticSizes[dim] != ShapedType::kDynamic)
          return arith::ConstantOp::create(
                     b, loc, b.getIndexAttr(physStaticSizes[dim]))
              .getResult();
        int pos = 0;
        for (unsigned d = 0; d < dim; ++d)
          if (physStaticSizes[d] == ShapedType::kDynamic)
            ++pos;
        return physDynSizes[pos];
      };
      for (int k = (int)physRank - 2; k >= 0; --k) {
        Value innerSize = getSizeVal(k + 1);
        strideSsaVals[k] =
            arith::MulIOp::create(b, loc, strideSsaVals[k + 1], innerSize)
                .getResult();
      }
      for (unsigned k = 0; k < physRank; ++k) {
        physStaticStrides[k] = ShapedType::kDynamic;
        physDynStrides.push_back(strideSsaVals[k]);
      }
    }
    return {std::move(physStaticStrides), std::move(physDynStrides)};
  }

  // Compute physical strides from logical strides via the coordinate map.
  static FailureOr<std::pair<SmallVector<int64_t>, SmallVector<Value>>>
  buildLogicalStrides(unsigned physRank, ArrayRef<int64_t> physSrc,
                      ArrayRef<int64_t> physOp, ArrayRef<int64_t> physArg,
                      ArrayRef<int64_t> logStaticStrides,
                      ArrayRef<Value> logDynStrides,
                      ArrayRef<int> logDynStrideIdx, OpBuilder &b, Location loc,
                      llvm::function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<int64_t> physStaticStrides(physRank);
    SmallVector<Value> physDynStrides;

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
        physStaticStrides[k] = logSt;
      }
    }
    if (hasAnyDynStride) {
      for (unsigned k = 0; k < physRank; ++k) {
        if (physStaticStrides[k] != ShapedType::kDynamic)
          continue;
        int64_t s = physSrc[k];
        auto op = static_cast<CoordOp>(physOp[k]);
        int64_t arg = physArg[k];
        if (logDynStrideIdx[s] < 0)
          return emitError()
                 << "spyre_tensor_layout: expected dynamic stride for dim";
        Value logDynSt = logDynStrides[logDynStrideIdx[s]];
        if (op == CoordOp::FloorDiv) {
          Value argVal =
              arith::ConstantOp::create(b, loc, b.getIndexAttr(arg));
          physDynStrides.push_back(
              arith::MulIOp::create(b, loc, logDynSt, argVal).getResult());
        } else {
          physDynStrides.push_back(logDynSt);
        }
      }
    }
    return std::make_pair(std::move(physStaticStrides),
                          std::move(physDynStrides));
  }

  // --- Loop rescaling ---

  // After rescaleEnclosingLoop(iv, factor), fix muli(iv, C) constants.
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

  // Rescale an scf.for loop from block units to stick units, where `factor` is
  // the number of sticks spanned by one block.  Pass 2 of rewriteAccessTile
  // consumes the IV directly as the physical stick index, so all three bounds
  // must be converted together to preserve the iteration space.
  void rescaleEnclosingLoop(scf::ForOp forOp, int64_t factor) {
    // factor == 1 means block and stick units already agree; there is nothing
    // to convert and the loop must be left exactly as it is.
    if (factor <= 1)
      return;
    LLVM_DEBUG(llvm::dbgs() << "  rescaling loop by factor " << factor << "\n");
    Type ivTy = forOp.getInductionVar().getType();
    OpBuilder b(forOp);
    Location loc = forOp.getLoc();
    Value factorV =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(ivTy, factor));
    auto scale = [&](Value v) -> Value {
      return arith::MulIOp::create(b, loc, v, factorV).getResult();
    };
    forOp.setLowerBound(scale(forOp.getLowerBound()));
    forOp.setUpperBound(scale(forOp.getUpperBound()));
    forOp.setStep(scale(forOp.getStep()));
  }

  // --- Access tile rewriting ---

  // Rebuild ConstructAccessTilesOp with the physical memView + block shape.
  LogicalResult rewriteAccessTile(mlir::ktdp::ConstructAccessTilesOp tileOp,
                                  Value newMemView,
                                  ArrayRef<int64_t> physSrc,
                                  ArrayRef<int64_t> physOp,
                                  ArrayRef<int64_t> physArg,
                                  triton::SpyreTensorLayoutOp marker) {
    OpBuilder b(tileOp);
    Location loc = tileOp.getLoc();

    auto logTileType = tileOp.getResult().getType();
    ArrayRef<int64_t> logBlock = logTileType.getShape();
    unsigned logRank = logBlock.size();
    unsigned physRank = physSrc.size();

    // Compute physical block shape via applyCoordMap.
    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, physSrc, physOp, physArg, physBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: cannot derive static block_shape");
    for (unsigned k = 0; k < physRank; ++k)
      if (physSrc[k] < 0 || physSrc[k] >= (int64_t)logRank)
        return tileOp.emitError("spyre_tensor_layout: phys_src out of range");

    LLVM_DEBUG({
      llvm::dbgs() << "  access tile physBlock: ";
      llvm::interleaveComma(physBlock, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Validate stick width.
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
    //
    // The op's base_map may have fewer inputs than logRank when the custom
    // parser deduplicates identical SSA operands (e.g. [%x, %x] becomes a
    // single operand with base_map (d0) -> (d0, d0)).  Expand through the
    // base_map to recover per-logical-dim values.
    SmallVector<Value> rawIndices(tileOp.getIndices().begin(),
                                  tileOp.getIndices().end());
    AffineMap baseMap = tileOp.getBaseMap();
    SmallVector<Value> logIdx(logRank);
    if (baseMap.getNumResults() == logRank &&
        baseMap.getNumInputs() == rawIndices.size()) {
      for (unsigned d = 0; d < logRank; ++d) {
        AffineExpr expr = baseMap.getResult(d);
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
          logIdx[d] = rawIndices[dimExpr.getPosition()];
        else
          logIdx[d] = rawIndices[0];
      }
    } else {
      logIdx.assign(rawIndices.begin(), rawIndices.end());
    }

    // Two passes: first rescale loops (mutating side effect), then compute
    // indices.  This avoids a latent issue where two physical dims sharing the
    // same logical source SSA value could see inconsistent IR if rescaling for
    // the first dim mutated state read by the second.

    // Pass 1: rescale enclosing loops for all FloorDiv dims (idempotent via
    // rescaledLoops set).
    for (unsigned k = 0; k < physRank; ++k) {
      if (static_cast<CoordOp>(physOp[k]) != CoordOp::FloorDiv)
        continue;
      Value logI = logIdx[physSrc[k]];
      BlockArgument iv = traceToMLIRBlockArg(logI);
      scf::ForOp forOp = iv ? dyn_cast_or_null<scf::ForOp>(
                                  iv.getOwner()->getParentOp())
                            : nullptr;
      if (forOp && forOp.getInductionVar() == iv) {
        if (rescaledLoops.insert(forOp).second) {
          rescaleEnclosingLoop(forOp, physBlock[k]);
          scaleDownIVMuls(iv, physBlock[k]);
        }
      }
    }

    // Pass 2: compute all physical index values from (now-stable) IR.
    SmallVector<Value> physIdx;
    for (unsigned k = 0; k < physRank; ++k) {
      int64_t src = physSrc[k];
      auto op = static_cast<CoordOp>(physOp[k]);
      int64_t arg = physArg[k];
      Value logI = logIdx[src];

      // FloorDiv on a rescaled loop IV: the IV itself is already the physical
      // index (rescaleEnclosingLoop in Pass 1 adjusted the trip count).
      if (op == CoordOp::FloorDiv) {
        BlockArgument iv = traceToMLIRBlockArg(logI);
        scf::ForOp forOp = iv ? dyn_cast_or_null<scf::ForOp>(
                                    iv.getOwner()->getParentOp())
                              : nullptr;
        if (forOp && forOp.getInductionVar() == iv) {
          Value ivIdx = iv.getType().isIndex()
                            ? iv
                            : arith::IndexCastOp::create(b, loc,
                                  b.getIndexType(), iv).getResult();
          physIdx.push_back(ivIdx);
          continue;
        }
      }

      // The split itself is built in `index`; lift its input there too, so no
      // fixed-width arithmetic is left between the subscript and the values it
      // derives from.
      physIdx.push_back(
          applyCoordOp(b, loc, rebuildInIndexDomain(b, loc, logI), op, arg));
    }

    Value newTileResult = mlir::triton::ktdp::buildAccessTile(
        b, loc, newMemView, physBlock, physIdx);

    // Update consumers (ktdp.load / ktdp.store).
    for (auto *user : llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        retypeLoad(ld, newTileResult, physBlock, marker);
      } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        redirectStoreAccessTile(st, newTileResult);
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  // Rebuild ConstructIndirectAccessTilesOp over the physical memView.
  LogicalResult rewriteIndirectAccessTile(
      mlir::ktdp::ConstructIndirectAccessTilesOp tileOp, Value newMemView,
      ArrayRef<int64_t> physSrc, ArrayRef<int64_t> physOp,
      ArrayRef<int64_t> physArg, triton::SpyreTensorLayoutOp marker) {
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

    // Capability gate.
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
    for (unsigned p = 0; p < physRank; ++p)
      if (physSrc[p] == 0 &&
          static_cast<CoordOp>(physOp[p]) != CoordOp::Identity)
        return tileOp.emitError(
            "spyre_tensor_layout: stick-splitting the indirect (gather) row "
            "dim is not supported");

    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, physSrc, physOp, physArg, physBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: cannot derive static block_shape for "
          "indirect access tile");

    unsigned newDimCount = numCaptured + physRank;
    auto newVar = [&](unsigned slot) { return getAffineDimExpr(slot, ctx); };

    // Reconstruct logical iteration variables from physical ones.
    SmallVector<AffineExpr> logicalFromPhysical(logRank);
    SmallVector<bool> contributed(logRank, false);
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t L = physSrc[p];
      if (L < 0 || L >= (int64_t)logRank)
        return tileOp.emitError(
            "spyre_tensor_layout: phys_src out of range for indirect tile");
      auto op = static_cast<CoordOp>(physOp[p]);
      int64_t arg = physArg[p];
      AffineExpr v = newVar(numCaptured + p);

      AffineExpr piece;
      switch (op) {
      case CoordOp::Identity: piece = v;       break;
      case CoordOp::FloorDiv: piece = v * arg; break;
      case CoordOp::Mod:      piece = v;       break;
      }

      logicalFromPhysical[L] =
          contributed[L] ? logicalFromPhysical[L] + piece : piece;
      contributed[L] = true;
    }

    // Build substitution from old domain to new domain.
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

      auto oldKindAttr = cast<BoolAttr>(oldKinds[L]);
      auto oldMapAttr  = cast<AffineMapAttr>(oldMaps[L]);

      AffineExpr oldExpr = oldMapAttr.getValue().getResult(0);
      AffineExpr reExpr = oldExpr.replaceDims(oldToNew);

      AffineExpr physExpr = applyCoordOpExpr(reExpr, op, arg);

      newKinds.push_back(oldKindAttr);
      newMaps.push_back(AffineMapAttr::get(
          AffineMap::get(newDimCount, /*symbolCount=*/0, physExpr, ctx)));
    }

    // Build new intermediate-variable space.
    SmallVector<AffineExpr> setConstraints;
    SmallVector<bool> setEqFlags;
    for (unsigned p = 0; p < physRank; ++p) {
      AffineExpr v = getAffineDimExpr(p, ctx);
      setConstraints.push_back(v);
      setEqFlags.push_back(false);
      setConstraints.push_back(
          getAffineConstantExpr(physBlock[p] - 1, ctx) - v);
      setEqFlags.push_back(false);
    }
    auto newSpaceSet = IntegerSet::get(physRank, 0, setConstraints, setEqFlags);
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

    // Update consumers.
    for (auto *user : llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        retypeLoad(ld, newTile.getResult(), physBlock, marker);
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of indirect access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  // --- Load/store consumer updates ---

  // Retype ktdp.load: replace with a new load of the physical tensor type.
  // Phase 1 stops here — it no longer forward-retypes the consuming chain;
  // Phase 2 decides each consuming op individually (see ContractionSynthesis.cpp).
  void retypeLoad(mlir::ktdp::LoadOp ld, Value newTile,
                  ArrayRef<int64_t> physBlock,
                  triton::SpyreTensorLayoutOp marker) {
    OpBuilder b(ld);
    auto elemTy = cast<RankedTensorType>(ld.getResult().getType())
                      .getElementType();
    auto physResTy = RankedTensorType::get(physBlock, elemTy);
    auto newLd = mlir::ktdp::LoadOp::create(b, ld.getLoc(), physResTy, newTile);
    ld.getResult().replaceAllUsesWith(newLd.getResult());
    ld.erase();
    // Seed Phase 2's reachability map: this is a physicalized load result,
    // the root of every forward elementwise chain Phase 2 will retype, and
    // carries this descriptor's marker for dispatchSource/findMarkerForStore
    // to read directly (no backward walk needed).
    physicalValues[newLd.getResult()] =
        mlir::triton::ktdp::PhysicalValueInfo{marker, {}};
  }

  // Redirect ktdp.store's access tile operand to the new physical tile.
  void redirectStoreAccessTile(mlir::ktdp::StoreOp st, Value newTile) {
    st.getAccessTileMutable().set(newTile);
  }

  // --- Phase 1: physicalize one descriptor ---

  /// Captures all analysis decisions for physicalizing a single descriptor.
  /// Pure data — no IR references that become stale during materialization
  /// (memViewOp, basePtr, tiles are read before any mutation).
  struct PhysicalViewPlan {
    // Coord map from the marker.
    ArrayRef<int64_t> physSrc, physOp, physArg;
    unsigned physRank = 0;

    // Logical sizes/strides from the original memory view.
    SmallVector<int64_t> logStaticSizes;
    SmallVector<int64_t> logStaticStrides;
    SmallVector<Value> logDynSizes;
    SmallVector<Value> logDynStrides;
    SmallVector<int> logDynIdx;
    SmallVector<int> logDynStrideIdx;

    // Computed physical sizes.
    SmallVector<int64_t> physStaticSizes;
    // Indices into logDynSizes for dynamic physical dims (floordiv/identity).
    // -1 means the physical dim is static. For floordiv dims, the materializer
    // will emit a ceildiv; for identity dims it forwards the dynamic value.
    SmallVector<int> physDynSizeSource;
    SmallVector<CoordOp> physDynSizeOp;
    SmallVector<int64_t> physDynSizeArg;

    // Physical strides (fully static case only; empty if any stride is dynamic).
    SmallVector<int64_t> physStaticStrides;
    bool stridesAreAllStatic = false;

    // Access tiles to rewrite.
    SmallVector<mlir::ktdp::ConstructAccessTilesOp> directTiles;
    SmallVector<mlir::ktdp::ConstructIndirectAccessTilesOp> indirectTiles;

    // The original memory view op (loc is derived via memViewOp.getLoc()).
    mlir::ktdp::ConstructMemoryViewOp memViewOp = nullptr;
    Value memView; // result of the original construct_memory_view
  };

  /// Pure analysis: compute the plan from the marker. No IR mutation.
  FailureOr<PhysicalViewPlan>
  planPhysicalization(triton::SpyreTensorLayoutOp marker) {
    Value desc = marker.getDesc();

    if (!isLoweredDescriptor(desc))
      return marker.emitError(
          "spyre_tensor_layout: desc operand is not a lowered descriptor — "
          "pass must run after LowerDescriptorMemory");

    Value memView = getDescriptorMemView(desc);
    auto memViewOp = memView.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
    if (!memViewOp)
      return marker.emitError(
          "spyre_tensor_layout: cannot locate construct_memory_view behind cast");

    PhysicalViewPlan plan;
    plan.physSrc = marker.getPhysSrc();
    plan.physOp = marker.getPhysOp();
    plan.physArg = marker.getPhysArg();
    plan.physRank = plan.physSrc.size();
    plan.memViewOp = memViewOp;
    plan.memView = memView;

    LLVM_DEBUG({
      llvm::dbgs() << "[rewrite-descriptor-layout] physicalizing: physRank="
                   << plan.physRank << "\n";
      llvm::dbgs() << "  physSrc="; llvm::interleaveComma(plan.physSrc, llvm::dbgs()); llvm::dbgs() << "\n";
      llvm::dbgs() << "  physOp="; llvm::interleaveComma(plan.physOp, llvm::dbgs()); llvm::dbgs() << "\n";
      llvm::dbgs() << "  physArg="; llvm::interleaveComma(plan.physArg, llvm::dbgs()); llvm::dbgs() << "\n";
    });

    // Read logical sizes and strides from the original memory view.
    plan.logStaticSizes.assign(memViewOp.getStaticSizes().begin(),
                               memViewOp.getStaticSizes().end());
    plan.logStaticStrides.assign(memViewOp.getStaticStrides().begin(),
                                 memViewOp.getStaticStrides().end());
    plan.logDynSizes.assign(memViewOp.getSizes().begin(),
                            memViewOp.getSizes().end());
    plan.logDynStrides.assign(memViewOp.getStrides().begin(),
                              memViewOp.getStrides().end());

    // Build index maps for dynamic dims.
    plan.logDynIdx.assign(plan.logStaticSizes.size(), -1);
    {
      int dynPos = 0;
      for (unsigned i = 0; i < plan.logStaticSizes.size(); ++i)
        if (plan.logStaticSizes[i] == ShapedType::kDynamic)
          plan.logDynIdx[i] = dynPos++;
    }
    plan.logDynStrideIdx.assign(plan.logStaticStrides.size(), -1);
    {
      int dynPos = 0;
      for (unsigned i = 0; i < plan.logStaticStrides.size(); ++i)
        if (plan.logStaticStrides[i] == ShapedType::kDynamic)
          plan.logDynStrideIdx[i] = dynPos++;
    }

    // Compute physical static sizes and note dynamic dim sources.
    for (unsigned k = 0; k < plan.physRank; ++k) {
      int64_t src = plan.physSrc[k];
      auto op = static_cast<CoordOp>(plan.physOp[k]);
      int64_t arg = plan.physArg[k];
      if (src < 0 || src >= (int64_t)plan.logStaticSizes.size())
        return marker.emitError("spyre_tensor_layout: phys_src out of range");

      int64_t logSz = plan.logStaticSizes[src];
      auto physSz = applyStatic(logSz, op, arg);
      if (physSz) {
        plan.physStaticSizes.push_back(*physSz);
        plan.physDynSizeSource.push_back(-1);
        plan.physDynSizeOp.push_back(CoordOp::Identity);
        plan.physDynSizeArg.push_back(0);
      } else {
        plan.physStaticSizes.push_back(ShapedType::kDynamic);
        if (op == CoordOp::FloorDiv) {
          if (plan.logDynIdx[src] < 0)
            return marker.emitError(
                "spyre_tensor_layout: expected dynamic size for floordiv dim");
          plan.physDynSizeSource.push_back(plan.logDynIdx[src]);
          plan.physDynSizeOp.push_back(CoordOp::FloorDiv);
          plan.physDynSizeArg.push_back(arg);
        } else {
          if (plan.logDynIdx[src] < 0)
            return marker.emitError(
                "spyre_tensor_layout: expected dynamic size for identity dim");
          plan.physDynSizeSource.push_back(plan.logDynIdx[src]);
          plan.physDynSizeOp.push_back(CoordOp::Identity);
          plan.physDynSizeArg.push_back(0);
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  physical sizes: ";
      llvm::interleaveComma(plan.physStaticSizes, llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Compute physical strides if all static; otherwise leave for materializer.
    if (hwDataLayout) {
      // Row-major strides: check if all sizes are static.
      bool allStatic = llvm::none_of(plan.physStaticSizes, [](int64_t s) {
        return s == ShapedType::kDynamic;
      });
      if (allStatic) {
        plan.physStaticStrides.resize(plan.physRank);
        plan.physStaticStrides[plan.physRank - 1] = 1;
        for (int k = (int)plan.physRank - 2; k >= 0; --k)
          plan.physStaticStrides[k] =
              plan.physStaticStrides[k + 1] * plan.physStaticSizes[k + 1];
        plan.stridesAreAllStatic = true;
      }
    } else {
      // Logical strides: check if all source logical strides are static.
      bool allStatic = true;
      for (unsigned k = 0; k < plan.physRank; ++k) {
        int64_t s = plan.physSrc[k];
        if (plan.logStaticStrides[s] == ShapedType::kDynamic) {
          allStatic = false;
          break;
        }
      }
      if (allStatic) {
        plan.physStaticStrides.resize(plan.physRank);
        for (unsigned k = 0; k < plan.physRank; ++k) {
          int64_t s = plan.physSrc[k];
          auto op = static_cast<CoordOp>(plan.physOp[k]);
          int64_t arg = plan.physArg[k];
          int64_t logSt = plan.logStaticStrides[s];
          if (op == CoordOp::FloorDiv)
            plan.physStaticStrides[k] = logSt * arg;
          else
            plan.physStaticStrides[k] = logSt;
        }
        plan.stridesAreAllStatic = true;
      }
    }

    // Collect access tiles that use the original memory view.
    for (auto *user : memView.getUsers()) {
      if (auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(user))
        plan.directTiles.push_back(tile);
      else if (auto tile =
                   dyn_cast<mlir::ktdp::ConstructIndirectAccessTilesOp>(user))
        plan.indirectTiles.push_back(tile);
    }

    return plan;
  }

  /// Emission: create new ops from the plan. Mutates IR.
  LogicalResult materializePhysicalView(PhysicalViewPlan &plan,
                                        triton::SpyreTensorLayoutOp marker) {
    OpBuilder b(plan.memViewOp);
    Location loc = plan.memViewOp.getLoc();

    // Build dynamic physical sizes.
    SmallVector<Value> physDynSizes;
    for (unsigned k = 0; k < plan.physRank; ++k) {
      if (plan.physDynSizeSource[k] < 0)
        continue; // static dim
      Value logDynSize = plan.logDynSizes[plan.physDynSizeSource[k]];
      if (plan.physDynSizeOp[k] == CoordOp::FloorDiv) {
        Value argIdx = arith::ConstantOp::create(
            b, loc, b.getIndexAttr(plan.physDynSizeArg[k]));
        physDynSizes.push_back(
            arith::CeilDivSIOp::create(b, loc, logDynSize, argIdx).getResult());
      } else {
        physDynSizes.push_back(logDynSize);
      }
    }

    // Compute physical strides.
    SmallVector<int64_t> physStaticStrides;
    SmallVector<Value> physDynStrides;
    if (plan.stridesAreAllStatic) {
      physStaticStrides = plan.physStaticStrides;
    } else if (hwDataLayout) {
      std::tie(physStaticStrides, physDynStrides) =
          buildPhysicalStrides(plan.physRank, plan.physStaticSizes, physDynSizes,
                               b, loc);
    } else {
      auto result = buildLogicalStrides(
          plan.physRank, plan.physSrc, plan.physOp, plan.physArg,
          plan.logStaticStrides, plan.logDynStrides, plan.logDynStrideIdx, b,
          loc, [&]() { return marker.emitError(); });
      if (failed(result))
        return failure();
      std::tie(physStaticStrides, physDynStrides) = std::move(*result);
    }

    // Physical memref type.
    auto logMemrefType = cast<MemRefType>(plan.memViewOp.getResult().getType());
    Value newMemView = mlir::triton::ktdp::buildMemoryView(
        b, loc, plan.memViewOp.getOffset(), plan.physStaticSizes,
        physStaticStrides, physDynSizes, physDynStrides,
        logMemrefType.getElementType(), plan.memViewOp.getMemorySpace());

    // Record the physical memView -> marker mapping for Phase 2.
    physMemViewToMarker[newMemView] = marker;

    // Rebuild each access tile that uses the old memView.
    for (auto tileOp : plan.directTiles) {
      if (failed(rewriteAccessTile(tileOp, newMemView, plan.physSrc, plan.physOp,
                                   plan.physArg, marker)))
        return failure();
    }
    for (auto tileOp : plan.indirectTiles) {
      if (failed(rewriteIndirectAccessTile(tileOp, newMemView, plan.physSrc,
                                           plan.physOp, plan.physArg, marker)))
        return failure();
    }

    // The logical view is now superseded by the physical one. Its only
    // remaining user is the marker's bridge cast, which Phase 3 erases; queue
    // the view so Phase 3 can drop it too.
    deadLogicalMemViews.push_back(plan.memViewOp);

    return success();
  }

  LogicalResult rewriteOnePhysicalize(triton::SpyreTensorLayoutOp marker) {
    auto planOrErr = planPhysicalization(marker);
    if (failed(planOrErr))
      return failure();
    return materializePhysicalView(*planOrErr, marker);
  }

  // --- Phase 3: marker cleanup ---

  // Erase a marker and its now-dead bridge cast.
  void eraseMarker(triton::SpyreTensorLayoutOp marker) {
    if (!marker->getBlock())
      return;
    Value desc = marker.getDesc();
    auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
    marker.erase();
    if (castOp && castOp.use_empty())
      castOp.erase();
  }

  // --- Pass entry point ---

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Resolve the data-layout option. Reject anything unrecognized rather
    // than silently falling through to the "host" branch below — the pass is
    // also invocable directly (spyre-triton-opt), bypassing the frontend's
    // own validation.
    if (dataLayout != "device" && dataLayout != "host") {
      module.emitError("rewrite-descriptor-layout: data-layout must be "
                       "'device' or 'host', got '")
          << dataLayout << "'";
      return signalPassFailure();
    }
    hwDataLayout = (dataLayout == "device");

    // Collect markers up front; mutating while walking invalidates the cursor.
    SmallVector<triton::SpyreTensorLayoutOp> markers;
    module.walk([&](triton::SpyreTensorLayoutOp op) { markers.push_back(op); });

    LLVM_DEBUG(llvm::dbgs() << "[rewrite-descriptor-layout] found "
                            << markers.size() << " layout markers\n");

    // Phase 1: physicalize each annotated descriptor.
    for (auto marker : markers)
      if (failed(rewriteOnePhysicalize(marker)))
        return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "[rewrite-descriptor-layout] Phase 1 complete, "
                            << "entering Phase 2 (contraction synthesis)\n");

    // Phase 2A: decide, for every value reachable from Phase 1's roots, the
    // final physical type it will carry -- before Phase 2 rewrites anything.
    // Mutates no IR, creates no ops (see PhysicalTypeAnalysis.h).
    PassContext ctx{physMemViewToMarker, physicalValues};
    PhysicalTypeMap physicalTypeMap = runPhysicalTypeAnalysis(module, ctx);
    ctx.physicalTypeAnalysis = &physicalTypeMap;
    // Phase 2B reads that map through the const pointer above and writes it
    // through this handle, which grants exactly one operation -- see
    // PhysicalTypeCarryForward.
    ctx.physicalTypes = PhysicalTypeCarryForward(physicalTypeMap);

    // Phase 2: synthesize contractions via greedy pattern rewrite.
    {
      RewritePatternSet patterns(module.getContext());
      populateContractionPatterns(patterns, ctx);
      // Collect candidate ops: the four contraction-family ops, plus every
      // other op that consumes a physical value (elementwise ops and
      // linalg.transpose) so Phase 2 decides them too, rather than Phase 1
      // retyping/erasing them in passing.
      //
      // The elementwise membership test here is deliberately not
      // isSingleTensorElementwiseOp (see its comment): that predicate would
      // silently exclude multi-tensor-operand elementwise ops like arith.addf
      // from ever being retyped by RewriteElementwisePattern. This test only
      // needs to be a superset of what that pattern's own local rule
      // matches: any single-result op with at least one tensor operand.
      SmallVector<Operation *> candidates;
      module.walk([&](Operation *op) {
        if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ReduceOp,
                mlir::ktdp::StoreOp, linalg::TransposeOp>(op)) {
          candidates.push_back(op);
          return;
        }
        if (op->getNumResults() != 1 ||
            !isa<RankedTensorType>(op->getResult(0).getType()))
          return;
        bool hasTensorOperand = llvm::any_of(op->getOperands(), [](Value v) {
          return isa<RankedTensorType>(v.getType());
        });
        if (hasTensorOperand)
          candidates.push_back(op);
      });
      GreedyRewriteConfig config;
      config.enableFolding(false);
      config.enableConstantCSE(false);
      config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
      // Converges only if every pattern's match condition is falsified by its
      // own rewrite: the driver re-enqueues an op whenever a neighbour it feeds
      // or consumes changes, so a pattern that stays matchable spins to the
      // iteration cap. Discarding this result would report success for that.
      if (failed(applyOpPatternsGreedily(
              candidates, FrozenRewritePatternSet(std::move(patterns)),
              config))) {
        module.emitError("rewrite-descriptor-layout: Phase 2 did not reach a "
                         "fixpoint; a pattern is re-matching its own output");
        return signalPassFailure();
      }
      if (ctx.hadError)
        return signalPassFailure();
      // At end of Phase 2, everything Phase 2 found to be physical was
      // predicted by Phase 2A.
      verifyPhysicalTypeAgreement(module, ctx, physicalTypeMap,
                                  "end of Phase 2");
    }

    LLVM_DEBUG(llvm::dbgs() << "[rewrite-descriptor-layout] Phase 2 complete, "
                            << "erasing " << markers.size() << " markers\n");

    // Phase 3: erase all markers (and their now-dead bridge casts).
    for (auto marker : markers)
      eraseMarker(marker);

    // With the bridge casts gone, the logical views Phase 1 superseded have no
    // users left. Erase them -- but check rather than assume: a view whose
    // access tiles Phase 1 could not fully re-point still has real users.
    for (auto memViewOp : deadLogicalMemViews)
      if (memViewOp->getBlock() && memViewOp->use_empty())
        memViewOp.erase();
  }
};

} // namespace
