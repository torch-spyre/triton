//===- LowerInterTile.cpp - Lower tt.inter_tile_reduce to KTDP ops --------===//
//
// Expands each tt.inter_tile_reduce into:
//   ktdp.inter_tile_produce  (per-tile partial, producer region)
//     + one delivery op (ktdp.inter_tile_reduce for all_reduce / reduce_to_one)
//
// Algorithm:
//   1. Collect all tt.inter_tile_reduce ops (collect-then-rewrite to avoid
//      invalidating the walk cursor when expansions insert/erase ops).
//   2. For each op:
//      a. Fold-away guard  — W[axis]==1 → forward partial(s), erase.
//      b. Validate         — axis, mode, scatter_dimension, combiner.
//      c. Build group sets — derive gsize/ngroups, emit affine_set attrs.
//      d. Select delivery  — all_reduce or reduce_to_one.
//      e. Emit produce     — ktdp.inter_tile_produce + yield_partial region.
//      f. Build combiner   — shorthand → linalg.fill+op; region → transcribe.
//      g. Emit delivery    — ktdp.inter_tile_reduce + yield_reduced region.
//      h. Emit dep set     — producer_dependency_per_consumer from depWkSlices.
//      i. RAUW + erase     — replace tt op uses with delivery results.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "InterTile/Grouping.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_LOWERINTERTILE
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

// The shared inter-tile grouping helpers, so the pass body below reads the same
// as it did when they were file-local.
using mlir::triton::ktdp::buildGroupSets;
using mlir::triton::ktdp::buildPick0Set;
using mlir::triton::ktdp::combinerEmitOp;
using mlir::triton::ktdp::GroupSets;
using mlir::triton::ktdp::readWorkSliceAttrs;
using mlir::triton::ktdp::WorkSliceAttrs;

namespace {

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct LowerInterTilePass
    : public mlir::triton::ktdp::impl::LowerInterTileBase<LowerInterTilePass> {

  using LowerInterTileBase::LowerInterTileBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // The DMV lowering is selected but not yet implemented, so refuse rather
    // than silently emitting the delivery pair the caller did not ask for.
    if (interTileLowering == "dmv") {
      mod.emitError("inter-tile-lowering='dmv' is not yet implemented");
      return signalPassFailure();
    }
    if (interTileLowering != "delivery") {
      mod.emitError("inter-tile-lowering must be 'delivery' or 'dmv', got '")
          << interTileLowering << "'";
      return signalPassFailure();
    }

    // Collect all inter_tile_reduce ops first (collect-then-rewrite).
    SmallVector<triton::InterTileReduceOp> ops;
    mod.walk([&](triton::InterTileReduceOp op) { ops.push_back(op); });

    for (auto op : ops) {
      if (failed(lowerOne(op, rewriter)))
        return signalPassFailure();
    }
  }

  LogicalResult lowerOne(triton::InterTileReduceOp op, IRRewriter &rewriter) {
    Location loc = op.getLoc();
    MLIRContext *ctx = &getContext();

    // --- find enclosing tt.func (pass runs before ConvertFunctions) ---
    auto func = op->getParentOfType<triton::FuncOp>();
    if (!func)
      return op.emitError("inter_tile_reduce must be inside a tt.func");


    // --- read work-slice attributes (carried on the op, set by frontend) ---
    auto attrsOrErr = readWorkSliceAttrs(op);
    if (failed(attrsOrErr)) return failure();
    WorkSliceAttrs attrs = *attrsOrErr;

    StringRef axis    = op.getAxis();
    StringRef mode    = op.getMode();
    StringRef combiner = op.getCombiner();
    auto partials     = op.getPartials();

    // --- validate axis present in W ---
    if (!attrs.numWkSlicesPerDim.getAs<IntegerAttr>(axis))
      return op.emitError("axis '") << axis
             << "' not in numWkSlicesPerDim";

    // --- validate mode ---
    if (mode != "all_reduce" && mode != "reduce_to_one" &&
        mode != "reduce_scatter" && mode != "broadcast")
      return op.emitError("unknown mode '") << mode << "'";

    // --- reject deferred modes ---
    if (mode == "broadcast" || mode == "reduce_scatter")
      return op.emitError("mode '") << mode << "' is not yet supported";

    // --- scatter_dimension present iff reduce_scatter ---
    bool hasSd = (bool)op.getScatterDimension();
    if (mode == "reduce_scatter" && !hasSd)
      return op.emitError("reduce_scatter requires scatter_dimension");
    if (mode != "reduce_scatter" && hasSd)
      return op.emitError("scatter_dimension only valid for reduce_scatter");

    if (combiner.empty())
      return op.emitError("custom combiner regions are not yet supported");
    auto identities = op.getIdentities();
    if (identities.empty())
      return op.emitError("identities operand group must not be empty");

    // --- derive group sets ---
    auto gsOrErr = buildGroupSets(ctx, attrs, axis, op);
    if (failed(gsOrErr)) return failure();
    GroupSets gs = *gsOrErr;

    // --- fold-away: gsize == 1 → each tile is its own group ---
    if (gs.gsize == 1) {
      // No cooperation needed — forward partials as results. The tt op's
      // result type equals its partial type (no rank reduction), so
      // rewriting each downstream use of a tt result to the matching
      // partial value is type-safe by construction.
      rewriter.setInsertionPoint(op);
      op.replaceAllUsesWith(partials);
      rewriter.eraseOp(op);
      return success();
    }

    // --- select consumer set ---
    IntegerSet consumerSet = gs.producerTilesPerGroup;  // all_reduce default
    if (mode == "reduce_to_one") {
      auto pick0OrErr = buildPick0Set(ctx, gs, op);
      if (failed(pick0OrErr)) return failure();
      consumerSet = *pick0OrErr;
    }

    // --- emit ktdp.inter_tile_produce ---
    rewriter.setInsertionPoint(op);

    SmallVector<Type> partialTypes(partials.getTypes());
    SmallVector<RankedTensorType> rankedPartials;
    rankedPartials.reserve(partialTypes.size());
    for (Type pTy : partialTypes)
      rankedPartials.push_back(cast<RankedTensorType>(pTy));
    auto futureType = ktdp::TileFutureType::get(rankedPartials, gs.groups);

    auto produceOp = ktdp::InterTileProduceOp::create(
        rewriter, loc,
        futureType,
        IntegerSetAttr::get(gs.producerTilesPerGroup));

    // Producer region: single block with %gid: index arg, yield_partial.
    Block *produceBlock = &produceOp.getBody().emplaceBlock();
    produceBlock->addArgument(rewriter.getIndexType(), loc);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(produceBlock);
      ktdp::YieldPartialOp::create(rewriter, loc, partials);
    }

    // Result types == partial types. The ktdp.inter_tile_reduce verifier
    // enforces this equality; grouping is expressed by the affine sets,
    // not by a tensor axis, so no dim is collapsed here.
    SmallVector<Type> resultTypes(partialTypes.begin(), partialTypes.end());

    // Identities are always provided on the tt op (semantic.py materializes
    // them for shorthand combiners at TTIR construction time).
    SmallVector<Value> identityValues(identities.begin(), identities.end());

    // --- emit ktdp.inter_tile_reduce ---
    auto reduceOp = ktdp::InterTileReduceOp::create(
        rewriter, loc,
        resultTypes,
        produceOp.getFuture(),
        identityValues,
        IntegerSetAttr::get(consumerSet),
        /*producer_dependency_per_consumer=*/IntegerSetAttr{});

    // Remove the placeholder null dep attr (create with no dep).
    reduceOp->removeAttr("producer_dependency_per_consumer");

    // --- emit per-consumer dependency ---
    if (attrs.depWkSlices) {
      if (failed(attachDepSet(rewriter, loc, ctx, gs, attrs, reduceOp, op, mode)))
        return failure();
    }
    // Absent D → full-barrier (attribute omitted — already done above).

    // --- build reducer region ---
    if (failed(buildReducerRegion(rewriter, loc, reduceOp,
                                  partialTypes, combiner)))
      return failure();

    // --- RAUW + erase ---
    rewriter.replaceOp(op, reduceOp.getResults());
    return success();
  }

  LogicalResult buildReducerRegion(IRRewriter &rewriter, Location loc,
                                   ktdp::InterTileReduceOp dstOp,
                                   ArrayRef<Type> partialTypes,
                                   StringRef combiner) {
    Block *block = &dstOp.getCombiner().emplaceBlock();
    SmallVector<Value> lhs, rhs;
    for (auto t : partialTypes)
      lhs.push_back(block->addArgument(t, loc));
    for (auto t : partialTypes)
      rhs.push_back(block->addArgument(t, loc));

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(block);

    SmallVector<Value> reduced;
    for (auto [l, r] : llvm::zip(lhs, rhs)) {
      auto tensorType = cast<RankedTensorType>(l.getType());
      Value out = tensor::EmptyOp::create(rewriter, loc,
                                          tensorType.getShape(),
                                          tensorType.getElementType());
      auto result = combinerEmitOp(rewriter, loc, combiner, l, r, out);
      if (failed(result))
        return dstOp.emitError("unknown shorthand combiner '") << combiner << "'";
      reduced.push_back(*result);
    }

    ktdp::YieldReducedOp::create(rewriter, loc, reduced);
    return success();
  }

  // Attach producer_dependency_per_consumer from depWkSlices.
  LogicalResult attachDepSet(IRRewriter &rewriter, Location loc,
                             MLIRContext *ctx,
                             const GroupSets &gs,
                             const WorkSliceAttrs &attrs,
                             ktdp::InterTileReduceOp dstOp,
                             Operation *srcLoc,
                             StringRef mode) {
    // reduce_to_one: only pick0 (local index 0) is a consumer.
    // all_reduce: every member of the group is a consumer.
    int64_t numConsumers = (mode == "reduce_to_one") ? 1 : gs.gsize;

    // D: consumer local index (str) → list of producer local indices.
    // Validate: all indices in [0, gsize-1], non-empty lists, full coverage.
    SmallVector<SmallVector<int64_t>> depTable(gs.gsize);
    SmallVector<bool> consumerCovered(gs.gsize, false);
    SmallVector<bool> producerCovered(gs.gsize, false);

    for (auto entry : attrs.depWkSlices) {
      int64_t consLocal;
      if (entry.getName().getValue().getAsInteger(10, consLocal))
        return srcLoc->emitError("depWkSlices key '")
               << entry.getName().getValue()
               << "' is not a valid integer local index";
      if (consLocal < 0 || consLocal >= gs.gsize)
        return srcLoc->emitError("depWkSlices key ") << consLocal
               << " out of range [0, " << gs.gsize << ")";
      if (consLocal >= numConsumers)
        return srcLoc->emitError("depWkSlices key ") << consLocal
               << " is not a valid consumer for mode '" << mode
               << "' (only indices [0, " << numConsumers << ") are consumers)";
      auto prodList = dyn_cast<ArrayAttr>(entry.getValue());
      if (!prodList || prodList.empty())
        return srcLoc->emitError("depWkSlices[") << consLocal
               << "] is empty or not an array";
      consumerCovered[consLocal] = true;
      for (auto prodAttr : prodList) {
        int64_t prodLocal = cast<IntegerAttr>(prodAttr).getInt();
        if (prodLocal < 0 || prodLocal >= gs.gsize)
          return srcLoc->emitError("depWkSlices producer index ") << prodLocal
                 << " out of range [0, " << gs.gsize << ")";
        depTable[consLocal].push_back(prodLocal);
        producerCovered[prodLocal] = true;
      }
    }
    // Check consumer coverage (only consumers valid for this mode).
    for (int64_t i = 0; i < numConsumers; ++i)
      if (!consumerCovered[i])
        return srcLoc->emitError("depWkSlices missing consumer local index ") << i;
    // Check producer coverage.
    for (int64_t i = 0; i < gs.gsize; ++i)
      if (!producerCovered[i])
        return srcLoc->emitError("depWkSlices producer local index ") << i
               << " not depended upon by any consumer";

    // Build the affine set Dep(p)[c,g]:
    //   p in members(g) AND localIdx(p) in D[localIdx(c)]
    // MLIR IntegerSet doesn't natively support OR; for now emit a placeholder
    // (p >= 0, always true) so the attribute is present and the downstream
    // validator accepts it. Full affine set rendering is deferred.
    auto pExpr = getAffineDimExpr(0, ctx);  // p
    auto cExpr = getAffineDimExpr(1, ctx);  // c (unused in placeholder)
    auto gSym  = getAffineSymbolExpr(0, ctx);
    (void)pExpr; (void)cExpr; (void)gSym;

    // Placeholder: p >= 0 (always true).
    SmallVector<AffineExpr> cons = {getAffineDimExpr(0, ctx)};
    IntegerSet depSet = IntegerSet::get(2, 1, cons, {false});
    dstOp->setAttr("producer_dependency_per_consumer",
                   IntegerSetAttr::get(depSet));
    return success();
  }
};

} // namespace

namespace mlir::triton::ktdp {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerInterTilePass(LowerInterTileOptions options) {
  return std::make_unique<LowerInterTilePass>(options);
}

} // namespace mlir::triton::ktdp
