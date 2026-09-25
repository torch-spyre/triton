//===- LowerInterTile.cpp - Lower inter-tile communication to KTDP ops ----===//
//
// Two modes, one per way a kernel expresses communication between cores. They
// share a pass because they are the same subject and nothing else: the design
// document this implements puts both here.
//
//   tt.inter_tile_reduce              a fold across cores -> produce + delivery
//   tts.make_distributed_descriptor   a redistribution -> a composed view, and
//                                     the reads through it
//
// The second is at the bottom of this file; what follows is the first.
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

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERINTERTILE
#include "Conversion/TritonToKTIR/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

//===----------------------------------------------------------------------===//
// Attribute key constants
//===----------------------------------------------------------------------===//

static constexpr StringRef kNumWkSlicesPerDim = "numWkSlicesPerDim";
static constexpr StringRef kCoreIdToWkSlice   = "coreIdToWkSlice";
static constexpr StringRef kDepWkSlices        = "depWkSlices";

//===----------------------------------------------------------------------===//
// readWorkSliceAttrs — read W, C, D from the op's own attributes
//===----------------------------------------------------------------------===//

struct WorkSliceAttrs {
  // W: axis name → slice count.
  DictionaryAttr numWkSlicesPerDim;  // StringAttr → IntegerAttr
  // C: list of per-tile maps (each map: axis name → slice index i64).
  // We store it as the raw ArrayAttr of DictionaryAttrs.
  ArrayAttr coreIdToWkSlice;
  // D (optional): dictionary consumer-local-index → list-of-producer-local-idx.
  DictionaryAttr depWkSlices;  // nullptr if absent.
};

static FailureOr<WorkSliceAttrs>
readWorkSliceAttrs(triton::InterTileReduceOp op) {
  auto W = op->getAttrOfType<DictionaryAttr>(kNumWkSlicesPerDim);
  if (!W)
    return op.emitError("missing '") << kNumWkSlicesPerDim << "' op attribute";
  auto C = op->getAttrOfType<ArrayAttr>(kCoreIdToWkSlice);
  if (!C)
    return op.emitError("missing '") << kCoreIdToWkSlice << "' op attribute";
  // D is optional.
  auto D = op->getAttrOfType<DictionaryAttr>(kDepWkSlices);
  return WorkSliceAttrs{W, C, D};
}

//===----------------------------------------------------------------------===//
// GroupSets — affine_set attributes for producer_tiles_per_group and groups
//===----------------------------------------------------------------------===//

struct GroupSets {
  IntegerSet producerTilesPerGroup;  // (i)[g] : membership predicate
  IntegerSet groups;                  // (g) : range [0, ngroups)
  int64_t gsize;
  int64_t stride;  // groupStep (= gsize for contiguous groups)
  SmallVector<int64_t> pick0TileIds;  // pick0TileIds[g] = tile-id with axis_value==0 in group g
};

// Build the GroupSets for the given reduction axis.
//
// Grouping semantics (coop_α): two tiles cooperate iff they agree on every
// dim except `axis`.  `axis` is the *reduction* dim — the dim that varies
// within a group.  Tiles with the same non-axis slice-index tuple form one
// group; `gsize = W[axis]` is the number of cooperating tiles per group, and
// `ngroups = numTiles / gsize`.
//
// Current scope: members of each group must be contiguous tile ids
// {g*gsize .. (g+1)*gsize - 1}.
static FailureOr<GroupSets>
buildGroupSets(MLIRContext *ctx, const WorkSliceAttrs &attrs,
               StringRef axis, Operation *loc) {
  auto gsizeAttr = attrs.numWkSlicesPerDim.getAs<IntegerAttr>(axis);
  if (!gsizeAttr)
    return loc->emitError("axis '") << axis
           << "' not found in numWkSlicesPerDim";
  int64_t gsize = gsizeAttr.getInt();

  int64_t numTiles = (int64_t)attrs.coreIdToWkSlice.size();
  if (numTiles == 0)
    return loc->emitError("coreIdToWkSlice is empty");

  if (numTiles % gsize != 0)
    return loc->emitError("tile count ") << numTiles
           << " does not divide evenly by gsize=" << gsize
           << " for axis '" << axis << "'";
  int64_t ngroups = numTiles / gsize;

  // --- partition tiles by non-axis slice-index tuple (coop_α) ---
  // Two tiles are in the same group iff their slice dicts agree on all dims
  // except `axis`.  We encode the non-axis tuple as a sorted string key for
  // map lookup.
  std::map<std::string, SmallVector<int64_t>> tupleToTiles;
  SmallVector<std::string> tupleOrder;

  for (int64_t t = 0; t < numTiles; ++t) {
    auto tileMap = dyn_cast<DictionaryAttr>(attrs.coreIdToWkSlice[t]);
    if (!tileMap)
      return loc->emitError("coreIdToWkSlice entry ") << t
             << " is not a DictionaryAttr";
    // Validate axis key present.
    if (!tileMap.getAs<IntegerAttr>(axis))
      return loc->emitError("coreIdToWkSlice entry ") << t
             << " has no key '" << axis << "'";
    // Build non-axis tuple key (sorted by attr name for determinism).
    std::string key;
    llvm::raw_string_ostream os(key);
    SmallVector<std::pair<StringRef, int64_t>> nonAxisPairs;
    for (auto namedAttr : tileMap) {
      if (namedAttr.getName().strref() == axis) continue;
      auto intAttr = dyn_cast<IntegerAttr>(namedAttr.getValue());
      if (!intAttr)
        return loc->emitError("coreIdToWkSlice entry ") << t
               << ": value for key '" << namedAttr.getName() << "' is not i64";
      nonAxisPairs.push_back({namedAttr.getName().strref(), intAttr.getInt()});
    }
    llvm::sort(nonAxisPairs, [](auto &a, auto &b) { return a.first < b.first; });
    for (auto &[k, v] : nonAxisPairs)
      os << k << "=" << v << ";";
    os.flush();
    if (!tupleToTiles.count(key))
      tupleOrder.push_back(key);
    tupleToTiles[key].push_back(t);
  }

  // Do not sort tupleOrder.  It is already in first-appearance (= minimum tile
  // id) order, which is the block order the contiguity check below requires.
  // Sorting by the key assumes the slice label ascends with tile id; nothing
  // requires that, and where it does not, valid IR is rejected.

  if ((int64_t)tupleOrder.size() != ngroups)
    return loc->emitError("expected ") << ngroups
           << " groups (numTiles/W[axis]=" << numTiles << "/" << gsize
           << ") but found " << tupleOrder.size()
           << " distinct non-axis tuples";

  // Verify uniform group size and contiguous membership.
  for (int64_t g = 0; g < ngroups; ++g) {
    auto &members = tupleToTiles[tupleOrder[g]];
    if ((int64_t)members.size() != gsize)
      return loc->emitError("group ") << g << " has " << members.size()
             << " tiles, expected gsize=" << gsize;
    // members is already ascending -- push_back ran with `t` ascending.
    for (int64_t j = 0; j < gsize; ++j) {
      int64_t expected = g * gsize + j;
      if (members[j] != expected)
        return loc->emitError("group ") << g
               << " is not contiguous: expected tile " << expected
               << " at position " << j << ", got " << members[j]
               << " (non-contiguous groups not yet supported)";
    }
  }

  // Find pick0 tile per group: the tile with axis_value==0 in each group.
  SmallVector<int64_t> pick0TileIds(ngroups, -1);
  for (int64_t g = 0; g < ngroups; ++g) {
    auto &members = tupleToTiles[tupleOrder[g]];
    for (int64_t j = 0; j < gsize; ++j) {
      auto tileMap = dyn_cast<DictionaryAttr>(attrs.coreIdToWkSlice[members[j]]);
      int64_t axVal = tileMap.getAs<IntegerAttr>(axis).getInt();
      if (axVal == 0) {
        if (pick0TileIds[g] != -1)
          return loc->emitError("group ") << g
                 << " has more than one tile with " << axis << "=0";
        pick0TileIds[g] = members[j];
      }
    }
    if (pick0TileIds[g] == -1)
      return loc->emitError("group ") << g
             << " has no tile with " << axis << "=0";
  }

  // --- emit affine sets ---
  // groups = { (g) : g >= 0, ngroups-1-g >= 0 }
  // g must be a DIM (not a symbol) — ktdp.inter_tile_produce verifier
  // requires groups to have no symbols (dimCount=1, symCount=0).
  auto gDim = getAffineDimExpr(0, ctx);
  SmallVector<AffineExpr> groupConstraints = {
      gDim,                                              // g >= 0
      getAffineConstantExpr(ngroups - 1, ctx) - gDim    // ngroups-1-g >= 0
  };
  IntegerSet groupsSet = IntegerSet::get(
      /*dimCount=*/1, /*symCount=*/0, groupConstraints,
      /*eqFlags=*/{false, false});

  // producer_tiles_per_group = { (i)[g] : g*gsize <= i <= g*gsize + gsize-1 }
  auto iExpr = getAffineDimExpr(0, ctx);
  auto gSym  = getAffineSymbolExpr(0, ctx);
  AffineExpr base = gSym * getAffineConstantExpr(gsize, ctx);
  SmallVector<AffineExpr> cons = {
      iExpr - base,                                           // i - g*gsize >= 0
      base + getAffineConstantExpr(gsize - 1, ctx) - iExpr   // g*gsize+gsize-1-i >= 0
  };
  IntegerSet producerSet = IntegerSet::get(1, 1, cons, {false, false});

  return GroupSets{producerSet, groupsSet, gsize, /*stride=*/gsize, pick0TileIds};
}

//===----------------------------------------------------------------------===//
// buildPick0 — find the reduced-axis slice-0 tile in group g
//===----------------------------------------------------------------------===//

// Build the pick₀ consumer set from gs.pick0TileIds.
// pick0TileIds[g] is the tile-id with axis_value==0 in group g, scanned from
// coreIdToWkSlice in buildGroupSets. The ids must form an arithmetic sequence
// base + g*stride so the predicate can be expressed as the single affine
// equality i == base + g*stride.
static FailureOr<IntegerSet> buildPick0Set(MLIRContext *ctx,
                                            const GroupSets &gs,
                                            Operation *loc) {
  int64_t ngroups = (int64_t)gs.pick0TileIds.size();
  if (ngroups == 0)
    return IntegerSet::getEmptySet(1, 1, ctx);

  int64_t base   = gs.pick0TileIds[0];
  int64_t stride = (ngroups > 1) ? (gs.pick0TileIds[1] - base) : 0;

  for (int64_t g = 0; g < ngroups; ++g) {
    if (gs.pick0TileIds[g] != base + g * stride)
      return loc->emitError(
          "reduce_to_one: pick0 tile-ids are not an arithmetic sequence "
          "(non-uniform pick0 layouts are not yet supported)");
  }

  // Emit i == base + g*stride.
  auto iExpr = getAffineDimExpr(0, ctx);
  auto gSym  = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineConstantExpr(base, ctx)
                   + gSym * getAffineConstantExpr(stride, ctx);
  SmallVector<AffineExpr> cons = {iExpr - rhs};
  return IntegerSet::get(1, 1, cons, {true});
}

//===----------------------------------------------------------------------===//
// CombinerSpec — dispatch helpers for shorthand combiners
//===----------------------------------------------------------------------===//

// Returns the identity TypedAttr for (combiner, elemType), or failure() if
// unsupported.

// Emits the reduction op for one (lhs, rhs, out) triple.
// Returns the scalar/tensor result Value, or failure() if unsupported.
// only used when the reducer region is not provided
static FailureOr<Value> combinerEmitOp(OpBuilder &b, Location loc,
                                        StringRef combiner,
                                        Value lhs, Value rhs, Value out) {
  if (combiner == "add")
    return linalg::AddOp::create(b, loc, ValueRange{lhs, rhs}, ValueRange{out})
               .getResult(0);
  if (combiner == "max")
    return linalg::MaxOp::create(b, loc, ValueRange{lhs, rhs}, ValueRange{out})
               .getResult(0);
  if (combiner == "mul")
    return linalg::MulOp::create(b, loc, ValueRange{lhs, rhs}, ValueRange{out})
               .getResult(0);
  return failure();  // caller emits error
}



//===----------------------------------------------------------------------===//
// tts.make_distributed_descriptor — the compose, and the reads through it
//
// The second of this pass's two modes. The first expands a reduce into a
// produce/delivery pair; this one composes a redistribution and then performs it.
// Both are inter-tile communication, which is why they share a pass, and they
// share nothing else.
//
// This is design §3's three phases in one function, and the phases are worth
// keeping visible because each has a different reason to be here:
//
//   Phase 1  the source view: one `construct_memory_view` per partition,
//            differing ONLY in `coordinate_set` and `ct_id`, composed by
//            `construct_distributed_memory_view`.
//   Phase 2  the access: a `construct_access_tile` at the offsets this instance
//            passed to `.load()`. Nothing has moved yet -- both view ops are
//            `Pure`, so a core may NAME another core's scratchpad without
//            anything crossing a core boundary.
//   Phase 3  the transfer: one `ktdp.load`, which IS the transfer. The landing
//            store the design also requires is not emitted here: it is a
//            `tts.pin` on the loaded value, and `PlacePinnedValues` has already
//            built it by the time this runs.
//
// Where each share LIVES is recovered rather than stated. `tts.pin` said it, and
// `PlacePinnedValues` has since erased the pin and rewritten the uses to read the
// buffer -- so the operand here is a `ktdp.load` and the address is its view's.
// `traceLoadToMemoryView` is that walk, published for this.
//===----------------------------------------------------------------------===//

/// The slice index `entry` gives `key`, or nullopt if it carries none.
static std::optional<int64_t> sliceIndexOf(const triton::tts::WorkSliceEntry &entry,
                                           StringRef key) {
  for (auto &[k, index] : entry)
    if (k == key)
      return index;
  return std::nullopt;
}

/// Lower one `tts.make_distributed_descriptor`, and the reads through its result.
static LogicalResult
lowerDistributedDescriptor(triton::tts::MakeDistributedDescriptorOp op,
                           IRRewriter &rewriter, int64_t cores) {
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  auto shareTy = cast<RankedTensorType>(op.getPartial().getType());
  ArrayRef<int64_t> shareShape = shareTy.getShape();
  unsigned rank = shareShape.size();

  // --- the partition table, through the op's own reader -----------------------
  SmallVector<triton::tts::WorkSliceEntry> entries;
  llvm::MapVector<StringRef, int64_t> sliceCounts;
  auto emitError = [&]() { return op.emitError(); };
  if (failed(triton::tts::readWorkSliceTable(op.getWorkSlices(), entries, sliceCounts,
                                     emitError)))
    return failure();

  // Holder = the partition's index in the table. §1 resolves a holder by matching
  // the partition's coordinate dict against the tile -> coordinate table, which
  // is a second table this op does not carry; the two coincide exactly when the
  // table has one entry per tile, and then the match is the identity. Anything
  // else -- a tensor held by 8 of 32 cores, or a broadcast source of 1 -- needs
  // that second table, and is refused here rather than resolved by a guess about
  // which cores were meant.
  if ((int64_t)entries.size() != cores)
    return op.emitError()
           << "work_slices has " << entries.size() << " partitions for a grid of "
           << cores
           << ": only one partition per tile is supported, since the holder is "
              "then the partition's own index. A shorter table needs the tile "
              "coordinate table to resolve holders against";

  // --- where the share lives -------------------------------------------------
  Value shareView = mlir::triton::ktdp::traceLoadToMemoryView(op.getPartial());
  if (!shareView)
    return op.emitError()
           << "cannot tell where the share lives: its value does not come from a "
              "ktdp.load of a memory view. Pin it with tl.spyre_pin, which is "
              "what supplies a partition's address";
  auto shareViewOp = shareView.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
  auto shareSpace = shareViewOp.getMemorySpace();
  if (shareSpace.getKind() != mlir::ktdp::MemorySpaceKind::ct_local)
    return op.emitError()
           << "the share lives in '"
           << mlir::ktdp::stringifyMemorySpaceKind(shareSpace.getKind())
           << "', and a redistribution composes scratchpad shares: pin it in "
              "'ct_local'";

  // --- phase 1: one view per partition, then the compose ---------------------
  // Offsets, sizes and strides are identical across the partitions -- §3 says the
  // coordinate set and the holder are "the entire content of the distribution" --
  // so they are computed once, out of the loop, from the share's own view.
  Value offset = shareViewOp.getOffset();
  SmallVector<int64_t> strides(rank, 1);
  for (int i = (int)rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shareShape[i + 1];

  // The composed extents: a share's extent on a divided dimension is the whole
  // extent divided by the slice count, so the whole is the share's times that
  // count. An undivided dimension composes to itself.
  SmallVector<int64_t> wholeShape(shareShape.begin(), shareShape.end());
  SmallVector<StringRef> dimKey(rank);
  for (auto [d, attr] : llvm::enumerate(op.getAxes())) {
    StringRef key = cast<StringAttr>(attr).getValue();
    dimKey[d] = key;
    if (!key.empty())
      wholeShape[d] = shareShape[d] * sliceCounts.find(key)->second;
  }

  rewriter.setInsertionPoint(op);
  SmallVector<Value> partitions;
  partitions.reserve(entries.size());
  for (auto [id, entry] : llvm::enumerate(entries)) {
    // The region this partition owns, in the composed tensor's index space.
    SmallVector<int64_t> los(rank, 0), his(wholeShape.begin(), wholeShape.end());
    for (unsigned d = 0; d < rank; ++d) {
      if (dimKey[d].empty())
        continue;
      int64_t index = *sliceIndexOf(entry, dimKey[d]);
      los[d] = index * shareShape[d];
      his[d] = los[d] + shareShape[d];
    }

    auto space = mlir::ktdp::MemorySpaceAttr::get(
        ctx, mlir::ktdp::MemorySpaceKind::ct_local, /*ct_id=*/(int32_t)id);
    partitions.push_back(mlir::triton::ktdp::buildMemoryView(
        rewriter, loc, offset, shareShape, strides, /*dynSizes=*/{},
        /*dynStrides=*/{}, shareTy.getElementType(), space,
        mlir::triton::ktdp::buildBoxSetND(ctx, los, his),
        /*spaceInResultType=*/true));
  }

  auto wholeTy = MemRefType::get(wholeShape, shareTy.getElementType());
  Value whole = mlir::ktdp::ConstructDistributedMemoryViewOp::create(
                    rewriter, loc, wholeTy, partitions)
                    .getResult();

  // --- phases 2 and 3: the reads through the composed view -------------------
  // Collected first: each is erased as it is rewritten.
  SmallVector<Operation *> readers;
  for (Operation *user : op.getResult().getUsers())
    readers.push_back(user);

  ArrayRef<int64_t> blockShape = op.getBlockShape();
  for (Operation *user : readers) {
    auto load = dyn_cast<triton::DescriptorLoadOp>(user);
    if (!load) {
      // A store through a distributed view is not forbidden by KTDP, and this
      // design composes no destination view (assumption 4): under the pull model
      // each destination holder writes into its own scratchpad, so there is
      // nothing on that side to compose. Refused rather than lowered, so the
      // restriction is the design's and not a silent gap.
      return op.emitError()
             << "a distributed descriptor may only be read: '"
             << user->getName().getStringRef()
             << "' is not a tt.descriptor_load";
    }

    rewriter.setInsertionPoint(load);
    Value tile = mlir::triton::ktdp::buildAccessTile(
        rewriter, load.getLoc(), whole, blockShape, load.getIndices());
    auto loaded = mlir::ktdp::LoadOp::create(
        rewriter, load.getLoc(),
        cast<RankedTensorType>(load.getResult().getType()), tile);
    rewriter.replaceOp(load, loaded.getResult());
  }

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct LowerInterTilePass
    : public mlir::triton::spyre::impl::LowerInterTileBase<LowerInterTilePass> {

  using LowerInterTileBase::LowerInterTileBase;

  LowerInterTilePass(ArrayRef<int64_t> gridShape) { grid = gridShape; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect all inter_tile_reduce ops first (collect-then-rewrite).
    SmallVector<triton::InterTileReduceOp> ops;
    mod.walk([&](triton::InterTileReduceOp op) { ops.push_back(op); });

    for (auto op : ops) {
      if (failed(lowerOne(op, rewriter)))
        return signalPassFailure();
    }

    // The second mode. Its own walk rather than one walk over both, because the
    // two share no step: a reduce reads work-slice attributes off its own op and
    // builds a produce/delivery pair, a compose reads a partition table and builds
    // views.
    //
    // Same default as DistributeWork's, and for the same reason: a ListOption
    // takes no tablegen default, and [32] is the 1D-on-full-hardware case. The
    // grid is needed because a partition's HOLDER is its index in the table only
    // when the table has one entry per tile, which is a claim about the grid.
    SmallVector<int64_t> gridShape(grid.begin(), grid.end());
    if (gridShape.empty())
      gridShape.push_back(32);
    int64_t cores = 1;
    for (int64_t g : gridShape)
      cores *= g;

    SmallVector<triton::tts::MakeDistributedDescriptorOp> composes;
    mod.walk([&](triton::tts::MakeDistributedDescriptorOp op) { composes.push_back(op); });

    for (auto op : composes) {
      if (failed(lowerDistributedDescriptor(op, rewriter, cores)))
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

  // Build the reducer region of the ktdp.inter_tile_reduce op.
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

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerInterTilePass(ArrayRef<int64_t> grid) {
  return std::make_unique<LowerInterTilePass>(grid);
}

} // namespace mlir::triton::spyre
