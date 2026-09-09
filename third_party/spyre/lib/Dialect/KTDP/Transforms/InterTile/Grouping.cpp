//===- Grouping.cpp - Work-slice group derivation for inter-tile lowering -===//
//
// Shared by the inter-tile lowerings: reading the work-slice attributes off a
// tt.inter_tile_reduce, deriving the cooperation groups they describe, and
// picking the combiner arithmetic for a shorthand combiner name. None of it
// names a target op, so a lowering that emits something other than the
// ktdp delivery pair reuses it unchanged — which is the point of it living
// here rather than in one pass's translation unit.
//
//===----------------------------------------------------------------------===//

#include "InterTile/Grouping.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

namespace mlir::triton::ktdp {

FailureOr<WorkSliceAttrs> readWorkSliceAttrs(triton::InterTileReduceOp op) {
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

FailureOr<GroupSets> buildGroupSets(MLIRContext *ctx, const WorkSliceAttrs &attrs,
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

  // Sort group keys for deterministic group-index assignment.
  llvm::sort(tupleOrder);

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
    llvm::sort(members);
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

FailureOr<IntegerSet> buildPick0Set(MLIRContext *ctx, const GroupSets &gs,
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

FailureOr<Value> combinerEmitOp(OpBuilder &b, Location loc, StringRef combiner,
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

} // namespace mlir::triton::ktdp
