#ifndef KTDP_TRANSFORMS_INTERTILE_GROUPING_H
#define KTDP_TRANSFORMS_INTERTILE_GROUPING_H

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::ktdp {

//===----------------------------------------------------------------------===//
// Attribute key constants
//===----------------------------------------------------------------------===//

inline constexpr StringRef kNumWkSlicesPerDim = "numWkSlicesPerDim";
inline constexpr StringRef kCoreIdToWkSlice   = "coreIdToWkSlice";
inline constexpr StringRef kDepWkSlices        = "depWkSlices";

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

FailureOr<WorkSliceAttrs> readWorkSliceAttrs(triton::InterTileReduceOp op);

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
FailureOr<GroupSets> buildGroupSets(MLIRContext *ctx, const WorkSliceAttrs &attrs,
                                    StringRef axis, Operation *loc);

//===----------------------------------------------------------------------===//
// buildPick0 — find the reduced-axis slice-0 tile in group g
//===----------------------------------------------------------------------===//

// Build the pick₀ consumer set from gs.pick0TileIds.
// pick0TileIds[g] is the tile-id with axis_value==0 in group g, scanned from
// coreIdToWkSlice in buildGroupSets. The ids must form an arithmetic sequence
// base + g*stride so the predicate can be expressed as the single affine
// equality i == base + g*stride.
FailureOr<IntegerSet> buildPick0Set(MLIRContext *ctx, const GroupSets &gs,
                                    Operation *loc);

//===----------------------------------------------------------------------===//
// CombinerSpec — dispatch helpers for shorthand combiners
//===----------------------------------------------------------------------===//

// Emits the reduction op for one (lhs, rhs, out) triple.
// Returns the scalar/tensor result Value, or failure() if unsupported.
// only used when the reducer region is not provided
FailureOr<Value> combinerEmitOp(OpBuilder &b, Location loc, StringRef combiner,
                                Value lhs, Value rhs, Value out);

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_INTERTILE_GROUPING_H
