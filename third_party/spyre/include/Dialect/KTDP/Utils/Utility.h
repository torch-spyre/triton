//===- Utility.h - Shared ktdp-building helpers ---------------------------===//
//
// Helpers whose subject is the KTDP dialect: they build ktdp memory views and
// access tiles, or recognise the bridge cast that carries one.  Hence namespace
// mlir::triton::ktdp -- which is why the TritonToKTIR conversions, being in
// mlir::triton::spyre, call them qualified.
//
// Published because each has a consumer outside this library -- most of them in
// both pass libraries.  Helpers that are about no dialect live in
// Utils/Utility.h; helpers with a single consuming library are private to it and
// not published at all.

#ifndef TRITON_SPYRE_DIALECT_KTDP_UTILS_UTILITY_H
#define TRITON_SPYRE_DIALECT_KTDP_UTILS_UTILITY_H

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir::triton::ktdp {

/// True iff `desc` is a memref-backed lowered descriptor (the
/// UnrealizedConversionCast bridge left by LowerDescriptorMemory).
bool isLoweredDescriptor(Value desc);

/// Unwrap the bridge cast to recover the ktdp.construct_memory_view result.
Value getDescriptorMemView(Value desc);

/// Build the dense range set of an N-D coordinate space: per dim, the pair of
/// constraints bounding it to [0, extent). An entry equal to
/// `ShapedType::kDynamic` contributes an IntegerSet symbol in place of a
/// constant upper bound, in dim order. A rank-0 shape gives the single
/// always-true constraint `0 >= 0`, because an IntegerSet cannot be built with
/// no constraints at all.
///
/// This is the coordinate set `buildMemoryView` and `buildAccessTile` derive,
/// published because a pass that rebuilds views and tiles itself has to build
/// the same set.
IntegerSet buildRangeSetND(MLIRContext *ctx, ArrayRef<int64_t> shape);

/// Build a `ktdp.construct_memory_view` of `staticSizes`/`staticStrides`
/// anchored at `baseIndex`. `staticSizes`/`staticStrides` may be empty for a
/// rank-0 view; entries equal to `ShapedType::kDynamic` draw their runtime
/// value from `dynSizes`/`dynStrides` in order, one per sentinel — the same
/// convention `ktdp.construct_memory_view`'s ODS builder and verifier use.
/// The coordinate set is derived from `staticSizes`.
Value buildMemoryView(OpBuilder &builder, Location loc, Value baseIndex,
                      ArrayRef<int64_t> staticSizes,
                      ArrayRef<int64_t> staticStrides, ValueRange dynSizes,
                      ValueRange dynStrides, Type elemType,
                      mlir::ktdp::MemorySpaceAttr memorySpace);

/// Build a `ktdp.construct_access_tile` of `blockShape` over `memView`,
/// anchored at `indices` (one per view dim, per the op's `base_map`
/// contract — an empty `blockShape`/`indices` pair builds a rank-0 tile).
/// Indices not already `index`-typed are cast to `index` first, since
/// Triton hands block/loop indices over as `i32`.
Value buildAccessTile(OpBuilder &builder, Location loc, Value memView,
                      ArrayRef<int64_t> blockShape, ValueRange indices);

//===----------------------------------------------------------------------===//
// The layout rewrite's seed traversals
//
// Two walks out of a `ktdp.construct_memory_view`, published because the layout
// rewrite is no longer their only caller: a pass that has to know WHICH ops that
// rewrite will reach -- to decline something it would bridge badly -- has to ask
// the same question, and a second copy of the walk would diverge from this one
// on the next change, in both directions.  They are stated in terms of ktdp ops
// on one side and a `linalg.generic` on the other, which is what makes them
// KTDP's rather than any one pass's.
//
// `Operation *` rather than the op classes, so this header stays the light
// include it is; callers cast, which they must do anyway to key their own sets.
//===----------------------------------------------------------------------===//

/// The loads and stores that reach `memView` through its access tiles, direct or
/// indirect.  ONE HOP each way: view -> `ktdp.construct_access_tile` /
/// `ktdp.construct_indirect_access_tile` -> `ktdp.load` / `ktdp.store`.  A user
/// of the view that is not an access tile is skipped, and so is a user of a tile
/// that is neither a load nor a store.  Appends; does not clear.
void collectViewAccesses(Operation *memView,
                         SmallVectorImpl<Operation *> &loads,
                         SmallVectorImpl<Operation *> &stores);

/// The `linalg.generic` ops ADJACENT to `memViews`: for each view, the direct
/// consumers of its loads plus the direct data producer of its stores.  Each
/// listed once, in the order the views are given.
///
/// This is the SCOPE of the layout rewrite -- one hop off a view's accesses, not
/// a transitive closure -- so it is also the scope any pass predicting that
/// rewrite has to use.
void collectAdjacentGenerics(ArrayRef<Operation *> memViews,
                             SmallVectorImpl<Operation *> &out);

} // namespace mlir::triton::ktdp

#endif // TRITON_SPYRE_DIALECT_KTDP_UTILS_UTILITY_H
