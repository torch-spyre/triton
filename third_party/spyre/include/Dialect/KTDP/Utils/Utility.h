//===- Utility.h - Shared ktdp-building helpers ---------------------------===//
//
// Helpers whose subject is the KTDP dialect: they build ktdp memory views and
// access tiles, or recognise the bridge cast that carries one.  Hence namespace
// mlir::triton::ktdp -- which is why the TritonToKTIR conversions, being in
// mlir::triton::spyre, call them qualified.
//
// Published because each has a consumer in both pass libraries.  Helpers that
// are about no dialect live in Utils/Utility.h; helpers with a single consuming
// library are private to it and not published at all.

#ifndef TRITON_SPYRE_DIALECT_KTDP_UTILS_UTILITY_H
#define TRITON_SPYRE_DIALECT_KTDP_UTILS_UTILITY_H

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "mlir/IR/Builders.h"

namespace mlir::triton::ktdp {

/// True iff `desc` is a memref-backed lowered descriptor (the
/// UnrealizedConversionCast bridge left by LowerDescriptorMemory).
bool isLoweredDescriptor(Value desc);

/// Unwrap the bridge cast to recover the ktdp.construct_memory_view result.
Value getDescriptorMemView(Value desc);

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

} // namespace mlir::triton::ktdp

#endif // TRITON_SPYRE_DIALECT_KTDP_UTILS_UTILITY_H
