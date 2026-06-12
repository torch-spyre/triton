//===- LowerDescriptorMemory.cpp - Lower tt.descriptor_* to ktdp ops -----===//
//
// Lowers the tensor descriptor memory path to KTDP dialect ops:
//   tt.descriptor_load/store  -> ktdp.construct_memory_view +
//                                ktdp.construct_access_tile +
//                                ktdp.load/store
//   tt.descriptor_gather/scatter -> ktdp.construct_indirect_access_tile +
//                                   ktdp.load/store
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_LOWERDESCRIPTORMEMORY
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// True iff `desc` is a memref-backed descriptor produced by walk 1
/// of `runOnOperation`.  Walk 1 attaches a memref to every
/// `tt.make_tensor_descriptor` it rewrites, and walk 2's access-op
/// patterns rely on that memref to lower the load/store/gather/scatter.
/// The precondition walk in `runOnOperation` uses this same predicate
/// so the diagnostics and the pattern call sites stay in sync.
static bool isLoweredDescriptor(Value desc) {
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp && !castOp.getInputs().empty() &&
         isa<MemRefType>(castOp.getInputs()[0].getType());
}

/// Recover the memref that backs an access op's `desc` operand.
///
/// Walk 1 of `runOnOperation` replaces every `tt.make_tensor_descriptor`
/// with a memref view of the underlying buffer; this helper hands that
/// memref back to each access-op pattern, which consumes it directly.
/// The pre-condition walk in `runOnOperation` validates that every
/// legal access op has its descriptor lowered, so on the success path
/// the predicate is guaranteed to hold.  The assert keeps the invariant
/// *local* to the helper so a future caller bypassing `runOnOperation`
/// (e.g. a stand-alone unit test of the patterns) still trips a clear
/// failure rather than indexing into an empty operand list.
static Value getDescriptorMemView(Value desc) {
  assert(isLoweredDescriptor(desc) &&
         "descriptor operand was not lowered by walk 1 — "
         "precondition check should have caught this");
  auto castOp = desc.getDefiningOp<UnrealizedConversionCastOp>();
  return castOp.getInputs()[0];
}

static Value getBasePtrAsIndex(OpBuilder &builder, Location loc,
                               Value basePtr) {
  if (basePtr.getType().isIndex())
    return basePtr;
  return UnrealizedConversionCastOp::create(builder, loc,
                                              builder.getIndexType(), basePtr)
      .getResult(0);
}

/// Build a range-set constraint for an N-D coordinate space.
/// Static dims use arith constants; dynamic dims use IntegerSet symbols,
/// which are bound positionally to the op's dynamic sizes operands.
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
    if (shape[i] == ShapedType::kDynamic) {
      // Symbol s_j is bound to dynSizes[j] by the op (positional binding).
      // Constraint: d_i <= s_j - 1  =>  s_j - 1 - d_i >= 0
      upper = getAffineSymbolExpr(symIdx++, ctx) - 1;
    } else {
      upper = getAffineConstantExpr(shape[i] - 1, ctx);
    }
    constraints.push_back(di);           // d_i >= 0
    eqFlags.push_back(false);
    constraints.push_back(upper - di);   // upper - d_i >= 0
    eqFlags.push_back(false);
  }
  return IntegerSet::get(rank, symCount, constraints, eqFlags);
}

/// Try to extract a compile-time int64 from an SSA value produced by
/// arith.constant.  Returns std::nullopt if the value is not a constant.
static std::optional<int64_t> getConstantInt(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Shared memory view construction
//===----------------------------------------------------------------------===//

/// Build a ktdp.construct_memory_view over the full tensor described by
/// descOp.  Shape/strides are extracted from the descriptor's SSA values
/// as compile-time constants when available, with kDynamic fallback for
/// non-constant values.  Hardcodes HBM memory space.
static Value buildBaseMemoryView(OpBuilder &builder, Location loc,
                                 triton::MakeTensorDescOp descOp,
                                 Type elemType) {
  MLIRContext *ctx = builder.getContext();
  Value baseIndex = getBasePtrAsIndex(builder, loc, descOp.getBase());

  // Extract shape/strides as constants when possible, kDynamic otherwise.
  SmallVector<int64_t> shape;
  SmallVector<Value> dynSizes;
  for (auto s : descOp.getShape()) {
    if (auto c = getConstantInt(s)) {
      shape.push_back(*c);
    } else {
      shape.push_back(ShapedType::kDynamic);
      dynSizes.push_back(
          arith::IndexCastOp::create(builder, loc, builder.getIndexType(), s));
    }
  }
  // Fallback: if the descriptor has no explicit shape, use the block shape.
  if (shape.empty()) {
    auto blockType =
        cast<triton::TensorDescType>(descOp.getResult().getType()).getBlockType();
    shape.assign(blockType.getShape().begin(), blockType.getShape().end());
  }

  SmallVector<int64_t> strides;
  SmallVector<Value> dynStrides;
  for (auto s : descOp.getStrides()) {
    if (auto c = getConstantInt(s)) {
      strides.push_back(*c);
    } else {
      strides.push_back(ShapedType::kDynamic);
      dynStrides.push_back(
          arith::IndexCastOp::create(builder, loc, builder.getIndexType(), s));
    }
  }
  // Fallback: compute default row-major strides from the shape.
  if (strides.empty()) {
    int64_t stride = 1;
    strides.resize(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      if (shape[i] != ShapedType::kDynamic)
        stride *= shape[i];
    }
  }

  auto memrefType = MemRefType::get(shape, elemType);
  auto memSpaceAttr = mlir::ktdp::SpyreMemorySpaceAttr::get(
      ctx, mlir::ktdp::SpyreMemorySpaceKind::HBM, /*core=*/-1);

  auto memView = mlir::ktdp::ConstructMemoryViewOp::create(
      builder, loc, memrefType, baseIndex,
      dynSizes, dynStrides,
      builder.getDenseI64ArrayAttr(SmallVector<int64_t>(shape)),
      builder.getDenseI64ArrayAttr(strides),
      memSpaceAttr,
      IntegerSetAttr::get(buildRangeSetND(ctx, shape)));

  return memView.getResult();
}

//===----------------------------------------------------------------------===//
// Direct access tile construction (load/store)
//===----------------------------------------------------------------------===//

/// Build ktdp.construct_access_tile for a direct (contiguous block) access.
/// The memory view describes the full tensor; the block indices position the
/// tile within it.  No manual offset computation needed.
static Value buildDirectAccessTile(OpBuilder &builder, Location loc,
                                   Value memView, ArrayRef<int64_t> blockShape,
                                   ValueRange indices) {
  MLIRContext *ctx = builder.getContext();
  auto indexType = builder.getIndexType();

  auto accessTileType = mlir::ktdp::AccessTileType::get(blockShape, indexType);
  unsigned rank = blockShape.size();
  auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);

  // Cast index operands to index type (they arrive as i32 from Triton).
  SmallVector<Value> indexOperands;
  for (auto idx : indices)
    indexOperands.push_back(
        arith::IndexCastOp::create(builder, loc, indexType, idx));

  auto accessTile = mlir::ktdp::ConstructAccessTilesOp::create(
      builder, loc, accessTileType, memView,
      identityMap, indexOperands, /*symbol_operands=*/ValueRange{},
      buildRangeSetND(ctx, blockShape), identityMap);

  return accessTile.getResult();
}

//===----------------------------------------------------------------------===//
// Indirect access tile helpers (gather/scatter)
//===----------------------------------------------------------------------===//

/// Resolved index view + descriptor_load anchor offset for a
/// gather/scatter `x_offsets` tensor.  `view` is the memref the indirect
/// access tile reads from (e.g. `memref<N_ELEMENTS x si32>`); `offset`
/// is the descriptor_load anchor offset — i.e. the `[offset]` argument
/// to `idx_desc.load([offset])`.  Carrying `offset` out of the trace
/// lets the gather lowering consume `view[offset + d0]` instead of the
/// offset-less `view[d0]`, which would otherwise silently drop the
/// descriptor_load offset and always read the buffer's prefix (the bug
/// fixed by this trace path).  Both fields are non-null on the success
/// path.
struct ResolvedIndexView {
  Value view;
  Value offset;
};

/// Try to resolve a tensor that was loaded via
///   ktdp.load ← construct_access_tile ← construct_memory_view
/// to its source memory view and the descriptor_load anchor offset.
///
/// When `tensor` is the SSA result of a `tt.descriptor_load` that has
/// already been lowered to ktdp (which is the only legal provenance for
/// a gather's `x_offsets` post-fallback-removal), this walks the chain
/// and reuses the source memory view directly — *and* recovers the
/// descriptor_load's `[offset]` operand so the gather indirect map can
/// apply it (RFC 0682; without this, the offset is silently dropped).
///
/// Returns `std::nullopt` on a trace miss: the chain is not present
/// (e.g. `tensor` is a function argument with tensor type, not a Spyre
/// kernel-arg shape).  The caller turns this into `failure()` so
/// `applyPartialConversion` surfaces "failed to legalize" against the
/// unconverted gather/scatter op.
///
/// Asserts (invariants from `ConvertDescriptorLoad`): when the chain
/// matches, the access tile carries exactly one base index, since the
/// only ops in this pipeline that emit `construct_access_tile` for an
/// index buffer come from rank-1 descriptor_loads.  An assert failure
/// here is a compiler bug, not bad user input.
static std::optional<ResolvedIndexView>
traceToSourceMemoryView(Value tensor) {
  auto loadOp = tensor.getDefiningOp<mlir::ktdp::LoadOp>();
  if (!loadOp)
    return std::nullopt;

  auto tileOp = loadOp.getAccessTile()
                    .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
  if (!tileOp)
    return std::nullopt;

  // `indices` are the `index`-typed offset operands the descriptor_load
  // lowering passed to construct_access_tile (one per dim of the view).
  auto indices = tileOp.getIndices();
  assert(indices.size() == 1 &&
         "traced rank-1 index buffer must carry exactly one base index — "
         "ConvertDescriptorLoad invariant violated");
  return ResolvedIndexView{tileOp.getBase(), indices.front()};
}

/// Resolve an `x_offsets` tensor to a `ResolvedIndexView` for use by
/// `ktdp.construct_indirect_access_tile`.  Thin adapter over
/// :func:`traceToSourceMemoryView` that translates a trace miss
/// (`std::nullopt`) into `failure()` for the conversion driver, and
/// asserts the trace's element-type invariant.
///
/// Worked example.  Suppose the kernel has a 1024-element index
/// buffer and gathers it in tiles of 32 in a loop:
///
///     idx_desc = make_tensor_descriptor(idx_ptr, [1024], [1])
///     for m in range(0, 1024, 32):
///         x_offsets = idx_desc.load([m])      # shape: <32xi32>
///         tile      = desc.gather(x_offsets, y_off)
///
///   - resolved->view   : memref<1024xi32>     (full index buffer)
///   - xOffsets type    : tensor<32xi32>       (one tile)
///   - resolved->offset : %m                   (loop iv, in [0, 1024))
///
/// The descriptor's full size (1024) may exceed the tile size (32);
/// the indirect map then reads
///   x_offsets[%m + d0]   for d0 ∈ [0, 32)
/// which traverses indices [%m, %m+32) of the full buffer — exactly
/// the tile the descriptor_load would have materialised.  Without
/// capturing `%m`, the map would degenerate to `x_offsets[d0]` and
/// every iteration would re-read the buffer's first 32 entries.
static FailureOr<ResolvedIndexView>
resolveIndexView(Value xOffsets) {
  auto resolved = traceToSourceMemoryView(xOffsets);
  if (!resolved)
    return failure();

  // Invariant from ConvertDescriptorLoad: the traced view is a ranked
  // memref whose rank and element-storage-width match the descriptor's
  // block type, which is also the type of `xOffsets`.  An assert here
  // means the lowering pipeline is broken, not the user input.
  //
  // Element types are compared by integer bit width because the
  // descriptor block type can be `si32` (signed) while
  // `tt.descriptor_load` canonicalises its result tensor to `i32`
  // (signless): the two sides describe the same storage but disagree
  // on signedness.
  auto memrefType = cast<MemRefType>(resolved->view.getType());
  auto tensorType = cast<RankedTensorType>(xOffsets.getType());
  auto memInt = dyn_cast<IntegerType>(memrefType.getElementType());
  auto tenInt = dyn_cast<IntegerType>(tensorType.getElementType());
  bool elemSameStorage =
      (memInt && tenInt && memInt.getWidth() == tenInt.getWidth()) ||
      memrefType.getElementType() == tensorType.getElementType();
  (void)elemSameStorage;
  assert(memrefType.getRank() == tensorType.getRank() && elemSameStorage &&
         "traced index memory view disagrees with x_offsets tensor type — "
         "ConvertDescriptorLoad invariant violated");

  return *resolved;
}

/// Build the subscript kinds, subscript maps, variable space set, and
/// variable space order for an N-D gather pattern:
///   result[d_0, d_1, d_2, ..., d_{rank-1}]
///       = base[ x_offsets[x_offset + d_0],
///               y_offset + d_1,
///               d_2,
///               ...,
///               d_{rank-1} ]
///
/// `x_offset` is the descriptor_load offset into the index buffer
/// (the `[offset]` argument to `idx_desc.load([offset])`); it is
/// always captured because the only legal `x_offsets` provenance is a
/// descriptor_load (see resolveIndexView).
///
/// Captured variable layout in the affine domain (left to right):
///   c_x = x_offset, c_y = y_offset, then d_0 .. d_{rank-1}.
///
/// Affine maps:
///   dim 0  (indirect):     (c_x, c_y, d_0, ...) -> c_x + d_0
///   dim 1  (direct, y_off): (c_x, c_y, d_0, ...) -> c_y + d_1
///   dim i  (direct, plain, for i in [2, rank)):
///                          (c_x, c_y, d_0, ...) -> d_i
///
/// `variables_space_set` constrains each d_i to [0, resultShape[i]);
/// `variables_space_order` is the identity over (d_0, ..., d_{rank-1}).
///
/// `numCaptured` is fixed at 2 regardless of rank: tt.descriptor_gather
/// carries one descriptor, one 1-D index tensor, and one scalar y_offset
/// at every rank, so the captured-variable list never grows.
struct GatherSubscriptInfo {
  ArrayAttr subscriptKinds;
  ArrayAttr subscriptMaps;
  IntegerSet spaceSet;
  AffineMap spaceOrder;
};

static GatherSubscriptInfo
buildGatherSubscriptMaps(MLIRContext *ctx, ArrayRef<int64_t> resultShape) {
  assert(resultShape.size() >= 2 &&
         "gather subscript maps require rank >= 2");

  auto kindTrue = BoolAttr::get(ctx, true);
  auto kindFalse = BoolAttr::get(ctx, false);

  // Two captured scalars (c_x, c_y) followed by `rank` iteration
  // variables (d_0 .. d_{rank-1}).
  constexpr unsigned numCaptured = 2u;
  unsigned rank = resultShape.size();
  unsigned dimCount = numCaptured + rank;
  constexpr unsigned cxSlot = 0u;          // captured x_offset
  constexpr unsigned cySlot = 1u;          // captured y_offset
  unsigned d0Slot = numCaptured;           // first iteration variable

  SmallVector<Attribute> kinds;
  SmallVector<Attribute> maps;
  kinds.reserve(rank);
  maps.reserve(rank);

  // dim 0: indirect, c_x + d_0
  kinds.push_back(kindTrue);
  maps.push_back(AffineMapAttr::get(AffineMap::get(
      dimCount, /*symbolCount=*/0,
      getAffineDimExpr(cxSlot, ctx) + getAffineDimExpr(d0Slot, ctx), ctx)));

  // dim 1: direct with y_offset, c_y + d_1
  kinds.push_back(kindFalse);
  maps.push_back(AffineMapAttr::get(AffineMap::get(
      dimCount, /*symbolCount=*/0,
      getAffineDimExpr(cySlot, ctx) + getAffineDimExpr(d0Slot + 1, ctx), ctx)));

  // dims [2, rank): direct, no offset, d_i
  for (unsigned i = 2; i < rank; ++i) {
    kinds.push_back(kindFalse);
    maps.push_back(AffineMapAttr::get(AffineMap::get(
        dimCount, /*symbolCount=*/0,
        getAffineDimExpr(d0Slot + i, ctx), ctx)));
  }

  // Intermediate-variable space: 0 <= d_i < resultShape[i] for each i.
  SmallVector<AffineExpr> constraints;
  SmallVector<bool> eqFlags;
  constraints.reserve(2 * rank);
  eqFlags.reserve(2 * rank);
  for (unsigned i = 0; i < rank; ++i) {
    // d_i >= 0
    constraints.push_back(getAffineDimExpr(i, ctx));
    eqFlags.push_back(false);
    // resultShape[i] - 1 - d_i >= 0
    constraints.push_back(getAffineConstantExpr(resultShape[i] - 1, ctx) -
                          getAffineDimExpr(i, ctx));
    eqFlags.push_back(false);
  }
  auto spaceSet = IntegerSet::get(
      /*dimCount=*/rank, /*symbolCount=*/0, constraints, eqFlags);

  return {
      ArrayAttr::get(ctx, kinds),
      ArrayAttr::get(ctx, maps),
      spaceSet,
      AffineMap::getMultiDimIdentityMap(rank, ctx),
  };
}

/// Build ktdp.construct_indirect_access_tile for gather/scatter.
///
/// Mirrors :func:`buildDirectAccessTile` — caller pre-builds `memView`,
/// helper constructs the access tile and returns it.  Indirect-only
/// deviations:
///   - Takes a separately resolved `indexView` and its descriptor_load
///     anchor `xOffsetIndex` for the indirect dimension, plus a
///     `yOffset` for the direct dimension.  Both `indexView` and
///     `xOffsetIndex` are produced by :func:`resolveIndexView` and are
///     non-null on the success path; this helper asserts that contract.
///
/// `resultShape` is the shape of the gather/scatter result tile (which
/// for gather is also the access-tile shape: dim 0 is fanned out to
/// the index count, and trailing dims mirror `block_shape[1:]` of the
/// source descriptor). The helper supports any rank >= 2.
static Value
buildIndirectAccessTile(OpBuilder &builder, Location loc, Value memView,
                        Value indexView, Value xOffsetIndex,
                        ArrayRef<int64_t> resultShape, Value yOffset) {
  assert(xOffsetIndex && "x_offset must be present (resolveIndexView "
                         "returns failure() on trace miss)");
  assert(resultShape.size() >= 2 &&
         "indirect access tile requires rank >= 2");
  MLIRContext *ctx = builder.getContext();
  auto indexType = builder.getIndexType();

  auto accessTileType = mlir::ktdp::AccessTileType::get(resultShape, indexType);
  auto sub = buildGatherSubscriptMaps(ctx, resultShape);

  // Cast yOffset (i32 from Triton) to index, mirroring how direct casts
  // its `indices` operands.
  Value yOffsetIndex =
      arith::IndexCastOp::create(builder, loc, indexType, yOffset);

  // Captured-variable order must match the dim ordering used by
  // buildGatherSubscriptMaps: x_offset first (c_x), then y_offset (c_y).
  SmallVector<Value> capturedVars{xOffsetIndex, yOffsetIndex};

  auto indirectTile = mlir::ktdp::ConstructIndirectAccessTilesOp::create(
      builder, loc, accessTileType,
      /*base=*/memView,
      sub.subscriptKinds, sub.subscriptMaps,
      /*indirect_memrefs=*/ValueRange{indexView},
      /*captured_variables=*/ValueRange{capturedVars},
      /*symbol_operands=*/ValueRange{},
      sub.spaceSet, sub.spaceOrder);

  return indirectTile.getResult();
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertDescriptorLoad
    : public OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value memView = getDescriptorMemView(adaptor.getDesc());
    // Block shape comes from the descriptor's type, not the result tensor:
    // a rank-reduced load (e.g. desc <1x16x16xf32> -> result tensor<16x16xf32>)
    // would otherwise build a 2D access tile that doesn't match the 3D
    // memory view, producing IR that passes verification here but is wrong.
    auto descType = cast<triton::TensorDescType>(op.getDesc().getType());
    ArrayRef<int64_t> blockShape = descType.getBlockType().getShape();
    Value accessTile = buildDirectAccessTile(
        rewriter, loc, memView, blockShape, op.getIndices());

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto loadResult = mlir::ktdp::LoadOp::create(
        rewriter, loc, resultType, accessTile);

    rewriter.replaceOp(op, loadResult.getResult());
    return success();
  }
};

struct ConvertDescriptorStore
    : public OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value memView = getDescriptorMemView(adaptor.getDesc());
    // See ConvertDescriptorLoad: use the descriptor's block shape so a
    // rank-reduced store still fails verification rather than silently
    // producing a tile/view rank mismatch.
    auto descType = cast<triton::TensorDescType>(op.getDesc().getType());
    ArrayRef<int64_t> blockShape = descType.getBlockType().getShape();
    Value accessTile = buildDirectAccessTile(
        rewriter, loc, memView, blockShape, op.getIndices());

    mlir::ktdp::StoreOp::create(rewriter, loc, op.getSrc(), accessTile);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertDescriptorGather
    : public OpConversionPattern<triton::DescriptorGatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value memView = getDescriptorMemView(adaptor.getDesc());
    // Use adaptor to get the remapped x_offsets (post-conversion value from
    // the already-lowered descriptor_load, i.e. the ktdp.load result).
    auto indexRes = resolveIndexView(adaptor.getXOffsets());
    if (failed(indexRes))
      return failure();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value accessTile = buildIndirectAccessTile(
        rewriter, loc, memView, indexRes->view, indexRes->offset,
        resultType.getShape(), op.getYOffset());

    auto loadResult = mlir::ktdp::LoadOp::create(
        rewriter, loc, resultType, accessTile);

    rewriter.replaceOp(op, loadResult.getResult());
    return success();
  }
};

struct ConvertDescriptorScatter
    : public OpConversionPattern<triton::DescriptorScatterOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value memView = getDescriptorMemView(adaptor.getDesc());
    // Use adaptor to get the remapped x_offsets (post-conversion value).
    auto indexRes = resolveIndexView(adaptor.getXOffsets());
    if (failed(indexRes))
      return failure();

    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    Value accessTile = buildIndirectAccessTile(
        rewriter, loc, memView, indexRes->view, indexRes->offset,
        srcType.getShape(), op.getYOffset());

    mlir::ktdp::StoreOp::create(rewriter, loc, op.getSrc(), accessTile);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerDescriptorMemoryPass
    : public mlir::triton::ktdp::impl::LowerDescriptorMemoryBase<
          LowerDescriptorMemoryPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // ---- Walk 1: tt.make_tensor_descriptor -> ktdp.construct_memory_view.
    // Walk the entire module (including nested regions) so every
    // descriptor is rewritten in place at its original site:
    //   * Descriptor at function top  -> the view sits at function top
    //     too, so it is built once and reused across every loop
    //     iteration — never rebuilt per-iteration.
    //   * Descriptor inside scf.for / scf.if -> view is built once per
    //     loop iteration / once per branch entry, matching the
    //     descriptor's own visibility. This pass does not analyse
    //     whether it would be safe to lift the view out of the loop;
    //     it simply preserves the user's placement.
    //
    // Collect first, rewrite second: erasing during a `module.walk`
    // would invalidate the walker's cursor when the erased op contains
    // nested regions or has been visited mid-traversal.
    //
    // After this walk, every descriptor's uses are routed to a memref
    // view of the underlying buffer.  The access-op patterns below
    // pick up that memref via `getDescriptorMemView` and lower into
    // ktdp.construct_access_tile + ktdp.load/store.
    OpBuilder builder(ctx);
    SmallVector<triton::MakeTensorDescOp> descOps;
    module.walk([&](triton::MakeTensorDescOp op) { descOps.push_back(op); });
    for (auto descOp : descOps) {
      if (descOp.getResult().use_empty()) {
        descOp.erase();
        continue;
      }

      builder.setInsertionPoint(descOp);
      Type elemType =
          cast<triton::TensorDescType>(descOp.getResult().getType())
              .getBlockType()
              .getElementType();
      Value memView =
          buildBaseMemoryView(builder, descOp.getLoc(), descOp, elemType);
      Value casted = UnrealizedConversionCastOp::create(
                         builder, descOp.getLoc(),
                         descOp.getResult().getType(), memView)
                         .getResult(0);
      descOp.getResult().replaceAllUsesWith(casted);
      descOp.erase();
    }

    // ---- Precondition check: every remaining access op's `desc` operand
    // must be a memref-backed descriptor produced by walk 1.  The
    // remaining failure mode is a descriptor sourced from a function
    // argument (or any other producer that isn't
    // `tt.make_tensor_descriptor`); walk 1 doesn't match those, so the
    // access op is left with a raw `!tt.tensordesc` operand whose
    // shape/stride info we have no way to recover.
    auto preCheck = module.walk([&](Operation *op) -> WalkResult {
      if (!isa<triton::DescriptorLoadOp, triton::DescriptorStoreOp,
               triton::DescriptorGatherOp, triton::DescriptorScatterOp>(op))
        return WalkResult::advance();
      if (!isLoweredDescriptor(op->getOperand(0)))
        return op->emitError(
            "cannot lower descriptor op: shape and stride info is only "
            "available when the descriptor is defined by "
            "tt.make_tensor_descriptor in the same block");
      return WalkResult::advance();
    });
    if (preCheck.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(*ctx);
    // Illegal: Triton descriptor ops that have conversion patterns below.
    //   Direct path: descriptor_load, descriptor_store
    //   Indirect path: descriptor_gather, descriptor_scatter
    target.addIllegalOp<triton::DescriptorLoadOp, triton::DescriptorStoreOp,
                        triton::DescriptorGatherOp, triton::DescriptorScatterOp>();
    // Legal: output dialects that conversion patterns lower into.
    //   ktdp (construct_memory_view, construct_access_tile, load, store),
    //   arith (constants, index casts), memref (alloc for index buffers)
    target.addLegalDialect<mlir::ktdp::KtdpDialect, arith::ArithDialect,
                           memref::MemRefDialect>();
    // `UnrealizedConversionCastOp` is MLIR's built-in placeholder cast,
    // used when a pass needs to attach a value of one type to a use site
    // that still expects a different type.  Two places in this pass
    // create one, and both need it to remain legal during this partial
    // conversion:
    //
    //   1. Walk 1 above wraps each new memref in a placeholder cast so
    //      the descriptor's existing `!tt.tensordesc`-typed uses keep
    //      verifying.  The access-op patterns then reach *through* the
    //      cast (via `getDescriptorMemView`) and consume the memref
    //      directly.  After all access ops are rewritten, the
    //      descriptor side of the cast has no real consumers and is
    //      cleaned up by canonicalize/DCE in the next pipeline stage.
    //
    //   2. `getBasePtrAsIndex` uses one to convert a `!tt.ptr` base
    //      pointer to `index`.  This cast survives this pass and is
    //      consumed by the `ConvertFunctions` pass later in the
    //      pipeline, which rewrites `!tt.ptr` function arguments to
    //      `index` and erases the matching casts.
    //
    // Marking it legal here prevents `applyPartialConversion` from
    // treating either cast as an unconverted op and failing the pass.
    // --- added for spyre: tt.spyre_tensor_layout is an annotation marker that
    // must survive (legal, pass-through) until the v2 RewriteDescriptorLayout
    // pass consumes it after LowerComputeOps. Its desc operand auto-re-points at
    // the UnrealizedConversionCast built in walk 1, so it stays valid.
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp,
                      triton::SpyreTensorLayoutOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertDescriptorLoad, ConvertDescriptorStore,
                 ConvertDescriptorGather, ConvertDescriptorScatter>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError("LowerDescriptorMemory: failed to convert descriptor ops");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createLowerDescriptorMemoryPass() {
  return std::make_unique<LowerDescriptorMemoryPass>();
}
} // namespace mlir::triton::ktdp
