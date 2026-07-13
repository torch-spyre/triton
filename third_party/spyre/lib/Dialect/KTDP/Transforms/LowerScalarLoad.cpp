//===- LowerScalarLoad.cpp - Lower scalar tt.load to ktdp ops ------------===//
//
// Lowers a *scalar* `tt.load` (pointer operand `!tt.ptr<ElemT>`, scalar
// result `ElemT`) to the minimal legal KTDP read:
//   tt.load %ptr [, %mask [, %other]]
//     -> ktdp.construct_memory_view (rank 0)
//     -> ktdp.construct_access_tile (rank 0)
//     -> ktdp.load                  (-> tensor<ElemT>)
//     -> tensor.extract             (-> ElemT)
//
// Spyre has no user-programmable control-flow divergence therefore
// a data-dependent branch per lane is not expressible.
// A `tt.load` mask is therefore only lowered
// when it is a compile-time constant (a materialized `arith.constant` i1):
// constant-true drops straight through to the unconditional read;
// constant-false skips the read entirely and yields `other` (or a
// materialized zero of the element type). Any other mask — anything whose
// value could depend on runtime data (for now including a comparison of
// two constants) is refused with a diagnostic before conversion runs; the
// pass never emits a runtime branch (e.g. `scf.if`) on the mask.
//
// Tensor-of-pointers `tt.load` (pointer operand shaped as a tensor of
// `!tt.ptr<ElemT>`) is out of scope for this pass and remains
// legal/untouched; see `[LowerPointerChainMemory]` (not yet implemented,
// Passes.td pipeline diagram) for that path.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "Ktdp/KtdpAttrs.hpp"
#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "Ktdp/KtdpTypes.hpp"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
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
#define GEN_PASS_DEF_LOWERSCALARLOAD
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// True iff `ptr` is a scalar Triton pointer (`!tt.ptr<ElemT>`), as opposed
/// to a tensor of pointers. Only the scalar form is handled by this pass.
static bool isScalarPtr(Value ptr) {
  return isa<triton::PointerType>(ptr.getType());
}

/// Try to extract a compile-time bool from an SSA value produced by
/// `arith.constant`. Returns std::nullopt if the value is not a
/// materialized constant — in particular, a comparison of two constants
/// (`arith.cmpi`) is deliberately *not* folded here; only a value that is
/// itself `arith.constant` counts, mirroring `LowerDescriptorMemory.cpp`'s
/// `getConstantInt`.
static std::optional<bool> getConstantMask(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt() != 0;
  return std::nullopt;
}

/// Cast a `!tt.ptr` value to `index` via an `unrealized_conversion_cast`.
/// Mirrors `LowerDescriptorMemory.cpp`'s `getBasePtrAsIndex`: the cast
/// survives this pass and is consumed by the later `ConvertFunctions`
/// pass, which rewrites `!tt.ptr` function arguments to `index` and
/// erases the matching casts.
static Value ptrToIndex(OpBuilder &builder, Location loc, Value ptr) {
  if (ptr.getType().isIndex())
    return ptr;
  return UnrealizedConversionCastOp::create(builder, loc,
                                             builder.getIndexType(), ptr)
      .getResult(0);
}

/// Walk a chain of scalar `tt.addptr` ops back to its root pointer,
/// folding every offset into a single `index` value with plain adds — no
/// element-size scaling, matching the descriptor path's element-unit
/// convention (striding is the kernel author's responsibility per the
/// maintainer). The `tt.addptr` ops themselves are left in place; they
/// become dead once the caller replaces the `tt.load` and are swept up by
/// `cleanupDeadOps` at the end of `runOnOperation`.
static Value resolveScalarAddress(OpBuilder &builder, Location loc,
                                   Value ptr) {
  SmallVector<Value> offsets;
  while (auto addPtr = ptr.getDefiningOp<triton::AddPtrOp>()) {
    Value offset = addPtr.getOffset();
    Type indexType = builder.getIndexType();
    if (offset.getType() != indexType)
      offset = arith::IndexCastOp::create(builder, loc, indexType, offset);
    offsets.push_back(offset);
    ptr = addPtr.getPtr();
  }

  Value baseIndex = ptrToIndex(builder, loc, ptr);
  for (Value offset : llvm::reverse(offsets))
    baseIndex =
        arith::AddIOp::create(builder, loc, baseIndex, offset).getResult();
  return baseIndex;
}

/// A rank-0 "trivially true" IntegerSet: 0 dims, 0 symbols, and a single
/// `0 >= 0` constraint. `IntegerSet::get` derives its owning context from
/// `constraints[0]`, so a genuinely empty constraint list (`{}`) is not
/// constructible — it indexes past the end of an empty array and crashes.
/// The single constant constraint is a no-op (always satisfied) and keeps
/// the set a valid handle for the single point of a rank-0 space.
static IntegerSet trivialIntegerSet(MLIRContext *ctx) {
  return IntegerSet::get(/*dimCount=*/0, /*symbolCount=*/0,
                         {getAffineConstantExpr(0, ctx)}, {false});
}

/// Build a rank-0 `ktdp.construct_memory_view` anchored at `baseIndex`,
/// i.e. the memory view of a single scalar element. Mirrors
/// `LowerDescriptorMemory.cpp`'s `buildBaseMemoryView`, specialized to
/// rank 0: no size/stride operands or attrs, and a trivially-true
/// coordinate set (see `trivialIntegerSet`) matching the single point of a
/// rank-0 space.
static Value buildScalarMemoryView(OpBuilder &builder, Location loc,
                                    Value baseIndex, Type elemType) {
  MLIRContext *ctx = builder.getContext();
  auto memrefType = MemRefType::get({}, elemType);
  auto memSpaceAttr = mlir::ktdp::SpyreMemorySpaceAttr::get(
      ctx, mlir::ktdp::SpyreMemorySpaceKind::HBM, /*core=*/-1);
  auto coordinateSet = trivialIntegerSet(ctx);
  auto memView = mlir::ktdp::ConstructMemoryViewOp::create(
      builder, loc, memrefType, baseIndex,
      /*sizes=*/ValueRange{}, /*strides=*/ValueRange{},
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
      memSpaceAttr, IntegerSetAttr::get(coordinateSet));
  return memView.getResult();
}

/// Build a rank-0 `ktdp.construct_access_tile` over `memView`. Mirrors
/// `LowerDescriptorMemory.cpp`'s `buildDirectAccessTile`, specialized to
/// rank 0: no block indices, identity maps over zero dims.
static Value buildScalarAccessTile(OpBuilder &builder, Location loc,
                                    Value memView) {
  MLIRContext *ctx = builder.getContext();
  auto indexType = builder.getIndexType();
  auto accessTileType = mlir::ktdp::AccessTileType::get({}, indexType);
  auto identityMap = AffineMap::getMultiDimIdentityMap(0, ctx);
  auto coordinateSet = trivialIntegerSet(ctx);
  auto accessTile = mlir::ktdp::ConstructAccessTilesOp::create(
      builder, loc, accessTileType, memView, identityMap,
      /*indices=*/ValueRange{}, /*symbol_operands=*/ValueRange{},
      coordinateSet, identityMap);
  return accessTile.getResult();
}

/// Emit the full rank-0 read: memory view -> access tile -> ktdp.load ->
/// tensor.extract, returning the scalar `elemType` value.
static Value emitScalarRead(OpBuilder &builder, Location loc,
                            Value baseIndex, Type elemType) {
  Value memView = buildScalarMemoryView(builder, loc, baseIndex, elemType);
  Value accessTile = buildScalarAccessTile(builder, loc, memView);
  auto tensorType = RankedTensorType::get({}, elemType);
  auto loadResult =
      mlir::ktdp::LoadOp::create(builder, loc, tensorType, accessTile);
  return tensor::ExtractOp::create(builder, loc, loadResult.getResult(),
                                   ValueRange{})
      .getResult();
}

//===----------------------------------------------------------------------===//
// Conversion pattern
//===----------------------------------------------------------------------===//

struct ConvertScalarLoad : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type elemType = op.getResult().getType();

    Value baseIndex = resolveScalarAddress(rewriter, loc, adaptor.getPtr());

    Value mask = adaptor.getMask();
    if (!mask) {
      Value scalar = emitScalarRead(rewriter, loc, baseIndex, elemType);
      rewriter.replaceOp(op, scalar);
      return success();
    }

    // Masked load: the mask must be a compile-time constant (checked by
    // the precheck walk in `runOnOperation` before conversion starts — a
    // non-constant mask never reaches here). Constant-true drops straight
    // through to the unconditional read; constant-false skips the read
    // entirely and yields `other`, or a materialized zero of `elemType`
    // when `other` is absent. No runtime branch is ever emitted on the
    // mask value.
    std::optional<bool> constMask = getConstantMask(mask);
    assert(constMask.has_value() &&
           "non-constant mask should have been rejected by the precheck");

    if (*constMask) {
      Value scalar = emitScalarRead(rewriter, loc, baseIndex, elemType);
      rewriter.replaceOp(op, scalar);
      return success();
    }

    Value fallback = adaptor.getOther();
    if (!fallback)
      fallback = arith::ConstantOp::create(rewriter, loc, elemType,
                                           rewriter.getZeroAttr(elemType));
    rewriter.replaceOp(op, fallback);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerScalarLoadPass
    : public mlir::triton::ktdp::impl::LowerScalarLoadBase<
          LowerScalarLoadPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // ---- Precondition check: a masked scalar `tt.load`'s mask must be a
    // compile-time constant. Spyre has no user-programmable control-flow
    // divergence (see wiki/foundations/hardware/execution-model.md), so a
    // mask whose value could depend on runtime data — including a
    // comparison of two constants, which this deliberately does not fold
    // — cannot be lowered.
    auto preCheck = module.walk([&](triton::LoadOp op) -> WalkResult {
      if (!isScalarPtr(op.getPtr()) || !op.getMask())
        return WalkResult::advance();
      if (!getConstantMask(op.getMask()))
        return op->emitError(
            "cannot lower masked scalar tt.load: mask must be a "
            "compile-time constant on Spyre");
      return WalkResult::advance();
    });
    if (preCheck.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(*ctx);
    // Only a scalar `tt.load` (pointer operand, not tensor-of-pointers)
    // must be converted by `ConvertScalarLoad` below; the tensor form is
    // out of scope for this pass and stays legal/untouched — hence a
    // dynamic legality predicate rather than a blanket `addIllegalOp`.
    target.addDynamicallyLegalOp<triton::LoadOp>(
        [](triton::LoadOp op) { return !isScalarPtr(op.getPtr()); });
    // Legal: output dialects that the conversion pattern lowers into.
    //   ktdp (construct_memory_view, construct_access_tile, load),
    //   arith (index casts, adds, zero constants), tensor (extract)
    target.addLegalDialect<mlir::ktdp::KtdpDialect, arith::ArithDialect,
                           tensor::TensorDialect>();
    // `UnrealizedConversionCastOp` is used by `ptrToIndex` to convert a
    // `!tt.ptr` base pointer to `index`; the cast survives this pass and
    // is consumed by the later `ConvertFunctions` pass, exactly as the
    // descriptor path's `getBasePtrAsIndex` relies on (see
    // `LowerDescriptorMemory.cpp`).
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertScalarLoad>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError("LowerScalarLoad: failed to convert scalar tt.load");
      signalPassFailure();
      return;
    }

    // The address fold in `resolveScalarAddress` leaves any consumed
    // `tt.addptr` chain in place with no remaining uses; sweep exactly
    // those (and nothing else) here rather than relying on the pipeline's
    // later canonicalize/CSE stage. The predicate scopes the sweep by op
    // type, not by provenance: it erases *any* trivially-dead `AddPtrOp`
    // (whether left dead by this fold or already dead in the input
    // kernel), but it will never delete a dead op of any other type — in
    // particular, it will not delete this pass's own freshly emitted read
    // chain in the (rare) case the loaded scalar has no consumer. That
    // chain, and any other type of dead-code fallout, is left for
    // canonicalize/CSE, same as `LowerDescriptorMemory.cpp` already does
    // for its own leftover dead casts.
    mlir::triton::ktdp::cleanupDeadOps(
        module, [](Operation *op) { return isa<triton::AddPtrOp>(op); });
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createLowerScalarLoadPass() {
  return std::make_unique<LowerScalarLoadPass>();
}
} // namespace mlir::triton::ktdp
