//===- LowerScalarLoad.cpp - Lower scalar tt.load to ktdp ops ------------===//
//
// Lowers a *scalar* `tt.load` (pointer operand `!tt.ptr<ElemT>`, scalar
// result `ElemT`) to the minimal legal KTDP read:
//   tt.load %ptr [, %mask [, %other]]
//     -> ktdp.construct_memory_view (memref<1xElemT>)
//     -> ktdp.construct_access_tile (!ktdp.access_tile<1xindex>)
//     -> ktdp.load                  (-> tensor<1xElemT>)
//     -> tensor.extract             (-> ElemT)
// A single-element 1-D vector, not a rank-0 view/tile: rank-0 shaped types
// are not a supported interchange form downstream, while a single-element
// 1-D vector is.
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
// Pass-ordering note: when a scalar `tt.load` result feeds a dynamic shape
// operand of `tt.make_tensor_descriptor`, `LowerDescriptorMemory` reaches
// `kDynamic` for that dimension regardless of which pass runs first:
//   Flow A (`LowerScalarLoad` first) — the `tt.load` is already rewritten
//     to the read chain above; `LowerDescriptorMemory` sees the shape
//     operand's producer is `tensor.extract`, not `arith.constant`, and
//     emits `kDynamic`.
//   Flow B (`LowerDescriptorMemory` first — this repo's actual pipeline
//     order) — `LowerDescriptorMemory` sees the producer is the raw,
//     unconverted `tt.load`, also not `arith.constant`, and emits
//     `kDynamic` directly; this pass then lowers the `tt.load` as normal.
// Both flows agree today only because `getConstantInt` folds nothing but a
// materialized `arith.constant` — it never traces through either the read
// chain above or a raw `tt.load`. `test_dynamic_shape_from_scalar_load`
// (`test_lower_desc_memory.py`) pins Flow B; if `getConstantInt` is later
// extended to trace through one of these chains but not the other, that
// test will catch the resulting order dependency between the two passes.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

/// True iff `ptr` is a scalar Triton pointer (`!tt.ptr<ElemT>`) whose
/// pointee is an integer or float type, as opposed to a tensor of pointers
/// or a pointer-to-pointer (`!tt.ptr<!tt.ptr<ElemT>>` — valid Triton IR,
/// since `PointerType::verify` only rejects tensor pointees, not pointer
/// pointees). Only integer/float pointees are lowerable by this pass:
/// `emitScalarRead`'s memory view requires a `MemRefElementTypeInterface`
/// element type, and the constant-false-without-`other` fallback requires
/// `getZeroAttr` to have a case for the element type — neither holds for
/// `PointerType`.
static bool isScalarPtr(Value ptr) {
  auto ptrTy = dyn_cast<triton::PointerType>(ptr.getType());
  if (!ptrTy)
    return false;
  Type pointee = ptrTy.getPointeeType();
  return isa<IntegerType, FloatType>(pointee);
}

/// Try to extract a compile-time bool from an SSA value produced by
/// `arith.constant`. Returns std::nullopt if the value is not a
/// materialized constant — in particular, a comparison of two constants
/// (`arith.cmpi`) is deliberately *not* folded here; only a value that is
/// itself `arith.constant` counts. Thin bool-projecting wrapper around the
/// shared `getConstantInt`.
static std::optional<bool> getConstantMask(Value v) {
  if (auto c = mlir::triton::ktdp::getConstantInt(v))
    return *c != 0;
  return std::nullopt;
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

  Value baseIndex = mlir::triton::ktdp::getBasePtrAsIndex(builder, loc, ptr);
  for (Value offset : llvm::reverse(offsets))
    baseIndex =
        arith::AddIOp::create(builder, loc, baseIndex, offset).getResult();
  return baseIndex;
}

/// Emit the full single-element 1-D read: memory view -> access tile ->
/// ktdp.load -> tensor.extract, returning the scalar `elemType` value.
/// Built from the shared `buildMemoryView`/`buildAccessTile` helpers (also
/// used by `LowerDescriptorMemory.cpp` and `RewriteDescriptorLayout.cpp`),
/// with a single dim of extent 1 rather than rank 0 — rank-0 shaped types
/// are not a supported interchange form downstream, while a single-element
/// 1-D vector is. The one `arith.constant 0 : index` serves double duty, as
/// both the tile's sole anchor index and the extract index.
static Value emitScalarRead(OpBuilder &builder, Location loc,
                            Value baseIndex, Type elemType) {
  auto memSpaceAttr = mlir::ktdp::MemorySpaceAttr::get(
      builder.getContext(), mlir::ktdp::MemorySpaceKind::global,
      /*ct_id=*/-1);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value memView = mlir::triton::ktdp::buildMemoryView(
      builder, loc, baseIndex, /*staticSizes=*/{1}, /*staticStrides=*/{1},
      /*dynSizes=*/{}, /*dynStrides=*/{}, elemType, memSpaceAttr);
  Value accessTile = mlir::triton::ktdp::buildAccessTile(
      builder, loc, memView, /*blockShape=*/{1}, ValueRange{zero});
  auto tensorType = RankedTensorType::get({1}, elemType);
  auto loadResult =
      mlir::ktdp::LoadOp::create(builder, loc, tensorType, accessTile);
  return tensor::ExtractOp::create(builder, loc, loadResult.getResult(),
                                   ValueRange{zero})
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
    if (!constMask)
      return rewriter.notifyMatchFailure(
          op, "masked scalar tt.load has a non-constant mask — "
              "Spyre has no runtime control-flow divergence");

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
    // divergence, so a mask whose value could depend on runtime data —
    // including a comparison of two constants, which this deliberately does
    // not fold — cannot be lowered.
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
    // `UnrealizedConversionCastOp` is used by `getBasePtrAsIndex` to convert
    // a `!tt.ptr` base pointer to `index`; the cast survives this pass and
    // is consumed by the later `ConvertFunctions` pass.
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp,
                      triton::SpyreTensorLayoutOp>();

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
