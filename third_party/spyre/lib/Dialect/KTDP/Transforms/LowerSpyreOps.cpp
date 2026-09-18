//===- LowerSpyreOps.cpp - Lower scalar math ops to spyreop intrinsics ---===//
//
// Lowers scalar math/arith dialect ops to spyreop dialect intrinsics.
// spyreop's intrinsics are scalar-only (f16/df16/f32, or i32/i64 for the
// integer ops below), so this pass only matches an op that is already
// scalar -- typically the body of a linalg.generic after
// ConvertElementwiseToLinalg has scalarized a tensor-level op.
//
// math.sqrt/exp/rsqrt and arith.divf are matched unconditionally on scalar
// type: those ops only ever appear in real floating-point compute, never in
// address/index arithmetic, so every scalar occurrence is expected to be
// convertible and an unsupported type (e.g. f64) is reported as illegal
// rather than left alone.
//
// arith.divf has two targets rather than one: `1.0 / x` becomes the unary
// spyreop.reciprocal and everything else the binary spyreop.realdiv. That is
// not an optimization -- the pattern's own comment has why an immediate the
// device never sees is the only reliable one.
//
// arith.addi/arith.muli are different: plain scalar integer add/mul is used
// throughout a kernel for loop indices, offsets, and tile addressing, not
// just scalarized tensor compute. Converting every scalar occurrence would
// also rewrite that index arithmetic. So the integer patterns below only
// match inside a linalg.generic body (the structural signal that this is
// scalarized elementwise compute, not address math), and only for the
// specific bit-widths spyreop has an intrinsic for -- anything else (other
// widths, or outside a linalg.generic) is left legal rather than reported.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_LOWERSPYREOPS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// Whether spyreop's scalar float intrinsics accept this operand type.
static bool isSpyreOpScalarType(Type type) {
  return isa<Float16Type, Float32Type>(type);
}

/// Whether this op is (transitively) inside a linalg.generic body -- the
/// structural signal that it is scalarized elementwise compute rather than
/// address/index arithmetic.
static bool isInsideLinalgGeneric(Operation *op) {
  return op->getParentOfType<linalg::GenericOp>() != nullptr;
}

/// The bit width of `type` if it's a scalar integer, or 0 otherwise (e.g. for
/// a not-yet-scalarized tensor/vector of integers).
static unsigned getScalarIntBitWidth(Type type) {
  auto intTy = dyn_cast<IntegerType>(type);
  return intTy ? intTy.getWidth() : 0;
}

//===----------------------------------------------------------------------===//
// math.sqrt -> spyreop.sqrt
//===----------------------------------------------------------------------===//

struct ConvertMathSqrt : public OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::Sqrt>(op, op.getType(),
                                               adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// math.exp -> spyreop.exp
//===----------------------------------------------------------------------===//

struct ConvertMathExp : public OpConversionPattern<math::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::Exp>(op, op.getType(),
                                              adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// math.rsqrt -> spyreop.rsqrt
//===----------------------------------------------------------------------===//

struct ConvertMathRsqrt : public OpConversionPattern<math::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::RSqrt>(op, op.getType(),
                                                adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// arith.divf -> spyreop.realdiv, or spyreop.reciprocal when the numerator is 1
//===----------------------------------------------------------------------===//

struct ConvertArithDivF : public OpConversionPattern<arith::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    // `1.0 / x` becomes the UNARY intrinsic, so no float immediate reaches the
    // device at all. That is the whole point of the special case: an fp16
    // constant is emitted as its IEEE binary16 bit pattern and read by the
    // device as Spyre's 1-6-9 float (bias 31, not 15), so 0x3C00 -- IEEE 1.0 --
    // arrives as 0.5. The reinterpretation is not ours to fix, but a reciprocal
    // does not have to depend on it: spyreop.reciprocal carries no operand to
    // misread. Nothing here compensates for the mangling for any other value,
    // and nothing should -- see the softmax_on_stick banner in
    // fixtures/reduce/meta.py for why there is no table of pre-scaled
    // immediates to exploit.
    //
    // Matched through m_OneFloat -- the standard MLIR value matcher -- rather
    // than by poking at the constant's attribute, so the shape of the numerator
    // is not assumed: it accepts a scalar float constant (softmax's, a
    // `arith.constant 1.0 : f16` hoisted above the linalg.generic that uses it)
    // and equally a splat, without either being spelled here.
    if (matchPattern(adaptor.getLhs(), m_OneFloat())) {
      // The numerator's own op is erased with it when the divide was its only
      // reader, so nothing dead is left for a later stage to have to decide
      // about. Guarded rather than unconditional because two reciprocals may
      // have been CSE'd onto one constant; then it has another reader and
      // outliving this rewrite is correct.
      Operation *numerator = adaptor.getLhs().getDefiningOp();
      bool sole = numerator && adaptor.getLhs().hasOneUse();
      rewriter.replaceOpWithNewOp<spyreop::Reciprocal>(op, op.getType(),
                                                       adaptor.getRhs());
      if (sole)
        rewriter.eraseOp(numerator);
      return success();
    }
    rewriter.replaceOpWithNewOp<spyreop::RealDiv>(
        op, op.getType(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// arith.addi (i32/i64, inside a linalg.generic) -> spyreop.addi32toi32 /
// spyreop.addi64toi64
//===----------------------------------------------------------------------===//

struct ConvertArithAddI : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isInsideLinalgGeneric(op))
      return failure();
    unsigned width = getScalarIntBitWidth(op.getType());
    if (width == 32)
      rewriter.replaceOpWithNewOp<spyreop::AddI32ToI32>(
          op, op.getType(), adaptor.getLhs(), adaptor.getRhs());
    else if (width == 64)
      rewriter.replaceOpWithNewOp<spyreop::AddI64ToI64>(
          op, op.getType(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// arith.muli (i32, inside a linalg.generic) -> spyreop.muli32toi32
//===----------------------------------------------------------------------===//

struct ConvertArithMulI : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isInsideLinalgGeneric(op) || getScalarIntBitWidth(op.getType()) != 32)
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::MulI32ToI32>(
        op, op.getType(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerSpyreOpsPass
    : public mlir::triton::ktdp::impl::LowerSpyreOpsBase<LowerSpyreOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    // A math op still on a tensor/vector hasn't been scalarized yet (that's
    // ConvertElementwiseToLinalg's job) -- leave it legal, quietly, rather
    // than reporting it. Any scalar type is illegal here: the pattern
    // converts the ones spyreop supports and leaves the rest illegal so
    // conversion reports them instead of silently dropping them.
    target.addDynamicallyLegalOp<math::SqrtOp>([](math::SqrtOp op) {
      return isa<ShapedType>(op.getType());
    });
    target.addDynamicallyLegalOp<math::ExpOp>([](math::ExpOp op) {
      return isa<ShapedType>(op.getType());
    });
    target.addDynamicallyLegalOp<math::RsqrtOp>([](math::RsqrtOp op) {
      return isa<ShapedType>(op.getType());
    });
    target.addDynamicallyLegalOp<arith::DivFOp>([](arith::DivFOp op) {
      return isa<ShapedType>(op.getType());
    });
    // arith.addi/muli are also used for plain index/address arithmetic, so
    // (unlike the ops above) they stay legal everywhere except the one
    // context and bit-width this pass actually converts: scalarized
    // elementwise compute inside a linalg.generic, at a width spyreop has an
    // intrinsic for. Everything else -- other widths, or outside a
    // linalg.generic -- is left alone rather than reported.
    target.addDynamicallyLegalOp<arith::AddIOp>([](arith::AddIOp op) {
      unsigned width = getScalarIntBitWidth(op.getType());
      return !isInsideLinalgGeneric(op) || (width != 32 && width != 64);
    });
    target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp op) {
      return !isInsideLinalgGeneric(op) ||
             getScalarIntBitWidth(op.getType()) != 32;
    });
    target.addLegalDialect<spyreop::SpyreOpDialect>();
    target.addLegalOp<ModuleOp>();

    // No TypeConverter is installed, so adaptor operands are the original
    // ones, for every pattern below.
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMathSqrt, ConvertMathExp, ConvertMathRsqrt,
                 ConvertArithDivF, ConvertArithAddI, ConvertArithMulI>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError("LowerSpyreOps: failed to convert math ops");
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass() {
  return std::make_unique<LowerSpyreOpsPass>();
}
} // namespace mlir::triton::ktdp
