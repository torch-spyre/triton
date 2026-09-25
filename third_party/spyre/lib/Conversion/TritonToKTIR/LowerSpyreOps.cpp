//===- LowerSpyreOps.cpp - Lower elementwise math ops to spyreop ---------===//
//
// Lowers math/arith dialect ops to spyreop dialect intrinsics, at TENSOR level.
//
// Position: this pass runs BEFORE ConvertElementwiseToLinalg, so the ops it sees
// are still tensor-typed -- `math.sqrt` on `tensor<128xf32>`, not `f32`. That is
// the opposite of where this pass used to sit, and the reason for the move is in
// Pipeline.cpp, where the move is justified: a mask cast can only be absorbed
// into its comparison while the two are adjacent ops in one block, and after
// scalarization they are in separate linalg.generic bodies with the producer a
// block argument that has no defining op. No pattern here absorbs one yet; the
// position is what makes adding one possible.
//
// spyreop's elementwise intrinsics are SCALAR-ONLY by declaration -- their
// operands are AnyTypeOf<[F16, SpyreOp_DF16, F32]>, never a shaped type, so a
// tensor-level spyreop op does not verify. Each pattern therefore emits a
// `linalg.generic` whose body holds the scalar intrinsic: tensor in, tensor out,
// scalar op inside. `wrapElementwise` below builds that, and every pattern goes
// through it so the shape is written once.
//
// A generic built here is NOT re-wrapped by ConvertElementwiseToLinalg
// afterwards: that pass keys on the `ElementwiseMappable` trait, which a
// linalg.generic does not carry.
//
// What each op matches on:
//
//   math.sqrt/exp/rsqrt, arith.divf -- any tensor of a width spyreop supports.
//     These appear only in real floating-point compute, never in address
//     arithmetic, so every such tensor occurrence is expected to be convertible
//     and an unsupported element type (e.g. f64) is reported rather than ignored.
//
//   arith.divf has two targets rather than one: a numerator of constant 1
//     becomes the unary spyreop.reciprocal and everything else the binary
//     spyreop.realdiv.
//
//   arith.addi/arith.muli -- a TENSOR of i32/i64 only. Plain scalar integer
//     add/mul is used throughout a kernel for loop indices, offsets and tile
//     addressing; converting those would rewrite address arithmetic into compute
//     intrinsics. Being tensor-typed is what separates the two, and it separates
//     them exactly: address math is never a tensor. (This replaces an earlier
//     `isInsideLinalgGeneric` test that asked the same question by IR position
//     rather than by type, and was only meaningful after scalarization.)
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToKTIR/Passes.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERSPYREOPS
#include "Conversion/TritonToKTIR/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

/// Whether spyreop's float intrinsics accept this element type.
static bool isSpyreOpFloatElemType(Type elemType) {
  return isa<Float16Type, Float32Type>(elemType);
}

/// The ranked tensor `type` is, or null if it is not one.
///
/// Every pattern here starts by asking this. A null answer means the op is not
/// tensor-typed, which at this point in the pipeline means it is scalar
/// address/index arithmetic rather than elementwise compute -- so the pattern
/// declines and the op is left alone.
static RankedTensorType asRankedTensor(Type type) {
  return dyn_cast<RankedTensorType>(type);
}

/// The element type of a ranked tensor, or null for anything else.
static Type tensorElemType(Type type) {
  auto ty = asRankedTensor(type);
  return ty ? ty.getElementType() : nullptr;
}

/// The bit width of `type` if it is a TENSOR of integers, or 0 otherwise.
///
/// Zero for a scalar integer as well as for a non-integer, which is what the
/// integer patterns want: a scalar `arith.muli` here is a tile offset, not
/// compute.
static unsigned tensorIntBitWidth(Type type) {
  auto intTy = dyn_cast_or_null<IntegerType>(tensorElemType(type));
  return intTy ? intTy.getWidth() : 0;
}

/// Wrap a scalar spyreop intrinsic in a `linalg.generic` over `operands`.
///
/// This is the one place a generic is built, so the shape is stated once rather
/// than nine times. Everything is DERIVED from the operands -- rank, indexing
/// maps, iterator types, shape and destination -- because a literal rank or a
/// parsed `affine_map` string here would break on the first 2D or f16 kernel.
///
/// `resultElemType` is separate from the operands' element type because a
/// comparison's result width need not match what it compared in general; for
/// every op here they happen to coincide, and passing it explicitly keeps that a
/// statement rather than an assumption.
///
/// `body` receives the scalar block arguments for the inputs only -- the
/// destination's block argument is dropped, since an elementwise body never
/// reads its own uninitialized destination -- and returns the scalar result,
/// which this function yields.
static Value wrapElementwise(
    OpBuilder &b, Location loc, ValueRange operands, Type resultElemType,
    function_ref<Value(OpBuilder &, Location, ValueRange)> body) {
  auto operandTy = cast<RankedTensorType>(operands.front().getType());
  int64_t rank = operandTy.getRank();

  auto resultTy = RankedTensorType::get(operandTy.getShape(), resultElemType);
  Value dest = tensor::EmptyOp::create(b, loc, operandTy.getShape(),
                                       resultElemType);

  // One identity map per operand plus one for the destination. Identity is
  // correct because every op here is elementwise: iteration (d0..dn) reads
  // element (d0..dn) of each operand.
  auto identity = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
  SmallVector<AffineMap> maps(operands.size() + 1, identity);
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);

  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{resultTy}, operands, ValueRange{dest}, maps, iterators,
      /*doc=*/"", /*libraryCall=*/"",
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value scalar = body(nested, nestedLoc, args.drop_back());
        linalg::YieldOp::create(nested, nestedLoc, scalar);
      });
  return generic.getResult(0);
}

//===----------------------------------------------------------------------===//
// math.sqrt -> spyreop.sqrt
//===----------------------------------------------------------------------===//

struct ConvertMathSqrt : public OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elemTy = tensorElemType(op.getType());
    if (!elemTy || !isSpyreOpFloatElemType(elemTy))
      return failure();
    // Hoisted into a local rather than nested inside replaceOp: C++ leaves the
    // evaluation order of call arguments unspecified, and wrapElementwise has
    // the side effect of inserting ops.
    Value wrapped = wrapElementwise(
        rewriter, op.getLoc(), {adaptor.getOperand()}, elemTy,
        [](OpBuilder &b, Location loc, ValueRange args) {
          return spyreop::Sqrt::create(b, loc, args[0].getType(), args[0]);
        });
    rewriter.replaceOp(op, wrapped);
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
    Type elemTy = tensorElemType(op.getType());
    if (!elemTy || !isSpyreOpFloatElemType(elemTy))
      return failure();
    // Hoisted into a local rather than nested inside replaceOp: C++ leaves the
    // evaluation order of call arguments unspecified, and wrapElementwise has
    // the side effect of inserting ops.
    Value wrapped = wrapElementwise(
        rewriter, op.getLoc(), {adaptor.getOperand()}, elemTy,
        [](OpBuilder &b, Location loc, ValueRange args) {
          return spyreop::Exp::create(b, loc, args[0].getType(), args[0]);
        });
    rewriter.replaceOp(op, wrapped);
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
    Type elemTy = tensorElemType(op.getType());
    if (!elemTy || !isSpyreOpFloatElemType(elemTy))
      return failure();
    // Hoisted into a local rather than nested inside replaceOp: C++ leaves the
    // evaluation order of call arguments unspecified, and wrapElementwise has
    // the side effect of inserting ops.
    Value wrapped = wrapElementwise(
        rewriter, op.getLoc(), {adaptor.getOperand()}, elemTy,
        [](OpBuilder &b, Location loc, ValueRange args) {
          return spyreop::RSqrt::create(b, loc, args[0].getType(), args[0]);
        });
    rewriter.replaceOp(op, wrapped);
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
    Type elemTy = tensorElemType(op.getType());
    if (!elemTy || !isSpyreOpFloatElemType(elemTy))
      return failure();

    // A numerator of one becomes the UNARY intrinsic, so no float immediate
    // reaches the device at all. Matched through m_OneFloat, which accepts a
    // scalar float constant or a splat -- at tensor level it is always the
    // splat form, which is why no shape is assumed here.
    if (matchPattern(adaptor.getLhs(), m_OneFloat())) {
      // The numerator's own op goes with it when the divide was its only
      // reader. Guarded, because a CSE'd splat may have another.
      Operation *numerator = adaptor.getLhs().getDefiningOp();
      bool sole = numerator && adaptor.getLhs().hasOneUse();
      // Hoisted into a local rather than nested inside replaceOp: C++ leaves the
      // evaluation order of call arguments unspecified, and wrapElementwise has
      // the side effect of inserting ops.
      Value wrapped = wrapElementwise(
          rewriter, op.getLoc(), {adaptor.getRhs()}, elemTy,
          [](OpBuilder &b, Location loc, ValueRange args) {
            return spyreop::Reciprocal::create(b, loc, args[0].getType(),
                                               args[0]);
          });
      rewriter.replaceOp(op, wrapped);
      if (sole)
        rewriter.eraseOp(numerator);
      return success();
    }

    // Hoisted into a local rather than nested inside replaceOp: C++ leaves the
    // evaluation order of call arguments unspecified, and wrapElementwise has
    // the side effect of inserting ops.
    Value wrapped = wrapElementwise(
        rewriter, op.getLoc(), {adaptor.getLhs(), adaptor.getRhs()}, elemTy,
        [](OpBuilder &b, Location loc, ValueRange args) {
          return spyreop::RealDiv::create(b, loc, args[0].getType(), args[0],
                                          args[1]);
        });
    rewriter.replaceOp(op, wrapped);
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
    // Tensor-typed only: a scalar integer add here is a loop index or a tile
    // offset, and rewriting it would turn address arithmetic into a compute
    // intrinsic. tensorIntBitWidth returns 0 for a scalar, so this declines.
    unsigned width = tensorIntBitWidth(op.getType());
    if (width != 32 && width != 64)
      return failure();
    Type elemTy = tensorElemType(op.getType());
    // Hoisted into a local rather than nested inside replaceOp: C++ leaves the
    // evaluation order of call arguments unspecified, and wrapElementwise has
    // the side effect of inserting ops.
    Value wrapped = wrapElementwise(
        rewriter, op.getLoc(), {adaptor.getLhs(), adaptor.getRhs()}, elemTy,
        [width](OpBuilder &b, Location loc, ValueRange args) -> Value {
          if (width == 32)
            return spyreop::AddI32ToI32::create(b, loc, args[0].getType(),
                                                args[0], args[1]);
          return spyreop::AddI64ToI64::create(b, loc, args[0].getType(),
                                              args[0], args[1]);
        });
    rewriter.replaceOp(op, wrapped);
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
    // Tensor-typed i32 only -- see ConvertArithAddI on why a scalar declines.
    if (tensorIntBitWidth(op.getType()) != 32)
      return failure();
    Type elemTy = tensorElemType(op.getType());
    // Hoisted into a local rather than nested inside replaceOp: C++ leaves the
    // evaluation order of call arguments unspecified, and wrapElementwise has
    // the side effect of inserting ops.
    Value wrapped = wrapElementwise(
        rewriter, op.getLoc(), {adaptor.getLhs(), adaptor.getRhs()}, elemTy,
        [](OpBuilder &b, Location loc, ValueRange args) {
          return spyreop::MulI32ToI32::create(b, loc, args[0].getType(),
                                              args[0], args[1]);
        });
    rewriter.replaceOp(op, wrapped);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerSpyreOpsPass
    : public mlir::triton::spyre::impl::LowerSpyreOpsBase<LowerSpyreOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    // These callbacks say what is allowed to REMAIN. They are the mirror of the
    // patterns above: a pattern says what to convert, and the callback says
    // whether an unconverted leftover is an error or is fine.
    //
    // For the float ops, a TENSOR is now illegal -- that is the thing this pass
    // exists to convert. A scalar is legal and left alone: at this point in the
    // pipeline a scalar float op is not elementwise compute (nothing has
    // scalarized yet), so it is not ours. An unsupported element type such as
    // f64 on a tensor stays illegal, so conversion reports it rather than
    // silently shipping it to the device.
    //
    // Note this is the opposite of what these callbacks said when the pass ran
    // after scalarization, where `isa<ShapedType>` meant "a tensor has not been
    // scalarized yet, leave it". Keeping that here would make the pass a silent
    // no-op: the patterns would decline AND the leftovers would be declared
    // fine, so applyPartialConversion would report success having converted
    // nothing.
    target.addDynamicallyLegalOp<math::SqrtOp>([](math::SqrtOp op) {
      return !asRankedTensor(op.getType());
    });
    target.addDynamicallyLegalOp<math::ExpOp>([](math::ExpOp op) {
      return !asRankedTensor(op.getType());
    });
    target.addDynamicallyLegalOp<math::RsqrtOp>([](math::RsqrtOp op) {
      return !asRankedTensor(op.getType());
    });
    target.addDynamicallyLegalOp<arith::DivFOp>([](arith::DivFOp op) {
      return !asRankedTensor(op.getType());
    });
    // arith.addi/muli are also used for plain index/address arithmetic, so
    // (unlike the ops above) they stay legal everywhere except the one shape and
    // bit-width this pass converts: a TENSOR of i32/i64. A scalar add or
    // multiply is a loop index or a tile offset and is left alone rather than
    // reported -- tensorIntBitWidth returns 0 for one, so both callbacks below
    // call it legal.
    target.addDynamicallyLegalOp<arith::AddIOp>([](arith::AddIOp op) {
      unsigned width = tensorIntBitWidth(op.getType());
      return width != 32 && width != 64;
    });
    target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp op) {
      return tensorIntBitWidth(op.getType()) != 32;
    });

    target.addLegalDialect<spyreop::SpyreOpDialect>();
    // Created by the patterns above, so they must be legal or the conversion
    // driver rolls the pattern back and then reports the original op as
    // unlegalizable -- which reads as "the pattern never fired".
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
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

namespace mlir::triton::spyre {
std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass() {
  return std::make_unique<LowerSpyreOpsPass>();
}
} // namespace mlir::triton::spyre
