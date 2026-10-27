//===- GenericOperands.cpp - Checks and rewrites on linalg.generic operands ===//
//
// See GenericOperands.h.
//
//===----------------------------------------------------------------------===//

#include "Utils/GenericOperands.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::triton::spyre {

LogicalResult checkOperandRewrite(ArrayRef<std::pair<Value, AffineMap>> ins,
                                  ArrayRef<std::pair<Value, AffineMap>> outs,
                                  unsigned numLoops, std::string &why) {
  auto all = llvm::concat<const std::pair<Value, AffineMap>>(ins, outs);

  // D3: every loop dim named, as a bare dim, by some map.
  SmallVector<bool> named(numLoops, false);
  for (const auto &[value, map] : all)
    for (AffineExpr r : map.getResults())
      if (auto d = dyn_cast<AffineDimExpr>(r))
        named[d.getPosition()] = true;
  for (auto [d, isNamed] : llvm::enumerate(named))
    if (!isNamed) {
      why = "D3 loop dim d" + std::to_string(d) + " is named by no map";
      return failure();
    }

  // D4: no value in both ins and outs.
  for (const auto &in : ins)
    for (const auto &out : outs)
      if (in.first == out.first) {
        why = "D4 an ins would alias an outs";
        return failure();
      }
  return success();
}

LogicalResult foldSplatConstantOperand(RewriterBase &rewriter,
                                       linalg::GenericOp generic,
                                       OpOperand *operand) {
  DenseElementsAttr splat;
  if (!generic.isDpsInput(operand) ||
      !matchPattern(operand->get(), m_Constant(&splat)) || !splat.isSplat() ||
      !splat.getElementType().isIntOrFloat())
    return failure();

  unsigned index = operand->getOperandNumber();
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  maps.erase(maps.begin() + index);
  if (!inversePermutation(concatAffineMaps(maps, rewriter.getContext())))
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(generic);
  Value constant = arith::ConstantOp::create(
      rewriter, operand->get().getLoc(), splat.getSplatValue<TypedAttr>());
  Block *body = generic.getBlock();
  rewriter.replaceAllUsesWith(body->getArgument(index), constant);
  rewriter.modifyOpInPlace(generic, [&] {
    body->eraseArgument(index);
    generic.getInputsMutable().erase(index);
    generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
  });
  return success();
}

} // namespace mlir::triton::spyre
