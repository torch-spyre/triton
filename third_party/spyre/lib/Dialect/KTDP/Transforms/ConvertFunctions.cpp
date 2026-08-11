//===- ConvertFunctions.cpp - Convert tt.func/return to func dialect ------===//
//
// Converts Triton function-level ops to standard func dialect and finalizes
// !tt.ptr function arguments to index type.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_CONVERTFUNCTIONS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

struct ConvertFunctionsPass
    : public mlir::triton::ktdp::impl::ConvertFunctionsBase<
          ConvertFunctionsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    convertFunctions(module);
    convertReturns(module);
    finalizeFunctionSignatures(module);
  }

private:
  /// tt.func -> func.func, preserving the body and the visibility marker.
  void convertFunctions(ModuleOp module) {
    // make_early_inc_range advances the iterator before the body runs, so
    // erasing the current tt.func mid-iteration is safe.
    for (auto ttFunc :
         llvm::make_early_inc_range(module.getOps<triton::FuncOp>())) {
      OpBuilder builder(ttFunc);
      auto funcOp =
          func::FuncOp::create(builder, ttFunc.getLoc(), ttFunc.getName(),
                               ttFunc.getFunctionType());

      // func::FuncOp::create always produces a public function, so this must
      // be unconditional: a private or nested tt.func would otherwise come out
      // public, telling symbol-DCE and the inliner the opposite of the truth.
      funcOp.setVisibility(ttFunc.getVisibility());

      // Move the body rather than cloning it block by block. A move carries
      // every block, its arguments, and its successor references at once, so
      // there is no partial-copy state to get wrong on a multi-block body.
      // Moving (not cloning) is valid because tt.func is IsolatedFromAbove —
      // the body cannot reference a value defined outside it — and because the
      // tt.func is erased immediately below. A future version that needs to
      // keep the tt.func alive past this point must use cloneInto + IRMapping.
      funcOp.getBody().takeBody(ttFunc.getBody());

      ttFunc.erase();
    }
  }

  /// tt.return -> func.return. A 1:1 operand mapping with no type conversion,
  /// so a walk suffices; there is no legality fixpoint to reach.
  void convertReturns(ModuleOp module) {
    module.walk([](triton::ReturnOp op) {
      OpBuilder builder(op);
      func::ReturnOp::create(builder, op.getLoc(), op.getOperands());
      op.erase();
    });
  }

  void finalizeFunctionSignatures(ModuleOp module) {
    module.walk([&](func::FuncOp funcOp) {
      Block &entry = funcOp.getBody().front();
      OpBuilder builder(funcOp.getContext());

      bool changed = false;
      SmallVector<Type> newArgTypes;

      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        BlockArgument arg = entry.getArgument(i);
        if (isa<triton::PointerType>(arg.getType())) {
          newArgTypes.push_back(builder.getIndexType());
          changed = true;
        } else {
          newArgTypes.push_back(arg.getType());
        }
      }

      if (!changed)
        return;

      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        BlockArgument arg = entry.getArgument(i);
        if (!isa<triton::PointerType>(arg.getType()))
          continue;

        SmallVector<UnrealizedConversionCastOp> casts;
        for (auto *user : arg.getUsers())
          if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user))
            casts.push_back(cast);

        arg.setType(builder.getIndexType());

        for (auto cast : casts) {
          cast.getResult(0).replaceAllUsesWith(arg);
          cast.erase();
        }
      }

      auto newFuncType = FunctionType::get(
          funcOp.getContext(), newArgTypes,
          funcOp.getFunctionType().getResults());
      funcOp.setType(newFuncType);
    });
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createConvertFunctionsPass() {
  return std::make_unique<ConvertFunctionsPass>();
}
} // namespace mlir::triton::ktdp
