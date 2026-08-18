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
#include "mlir/Interfaces/FunctionInterfaces.h"
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

    // ---- Precondition check: every !tt.ptr function parameter must be
    // consumed only by the placeholder casts the memory passes leave behind.
    // Checked before anything is rewritten, so a rejected module is left
    // untouched rather than partially converted.
    if (failed(checkPointerArgsOnlyFeedCasts(module))) {
      signalPassFailure();
      return;
    }

    convertFunctions(module);
    convertReturns(module);
    retypePointerArgsToIndex(module);
  }

private:
  /// Reject any !tt.ptr function parameter that is read by something other than
  /// an unrealized_conversion_cast.
  ///
  /// retypePointerArgsToIndex below changes such a parameter's type to index and
  /// folds away the casts that consumed it. It can only fix up the casts, so a
  /// user of any other kind would be left holding an operand whose type changed
  /// underneath it — the pass would report success and the verifier would then
  /// fail on an op that did nothing wrong. Reaching this state means the memory
  /// passes have not consumed the pointer via getBasePtrAsIndex, which is a pass
  /// ordering error; the diagnostic says so rather than leaving the verifier to
  /// blame the surviving op.
  LogicalResult checkPointerArgsOnlyFeedCasts(ModuleOp module) {
    // Both tt.func and func.func are checked: this pass accepts a module that
    // already contains func.func (only tt.func is converted), and
    // retypePointerArgsToIndex walks every func.func regardless of origin.
    auto result = module.walk([](FunctionOpInterface fnOp) -> WalkResult {
      // A declaration has no entry block to hold the arguments, so it has no
      // argument users to inspect. Skipped for the same reason
      // retypePointerArgsToIndex skips it.
      if (fnOp.isExternal())
        return WalkResult::advance();

      for (BlockArgument arg : fnOp.getFunctionBody().front().getArguments()) {
        if (!isa<triton::PointerType>(arg.getType()))
          continue;

        for (Operation *user : arg.getUsers()) {
          if (isa<UnrealizedConversionCastOp>(user))
            continue;
          return user->emitError()
                 << "cannot convert function signature: !tt.ptr argument #"
                 << arg.getArgNumber() << " of '" << fnOp.getName()
                 << "' is used by an op that is not an "
                    "unrealized_conversion_cast, so replacing the argument "
                    "with an index would leave this operand wrongly typed; the "
                    "memory passes must consume every !tt.ptr use (via "
                    "getBasePtrAsIndex) before this pass runs";
        }
      }
      return WalkResult::advance();
    });

    return failure(result.wasInterrupted());
  }

  /// tt.func -> func.func, preserving the body and the visibility marker.
  ///
  /// Reaches tt.func at any depth, not just the direct children of the
  /// top-level module. tt.func is HasParent<"ModuleOp">, and a nested
  /// builtin.module satisfies that, so a tt.func can legally sit deeper than
  /// one level. Enumerating only direct children would skip it while
  /// convertReturns below still rewrote its tt.return, leaving a tt.func whose
  /// terminator is func.return — which fails verification, since tt.return is
  /// HasParent<"FuncOp">. All three traversals in this pass must agree on which
  /// functions they touch.
  void convertFunctions(ModuleOp module) {
    // Collect first, then rewrite. Erasing a tt.func inside the walk would
    // invalidate the walker's cursor, because the erased op owns nested regions
    // the walk is still traversing.
    SmallVector<triton::FuncOp> ttFuncs;
    module.walk([&](triton::FuncOp ttFunc) { ttFuncs.push_back(ttFunc); });

    for (auto ttFunc : ttFuncs) {
      OpBuilder builder(ttFunc);
      auto funcOp =
          func::FuncOp::create(builder, ttFunc.getLoc(), ttFunc.getName(),
                               ttFunc.getFunctionType());

      // func::FuncOp::create always produces a public function, so a non-public
      // marker has to be copied deliberately: a private or nested tt.func would
      // otherwise come out public, telling symbol-DCE and the inliner the
      // opposite of the truth.
      //
      // The marker is copied as the raw sym_visibility attribute rather than
      // via setVisibility(ttFunc.getVisibility()). tt.func declares
      // sym_visibility as an ordinary OptionalAttr<StrAttr> and does not
      // implement SymbolOpInterface, so getVisibility() does not read that
      // attribute — it reports Public for every input. Passing that to
      // setVisibility then *erases* the attribute on the result, because
      // SymbolTable::setSymbolVisibility drops it for Public. The round trip
      // collapsed private and nested to public.
      //
      // "public" is the default and is left implicit, matching how MLIR prints
      // func.func: copying it verbatim would emit a redundant `public` marker
      // that no other func.func producer emits.
      if (auto visibility = ttFunc.getSymVisibilityAttr();
          visibility && visibility.getValue() != "public")
        funcOp.setSymVisibilityAttr(visibility);

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

  /// Retype every !tt.ptr function parameter to index, and delete the
  /// placeholder casts that stood between the parameter and its index-typed
  /// uses.
  ///
  /// Only the casts are fixed up, so this requires that a pointer parameter has
  /// no other users. checkPointerArgsOnlyFeedCasts has already rejected the
  /// module otherwise, which is what makes the unconditional setType below safe.
  ///
  /// A declaration (a function with no body) is skipped: its parameters live
  /// only in the FunctionType, so there is no entry block to retype. A !tt.ptr
  /// in a declaration's signature therefore survives this pass — retyping one
  /// would also mean updating every caller, which is out of scope here.
  void retypePointerArgsToIndex(ModuleOp module) {
    module.walk([&](func::FuncOp funcOp) {
      // front() on a body with no blocks is undefined behaviour, not a
      // trapping error, so guard before reading the entry block.
      if (funcOp.isExternal())
        return;

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
