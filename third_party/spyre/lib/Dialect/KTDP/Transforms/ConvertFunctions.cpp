//===- ConvertFunctions.cpp - Convert tt.func/return to func dialect ------===//
//
// Converts Triton function-level ops to the func dialect and retypes !tt.ptr
// function arguments to index.
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

    // Checked before anything is rewritten, so a rejected module is left
    // untouched rather than half-converted.
    if (failed(checkPointerArgsOnlyFeedCasts(module))) {
      signalPassFailure();
      return;
    }

    convertFunctions(module);
    convertReturns(module);
    retypePointerArgsToIndex(module);
  }

private:
  /// The indices of the results of `cast` that carry operand `operandIdx`'s
  /// value, or nullopt if the cast's shape gives that operand no result of its
  /// own.
  ///
  /// builtin.unrealized_conversion_cast is variadic on both sides, and its arity
  /// decides how operands correspond to results. Upstream MLIR's folder pairs
  /// them by position, so this follows the same reading:
  ///
  ///   1 operand -> N results  ("fan-out": one value expanded, e.g. a tuple
  ///                            split into its element types). Every result
  ///                            carries the single operand, so all of them
  ///                            correspond to it.
  ///   N operands -> N results ("pairwise": result i carries operand i). Only
  ///                            result `operandIdx` corresponds to this operand.
  ///   N operands -> 1 result  ("collapse": the lone result is a joint function
  ///                            of every operand, e.g. a tuple built from
  ///                            several values). No single operand corresponds
  ///                            to it, so nullopt is returned.
  ///
  /// A collapse has no correct per-operand rewrite: forwarding the result to one
  /// operand would silently drop the others' contribution. Upstream agrees — it
  /// leaves a lone N-to-1 cast alone and only cancels it against a matching
  /// 1-to-N partner.
  static std::optional<SmallVector<unsigned>>
  getCorrespondingResultIndices(UnrealizedConversionCastOp cast,
                                unsigned operandIdx) {
    unsigned numResults = cast.getNumResults();
    if (cast.getNumOperands() == 1) {
      SmallVector<unsigned> allResults;
      for (unsigned i = 0; i < numResults; ++i)
        allResults.push_back(i);
      return allResults;
    }
    if (cast.getNumOperands() == numResults)
      return SmallVector<unsigned>{operandIdx};
    return std::nullopt;
  }

  /// The indices of every result of `cast` that carries `arg`, or nullopt if any
  /// operand position holding `arg` has no result of its own.
  ///
  /// An argument may occupy more than one operand position of the same cast
  /// (`cast %p, %p`), so this unions the correspondences of all of them.
  /// Duplicates are dropped: the fan-out shape maps every operand position to the
  /// same full result list, and forwarding one result twice would be wasted work.
  /// Indices come out in ascending order, since getCorrespondingResultIndices
  /// yields them that way per position and positions are visited in order.
  ///
  /// Both the precheck and the rewrite read this one function, so what gets
  /// type-checked is exactly what gets forwarded.
  static std::optional<SmallVector<unsigned>>
  getResultIndicesForArg(UnrealizedConversionCastOp cast, Value arg) {
    SmallVector<unsigned> indices;
    for (OpOperand &operand : cast->getOpOperands()) {
      if (operand.get() != arg)
        continue;
      auto perOperand =
          getCorrespondingResultIndices(cast, operand.getOperandNumber());
      if (!perOperand)
        return std::nullopt;
      for (unsigned resultIdx : *perOperand)
        if (!llvm::is_contained(indices, resultIdx))
          indices.push_back(resultIdx);
    }
    return indices;
  }

  /// Reject a cast that this pass cannot fold once the argument becomes index.
  ///
  /// Two ways it can fail. The cast may collapse several operands into one
  /// result, leaving no result that corresponds to this argument. Or a
  /// corresponding result may be typed something other than index, in which case
  /// forwarding the index-typed argument into it would retype a value its users
  /// still read at the old type — the pass would report success and the verifier
  /// would then blame an innocent op, which is the failure mode this precheck
  /// exists to prevent.
  ///
  /// Every result belonging to the argument is checked, not just the first. An
  /// argument repeated across operand positions owns a result per position, and a
  /// mistype in any of them is equally unfoldable.
  LogicalResult checkCastResultsAreIndex(UnrealizedConversionCastOp cast,
                                         BlockArgument arg,
                                         FunctionOpInterface fnOp) {
    auto resultIndices = getResultIndicesForArg(cast, arg);
    if (!resultIndices)
      return cast.emitError()
             << "cannot convert function signature: !tt.ptr argument #"
             << arg.getArgNumber() << " of '" << fnOp.getName()
             << "' feeds an unrealized_conversion_cast with "
             << cast.getNumOperands() << " operands and "
             << cast.getNumResults()
             << " results; a cast that collapses several operands into fewer "
                "results has no result belonging to this argument alone, so "
                "there is no way to fold it away when the argument becomes an "
                "index";

    for (unsigned resultIdx : *resultIndices)
      if (!cast.getResult(resultIdx).getType().isIndex())
        return cast.emitError()
               << "cannot convert function signature: !tt.ptr argument #"
               << arg.getArgNumber() << " of '" << fnOp.getName()
               << "' feeds an unrealized_conversion_cast whose result #"
               << resultIdx << " has type "
               << cast.getResult(resultIdx).getType()
               << ", not index; replacing the argument with an index would "
                  "leave that result wrongly typed for its users";

    return success();
  }

  /// Reject any !tt.ptr function parameter that is read by something other than
  /// an unrealized_conversion_cast this pass can fold.
  ///
  /// retypePointerArgsToIndex below retypes such a parameter to index and folds
  /// away the casts. It fixes up only the casts, so any other user would be left
  /// holding an operand whose type changed underneath it: the pass reports
  /// success and the verifier then fails on an op that did nothing wrong.
  ///
  /// That state means a memory pass never consumed the pointer via
  /// getBasePtrAsIndex — a pass ordering error. The diagnostic names that
  /// instead of leaving the verifier to blame the surviving op.
  ///
  /// Being an unrealized_conversion_cast is necessary but not sufficient: the op
  /// is variadic on both sides, so its arity decides whether any result even
  /// corresponds to a given argument. checkCastResultsAreIndex screens that.
  LogicalResult checkPointerArgsOnlyFeedCasts(ModuleOp module) {
    // FunctionOpInterface covers tt.func and func.func alike. Input may already
    // contain func.func, and retypePointerArgsToIndex walks every one of them.
    auto result = module.walk([&](FunctionOpInterface fnOp) -> WalkResult {
      // A declaration has no entry block, so it has no argument users to
      // inspect. retypePointerArgsToIndex skips it too.
      if (fnOp.isExternal())
        return WalkResult::advance();

      for (BlockArgument arg : fnOp.getFunctionBody().front().getArguments()) {
        if (!isa<triton::PointerType>(arg.getType()))
          continue;

        for (Operation *user : arg.getUsers()) {
          auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
          if (!cast)
            return user->emitError()
                   << "cannot convert function signature: !tt.ptr argument #"
                   << arg.getArgNumber() << " of '" << fnOp.getName()
                   << "' is used by an op that is not an "
                      "unrealized_conversion_cast, so replacing the argument "
                      "with an index would leave this operand wrongly typed; "
                      "the memory passes must consume every !tt.ptr use (via "
                      "getBasePtrAsIndex) before this pass runs";

          if (failed(checkCastResultsAreIndex(cast, arg, fnOp)))
            return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    return failure(result.wasInterrupted());
  }

  /// tt.func -> func.func, preserving the body and the visibility marker.
  ///
  /// Reaches tt.func at any depth, not just the top-level module's direct
  /// children: tt.func only requires a ModuleOp parent, and a nested
  /// builtin.module satisfies that. Enumerating direct children alone would skip
  /// a nested one while convertReturns still rewrote its tt.return, leaving a
  /// tt.func whose terminator is func.return — which fails verification, since
  /// tt.return requires a func.func parent. All traversals in this pass must
  /// agree on which functions they touch.
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
      // marker has to be copied deliberately. Otherwise a private or nested
      // tt.func comes out public, telling symbol-DCE and the inliner the
      // opposite of the truth.
      //
      // Copy the raw sym_visibility attribute, not setVisibility(getVisibility()).
      // tt.func declares sym_visibility as a plain OptionalAttr<StrAttr> and does
      // not implement SymbolOpInterface, so getVisibility() ignores it and reports
      // Public for every input; setVisibility then erases the attribute, since
      // SymbolTable::setSymbolVisibility drops it for Public. That round trip
      // collapsed private and nested to public.
      //
      // "public" is left implicit, matching how MLIR prints func.func — copying it
      // would emit a redundant marker no other func.func producer emits.
      if (auto visibility = ttFunc.getSymVisibilityAttr();
          visibility && visibility.getValue() != "public")
        funcOp.setSymVisibilityAttr(visibility);

      // Move the body rather than cloning it block by block: a move carries every
      // block, its arguments, and its successor references at once, so there is no
      // partial-copy state to get wrong on a multi-block body. Moving is safe
      // because tt.func is IsolatedFromAbove — the body cannot reference a value
      // defined outside it — and because the tt.func is erased just below. Keeping
      // the tt.func alive past this point would require cloneInto + IRMapping.
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
  /// no other users, and that every cast reading it has an index-typed result
  /// corresponding to that parameter. checkPointerArgsOnlyFeedCasts has already
  /// rejected the module otherwise, which is what makes the unconditional setType
  /// and result forwarding below safe.
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

      // Every cast reached while forwarding, kept so the erase sweep below can
      // run after all forwarding is done. Nothing is erased inside the loops:
      // erasing there would invalidate a cast that a later iteration still has
      // to visit, and a cast can be reached more than once — once per operand
      // position holding the same argument, and once per pointer argument
      // feeding it. Deferring the erase makes a repeat visit genuinely harmless,
      // because forwarding an already-forwarded result is a no-op.
      SmallVector<UnrealizedConversionCastOp> reachedCasts;

      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        BlockArgument arg = entry.getArgument(i);
        if (!isa<triton::PointerType>(arg.getType()))
          continue;

        // Collect before retyping: replaceAllUsesWith below mutates the use list
        // this range walks.
        SmallVector<UnrealizedConversionCastOp> casts;
        for (auto *user : arg.getUsers())
          if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user))
            casts.push_back(cast);

        arg.setType(builder.getIndexType());

        for (auto cast : casts) {
          // Forward every result this argument owns — all of them, not just the
          // first, since the argument may sit in several operand positions and
          // owns a result per position. checkCastResultsAreIndex has already
          // established that at least one result corresponds to this argument and
          // that every such result is index-typed, so each forward is an identity
          // now that the argument is an index.
          auto resultIndices = getResultIndicesForArg(cast, arg);
          for (unsigned resultIdx : *resultIndices)
            cast.getResult(resultIdx).replaceAllUsesWith(arg);

          reachedCasts.push_back(cast);
        }
      }

      // A pairwise cast's other results still carry other operands, so a cast is
      // erased only once nothing reads it: the whole cast for the one-operand
      // shapes, and for a pairwise cast only after every pointer argument feeding
      // it has been folded. use_empty() is only a reliable test for that here,
      // after the forwarding above has finished.
      //
      // The sweep must visit each cast once: reachedCasts records one entry per
      // forwarding visit, so the same op can appear several times, and erasing one
      // op twice is a double free.
      SmallPtrSet<Operation *, 8> seen;
      for (auto cast : reachedCasts)
        if (seen.insert(cast).second && cast->use_empty())
          cast.erase();

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
