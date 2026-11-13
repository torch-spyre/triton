//===- SpyreOpTypeProbe.cpp - Which types a spyreop rule lowers -----------===//
//
// probeSpyreOpRule. See SpyreOpTypeProbe.h for why the types are asked of the
// ops themselves, and what that requires of a rule's build.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpyreOp/Utils/SpyreOpTypeProbe.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace mlir::triton::spyre {

bool probeSpyreOpRule(const ShapeCheckedRule &shape, std::string &why) {
  const SpyreOpLoweringRule &rule = shape.rule();
  const GroupCandidate &candidate = shape.candidate();
  MLIRContext *ctx = candidate.anchor->getContext();

  // The handler goes first, so nothing the build or the verifiers emit reaches
  // the context's own. The reason is the first the verifiers emit.
  std::string diagnostic;
  ScopedDiagnosticHandler handler(ctx, [&](Diagnostic &diag) {
    if (diagnostic.empty())
      diagnostic = diag.str();
    return success();
  });

  // The scratch block's arguments stand for the operands, in order. A bare
  // OpBuilder: a rewriter's would record the ops it creates.
  Block scratch;
  for (Value operand : shape.operands())
    scratch.addArgument(operand.getType(), operand.getLoc());
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(&scratch);
  Value result = rule.build(builder, candidate, scratch.getArguments());

  // The contract on build: it reaches nothing outside the scratch block, and
  // returns a value of the type of the op it replaces. Checked here because a
  // violation would otherwise verify against the wrong types, or fail only
  // after LowerByRule has committed. A broken rule is a compiler bug, so it
  // aborts rather than declining, which would stop the rule lowering silently.
  auto insideScratch = [&](Value v) { return v.getParentBlock() == &scratch; };
  bool reachesOutside = llvm::any_of(scratch, [&](Operation &op) {
    return !llvm::all_of(op.getOperands(), insideScratch);
  });
  Operation *replaced =
      rule.readers.empty() ? candidate.anchor : candidate.readers.front();
  bool wrongResult = !result || !insideScratch(result) ||
                     result.getType() != replaced->getResult(0).getType();
  if (reachesOutside || wrongResult)
    llvm::report_fatal_error(
        "SpyreOpTypeProbe: the rule '" + rule.name +
        "' breaks the contract on its build: it must read only its operands "
        "and return the replaced op's type (SpyreOpTypeProbe.h)");

  // Only the verifiers' diagnostics give the reason.
  diagnostic.clear();
  for (Operation &op : scratch)
    if (failed(verify(&op, /*verifyRecursively=*/false))) {
      why = "'" + op.getName().getStringRef().str() +
            "' does not take these types: " +
            (diagnostic.empty() ? "its verifier failed" : diagnostic);
      return false;
    }
  return true;
}

} // namespace mlir::triton::spyre
