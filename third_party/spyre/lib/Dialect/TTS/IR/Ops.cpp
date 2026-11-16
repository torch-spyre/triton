//===- Ops.cpp - The tts dialect's ops ------------------------------------===//
//
// Three ops. `tensor_layout`'s verifier delegates rather than restates -- see
// the note on `tts::verifyTensorLayoutArrays` for why the rules have a single
// owner.
//
// `pin`'s rules are its own, and what is left of them after ODS is small: the
// offset is an attribute, so its spelling is a type constraint the generated
// verifier already enforces, and the number in it is a consumer's business. What
// stays here is the memory space, which no type constraint can check against
// ktdp's enum. What deliberately does NOT stay is any rule about the operand's
// provenance -- see the note in `PinOp::verify`.
//
// `spyre_op`'s verifier delegates too, to `verifySpyreIntrinsicCall` -- see the
// note there for why the real `spyreop.<name>` op, not a restatement of its
// shape, is the thing this checks against.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/Twine.h"

#define GET_OP_CLASSES
#include "Dialect/TTS/IR/Ops.cpp.inc"

namespace mlir::triton::tts {

LogicalResult TensorLayoutOp::verify() {
  // The logical rank comes from the descriptor's BLOCK type. The attribute form
  // reads it from the memory view's memref instead; the extents differ between
  // the two, the rank does not, so both measure the same bound.
  unsigned logicalRank = getDesc().getType().getBlockType().getRank();

  // `emitError` and not `emitOpError`: every message the shared checker
  // produces already names `tts.tensor_layout`, so the op-error prefix would
  // say it twice — and identical text either side of the lowering is what makes
  // the op and the attribute diagnosable as one contract.
  auto emitError = [&]() { return this->emitError(); };
  return verifyTensorLayoutArrays(getPhysSrc(), getPhysOp(), getPhysArg(),
                                  logicalRank, emitError);
}

//===----------------------------------------------------------------------===//
// tts.pin
//===----------------------------------------------------------------------===//

LogicalResult PinOp::verify() {
  // NO block-argument rule here, and its absence is the point. A pinned value does
  // need a defining op -- that op is the annotation's carrier -- but whether one
  // exists is a fact about the operand's PROVENANCE, which is IR outside this op.
  // A verifier runs after every pass, so a predicate of that shape lets a legal
  // rewrite somewhere else turn a valid module invalid and report it here: the
  // canonicalizer (Pipeline.cpp, before LowerTTSMarkers) folds `x * 1`, `x + 0` and
  // an identity reshape/broadcast/transpose, each of which RAUWs this op's operand
  // to the fold's input without touching this op. If that input is a block
  // argument the check would fire on IR the author never wrote.
  //
  // Contrast the field rules below: they read attributes STORED ON THIS OP, which
  // no pass can change without rewriting the op. That is the property "local"
  // has to mean for a verifier, and it is not the same as needing no pass context.
  //
  // The authoring question -- did the author name a block argument -- is answerable
  // where authoring information still exists, which is trace time; see
  // `semantic.spyre_pin`. By the time a module is verified the two cases are
  // indistinguishable. What is left for the lowering is a defining op it can
  // actually miss, which `resolvePin` diagnoses naming both of its causes.
  //
  // Everything below is a rule about the FIELDS, and this is the only place they
  // are checked. The attribute form is built from fields that have already passed
  // here, so the dialect's `verifyOperationAttribute` acknowledges the name and
  // re-derives nothing -- see the note there.
  StringRef space = getMemorySpace();

  // Two questions, in this order, because they have different answers. Whether the
  // name is one ktdp DEFINES is settled by symbolizing it -- the string is builtin,
  // so unlike a `#ktdp.memory_space` operand nothing earlier can have rejected a
  // name ktdp never heard of. `symbolizeMemorySpaceKind` needs no context and no
  // loaded dialect, which is what lets this check live here at all; see the note in
  // TTSOps.td on why the field is a name and not that attribute.
  std::optional<mlir::ktdp::MemorySpaceKind> kind =
      mlir::ktdp::symbolizeMemorySpaceKind(space);
  if (!kind)
    return emitError() << "tts.pin: '" << space
                       << "' is not a ktdp memory space kind";

  // Then which of the kinds that exist a pin may NAME. `global` is a known kind
  // that it may not -- an intermediate in HBM is written as a descriptor with an
  // explicit store and load.
  if (*kind != mlir::ktdp::MemorySpaceKind::ct_local)
    return emitError() << "tts.pin: memory space '" << space
                       << "' cannot be pinned: only 'ct_local' is";

  // No ct_id rule, and none is missing. A name cannot carry one, so `ct_id = 7` --
  // core 7's scratchpad, a different request and one nothing here honours -- is not
  // expressible rather than refused. A pin is always the running core's own.

  // An offsetless pin is the design's baseline -- the compiler places every
  // intermediate and a pin only overrides where -- and it is refused because
  // nothing in this tree can act on it. There is no placement analysis and nothing
  // that allocates a buffer which is not a kernel argument, so a pin naming no
  // offset names no location at all.
  //
  // TODO: admit this form once something can place it. The field stays optional in
  // ODS so the surface does not have to change shape when that happens.
  if (!getOffsetAttr())
    return emitError() << "tts.pin: no offset, and nothing here can choose one "
                          "-- there is no placement analysis and nothing "
                          "allocates a buffer that is not a kernel argument; "
                          "state an offset";

  // And that is every rule about the offset. A single `i32` has no shape left to
  // check: that it is one, and that it is an integer, is ODS's from the type
  // constraint. Whether the NUMBER fits the device is answerable only against a
  // device description this dialect deliberately does not hold, so it is
  // MaterializePinnedBuffers', which takes it as a pass option. Alignment is not
  // asked anywhere: the offset counts from an allocator-assigned base, and
  // aligning that base is the allocator's rule rather than the author's.
  return success();
}

//===----------------------------------------------------------------------===//
// tts.spyre_op
//===----------------------------------------------------------------------===//

/// Whether `name` is reduction-shaped: its real `spyreop.*` op's result
/// covers fewer tensor dimensions than its operands, so `tts.spyre_op`'s
/// `axis` names which one. Not inferable from the real op itself -- nothing
/// about `spyreop.exx2`'s own (scalar) definition says this; it is a fact
/// about the LOWERING this one intrinsic needs (Group D2, modeled on
/// `ConvertTTReduce`), which is why this is the one piece of per-intrinsic
/// knowledge this file has to state itself rather than read off `SpyreOp.td`.
/// Widen this, not `extraAttrs`, the day a second reduction-shaped intrinsic
/// exists.
static bool isReductionShapedIntrinsic(StringRef name) {
  return name == "exx2";
}

FailureOr<SmallVector<Type>> verifySpyreIntrinsicCall(
    StringRef name, ValueRange operands, DictionaryAttr extraAttrs,
    std::optional<int64_t> axis, Location loc,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = loc->getContext();

  // Requires `spyreop` LOADED, not merely registered -- a lookup only finds
  // ops of a dialect the context has actually loaded. `tts`'s own
  // `dependentDialects` (TTSDialect.td) guarantees this for every caller that
  // goes through loading `tts` itself, which both callers of this function
  // do (the verifier, by construction of being `tts.spyre_op`'s own; the
  // pybind builder, which loads `tts::TTSDialect` before building anything).
  std::string qualified = ("spyreop." + name).str();
  std::optional<RegisteredOperationName> info =
      RegisteredOperationName::lookup(qualified, ctx);
  if (!info)
    return emitError() << "tts.spyre_op: '" << name
                       << "' does not name a registered spyreop intrinsic";

  // Every entry of `extraAttrs` must be one of the real op's own attributes
  // -- caught here because `mlir::verify()` below only checks the attributes
  // the real op's ODS declares, never flags one it does not know about, so
  // `tl.spyre_op("layernormscale", mean, msq, eps=1e-5)` -- the exact shape
  // the design doc once wrote by mistake for an op with no `eps` at all --
  // would otherwise verify silently with the stray attribute just ignored.
  // `axis` is deliberately NOT one of `extraAttrs`'s entries -- see TTSOps.td
  // -- so it never reaches this check or the real op at all.
  if (extraAttrs) {
    ArrayRef<StringAttr> known = info->getAttributeNames();
    for (NamedAttribute attr : extraAttrs) {
      if (!llvm::is_contained(known, attr.getName()))
        return emitError() << "tts.spyre_op: '" << attr.getName()
                           << "' is not an attribute of spyreop." << name;
    }
  }

  bool reductionShaped = isReductionShapedIntrinsic(name);
  if (reductionShaped && !axis)
    return emitError() << "tts.spyre_op: 'spyreop." << name
                       << "' is reduction-shaped and needs an 'axis'";
  if (!reductionShaped && axis)
    return emitError() << "tts.spyre_op: 'spyreop." << name
                       << "' is not reduction-shaped; it takes no 'axis'";

  if (operands.empty())
    return emitError() << "tts.spyre_op: 'spyreop." << name
                       << "' needs at least one operand";
  auto firstTensorTy = dyn_cast<RankedTensorType>(operands[0].getType());
  if (!firstTensorTy)
    return emitError() << "tts.spyre_op: operand of 'spyreop." << name
                       << "' is not a ranked tensor";
  if (axis && (*axis < 0 || *axis >= firstTensorTy.getRank()))
    return emitError() << "tts.spyre_op: 'axis' " << *axis
                       << " is out of range for rank "
                       << firstTensorTy.getRank();

  // `operands` are TENSORS; every `spyreop.*` op in scope is SCALAR-typed
  // (confirmed against SpyreOp.td -- no tensor-typed `SpyreOp_IntrinsicOp`
  // exists). The request names a per-element computation, realized later by
  // Group D1/D2 putting a scalar of each element type into the real op's
  // body inside a `linalg.generic` (see the worked example in
  // spyreop-frontend-api.md: `spyreop.gelu %in : f16`, `%in` a block
  // argument, not `%x` itself). So the real op is checked against each
  // operand's ELEMENT type, not the tensor type -- a throwaway block
  // supplies one scalar argument per element type purely so `verify()` below
  // has real `Value`s of the right type to point at; nothing about this
  // block is kept, and nothing reads through these values (every
  // `inferReturnTypes` implementation in scope only reads `.getType()`,
  // confirmed directly against the generated `Exx2::inferReturnTypes`).
  SmallVector<Type> elementTypes;
  for (Value operand : operands) {
    auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorTy)
      return emitError() << "tts.spyre_op: operand of 'spyreop." << name
                         << "' is not a ranked tensor";
    elementTypes.push_back(tensorTy.getElementType());
  }
  Block scratchBlock;
  SmallVector<Location> argLocs(elementTypes.size(), loc);
  scratchBlock.addArguments(elementTypes, argLocs);
  SmallVector<Value> scalarOperands(scratchBlock.getArguments().begin(),
                                    scratchBlock.getArguments().end());

  // The real op's scalar result TYPES, inferred before anything is built:
  // every op in scope has InferTypeOpInterface (tablegen derives it
  // automatically from SameOperandsAndResultType with non-variadic results
  // -- see the note on this function in Dialect.h), and `OperationState`
  // needs its result `TypeRange` up front, so this is what answers "how
  // many results, and what type" without a probe op.
  auto *infer = info->getInterface<InferTypeOpInterface>();
  if (!infer)
    return emitError() << "tts.spyre_op: 'spyreop." << name
                       << "' is not an intrinsic this op can request";

  SmallVector<Type> scalarResultTypes;
  DictionaryAttr attrsForInfer =
      extraAttrs ? extraAttrs : DictionaryAttr::get(ctx);
  if (failed(infer->inferReturnTypes(ctx, loc, scalarOperands, attrsForInfer,
                                     /*properties=*/nullptr, /*regions=*/{},
                                     scalarResultTypes)))
    return emitError() << "tts.spyre_op: 'spyreop." << name
                       << "' refused these operands";

  // Build the real (scalar) op and verify it for everything type inference
  // does not itself check: operand/attribute type mismatches,
  // `SameOperandsAndResultType` itself, and a confined-attribute predicate
  // the real op's ODS states (`softplus`'s non-zero `beta`, among others).
  SmallVector<NamedAttribute> attrList;
  if (extraAttrs)
    attrList.assign(extraAttrs.begin(), extraAttrs.end());
  OperationState state(loc, *info, scalarOperands, scalarResultTypes,
                       attrList);
  OpBuilder builder(ctx);
  Operation *probe = builder.create(state);
  std::string diagnostic;
  bool probeFailed;
  {
    // Scoped to exactly the `verify()` call: an InFlightDiagnostic built
    // AFTER this handler is uninstalled is what the caller's own `emitError`
    // actually wants reported to the real user. Building it inside this
    // scope instead -- as a first version of this function did -- feeds the
    // outer diagnostic back into this same handler, which marks it `success()`
    // ("fully handled, do not print") and the real error vanishes silently:
    // confirmed the hard way, as a `tts.spyre_op` test case that should have
    // reported a clean diagnostic instead exited 1 with no output at all.
    llvm::raw_string_ostream os(diagnostic);
    ScopedDiagnosticHandler handler(ctx, [&](Diagnostic &diag) {
      diag.print(os);
      os << "; ";
      return success();
    });
    probeFailed = failed(verify(probe, /*verifyRecursively=*/false));
  }
  probe->erase();
  if (probeFailed)
    return emitError() << "tts.spyre_op: 'spyreop." << name
                       << "' rejects this call: " << diagnostic;

  // Lift each scalar result type back to the tensor shape `tts.spyre_op`
  // itself deals in. Elementwise (D1): every result shares the first
  // operand's shape, matching `SameOperandsAndResultType` one level up.
  // Reduction-shaped (D2, `axis` present): a result drops the `axis`
  // dimension -- the same rule `tt.reduce`'s own result shape follows
  // (`ConvertTTReduce`, `LowerComputeOps.cpp`: every dim except `axis`).
  SmallVector<int64_t> resultShape(firstTensorTy.getShape());
  if (axis)
    resultShape.erase(resultShape.begin() + *axis);
  SmallVector<Type> resultTypes;
  for (Type scalarTy : scalarResultTypes)
    resultTypes.push_back(RankedTensorType::get(resultShape, scalarTy));
  return resultTypes;
}

LogicalResult SpyreOpOp::verify() {
  auto emitError = [&]() { return this->emitError(); };
  std::optional<int64_t> axis;
  if (std::optional<uint32_t> a = getAxis())
    axis = *a;
  FailureOr<SmallVector<Type>> resultTypes = verifySpyreIntrinsicCall(
      getName(), getOperands(), getExtraAttrsAttr(), axis, getLoc(),
      emitError);
  if (failed(resultTypes))
    return failure();

  if (resultTypes->size() != getResults().size())
    return emitError() << "tts.spyre_op: 'spyreop." << getName()
                       << "' has " << resultTypes->size()
                       << " result(s), but this call declares "
                       << getResults().size();
  for (auto [declared, inferred] :
       llvm::zip_equal(getResults().getTypes(), *resultTypes))
    if (declared != inferred)
      return emitError() << "tts.spyre_op: 'spyreop." << getName()
                         << "' infers result type " << inferred
                         << ", but this call declares " << declared;
  return success();
}

} // namespace mlir::triton::tts
