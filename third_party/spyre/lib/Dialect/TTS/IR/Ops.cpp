//===- Ops.cpp - The tts dialect's ops ------------------------------------===//
//
// Four ops. `tensor_layout`'s verifier delegates rather than restates -- see the
// note on `tts::verifyTensorLayoutArrays` for why the rules have a single owner.
//
// `pin`'s rules are its own, and what is left of them after ODS is small: the
// offset is an attribute, so its spelling is a type constraint the generated
// verifier already enforces, and the number in it is a consumer's business. What
// stays here is the memory space, which no type constraint can check against
// ktdp's enum. What deliberately does NOT stay is any rule about the operand's
// provenance -- see the note in `PinOp::verify`.
//
// `spyre_op`'s rules are about its body, which is why they are a region
// verifier: the block's signature is the operands', its terminator yields the
// results, and nothing in it touches memory.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"

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

LogicalResult SpyreOpOp::verifyRegions() {
  Block &body = getBody().front();

  if (body.getArgumentTypes() != getInputs().getTypes())
    return emitOpError("body block arguments must have the operand types, one "
                       "for one");

  // Read as `back()` and not `getTerminator()`, which asserts on a block whose
  // last op is not one.
  auto yield = dyn_cast<SpyreOpYieldOp>(body.back());
  if (!yield)
    return emitOpError("body must end in tts.spyreop_yield, not '")
           << body.back().getName() << "'";
  if (yield.getValues().getTypes() != getResultTypes())
    return emitOpError("body yields ")
           << yield.getValues().getTypes() << " but the op's results are "
           << getResultTypes();

  // No memory op anywhere in the body, nested regions included. A call is the
  // one op admitted unasked: tracing reaches the fallback through a `tt.call`,
  // and what its callee does is IR outside this op, which a verifier must not
  // read. The `ttir` stage's inliner replaces the call with that body, which is
  // then held to this rule like any other.
  Operation *offender = nullptr;
  body.walk([&](Operation *op) {
    if (op == yield.getOperation() || isa<CallOpInterface>(op))
      return WalkResult::advance();
    // An op with no effect interface at all counts as having effects, so an
    // op this rule has never heard of is refused rather than admitted.
    if (isMemoryEffectFree(op))
      return WalkResult::advance();
    offender = op;
    return WalkResult::interrupt();
  });
  if (offender) {
    InFlightDiagnostic diag =
        emitOpError("body must not access memory: the request names an "
                    "intrinsic computing on values, and '")
        << offender->getName() << "' has memory effects";
    diag.attachNote(offender->getLoc()) << "the op is here";
    return diag;
  }
  return success();
}

} // namespace mlir::triton::tts
