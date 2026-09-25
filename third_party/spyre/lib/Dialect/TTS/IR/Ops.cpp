//===- Ops.cpp - The tts dialect's ops ------------------------------------===//
//
// Two ops. `tensor_layout`'s verifier delegates rather than restates -- see the
// note on `tts::verifyTensorLayoutArrays` for why the rules have a single owner.
//
// `pin`'s rules are its own, and what is left of them after ODS is small: the
// address is an attribute, so its two spellings are a type constraint the
// generated verifier already enforces, and the numbers in it are a consumer's
// business. What stays here is the memory space, which no type constraint can
// check against ktdp's enum, and the one array shape ODS cannot express.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/BuiltinTypes.h"

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
  // Delegates rather than restates, for the reason `tensor_layout`'s does: the
  // same rules are checked again on the ATTRIBUTE form by the dialect's
  // `verifyOperationAttribute`, and a second copy would answer differently the
  // first time anyone admits a new memory space.
  //
  // `emitError` and not `emitOpError`: every message the shared checker produces
  // already names `tts.pin`, so the op-error prefix would say it twice -- and
  // identical text either side of the lowering is what makes the op and the
  // attribute diagnosable as one contract.
  //
  // A block argument is deliberately NOT refused here. A `tensor` is a value
  // whether an op or a block argument defines it, so it is well formed as a pin;
  // what has no answer is which op would carry the annotation, and that is
  // `LowerTTSMarkers`' question rather than this one.
  auto emitError = [&]() { return this->emitError(); };
  return verifyPinFields(getMemorySpace(), getAddressAttr(), emitError);
}

} // namespace mlir::triton::tts
