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
  // A block argument is refused, and the reason is about WHERE IT ALREADY LIVES
  // rather than about the annotation. An entry input is global: it lives at the
  // address its base pointer supplies, which the launcher fills in, so a pin
  // asking for it in `ct_local` is asking to relocate a kernel argument. That is
  // not what a pin does -- a pin says where an INTERMEDIATE's buffer goes, and an
  // intermediate is by definition something an op in this function produced.
  //
  // Here rather than in the lowering because it is answerable from the op alone:
  // whether a value is a block argument needs no pass context. The lowering has
  // its own reason to want a defining op -- it is the attribute's carrier -- but
  // that is a consequence, not the rule.
  if (isa<BlockArgument>(getValue()))
    return emitError() << "tts.pin names a block argument, which is an entry "
                          "input and lives where its base pointer says; a pin "
                          "places an intermediate some op here produced";

  // The rest delegates rather than restates, for the reason `tensor_layout`'s
  // does: the same rules are checked again on the ATTRIBUTE form by the dialect's
  // `verifyOperationAttribute`, and a second copy would answer differently the
  // first time anyone admits a new memory space.
  //
  // `emitError` and not `emitOpError`: every message the shared checker produces
  // already names `tts.pin`, so the op-error prefix would say it twice -- and
  // identical text either side of the lowering is what makes the op and the
  // attribute diagnosable as one contract.
  auto emitError = [&]() { return this->emitError(); };
  return verifyPinFields(getMemorySpace(), getAddressAttr(), emitError);
}

} // namespace mlir::triton::tts
