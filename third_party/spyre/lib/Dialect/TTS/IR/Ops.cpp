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

  // Everything below is a rule about the FIELDS, and this is the only place they
  // are checked. The attribute form is built from fields that have already passed
  // here, so the dialect's `verifyOperationAttribute` acknowledges the name and
  // re-derives nothing -- see the note there.
  mlir::ktdp::MemorySpaceAttr space = getMemorySpaceAttr();

  // Which kinds EXIST is ktdp's business and is settled before this runs: the
  // attribute would not have parsed otherwise. What is left is which of them a pin
  // may name, and `global` is a known kind that it may not -- an intermediate in
  // HBM is written as a descriptor with an explicit store and load.
  if (space.getKind() != mlir::ktdp::MemorySpaceKind::ct_local)
    return emitError() << "tts.pin: memory space '"
                       << mlir::ktdp::stringifyMemorySpaceKind(space.getKind())
                       << "' cannot be pinned: only 'ct_local' is";

  // A pin means the scratchpad of whichever core is running, so there is no core
  // to name. `ct_id = 7` would ask for core 7's scratchpad, which is a different
  // request and one nothing here honours -- and an unhonoured ct_id would be
  // silent, since the buffer would simply be built somewhere else.
  if (space.hasCTIdSpecified())
    return emitError() << "tts.pin: memory space names ct_id " << space.getCtId()
                       << "; a pin is always the running core's own scratchpad, "
                          "so leave ct_id unspecified";

  // An addressless pin is the design's baseline -- the compiler places every
  // intermediate and a pin only overrides where -- and it is refused because
  // nothing in this tree can act on it. There is no address analysis and nothing
  // that allocates a buffer which is not a kernel argument, so a pin naming no
  // address names no location at all. Offset 0 is not the fallback: it is a
  // legitimate address that would collide with the scheduler's own pool.
  //
  // TODO: admit this form once something can place it. The field stays optional in
  // ODS so the surface does not have to change shape when that happens.
  Attribute address = getAddressAttr();
  if (!address)
    return emitError() << "tts.pin: no address, and nothing here can choose one "
                          "-- there is no address analysis and nothing allocates "
                          "a buffer that is not a kernel argument; state an "
                          "address";

  // An EMPTY array names no address for any core, so it is neither the uniform
  // spelling nor the per-core one, and a consumer indexing it by the program id
  // reads out of bounds on the first core. The only address rule left to check --
  // that it is an i32 or an i32 array is ODS's, from the type constraint.
  if (auto perCore = dyn_cast<DenseI32ArrayAttr>(address))
    if (perCore.empty())
      return emitError() << "tts.pin: address array is empty: state one address "
                            "per program id, or a single i32 for an address that "
                            "is the same on every core";

  return success();
}

} // namespace mlir::triton::tts
