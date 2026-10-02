//===- Ops.cpp - The tts dialect's ops ------------------------------------===//
//
// Two ops. `tensor_layout`'s verifier delegates rather than restates -- see the
// note on `tts::verifyTensorLayoutArrays` for why the rules have a single owner.
//
// `pin`'s rules are its own, and what is left of them after ODS is small: the
// offset is an attribute, so its spelling is a type constraint the generated
// verifier already enforces, and the number in it is a consumer's business. What
// stays here is the memory space, which no type constraint can check against
// ktdp's enum. What deliberately does NOT stay is any rule about the operand's
// provenance -- see the note in `PinOp::verify`.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

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
// tts.make_distributed_descriptor
//===----------------------------------------------------------------------===//

LogicalResult readWorkSliceTable(
    ArrayAttr table, SmallVectorImpl<WorkSliceEntry> &entries,
    llvm::MapVector<StringRef, int64_t> &sliceCounts,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  if (table.empty())
    return emitError() << "work_slices must not be empty: a view is composed "
                          "from at least one partition";

  SmallVector<StringRef> refKeys;
  for (auto [i, attr] : llvm::enumerate(table)) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return emitError() << "work_slices[" << i
                         << "] must be a dictionary of dimension key to slice "
                            "index";

    WorkSliceEntry entry;
    for (NamedAttribute named : dict) {
      auto idx = dyn_cast<IntegerAttr>(named.getValue());
      if (!idx || !idx.getType().isSignlessInteger(64))
        return emitError() << "work_slices[" << i << "]['"
                           << named.getName().strref()
                           << "'] must be an i64 slice index";
      if (idx.getInt() < 0)
        return emitError() << "work_slices[" << i << "]['"
                           << named.getName().strref()
                           << "'] is negative: " << idx.getInt();
      entry.emplace_back(named.getName().strref(), idx.getInt());
    }

    // Same key SET in every entry, checked as a sorted sequence because a
    // DictionaryAttr is already sorted by name -- so equality of the two
    // sequences is equality of the sets, with no set to build.
    SmallVector<StringRef> keys;
    for (auto &[key, _] : entry)
      keys.push_back(key);
    if (i == 0) {
      refKeys = keys;
    } else if (keys != refKeys) {
      auto diag = emitError() << "work_slices[" << i << "] has keys [";
      llvm::interleaveComma(keys, diag);
      diag << "], expected [";
      llvm::interleaveComma(refKeys, diag);
      return diag << "]: every entry describes the same grid, so the keys must "
                     "be identical";
    }

    for (auto &[key, index] : entry) {
      auto it = sliceCounts.find(key);
      if (it == sliceCounts.end())
        sliceCounts.insert({key, index + 1});
      else
        it->second = std::max(it->second, index + 1);
    }
    entries.push_back(std::move(entry));
  }
  return success();
}

LogicalResult MakeDistributedDescriptorOp::verify() {
  auto tensorTy = cast<RankedTensorType>(getPartial().getType());
  unsigned rank = tensorTy.getRank();

  // (1) The partition table, through the shared reader.
  SmallVector<WorkSliceEntry> entries;
  llvm::MapVector<StringRef, int64_t> sliceCounts;
  auto emitError = [&]() { return this->emitOpError(); };
  if (failed(readWorkSliceTable(getWorkSlices(), entries, sliceCounts,
                                emitError)))
    return failure();

  // (2) `axes` is per TENSOR DIMENSION, so its length is the share's rank and
  // not the number of keys: a dimension the work was not divided on still needs
  // an entry, spelled "", or the mapping would be positional against a shorter
  // list and silently shift.
  ArrayAttr axes = getAxes();
  if (axes.size() != rank)
    return emitOpError() << "axes has " << axes.size() << " entries for a rank-"
                         << rank
                         << " share: one per tensor dimension, with \"\" for a "
                            "dimension the work was not divided on";

  llvm::SmallDenseSet<StringRef> named;
  for (auto [d, attr] : llvm::enumerate(axes)) {
    auto key = dyn_cast<StringAttr>(attr);
    if (!key)
      return emitOpError() << "axes[" << d << "] must be a string";
    if (key.getValue().empty())
      continue;
    if (!sliceCounts.contains(key.getValue()))
      return emitOpError() << "axes[" << d << "] names '" << key.getValue()
                           << "', which no work_slices entry carries";
    // (3) One dimension per key. Two dimensions divided along one key would ask
    // for a single slice index to pick a region in both, which is a projection
    // the table cannot express.
    if (!named.insert(key.getValue()).second)
      return emitOpError() << "axes names '" << key.getValue()
                           << "' twice: a partition key divides one dimension";
  }

  // (4) Every key must reach a dimension. A key the table carries and `axes`
  // does not name divides nothing, so the regions it distinguishes are equal --
  // which makes partitions collide rather than merely wasting a key.
  for (auto &[key, count] : sliceCounts) {
    (void)count;
    if (!named.contains(key))
      return emitOpError() << "work_slices carries key '" << key
                           << "', which axes does not name, so it divides no "
                              "dimension";
  }

  // (5) The result's block shape is what one `.load()` takes, so it is
  // `block_shape`, over the share's element type.
  ArrayRef<int64_t> blockShape = getBlockShape();
  auto blockTy = getResult().getType().getBlockType();
  if (blockShape.size() != blockTy.getRank())
    return emitOpError() << "block_shape has " << blockShape.size()
                         << " entries but the result descriptor's block type is "
                            "rank "
                         << blockTy.getRank();
  if (blockShape != blockTy.getShape())
    return emitOpError() << "block_shape does not match the result "
                            "descriptor's block shape";
  if (blockTy.getElementType() != tensorTy.getElementType())
    return emitOpError() << "the result descriptor's element type "
                         << blockTy.getElementType()
                         << " does not match the share's "
                         << tensorTy.getElementType();

  // (6) `block_shape` is the extent of one ACCESS, bounded by the COMPOSED extent
  // and not by the share's. Those differ, and admitting the difference is the
  // point: a share is one slice, so the compose grows it by the slice count, and
  // an access may legitimately take
  //
  //   less than a share   a relayout, reading the region I end up holding;
  //   exactly a share     the common case, one region per load;
  //   more than a share   a gather, spanning several partitions -- and at the
  //                       limit an all-reduce, taking the whole composed axis so
  //                       that a fold over it reduces across every core.
  //
  // Nothing else bounds it. Triton relates `block_shape` only to the descriptor's
  // own block type, never to the share; and `ktdp.construct_access_tile`'s
  // verifier checks ranks and maps but not extents against the view it is taken
  // on. So an access past the composed domain would verify everywhere and address
  // memory no partition holds, which makes this the only place to refuse it.
  //
  // Every dimension, not only the divided ones: an undivided dimension composes
  // to itself, so a block larger than the share there is out of bounds too.
  for (auto [d, attr] : llvm::enumerate(axes)) {
    auto key = cast<StringAttr>(attr).getValue();
    int64_t composed = tensorTy.getDimSize(d);
    if (!key.empty())
      composed *= sliceCounts.find(key)->second;
    if (blockShape[d] > composed)
      return emitOpError() << "block_shape[" << d << "] is " << blockShape[d]
                           << ", larger than the composed extent " << composed
                           << " on that dimension (the share's "
                           << tensorTy.getDimSize(d) << " times "
                           << (composed / tensorTy.getDimSize(d)) << " slices)";
  }

  return success();
}
} // namespace mlir::triton::tts
