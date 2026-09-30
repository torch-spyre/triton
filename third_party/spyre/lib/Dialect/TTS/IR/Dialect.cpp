//===- Dialect.cpp - The tts dialect --------------------------------------===//
//
// The dialect registration, the structural checker for a `tts.tensor_layout`
// coordinate map, and the `verifyOperationAttribute` hook that gates the
// attribute on any op carrying it. The authoring op's own verifier is in
// Ops.cpp, and calls the same checker.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include "Dialect/TTS/IR/Dialect.cpp.inc"

namespace mlir::triton::tts {

void TTSDialect::initialize() {
  // One op and nothing else: no types, no attribute types. Most of what the
  // dialect is for is the attribute NAME and the verifier hook that name routes
  // to, neither of which is registered here.
  addOperations<
#define GET_OP_LIST
#include "Dialect/TTS/IR/Ops.cpp.inc"
      >();
}

std::optional<CoordOp> symbolizeCoordOp(int64_t code) {
  switch (static_cast<CoordOp>(code)) {
  case CoordOp::Identity:
  case CoordOp::FloorDiv:
  case CoordOp::Mod:
  case CoordOp::Splat:
    return static_cast<CoordOp>(code);
  }
  return std::nullopt;
}

std::optional<int64_t> applyStatic(int64_t logical, CoordOp op, int64_t arg) {
  switch (op) {
  case CoordOp::Identity:
    return (logical == ShapedType::kDynamic) ? std::nullopt
                                             : std::optional<int64_t>(logical);
  case CoordOp::FloorDiv:
    if (logical == ShapedType::kDynamic)
      return std::nullopt;
    // Ceiling, not floor -- see the header. arg == 0 is refused rather than
    // divided by; verifyTensorLayoutArrays already requires arg > 0 here, so
    // this only catches a caller that skipped it.
    return arg == 0 ? std::nullopt
                    : std::optional<int64_t>((logical + arg - 1) / arg);
  case CoordOp::Mod:
  case CoordOp::Splat:
    // Neither reads `logical`: both give exactly `arg` extents.
    return arg;
  }
  return std::nullopt;
}

bool applyCoordMap(ArrayRef<int64_t> logSizes, ArrayRef<int64_t> physSrc,
                   ArrayRef<int64_t> physOp, ArrayRef<int64_t> physArg,
                   SmallVectorImpl<int64_t> &out) {
  out.resize(physSrc.size());
  for (unsigned k = 0, e = physSrc.size(); k < e; ++k) {
    auto sz = applyStatic(logSizes[physSrc[k]], static_cast<CoordOp>(physOp[k]),
                          physArg[k]);
    if (!sz)
      return false;
    out[k] = *sz;
  }
  return true;
}

bool evaluateDeviceLayout(ArrayRef<int64_t> logSizes,
                          ArrayRef<int64_t> logStrides,
                          ArrayRef<int64_t> physSrc, ArrayRef<int64_t> physOp,
                          ArrayRef<int64_t> physArg,
                          SmallVectorImpl<int64_t> &deviceSize,
                          SmallVectorImpl<int64_t> &strideMap) {
  // The two logical arrays are indexed by the same physSrc[k] below.
  // getDescriptorLogicalLayout always produces them parallel; a caller reading a
  // shape and a stride list from different places may not.
  if (logSizes.size() != logStrides.size())
    return false;

  SmallVector<int64_t> extents;
  if (!applyCoordMap(logSizes, physSrc, physOp, physArg, extents))
    return false;
  unsigned rank = extents.size();

  // How far does one step along each device axis move the host pointer? Innermost
  // to outermost, so that an outer half of a partitioned dim can multiply up by
  // whatever the inner half turned out to be: `running[d]` is the stride the next
  // axis over logical dim d will take, seeded with the dim's own host stride.
  SmallVector<int64_t> strides(rank, 0);
  SmallVector<int64_t> running(logStrides.begin(), logStrides.end());
  for (int k = (int)rank - 1; k >= 0; --k) {
    int64_t d = physSrc[k];
    if (static_cast<CoordOp>(physOp[k]) == CoordOp::Splat || logSizes[d] == 1) {
      strides[k] = -1;
      continue;
    }
    strides[k] = running[d];
    // A dynamic logical stride stays dynamic rather than being multiplied: the
    // product would overflow. Unreachable for anything in tree -- a dynamic dim
    // only survives applyCoordMap under mod or splat -- and the alternative is
    // signed-overflow UB rather than a number.
    running[d] = strides[k] == ShapedType::kDynamic
                     ? ShapedType::kDynamic
                     : strides[k] * extents[k];
  }

  deviceSize.assign(extents.begin(), extents.end());
  strideMap.assign(strides.begin(), strides.end());

  // torch-spyre's DMA reads the axis THIRD FROM THE END as the tile-count half of
  // the stick split, and overwrites what it matched there. If ours is something
  // else, that read corrupts a real dimension -- so put a harmless unit axis where
  // it looks. Three shapes are harmless already; see the header for what each
  // rests on.
  if (rank < 2)
    return true;
  unsigned p = rank > 2 ? rank - 3 : 0;
  unsigned last = rank - 1;
  bool harmless =
      strides[p] == -1 || extents[p] == 1 ||
      (physSrc[p] == physSrc[last] &&
       static_cast<CoordOp>(physOp[p]) == CoordOp::FloorDiv &&
       static_cast<CoordOp>(physOp[last]) == CoordOp::Mod);
  if (harmless)
    return true;

  // Second from the end, so that after the insertion shifts everything later the
  // unit axis is the one third from the end -- where the DMA will look. Inserting
  // at `p` instead would leave that read pointing at a real axis.
  //
  // UNTESTED for rank >= 4, and this is the line to test first when a rank-4
  // layout reaches the device, because nothing in tree can tell the two positions
  // apart: they COINCIDE at rank 2 -- the splat case, the only one verified on
  // hardware -- and of the 8 insertions this performs across the fixtures, the 7
  // where they differ are all rank >= 4 (`matmul__bmm_*`,
  // `reduce__middle_axis_spyre_stick`), none of which declares
  // `compiles_to_binary`, so no tier reaches them. Confirmed by mutation:
  // `at = p` leaves both suites green.
  //
  // The failure mode is silent, which is why it needs a test rather than a
  // comment: a wrong position leaves the DMA's read on a real axis, `dcsi_sizes`
  // stays all ones and one element moves instead of the full extent, with no check
  // firing.
  unsigned at = rank - 2;
  deviceSize.insert(deviceSize.begin() + at, 1);
  strideMap.insert(strideMap.begin() + at, -1);
  return true;
}

LogicalResult verifyTensorLayoutArrays(
    ArrayRef<int64_t> src, ArrayRef<int64_t> op, ArrayRef<int64_t> arg,
    unsigned logicalRank,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  // The three arrays are parallel: every consumer indexes all of them with the
  // same physical-dim index k (see applyCoordMap).
  if (src.size() != op.size() || src.size() != arg.size())
    return emitError() << "tts.tensor_layout: phys_src, phys_op and phys_arg "
                          "must have the same number of entries, got "
                       << src.size() << ", " << op.size() << " and "
                       << arg.size();

  // A rank-0 coordinate map describes no physical dim, so there is no layout in
  // it to read; every consumer seeds itself from some physical dim and
  // immediately dereferences it.
  if (src.empty())
    return emitError()
           << "tts.tensor_layout: must describe at least one physical dim";

  // Tallies per logical dim, for the repeated-dim check below.
  SmallVector<int64_t> numIdentity(logicalRank, 0), numFloorDiv(logicalRank, 0),
      numMod(logicalRank, 0), numSplat(logicalRank, 0),
      numTotal(logicalRank, 0);

  for (unsigned k = 0; k < src.size(); ++k) {
    // Consumers index the logical shape/stride arrays with phys_src[k].
    if (src[k] < 0 || src[k] >= (int64_t)logicalRank)
      return emitError() << "tts.tensor_layout: phys_src[" << k
                         << "] must be in [0, " << logicalRank << "), got "
                         << src[k];

    // phys_op[k] is static_cast to a CoordOp enum and switched on without a
    // default; an unknown code leaves the derived coordinate expression unset.
    // Through symbolizeCoordOp, and the codes in the message through the enum,
    // so that the numbering is stated once -- in CoordOp -- rather than spelled
    // as literals a few lines from the enum that owns it.
    std::optional<CoordOp> coordOp = symbolizeCoordOp(op[k]);
    if (!coordOp)
      return emitError()
             << "tts.tensor_layout: phys_op[" << k << "] must be "
             << static_cast<int64_t>(CoordOp::Identity) << " (identity), "
             << static_cast<int64_t>(CoordOp::FloorDiv) << " (floordiv), "
             << static_cast<int64_t>(CoordOp::Mod) << " (mod) or "
             << static_cast<int64_t>(CoordOp::Splat) << " (splat), got "
             << op[k];

    // phys_arg is the floordiv divisor / mod modulus / splat lane count; 0
    // divides by zero when deriving physical extents and yields a zero-width
    // stick or a zero-lane splat.
    if (*coordOp != CoordOp::Identity && arg[k] <= 0)
      return emitError() << "tts.tensor_layout: phys_arg[" << k
                         << "] must be > 0 for a floordiv/mod/splat dim, got "
                         << arg[k];

    ++numTotal[src[k]];
    switch (*coordOp) {
    case CoordOp::Identity:
      ++numIdentity[src[k]];
      break;
    case CoordOp::FloorDiv:
      ++numFloorDiv[src[k]];
      break;
    case CoordOp::Mod:
      ++numMod[src[k]];
      break;
    case CoordOp::Splat:
      ++numSplat[src[k]];
      break;
    }
  }

  // A logical dim may legitimately span two physical dims in two ways:
  //
  //   Stick split: one floordiv (stick index) + one mod (lane). The two are the
  //   components of the dim's multi-index over the basis (ceildiv(extent, W), W),
  //   so exactly one of them carries the dim's position within a stick and
  //   exactly one carries which stick. Any other repetition puts two dims
  //   carrying the same role in the physical layout, and a consumer assigning a
  //   unique target position per role has no position left to give the second.
  //
  //   Splat re-stick: one identity (the dim itself, kept whole) + one splat (the
  //   same dim replicated across a fresh lane axis). That is the reduce-on-stick
  //   output layout, where a rank-1 logical result physicalizes to (dim, lanes).
  //   Its two entries do not partition the dim between them the way a split
  //   does, so it is a distinct pairing rather than a relaxation of the one
  //   above: mixing halves across the two (floordiv + splat, identity + mod)
  //   stays rejected.
  for (unsigned d = 0; d < logicalRank; ++d) {
    if (numTotal[d] < 2)
      continue;
    if (numTotal[d] == 2 && numFloorDiv[d] == 1 && numMod[d] == 1)
      continue;
    if (numTotal[d] == 2 && numIdentity[d] == 1 && numSplat[d] == 1)
      continue;
    return emitError()
           << "tts.tensor_layout: logical dim " << d << " appears in "
           << numTotal[d]
           << " physical dims; a repeated logical dim is only valid as a stick "
              "split (exactly one floordiv entry and one mod entry) or a splat "
              "re-stick (exactly one identity entry and one splat entry), got "
           << numIdentity[d] << " identity, " << numFloorDiv[d] << " floordiv, "
           << numMod[d] << " mod, " << numSplat[d] << " splat";
  }

  return success();
}

LogicalResult readTensorLayoutArrays(
    Attribute value, ArrayRef<int64_t> &physSrc, ArrayRef<int64_t> &physOp,
    ArrayRef<int64_t> &physArg,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  auto dict = dyn_cast<DictionaryAttr>(value);
  if (!dict)
    return emitError() << "tts.tensor_layout: must be a dictionary of "
                          "phys_src, phys_op and phys_arg";

  // Read by name, then check the count, so a missing entry is named and an
  // unexpected one is not silently tolerated. An unrecognised key is rejected
  // rather than ignored because the failure mode of ignoring it is a typo that
  // leaves the layout half-stated and still verifying.
  ArrayRef<int64_t> *slots[] = {&physSrc, &physOp, &physArg};
  StringRef names[] = {TTSDialect::kPhysSrcName, TTSDialect::kPhysOpName,
                       TTSDialect::kPhysArgName};
  for (auto [name, slot] : llvm::zip_equal(names, slots)) {
    Attribute entry = dict.get(name);
    if (!entry)
      return emitError() << "tts.tensor_layout: missing '" << name << "' entry";
    auto array = dyn_cast<DenseI64ArrayAttr>(entry);
    if (!array)
      return emitError() << "tts.tensor_layout: '" << name
                         << "' must be a dense i64 array";
    *slot = array.asArrayRef();
  }
  if (dict.size() != std::size(names))
    return emitError() << "tts.tensor_layout: expected exactly the entries "
                          "phys_src, phys_op and phys_arg, got "
                       << dict.size() << " entries";
  return success();
}

LogicalResult TTSDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attribute) {
  StringRef name = attribute.getName().strref();
  if (name != kTensorLayoutAttrName)
    return op->emitError("attribute '")
           << name << "' is not one the tts dialect defines";

  // The layout describes the tensor a memory view addresses, so there is
  // nothing for it to mean anywhere else — and on the wrong op it would be
  // inert rather than wrong, which is the failure worth catching here.
  auto view = dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(op);
  if (!view)
    return op->emitError("'")
           << kTensorLayoutAttrName
           << "' is only meaningful on a ktdp.construct_memory_view, which "
              "this op is not";

  auto emitError = [&]() { return op->emitError(); };
  ArrayRef<int64_t> physSrc, physOp, physArg;
  if (failed(readTensorLayoutArrays(attribute.getValue(), physSrc, physOp,
                                    physArg, emitError)))
    return failure();

  // The logical rank is the view's own rank. That is the same rank the op form
  // measured against — it read the descriptor's BLOCK type, whose extents
  // differ from the view's but whose rank does not.
  auto memrefTy = cast<MemRefType>(view.getResult().getType());
  return verifyTensorLayoutArrays(physSrc, physOp, physArg,
                                  memrefTy.getRank(), emitError);
}

} // namespace mlir::triton::tts
