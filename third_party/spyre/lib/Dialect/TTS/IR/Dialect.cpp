//===- Dialect.cpp - The tts dialect --------------------------------------===//
//
// The dialect registration, the structural checker for a `tts.tensor_layout`
// coordinate map, and the `verifyOperationAttribute` hook that gates the
// attribute on any op carrying it. The authoring op's own verifier is in
// Ops.cpp, and calls the same checker.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallVector.h"

#include "Dialect/TTS/IR/Dialect.cpp.inc"

namespace mlir::triton::tts {

namespace {

/// Admits inlining INTO a `tts.spyre_intrinsic` body. Tracing reaches the
/// registered fallback through a `tt.call` inside that body, and the `ttir`
/// stage's inliner asks the dialect of the region's parent op whether a
/// callee's body may land there; with no answer it keeps the call. Whether each
/// inlined op may move is still asked of that op's own dialect.
struct TTSInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return isa<SpyreIntrinsicOp>(dest->getParentOp());
  }
};

} // namespace

void TTSDialect::initialize() {
  // Ops and nothing else: no types, no attribute types. Most of what the
  // dialect is for is the attribute NAMES and the verifier hook those names
  // route to, neither of which is registered here.
  addOperations<
#define GET_OP_LIST
#include "Dialect/TTS/IR/Ops.cpp.inc"
      >();
  addInterfaces<TTSInlinerInterface>();
}

DictionaryAttr getSpyreIntrinsic(Operation *op) {
  return dyn_cast_or_null<DictionaryAttr>(
      op->getDiscardableAttr(TTSDialect::kSpyreIntrinsicAttrName));
}

StringRef getSpyreIntrinsicTarget(DictionaryAttr hint) {
  return hint.getAs<StringAttr>(TTSDialect::kSpyreIntrinsicTargetKey)
      .getValue();
}

int64_t getSpyreIntrinsicId(DictionaryAttr hint) {
  return hint.getAs<IntegerAttr>(TTSDialect::kSpyreIntrinsicIdKey).getInt();
}

int64_t getSpyreIntrinsicArity(DictionaryAttr hint) {
  return hint.getAs<IntegerAttr>(TTSDialect::kSpyreIntrinsicArityKey).getInt();
}

DenseI64ArrayAttr getSpyreIntrinsicOperands(DictionaryAttr hint) {
  return hint.getAs<DenseI64ArrayAttr>(TTSDialect::kSpyreIntrinsicOperandsKey);
}

DictionaryAttr getBodyCallSite(Block &body) {
  DictionaryAttr found;
  WalkResult walked = body.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::ConstantLike>())
      return WalkResult::advance();
    DictionaryAttr hint = getSpyreIntrinsic(op);
    if (!hint)
      return WalkResult::advance();
    if (!found)
      found = hint;
    else if (getSpyreIntrinsicId(found) != getSpyreIntrinsicId(hint))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walked.wasInterrupted() ? DictionaryAttr() : found;
}

void copyTTSAttrs(Operation *from, Operation *to) {
  StringRef prefix = TTSDialect::getDialectNamespace();
  for (NamedAttribute attr : from->getDiscardableAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name.starts_with(prefix) &&
        name.drop_front(prefix.size()).starts_with("."))
      to->setDiscardableAttr(attr.getName(), attr.getValue());
  }
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

  // How far does one step along each device axis move the host pointer? A splat
  // axis, or an axis over a logical dim of extent one, moves it nowhere. An
  // identity axis and a split's mod (lane) axis move it by the logical dim's own
  // host stride; a split's floordiv (stick) axis moves it by one whole stick,
  // that stride times the width. The mod axis, when there is one, is the last
  // axis and its width is the floordiv's divisor, so the stick stride is stated
  // directly from phys_arg.
  SmallVector<int64_t> strides(rank, 0);
  for (unsigned k = 0; k < rank; ++k) {
    int64_t d = physSrc[k];
    CoordOp c = static_cast<CoordOp>(physOp[k]);
    if (c == CoordOp::Splat || logSizes[d] == 1) {
      strides[k] = -1;
      continue;
    }
    strides[k] = logStrides[d];
    // A dynamic logical stride stays dynamic rather than being multiplied: the
    // product would overflow. Unreachable for anything in tree -- a dynamic dim
    // only survives applyCoordMap under mod or splat -- and the alternative is
    // signed-overflow UB rather than a number.
    if (c == CoordOp::FloorDiv && strides[k] != ShapedType::kDynamic)
      strides[k] *= physArg[k];
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
  // hardware -- and of the 6 insertions this performs across the fixtures, the 5
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
  // The floordiv divisor and mod modulus per logical dim, and the physical dims
  // carrying a mod or splat entry, for the stick-dim checks below.
  SmallVector<int64_t> floorDivArg(logicalRank, 0), modArg(logicalRank, 0);
  SmallVector<unsigned> stickDims;

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
      floorDivArg[src[k]] = arg[k];
      break;
    case CoordOp::Mod:
      ++numMod[src[k]];
      modArg[src[k]] = arg[k];
      stickDims.push_back(k);
      break;
    case CoordOp::Splat:
      ++numSplat[src[k]];
      stickDims.push_back(k);
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

  // The stick dim is the physical dim whose positions are the lanes of one
  // stick: a split's mod entry, or a splat entry. A layout has at most one, and
  // it is the innermost physical dim.
  if (stickDims.size() > 1)
    return emitError() << "tts.tensor_layout: a layout has at most one stick "
                          "dim (a mod or splat entry), got "
                       << stickDims.size();
  if (!stickDims.empty() && stickDims.front() != src.size() - 1)
    return emitError() << "tts.tensor_layout: the stick dim must be the last "
                          "physical dim, got it at phys dim "
                       << stickDims.front() << " of " << src.size();

  // A split's floordiv counts sticks of the width its mod counts lanes in.
  for (unsigned d = 0; d < logicalRank; ++d) {
    if (numFloorDiv[d] != 1 || numMod[d] != 1)
      continue;
    if (floorDivArg[d] != modArg[d])
      return emitError() << "tts.tensor_layout: logical dim " << d
                         << " is split with floordiv " << floorDivArg[d]
                         << " but mod " << modArg[d]
                         << "; the two must be the same stick width";
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
  auto emitError = [&]() { return op->emitError(); };

  if (name == kTensorLayoutAttrName) {
    // The layout describes the tensor a memory view addresses, so there is
    // nothing for it to mean anywhere else — and on the wrong op it would be
    // inert rather than wrong, which is the failure worth catching here.
    auto view = dyn_cast<mlir::ktdp::ConstructMemoryViewOp>(op);
    if (!view)
      return op->emitError("'")
             << kTensorLayoutAttrName
             << "' is only meaningful on a ktdp.construct_memory_view, which "
                "this op is not";

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

  // `tts.pin` is ours, and that is the whole of what this branch says. Its rules
  // are established BEFORE the attribute exists: the op's verifier checks the
  // fields an author wrote, and LowerTTSMarkers checks the move -- that the value
  // resolves to a carrier, that the carrier can hold one annotation, and that no
  // other marker already claimed it. The attribute is then built from fields that
  // have already passed, so re-deriving those rules out of the dictionary here
  // would establish nothing and would state them a second place to drift.
  //
  // The branch cannot simply be absent: an unhandled `tts.` name falls through to
  // the refusal below, so without this the annotation would fail to verify the
  // moment it was attached.
  //
  // When something consumes this attribute it will read the dictionary back, and
  // being invocable on hand-written IR it will have to check the spelling it finds
  // -- as RewriteDescriptorLayoutGeneric's readCoordMap does for the layout, for
  // exactly that reason. That reader belongs with that consumer.
  if (name == kPinAttrName)
    return success();

  // The `tts.spyre_intrinsic` attribute is written by one pass,
  // LowerTTSMarkers, and READ by two in `spyrecode` --
  // FuseComputeAndDataMovement groups bodies by the hint's `id`, and
  // LowerSpyreIntrinsicCalls builds the op its `target` names from the call
  // operands its `operands` bind. So what one op can show is checked here,
  // which is what lets getSpyreIntrinsic hand those readers a dictionary they
  // need not re-check: the three required fields, each of its type, an optional
  // `operands` of the op's own operand count with every entry a call operand
  // index or -1, no other key, a `target` that is a registered op, and an op
  // that is not a constant, which LowerTTSMarkers never hints. Nothing about
  // other ops -- an `id` being unique, a call site being whole, its ops
  // agreeing on `target` and `arity`, every call operand being bound -- is
  // checkable from one op; FuseComputeAndDataMovement and
  // LowerSpyreIntrinsicCalls check those.
  if (name == kSpyreIntrinsicAttrName) {
    auto hint = dyn_cast<DictionaryAttr>(attribute.getValue());
    auto target =
        hint ? hint.getAs<StringAttr>(kSpyreIntrinsicTargetKey) : StringAttr();
    auto id =
        hint ? hint.getAs<IntegerAttr>(kSpyreIntrinsicIdKey) : IntegerAttr();
    auto arity =
        hint ? hint.getAs<IntegerAttr>(kSpyreIntrinsicArityKey) : IntegerAttr();
    Attribute operandsEntry =
        hint ? hint.get(kSpyreIntrinsicOperandsKey) : nullptr;
    auto operands = dyn_cast_or_null<DenseI64ArrayAttr>(operandsEntry);
    size_t expected = operandsEntry ? 4 : 3;
    if (!hint || hint.size() != expected || !target || !id ||
        !id.getType().isInteger(64) || !arity ||
        !arity.getType().isInteger(64) || (operandsEntry && !operands))
      return op->emitError("'")
             << kSpyreIntrinsicAttrName
             << "' must be a dictionary of a string '"
             << kSpyreIntrinsicTargetKey << "', an i64 '"
             << kSpyreIntrinsicIdKey << "', an i64 '" << kSpyreIntrinsicArityKey
             << "' and optionally an array<i64> '" << kSpyreIntrinsicOperandsKey
             << "', and nothing else, got " << attribute.getValue();
    if (!RegisteredOperationName::lookup(target.getValue(), getContext()))
      return op->emitError("'")
             << kSpyreIntrinsicAttrName << "' targets " << target
             << ", which is not a registered op";
    if (op->hasTrait<OpTrait::ConstantLike>())
      return op->emitError("'")
             << kSpyreIntrinsicAttrName << "' is not set on a constant";
    int64_t n = arity.getInt();
    if (n < 1)
      return op->emitError("'")
             << kSpyreIntrinsicAttrName << "' has " << kSpyreIntrinsicArityKey
             << " " << n << ", and a call site has at least one operand";
    if (!operands)
      return success();
    if (operands.size() != static_cast<int64_t>(op->getNumOperands()))
      return op->emitError("'")
             << kSpyreIntrinsicAttrName << "' has " << operands.size() << " '"
             << kSpyreIntrinsicOperandsKey
             << "' entries, one per operand of an op "
             << "with " << op->getNumOperands();
    for (auto [j, k] : llvm::enumerate(operands.asArrayRef()))
      if (k < -1 || k >= n)
        return op->emitError("'")
               << kSpyreIntrinsicAttrName << "' binds operand " << j
               << " to call operand " << k << ", and a call site of "
               << kSpyreIntrinsicArityKey << " " << n << " has operands 0 to "
               << n - 1 << ", or -1 for none";
    return success();
  }

  return op->emitError("attribute '")
         << name << "' is not one the tts dialect defines";
}

} // namespace mlir::triton::tts
