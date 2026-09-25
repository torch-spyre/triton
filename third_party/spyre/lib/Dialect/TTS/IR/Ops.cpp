//===- Ops.cpp - The tts dialect's ops ------------------------------------===//
//
// Two ops. `tensor_layout`'s verifier delegates rather than restates -- see the
// note on `tts::verifyTensorLayoutArrays` for why the rules have a single owner.
// `pin`'s rules are its own, and they are all STRUCTURAL: what shape an address
// expression may have, not what numbers it may hold. The numeric rules --
// scratchpad capacity, stick alignment -- need the device description, which this
// tree deliberately does not read (see `SpyreUtils.get_device_properties`), so
// they belong to `PlacePinnedValues`, which takes them as pass options.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
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

namespace {

/// The integer a value is, if it is a constant.
///
/// `m_ConstantInt` rather than `arith::ConstantOp`, so this admits any
/// ConstantLike op folding to an integer attribute. A pinned address written as
/// `BASE + pid * STRIDE` with `tl.constexpr` coefficients reaches us as
/// `arith.constant`s today; nothing here depends on that staying true.
std::optional<int64_t> matchConstantInt(Value v) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue();
  return std::nullopt;
}

/// The coefficient of a `tl.program_id(0)` term: `pid`, `pid * k` or `k * pid`.
///
/// Axis X only. The launch grid this backend distributes over is
/// one-dimensional -- `TTIRToKTIRPipelineOptions.grid` is what DistributeWork
/// divides and what `wk_slice_coord` indexes -- so X is the only axis whose
/// address set is a finite enumerable `{base + i*stride : i < grid}`. A Y or Z
/// term would have no bound to enumerate against, which is the whole reason the
/// form is restricted.
std::optional<int64_t> matchProgramIdTerm(Value v) {
  auto isPid = [](Value v) {
    auto pid = v.getDefiningOp<triton::GetProgramIdOp>();
    return pid && pid.getAxis() == triton::ProgramIDDim::X;
  };

  if (isPid(v))
    return 1;
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    if (isPid(mul.getLhs()))
      return matchConstantInt(mul.getRhs());
    if (isPid(mul.getRhs()))
      return matchConstantInt(mul.getLhs());
  }
  return std::nullopt;
}

} // namespace

/// Addition is commutative and so is the multiply inside the program-id term, so
/// all of `B + pid*S`, `pid*S + B`, `S*pid + B` and a bare `pid` match. A missing
/// base is 0 and a missing stride is 0, which is what makes `(base, stride)` a
/// description of the address SET rather than of the syntax that spelled it.
///
/// CONSERVATIVE, in one direction worth naming. An expression with two
/// program-id terms -- `pid*256 + pid*512` -- describes an admissible set,
/// `pid*768`, and is refused here only because this does not sum strides. The
/// hazard the refusal avoids is the opposite mistake, matching one term and
/// silently ignoring the other, which would leave the pin occupying a range
/// nobody computed; summing would avoid it too. The frontend folds `tl.constexpr`
/// coefficients in Python, so `BASE + pid*STRIDE` arrives as exactly one
/// multiply and one add and nothing in tree spells two terms -- which is why the
/// cheaper of the two is the one implemented.
LogicalResult matchPinAddress(Value addr, int64_t &base, int64_t &stride) {
  if (!addr)
    return failure();

  if (auto c = matchConstantInt(addr)) {
    base = *c;
    stride = 0;
    return success();
  }
  if (auto s = matchProgramIdTerm(addr)) {
    base = 0;
    stride = *s;
    return success();
  }
  if (auto add = addr.getDefiningOp<arith::AddIOp>()) {
    Value lhs = add.getLhs(), rhs = add.getRhs();
    for (auto [constSide, pidSide] : {std::pair<Value, Value>{lhs, rhs},
                                      std::pair<Value, Value>{rhs, lhs}}) {
      auto c = matchConstantInt(constSide);
      if (!c)
        continue;
      if (auto s = matchProgramIdTerm(pidSide)) {
        base = *c;
        stride = *s;
        return success();
      }
    }
  }
  return failure();
}

LogicalResult PinOp::verify() {
  // (1) The memory space, spelled as a string because this dialect defines no
  // attribute type. Checking it against ktdp's enum is what makes the string as
  // safe as the attribute would have been, and the lowering builds the real
  // `#ktdp.memory_space` from the same symbolization -- but only ONE of the two
  // kinds is admitted.
  StringRef space = getMemorySpace();
  auto kind = mlir::ktdp::symbolizeMemorySpaceKind(space);
  if (!kind)
    return emitOpError() << "unknown memory space '" << space
                         << "': expected 'ct_local'";

  // (2) `global` is refused rather than admitted and then found unplaceable.
  // lx-placement.md's first assumption puts HBM intermediates outside a pin
  // altogether: an author who wants one "writes a tl.make_tensor_descriptor for
  // it, with a tl.spyre_tensor_layout, and an explicit store and load", which
  // reaches a binary today. Nothing in this tree allocates an anonymous global
  // buffer, so admitting the kind here would only make a surface that never
  // compiles -- and would say so with a diagnostic about the missing ADDRESS,
  // which is not what such an author got wrong.
  //
  // The parameter stays a memory space rather than collapsing into the op's
  // name, so that the surface does not change shape if `global` ever becomes
  // placeable.
  if (*kind != mlir::ktdp::MemorySpaceKind::ct_local)
    return emitOpError()
           << "memory space '" << space
           << "' cannot be pinned: only 'ct_local' is, since an intermediate in "
              "HBM is written as a descriptor with an explicit store and load";

  // A block argument is deliberately NOT refused. A `tensor` value is a value
  // whether an op or a block argument defines it -- the same state a
  // `ktdp.load` result is in -- so pinning one is the same request, and nothing
  // here reads the producing op (the store goes at the pin, and dominance is
  // measured against it). The "only a value the author named" rule the design
  // states is enforced by the surface, where an anonymous subexpression simply
  // has no name to pass, and needs no restatement as an op rule.

  // (3) The address expression's shape. Rejected here rather than in the
  // frontend because this is where the expression is: the frontend hands over a
  // value it has already built, and a module parsed from text never went through
  // the frontend at all.
  if (getAddress()) {
    int64_t base = 0, stride = 0;
    if (failed(matchPinAddress(getAddress(), base, stride)))
      return emitOpError()
             << "address must be a constant or `base + tl.program_id(0) * "
                "stride` with constant coefficients; anything else leaves the "
                "pinned range unenumerable, so capacity and disjointness "
                "cannot be checked";
  }

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
