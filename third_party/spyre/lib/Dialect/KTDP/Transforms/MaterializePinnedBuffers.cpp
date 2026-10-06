//===- MaterializePinnedBuffers.cpp - Honour a tts.pin annotation ---------===//
//
// A `tts.pin` says where an intermediate's buffer lives. `LowerTTSMarkers` has
// already moved that request onto the op PRODUCING the value; honouring it means
// building the buffer and routing the value's uses through it -- a memory view in
// the named space, a store of the value, and a load per use.
//
// The annotation is read here and nowhere else. That is the same division
// `RewriteDescriptorLayoutGeneric` has with `tts.tensor_layout`: the dialect owns
// the attribute's NAME and the fields' rules, and a reader for the dictionary
// belongs with the consumer, written against what that consumer needs.
//
// Two shapes worth naming, because both could plausibly have gone the other way:
//
//   * ONE VIEW, shared by the store and every load, with a fresh access tile per
//     access. That is what LowerDescriptorMemory does for a descriptor -- the view
//     is built once and `getDescriptorMemView` hands the same value to every load
//     and store of it -- and that path runs on hardware. lx-placement.md
//     contemplates a view per group instead, on the grounds that "nothing may be
//     reachable from two groups"; if that constraint is real it is not the one
//     descriptors already violate, so this follows the shape that is known to work
//     rather than the stricter one that is not.
//
//   * The buffer is the value's OWN shape, row-major. A pin states no layout --
//     the buffer is the compiler's, so there is nothing for an author to agree
//     with -- and the physical type is the producing compute's result type.
//
// What this pass does not do: place the intermediates nobody pinned. Every
// compute-to-compute edge needs a buffer whether or not the author could name it,
// and that is the whole of lx-placement.md's proposal; this pass does only what an
// annotation asked for.
//
// One check the earlier marker-consuming version of this pass had is GONE, and its
// absence is the point rather than an omission. That version refused a use of the
// pinned value that the pin did not dominate, because a `tts.pin` op could sit
// BELOW a use -- ordinary Python reaches that shape. An annotation has no position
// of its own: it rides on the op defining the value, every use of a value is
// dominated by its definition, and the store goes directly after it. So the
// position that made the refusal possible cannot exist, and DominanceInfo is not
// needed to establish it.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"

#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_MATERIALIZEPINNEDBUFFERS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

using mlir::triton::tts::TTSDialect;

//===----------------------------------------------------------------------===//
// The numeric obligations on a pinned offset
//===----------------------------------------------------------------------===//

/// The half-open BYTE range a pin occupies, measured from the base of this
/// kernel's scratchpad allocation.
///
/// Bytes and not element indices, although the offset is an element index and the
/// buffer is described in elements. Two pins of different dtypes have element
/// ranges that say nothing about each other: elements [0, 64) of an f32 value and
/// elements [64, 128) of an f16 one do not intersect as indices and do overlap as
/// memory, at bytes [0, 256) against [128, 256). Comparing indices would have
/// missed that silently, so the comparison is done in the unit the scratchpad is
/// actually addressed in.
struct ByteRange {
  int64_t lo = 0;
  int64_t hi = 0; // exclusive

  /// A range that covers no byte, which is one meaning and no longer two. It used
  /// to also stand for "the byte size is not known", which a sub-byte element type
  /// produced -- and sharing one guard made that case exempt from the capacity and
  /// overlap rules in silence. An unknown size is now refused in `checkOffset`, so
  /// what reaches here is always a real size and `hi == lo` always means a value
  /// with no elements. Such a value cannot overlap anything, which is why this is
  /// still a reason to skip the comparison.
  bool occupiesNothing() const { return hi <= lo; }
};

//===----------------------------------------------------------------------===//
// The layout a pin's buffer inherits
//===----------------------------------------------------------------------===//

/// A coordinate map, in the three parallel arrays `tts.tensor_layout` spells one
/// as. Copied rather than borrowed, because the search below projects one and
/// compares two.
struct Layout {
  SmallVector<int64_t> src, op, arg;

  bool operator==(const Layout &other) const {
    return src == other.src && op == other.op && arg == other.arg;
  }

  DictionaryAttr toAttr(Builder &b) const {
    return b.getDictionaryAttr(
        {b.getNamedAttr(TTSDialect::kPhysSrcName, b.getDenseI64ArrayAttr(src)),
         b.getNamedAttr(TTSDialect::kPhysOpName, b.getDenseI64ArrayAttr(op)),
         b.getNamedAttr(TTSDialect::kPhysArgName,
                        b.getDenseI64ArrayAttr(arg))});
  }
};

/// The layout the view behind `access` states, carried onto the dims of the tensor
/// the search started from.
///
/// `toStart[d]` is the starting tensor's dim that this access's logical dim `d`
/// carries, or -1 when it carries none -- which is how a dim-dropping op is walked
/// through. The carry is a DROP by `src`: a physical dim names exactly one logical
/// dim, so removing the entries of a dim that survives into nothing leaves a
/// coordinate map over the dims that do, with the survivors renumbered.
///
/// `out` stays empty, with success, when there is nothing to read or nothing sound
/// to carry: `access` is not a load or a store, the view behind it carries no
/// layout, the carry would have to drop a dim that was stick-SPLIT rather than named
/// whole, or it leaves no entry at all (a full reduction, whose rank-0 result no
/// layout describes).
/// Failure means the annotation is there and malformed, and a diagnostic has been
/// emitted -- through the dialect's own checker, so this pass states none of those
/// rules itself.
static LogicalResult layoutThrough(Operation *access, ArrayRef<int64_t> toStart,
                                   std::optional<Layout> &out) {
  Operation *view = mlir::triton::ktdp::viewBehindAccess(access);
  if (!view)
    return success();
  Attribute attr = view->getAttr(TTSDialect::kTensorLayoutAttrName);
  if (!attr)
    return success();

  auto anchor = [&]() { return view->emitError(); };
  ArrayRef<int64_t> src, op, arg;
  if (failed(mlir::triton::tts::readTensorLayoutArrays(attr, src, op, arg,
                                                       anchor)))
    return failure();

  Layout carried;
  for (auto [s, o, a] : llvm::zip_equal(src, op, arg)) {
    if (s < 0 || s >= (int64_t)toStart.size() || toStart[s] < 0) {
      // A dropped dim is only droppable while it was named WHOLE. Dropping a dim
      // that was stick-SPLIT would leave a map with no stick structure over it,
      // and that is not what the result of such a reduction wants: reducing away
      // a split dim destroys the stick structure, so the output is re-stuck by a
      // splat instead -- `phys_op [identity, splat]` over one `phys_src`, which
      // replicates the surviving dim across a stick's lanes. See
      // RewriteDescriptorLayoutGeneric/rebuild-reduction.mlir, case 2.
      //
      // A splat is not something a carry can produce: nothing upstream states the
      // width it would broadcast over. So this is where the walk stops rather than
      // where it guesses -- and the layout for such a buffer can still arrive from
      // the other direction, where a store of the re-stuck result states it.
      if (o != (int64_t)mlir::triton::tts::CoordOp::Identity) {
        out.reset();
        return success();
      }
      continue;
    }
    carried.src.push_back(toStart[s]);
    carried.op.push_back(o);
    carried.arg.push_back(a);
  }
  if (carried.src.empty())
    return success();

  // The carried map is checked rather than trusted. A drop by `src` cannot break
  // the pairing rules -- a split's two halves name the same logical dim, so they
  // leave together or stay together -- but this says so instead of arguing it, and
  // it is also what checks the renumbering against the pinned value's own rank.
  if (failed(mlir::triton::tts::verifyTensorLayoutArrays(
          carried.src, carried.op, carried.arg, toStart.size(), anchor)))
    return failure();

  out = std::move(carried);
  return success();
}

//===----------------------------------------------------------------------===//
// Reading the annotation
//===----------------------------------------------------------------------===//

/// One `tts.pin` dictionary, as this pass needs it, with everything the pass
/// derives from it on the way to building the buffer.
///
/// `range` lives here rather than in a parallel vector kept in step by index,
/// which is what the two used to be: a pin and the bytes it occupies are one
/// fact, and the overlap comparison needs them together.
struct Pin {
  /// The op CARRYING the annotation, which is also the op producing the value.
  Operation *op = nullptr;
  mlir::ktdp::MemorySpaceAttr space;
  /// Element index from the base of this kernel's scratchpad allocation.
  int64_t offset = 0;
  /// The single result the annotation is about, and the value to be stored.
  Value value;
  RankedTensorType type;
  /// What the buffer's view is annotated with, from `LayoutSearch`. Absent when no
  /// neighbour states one, which leaves the buffer logical -- the same thing an
  /// unannotated descriptor leaves behind, and the author's responsibility in the
  /// same way.
  std::optional<Layout> layout;
  /// Filled by `checkOffset`, which is also where it is checked.
  ByteRange range;
};

/// Read the `tts.pin` attribute on `op`, or diagnose and fail.
///
/// Every rule here has been checked once already, by `tts.pin`'s verifier, before
/// `LowerTTSMarkers` built this dictionary out of the op -- and each is checked again
/// anyway, because an ATTRIBUTE reaching this pass need never have been an op. This
/// pass runs on hand-written IR in its own lit tests, so a malformed dictionary is a
/// real input and a missing or wrong entry is a diagnostic rather than an assertion.
/// That is why the duplication is deliberate: the op's verifier speaks for what an
/// author wrote, and this speaks for what the pass is handed.
static LogicalResult readPin(Operation *op, Pin &pin) {
  pin.op = op;

  auto dict = dyn_cast<DictionaryAttr>(op->getAttr(TTSDialect::kPinAttrName));
  if (!dict)
    return op->emitError() << "'" << TTSDialect::kPinAttrName
                           << "' is not a dictionary; LowerTTSMarkers writes one";

  // The name becomes the enum HERE, which is the whole reason the annotation holds a
  // string. `tts` must not construct a `#ktdp.memory_space`: the marker op it builds
  // that dictionary from is created during tracing, and loading ktdp there loads
  // `func`, whose promised DialectInlinerInterface nothing registers, which aborts
  // the `ttir` stage's Inliner. This pass is in `spyrecode` and declares ktdp a
  // dependent dialect, so here the attribute is ordinary to build -- and it has to
  // be built anyway, since a memory view takes one.
  auto spaceName =
      dyn_cast_or_null<StringAttr>(dict.get(TTSDialect::kMemorySpaceName));
  if (!spaceName)
    return op->emitError() << "'" << TTSDialect::kPinAttrName << "' has no '"
                           << TTSDialect::kMemorySpaceName
                           << "' entry naming a memory space";
  auto kind = mlir::ktdp::symbolizeMemorySpaceKind(spaceName.getValue());
  if (!kind)
    return op->emitError() << "pinned memory space '" << spaceName.getValue()
                           << "' is not a ktdp memory space kind";

  // And that the kind is one a pin may name, which existing is not. `global` is the
  // reachable case and the reason the two checks are separate: it is a real ktdp
  // kind, so symbolizing accepts it, and it still cannot be pinned -- an
  // intermediate in HBM is written as a tl.make_tensor_descriptor with an explicit
  // store and load, and nothing here allocates an anonymous device buffer.
  //
  // Re-checked here although `tts.pin`'s verifier says it too, for the reason every
  // other rule in this function is re-checked: an attribute written by hand never
  // passed that verifier, and this pass runs on hand-written IR in its own tests.
  if (*kind != mlir::ktdp::MemorySpaceKind::ct_local)
    return op->emitError() << "pinned memory space '" << spaceName.getValue()
                           << "' cannot be pinned: only 'ct_local' is";

  // No ct_id: a pin means the scratchpad of whichever core is running the kernel,
  // never a named peer's, which is also why the annotation carries no such field to
  // read. -1 is what every site in this tree that constructs one passes.
  pin.space = mlir::ktdp::MemorySpaceAttr::get(op->getContext(), *kind,
                                               /*ct_id=*/-1);

  // Required HERE and optional in the attribute, which is the same asymmetry
  // `tts.pin` itself carries and has the same reason: the design's baseline is that
  // the compiler places every intermediate and a pin only overrides where, but
  // nothing in this tree allocates a buffer that is not a kernel argument, so an
  // offsetless pin names no location a view could be built at.
  Attribute offsetAttr = dict.get(TTSDialect::kOffsetName);
  if (!offsetAttr)
    return op->emitError() << "pinned value has no offset, so there is no "
                              "location to build its buffer at";

  // A SIGNLESS i32, which is what `tts.pin` declares the offset to be
  // (`OptionalAttr<I32Attr>`). Said here as one predicate rather than assumed, and
  // the reason is stronger than the rest of this function's: `IntegerAttr::getInt()`
  // does not merely answer wrongly for the other spellings, it ASSERTS -- "must be
  // signless integer" for a `ui32`, "Too many bits for int64_t" for anything wider
  // than 64 -- so a hand-written dictionary could crash the pass. A `FloatAttr` was
  // worse than a crash: it reported a MISSING offset, which it is not.
  //
  // It also bounds the offset, which is what makes the range arithmetic in
  // `checkOffset` representable rather than merely checked.
  auto offset = dyn_cast<IntegerAttr>(offsetAttr);
  if (!offset || !offset.getType().isSignlessInteger(32))
    return op->emitError() << "a pinned offset must be a signless i32, and this "
                              "one is " << offsetAttr;
  pin.offset = offset.getInt();

  // The op's operand constraint said this for an authored pin -- a statically
  // shaped tensor, one result. An annotation on an op reaches here without having
  // passed that, so it is said again rather than assumed.
  if (op->getNumResults() != 1)
    return op->emitError() << "a pin annotation is one buffer for one value, and "
                              "this op has " << op->getNumResults() << " results";
  pin.value = op->getResult(0);
  pin.type = dyn_cast<RankedTensorType>(pin.value.getType());
  if (!pin.type || !pin.type.hasStaticShape())
    return op->emitError() << "a pinned value must be a statically shaped "
                              "tensor, and this one is " << pin.value.getType();

  return success();
}

//===----------------------------------------------------------------------===//
// Deciding that layout
//===----------------------------------------------------------------------===//

/// The search for the layout a pin's buffer carries, out from the pinned value to
/// the nearest annotated access.
///
/// A pin's buffer is a memory view like any other and should leave `spyrecode`
/// physical; what is particular about it is only that the AUTHOR cannot annotate
/// it, since it is the compiler's buffer. So the compiler decides it, the way a
/// `linalg.generic` has its domain decided: from what is next to it.
///
/// Both directions, and they are not symmetric. Backward -- towards the value's
/// producers -- may pass through an op that DROPS dims, because the layout found
/// upstream describes a superset of this tensor's dims and the surplus entries can
/// be dropped. Forward may not: a layout found downstream of a dim-dropping op
/// describes fewer dims than the pinned value has, and the missing entries cannot
/// be invented. So forward steps only where the shape is unchanged.
struct LayoutSearch {
  const Pin &pin;

  /// Each distinct layout found, with the access that stated it, so that a
  /// disagreement can name both sides.
  SmallVector<std::pair<Layout, Operation *>> found;
  /// One set per direction. Shared, they would let one walk's visit suppress the
  /// other's, which loses a candidate rather than merely repeating work.
  SmallPtrSet<Operation *, 8> seenBack, seenFwd;

  LogicalResult run() {
    SmallVector<int64_t> identity;
    identity.reserve(pin.type.getRank());
    for (int64_t d = 0, e = pin.type.getRank(); d < e; ++d)
      identity.push_back(d);
    if (failed(backward(pin.value, identity)))
      return failure();
    return forward(pin.value);
  }

  /// The layout to use, or nullopt when nothing stated one. Refuses a
  /// disagreement rather than choosing: a buffer has one layout, and its extent is
  /// what the author's offset is denominated in, so picking a side would make the
  /// footprint depend on which neighbour the walk reached first.
  LogicalResult decide(std::optional<Layout> &out) {
    if (found.empty())
      return success();
    if (found.size() > 1) {
      InFlightDiagnostic diag =
          pin.op->emitError()
          << "a pinned value's neighbours state different layouts, so the "
             "layout of its buffer is not determined";
      for (auto &[layout, access] : found)
        diag.attachNote(access->getLoc()) << "one of them is stated here";
      return failure();
    }
    out = found.front().first;
    return success();
  }

private:
  void note(Layout layout, Operation *access) {
    for (auto &[seen, by] : found)
      if (seen == layout)
        return;
    found.push_back({std::move(layout), access});
  }

  LogicalResult backward(Value v, ArrayRef<int64_t> toPin) {
    Operation *def = v.getDefiningOp();
    if (!def || !seenBack.insert(def).second)
      return success();

    // A `ktdp.load` is where a branch ends, found or not: it is the one op that
    // reaches a view, so there is nothing further back to ask.
    if (isa<mlir::ktdp::LoadOp>(def)) {
      std::optional<Layout> layout;
      if (failed(layoutThrough(def, toPin, layout)))
        return failure();
      if (layout)
        note(std::move(*layout), def);
      return success();
    }

    // A REDUCTION is walked THROUGH rather than stopped at. It states which dims
    // it removes, so its input's layout is a layout for this tensor once the
    // removed dims' entries are dropped -- which is what `toPin` carries.
    if (auto reduce = dyn_cast<linalg::ReduceOp>(def)) {
      ArrayRef<int64_t> dropped = reduce.getDimensions();
      for (Value in : reduce.getDpsInputs()) {
        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        if (!inTy)
          continue;
        SmallVector<int64_t> toPinIn(inTy.getRank(), -1);
        int64_t surviving = 0;
        for (int64_t d = 0, e = inTy.getRank(); d < e; ++d) {
          if (llvm::is_contained(dropped, d))
            continue;
          if (surviving < (int64_t)toPin.size())
            toPinIn[d] = toPin[surviving];
          ++surviving;
        }
        if (failed(backward(in, toPinIn)))
          return failure();
      }
      return success();
    }

    // Anything else: the operands whose shape is this tensor's, which is the hop
    // that needs no carry. An operand of a different shape with no stated
    // correspondence ends the branch, because guessing one is how a layout would
    // come to describe the wrong dims.
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    if (!vTy)
      return success();
    for (Value operand : def->getOperands()) {
      auto oTy = dyn_cast<RankedTensorType>(operand.getType());
      if (oTy && oTy.getShape() == vTy.getShape())
        if (failed(backward(operand, toPin)))
          return failure();
    }
    return success();
  }

  LogicalResult forward(Value v) {
    auto vTy = dyn_cast<RankedTensorType>(v.getType());
    if (!vTy)
      return success();
    SmallVector<int64_t> identity;
    identity.reserve(vTy.getRank());
    for (int64_t d = 0, e = vTy.getRank(); d < e; ++d)
      identity.push_back(d);

    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();

      // A `ktdp.store` of this value is an end, the mirror of the load above. Only
      // of THIS value: a store's other operand is its access tile.
      if (auto store = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        if (store.getDataTile() != v)
          continue;
        std::optional<Layout> layout;
        if (failed(layoutThrough(store, identity, layout)))
          return failure();
        if (layout)
          note(std::move(*layout), store);
        continue;
      }

      if (!seenFwd.insert(user).second)
        continue;
      for (Value result : user->getResults()) {
        auto rTy = dyn_cast<RankedTensorType>(result.getType());
        if (rTy && rTy.getShape() == vTy.getShape())
          if (failed(forward(result)))
            return failure();
      }
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Building the buffer
//===----------------------------------------------------------------------===//

/// A `ktdp.construct_access_tile` over the whole of `memView`, anchored at the
/// origin. The tile is the whole buffer because the pin names a whole value: the
/// store writes all of it and each load reads all of it.
static Value buildWholeTile(OpBuilder &builder, Location loc, Value memView,
                            ArrayRef<int64_t> shape) {
  SmallVector<Value> zeros;
  zeros.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i)
    zeros.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
  return mlir::triton::ktdp::buildAccessTile(builder, loc, memView, shape,
                                             zeros);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct MaterializePinnedBuffersPass
    : public mlir::triton::ktdp::impl::MaterializePinnedBuffersBase<
          MaterializePinnedBuffersPass> {

  using MaterializePinnedBuffersBase::MaterializePinnedBuffersBase;

  /// Check one pin's offset, and record the byte range it occupies for the
  /// cross-pin comparison `checkNoOverlap` then makes.
  ///
  /// Two rules between the two functions, and what they have in common is that both
  /// hold for ANY base the scratchpad allocator hands this kernel: a range cannot be
  /// longer than a whole scratchpad, which is here, and two pins of one kernel share
  /// one base and so must not overlap, which is there. That is what makes them
  /// askable at all.
  ///
  /// ALIGNMENT is deliberately not among them. An offset counts from a base this
  /// pass never sees, so a stick-aligned offset is neither necessary nor sufficient
  /// for a stick-aligned buffer -- aligning the base it hands out is the
  /// allocator's rule, and asserting a multiple here would be asking the author to
  /// guarantee something they cannot see either.
  LogicalResult checkOffset(Pin &pin) {
    Operation *op = pin.op;

    // Read out of the pass option into a plain integer. A `Pass::Option<int64_t>`
    // streams into a diagnostic as garbage -- it is an llvm::cl::opt, and the
    // overload that wins is not the integer one -- so the copy is what makes the
    // message below readable rather than a stray byte.
    //
    // Already known non-negative: `runOnOperation` refuses a negative option before
    // it looks at any pin, because an option is wrong independently of the input.
    int64_t capacity = lxCapacityBytes;

    if (pin.offset < 0)
      return op->emitError() << "pinned offset is negative: " << pin.offset;

    // A SUB-BYTE element type, refused rather than waved through. `i1` is the
    // reachable one: `arith.cmpf` on tensors yields `tensor<...xi1>`, which is every
    // comparison and every `tl.where` mask, and `tl.spyre_pin(mask, ...)` reaches
    // here from Python. It used to give `elemBytes == 0` and an early success, which
    // took the capacity and overlap rules with it silently -- two such pins at the
    // same offset produced two buffers and no diagnostic.
    //
    // Refused and not supported, because an offset in ELEMENTS cannot address a
    // sub-byte value: `ktdp.construct_memory_view` takes element strides, and what
    // stride a packed i1 buffer has is a question about the device's packing rather
    // than about this pass. So this is the honest answer until something states it.
    Type elemTy = pin.type.getElementType();
    if (!elemTy.isIntOrFloat())
      return op->emitError()
             << "a pinned value's element type must be an integer or a float, "
                "and this one is " << elemTy;
    int64_t bits = elemTy.getIntOrFloatBitWidth();
    if (bits % 8 != 0)
      return op->emitError()
             << "a pinned value's element type must occupy whole bytes, and "
             << elemTy << " is " << bits
             << " bits; an offset counts elements, which cannot address a value "
                "narrower than a byte";
    int64_t elemBytes = bits / 8;

    // CHECKED, although the i32 offset rule in `readPin` already bounds the first
    // product: the element count comes from the tensor type rather than from the
    // annotation, so it is the operand this function cannot bound by reading a rule
    // elsewhere. An unchecked product wraps to a small range that passes both the
    // capacity and the overlap test and then builds a view at that offset with no
    // diagnostic, which is the one failure mode here that is silent.
    // The element count the buffer OCCUPIES, which is the physical one once it
    // carries a layout. A stick-tiled extent is rounded up to a whole stick, so a
    // buffer can hold more elements than its logical shape has -- and measuring the
    // logical count would under-report the range, which is silent twice over: two
    // pins an author packed as disjoint would pass the overlap test, and a pin past
    // the budget would pass the capacity test.
    int64_t elements = pin.type.getNumElements();
    if (pin.layout) {
      SmallVector<int64_t> physSizes;
      if (!mlir::triton::tts::applyCoordMap(pin.type.getShape(),
                                            pin.layout->src, pin.layout->op,
                                            pin.layout->arg, physSizes))
        return op->emitError()
               << "a pinned value has no static physical extents under the "
                  "layout its buffer takes from a neighbour";
      elements = 1;
      for (int64_t extent : physSizes)
        if (llvm::MulOverflow(elements, extent, elements))
          return op->emitError()
                 << "a pinned value's physical element count is not "
                    "representable";
    }

    int64_t size = 0;
    if (llvm::MulOverflow(pin.offset, elemBytes, pin.range.lo) ||
        llvm::MulOverflow(elements, elemBytes, size) ||
        llvm::AddOverflow(pin.range.lo, size, pin.range.hi))
      return op->emitError()
             << "pinned range is not representable: offset " << pin.offset
             << " of " << elements << " elements at " << elemBytes
             << " bytes each";

    if (capacity > 0 && pin.range.hi > capacity)
      return op->emitError()
             << "pinned range reaches byte " << pin.range.hi
             << " from the base of this kernel's allocation, past the "
             << capacity << " a core's scratchpad holds in total";

    return success();
  }

  /// Refuse a pin the `bind-base-addresses` mode cannot carry.
  ///
  /// That mode adds a canonicalize and a CSE after MaterializeBaseAddresses, and the
  /// CSE is fatal to a pin: every pin builds one `ktdp.construct_access_tile` for the
  /// store and an identical one per load -- same view, same block, same zero indices,
  /// both `Pure` -- so CSE merges them and dbo-opt's compute-group extraction then
  /// aborts. Nowhere else in the pipeline runs CSE, and both canonicalizes omit it
  /// for exactly this reason. See issue #161.
  ///
  /// Refused HERE rather than left to abort downstream, which is what it did: the IR
  /// compiled, and the failure then arrived from dbo-opt with nothing naming its
  /// cause.
  ///
  /// A pin whose value has NO use is exempt, and that is the whole of the
  /// granularity: it has a store and no load, so there is one access tile and
  /// nothing for CSE to merge.
  ///
  /// #199 removes the mode, and this check leaves with it.
  LogicalResult checkBindMode(const Pin &pin) {
    if (!bindBaseAddresses || pin.value.use_empty())
      return success();
    return pin.op->emitError()
           << "a pin is not supported with bind-base-addresses: the stage's CSE "
              "merges the pin's store and load access tiles, which aborts the "
              "scheduler";
  }

  /// Build the buffer for one pin and route the value's uses through it.
  void materialize(const Pin &pin) {
    Operation *op = pin.op;
    ArrayRef<int64_t> shape = pin.type.getShape();
    Location loc = op->getLoc();

    // Directly after the producer, which is the earliest point the value exists
    // and therefore dominates every use of it. The annotation carries no position
    // of its own, so there is no other candidate.
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    // Collected BEFORE the store is built, because the store is itself a use of
    // the value and must not be rewritten into a read of its own result.
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : pin.value.getUses())
      uses.push_back(&use);

    Value offset = arith::ConstantIndexOp::create(builder, loc, pin.offset);
    // Row-major, because a pin states no layout: the buffer is the compiler's, so
    // there is nothing for an author to agree with. `computeStrides` is upstream's
    // suffix product, which is exactly that.
    SmallVector<int64_t> strides = mlir::computeStrides(shape);
    Value memView = mlir::triton::ktdp::buildMemoryView(
        builder, loc, offset, shape, strides, /*dynSizes=*/{},
        /*dynStrides=*/{}, pin.type.getElementType(), pin.space);

    // The layout a neighbour stated, if one did, and this is the whole of how the
    // buffer becomes physical: annotated, it is stick-tiled by
    // RewriteDescriptorLayoutGeneric like any other annotated view, which then
    // finds it from the producing compute and restates that compute to match. The
    // view is built LOGICAL either way -- that pass is where logical becomes
    // physical, and it is below this one.
    if (pin.layout)
      if (Operation *viewOp = memView.getDefiningOp())
        viewOp->setAttr(TTSDialect::kTensorLayoutAttrName,
                        pin.layout->toAttr(builder));

    Value storeTile = buildWholeTile(builder, loc, memView, shape);
    mlir::ktdp::StoreOp::create(builder, loc, pin.value, storeTile);

    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();
      OpBuilder useBuilder(user);
      Value loadTile =
          buildWholeTile(useBuilder, user->getLoc(), memView, shape);
      auto loaded = mlir::ktdp::LoadOp::create(useBuilder, user->getLoc(),
                                               pin.type, loadTile);
      use->set(loaded.getResult());
    }

    // The request has been honoured, so it stops being one. That is what makes the
    // pass idempotent and keeps a stale annotation from reaching a consumer that
    // would read it as still outstanding -- the same reason
    // RewriteDescriptorLayoutGeneric removes the layout it has applied.
    op->removeAttr(TTSDialect::kPinAttrName);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The option, before the input. A negative capacity used to behave exactly like
    // the 0 that means "do not ask" -- `capacity > 0` admitted every negative value
    // -- so a pin at byte 400M passed under `lx-capacity-bytes=-1`. Refused here
    // rather than in `checkOffset` because an option is wrong independently of what
    // it is handed: a module with no pins at all must not launder one.
    // The copy into a plain integer is for the same reason `checkOffset` makes one:
    // a `Pass::Option<int64_t>` streams into a diagnostic as a stray byte.
    int64_t capacity = lxCapacityBytes;
    if (capacity < 0) {
      module.emitError() << "lx-capacity-bytes must not be negative, and this one "
                            "is " << capacity;
      return signalPassFailure();
    }

    // Collect first: `materialize` inserts ops around the annotated one, which
    // invalidates a walker's cursor.
    //
    // No one-pin-per-value check here, and none is needed: the annotation is a
    // single attribute on the op defining the value, so a second pin on the same
    // value has nowhere to go. LowerTTSMarkers is where that hazard lives and
    // where it is refused, because `setAttr` would otherwise resolve a collision
    // by overwriting and lose the earlier pin silently.
    SmallVector<Operation *> pinned;
    module.walk([&](Operation *op) {
      if (op->hasAttr(TTSDialect::kPinAttrName))
        pinned.push_back(op);
    });
    if (pinned.empty())
      return;

    // Every offset checked before anything is built: a diagnostic on the second
    // pin is more useful next to an unchanged module than next to a half-rewritten
    // one. Module-wide, because each of these rules is about one pin.
    SmallVector<Pin> pins;
    for (Operation *op : pinned) {
      Pin pin;
      if (failed(readPin(op, pin)))
        return signalPassFailure();
      // Before the numbers: the mode makes the pin unsupported whatever they are.
      if (failed(checkBindMode(pin)))
        return signalPassFailure();
      // And before them for a second reason: what the buffer's layout turns out to
      // be is what its extent is measured in.
      LayoutSearch search{pin};
      if (failed(search.run()) || failed(search.decide(pin.layout)))
        return signalPassFailure();
      if (failed(checkOffset(pin)))
        return signalPassFailure();
      pins.push_back(pin);
    }

    // Overlap, PER FUNCTION. The rule is that two pins of one kernel share one base
    // and so must not overlap -- two kernels share nothing, and a module with two
    // functions each pinning at offset 0 was being refused for it. A multi-function
    // module is a live shape, not only a hand-written one.
    //
    // Keyed by FunctionOpInterface so that both spellings are covered: the pass's
    // own lit tests are `tt.func`, and by the time the pipeline reaches this pass
    // ConvertFunctions has made them `func.func`. A pin outside any function is
    // grouped under null rather than dropped, so it is still checked against its
    // own kind.
    llvm::MapVector<Operation *, SmallVector<const Pin *>> byFunction;
    for (const Pin &pin : pins) {
      auto func = pin.op->getParentOfType<FunctionOpInterface>();
      byFunction[func ? func.getOperation() : nullptr].push_back(&pin);
    }

    for (auto &[func, group] : byFunction)
      if (failed(checkNoOverlap(group)))
        return signalPassFailure();

    for (const Pin &pin : pins)
      materialize(pin);
  }

  /// Refuse two pins of ONE function whose byte ranges intersect.
  ///
  /// An IntervalMap rather than every pair: it answers "does anything already cover
  /// this" in log time, and it is also the structure a relaxed rule would want,
  /// since live-range reuse is a question about what a range is occupied BY and not
  /// merely whether it is occupied.
  ///
  /// Its intervals are CLOSED, so a half-open [lo, hi) is inserted as [lo, hi-1].
  ///
  /// TODO: Relax this rule to allow scratchpad areas to be reused. One buffer
  /// serving two values whose live ranges do not overlap is a legitimate thing to
  /// want, and refusing every intersection forbids it. It is refused for now
  /// because nothing here can tell a deliberate reuse from a collision, and
  /// getting it wrong is silent; when something can, this should narrow to what is
  /// actually unsafe rather than being dropped.
  LogicalResult checkNoOverlap(ArrayRef<const Pin *> group) {
    using OccupiedBy = llvm::IntervalMap<int64_t, Operation *>;
    OccupiedBy::Allocator allocator;
    OccupiedBy occupied(allocator);
    for (const Pin *pin : group) {
      const ByteRange &range = pin->range;
      if (range.occupiesNothing())
        continue;
      auto it = occupied.find(range.lo);
      if (it.valid() && it.start() <= range.hi - 1) {
        // A note on the in-flight error and not a remark of its own: the other
        // pin's location is part of THIS diagnostic, so it groups with it rather
        // than arriving as a second message a lit case has to expect separately.
        InFlightDiagnostic diag =
            pin->op->emitError()
            << "pinned range [" << range.lo << ", " << range.hi
            << ") bytes overlaps another pin's [" << it.start() << ", "
            << (it.stop() + 1) << ")";
        diag.attachNote(it.value()->getLoc()) << "the other pin is here";
        return failure();
      }
      occupied.insert(range.lo, range.hi - 1, pin->op);
    }
    return success();
  }
};

} // namespace
