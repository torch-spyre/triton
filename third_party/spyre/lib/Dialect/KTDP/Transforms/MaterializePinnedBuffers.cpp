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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_MATERIALIZEPINNEDBUFFERS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

using mlir::triton::tts::TTSDialect;

//===----------------------------------------------------------------------===//
// Reading the annotation
//===----------------------------------------------------------------------===//

/// One `tts.pin` dictionary, as this pass needs it.
struct Pin {
  mlir::ktdp::MemorySpaceAttr space;
  /// Element index from the base of this kernel's scratchpad allocation.
  int64_t offset = 0;
  /// The single result the annotation is about, and the value to be stored.
  Value value;
  RankedTensorType type;
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
  auto offset = dyn_cast_or_null<IntegerAttr>(dict.get(TTSDialect::kOffsetName));
  if (!offset)
    return op->emitError() << "pinned value has no offset, so there is no "
                              "location to build its buffer at";
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
// Building the buffer
//===----------------------------------------------------------------------===//

/// Row-major strides for `shape`, innermost 1.
static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

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
  /// cross-pin comparison.
  ///
  /// Two rules, and what they have in common is that both hold for ANY base the
  /// scratchpad allocator hands this kernel: a range cannot be longer than a whole
  /// scratchpad, and two pins of one kernel share one base and so must not overlap.
  /// That is what makes them askable here at all.
  ///
  /// ALIGNMENT is deliberately not among them. An offset counts from a base this
  /// pass never sees, so a stick-aligned offset is neither necessary nor sufficient
  /// for a stick-aligned buffer -- aligning the base it hands out is the
  /// allocator's rule, and asserting a multiple here would be asking the author to
  /// guarantee something they cannot see either.
  LogicalResult checkOffset(Operation *op, const Pin &pin, ByteRange &range) {
    // Read out of the pass option into a plain integer. A `Pass::Option<int64_t>`
    // streams into a diagnostic as garbage -- it is an llvm::cl::opt, and the
    // overload that wins is not the integer one -- so the copy is what makes the
    // message below readable rather than a stray byte.
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

    range.lo = pin.offset * elemBytes;
    range.hi = range.lo + pin.type.getNumElements() * elemBytes;

    if (capacity > 0 && range.hi > capacity)
      return op->emitError()
             << "pinned range reaches byte " << range.hi
             << " from the base of this kernel's allocation, past the "
             << capacity << " a core's scratchpad holds in total";

    return success();
  }

  /// Build the buffer for one pin and route the value's uses through it.
  void materialize(Operation *op, const Pin &pin) {
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
    SmallVector<int64_t> strides = rowMajorStrides(shape);
    Value memView = mlir::triton::ktdp::buildMemoryView(
        builder, loc, offset, shape, strides, /*dynSizes=*/{},
        /*dynStrides=*/{}, pin.type.getElementType(), pin.space);

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
    // one.
    SmallVector<std::pair<Operation *, Pin>> pins;
    SmallVector<ByteRange> ranges;
    for (Operation *op : pinned) {
      Pin pin;
      if (failed(readPin(op, pin)))
        return signalPassFailure();
      ByteRange range;
      if (failed(checkOffset(op, pin, range)))
        return signalPassFailure();
      pins.push_back({op, pin});
      ranges.push_back(range);
    }

    // Cross-pin overlap, in bytes. An IntervalMap rather than every pair: it
    // answers "does anything already cover this" in log time, and it is also the
    // structure a relaxed rule would want, since live-range reuse is a question
    // about what a range is occupied BY and not merely whether it is occupied.
    //
    // Its intervals are CLOSED, so a half-open [lo, hi) is inserted as [lo, hi-1].
    //
    // TODO: Relax this rule to allow scratchpad areas to be reused. One buffer
    // serving two values whose live ranges do not overlap is a legitimate thing to
    // want, and refusing every intersection forbids it. It is refused for now
    // because nothing here can tell a deliberate reuse from a collision, and
    // getting it wrong is silent; when something can, this should narrow to what is
    // actually unsafe rather than being dropped.
    using OccupiedBy = llvm::IntervalMap<int64_t, Operation *>;
    OccupiedBy::Allocator allocator;
    OccupiedBy occupied(allocator);
    for (auto [i, range] : llvm::enumerate(ranges)) {
      if (range.occupiesNothing())
        continue;
      auto it = occupied.find(range.lo);
      if (it.valid() && it.start() <= range.hi - 1) {
        // A note on the in-flight error and not a remark of its own: the other
        // pin's location is part of THIS diagnostic, so it groups with it rather
        // than arriving as a second message a lit case has to expect separately.
        InFlightDiagnostic diag =
            pins[i].first->emitError()
            << "pinned range [" << range.lo << ", " << range.hi
            << ") bytes overlaps another pin's [" << it.start() << ", "
            << (it.stop() + 1) << ")";
        diag.attachNote(it.value()->getLoc()) << "the other pin is here";
        return signalPassFailure();
      }
      occupied.insert(range.lo, range.hi - 1, pins[i].first);
    }

    for (auto &[op, pin] : pins)
      materialize(op, pin);
  }
};

} // namespace
