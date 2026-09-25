//===- PlacePinnedValues.cpp - Give a tts.pin'd value its buffer ----------===//
//
// A `tts.pin` says where an intermediate's buffer lives. Honouring it means
// building that buffer and routing the value's uses through it: a memory view
// in the named space, a store of the value, and a load per use.
//
// So this is NOT the marker-to-attribute lowering its neighbour LowerTTSMarkers
// performs, and the difference is forced rather than stylistic. An attribute
// needs an op to live on, and a value pin's only candidate is the op PRODUCING
// the value -- `math.exp`, `arith.mulf`, a `tt.reduce`. Those ops do not
// survive `convert-elementwise-to-linalg` and `linalg-generalize-named-ops`,
// which REPLACE them, and a discardable attribute on a replaced op is dropped
// in silence. `tts.tensor_layout` has no such problem: it lands on a
// `ktdp.construct_memory_view`, which nothing later replaces. So the pin has to
// become ops while the marker is still here, which is what this pass does.
//
// Two shapes worth naming, because both could plausibly have gone the other
// way:
//
//   * ONE VIEW, shared by the store and every load, with a fresh access tile
//   per
//     access. That is what LowerDescriptorMemory does for a descriptor -- the
//     view is built once and `getDescriptorMemView` hands the same value to
//     every load and store of it -- and that path runs on hardware.
//     lx-placement.md contemplates a view per group instead, on the grounds
//     that "nothing may be reachable from two groups"; if that constraint is
//     real it is not the one descriptors already violate, so this follows the
//     shape that is known to work rather than the stricter one that is not.
//
//   * The buffer is the value's OWN shape, row-major. A pin states no layout --
//     the buffer is the compiler's, so there is nothing for an author to agree
//     with -- and the physical type is the producing compute's result type.
//
// What this pass does not do: place the intermediates nobody pinned. Every
// compute-to-compute edge needs a buffer whether or not the author could name
// it, and that is the whole of lx-placement.md's proposal; this pass does only
// what a marker asked for.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/Transforms/Passes.h"

#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::tts {
#define GEN_PASS_DEF_PLACEPINNEDVALUES
#include "Dialect/TTS/Transforms/Passes.h.inc"
} // namespace mlir::triton::tts

namespace {

using mlir::triton::tts::PinOp;

//===----------------------------------------------------------------------===//
// The numeric obligations on a pinned address
//===----------------------------------------------------------------------===//

/// The half-open element range a pin occupies, as every core has to reserve it.
///
/// For a constant address that is just `[base, base + elements)`. For an
/// address affine in the program id it is the HULL of the enumerated set
/// `{base + i*stride : i < cores}`, and the hull rather than the individual
/// members because the scratchpad is per-core: a `program_id` term does not
/// spread one tensor across cores, it moves each core's own tensor to a
/// different offset inside that core's own scratchpad. Which offset a given
/// core uses is a runtime answer, so every core has to leave room for all of
/// them.
struct PinRange {
  int64_t lo = 0;
  int64_t hi = 0; // exclusive

  bool overlaps(const PinRange &other) const {
    return lo < other.hi && other.lo < hi;
  }
};

static PinRange hullOf(int64_t base, int64_t stride, int64_t cores,
                       int64_t elements) {
  int64_t first = base;
  int64_t last = base + (cores - 1) * stride; // stride may be 0 or negative
  PinRange r;
  r.lo = std::min(first, last);
  r.hi = std::max(first, last) + elements;
  return r;
}

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
/// origin. The tile is the whole buffer because the pin names a whole value:
/// the store writes all of it and each load reads all of it.
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

struct PlacePinnedValuesPass
    : public mlir::triton::tts::impl::PlacePinnedValuesBase<
          PlacePinnedValuesPass> {

  using PlacePinnedValuesBase::PlacePinnedValuesBase;

  PlacePinnedValuesPass(ArrayRef<int64_t> gridShape) { grid = gridShape; }

  /// Check one pin's address against the two hardware numbers, and record its
  /// range for the cross-pin comparison. Returns failure after diagnosing.
  ///
  /// `elemBytes` is 0 for an element type whose width is not a byte count,
  /// which no compute produces today; both checks are then skipped rather than
  /// guessed.
  LogicalResult checkAddress(PinOp pin, int64_t cores, PinRange &range) {
    auto tensorTy = cast<RankedTensorType>(pin.getValue().getType());
    int64_t elements = tensorTy.getNumElements();
    Type elemTy = tensorTy.getElementType();
    int64_t elemBytes =
        elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() / 8 : 0;

    // Read out of the pass options into plain integers. A
    // `Pass::Option<int64_t>` streams into a diagnostic as garbage -- it is an
    // llvm::cl::opt, and the overload that wins is not the integer one -- so
    // the copy is what makes the messages below readable rather than a stray
    // byte.
    int64_t alignment = alignmentBytes;
    int64_t capacity = lxCapacityBytes;

    int64_t base = 0, stride = 0;
    // The op's verifier has already refused anything this cannot match, so a
    // failure here means the pass was handed IR that never verified.
    if (failed(
            mlir::triton::tts::matchPinAddress(pin.getAddress(), base, stride)))
      return pin.emitError()
             << "pinned address is not a constant or an affine function of "
                "tl.program_id(0); this should have been refused by the op's "
                "verifier";

    if (base < 0)
      return pin.emitError() << "pinned address is negative: " << base;

    range = hullOf(base, stride, cores, elements);

    // Alignment, on the two coefficients rather than on `cores` addresses:
    // every member is aligned exactly when the base and the stride both are.
    if (alignment > 0 && elemBytes > 0) {
      for (auto [what, coeff] :
           {std::pair<StringRef, int64_t>{"base", base},
            std::pair<StringRef, int64_t>{"stride", stride}})
        if ((coeff * elemBytes) % alignment != 0)
          return pin.emitError()
                 << "pinned address " << what << " " << coeff << " is "
                 << (coeff * elemBytes) << " bytes, which is not a multiple of "
                 << alignment << " (one stick)";
    }

    // Capacity, at the hull's top.
    if (capacity > 0 && elemBytes > 0) {
      int64_t needed = range.hi * elemBytes;
      if (needed > capacity)
        return pin.emitError()
               << "pinned range reaches " << needed << " bytes, past the "
               << capacity << " a core's scratchpad holds"
               << (stride != 0 ? " (an address affine in the program id makes "
                                 "every core reserve the whole set)"
                               : "");
    }

    return success();
  }

  /// Build the buffer for one pin and route the value's uses through it.
  LogicalResult place(PinOp pin) {
    Value value = pin.getValue();
    auto tensorTy = cast<RankedTensorType>(value.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    Location loc = pin.getLoc();

    auto kind = mlir::ktdp::symbolizeMemorySpaceKind(pin.getMemorySpace());
    if (!kind)
      return pin.emitError() << "unknown memory space '" << pin.getMemorySpace()
                             << "'; this should have been refused by the op's "
                                "verifier";
    // No ct_id: one address, the same on every core. A pin names the space a
    // value lives in for the core running it, never another core's copy.
    auto spaceAttr = mlir::ktdp::MemorySpaceAttr::get(pin.getContext(), *kind,
                                                      /*ct_id=*/-1);

    // Collected, and checked, BEFORE anything is built. Two reasons, and both
    // have bitten: the store this pass is about to create is itself a use of
    // the value and must not be rewritten into a read of its own result; and a
    // DominanceInfo built over the module goes stale the moment ops are
    // inserted, so the question has to be asked while the IR is still the one
    // it describes.
    //
    // The pin's own position is what the uses are measured against, because
    // that is where the store is about to go -- so `pin dominates user` is the
    // same question as `the store will dominate user`.
    //
    // A use the pin does not dominate cannot read the buffer. This is NOT only a
    // hand-written-IR case -- ordinary Python reaches it, by using a value before
    // pinning it:
    //
    //     e = tl.exp(x); y = tl.sqrt(e); tl.spyre_pin(e, ...); ... e + y
    //
    // so it is diagnosed rather than resolved. The alternative reading is
    // positional -- uses after the pin read the buffer, uses before it read the
    // register -- and it is coherent, but it splits one value across a register
    // path and a memory path, which makes the pin's observable effect (severing
    // the producer from its consumers) hold for only some of them. Telling the
    // author to move the pin above the use is the answer that keeps the op's
    // meaning whole.
    DominanceInfo dom(pin->getParentOp());
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : value.getUses()) {
      if (use.getOwner() == pin.getOperation())
        continue;
      if (!dom.properlyDominates(pin.getOperation(), use.getOwner()))
        return pin.emitError()
               << "a use of the pinned value is not dominated by this pin, so "
                  "it cannot read the pinned buffer; move the pin above that "
                  "use";
      uses.push_back(&use);
    }

    // At the pin, not at the producer: the pin op uses both the value and the
    // address, so its position dominates both definitions by construction, and
    // an address written as `BASE + pid*STRIDE` may well be computed after the
    // value it is pinning.
    OpBuilder builder(pin);

    // The address flows straight into the view: `construct_memory_view` takes
    // its offset as an SSA `index` operand, so an address affine in the program
    // id needs no reconstruction here -- only the width change from the `i32`
    // the frontend built it at.
    Value offset = arith::IndexCastOp::create(
                       builder, loc, builder.getIndexType(), pin.getAddress())
                       .getResult();

    SmallVector<int64_t> strides = rowMajorStrides(shape);
    Value memView = mlir::triton::ktdp::buildMemoryView(
        builder, loc, offset, shape, strides, /*dynSizes=*/{},
        /*dynStrides=*/{}, tensorTy.getElementType(), spaceAttr);

    Value storeTile = buildWholeTile(builder, loc, memView, shape);
    mlir::ktdp::StoreOp::create(builder, loc, value, storeTile);

    for (OpOperand *use : uses) {
      Operation *user = use->getOwner();
      OpBuilder useBuilder(user);
      Value loadTile =
          buildWholeTile(useBuilder, user->getLoc(), memView, shape);
      auto loaded = mlir::ktdp::LoadOp::create(useBuilder, user->getLoc(),
                                               tensorTy, loadTile);
      use->set(loaded.getResult());
    }

    pin.erase();
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Same default, and the same reason, as DistributeWork's: a ListOption
    // takes no tablegen default, and [32] is the 1D-on-full-hardware case.
    SmallVector<int64_t> gridShape(grid.begin(), grid.end());
    if (gridShape.empty())
      gridShape.push_back(32);
    int64_t cores = 1;
    for (int64_t g : gridShape)
      cores *= g;
    if (cores < 1)
      cores = 1;

    // Collect first: `place` erases the marker and inserts ops around it, which
    // invalidates a walker's cursor.
    SmallVector<PinOp> pins;
    module.walk([&](PinOp pin) { pins.push_back(pin); });
    if (pins.empty())
      return;

    // One pin per value. Two pins would name two buffers for one value and
    // nothing decides which the uses read; refused rather than resolved by walk
    // order.
    DenseMap<Value, PinOp> pinned;
    for (PinOp pin : pins) {
      auto [it, inserted] = pinned.try_emplace(pin.getValue(), pin);
      if (!inserted) {
        pin.emitError() << "value is pinned twice; a value has one buffer";
        it->second.emitRemark() << "first pinned here";
        return signalPassFailure();
      }
    }

    // The address checks, all of them, before anything is built: a diagnostic
    // on the second pin is more useful next to an unchanged module than next to
    // a half-rewritten one.
    SmallVector<std::pair<PinOp, PinRange>> ranges;
    for (PinOp pin : pins) {
      // An address is required HERE and optional on the op, and the asymmetry
      // is the honest statement of what exists. The op's surface is the
      // design's, in which the compiler places every intermediate and a pin is
      // an override; but nothing in this tree allocates a buffer that is not a
      // kernel argument, so an unaddressed pin names no location this pass
      // could build a view at. Offset 0 would be a fabricated address, not an
      // allocation, and it would collide with the scheduler's own pool -- which
      // is the very thing an author cannot check and the build must.
      //
      // This is why a `global` pin never compiles today: the verifier admits an
      // address only in `ct_local`, so a global pin is necessarily unaddressed.
      if (!pin.getAddress()) {
        pin.emitError()
            << "a pin with no address cannot be placed: nothing here allocates "
               "a buffer that is not a kernel argument, so state an address in "
               "'ct_local'";
        return signalPassFailure();
      }
      PinRange range;
      if (failed(checkAddress(pin, cores, range)))
        return signalPassFailure();
      ranges.push_back({pin, range});
    }

    // Cross-pin overlap, over the hulls. Quadratic in the number of ADDRESSED
    // pins in a module, which is small and author-written.
    //
    // TODO: Relax this rule to allow scratchpad areas to be reused. One buffer
    // serving two values whose live ranges do not overlap is a legitimate thing
    // to want, and refusing every intersection forbids it. It is refused for now
    // because nothing here can tell a deliberate reuse from a collision, and
    // getting it wrong is silent; when something can, this should narrow to what
    // is actually unsafe rather than being dropped.
    for (size_t i = 0; i < ranges.size(); ++i)
      for (size_t j = i + 1; j < ranges.size(); ++j)
        if (ranges[i].second.overlaps(ranges[j].second)) {
          ranges[j].first.emitError()
              << "pinned range [" << ranges[j].second.lo << ", "
              << ranges[j].second.hi << ") overlaps another pin's ["
              << ranges[i].second.lo << ", " << ranges[i].second.hi << ")";
          ranges[i].first.emitRemark() << "the other pin is here";
          return signalPassFailure();
        }

    for (PinOp pin : pins)
      if (failed(place(pin)))
        return signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::tts {

std::unique_ptr<OperationPass<ModuleOp>>
createPlacePinnedValuesPass(llvm::ArrayRef<int64_t> grid) {
  return std::make_unique<PlacePinnedValuesPass>(grid);
}

} // namespace mlir::triton::tts
