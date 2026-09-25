//===- LowerTTSMarkers.cpp - Move tts marker annotations onto their subject ===//
//
// A `tts` marker op names a value in order to say something about it. The thing
// it says is already an attribute -- a marker has no result and no body -- so
// lowering it is not a conversion: the attribute moves onto the op the named
// value resolves to, and the marker goes away.
//
// That rule is the same for every marker, which is why it is one pass rather
// than a clause in whichever conversion happens to have built the op the
// attribute lands on. What differs per marker is only *how its operand
// resolves*, and that is also the marker's "which ops may carry me" predicate:
//
//   tts.tensor_layout  a lowered tensor descriptor, and nothing else. The
//                      attribute lands on the `ktdp.construct_memory_view` the
//                      descriptor became.
//   tts.pin            any value with a DEFINING OP, which is the carrier,
//                      because a pin names a value and a value's only op is the
//                      one defining it.
//
// Two markers, two resolution functions and two attribute builders, which is
// what the shape below was for.
//
// A pinned BLOCK ARGUMENT is therefore refused. It is a value like any other and
// the op admits one, but it has no defining op, so there is nothing for the
// attribute to live on -- a function argument attribute would be the candidate,
// and `ConvertFunctions` does not carry those across `tt.func` -> `func.func`,
// so it would not reach a consumer. Refused here rather than in the op's
// verifier because "what counts as resolvable" is this pass's question.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/Transforms/Passes.h"

#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::tts {
#define GEN_PASS_DEF_LOWERTTSMARKERS
#include "Dialect/TTS/Transforms/Passes.h.inc"
} // namespace mlir::triton::tts

namespace {

//===----------------------------------------------------------------------===//
// The generic driver
//===----------------------------------------------------------------------===//

/// Lower every `MarkerOp` in `module`.
///
/// This is the whole generic half, and deliberately knows nothing about any
/// particular marker:
///
///   1. resolve the marker to the op its attribute belongs on;
///   2. set `attrName` on that op to what `buildAttr` makes of the marker;
///   3. erase the marker;
///   4. erase any `builtin.unrealized_conversion_cast` that step 3 left dead.
///
/// Step 4 is generic and not a `tts.tensor_layout` special case. A marker's
/// operand is typed as whatever the authoring form spoke about, so after a
/// lowering pass has retyped that value the operand is reached through a bridge
/// cast; that cast exists for the marker's sake and nothing else wants it once
/// the marker is gone. Guarded on `use_empty()`, so a cast with another user --
/// a second marker on the same value, or an op this pass has never heard of --
/// stays, and the dead-cast question stops being anyone's to manage.
///
/// `resolve` returns null after emitting its own diagnostic: what counts as
/// resolvable is the marker's to say, so the message naming what was expected
/// has to come from there.
template <typename MarkerOp, typename ResolveFn, typename AttrFn>
static LogicalResult lowerMarkers(ModuleOp module, StringRef attrName,
                                  ResolveFn resolve, AttrFn buildAttr) {
  // Collect first, rewrite second: step 3 and step 4 both erase, and an erase
  // during a `module.walk` invalidates the walker's cursor.
  SmallVector<MarkerOp> markers;
  module.walk([&](MarkerOp op) { markers.push_back(op); });

  for (MarkerOp marker : markers) {
    Operation *subject = resolve(marker);
    if (!subject)
      return failure();

    // Two markers of one kind resolving to the same subject would collide, and
    // `setAttr` resolves a collision by overwriting: the later marker wins and
    // the earlier one is gone with nothing said. Refused instead, because which
    // of the two survived would be a fact about this loop's order rather than
    // about anything the author wrote.
    //
    // Generic, like the rest of this driver, because the hazard is: one subject
    // can be reached from more than one marker. For `tts.pin` that is two pins
    // on one value -- they share a producer. For `tts.tensor_layout` it is two
    // layouts on one descriptor -- they share a memory view.
    if (Attribute existing = subject->getAttr(attrName)) {
      InFlightDiagnostic diag = marker.emitError()
                                << "second " << MarkerOp::getOperationName()
                                << " resolving to the same op, which can carry "
                                   "only one; the first states " << existing;
      diag.attachNote(subject->getLoc()) << "the op both resolve to is here";
      return failure();
    }

    subject->setAttr(attrName, buildAttr(marker));

    // Read the operands out before the erase invalidates them.
    SmallVector<Value> named(marker->getOperands());
    marker->erase();
    for (Value value : named) {
      Operation *def = value.getDefiningOp();
      if (def && isa<UnrealizedConversionCastOp>(def) && def->use_empty())
        def->erase();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// tts.tensor_layout
//===----------------------------------------------------------------------===//

/// `tts.tensor_layout` names a `!tt.tensordesc`, and its attribute belongs on
/// the `ktdp.construct_memory_view` that descriptor became. The route is the one
/// LowerDescriptorMemory leaves: the marker's operand is the
/// `builtin.unrealized_conversion_cast` bridging the view's memref back to
/// `!tt.tensordesc` so the operand keeps verifying, and the view is the cast's
/// input.
///
/// `isLoweredDescriptor` / `getDescriptorMemView` are that route, already
/// written and published by KTDPUtils for exactly this -- the same pair the
/// access-op patterns and the named layout pass reach the memref through.
///
/// The two checks together are this marker's admissibility rule, and both
/// failures mean the same thing in practice: the marker was reached before
/// LowerDescriptorMemory built anything for it, or on a descriptor that pass
/// declined. Diagnosed rather than skipped -- a marker silently dropped would
/// take a kernel's layout with it and produce a correct-looking logical
/// artifact -- and rather than asserted, since this pass is invocable on
/// hand-written IR.
static Operation *
resolveTensorLayout(mlir::triton::tts::TensorLayoutOp marker) {
  Value desc = marker.getDesc();
  if (!mlir::triton::ktdp::isLoweredDescriptor(desc)) {
    marker.emitError()
        << "tts.tensor_layout does not annotate a lowered tensor descriptor: "
           "expected its operand to be the builtin.unrealized_conversion_cast "
           "that lower-descriptor-memory leaves bridging a memref back to "
           "!tt.tensordesc";
    return nullptr;
  }

  Value memView = mlir::triton::ktdp::getDescriptorMemView(desc);
  auto memViewOp = memView.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
  if (!memViewOp) {
    marker.emitError()
        << "tts.tensor_layout does not annotate a lowered tensor descriptor: "
           "its operand bridges a memref that no ktdp.construct_memory_view "
           "defines, so there is no memory view to carry the layout";
    return nullptr;
  }
  return memViewOp;
}

/// The attribute form of the same layout: a dictionary of the marker's three
/// dense i64 arrays, reused as-is under the names the dialect publishes. Builtin
/// attributes only, which is what lets a consumer that does not load `tts` still
/// parse the result -- see the note in TTSDialect.td.
static Attribute buildTensorLayoutAttr(mlir::triton::tts::TensorLayoutOp marker) {
  using mlir::triton::tts::TTSDialect;
  Builder builder(marker.getContext());
  return builder.getDictionaryAttr({
      builder.getNamedAttr(TTSDialect::kPhysSrcName, marker.getPhysSrcAttr()),
      builder.getNamedAttr(TTSDialect::kPhysOpName, marker.getPhysOpAttr()),
      builder.getNamedAttr(TTSDialect::kPhysArgName, marker.getPhysArgAttr()),
  });
}

//===----------------------------------------------------------------------===//
// tts.pin
//===----------------------------------------------------------------------===//

/// `tts.pin` names a value, and its attribute belongs on the op DEFINING that
/// value -- there is no other op the annotation could be about.
///
/// No admissibility test on what that op is. A pin says where a value's buffer
/// goes, which is a statement about the value and not about how it was computed,
/// so a `math.exp`, a `linalg.reduce` and a `ktdp.load` are equally valid
/// carriers and the consumer never reads the op's identity.
///
/// Two things are checked here rather than by the op, and both are about the MOVE
/// rather than about what the author wrote. The op holds its value as an operand
/// and can say whether that value is well formed; only resolution knows which op
/// is about to carry the annotation, and therefore whether it can.
static Operation *resolvePin(mlir::triton::tts::PinOp marker) {
  Value value = marker.getValue();

  // A value with no defining op is unreachable, because the op's verifier refuses
  // a pinned block argument -- an entry input lives where its base pointer says,
  // so it is not an intermediate to place. Checked anyway rather than asserted,
  // since this pass is invocable on hand-written IR and a null would otherwise be
  // a crash instead of a diagnostic.
  Operation *producer = value.getDefiningOp();
  if (!producer) {
    marker.emitError()
        << "tts.pin names a value with no defining op; this should have been "
           "refused by the op's verifier";
    return nullptr;
  }

  // UNSUPPORTED rather than ill formed, and the difference is worth the wording:
  // the value is a perfectly good thing to pin, and what is missing is a spelling.
  // An attribute attaches to an OP and not to a value, so one dictionary can carry
  // one pin; a producer with several results needs the annotation to say which
  // result each pin is for, and it has no way to.
  //
  // Reachable from a loop carrying more than one value -- `%acc:2 = scf.for ... ->
  // (T, T)` -- so the workaround is at the kernel level: carry one value per loop,
  // which every loop in the softmax fixture already does and which makes both its
  // accumulators pinnable.
  //
  // TODO: when this is wanted, make the attribute a LIST positional in the
  // results -- `[{}, {...}]` -- with an empty dictionary for a result nobody
  // pinned and a length equal to the result count. Position carries the result
  // number, so no entry needs to name its own index.
  //
  // List-only, replacing today's bare dictionary rather than joining it. The
  // attribute is an internal handoff: LowerTTSMarkers writes it and one pass in
  // `spyrecode` consumes it, and nothing outside this tree ever reads it, since it
  // is gone before the artifact goes out. So changing its shape costs nothing now,
  // while two accepted spellings would cost every consumer forever.
  if (producer->getNumResults() != 1) {
    InFlightDiagnostic diag =
        marker.emitError()
        << "pinning one result of a " << producer->getNumResults()
        << "-result op is not supported: the annotation is one attribute on the "
           "producer, so it cannot say which result it is for. Carry one value "
           "per op -- for a loop, one value per loop";
    diag.attachNote(producer->getLoc()) << "the producer is here";
    return nullptr;
  }

  return producer;
}

/// The attribute form of the same pin: the memory space, and the address when the
/// marker stated one. The address attribute is reused as-is, so which of its two
/// spellings the author chose survives into the attribute for a consumer to read.
static Attribute buildPinAttr(mlir::triton::tts::PinOp marker) {
  using mlir::triton::tts::TTSDialect;
  Builder builder(marker.getContext());

  SmallVector<NamedAttribute> entries;
  entries.push_back(builder.getNamedAttr(TTSDialect::kMemorySpaceName,
                                         marker.getMemorySpaceAttr()));
  // Absent rather than zero when the pin named no address: 0 is a legitimate
  // element index, so it cannot double as "unstated".
  if (Attribute address = marker.getAddressAttr())
    entries.push_back(
        builder.getNamedAttr(TTSDialect::kAddressName, address));

  return builder.getDictionaryAttr(entries);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerTTSMarkersPass
    : public mlir::triton::tts::impl::LowerTTSMarkersBase<LowerTTSMarkersPass> {

  void runOnOperation() override {
    using mlir::triton::tts::PinOp;
    using mlir::triton::tts::TensorLayoutOp;
    using mlir::triton::tts::TTSDialect;

    // One statement per marker kind, and the three things a marker brings: its
    // op type, its attribute name, and the resolve/build pair above.
    if (failed(lowerMarkers<TensorLayoutOp>(
            getOperation(), TTSDialect::kTensorLayoutAttrName,
            resolveTensorLayout, buildTensorLayoutAttr)))
      return signalPassFailure();

    if (failed(lowerMarkers<PinOp>(getOperation(), TTSDialect::kPinAttrName,
                                   resolvePin, buildPinAttr)))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::tts {

std::unique_ptr<OperationPass<ModuleOp>> createLowerTTSMarkersPass() {
  return std::make_unique<LowerTTSMarkersPass>();
}

} // namespace mlir::triton::tts
