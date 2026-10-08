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
// And one op that is NOT a marker: `tts.spyre_op` names no value and has a body,
// so it does not fit the driver's resolve-and-move shape and has a function of
// its own, inlineSpyreOpCallSites. Its lowering inlines the body in place and
// leaves the call site as the `tts.spyreop_hint` on every op inlined -- an
// attribute on the ops rather than on one subject, which is the same rule (the op
// goes, a builtin attribute stays) applied to a call site that is about a
// computation.
//
// A pinned BLOCK ARGUMENT is therefore refused HERE, and this is the only place it
// can be. It is a value like any other and the op admits one, but it has no defining
// op, so there is nothing for the attribute to live on -- a function argument
// attribute would be the candidate, and `ConvertFunctions` does not carry those
// across `tt.func` -> `func.func`, so it would not reach a consumer.
//
// Not in the op's verifier, which is where it used to be. A verifier's checks have
// to read the op's OWN IR: whether the operand has a producer is a fact about the
// operand's provenance, and the canonicalizer that runs before this pass folds
// `x * 1`, `x + 0` and identity reshape/broadcast/transpose, RAUWing a marker's
// operand without touching the marker. A rule of that shape lets a legal rewrite
// elsewhere invalidate a module and report it against an op the author wrote
// correctly. The authoring question is asked at trace time instead, in
// `semantic.spyre_pin`, which is where the two kinds of block argument are still
// distinguishable.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/Transforms/Passes.h"

#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
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
/// lowering pass has retyped that value the operand is reached THROUGH a bridge
/// cast; that cast exists for the marker's sake and nothing else wants it once
/// the marker is gone. Guarded on `use_empty()`, so a cast with another user --
/// a second marker on the same value, or an op this pass has never heard of --
/// stays, and the dead-cast question stops being anyone's to manage.
///
/// "Through" is the premise, and it is written as a CONDITION rather than assumed,
/// because it holds for only one of the two markers. `tts.tensor_layout` resolves to
/// the `construct_memory_view` BEHIND the cast, so the cast is an intermediate hop
/// and erasing it cannot reach the subject. `tts.pin` resolves to
/// `value.getDefiningOp()`, so when a pin's producer happens to be a cast the
/// subject IS that cast -- and step 4 would erase the op step 2 just annotated,
/// silently, since step 3 is what made it use-empty. `def != subject` keeps the
/// genuine genericity: step 4 is inert for a marker that resolves TO its operand's
/// producer, and still fires for any marker that resolves through a cast.
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
    //
    // A GENUINE conflict only, which is what the comparison is for. Two markers
    // stating the same thing are not a collision at all: overwriting writes the
    // value that is already there, nothing is lost, and refusing would break a
    // shape nobody wrote deliberately. A helper that builds a descriptor and
    // annotates it, inlined twice with the same arguments, is exactly that -- the
    // `ttir` stage's CSE merges the `Pure` `tt.make_tensor_descriptor` ops and keeps
    // both markers, since a `tts` marker has no traits for CSE to act on. So the
    // refusal is about two markers DISAGREEING, and the message says which two.
    Attribute built = buildAttr(marker);
    Attribute existing = subject->getAttr(attrName);
    if (existing && existing != built) {
      InFlightDiagnostic diag = marker.emitError()
                                << "second " << MarkerOp::getOperationName()
                                << " resolving to the same op with a different "
                                   "statement, which can carry only one; this one "
                                   "states " << built << ", the first states "
                                << existing;
      diag.attachNote(subject->getLoc()) << "the op both resolve to is here";
      return failure();
    }

    subject->setAttr(attrName, built);

    // Read the operands out before the erase invalidates them.
    SmallVector<Value> named(marker->getOperands());
    marker->erase();
    for (Value value : named) {
      Operation *def = value.getDefiningOp();
      if (def && def != subject && isa<UnrealizedConversionCastOp>(def) &&
          def->use_empty())
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

  // A value with no defining op, which is REACHABLE and has exactly two causes by
  // the time control is here. Tracing refuses a block argument the author named, so
  // what is left is IR nobody traced -- this pass is invocable on hand-written IR,
  // which its own lit cases are -- or a fold that replaced the producer: the
  // canonicalizer before this pass rewrites `x * 1`, `x + 0` and an identity
  // reshape/broadcast/transpose to their input, and if that input is a block
  // argument the marker is left naming one.
  //
  // Both are named, because the second is the one that will surprise someone: the
  // kernel pinned a value an op produced, and the op is gone.
  Operation *producer = value.getDefiningOp();
  if (!producer) {
    marker.emitError()
        << "tts.pin names a block argument, which no op produces, so there is "
           "nothing to carry the annotation. Either this IR was written by hand "
           "-- tracing refuses a pinned kernel argument or loop-carried value at "
           "the kernel line -- or a fold replaced the producer with its own "
           "input, which an identity multiply, add, reshape, broadcast or "
           "transpose does";
    return nullptr;
  }

  // UNSUPPORTED rather than ill formed, and the difference is worth the wording:
  // the value is a perfectly good thing to pin, and what is missing is a spelling.
  // An attribute attaches to an OP and not to a value, so one dictionary can carry
  // one pin; a producer with several results needs the annotation to say which
  // result each pin is for, and it has no way to.
  //
  // What reaches it is ARGMAX, now that this pass runs at the end of the stage:
  // `tl.max(..., return_indices=True)` is one `linalg.reduce` with two results by
  // here, a value and an index. So there is no kernel-level workaround to offer --
  // the two results are one reduction and cannot be written as two -- and the
  // message says which result cannot be named rather than suggesting a rewrite.
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
           "producer, so it cannot say which result it is for. An argmax is the "
           "shape that reaches this -- tl.max(..., return_indices=True) is one "
           "reduction producing a value and an index -- and pinning either result "
           "needs the annotation to be a list positional in the results";
    diag.attachNote(producer->getLoc()) << "the producer is here";
    return nullptr;
  }

  return producer;
}

/// The attribute form of the same pin: the memory space, and the offset when the
/// marker stated one. Both are reused as-is, so the attribute holds what the op
/// held and this pass decides nothing about either.
///
/// Also what the driver's collision check compares, which is why it is a pure
/// function of the marker: two markers agreeing means two dictionaries that are
/// equal, and that only works if nothing here varies with the call.
static Attribute buildPinAttr(mlir::triton::tts::PinOp marker) {
  using mlir::triton::tts::TTSDialect;
  Builder builder(marker.getContext());

  SmallVector<NamedAttribute> entries;
  entries.push_back(builder.getNamedAttr(TTSDialect::kMemorySpaceName,
                                         marker.getMemorySpaceAttr()));
  // Absent rather than zero when the pin named no offset: 0 is a legitimate
  // element index, so it cannot double as "unstated".
  if (Attribute offset = marker.getOffsetAttr())
    entries.push_back(builder.getNamedAttr(TTSDialect::kOffsetName, offset));

  return builder.getDictionaryAttr(entries);
}

//===----------------------------------------------------------------------===//
// tts.spyre_op
//===----------------------------------------------------------------------===//

/// Inline every `tts.spyre_op` in `module` in place, hinting each inlined op.
///
/// Per call site, in this order:
///
///   1. pick its `id`: one per op, so two call sites of the same intrinsic --
///      one helper inlined twice, say -- stay two after both land in one
///      function;
///   2. set `{name = name, id = id}` on every op of the body but its
///      constants and its terminator, which is about to go. The op's verifier
///      has held every one of them to be elementwise, so none has a region;
///   3. replace each block argument by its operand and move the body's ops in
///      front of the call site;
///   4. replace each result by its yielded value and erase the call site, whose
///      terminator goes with it.
///
/// Constants are not hinted. The canonicalizer hoists, merges and folds them
/// freely, so a constant cannot stay a member of one call site, and every
/// reader of the hint skips constants.
///
/// Ids start above any already in the module, so IR that already carries hints
/// -- written by hand, or lowered once already -- does not collide with the
/// ones written here.
///
/// A call site with a constant operand is refused, before anything is inlined.
/// Every op of its body then reads only constants, so the first folding driver
/// in `spyrecode` folds the whole call site to one constant and no intrinsic is
/// selected. The check is here rather than at trace time because this pass runs
/// after the `ktir` stage's canonicalize: an operand that only folds to a
/// constant there (`tl.full` traces to a `tt.splat` of a scalar) is a constant
/// by now.
static LogicalResult inlineSpyreOpCallSites(ModuleOp module) {
  using mlir::triton::tts::SpyreOpOp;
  using mlir::triton::tts::TTSDialect;

  int64_t nextId = 0;
  module.walk([&](Operation *op) {
    if (DictionaryAttr hint = mlir::triton::tts::getSpyreopHint(op))
      if (auto id = hint.getAs<IntegerAttr>(TTSDialect::kSpyreopHintIdKey))
        nextId = std::max(nextId, id.getInt() + 1);
  });

  // Collect first: the rewrite erases. None is nested in another's body,
  // since the op's verifier admits only elementwise ops there.
  SmallVector<SpyreOpOp> callSites;
  module.walk([&](SpyreOpOp op) { callSites.push_back(op); });

  for (SpyreOpOp callSite : callSites)
    for (auto [i, operand] : llvm::enumerate(callSite.getInputs()))
      if (matchPattern(operand, m_Constant()))
        return callSite.emitError()
               << "tl.spyre_op(\"" << callSite.getName() << "\"): operand " << i
               << " is a constant, so the call site would fold to a constant "
                  "and no intrinsic would be selected";

  Builder builder(module.getContext());
  for (SpyreOpOp callSite : callSites) {
    DictionaryAttr hint = builder.getDictionaryAttr({
        builder.getNamedAttr(TTSDialect::kSpyreopHintNameKey,
                             callSite.getNameAttr()),
        builder.getNamedAttr(TTSDialect::kSpyreopHintIdKey,
                             builder.getI64IntegerAttr(nextId++)),
    });

    Block &body = callSite.getBody().front();
    Operation *yield = body.getTerminator();
    for (Operation &op : body.without_terminator())
      if (!op.hasTrait<OpTrait::ConstantLike>())
        op.setAttr(TTSDialect::kSpyreopHintAttrName, hint);

    for (auto [arg, operand] :
         llvm::zip_equal(body.getArguments(), callSite.getInputs()))
      arg.replaceAllUsesWith(operand);
    Block *parent = callSite->getBlock();
    parent->getOperations().splice(callSite->getIterator(),
                                   body.getOperations(), body.begin(),
                                   yield->getIterator());

    callSite->replaceAllUsesWith(yield->getOperands());
    callSite->erase();
  }
  return success();
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

    // FIRST, ahead of the markers. A pin may name a call site's result, and
    // the op producing that value exists only once the body is inlined -- run
    // after, the pin would land on the call site and be erased with it. Not a
    // marker, so not the driver's: see inlineSpyreOpCallSites, which refuses
    // a call site with a constant operand.
    if (failed(inlineSpyreOpCallSites(getOperation())))
      return signalPassFailure();

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
