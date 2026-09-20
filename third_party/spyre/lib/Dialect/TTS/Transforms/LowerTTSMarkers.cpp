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
// `tts.tensor_layout` admits a lowered tensor descriptor and nothing else.
//
// One marker exists today. A second is expected -- the shape below is the
// answer to what adding it costs, and the answer is a resolution function, an
// attribute builder, and one line in `runOnOperation`.
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
// Pass
//===----------------------------------------------------------------------===//

struct LowerTTSMarkersPass
    : public mlir::triton::tts::impl::LowerTTSMarkersBase<LowerTTSMarkersPass> {

  void runOnOperation() override {
    using mlir::triton::tts::TensorLayoutOp;
    using mlir::triton::tts::TTSDialect;

    // One statement per marker kind, and the three things a marker brings: its
    // op type, its attribute name, and the resolve/build pair above.
    if (failed(lowerMarkers<TensorLayoutOp>(
            getOperation(), TTSDialect::kTensorLayoutAttrName,
            resolveTensorLayout, buildTensorLayoutAttr)))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::tts {

std::unique_ptr<OperationPass<ModuleOp>> createLowerTTSMarkersPass() {
  return std::make_unique<LowerTTSMarkersPass>();
}

} // namespace mlir::triton::tts
