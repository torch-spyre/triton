//===- Dialect.h - The tts dialect ----------------------------------------===//
//
// The `tts` dialect: the Spyre backend's authoring annotations. No types and no
// attribute types — see the dialect's description in TTSDialect.td for why a
// dialect that defines almost nothing is still the right shape for this.
//
// What it publishes is one layout contract in two spellings — the
// `tts.tensor_layout` *op*, which a kernel authors on a `!tt.tensordesc`, and
// the `tts.tensor_layout` *attribute*, which the lowered IR carries on the
// memory view — and the one structural checker both are enforced by.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_DIALECT_TTS_IR_DIALECT_H
#define TRITON_SPYRE_DIALECT_TTS_IR_DIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

// For the generated op classes: ODS emits Op<> subclasses that need the op
// definition machinery, and TensorLayoutOp's operand is a Triton type.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "Dialect/TTS/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/TTS/IR/Ops.h.inc"

namespace mlir::triton::tts {

/// The coordinate op a `tts.tensor_layout`'s `phys_op[k]` names, by the
/// attribute's own numbering.
enum class CoordOp : int64_t { Identity = 0, FloorDiv = 1, Mod = 2, Splat = 3 };

/// `code` as a CoordOp, or `std::nullopt` if it names none.
///
/// The range check and the enum in one place, so that a consumer validating
/// `phys_op[k]` cannot spell the bound as a literal that a new enumerator would
/// silently invalidate — which is what `verifyTensorLayoutArrays` used to do.
/// Written as a switch rather than as `code <= 3` for the same reason: adding an
/// enumerator makes the compiler ask about this function.
std::optional<CoordOp> symbolizeCoordOp(int64_t code);

/// The physical extent one coordinate op gives one logical extent, or
/// `std::nullopt` when it is not a compile-time answer.
///
/// `logical` may be `ShapedType::kDynamic`; `arg` is the op's `phys_arg[k]`, and
/// is read only by the three non-identity ops.
///
/// Two behaviours here are load-bearing and neither is obvious from the name:
///
///   - **FloorDiv rounds UP.** The name is the coordinate map's, not this
///     function's: the *coordinate* of an element is `i floordiv arg`, and the
///     number of distinct such coordinates over `[0, logical)` is
///     `ceil(logical / arg)`. A floor here would drop the partial last stick.
///   - **Mod and Splat do not read `logical` at all.** Both give exactly `arg`
///     extents, so both answer even for a dynamic logical extent -- which is
///     why a dynamic dim can still have a statically known stick width.
std::optional<int64_t> applyStatic(int64_t logical, CoordOp op, int64_t arg);

/// The physical extents a coordinate map gives `logSizes`, one per physical dim.
///
/// Returns false, leaving `out` unspecified, if any physical dim's extent is not
/// a compile-time answer -- which makes a partial result impossible to mistake
/// for a whole one. The three arrays must already satisfy
/// `verifyTensorLayoutArrays`; this indexes `logSizes` with `physSrc[k]` without
/// rechecking the bound.
///
/// This is the evaluator of a coordinate map for everything that survives the
/// layout-attribute migration, and that is the point of it being here:
/// `rewrite-descriptor-layout-generic` derives a physical memref from it, and
/// `evaluateDeviceLayout` below derives the device footprint `SpyreBackend`
/// records in the compiled metadata from it, so the footprint a launcher
/// bounds-checks against and the extents the IR is actually built with cannot
/// disagree. (The named `rewrite-descriptor-layout` pass carries its own copy,
/// in RewriteDescriptorLayout/PermutationUtils.h, which knows no Splat; it goes
/// when that pass does.)
bool applyCoordMap(ArrayRef<int64_t> logSizes, ArrayRef<int64_t> physSrc,
                   ArrayRef<int64_t> physOp, ArrayRef<int64_t> physArg,
                   SmallVectorImpl<int64_t> &out);

/// The device layout a coordinate map gives a logical tensor.
///
/// A stick-tiled buffer can hold MORE elements than its host tensor does. A `[64]`
/// statistic written through a splat layout occupies `[64, S]` on the device, so a
/// host shape alone cannot size the allocation -- and getting it wrong is an
/// out-of-bounds write rather than a wrong answer: measured on that statistic, 128
/// bytes allocated against 8192 needed, 64x short.
///
/// This function is what closes that gap. Given a descriptor's layout annotation
/// (the coordinate map) and its logical shape and strides, it derives the two
/// arrays torch-spyre needs to describe the device buffer, and which its explicit
/// `SpyreTensorLayout` constructor takes: `device_size`, the extent of each device
/// axis, and `stride_map`, how far the host pointer moves per step along each. The
/// launcher checks an argument against them; a caller allocates a buffer from them.
///
/// Returns false, leaving both outputs unspecified, when any physical extent is
/// not a compile-time answer -- the whole pair is then absent rather than half
/// stated, because a stride map is meaningless without the extents it was
/// derived over.
///
/// It is one function because it is one derivation. Extents, stride rule and
/// unit-axis padding are three steps over the same coordinate-op numbering, so a
/// caller holding only some of them has to restate that numbering to do its
/// share. That is not hypothetical: `backend/tensor_layout.py` held the last two,
/// and paid for it with a fourth copy of `CoordOp`. Whole, the numbering lives
/// once, in the enum that owns it.
///
/// The three steps, each of which has a reason not to be obvious:
///
///   - **The extents.** How many positions does each device axis have? That is
///     `applyCoordMap`'s answer, so a floordiv rounds up and a splat contributes
///     its own width.
///
///   - **The stride map.** Stepping one position along a device axis moves the
///     host pointer how far? For a partitioned dim's two halves that is one lane
///     and one whole stick respectively. An axis the host tensor does not address
///     at all moves it nowhere, and that case is spelled `-1`; it arises twice --
///     a SPLAT, which replicates rather than partitions, so there is no host
///     stride to advance by, and a logical dim of extent ONE, which has nothing to
///     advance over. torch-spyre treats the two identically (same branch of
///     `dim_map_to_stride_map`, `spyre_tensor_impl.cpp`).
///
///     A table of cases would be wrong here, which is why this is their loop
///     re-derived instead: the outer half's stride is a PRODUCT, and stating it as
///     `logStrides[d] x phys_arg` is the right number only while the mod half is
///     the last axis -- true of every layout in tree and not a rule anything
///     checks. The loop gets it right by construction, running INNERMOST TO
///     OUTERMOST and carrying a per-logical-dim running stride: a stick split's
///     inner (mod) half takes `logStrides[d]`, and its outer (floordiv) half then
///     takes `logStrides[d] x (the inner half's device extent)`, whatever axis
///     that inner half turned out to be. Neither `-1` case updates the running
///     stride, so an outer axis over the same logical dim is unaffected by a splat
///     inside it.
///
///   - **The unit-axis padding.** torch-spyre's DMA setup reads one particular
///     device axis as the tile-count half of the stick split: the one THIRD FROM
///     THE END, paired with the LAST axis as its lanes. Our coordinate maps can
///     put something else there -- a real dimension, or a splat's replicated axis
///     -- and then that read lands on a dimension it corrupts. So a dummy unit
///     axis goes in where the DMA looks, which makes the read harmless. Where it
///     looks depends on the rank, and inserting shifts the later positions, which
///     is all the arithmetic below is about. Note this is therefore NOT "pad to
///     rank 3": the padding neutralizes one read, it does not reach a rank.
///
///     Concretely, `get_dim_map` (`spyre_mem.cpp`) matches each device axis to a
///     logical dim by a greedy stride scan and then FORCIBLY OVERWRITES the
///     third-from-last entry with the dim it found for the last axis. Sound for
///     every layout their own constructor builds, and wrong for a splat layout,
///     whose last axis addresses no logical dim at all: feed the rank-2
///     `device_size [64, 64]` / `stride_map [1, -1]` in unpadded and the scan
///     finds `dim_map = [0, -1]`, the forcing turns it into `[-1, -1]`, every
///     logical dim is then skipped downstream, and the DMA moves ONE ELEMENT
///     instead of 64, silently, with no check firing.
///
///     Three shapes already neutralize it, and get no padding: the
///     third-from-last axis has stride `-1` (the scan never matches a negative
///     stride, so that entry is already `-1` and the forcing is guarded out); or
///     its extent is 1 (the scan skips unit axes, same conclusion); or it really
///     is the floordiv half of the same logical dim the last axis takes modulo,
///     the canonical stick split, where the forcing writes back the value that
///     was already there. Rank 1 needs nothing either, since the two positions
///     coincide there and the forcing is a self-assignment.
///
///     Otherwise a unit axis goes in, and it goes in SECOND FROM THE END of the
///     unpadded layout -- not at the position being neutralized. Inserting shifts
///     every later position by one, so second-from-last before the insertion is
///     third-from-last after it, which is where the DMA will look. Inserting at
///     the old third-from-last would leave the DMA's read pointing at a real axis
///     and change nothing.
///
///     Worth raising upstream: their assumption has no test on their side, and
///     nothing in the layout they are handed lets them detect its violation.
bool evaluateDeviceLayout(ArrayRef<int64_t> logSizes,
                          ArrayRef<int64_t> logStrides,
                          ArrayRef<int64_t> physSrc, ArrayRef<int64_t> physOp,
                          ArrayRef<int64_t> physArg,
                          SmallVectorImpl<int64_t> &deviceSize,
                          SmallVectorImpl<int64_t> &strideMap);

/// The structural rules a `tts.tensor_layout` coordinate map obeys, checked
/// once for the three callers that need them:
///   - `TensorLayoutOp::verify`, the authoring op's own verifier;
///   - the dialect's `verifyOperationAttribute`, which sees every annotated op
///     at every verification point;
///   - `readCoordMap` in RewriteDescriptorLayoutGeneric, which is invocable on
///     hand-written IR and so cannot assume the verifier ran with the rank it
///     measures against.
///
/// One checker rather than three is the decision recorded here: the rules were
/// stated twice while the layout lived in the Triton dialect — in
/// `SpyreTensorLayoutOp::verify` and again in that pass — and the pass's half
/// would have quietly become the only half when the op went away. The op form
/// moving into this dialect is what lets it share the checker instead of adding
/// a third copy.
///
/// `logicalRank` is the rank of the thing the layout describes, and it is read
/// from a different place per caller: the descriptor's block type for the op,
/// the `construct_memory_view`'s result memref for the attribute. The extents
/// differ between those two; the rank does not. `emitError` supplies the
/// diagnostic's anchor; every message this emits names `tts.tensor_layout`
/// itself, so the anchor only has to say *where*.
///
/// What it enforces, and nothing else:
///   - the three arrays are parallel (equal length) and non-empty;
///   - `phys_src[k]` is in `[0, logicalRank)`;
///   - `phys_op[k]` names a `CoordOp` (checked through `symbolizeCoordOp`, so
///     the numbering is not restated here);
///   - `phys_arg[k] > 0` wherever `phys_op[k]` is not identity;
///   - a logical dim named by more than one physical dim is named either as a
///     stick split (one floordiv + one mod) or a splat re-stick (one identity +
///     one splat), and by exactly two dims.
///
/// Deliberately NOT here: that every logical dim is named at least once, and
/// that a split's two halves and a splat's identity companion are both present.
/// Those are consumer rules — a layout that drops a dim or carries half a split
/// is well formed as a coordinate map and merely unusable by the rewrite — and
/// they are checked where the rewrite would otherwise build an unaddressable
/// map. See readCoordMap.
LogicalResult verifyTensorLayoutArrays(
    ArrayRef<int64_t> physSrc, ArrayRef<int64_t> physOp,
    ArrayRef<int64_t> physArg, unsigned logicalRank,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// Read `tts.tensor_layout` off `op` and check its shape as an attribute: a
/// dictionary of exactly the three `DenseI64ArrayAttr` entries. On success the
/// three arrays are handed back; on failure a diagnostic has been emitted
/// through `emitError`.
///
/// Separate from verifyTensorLayoutArrays because the two questions have
/// different callers: a consumer that already has three arrays in hand (the op
/// form, a builder) needs only the structural rules, while anything reading the
/// attribute out of IR has to get past its spelling first.
LogicalResult readTensorLayoutArrays(
    Attribute value, ArrayRef<int64_t> &physSrc, ArrayRef<int64_t> &physOp,
    ArrayRef<int64_t> &physArg,
    llvm::function_ref<InFlightDiagnostic()> emitError);

} // namespace mlir::triton::tts

#endif // TRITON_SPYRE_DIALECT_TTS_IR_DIALECT_H
