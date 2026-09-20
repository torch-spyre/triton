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
/// This is the one evaluator of a coordinate map in the tree, and that is the
/// point of it being here: the rewrite pass derives a physical memref from it,
/// and `SpyreBackend` derives the device footprint it records in the compiled
/// metadata from it, so the footprint a launcher bounds-checks against and the
/// extents the IR is actually built with cannot disagree.
bool applyCoordMap(ArrayRef<int64_t> logSizes, ArrayRef<int64_t> physSrc,
                   ArrayRef<int64_t> physOp, ArrayRef<int64_t> physArg,
                   SmallVectorImpl<int64_t> &out);

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
///   - `phys_op[k]` is one of 0 identity, 1 floordiv, 2 mod, 3 splat;
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
