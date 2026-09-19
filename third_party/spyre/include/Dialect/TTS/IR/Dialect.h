//===- Dialect.h - The tts dialect ----------------------------------------===//
//
// The `tts` dialect: the Spyre backend's authoring annotations. No ops, no
// types, no attribute types — see the dialect's description in TTSDialect.td for
// why a dialect that defines nothing is still the right shape for this.
//
// What it publishes is one attribute contract, `tts.tensor_layout`, and the one
// structural checker that contract is enforced by.
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

#include "Dialect/TTS/IR/Dialect.h.inc"

namespace mlir::triton::tts {

/// The structural rules a `tts.tensor_layout` coordinate map obeys, checked
/// once for the two callers that need them: the dialect's own
/// `verifyOperationAttribute`, which sees every annotated op at every
/// verification point, and `readCoordMap` in RewriteDescriptorLayoutGeneric,
/// which is invocable on hand-written IR and so cannot assume the verifier ran
/// with the rank it measures against.
///
/// One checker rather than two is the decision recorded here: when the layout
/// was an op, its verifier and that pass re-stated the same rules in two
/// places, and the pass's half would have quietly become the only half when the
/// op went away.
///
/// `logicalRank` is the rank of the thing the layout describes — the
/// `construct_memory_view`'s result memref for the attribute. `emitError`
/// supplies the diagnostic's anchor; every message this emits names
/// `tts.tensor_layout` itself, so the anchor only has to say *where*.
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
