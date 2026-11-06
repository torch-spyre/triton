//===- SpyreOpLoweringRules.h - What lowers to a spyreop intrinsic --------===//
//
// The lowering rules LowerSpyreOps and FuseGenerics both read, so the two
// passes cannot disagree on what lowers.
//
// Each group shape -- an anchor and the op reading it, which LowerSpyreOps
// lowers together only when they share a body -- is one entry in kGroupShapes
// in SpyreOpLoweringRules.cpp: a match function and the rewrite that lowers
// what it matched. LowerSpyreOps' group pattern, rooted on the anchor in a
// body, and FuseGenerics, before it puts the ops in one body, both ask
// matchLowerableGroup, which tries each shape. LowerSpyreOps then lowers what
// matched with rewriteLowerableGroup.
//
// A group has exactly one reader. An anchor read by more than one op matches
// no shape, so FuseGenerics does not group it and LowerSpyreOps does not lower
// it, and the `i1` left behind is reported by whichever pass holds it.
//
// The remaining exports are the questions the group shapes share with
// LowerSpyreOps' own patterns and diagnostics: which scalar types the float
// intrinsics take, which spyreop predicate a cmpf predicate maps to, and what
// a body value names from outside the body.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H
#define TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H

#include "ktir/Dialect/SpyreOp/SpyreOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::triton::spyre {

/// The ops a group would lower together: an anchor and every op that reads its
/// result.
struct GroupCandidate {
  /// The op whose result the group consumes: in the producer generic when the
  /// group spans two generics, in the one generic when it does not.
  Operation *anchor;
  /// The generic whose body holds `anchor`. A compare operand that is one of
  /// its block arguments is looked up through it.
  linalg::GenericOp anchorBody;
  /// The ops that read `anchor`'s result, wherever they are: in `anchorBody`
  /// when the group is in one body, in the consumer's body when it spans two
  /// generics.
  SmallVector<Operation *> readers;
};

/// What a group shape found: which ops of the anchor's body are in the group.
struct GroupMatch {
  StringRef shape;                      // e.g. "compare-cast", for the trace
  SmallVector<Operation *> producerOps; // the anchor body's ops the group consumes
};

/// The group shape `candidate` lowers as: the first shape in kGroupShapes that
/// matches its anchor and reader. Failure when `candidate` has other than
/// exactly one reader, or when no shape matches, with `why`, when non-null,
/// set to the reason.
FailureOr<GroupMatch> matchLowerableGroup(const GroupCandidate &candidate,
                                          std::string *why = nullptr);

/// Lowers `candidate` as the shape `match` names, `match` being what
/// matchLowerableGroup returned for it: the reader is replaced by the spyreop
/// ops. The anchor is left in place, for the caller to erase.
void rewriteLowerableGroup(PatternRewriter &rewriter,
                           const GroupCandidate &candidate,
                           const GroupMatch &match);

/// Whether spyreop's scalar float intrinsics accept this operand type.
bool isSpyreOpScalarType(Type type);

/// The spyreop predicate computing the same thing as `p`, or nothing:
///
///   - an ordering maps from both spellings, e.g. `oeq` and `ueq` -> `equal`
///     (they differ only on NaN, and no NaN reaches a compute body);
///   - `ord`/`uno` ask about NaN and `false`/`true` about nothing, so they have
///     no counterpart.
std::optional<spyreop::ComparePredicate>
spyrePredicateFor(arith::CmpFPredicate p);

/// What `v` names from OUTSIDE `generic`'s body: the matching `ins` operand
/// when `v` is one of the body's input block arguments, and `v` itself
/// otherwise.
///
/// This is what lets a rule ask about an operand's DEFINITION without caring
/// how the value reached the body. The same `1.0` is a scalar `arith.constant`
/// above the generic once CanonicalizeGenericOperands has folded the splat in,
/// and a splat `ins` operand if that pass has not run; resolved, both answer
/// the same question.
///
/// SOUND ONLY BECAUSE OF WHAT CALLERS ASK. A body value corresponds to the same
/// tensor element only for a value uniform across the operand, so a caller may
/// ask a resolved value whether it is a SPLAT or a scalar constant -- which is
/// exactly what `m_OneFloat` tests -- and may not ask it anything about a
/// particular element. A non-splat constant tensor resolves here too and
/// correctly matches nothing.
///
/// The `outs` block arguments are deliberately not resolved: an `outs` of a
/// `tensor.empty` has no defined value to name, so forwarding it would invite a
/// rule to read one.
Value resolveThroughBody(linalg::GenericOp generic, Value v);

} // namespace mlir::triton::spyre

#endif // TRITON_SPYRE_DIALECT_SPYREOP_UTILS_SPYREOPLOWERINGRULES_H
