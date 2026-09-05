#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::triton::ktdp {

enum class CoordOp : int64_t { Identity = 0, FloorDiv = 1, Mod = 2 };

/// Where a synthesized op's OUTPUT axes live — the space a `role` (and hence
/// `dimRoles`, `targetOrder` and the accumulator's axes) numbers positions in.
/// A property of the op *instance*, not of the op kind: the same linalg.reduce
/// lands in either space depending on whether the descriptor its result is
/// stored to carries the layout the operand's surviving stick structure
/// induces. See "Which physical shape" in docs/spyre-tensor-layouts.md.
enum class OutputAxisSpace {
  /// One output axis per surviving LOGICAL dim. A surviving stick-index dim is
  /// not an output axis at all: it is sliced away (extent 1) or scattered by an
  /// outer loop, so it is bucketed as a scatter dim and the accumulator carries
  /// the op's logical rank. Every matmul-like op is here, and so is a reduce
  /// whose result acquires no layout of its own.
  Logical,
  /// One output axis per surviving PHYSICAL dim, in physical order. A surviving
  /// stick-index dim rides along as a batch dim of the emitted op and gets an
  /// accumulator axis of its own, so it belongs in the op tile rather than in
  /// scatterDims. This is the space in which one logical axis can occupy two
  /// output axes (its stick index and its lane), which is exactly what a
  /// role numbered per logical dim cannot express.
  Physical,
};

/// Is this physical dim a stick index? A property of the coordinate op alone:
/// the dim carries the `floordiv` half of a stick split, so it counts sticks.
/// Says nothing about what the op consuming it does with the dim.
inline bool isFloorCoord(CoordOp op) { return op == CoordOp::FloorDiv; }

/// Must this physical dim be SCATTERED — driven from outside the op tile, so
/// that each iteration of its loop writes a different output slice — rather
/// than being an axis of the emitted op? True for exactly one situation, and it
/// takes three facts to name it:
///
///   role >= 0            the dim survives the op (it is not reduced away)
///   isFloorCoord(op)     it is a stick index
///   space == Logical     output axes are numbered per surviving *logical* dim
///
/// The third is what makes this a decision rather than a property: in the
/// Logical space a surviving stick index shares its logical axis with the lane
/// dim beside it and has nowhere to go, so it is bucketed into `scatterDims` and
/// driven from outside; in the Physical space it has an output axis of its own —
/// a batch dim of the emitted op — so it stays in the op tile and this is false.
///
/// The three conjuncts used to sit behind the name `isFloorDim`, which read as
/// a property of the dim. `isFloorCoord` is the property; this is the decision,
/// named for what the dim's loop *does* — scatter — because the sibling bucket
/// (`reduceLoopDims`) is sliced identically and differs only in that its loop
/// accumulates.
///
/// Shared by classify() and by any caller deriving the same membership from a
/// role/coord-op pair without re-running classify() (building a target order
/// from a marker directly, say). `space` is a parameter rather than a default
/// precisely because those callers must agree on it — that agreement is what
/// keeps a target order lined up 1:1 with opTileDims.
inline bool isScatterDim(int64_t role, CoordOp op, OutputAxisSpace space) {
  return role >= 0 && isFloorCoord(op) &&
         space == OutputAxisSpace::Logical;
}

/// Apply one coordinate op to a static (compile-time) logical extent.
/// Returns a non-kDynamic int64 on success, or std::nullopt when the result
/// is dynamic (i.e. needs a runtime SSA value).
inline std::optional<int64_t> applyStatic(int64_t logical, CoordOp op,
                                          int64_t arg) {
  switch (op) {
  case CoordOp::Identity:
    if (logical == mlir::ShapedType::kDynamic)
      return std::nullopt;
    return logical;
  case CoordOp::FloorDiv:
    if (logical == mlir::ShapedType::kDynamic)
      return std::nullopt;
    return arg == 0 ? std::optional<int64_t>(std::nullopt)
                    : std::optional<int64_t>((logical + arg - 1) / arg);
  case CoordOp::Mod:
    return arg;
  }
  return std::nullopt;
}

/// Compute physical static extents from logical static extents via a coord map.
/// Returns true on success (all physical extents are static).
inline bool applyCoordMap(llvm::ArrayRef<int64_t> logSizes,
                          llvm::ArrayRef<int64_t> physSrc,
                          llvm::ArrayRef<int64_t> physOp,
                          llvm::ArrayRef<int64_t> physArg,
                          llvm::SmallVectorImpl<int64_t> &out) {
  unsigned physRank = physSrc.size();
  out.resize(physRank);
  for (unsigned k = 0; k < physRank; ++k) {
    auto sz = applyStatic(logSizes[physSrc[k]],
                          static_cast<CoordOp>(physOp[k]), physArg[k]);
    if (!sz)
      return false;
    out[k] = *sz;
  }
  return true;
}

/// Compute the permutation that reorders opTileDims from physical order into
/// the target op's axis order. `targetOrder[t]` names the role (or -1 for the
/// contracted/reduced slot) that must land at target axis position `t`, so the
/// index into `targetOrder` *is* the destination position.
/// Returns empty vector if already identity (no transpose needed).
inline llvm::SmallVector<int64_t> computeTransposePerm(
    llvm::ArrayRef<int> opTileDims,
    llvm::ArrayRef<int64_t> dimRoles,
    llvm::ArrayRef<int64_t> targetOrder) {
  unsigned nTile = opTileDims.size();
  llvm::SmallVector<int64_t> perm(nTile, -1);
  llvm::SmallVector<bool> used(nTile, false);

  // Repeated roles cannot occur (roles are unique); -1 slots are matched
  // left-to-right against the physical reduction dims via `used`.
  for (unsigned t = 0; t < targetOrder.size(); ++t) {
    int64_t wantRole = targetOrder[t];
    for (unsigned j = 0; j < nTile; ++j) {
      if (!used[j] && dimRoles[opTileDims[j]] == wantRole) {
        perm[j] = (int64_t)t;
        used[j] = true;
        break;
      }
    }
  }
  // Check if identity.
  bool isIdentity = true;
  for (unsigned j = 0; j < nTile; ++j)
    if (perm[j] != (int64_t)j) { isIdentity = false; break; }
  return isIdentity ? llvm::SmallVector<int64_t>{} : perm;
}

/// Invert a permutation vector.
inline llvm::SmallVector<int64_t> invertPerm(llvm::ArrayRef<int64_t> perm) {
  llvm::SmallVector<int64_t> inv(perm.size());
  for (unsigned i = 0; i < perm.size(); ++i)
    inv[perm[i]] = i;
  return inv;
}

/// Compose two permutations: result[i] = first[second[i]]. `first` is applied
/// after `second` (i.e. `second` runs first, `first` runs second).
inline llvm::SmallVector<int64_t> composePerm(llvm::ArrayRef<int64_t> first,
                                              llvm::ArrayRef<int64_t> second) {
  llvm::SmallVector<int64_t> result(second.size());
  for (unsigned i = 0; i < second.size(); ++i)
    result[i] = first[second[i]];
  return result;
}

} // namespace mlir::triton::ktdp

#endif // KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H
