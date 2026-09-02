#ifndef KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H
#define KTDP_TRANSFORMS_REWRITEDESCRIPTORLAYOUT_PERMUTATIONUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::triton::ktdp {

enum class CoordOp : int64_t { Identity = 0, FloorDiv = 1, Mod = 2 };

/// A physical dim is a floor (stick-index) dim iff it carries a parallel
/// (non-reduced) role and a FloorDiv coord op. Shared by classify() and any
/// caller that must derive floor-dim membership from the same role/coord-op
/// pair without re-running classify() (e.g. building a target order from a
/// marker directly).
inline bool isFloorDim(int64_t role, CoordOp op) {
  return role >= 0 && op == CoordOp::FloorDiv;
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
