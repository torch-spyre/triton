//===- RequirementAnalysis.cpp - Phase 2A backward: what is wanted --------===//
//
// Analysis only. Walks backward from the physicalized ktdp.stores and computes,
// for every value it reaches, the layout that store wants of it. Creates no ops
// and mutates nothing.
//
// The result is what ReducePropagation reads to decide whether a reduce's result
// is physical, so this runs before the forward analysis. See "a backward
// requirement analysis" in docs/spyre-tensor-layouts.md for the rule table and
// the questions still owed answers.
//
//===----------------------------------------------------------------------===//

#include "RewriteDescriptorLayout/RequirementAnalysis.h"
#include "RewriteDescriptorLayout/PermutationUtils.h"

#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "rewrite-descriptor-layout"

using namespace mlir;

namespace mlir::triton::ktdp {

namespace {

/// Reorder a requirement's four per-physical-dim arrays by `gather`:
/// out[i] = in[gather[i]]. The marker rides along unchanged -- it names where
/// the requirement came from, not what order its dims are in.
LayoutRequirement permuteRequirement(const LayoutRequirement &req,
                                     llvm::ArrayRef<int64_t> gather) {
  LayoutRequirement out;
  out.marker = req.marker;
  out.physSrc.reserve(gather.size());
  out.physOp.reserve(gather.size());
  out.physArg.reserve(gather.size());
  out.physExtents.reserve(gather.size());
  for (int64_t g : gather) {
    out.physSrc.push_back(req.physSrc[g]);
    out.physOp.push_back(req.physOp[g]);
    out.physArg.push_back(req.physArg[g]);
    out.physExtents.push_back(req.physExtents[g]);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Backward patterns: one per op kind
//===----------------------------------------------------------------------===//

/// Elementwise / single-tensor shape-preserving: the requirement passes through
/// unchanged, to every tensor operand. Rank-agnostic, so there is nothing to
/// recompute.
///
/// The match is the forward ElementwisePropagation rule, verbatim: one result, a
/// RankedTensorType, every tensor operand agreeing on a shape -- so the same op
/// set carries both facts.
///
/// It deliberately does NOT also require the result's shape to equal the
/// operands'. That reads like the stronger test but terminates every requirement
/// at the first elementwise op on a chain: Phase 1 has already physicalized the
/// loads and stops there, so a mid-chain `arith.addf` has physical operands and
/// a still-logical result until Phase 2 retypes it.
///
/// Local is safe for the same reason it is forward -- reachability comes from the
/// seeded walk -- and the rank-changing ops that satisfy this shape rule
/// (reshape family, broadcast) have explicit rules registered ahead of it.
///
/// Nothing here requires the operands' own forward layouts to agree with each
/// other; a mixed pair at an `arith` op is an owed answer in the doc.
struct ElementwiseRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    if (op->getNumResults() != 1 ||
        !isa<RankedTensorType>(op->getResult(0).getType()))
      return false;
    // Deliberately does NOT compare operand shapes to each other. Mid-analysis
    // the IR is half-retyped -- Phase 1 physicalizes loads and stops -- so a
    // sibling operand being a different shape is the state this pass exists to
    // resolve, not evidence that the op is unknown. An elementwise op's
    // requirement is the same on every operand regardless (induce is `return
    // req`), so the comparison gates nothing it needs.
    //
    // Shape-changing ops are excluded by KIND; isShapeChangingOp (Types.h) is
    // the one list all three elementwise predicates share, and states why.
    if (isShapeChangingOp(op))
      return false;
    bool sawTensorOperand = false;
    for (Value o : op->getOperands())
      if (isa<RankedTensorType>(o.getType()))
        sawTensorOperand = true;
    return sawTensorOperand;
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    return req;
  }
};

/// linalg.transpose: the input is the same data in a permuted dim order, so the
/// requirement on it is the requirement on the result read through the inverse
/// permutation -- result dim i is input dim permutation[i], hence input dim j
/// wants the result's entry invertPerm(permutation)[j].
///
/// The init operand carries the RESULT's shape, so it gets the requirement
/// unpermuted.
struct TransposeRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<linalg::TransposeOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    auto tr = cast<linalg::TransposeOp>(op);
    if (operand == tr.getInit())
      return req;
    if (operand != tr.getInput())
      return failure();
    auto perm = tr.getPermutation();
    // A requirement is indexed per PHYSICAL dim, so a permutation stated over
    // logical dims cannot reorder it. The two coincide only when the layout is
    // unsplit; otherwise there is no defined reordering and nothing crosses.
    if (perm.size() != req.physSrc.size())
      return failure();
    return permuteRequirement(req, invertPerm(perm));
  }
};

/// linalg.reduce: the requirement is CONSUMED here. It is recorded at the
/// reduce's result, which is what the reduce needs -- the decision compares it
/// against what the operand's forward layout induces, and the operand's own
/// layout is a forward fact. Nothing crosses, and nothing partial is ever built
/// (see "A requirement is total" in the doc).
struct ReduceRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<linalg::ReduceOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    return failure();
  }
};

/// linalg.matmul / linalg.batch_matmul: the requirement is DISCHARGED here, as
/// the widen the store already emits. The op is fixed at logical rank so its
/// result cannot carry the requirement, and the operands are narrowed from their
/// own forward layouts without it having to cross.
struct MatmulRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    return failure();
  }
};

/// tensor.expand_shape / collapse_shape / reshape: the requirement crosses only
/// a reshape that adds or removes SIZE-1 dims, and terminates otherwise.
///
/// The general case cannot cross, for the reason the forward rule states: a
/// requirement is indexed per PHYSICAL dim, and a reassociation that fuses two
/// real axes leaves no dim for one of them to map to. Collapsing physical
/// [1, 64, 64] under phys_op = [floor, id, mod] via [[0, 1], [2]] fuses the
/// stick index with the row, and no coordinate map describes the result.
///
/// A reshape that only inserts or drops size-1 dims is different: it touches no
/// real axis, and a size-1 dim carries no coordinate information. So the
/// requirement crosses with phys_src renumbered. That narrow class is what
/// LowerComputeOps emits between a reduce and a broadcast -- rules A3 and A4
/// lower tt.expand_dims and tt.broadcast independently, so A3 expands 1 -> 1x1
/// and A4 immediately collapses it back.
///
/// The reassociation groups always index the HIGHER-rank side: the result for
/// expand_shape, the operand for collapse_shape. So the two directions are not
/// symmetric and are handled separately below.
///
/// The safety condition is on the marker, not on extents: a stick index can
/// itself have extent 1 while still carrying coordinate meaning, so a group of
/// size > 1 must contain no floordiv or mod dim.
struct ReshapeRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
               tensor::ReshapeOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    // tensor.reshape takes a runtime shape operand, so there is no static
    // reassociation to reason about.
    auto expand = dyn_cast<tensor::ExpandShapeOp>(op);
    auto collapse = dyn_cast<tensor::CollapseShapeOp>(op);
    if (!expand && !collapse)
      return failure();

    auto inTy = dyn_cast<RankedTensorType>(operand.getType());
    auto resTy = dyn_cast<RankedTensorType>(result.getType());
    if (!inTy || !resTy)
      return failure();

    // The requirement is stated over the RESULT's logical dims.
    unsigned logicalRank = 0;
    for (int64_t d : req.physSrc)
      logicalRank = std::max(logicalRank, (unsigned)(d + 1));
    if (logicalRank != (unsigned)resTy.getRank())
      return failure();

    auto reassoc = expand ? expand.getReassociationIndices()
                          : collapse.getReassociationIndices();
    // Groups index the higher-rank side. Verify that side's rank matches, so a
    // malformed pairing is refused rather than mis-indexed.
    llvm::ArrayRef<int64_t> bigShape =
        expand ? resTy.getShape() : inTy.getShape();
    if (reassoc.size() != (unsigned)(expand ? inTy.getRank()
                                            : resTy.getRank()))
      return failure();

    // remap[result dim] -> operand dim, or -1 when the dim disappears.
    llvm::SmallVector<int64_t> remap(logicalRank, -1);
    for (unsigned g = 0; g < reassoc.size(); ++g) {
      // Within a group, at most one dim of the higher-rank side may be
      // non-unit; it is the one the lower-rank side's dim corresponds to.
      int64_t nonUnit = -1;
      for (int64_t d : reassoc[g]) {
        if (d < 0 || d >= (int64_t)bigShape.size())
          return failure();
        if (bigShape[d] != 1) {
          if (nonUnit >= 0)
            return failure();     // two real axes in one group: cannot cross
          nonUnit = d;
        }
      }
      // A floordiv/mod dim inside a multi-dim group would be fused or split.
      if (reassoc[g].size() > 1)
        for (int64_t d : reassoc[g]) {
          int64_t reqDim = expand ? d : (int64_t)g;
          for (unsigned p = 0; p < req.physSrc.size(); ++p)
            if (req.physSrc[p] == reqDim &&
                static_cast<CoordOp>(req.physOp[p]) != CoordOp::Identity)
              return failure();
        }

      if (expand) {
        // Groups index the RESULT. Group g corresponds to operand dim g, and
        // the surviving result dim within it is the non-unit one (or the first).
        int64_t keep = nonUnit >= 0 ? nonUnit : reassoc[g].front();
        remap[keep] = (int64_t)g;
      } else {
        // Groups index the OPERAND. Result dim g corresponds to the group's
        // non-unit operand dim (or its first).
        int64_t keep = nonUnit >= 0 ? nonUnit : reassoc[g].front();
        if ((unsigned)g >= logicalRank)
          return failure();
        remap[g] = keep;
      }
    }

    // Build the operand-side requirement. The two directions differ in whether
    // entries are dropped or added:
    //
    //   expand_shape   the operand has FEWER dims, so a requirement entry whose
    //                  logical dim is a newly inserted size-1 dim is dropped.
    //   collapse_shape the operand has MORE dims, so an entry must be ADDED for
    //                  each size-1 operand dim the collapse removed. Such a dim
    //                  is Identity with extent 1 -- it carries no coordinate
    //                  information, which is exactly why crossing is sound.
    //
    // Either way the result is indexed per operand dim, so it is assembled by
    // walking the operand's dims rather than the requirement's.
    LayoutRequirement out;
    out.marker = req.marker;
    llvm::SmallVector<int64_t> reqDimForOperandDim(inTy.getRank(), -1);
    for (unsigned d = 0; d < logicalRank; ++d)
      if (remap[d] >= 0 && remap[d] < inTy.getRank())
        reqDimForOperandDim[remap[d]] = (int64_t)d;

    for (int64_t od = 0; od < inTy.getRank(); ++od) {
      int64_t reqDim = reqDimForOperandDim[od];
      if (reqDim < 0) {
        // A size-1 operand dim the reshape removed. Only sound because it is
        // size 1; refuse anything else rather than invent a coordinate for it.
        if (inTy.getDimSize(od) != 1)
          return failure();
        out.physSrc.push_back(od);
        out.physOp.push_back((int64_t)CoordOp::Identity);
        out.physArg.push_back(0);
        out.physExtents.push_back(1);
        continue;
      }
      // Carry every requirement entry naming this logical dim.
      bool found = false;
      for (unsigned p = 0; p < req.physSrc.size(); ++p) {
        if (req.physSrc[p] != reqDim)
          continue;
        out.physSrc.push_back(od);
        out.physOp.push_back(req.physOp[p]);
        out.physArg.push_back(req.physArg[p]);
        out.physExtents.push_back(req.physExtents[p]);
        found = true;
      }
      if (!found)
        return failure();
    }
    if ((int64_t)out.physSrc.size() != inTy.getRank())
      return failure();
    return out;
  }
};

/// linalg.broadcast: the requirement PROJECTS onto the carried axes.
///
/// The result has dims the operand does not -- that is what a broadcast is --
/// so the requirement cannot cross unchanged. But it does project: keep the
/// physical dims whose phys_src names a logical axis the operand CARRIES, drop
/// those naming an axis the broadcast ADDS. What survives is a requirement of
/// exactly the operand's logical rank.
///
/// For softmax that projection is trivial -- the surviving dim is Identity, so
/// it asks nothing a rank-1 logical value does not already satisfy. It is still
/// load-bearing: it is what carries the requirement past this op to the reduce,
/// which needs `want` non-null to take the Physical space. Once the reduce's
/// result is in the forward map, the reshapes and then this broadcast acquire a
/// physical operand, which is what lets BroadcastPropagation be asked at all.
/// The real shape work happens there, from the store's marker.
///
/// Declines when a CARRIED axis is split across two physical dims. Projecting
/// would then demand an operand of higher rank than it has -- linalg.broadcast
/// matches its input against the non-broadcast init dims positionally -- so
/// there is no requirement the operand could satisfy. That is the case the
/// consuming elementwise op has to repair instead.
struct BroadcastRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<linalg::BroadcastOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    auto bc = cast<linalg::BroadcastOp>(op);
    // The init carries the RESULT's shape, so it gets the requirement whole --
    // the same split TransposeRequirement makes for its own init.
    if (operand == bc.getInit())
      return req;
    if (operand != bc.getInput())
      return failure();

    // `dimensions` names the added LOGICAL output axes, so the requirement's
    // logical rank must be the result's rank for the two to be comparable.
    unsigned logicalRank = 0;
    for (int64_t d : req.physSrc)
      logicalRank = std::max(logicalRank, (unsigned)(d + 1));
    auto resTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resTy || logicalRank != (unsigned)resTy.getRank())
      return failure();

    llvm::SmallDenseSet<int64_t> added;
    for (int64_t d : bc.getDimensions()) {
      if (d < 0 || d >= (int64_t)logicalRank)
        return failure();
      added.insert(d);
    }

    // A carried axis split across two physical dims cannot be projected: the
    // operand would need a rank it does not have.
    for (unsigned p = 0; p < req.physSrc.size(); ++p)
      if (!added.contains(req.physSrc[p]) &&
          static_cast<CoordOp>(req.physOp[p]) != CoordOp::Identity)
        return failure();

    // Renumber the carried logical axes down, since the added ones are gone.
    llvm::SmallVector<int64_t> logicalRemap(logicalRank, -1);
    int64_t next = 0;
    for (unsigned d = 0; d < logicalRank; ++d)
      if (!added.contains((int64_t)d))
        logicalRemap[d] = next++;

    LayoutRequirement out;
    out.marker = req.marker;
    for (unsigned p = 0; p < req.physSrc.size(); ++p) {
      int64_t mapped = logicalRemap[req.physSrc[p]];
      if (mapped < 0)
        continue;                       // a dim of an added axis: dropped
      out.physSrc.push_back(mapped);
      out.physOp.push_back(req.physOp[p]);
      out.physArg.push_back(req.physArg[p]);
      out.physExtents.push_back(req.physExtents[p]);
    }
    // The projection must have the operand's own rank, or it describes a
    // different value than the one it is about.
    auto inTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!inTy || (int64_t)out.physSrc.size() != inTy.getRank())
      return failure();
    return out;
  }
};

/// ktdp.load: the terminus, where `want` meets `have`. It has no tensor operand,
/// so there is nothing to induce; the pattern exists so a load is a taught op
/// rather than an unruled one.
struct LoadRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<mlir::ktdp::LoadOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    return failure();
  }
};

/// The pattern whose rule covers `op`, or null when this pass has not been
/// taught the op.
const RequirementBackwardPattern *
lookupPattern(Operation *op, const RequirementBackwardPatternSet &patterns) {
  for (const auto &p : patterns)
    if (p->match(op))
      return p.get();
  return nullptr;
}

/// The requirement a physicalized store places on its data tile, or failure when
/// this store is not a seed.
///
/// A store is a seed iff Phase 1 physicalized its access tile. There is no map
/// to ask: `redirectStoreAccessTile` records nothing, it only re-points the
/// operand. What identifies one is the tile's own base -- Phase 1 built the new
/// tile over the physical memory view it registered in `physMemViewToMarker`, so
/// a marker resolving from the tile IS the record that the tile is physical.
/// The same test RewriteStorePattern makes of its own access tile, asked of every
/// store up front rather than one at a time during the rewrite.
llvm::FailureOr<LayoutRequirement> seedFromStore(mlir::ktdp::StoreOp st,
                                                const MarkerByMemView &markers) {
  auto tileOp =
      st.getAccessTile().getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
  if (!tileOp)
    return failure();
  auto found = markers.find(tileOp.getBase());
  if (found == markers.end())
    return failure();
  auto tileTy = dyn_cast<mlir::ktdp::AccessTileType>(tileOp.getResult().getType());
  if (!tileTy)
    return failure();

  triton::SpyreTensorLayoutOp marker = found->second;
  LayoutRequirement req;
  req.marker = marker;
  req.physSrc.assign(marker.getPhysSrc().begin(), marker.getPhysSrc().end());
  req.physOp.assign(marker.getPhysOp().begin(), marker.getPhysOp().end());
  req.physArg.assign(marker.getPhysArg().begin(), marker.getPhysArg().end());
  req.physExtents.assign(tileTy.getShape().begin(), tileTy.getShape().end());
  return req;
}

} // namespace

//===----------------------------------------------------------------------===//
// populateRequirementBackwardPatterns
//===----------------------------------------------------------------------===//

void populateRequirementBackwardPatterns(
    RequirementBackwardPatternSet &patterns) {
  // Order matters only where two patterns could match the same op, and the
  // constraint is the forward side's, for the same reason: every named-op
  // pattern must be asked before the structural elementwise rule. The reshape
  // family and linalg.broadcast are again the load-bearing cases -- one tensor
  // operand each, so "every tensor operand agrees on a shape" holds trivially
  // and the elementwise rule WOULD claim them, crossing a requirement over a
  // change of physical dim count.
  patterns.push_back(std::make_unique<TransposeRequirement>());
  patterns.push_back(std::make_unique<MatmulRequirement>());
  patterns.push_back(std::make_unique<ReduceRequirement>());
  patterns.push_back(std::make_unique<ReshapeRequirement>());
  patterns.push_back(std::make_unique<BroadcastRequirement>());
  patterns.push_back(std::make_unique<LoadRequirement>());
  patterns.push_back(std::make_unique<ElementwiseRequirement>());
}

//===----------------------------------------------------------------------===//
// propagateRequirement
//===----------------------------------------------------------------------===//

void propagateRequirement(Value value, const LayoutRequirement &req,
                          const RequirementBackwardPatternSet &patterns,
                          RequirementAnalysis &result,
                          llvm::SmallVector<Value> &visitStack) {
  // Cycle detection; the visitStack contract is stated on the declaration.
  if (llvm::is_contained(visitStack, value)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  [2A-bwd] cycle on value, requirement stops: " << value
               << "\n");
    return;
  }

  auto [it, inserted] = result.requirements.try_emplace(value, req);
  if (!inserted) {
    // Two requirements reached the same value. Detect and report; the first
    // entry stands and neither is chosen as the answer -- resolution is the
    // doc's owed answer, not this slice's.
    if (!(it->second == req)) {
      result.conflicts.insert(value);
      LLVM_DEBUG(llvm::dbgs()
                 << "  [2A-bwd] conflicting requirements on: " << value << "\n");
    }
    return;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return; // A BlockArgument: the region boundary, as forward stops there too.

  const RequirementBackwardPattern *pattern = lookupPattern(defOp, patterns);
  if (!pattern) {
    // No pattern is this op's rule. Visible rather than defaulted, exactly as
    // getPhysicalizedType leaves an untaught op visible: guessing a rule would
    // silently require the wrong layout of the operands.
    result.opsWithNoRule.push_back(defOp);
    LLVM_DEBUG(llvm::dbgs() << "  [2A-bwd] no backward rule for "
                            << defOp->getName()
                            << "; requirement stops here\n");
    return;
  }

  visitStack.push_back(value);
  llvm::scope_exit popStack([&] { visitStack.pop_back(); });

  for (Value o : defOp->getOperands()) {
    if (!isa<RankedTensorType>(o.getType()))
      continue;
    auto induced = pattern->induce(defOp, value, o, req);
    if (failed(induced))
      continue;
    propagateRequirement(o, *induced, patterns, result, visitStack);
  }
}

//===----------------------------------------------------------------------===//
// runRequirementAnalysis
//===----------------------------------------------------------------------===//

RequirementAnalysis runRequirementAnalysis(ModuleOp module,
                                           const PassContext &ctx) {
  RequirementBackwardPatternSet patterns;
  populateRequirementBackwardPatterns(patterns);

  RequirementAnalysis result;
  unsigned seeds = 0;
  module.walk([&](mlir::ktdp::StoreOp st) {
    auto req = seedFromStore(st, ctx.physMemViewToMarker);
    if (failed(req))
      return;
    ++seeds;
    llvm::SmallVector<Value> visitStack;
    propagateRequirement(st.getDataTile(), *req, patterns, result, visitStack);
  });

  LLVM_DEBUG({
    llvm::dbgs() << "[rewrite-descriptor-layout] Phase 2A backward: " << seeds
                 << " seed(s) reached " << result.requirements.size()
                 << " value(s), " << result.conflicts.size() << " conflict(s), "
                 << result.opsWithNoRule.size() << " unruled op(s)\n";
  });
  (void)seeds;
  return result;
}

} // namespace mlir::triton::ktdp
