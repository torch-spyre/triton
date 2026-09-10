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
    ArrayRef<int64_t> commonShape;
    bool sawTensorOperand = false;
    for (Value o : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(o.getType());
      if (!t)
        continue;
      if (!sawTensorOperand) {
        commonShape = t.getShape();
        sawTensorOperand = true;
      } else if (t.getShape() != commonShape) {
        return false;
      }
    }
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

/// tensor.expand_shape / collapse_shape / reshape: the requirement terminates.
/// The physical dim count changes across a reassociation map, so a
/// per-physical-dim requirement on the result says nothing about the operand --
/// the mirror of why ReshapePropagation declines forward.
///
/// Registered ahead of the structural elementwise rule; see
/// populateRequirementBackwardPatterns.
struct ReshapeRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
               tensor::ReshapeOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    return failure();
  }
};

/// linalg.broadcast: the requirement terminates. The result has dims the operand
/// does not, so no per-dim requirement on it constrains the operand. This is
/// also what keeps softmax's and layernorm's reduce results out of the map: they
/// reach their store only through a broadcast.
struct BroadcastRequirement : RequirementBackwardPattern {
  bool match(Operation *op) const override {
    return isa<linalg::BroadcastOp>(op);
  }

  llvm::FailureOr<LayoutRequirement>
  induce(Operation *op, Value result, Value operand,
         const LayoutRequirement &req) const override {
    return failure();
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
