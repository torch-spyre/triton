//===- FoldDataMovementGenerics.cpp - Fold re-indexing into consumers -----===//
//
// Folds a *data-movement* op into the indexing map of the op that consumes it,
// so no op whose only effect is to re-index survives into the emitted KTIR.
//
// Why:
//   Every compute op reaches this pass as a linalg.generic. A coordinate change
//   -- broadcast, transpose, inserted unit dim -- arrives as its own generic
//   whose body computes nothing: it yields its input, and only its indexing maps
//   differ. That op has no descriptor and no layout marker, so
//   RewriteDescriptorLayoutGeneric leaves its result LOGICAL while its consumer
//   is physical; the consumer's operand map then has to bridge the two and the
//   pass folds a LINEARIZATION into it (`(d0, d1, d2) -> (d1, d0 * 64 + d2)`).
//   dbo-opt cannot schedule that:
//
//     error: could not locate matching linalg operand to project loop IVs and
//     tile sizes through; the data_transfer rank would not match the underlying
//     memref
//
//   Folded into the consumer instead, the coordinate change becomes part of an
//   operand map the layout pass restates at physical rank like any other -- a
//   statistic read at a constant lane comes out as `(d0, d1, d2) -> (d1, 0)`,
//   which is what hand-written reference KTIR states.
//
// The mechanism is upstream's, the POLICY is ours:
//   Elementwise fusion is upstream's `populateElementwiseOpsFusionPatterns`,
//   which takes a `ControlFusionFn` deciding each fusion. Upstream's own pass
//   always accepts, which over-fuses for us in two measured ways: two plain
//   computes merge into one generic holding both, and a compute already carrying
//   a shape change in its operand map merges with a downstream compute. The
//   backend takes one compute op per group, so blanket fusion manufactures
//   exactly the multi-compute bodies other machinery then has to undo.
//
//   Our control function permits a fusion only when the PRODUCER is pure data
//   movement, tested STRUCTURALLY: a linalg.generic whose body is exactly one
//   operation, a linalg.yield of an input block argument. Nothing computed, only
//   re-indexed. Deliberately not an attribute -- a stale or missing label would
//   silently change which fusions happen and therefore change the emitted
//   program, while a property of the op cannot go stale. In particular the `doc`
//   string LowerComputeOps attaches ("tt.broadcast", ...) is for humans and
//   nothing here reads it.
//
//   Two things fall out rather than needing cases of their own:
//     - A generic that is both a compute and a shape change is declined as a
//       producer and stays eligible as a consumer. Correct in both roles.
//     - The predicate keys on the producer, so a REDUCTION consumer is fine: the
//       shape change folds into its operand map while its iterator_types and its
//       single-compute body are untouched. That also keeps DropReductionInitFill's
//       precondition (a reduce body of exactly two ops) true, since a pure
//       data-movement producer contributes no body op.
//
// The one reshape, and why upstream's reshape patterns are NOT used:
//   `tt.broadcast` lowers to `tensor.collapse_shape` + `linalg.broadcast`,
//   because linalg.broadcast takes its input rank-reduced. So a unit-dim
//   `tensor.collapse_shape` sits in front of the broadcast generic and, being a
//   tensor op rather than a generic, blocks the composition: fusing the
//   broadcast alone leaves the consumer reading a rank-1 value that no longer
//   matches the rank-2 access tile its load came from, and dbo-opt reports the
//   same diagnostic as before (measured).
//
//   Upstream exposes `populateFoldReshapeOpsBy{Expansion,Collapsing}Patterns`
//   with the same control-function type, but they solve a different problem: a
//   general reshape is not expressible as an operand map, so they change the
//   consumer's ITERATION SPACE and push a reshape onto the other operands. On
//   this kernel that emits `tensor.expand_shape` on the physicalized data path,
//   and RewriteDescriptorLayoutGeneric rejects it outright:
//
//     error: this op reads a value on a physicalized chain, but the rewrite
//     restates only linalg.generic; spell this op as one
//
//   A reshape that only inserts or drops UNIT dims is the special case that IS
//   expressible as an operand map -- a constant 0 index in the unit position --
//   and it is the only reshape our emission puts in front of a data-movement
//   generic. AbsorbUnitDimCollapse below does exactly that substitution and
//   nothing else; anything that merges two non-unit dims is a genuine
//   linearization and is left alone.
//
//   It is deliberately not gated on what consumes the reshape, so the fixpoint
//   does not depend on the order the greedy driver picks: absorbed first, the
//   broadcast generic reads the uncollapsed tensor and fusion composes that map
//   into the consumer; fused first, the consumer reads the collapsed value and
//   the absorption then rewrites its map. Both reach the same IR.
//
// Position in the pipeline: after unalias_linalg_outs (fusion needs generics on
// both sides, so after convert_elementwise_to_linalg and after
// linalg_generalize_named_ops) and before lower_inter_tile /
// rewrite_descriptor_layout_generic (the layout pass must see the folded maps,
// so that no data-movement generic is left for it to linearize).
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_FOLDDATAMOVEMENTGENERICS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// True iff `op` is a `linalg.generic` that only re-indexes its input: its body
/// is exactly one operation, a `linalg.yield` of a block argument belonging to
/// one of its `ins`.
///
/// This is the whole fusion policy, and it is a property of the op rather than a
/// claim about it -- see the header on why it must not be an attribute.
///
/// Requiring the yielded value to be an INPUT block argument, not merely any
/// block argument, excludes a generic that yields its `outs` argument: that op
/// forwards its init rather than its input, and the init of a `tensor.empty` is
/// unspecified, so it is not a coordinate change on data.
bool isPureDataMovement(Operation *op) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic)
    return false;
  Block *body = generic.getBlock();
  if (!body || body->getOperations().size() != 1)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield->getNumOperands() != 1)
    return false;
  auto arg = dyn_cast<BlockArgument>(yield->getOperand(0));
  if (!arg || arg.getOwner() != body)
    return false;
  // Block arguments are the `ins` first, then the `inits`.
  return arg.getArgNumber() < static_cast<unsigned>(generic.getNumDpsInputs());
}

/// Rewrites a `linalg.generic` input operand defined by a unit-dim-only
/// `tensor.collapse_shape` to read the uncollapsed value, substituting a
/// constant 0 for each unit dim the collapse absorbed.
///
/// A group of the reassociation may merge any number of unit dims with AT MOST
/// ONE non-unit dim. Under that condition the collapse is exact as an index
/// substitution: the collapsed index equals the non-unit dim's index, and every
/// other dim of the group can only be read at 0. A group merging two non-unit
/// dims is a genuine linearization, cannot be written as an affine operand map,
/// and is left alone -- which is also why this is not upstream's reshape
/// folding, whose answer to that case is to change the iteration space.
///
/// `outs` is not touched: repointing it would change the op's result type, which
/// is a different rewrite with a different consumer to satisfy.
struct AbsorbUnitDimCollapse : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    for (OpOperand *operand : op.getDpsInputOperands()) {
      auto collapse = operand->get().getDefiningOp<tensor::CollapseShapeOp>();
      if (!collapse)
        continue;

      ArrayRef<int64_t> srcShape = collapse.getSrcType().getShape();
      AffineMap map = op.getMatchingIndexingMap(operand);

      SmallVector<AffineExpr> results;
      bool foldable = true;
      for (auto [group, collapsedExpr] :
           llvm::zip_equal(collapse.getReassociationIndices(),
                           map.getResults())) {
        // The dim of the group that carries the collapsed index, if any. With
        // more than one the collapse is a linearization: bail on the whole
        // operand rather than folding part of it.
        int64_t carrier = -1;
        for (int64_t dim : group) {
          if (srcShape[dim] == 1)
            continue;
          if (carrier >= 0) {
            foldable = false;
            break;
          }
          carrier = dim;
        }
        if (!foldable)
          break;
        // An all-unit group collapses to an extent-1 dim, so every dim of it --
        // including the one the collapsed expression addressed -- can only be
        // read at 0.
        for (int64_t dim : group)
          results.push_back(dim == carrier ? collapsedExpr
                                           : getAffineConstantExpr(0, ctx));
      }
      if (!foldable)
        continue;

      SmallVector<AffineMap> maps = op.getIndexingMapsArray();
      maps[operand->getOperandNumber()] =
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, ctx);

      // The element type is unchanged, so the body and its block arguments are
      // untouched and this is a genuine in-place edit.
      rewriter.modifyOpInPlace(op, [&]() {
        operand->set(collapse.getSrc());
        op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      });
      return success();
    }
    return failure();
  }
};

struct FoldDataMovementGenericsPass
    : public mlir::triton::ktdp::impl::FoldDataMovementGenericsBase<
          FoldDataMovementGenericsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // Upstream does the map composition; the control function decides which
    // fusions are allowed to happen at all.
    linalg::ControlFusionFn onlyDataMovementProducers =
        [](OpOperand *fusedOperand) {
          return isPureDataMovement(fusedOperand->get().getDefiningOp());
        };
    linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                onlyDataMovementProducers);
    patterns.add<AbsorbUnitDimCollapse>(ctx);

    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::ktdp {

std::unique_ptr<OperationPass<ModuleOp>> createFoldDataMovementGenericsPass() {
  return std::make_unique<FoldDataMovementGenericsPass>();
}

} // namespace mlir::triton::ktdp
