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
//   (--linalg-fuse-elementwise-ops) supplies `producer && producer->hasOneUse()`
//   -- a claim about the producer's USE COUNT and nothing about what it computes.
//   That is blanket as far as we are concerned, and it over-fuses in two measured
//   ways: two plain computes merge into one generic holding both, and a compute
//   already carrying a shape change in its operand map merges with a downstream
//   compute. The backend takes one compute op per group, so blanket fusion
//   manufactures exactly the multi-compute bodies other machinery then has to
//   undo. Our control function is narrower, and narrow along a different axis: it
//   asks what the producer IS.
//
//   It permits a fusion only when the PRODUCER is pure data movement, tested
//   STRUCTURALLY: a linalg.generic whose body is exactly one operation, a
//   linalg.yield of an input block argument. Nothing computed, only re-indexed.
//
//   Structural rather than an attribute, because nothing in this pipeline labels
//   these ops and nothing would maintain a label if it did. They arrive here from
//   --linalg-generalize-named-ops rewriting the linalg.broadcast and
//   linalg.transpose LowerComputeOps emits, neither of which records its
//   provenance; and RewriteDescriptorLayoutGeneric REBUILDS every generic on a
//   physicalized chain, so any label would additionally have to be copied by a
//   pass with no other reason to know it exists. A stale or missing label would
//   not merely mislead a reader: this predicate gates every fusion decision, so
//   it would silently change the emitted program. A property of the op cannot go
//   stale.
//
//   The predicate deliberately does NOT check hasOneUse. A data-movement generic
//   feeding two consumers is folded into each and then dies, and that is what we
//   want -- the goal is that NO data-movement generic survives, not that each is
//   folded at most once. Duplicating a coordinate change costs nothing, because
//   there is no body to duplicate: what the second consumer gains is an operand
//   map, not an op. This is the one place our control function is LOOSER than
//   upstream's, whose hasOneUse would decline exactly that fusion.
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
// The one reshape absorbed here, and when it is even on the path:
//   `tt.broadcast` lowers to `tensor.collapse_shape` + `linalg.broadcast`,
//   because linalg.broadcast takes its input rank-reduced. But that collapse only
//   SURVIVES to this pass when the broadcast's source already carries the unit
//   dim as a real tensor dim -- a `ktdp.load` of `tensor<64x1xf16>` off a narrow
//   access tile, say. Where the unit dim was itself just inserted, the usual
//   `tt.expand_dims` on a rank-1 reduce result, the collapse is the exact inverse
//   of that `tensor.expand_shape` and the pair cancels in the canonicalize that
//   closes the `ktir` stage; what reaches this pass is then a bare
//   `linalg.broadcast` over a rank-1 value with no tensor op in front of it at
//   all (measured). So the absorber is not always on the path. It is on it for
//   the loaded-statistic shape, which is the shape that needed it.
//
//   Where it is on the path it is load-bearing: being a tensor op rather than a
//   generic it blocks the composition, and fusing the broadcast alone leaves the
//   consumer reading a rank-1 value that no longer matches the rank-2 access tile
//   its load came from -- dbo-opt then reports the same diagnostic as above
//   (measured).
//
// Why upstream's reshape patterns are NOT used:
//   Upstream exposes `populateFoldReshapeOpsBy{Expansion,Collapsing}Patterns`
//   with the same control-function type, but they solve a different problem: a
//   general reshape is not expressible as an operand map, so they change the
//   consumer's ITERATION SPACE and push a reshape onto the other operands. On
//   this kernel that emits `tensor.expand_shape` on the physicalized data path,
//   and RewriteDescriptorLayoutGeneric rejects it outright:
//
//     error: this op reads a value the rewrite retyped, but the rewrite restates
//     only linalg.generic, so this op still names the logical type
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
// What is NOT absorbed, and is left silently:
//   LowerComputeOps' Group A emits five kinds of tensor shape op. This pass
//   handles one of them:
//
//     A2  tt.reshape      tensor.reshape                 no: an arbitrary
//                                                        reshape is not affine
//     A3  tt.expand_dims  tensor.expand_shape            no: NOT YET
//                                                        IMPLEMENTED. A unit-dim
//                                                        expand is exactly as
//                                                        absorbable as the
//                                                        collapse below, by the
//                                                        mirror-image argument;
//                                                        only the pattern is
//                                                        missing.
//     A4  tt.broadcast    collapse_shape +               YES, in the unit-dim
//                         linalg.broadcast               case described above
//     A6  tt.join         expand_shape x2 +              no: a concat is not
//                         tensor.concat                  expressible as one
//                                                        operand map at all
//     A7  tt.split        extract_slice x2 +             the collapse yes; the
//                         collapse_shape x2              extract_slice needs no
//                                                        absorber, the
//                                                        scheduler's
//                                                        findIndexingMapForLoadResult
//                                                        looking through an
//                                                        extract_slice and
//                                                        nothing else
//
//   Every unabsorbed case is left SILENTLY -- this pass emits no diagnostic, and
//   nothing downstream emits one either. RewriteDescriptorLayoutGeneric's
//   precondition sweep, the one that produces the "restates only linalg.generic"
//   error quoted above, walks only the users of a GENERIC'S RESULTS
//   (`for (unsigned i = numIns; i < numOps; ++i)` over the operand range past the
//   ins), so a `tensor.reshape` consuming a `ktdp.load` directly is invisible to
//   it. A kernel using tt.reshape or tt.join therefore fails later, in dbo-opt,
//   with a diagnostic that does not name the reshape.
//
// Position in the pipeline: after unalias_linalg_outs, and in any case after
// convert_elementwise_to_linalg and linalg_generalize_named_ops -- fusion matches
// generic -> generic only, so a named producer or consumer blocks it whatever the
// control function says. Before rewrite_descriptor_layout_generic, which must see
// the folded maps so that no data-movement generic is left for it to linearize.
// Nothing to say about lower_inter_tile: that pass runs in the `ktir` stage, a
// whole stage earlier, so this pass necessarily runs after it -- harmlessly,
// since by then every !ktdp.tile_future has been retired.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

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

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_FOLDDATAMOVEMENTGENERICS
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::spyre

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
    : public mlir::triton::spyre::impl::FoldDataMovementGenericsBase<
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

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createFoldDataMovementGenericsPass() {
  return std::make_unique<FoldDataMovementGenericsPass>();
}

} // namespace mlir::triton::spyre
