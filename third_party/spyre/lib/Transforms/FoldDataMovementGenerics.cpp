//===- FoldDataMovementGenerics.cpp - Fold re-indexing into consumers -----===//
//
// Folds a *data-movement* op into the indexing map of the op that consumes it,
// so no op whose only effect is to re-index survives into the emitted KTIR --
// and REJECTS the ones it cannot fold, where leaving them would emit a program
// the scheduler silently cannot take.
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
// So the standing goal is: leave NOTHING between a load and a compute except
// indexing maps. Three parts, and they are separable on purpose --
//
//   resultToSourceMap    given one shape op, the map from its RESULT
//            coordinates to its SOURCE coordinates. Knows about reassociations
//            and nothing about consumers. The genuine map, floordiv and mod
//            included; it does not decide whether anyone can use it.
//   AbsorbCoordinateOp   for each `ins` operand a map exists for, COMPOSE that
//            map with the consumer's operand map, repoint the operand at the
//            shape op's source, and set the composed map.
//   the projection check the composed map must have every result a bare loop dim
//            or a constant. Not the raw map -- see below.
//
// WHY THE CHECK BELONGS ON THE COMPOSED MAP.
//   The reason to decline a linearizing collapse is NOT that it is inexpressible.
//   AffineExpr has FloorDiv and Mod, so `[64,2] -> [128]` is perfectly writable
//   as `(d0) -> (d0 floordiv 2, d0 mod 2)` and this file emits exactly that. The
//   real reason is downstream: the scheduler projects loop IVs and tile sizes
//   through an operand map, and it cannot project them through floordiv or mod.
//   That is a property of the map the CONSUMER ends up holding, not of the
//   reassociation -- and the two can differ, because composition folds. A
//   linearizing collapse read at a constant coordinate composes to
//   `(d0, d1) -> (0, 0)`, which projects fine and is absorbed. Checking the
//   reassociation instead would decline it for a linearization that is not there
//   in the map anyone would schedule.
//
// THE STORE SIDE IS REJECTION ONLY, NEVER ABSORPTION.
//   Everything above is the `ins` side. A shape op between a generic and the
//   `ktdp.store` that consumes its result is the same hazard mirrored, and this
//   pass refuses it -- but never absorbs it, and that is a decision rather than
//   an unfinished case. Absorbing there is not a variation on
//   AbsorbCoordinateOp: it would have to change the generic's RESULT type,
//   replace the `tensor.empty` behind `outs`, restate the outs map and repoint
//   the store's data operand -- and it needs the map in the OPPOSITE direction,
//   which is an inversePermutation that is null for exactly the linearizing
//   cases this is about. AbsorbCoordinateOp's `outs` is untouched, as it says.
//
//   The eventual fix is an open choice between doing that and teaching
//   RewriteDescriptorLayoutGeneric's findLayoutForResult to look through a
//   unit-dim reshape. The latter is arguably the better home: `ktdp.store`
//   carries no indexing map, so on the store side there is nothing to absorb
//   INTO, and what goes wrong is a walk that stops one op too early.
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
// The shape ops absorbed, and when they are even on the path:
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
//   The `tensor.expand_shape` from `tt.expand_dims` is absorbed by the same
//   machinery and needs no case of its own -- it is the mirror image, its
//   reassociation reading source dim -> result dims instead of result dim ->
//   source dims. It survives to this pass where the collapse that would have
//   cancelled it is not there, which is the `softmax_2pass` shape:
//   `tensor<4xf32>` expanded to `tensor<4x1xf32>` feeding a rank-2 loop-carried
//   accumulator.
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
//   The absorber keeps the iteration space and moves the coordinate change into
//   one operand's map, which is the whole of the difference.
//
//   It is deliberately not gated on what consumes the reshape, so the fixpoint
//   does not depend on the order the greedy driver picks: absorbed first, the
//   broadcast generic reads the uncollapsed tensor and fusion composes that map
//   into the consumer; fused first, the consumer reads the collapsed value and
//   the absorption then rewrites its map. Both reach the same IR.
//
// WHAT IS NOT ABSORBED, AND WHY REJECTING IT IS GATED.
//   Two shape ops have no result-to-source map at all. `tensor.reshape` takes its
//   shape as a tensor OPERAND, so there is no static structure to derive a map
//   from; `tensor.concat` selects between operands per coordinate, which one
//   operand map cannot state whatever the coordinates are. A third case has a map
//   that linearizes, which the projection check declines.
//
//   Left in front of a generic the layout pass will physicalize, each of those is
//   a program that fails later, in dbo-opt, with a diagnostic that names neither
//   the op nor this pass. So they are rejected here -- and ONLY on such a path.
//
//   WHY THE REJECTION IS GATED, and the reason is forward-looking rather than
//   protective. As annotation coverage grows toward everything on a device path
//   being physicalized, the gate fires more often and converges to the ungated
//   behaviour by itself, so it never needs removing. Shipping ungated would
//   instead mean ADDING a gate later, the moment an unannotated path has to be
//   legal. It is kept pending the LX roundtrip, at which point the better move
//   may be to INVERT it -- into an assertion that anything on a device path IS
//   annotated.
//
//   THE GATE is the layout rewrite's own scope, asked with the layout rewrite's
//   own traversals (Dialect/KTDP/Utils, shared rather than copied): seed from
//   every `tts.tensor_layout`-annotated `ktdp.construct_memory_view`, reach its
//   loads and stores through its access tiles, and take the generics one hop off
//   those -- `collectAdjacentGenerics`. One hop, not a transitive closure, which
//   is what that pass actually rewrites.
//
//   Three seeds, because the hazard has two sides and a forward-only walk
//   reproduces an existing blind spot on both:
//
//     BACKWARD from each `ins` operand of a to-be-rewritten generic, through
//     every generic that is not itself in the rewrite set, stopping at anything
//     that is not a coordinate restatement. This is the half that matters. It
//     catches a re-indexing op whose OWN load is on an UNANNOTATED view -- which
//     is the real `stat_chain_on_stick` situation, its statistic read view
//     deliberately carrying no layout because its logical shape already is its
//     physical one. Nothing else looks there: RewriteDescriptorLayoutGeneric
//     calls retypeToPhysical only for an operand that HAS a layout, so an operand
//     that has none is bridged by rebuildMap with no check at all.
//
//     BACKWARD from the DATA of each store over an annotated view, which is the
//     exact mirror. findLayoutForResult walks a generic's result users for a
//     `ktdp.store` DIRECTLY -- `dyn_cast<StoreOp>`, `continue` otherwise -- so a
//     shape op between the generic and the store makes it return null, the outs
//     gets CoordOp::Identity, and the LINEARIZATION lands on the outs map
//     instead of an ins map. `gather__1d` carries an 8x1 -> 8 `tt.reshape` into a
//     store, so the shape is real and in the tree today -- on a kernel this pass
//     never sees, since it stops at the `ktir` stage. Rejection only on this
//     side, never absorption: see THE STORE SIDE, above.
//
//     FORWARD, one hop off each physicalized load, for a `load -> reshape`
//     whose result goes somewhere neither of the backward walks reaches. Where it
//     does go to a store the store seed reaches it too, and where it goes to a
//     generic this is REDUNDANT with RewriteDescriptorLayoutGeneric's own
//     checkConsumersAreRewritable, which already refuses a non-generic non-store
//     consumer of such a load. It is kept because it fires a stage earlier and
//     says which property of the op is the problem, where that check can only say
//     the op is not a linalg.generic.
//
//   REJECTION IS DERIVED, not a declared op list. The condition is exactly
//   *absorption failed* and *on a physicalized path*, and "absorption failed" is
//   asked of `resultToSourceMap` -- the same function the absorber asks. Teaching
//   that function a static `tensor.reshape` therefore stops the rejection for it
//   with nothing here to update. What `resultToSourceMap` does own is the
//   three-way distinction that makes this possible: an op is not a coordinate
//   restatement at all (a compute, an scf result, a cast -- nothing here is about
//   it), or it is one with a usable map, or it is one without. The middle
//   category is the absorber's inventory of what it handles; the last is its
//   inventory of what it has yet to. Neither is a list of things to refuse.
//
//   Two limits worth naming rather than hiding. The backward walk stops at any op
//   that is neither a coordinate restatement nor a linalg.generic, so a reshape
//   behind an `scf.for` result is not found -- and could not be absorbed either,
//   so reporting it would name a problem this pass cannot describe. And the
//   linearization hazard is wider than re-indexing ops: ANY unphysicalized
//   operand of a generic whose domain got split is bridged by rebuildMap's
//   linearizeStickLane, a plain compute chain included. That is not this gate's,
//   because absorption is not its fix.
//
// SEEING WHAT IT DECIDED: every decision here is a silent accept-or-decline, so
// `--debug-only=fold-data-movement-generics` traces the decisions rather than the
// control flow -- one line per restatement asked for and the answer, one per
// operand the absorber considered with the composition spelled out, the gate's
// seeds and scope, and each rejection with the PATH that reached it. The path is
// the part worth having: a bare "rejected op X" leaves the reader to re-derive
// why X was on a physicalized path at all.
//
// Position in the pipeline: after unalias_linalg_outs, and in any case after
// convert_elementwise_to_linalg and linalg_generalize_named_ops -- fusion matches
// generic -> generic only, so a named producer or consumer blocks it whatever the
// control function says. Before rewrite_descriptor_layout_generic, which must see
// the folded maps so that no data-movement generic is left for it to linearize,
// and which is also the pass whose scope the gate above predicts. Nothing is owed
// to lower_inter_tile, which runs in the `ktir` stage a whole stage earlier.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"

#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#define DEBUG_TYPE "fold-data-movement-generics"

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
/// Terse on purpose: the greedy driver asks this once per candidate operand per
/// sweep, so a chatty decline would bury everything else in the trace. One line,
/// naming which of the five tests said no.
bool isPureDataMovement(Operation *op) {
  auto decline = [&](const char *test) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] not data movement (" << test
                            << "): " << op->getName() << " at " << op->getLoc()
                            << "\n");
    return false;
  };
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic) {
    LLVM_DEBUG(if (op) llvm::dbgs()
               << "[" DEBUG_TYPE "] not data movement (not a generic): "
               << op->getName() << " at " << op->getLoc() << "\n");
    return false;
  }
  Block *body = generic.getBlock();
  if (!body || body->getOperations().size() != 1)
    return decline("body is not one op");
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield->getNumOperands() != 1)
    return decline("terminator is not a single-value yield");
  auto arg = dyn_cast<BlockArgument>(yield->getOperand(0));
  if (!arg || arg.getOwner() != body)
    return decline("does not yield a block argument");
  // Block arguments are the `ins` first, then the `inits`.
  if (arg.getArgNumber() >= static_cast<unsigned>(generic.getNumDpsInputs()))
    return decline("yields its outs argument, not an ins");
  return true;
}

//===----------------------------------------------------------------------===//
// Piece 1: the result-to-source map of one shape op
//===----------------------------------------------------------------------===//

/// One op's coordinate restatement, as far as this pass can state it.
///
/// THREE answers, not two, and the third is what lets Part 2's rejection be
/// derived rather than declared:
///
///   `source` null          -- not a coordinate restatement at all. A compute, an
///                             scf result, a cast, a load. Nothing in this pass
///                             is about it, and it is neither absorbed nor
///                             refused.
///   `source` set, `why` empty
///                          -- a restatement whose map is derivable AND is a
///                             projected coordinate map. Absorbable.
///   `source` set, `why` set
///                          -- a restatement this pass cannot hand to a
///                             scheduler. `map` may still be present (a genuine
///                             linearization) or absent (no map exists at all);
///                             `why` says which, in the words the diagnostic
///                             uses.
///
/// `why` is filled HERE, by the code that read the reassociation, because that is
/// the only place that knows how many non-unit dims a group fused. The absorber
/// does not consult it: its authority is the projection check on the COMPOSED
/// map, which can accept a map `why` describes as linearizing (composition folds
/// a floordiv against a constant). The two never disagree in a way that matters,
/// because the absorber runs first and removes what it accepts, so the gate only
/// ever sees ops the absorber declined.
struct CoordinateRestatement {
  /// The value a consumer should read instead of this op's result.
  Value source;
  /// This op's RESULT coordinates -> `source`'s coordinates. Null when no such
  /// map is derivable.
  AffineMap map;
  /// Why this restatement is not a projected coordinate map; empty when it is.
  std::string why;

  bool isReindex() const { return source != nullptr; }
};

/// Not a coordinate restatement.
CoordinateRestatement notReindex() { return CoordinateRestatement{}; }

/// A restatement with no derivable map.
CoordinateRestatement noMap(Value source, StringRef why) {
  return CoordinateRestatement{source, AffineMap(), why.str()};
}

/// True iff every result of `map` is a bare loop dim or a constant -- the form a
/// loop IV can be projected through. No floordiv, no mod, no mul-add.
bool isProjectedCoordinateMap(AffineMap map) {
  return llvm::all_of(map.getResults(), [](AffineExpr e) {
    return isa<AffineDimExpr, AffineConstantExpr>(e);
  });
}

/// The extents of `shape` at `dims`, multiplied -- the row-major stride of the
/// dim just outside them. Every entry must be static; the caller checks.
int64_t productOf(ArrayRef<int64_t> shape, ArrayRef<int64_t> dims) {
  int64_t p = 1;
  for (int64_t d : dims)
    p *= shape[d];
  return p;
}

/// The dims of `group` whose extent in `shape` is not 1. A unit dim carries no
/// coordinate: it can only ever be read at 0, and it contributes nothing to a
/// linearization either.
SmallVector<int64_t> nonUnitDims(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> group) {
  SmallVector<int64_t> out;
  for (int64_t d : group)
    if (shape[d] != 1)
      out.push_back(d);
  return out;
}

/// `tensor.collapse_shape`: RESULT coords -> SRC coords.
///
/// The reassociation has one group per RESULT dim, listing the SRC dims it
/// merged, so each result dim's index has to be DELINEARIZED back over the src
/// extents of its group. Three cases per group, and only the first two give a map
/// a loop IV can be projected through:
///
///   no non-unit dim   the group collapses to an extent-1 dim, so every src dim
///                     of it -- including the one the collapsed index addressed
///                     -- can only be read at 0.
///   one non-unit dim  that dim carries the collapsed index unchanged, every
///                     other dim of the group reads at 0. This is the
///                     `tt.broadcast` shape, `[64,1] -> [64]` giving
///                     `(d0) -> (d0, 0)`.
///   two or more       a genuine linearization, and the map says so:
///                     `[64,2] -> [128]` gives
///                     `(d0) -> (d0 floordiv 2, d0 mod 2)`. Emitted, not
///                     refused -- the projection check on the composed map is
///                     what decides, and composition can fold this away.
CoordinateRestatement collapseRestatement(tensor::CollapseShapeOp op) {
  MLIRContext *ctx = op.getContext();
  ArrayRef<int64_t> srcShape = op.getSrcType().getShape();
  unsigned srcRank = srcShape.size();
  unsigned resultRank = op.getResultType().getRank();
  // Bound to a local: getReassociationIndices returns by value, and enumerating
  // the temporary directly is a lifetime question nobody should have to ask.
  auto groups = op.getReassociationIndices();

  SmallVector<AffineExpr> results(srcRank);
  std::string why;
  for (auto [g, group] : llvm::enumerate(groups)) {
    AffineExpr collapsed = getAffineDimExpr(g, ctx);
    SmallVector<int64_t> nonUnit = nonUnitDims(srcShape, group);
    if (nonUnit.size() <= 1) {
      for (int64_t d : group)
        results[d] = (!nonUnit.empty() && d == nonUnit.front())
                         ? collapsed
                         : getAffineConstantExpr(0, ctx);
      continue;
    }
    // Linearizing. Every dim of the group must have a static extent for the
    // strides to exist at all; a unit dim is static by definition, so this only
    // asks about the non-unit ones.
    if (llvm::any_of(nonUnit, [&](int64_t d) {
          return ShapedType::isDynamic(srcShape[d]);
        }))
      return noMap(op.getSrc(),
                   "reassociation group " + std::to_string(g) +
                       " fuses a dynamic extent, whose "
                       "delinearization strides are not stateable");
    if (why.empty())
      why = "reassociation group " + std::to_string(g) + " fuses " +
            std::to_string(nonUnit.size()) +
            " non-unit dims, which only a linearizing map can state";
    for (auto [idx, d] : llvm::enumerate(group)) {
      if (srcShape[d] == 1) {
        results[d] = getAffineConstantExpr(0, ctx);
        continue;
      }
      int64_t stride =
          productOf(srcShape, ArrayRef<int64_t>(group).drop_front(idx + 1));
      AffineExpr e = stride == 1 ? collapsed : collapsed.floorDiv(stride);
      // The outermost non-unit dim needs no modulus: the collapsed index is
      // already bounded by the product of the group's extents.
      if (d != nonUnit.front())
        e = e % srcShape[d];
      results[d] = e;
    }
  }

  // A src dim in no group would leave its coordinate unstated. The op's own
  // verifier rules this out; be explicit rather than building a short map.
  if (llvm::any_of(results, [](AffineExpr e) { return !e; }))
    return noMap(op.getSrc(),
                 "its reassociation leaves a source dim unnamed, leaving no "
                 "map over all of them");

  return CoordinateRestatement{
      op.getSrc(), AffineMap::get(resultRank, /*symbolCount=*/0, results, ctx),
      why};
}

/// `tensor.expand_shape`: RESULT coords -> SRC coords.
///
/// The mirror image of the collapse, and the asymmetry is in the reassociation
/// rather than in the reasoning: here one group per SRC dim lists the RESULT dims
/// it was split into, so each src coordinate is a LINEARIZATION of its group's
/// result coordinates rather than a delinearization of one.
///
///   no non-unit result dim   the src dim has extent 1; its coordinate is 0.
///   one non-unit result dim  that dim carries the src coordinate and the
///                            inserted unit dims drop out. This is the
///                            `tt.expand_dims` shape, `[4] -> [4,1]` giving
///                            `(d0, d1) -> (d0)`.
///   two or more              a genuine linearization, `d_i * stride + d_j`,
///                            emitted and then declined by the projection check.
CoordinateRestatement expandRestatement(tensor::ExpandShapeOp op) {
  MLIRContext *ctx = op.getContext();
  ArrayRef<int64_t> resultShape = op.getResultType().getShape();
  unsigned resultRank = resultShape.size();
  auto groups = op.getReassociationIndices();

  SmallVector<AffineExpr> results;
  std::string why;
  for (auto [s, group] : llvm::enumerate(groups)) {
    SmallVector<int64_t> nonUnit = nonUnitDims(resultShape, group);
    if (nonUnit.empty()) {
      results.push_back(getAffineConstantExpr(0, ctx));
      continue;
    }
    if (nonUnit.size() == 1) {
      results.push_back(getAffineDimExpr(nonUnit.front(), ctx));
      continue;
    }
    if (llvm::any_of(nonUnit, [&](int64_t d) {
          return ShapedType::isDynamic(resultShape[d]);
        }))
      return noMap(op.getSrc(),
                   "reassociation group " + std::to_string(s) +
                       " expands into a dynamic extent, whose "
                       "linearization strides are not stateable");
    if (why.empty())
      why = "reassociation group " + std::to_string(s) + " expands into " +
            std::to_string(nonUnit.size()) +
            " non-unit dims, which only a linearizing map can state";
    AffineExpr e = getAffineConstantExpr(0, ctx);
    for (auto [idx, d] : llvm::enumerate(group)) {
      if (resultShape[d] == 1)
        continue;
      int64_t stride =
          productOf(resultShape, ArrayRef<int64_t>(group).drop_front(idx + 1));
      AffineExpr term = getAffineDimExpr(d, ctx);
      e = e + (stride == 1 ? term : term * stride);
    }
    results.push_back(e);
  }

  return CoordinateRestatement{
      op.getSrc(), AffineMap::get(resultRank, /*symbolCount=*/0, results, ctx),
      why};
}

/// The map from `shapeOp`'s RESULT coordinates to its SOURCE coordinates, if
/// `shapeOp` is a coordinate restatement at all.
///
/// This function is the pass's whole inventory of what a coordinate restatement
/// IS, which is what makes both the absorption and the rejection follow from one
/// place. Teaching it a new op makes that op absorbable and stops it being
/// refused, with nothing else to edit.
///
/// The five tensor shape ops LowerComputeOps' Group A can emit:
///
///   A2  tt.reshape      tensor.reshape       no map: the shape is a tensor
///                                            OPERAND, so there is no static
///                                            structure to read one off
///   A3  tt.expand_dims  tensor.expand_shape  yes, expandRestatement
///   A4  tt.broadcast    collapse_shape +     yes, collapseRestatement
///                       linalg.broadcast
///   A6  tt.join         expand_shape x2 +    no map: a concat selects between
///                       tensor.concat        operands per coordinate
///   A7  tt.split        extract_slice x2 +   the collapse yes; the
///                       collapse_shape x2    extract_slice deliberately NOT a
///                                            restatement -- see below
///
/// WHY `tensor.extract_slice` IS NOT HERE, and it is not the reason one might
/// expect. Its coordinate map is perfectly derivable -- static offsets and unit
/// strides give `(d0, d1) -> (d0 + o0, d1 + o1)`. What is not derivable is its
/// EXTENT change, and a linalg operand map cannot state one: linalg infers its
/// loop bounds from the operand shapes THROUGH the indexing maps, so repointing
/// an operand at the larger, unsliced source under any map makes the inferred
/// extents inconsistent and fails linalg's own verifier. A slice crops; an
/// operand map re-indexes. Those are different powers, and no amount of affine
/// expression closes the gap.
///
/// Two further facts make the omission cost nothing. The offset part would be
/// declined by the projection check anyway (`d0 + o0` is neither a bare dim nor a
/// constant), so a sound version of this case would absorb only a zero-offset
/// slice, which is the case the canonicalizer already removes. And the scheduler
/// already looks through the op: `findIndexingMapForLoadResult` tries the direct
/// operand and then exactly one alternative, an extract_slice user -- verified in
/// its source, not assumed. So it is the one shape op downstream handles, and
/// there are ZERO occurrences of it across all 163 fixture artifacts. It is left
/// as "not a coordinate restatement", which means it is neither absorbed nor
/// refused: exactly the status quo, deliberately.
CoordinateRestatement resultToSourceMap(Operation *shapeOp) {
  CoordinateRestatement r = [&]() -> CoordinateRestatement {
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(shapeOp))
      return collapseRestatement(collapse);
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(shapeOp))
      return expandRestatement(expand);
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(shapeOp))
      return noMap(reshape.getSource(),
                   "no affine map from its result coordinates to its source "
                   "coordinates is derivable -- its shape is an operand rather "
                   "than a reassociation");
    if (auto concat = dyn_cast<tensor::ConcatOp>(shapeOp))
      return noMap(concat.getInputs().front(),
                   "it selects between operands per coordinate, which one "
                   "operand map cannot state");
    return notReindex();
  }();

  // The single most useful line in the pass: one op, and the answer the whole
  // rest of the file is derived from. A non-restatement says nothing -- every
  // load, empty, constant and cast the gate's walk touches goes through here.
  LLVM_DEBUG(if (r.isReindex()) {
    llvm::dbgs() << "[" DEBUG_TYPE "] restatement of " << shapeOp->getName()
                 << " at " << shapeOp->getLoc() << ": ";
    if (r.map)
      llvm::dbgs() << "result->source " << r.map;
    else
      llvm::dbgs() << "no map";
    if (!r.why.empty())
      llvm::dbgs() << " [not projectable: " << r.why << "]";
    llvm::dbgs() << "\n";
  });
  return r;
}

/// What the pass did, for the one-line summary. Not a pass Statistic: these are
/// for reading a trace, and a Statistic would put them in a different place than
/// the lines they summarise.
struct Tally {
  /// Operands the absorber repointed.
  unsigned absorbed = 0;
  /// Operands whose COMPOSED map failed the projection check.
  unsigned declined = 0;
  /// Ops the gate refused.
  unsigned rejected = 0;
  /// linalg.generic ops that went away -- fusion's work, counted as a
  /// difference rather than per decision, because the control function is
  /// consulted more often than it is acted on.
  int genericsRemoved = 0;
};

//===----------------------------------------------------------------------===//
// Piece 2: absorbing one into its consumer's operand map
//===----------------------------------------------------------------------===//

/// Rewrites a `linalg.generic` input operand defined by a coordinate restatement
/// to read that restatement's SOURCE, under the composed map.
///
/// The composition is `resultToSource.compose(consumerOperandMap)`: the consumer
/// map takes loop dims to the shape op's result coordinates, the restatement
/// takes those to the source's, so the composite takes loop dims to the source's
/// -- which is exactly an operand map on the repointed operand.
///
/// Accepted only when that COMPOSED map is a projected coordinate map. The check
/// is on the composite and not on the restatement, for the reason in the header:
/// what has to be projectable is the map the consumer is left holding.
///
/// `outs` is not touched: repointing it would change the op's result type, which
/// is a different rewrite with a different consumer to satisfy. The element type
/// is unchanged, so the body and its block arguments are untouched too, and this
/// is a genuine in-place edit.
struct AbsorbCoordinateOp : public OpRewritePattern<linalg::GenericOp> {
  AbsorbCoordinateOp(MLIRContext *ctx, Tally &tally)
      : OpRewritePattern(ctx), tally(tally) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    for (OpOperand *operand : op.getDpsInputOperands()) {
      Operation *def = operand->get().getDefiningOp();
      if (!def)
        continue;
      CoordinateRestatement restatement = resultToSourceMap(def);
      if (!restatement.map)
        continue;

      AffineMap consumerMap = op.getMatchingIndexingMap(operand);
      // Holds by linalg's own contract -- an operand's map has one result per
      // tensor dim, and the restatement's domain IS that tensor's coordinates.
      if (restatement.map.getNumDims() != consumerMap.getNumResults())
        continue;
      AffineMap composed = restatement.map.compose(consumerMap);
      bool projectable = isProjectedCoordinateMap(composed);

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] operand #" << operand->getOperandNumber()
                 << " of generic at " << op.getLoc() << ": consumer "
                 << consumerMap << " o result->source " << restatement.map
                 << " = " << composed << " -> "
                 << (projectable ? "ABSORB" : "decline, not a projected "
                                              "coordinate map")
                 << "\n");

      if (!projectable) {
        ++tally.declined;
        continue;
      }

      SmallVector<AffineMap> maps = op.getIndexingMapsArray();
      maps[operand->getOperandNumber()] = composed;
      rewriter.modifyOpInPlace(op, [&]() {
        operand->set(restatement.source);
        op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      });
      ++tally.absorbed;
      return success();
    }
    return failure();
  }

private:
  Tally &tally;
};

//===----------------------------------------------------------------------===//
// Piece 3 (Part 2): rejecting what could not be absorbed, on a path the layout
// pass will physicalize
//===----------------------------------------------------------------------===//

/// The diagnostic. `why` comes from `resultToSourceMap`, so the message is
/// derived from the same answer the absorption was.
void rejectUnabsorbable(Operation *op, StringRef why) {
  op->emitError(
      "fold-data-movement-generics: this op re-indexes a value on a path the "
      "layout pass will physicalize and it cannot be restated as an indexing "
      "map on its consumer (")
      << why
      << "), so the layout pass would bridge it with a linearizing map, which "
         "the scheduler cannot project loop IVs through";
}

/// Refuse every coordinate restatement the absorber declined that sits on a path
/// RewriteDescriptorLayoutGeneric will physicalize. Silent when the module has no
/// annotated view at all, which is the gate.
LogicalResult rejectOnPhysicalizedPaths(ModuleOp mod, Tally &tally) {
  // The seed: the layout pass's own roots.
  SmallVector<Operation *> annotatedViews;
  mod.walk([&](mlir::ktdp::ConstructMemoryViewOp view) {
    if (view->hasAttr(triton::tts::TTSDialect::kTensorLayoutAttrName))
      annotatedViews.push_back(view.getOperation());
  });
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] gate: "
                          << annotatedViews.size()
                          << " tts.tensor_layout-annotated memory view(s)\n");
  if (annotatedViews.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] gate closed: nothing in this module is "
                  "physicalized, so nothing is refused\n");
    return success();
  }

  // The scope, from the shared traversals rather than from a copy of them.
  SmallVector<Operation *> adjacent;
  triton::ktdp::collectAdjacentGenerics(annotatedViews, adjacent);
  SmallPtrSet<Operation *, 8> inRewriteSet(adjacent.begin(), adjacent.end());

  SmallVector<Operation *> physLoads, physStores;
  for (Operation *view : annotatedViews)
    triton::ktdp::collectViewAccesses(view, physLoads, physStores);

  LLVM_DEBUG({
    llvm::dbgs() << "[" DEBUG_TYPE "] gate open: " << physLoads.size()
                 << " physicalized load(s), " << physStores.size()
                 << " physicalized store(s), " << adjacent.size()
                 << " generic(s) in the rewrite set\n";
    for (Operation *g : adjacent)
      llvm::dbgs() << "  rewrite set: generic at " << g->getLoc() << "\n";
  });

  LogicalResult result = success();
  SmallPtrSet<Operation *, 16> asked;
  // How each op was reached, so a rejection can print the path rather than
  // leaving the reader to re-derive why the op was on a physicalized path at
  // all. Maps an op to the op whose operand led the walk to it.
  llvm::DenseMap<Operation *, Operation *> cameFrom;

  // `why` empty means the restatement IS a projected coordinate map. Such an op
  // is still here only because it was never offered to the absorber -- nothing
  // between it and the memory access is a linalg.generic. RewriteDescriptorLayout
  // Generic names that case itself, and more precisely than this could: its
  // checkConsumersAreRewritable for the ins side, its checkStoreDataIsRestatable
  // for the store side. So say nothing.
  auto ask = [&](Operation *op, const CoordinateRestatement &restatement) {
    if (restatement.why.empty() || !asked.insert(op).second)
      return;
    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "] REJECT " << op->getName() << " at "
                   << op->getLoc() << "; reached by";
      // Capped: the chain follows reverse def-use edges and so cannot cycle,
      // but a debug print is not the place to bet the process on that.
      unsigned hops = 0;
      for (Operation *step = cameFrom.lookup(op); step && hops < 16;
           step = cameFrom.lookup(step), ++hops)
        llvm::dbgs() << " <- " << step->getName() << " at " << step->getLoc();
      llvm::dbgs() << "\n";
    });
    rejectUnabsorbable(op, restatement.why);
    ++tally.rejected;
    result = failure();
  };

  // BACKWARD, from every value a generic the layout pass will rewrite reads, and
  // from the data of every store over an annotated view.
  //
  // The two seeds are the two sides of one hazard, and neither is optional:
  //   the generic's `ins` -- findLayoutForInput takes a ktdp.load and nothing
  //   else, so an operand behind a shape op has no layout and rebuildMap bridges
  //   it with linearizeStickLane;
  //   the store's data -- findLayoutForResult walks the generic's result users
  //   for a ktdp.store DIRECTLY (`dyn_cast<StoreOp>`, `continue` otherwise), so a
  //   shape op between the two makes it return null and the LINEARIZATION lands
  //   on the `outs` map instead. `gather__1d` carries an 8x1 -> 8 tt.reshape into
  //   a store, which is that shape in the tree today -- unannotated, hence the
  //   gate.
  //
  // Nothing here absorbs on the store side: that rewrite would have to change
  // the generic's result type, replace the tensor.empty behind `outs`, restate
  // the outs map and repoint the store, and it needs the map in the opposite
  // direction, which is an inversePermutation that is null for exactly the
  // linearizing cases. Rejection only.
  SmallPtrSet<Operation *, 16> visited;
  SmallVector<std::pair<Value, Operation *>> worklist;
  auto seed = [&](Value v, Operation *user) { worklist.emplace_back(v, user); };
  for (Operation *g : adjacent)
    for (Value in : cast<linalg::GenericOp>(g).getDpsInputs())
      seed(in, g);
  for (Operation *st : physStores)
    seed(cast<mlir::ktdp::StoreOp>(st).getDataTile(), st);

  while (!worklist.empty()) {
    auto [v, user] = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || !visited.insert(def).second)
      continue;
    cameFrom.try_emplace(def, user);
    if (isa<linalg::GenericOp>(def)) {
      // A generic already in the rewrite set is its own entry point above; one
      // that is not is walked through, because the absorber is not gated on
      // physicalization and could have folded something above it into it.
      if (!inRewriteSet.contains(def))
        for (Value in : cast<linalg::GenericOp>(def).getDpsInputs())
          seed(in, def);
      continue;
    }
    CoordinateRestatement restatement = resultToSourceMap(def);
    if (!restatement.isReindex())
      continue; // a load, an empty, a constant, a fill, an scf result, a cast
    ask(def, restatement);
    seed(restatement.source, def);
  }

  // FORWARD, one hop off each load the layout pass will retype -- for a
  // `load -> reshape -> store` chain with no generic in it anywhere, which the
  // backward walk from the store's data also reaches, and a `load -> reshape`
  // whose result goes somewhere else entirely, which it does not.
  for (Operation *load : physLoads)
    for (Operation *user : load->getResult(0).getUsers()) {
      if (isa<linalg::GenericOp, mlir::ktdp::StoreOp>(user))
        continue;
      cameFrom.try_emplace(user, load);
      ask(user, resultToSourceMap(user));
    }

  return result;
}

struct FoldDataMovementGenericsPass
    : public mlir::triton::spyre::impl::FoldDataMovementGenericsBase<
          FoldDataMovementGenericsPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    Tally tally;
    auto countGenerics = [&] {
      int n = 0;
      mod.walk([&](linalg::GenericOp) { ++n; });
      return n;
    };
    int genericsBefore = countGenerics();

    RewritePatternSet patterns(ctx);
    // Upstream does the map composition; the control function decides which
    // fusions are allowed to happen at all.
    linalg::ControlFusionFn onlyDataMovementProducers =
        [](OpOperand *fusedOperand) {
          return isPureDataMovement(fusedOperand->get().getDefiningOp());
        };
    linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                onlyDataMovementProducers);
    patterns.add<AbsorbCoordinateOp>(ctx, tally);

    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();
    tally.genericsRemoved = genericsBefore - countGenerics();

    // After the fixpoint, not during it: what is left is precisely what the
    // absorber declined, so presence is the evidence and no pattern has to
    // report anything.
    LogicalResult gate = rejectOnPhysicalizedPaths(mod, tally);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] done: " << tally.absorbed
               << " operand(s) absorbed, " << tally.genericsRemoved
               << " generic(s) removed by fusion, " << tally.declined
               << " operand(s) declined, " << tally.rejected
               << " op(s) rejected\n");

    if (failed(gate))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>> createFoldDataMovementGenericsPass() {
  return std::make_unique<FoldDataMovementGenericsPass>();
}

} // namespace mlir::triton::spyre
