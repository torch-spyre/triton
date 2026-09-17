//===- RewriteDescriptorLayoutGeneric.cpp ---------------------------------===//
//
// Rewrites logical tensor descriptors to their physical (stick-tiled) layout,
// driven by tt.spyre_tensor_layout markers, and retypes the compute ops on the
// annotated chain — which are all linalg.generic.
//
// The marker carries the physical layout as the OpSpec `device_coordinates`
// form, three i64 arrays with one entry per physical dim:
//   phys_src[p] : the logical dim physical dim p derives from
//   phys_op[p]  : 0 = identity, 1 = floordiv, 2 = mod, 3 = broadcast
//   phys_arg[p] : divisor (floordiv) / modulus (mod) / lane count (broadcast);
//                 ignored for identity
// e.g. [M,N] stick-on-N -> phys_src=[1,0,1] phys_op=[1,0,2] phys_arg=[64,0,64]
//   => physical size [ceil(N/64), M, 64].
// A broadcast dim replicates its source logical dim across `phys_arg` lanes
// instead of partitioning it, so [M] -> phys_src=[0,0] phys_op=[0,3]
// phys_arg=[0,64] gives physical size [M, 64].
//
//   Phase 1  physicalize each annotated descriptor: memory view, access tiles,
//            loads. Stores have their access tile redirected.
//   Phase 2  one rewrite over generics and stores, applied greedily.
//   Phase 3  erase the markers and their now-dead bridge casts.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "RewriteDescriptorLayout/IndexDomain.h"
#include "RewriteDescriptorLayout/PermutationUtils.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "rewrite-descriptor-layout-generic"

namespace mlir::triton::ktdp {

#define GEN_PASS_DEF_REWRITEDESCRIPTORLAYOUTGENERIC
#include "Dialect/KTDP/Transforms/Passes.h.inc"

} // namespace mlir::triton::ktdp

namespace {

using namespace mlir;
using namespace mlir::triton::ktdp;

//===----------------------------------------------------------------------===//
// The coordinate map, read off a marker
//===----------------------------------------------------------------------===//

/// One descriptor's physical layout: the marker's three arrays, plus the
/// logical rank they index into.
///
/// The marker is an *instruction* — it says how to split logical dims. It is
/// not a source of truth about the tensor: element type, memory space, base
/// offset, strides, coordinate set and dynamic extents all come from the ops
/// being rewritten, and where the two could disagree the op wins.
struct CoordMap {
  ArrayRef<int64_t> src, op, arg;
  unsigned logicalRank = 0;

  unsigned physRank() const { return src.size(); }
  CoordOp opAt(unsigned p) const { return static_cast<CoordOp>(op[p]); }

  /// Is logical dim `d` split into a (stick, elem) pair by this layout?
  ///
  /// A broadcast names `d` too, but it partitions nothing — it replicates `d`
  /// across a fresh axis — so it is not a split and the dim stays whole. Asking
  /// for the floordiv half is therefore the question, not "is some dim of `d`
  /// non-identity".
  bool splits(int64_t d) const {
    return findPhys(d, CoordOp::FloorDiv) >= 0;
  }

  /// The physical dim carrying `wanted` for logical dim `d`, or -1.
  int findPhys(int64_t d, CoordOp wanted) const {
    for (unsigned p = 0, e = physRank(); p < e; ++p)
      if (src[p] == d && opAt(p) == wanted)
        return p;
    return -1;
  }

  /// The stick width of the split of logical dim `d`. Only meaningful when
  /// `splits(d)`; read off the mod dim, which is where the width is the extent.
  int64_t stickWidth(int64_t d) const {
    int p = findPhys(d, CoordOp::Mod);
    return p < 0 ? 0 : arg[p];
  }
};

/// Name of a coord op, for the trace.
const char *coordOpName(CoordOp op) {
  switch (op) {
  case CoordOp::Identity:
    return "id";
  case CoordOp::FloorDiv:
    return "stick";
  case CoordOp::Mod:
    return "lane";
  case CoordOp::Broadcast:
    return "bcast";
  }
  return "?";
}

/// Print a coord map as one physical dim per entry, each naming the logical dim
/// it came from, the coord op that made it, and the op's argument where the
/// argument means something. This is the whole layout in one line, which is what
/// lets a reader check the marker against the shape derived from it below.
///
/// Reached only from an LLVM_DEBUG body, so a release build has no caller left;
/// [[maybe_unused]] keeps that from warning.
[[maybe_unused]] void printCoordMap(llvm::raw_ostream &os, const CoordMap &cm) {
  os << "logical rank " << cm.logicalRank << " -> phys [";
  for (unsigned p = 0, e = cm.physRank(); p < e; ++p) {
    if (p)
      os << ", ";
    os << "d" << cm.src[p] << ":" << coordOpName(cm.opAt(p));
    if (cm.opAt(p) != CoordOp::Identity)
      os << "(" << cm.arg[p] << ")";
  }
  os << "]";
}

/// Read the coord map off a marker, checking phys_src against `logicalRank`.
FailureOr<CoordMap> readCoordMap(triton::SpyreTensorLayoutOp marker,
                                 unsigned logicalRank) {
  CoordMap cm{marker.getPhysSrc(), marker.getPhysOp(), marker.getPhysArg(),
              logicalRank};
  if (cm.op.size() != cm.physRank() || cm.arg.size() != cm.physRank())
    return marker.emitError("spyre_tensor_layout: phys_src, phys_op and "
                            "phys_arg must have the same length");
  for (unsigned p = 0, e = cm.physRank(); p < e; ++p) {
    if (cm.src[p] < 0 || cm.src[p] >= (int64_t)logicalRank)
      return marker.emitError("spyre_tensor_layout: phys_src out of range for "
                              "logical rank ")
             << logicalRank;
    if (cm.op[p] < 0 || cm.op[p] > 3)
      return marker.emitError("spyre_tensor_layout: phys_op must be 0 "
                              "(identity), 1 (floordiv), 2 (mod) or 3 "
                              "(broadcast)");
  }
  // A split names the same logical dim twice, once floordiv and once mod. A
  // lone half would leave the rebuild unable to state where the dim's elements
  // live, so reject it here rather than emitting a map that cannot address
  // them.
  for (unsigned d = 0; d < logicalRank; ++d) {
    bool hasFloor = cm.findPhys(d, CoordOp::FloorDiv) >= 0;
    bool hasMod = cm.findPhys(d, CoordOp::Mod) >= 0;
    if (hasFloor != hasMod)
      return marker.emitError("spyre_tensor_layout: logical dim ")
             << d << " has a " << (hasFloor ? "floordiv" : "mod")
             << " physical dim without the matching "
             << (hasFloor ? "mod" : "floordiv") << " half";
    // A broadcast replicates the dim it names, so that dim must also be present
    // whole for the replication to have something to replicate. It cannot be
    // present as a split: the elements would then live in the floordiv/mod pair
    // and the broadcast axis would name a third copy of them.
    if (cm.findPhys(d, CoordOp::Broadcast) >= 0 &&
        cm.findPhys(d, CoordOp::Identity) < 0)
      return marker.emitError("spyre_tensor_layout: logical dim ")
             << d
             << " is broadcast but has no identity physical dim; a broadcast "
                "replicates a dim that is also carried whole";
  }
  return cm;
}

/// The composite that recovers logical dim `d`'s index from the two physical
/// dims a stick split gave it: the stick index counts whole sticks of `width`
/// elements, and the element offset picks one out of the stick it lands in.
///
/// This is the *only* arithmetic either carrier introduces, and both introduce
/// the same one. Phase 2 names its two halves after loop dims of the rebuilt
/// linalg domain; the indirect access tile names them after intermediate
/// variables of its own variable space. Different numbering, identical algebra —
/// so the algebra lives here once and each carrier passes in the exprs it
/// numbers the halves with.
inline AffineExpr composeStickSplit(AffineExpr stick, int64_t width,
                                    AffineExpr elem) {
  return stick * width + elem;
}

/// The physical tensor type `cm` prescribes for a logical shape.
/// Fails when a physical extent cannot be stated statically.
FailureOr<RankedTensorType> physicalTensorType(const CoordMap &cm,
                                               RankedTensorType logicalType) {
  SmallVector<int64_t> physShape;
  if (!applyCoordMap(logicalType.getShape(), cm.src, cm.op, cm.arg, physShape))
    return failure();
  return RankedTensorType::get(physShape, logicalType.getElementType());
}

//===----------------------------------------------------------------------===//
// Phase 2's map rebuild
//===----------------------------------------------------------------------===//

/// One value's place in the rebuild: its logical indexing map, and the layout
/// it is physicalized under. `layout` is null for a value that carries no
/// marker and therefore stays logical.
struct RebuildOperand {
  AffineMap logicalMap;
  const CoordMap *layout = nullptr;
  /// Loop dim assigned to each of this operand's BROADCAST physical dims, keyed
  /// by physical dim; -1 for every dim that is not a broadcast. Filled by
  /// buildLoopDomain, because a broadcast axis is the one physical dim that no
  /// logical loop dim accounts for — see LoopDomain.
  SmallVector<int> broadcastDim;
};

/// One piece of the refined loop domain: a logical dim's stick index, its
/// element offset within a stick, or — when nothing splits the dim — the whole
/// dim, which is spelled as the stick half.
///
/// A piece is the unit the numbering orders, because it is the unit an operand's
/// physical dim names: a physical dim carries exactly one of these, and that is
/// what lets an operand's physical order be read as an order on pieces.
struct DomainPiece {
  unsigned loop;
  /// True for the element offset within a stick, false for the stick index (or
  /// for an unsplit dim held whole).
  bool elem;

  bool operator==(const DomainPiece &o) const {
    return loop == o.loop && elem == o.elem;
  }
};

/// The rebuilt loop domain: how many physical loop dims there are, and where
/// each logical dim's pieces landed.
///
/// A logical dim split by any operand contributes TWO loop dims — a stick
/// index and an element offset within the stick — and an operand that holds
/// that dim whole addresses it as `stick * width + elem`. A dim no operand
/// splits contributes one.
///
/// Which loop number each piece gets is read off the operands' physical orders,
/// with the RESULT's taken as authoritative — see buildLoopDomain.
///
/// On top of that refinement, each BROADCAST physical dim contributes one loop
/// dim of its own. A broadcast is not a subdivision of a logical dim, so no
/// logical loop dim can stand for it: it replicates that dim across a fresh
/// axis, and a fresh axis is a fresh loop. Those loops are numbered after the
/// refinement, and recorded per operand in RebuildOperand::broadcastDim rather
/// than here, since two operands broadcasting the same logical dim replicate it
/// independently and so get separate loops.
struct LoopDomain {
  /// Loop dim carrying logical dim d's stick index, or its whole extent when
  /// the dim is unsplit.
  SmallVector<int> stickDim;
  /// Loop dim carrying logical dim d's element offset within a stick; -1 when
  /// the dim is unsplit.
  SmallVector<int> elemDim;
  /// The stick width to compose with, per logical dim; 0 when unsplit. Taken
  /// from the operands, which must agree — see buildLoopDomain.
  SmallVector<int64_t> width;
  unsigned numLoopDims = 0;

  bool isSplit(int64_t d) const { return elemDim[d] >= 0; }
};

/// The domain pieces one operand's physical dims name, in that operand's own
/// physical order.
///
/// This is the operand's physical order re-expressed as an order on pieces, and
/// it is the only thing the numbering below reads off an operand. A physical dim
/// names the piece its coord op picks out: a mod dim names the element half, a
/// floordiv dim the stick half, and an identity dim the stick half too — an
/// identity dim over a dim the domain splits addresses it as
/// `stick * width + elem`, whose leading term is the stick half, so the stick
/// half is where that physical position sits.
///
/// A broadcast dim names none: it is a replication axis, not a piece of any
/// logical dim, and it gets its own loop after the refinement.
///
/// An operand with no layout walks its own logical dims, every one held whole —
/// which is the same walk under an identity layout. Its logical order IS its
/// physical order, since nothing physicalized it.
void collectPieces(const RebuildOperand &o,
                   SmallVectorImpl<DomainPiece> &pieces) {
  unsigned numDims =
      o.layout ? o.layout->physRank() : o.logicalMap.getNumResults();
  for (unsigned p = 0; p < numDims; ++p) {
    CoordOp coordOp = o.layout ? o.layout->opAt(p) : CoordOp::Identity;
    if (coordOp == CoordOp::Broadcast)
      continue;
    int64_t logDim = o.layout ? o.layout->src[p] : p;
    auto dimExpr = dyn_cast<AffineDimExpr>(o.logicalMap.getResult(logDim));
    if (!dimExpr)
      continue; // a constant (a folded broadcast) names no loop dim
    pieces.push_back({dimExpr.getPosition(), coordOp == CoordOp::Mod});
  }
}

/// Build the loop domain over `logicalNumLoops` dims, splitting every logical
/// dim that any operand splits. A split dim contributes two loop dims, a stick
/// index and an element offset. `resultIdx` names the `outs` operand within
/// `operands`.
///
/// A loop domain is only ever defined up to a relabelling -- permuting the loop
/// dims and permuting every map's reference to them describes the same
/// computation -- so the numbering is free, and something has to fix it. Two
/// consumers care which way, and they care about different operands:
///
///   - ktir-cpu reads the RESULT's map. For an all-parallel generic it takes the
///     iteration shape to be the result's shape outright
///     (`if not reduction_dims: iter_shape = out_shape`), and for a reduction it
///     folds and squeezes the reduction loops and expects what remains, in loop
///     order, to be the result's shape. Both say: the result's physical order
///     must be the order its loops appear in.
///   - The scheduler reads an INPUT's map. Its ReductionLoopExposurePass
///     substitutes a loop index for an operand axis index, which is sound only
///     while a reduction loop sits AT the axis it reduces; moved off it, the
///     wrong physical dim gets narrowed and the report names a tensor type that
///     is in no input module. That says: a reduction loop keeps the positional
///     index of the axis it reduces.
///
///     Narrower than "the input's map is the identity", and the difference
///     matters. Permuting the PARALLEL dims is fine -- a fully reversed version
///     of the same reduce gets a correctly computed slice out of that pass -- and
///     the transpose cases below require the permutation. Note also that the
///     scheduler's KTIR frontend handles the permutation correctly and a later
///     pass discards that, so this is a defect on that side, filed upstream,
///     rather than a rule this emitter was wrong to violate. The rule below is
///     the better emission independently of it.
///
/// So the rule is a merge, seeded by the result and refined by the inputs:
///
///   1. Walk the result's physical dims and number the pieces they name, in that
///      order. The result is the one operand whose coordinate order the generic
///      does not get to choose -- its elements are written where its own type
///      says they live -- so its order is authoritative and nothing below
///      reorders it.
///   2. Walk each remaining operand's physical dims. A piece already numbered
///      only advances a cursor; a piece the result never named is INSERTED at
///      the cursor, i.e. at the place this operand's own physical order puts it
///      relative to the pieces the result did name.
///   3. Anything still unnumbered is appended in logical order.
///
/// Step 2 is what a reduce needs, and it is where the reduced dim's own axis
/// index comes from. The result of a reduce does not name the reduced dim at all,
/// so step 1 leaves it unplaced; appending it -- which is what this used to do --
/// puts it after pieces that come BEFORE it in the input's physical order, so its
/// loop number no longer matches the axis it occupies there. Stick-on-N is the
/// case: logical [M, N] -> physical [N/S, M, S], reduce over M, so the result's
/// physical order (stick, lane) is a subsequence of the input's (stick, M, lane).
/// Appending M gives `ins (d0, d2, d1)` -- the reduction is loop d2 sitting at
/// axis 1. Inserting it where the input puts it gives `ins (d0, d1, d2)`, the
/// reduction at loop d1 and axis 1, with `outs (d0, d2)` and iterators
/// [parallel, reduction, parallel]. That is exactly what
/// `linalg.reduce ... dimensions = [1]` desugars to.
///
/// The identity is how that particular case comes out, not the goal. A reduce
/// whose operands' physical orders genuinely disagree still gets a permuted input
/// map -- and correctly so; what the insertion guarantees is that the reduced
/// dim's loop number is the axis index it has in the input, which is the part
/// that is not free.
///
/// The two conventions do not collide, because inserting never moves a piece the
/// result named: the result's map stays monotone in the loop numbering, which is
/// all ktir-cpu asks of it. What changes is only that the result's map need no
/// longer project onto a *prefix* of the domain.
///
/// Numbering in *logical* order, which this did before the result became the
/// seed, is a different mistake and still one. A layout is free to reorder dims,
/// so a logical-order domain makes the result read `(d1, d0, d2)` on a
/// stick-on-N elementwise chain: faithful to the layout, stated in the logical
/// frame, and not the frame the result is written in. Nothing cancels it later,
/// because there is no later -- these maps are the output.
///
/// When the result's physical order already reaches every piece -- every
/// elementwise, broadcast and transpose case -- step 2 inserts nothing and the
/// numbering is exactly the result's own walk. So this generalizes the
/// result-as-frame rule rather than weakening it: it only decides where the
/// pieces that rule left unplaced go.
///
/// Fails when two operands split the same logical dim at different widths:
/// there is then no single `stick * width + elem` a third operand holding the
/// dim whole could use, and picking either width would silently address the
/// wrong elements.
FailureOr<LoopDomain>
buildLoopDomain(MutableArrayRef<RebuildOperand> operands, unsigned resultIdx,
                unsigned logicalNumLoops,
                llvm::function_ref<InFlightDiagnostic()> emitError) {
  LoopDomain dom;
  dom.stickDim.assign(logicalNumLoops, -1);
  dom.elemDim.assign(logicalNumLoops, -1);
  dom.width.assign(logicalNumLoops, 0);

  // Which logical loop dims are split, and at what width.
  for (const RebuildOperand &o : operands) {
    if (!o.layout)
      continue;
    for (unsigned r = 0, e = o.logicalMap.getNumResults(); r < e; ++r) {
      auto dimExpr = dyn_cast<AffineDimExpr>(o.logicalMap.getResult(r));
      if (!dimExpr)
        continue; // a constant (a folded broadcast) names no loop dim
      unsigned loop = dimExpr.getPosition();
      if (!o.layout->splits(r))
        continue;
      int64_t w = o.layout->stickWidth(r);
      if (dom.width[loop] && dom.width[loop] != w)
        return emitError() << "loop dim " << loop
                           << " is split at two different stick widths ("
                           << dom.width[loop] << " and " << w
                           << "), so no single composite addresses it";
      dom.width[loop] = w;
    }
  }

  // Step 1: the result's physical order seeds the numbering. The result is the
  // one operand whose coordinate order the generic does not get to choose, since
  // its elements are written where its own type says they live, so nothing below
  // reorders what this places.
  SmallVector<DomainPiece> order;
  auto seen = [&](DomainPiece pc) { return llvm::is_contained(order, pc); };
  {
    SmallVector<DomainPiece> walk;
    collectPieces(operands[resultIdx], walk);
    for (DomainPiece pc : walk)
      if (!seen(pc))
        order.push_back(pc);
  }

  // Step 2: merge each remaining operand's physical order in. A piece the result
  // already placed only advances the cursor -- and only forwards, so an operand
  // that disagrees with the established order cannot drag it back. A piece the
  // result never named is inserted AT the cursor, which is the position this
  // operand's own physical order puts it in relative to the result's pieces.
  //
  // This is what makes a reduce's input map the identity: the reduced dim is a
  // piece the result never names, and the input's walk says where it sits.
  for (auto [i, o] : llvm::enumerate(operands)) {
    if (i == resultIdx)
      continue;
    SmallVector<DomainPiece> walk;
    collectPieces(o, walk);
    unsigned cursor = 0;
    for (DomainPiece pc : walk) {
      auto it = llvm::find(order, pc);
      if (it != order.end()) {
        unsigned idx = std::distance(order.begin(), it);
        if (idx >= cursor)
          cursor = idx + 1;
        continue;
      }
      order.insert(order.begin() + cursor, pc);
      ++cursor;
    }
  }

  // Step 3: anything no operand's walk named -- a dim every operand addresses
  // through a constant, or the element half of a dim only the domain knows is
  // split -- goes last, in logical order. Every logical dim needs a stick slot,
  // and a split one needs an element slot, since the rebuild indexes both.
  for (unsigned d = 0; d < logicalNumLoops; ++d) {
    if (!seen({d, /*elem=*/false}))
      order.push_back({d, /*elem=*/false});
    if (dom.width[d] && !seen({d, /*elem=*/true}))
      order.push_back({d, /*elem=*/true});
  }

  for (auto [n, pc] : llvm::enumerate(order))
    (pc.elem ? dom.elemDim : dom.stickDim)[pc.loop] = n;
  dom.numLoopDims = order.size();

  // Then one loop per broadcast physical dim, after the refinement so that an
  // operand carrying no broadcast keeps exactly the numbering it would have had.
  for (RebuildOperand &o : operands) {
    if (!o.layout)
      continue;
    o.broadcastDim.assign(o.layout->physRank(), -1);
    for (unsigned p = 0, e = o.layout->physRank(); p < e; ++p)
      if (o.layout->opAt(p) == CoordOp::Broadcast)
        o.broadcastDim[p] = dom.numLoopDims++;
  }
  return dom;
}

/// Rebuild one operand's indexing map over `dom`.
///
/// Per map result, three cases, and they are the whole rule:
///   - the operand splits this dim too  -> name the two loop dims directly, in
///     the operand's own physical order (its marker says which is which);
///   - the operand holds the dim whole  -> address it as `stick * width + elem`;
///   - the result is not a dim at all   -> pass it through unchanged, which is
///     what keeps a folded broadcast's constant a constant.
/// A dim the domain does not split is named by its single loop dim in all three.
///
/// A broadcast physical dim is the fourth: it names the loop dim the domain gave
/// it, and nothing else. No arithmetic, because it addresses no element of the
/// logical dim — it is the replication axis itself.
AffineMap rebuildMap(const RebuildOperand &o, const LoopDomain &dom,
                     MLIRContext *ctx) {
  auto loopExpr = [&](int loopDim) { return getAffineDimExpr(loopDim, ctx); };

  // A physicalized operand's map is stated over its PHYSICAL dims, so it is
  // built by walking those, each naming the logical dim it came from and the
  // coord op that made it. An operand with no layout walks its own logical dims,
  // every one of them held whole — which is the same walk with an identity
  // layout, so the substitution below is shared rather than restated.
  unsigned numResults =
      o.layout ? o.layout->physRank() : o.logicalMap.getNumResults();

  SmallVector<AffineExpr> results;
  for (unsigned p = 0; p < numResults; ++p) {
    int64_t logDim = o.layout ? o.layout->src[p] : p;
    CoordOp coordOp = o.layout ? o.layout->opAt(p) : CoordOp::Identity;

    AffineExpr logResult = o.logicalMap.getResult(logDim);
    auto dimExpr = dyn_cast<AffineDimExpr>(logResult);
    if (!dimExpr) {
      // A constant survives every physical dim it is named by: the operand does
      // not vary along this loop dim, whatever the layout does to it.
      results.push_back(logResult);
      continue;
    }
    unsigned loop = dimExpr.getPosition();

    switch (coordOp) {
    case CoordOp::FloorDiv:
      results.push_back(loopExpr(dom.stickDim[loop]));
      break;
    case CoordOp::Mod:
      results.push_back(loopExpr(dom.elemDim[loop]));
      break;
    case CoordOp::Broadcast:
      // The replication axis, named by the loop the domain allocated for it. It
      // does not enter the loop dim `logDim` maps to at all, which is what makes
      // this loop appear in one map only.
      results.push_back(loopExpr(o.broadcastDim[p]));
      break;
    case CoordOp::Identity:
      // This operand holds the dim whole. If the domain split it — because some
      // other operand does — the two halves must be recombined here, and that
      // composite is the rule's only arithmetic.
      results.push_back(
          dom.isSplit(loop)
              ? composeStickSplit(loopExpr(dom.stickDim[loop]),
                                  dom.width[loop], loopExpr(dom.elemDim[loop]))
              : loopExpr(dom.stickDim[loop]));
      break;
    }
  }
  return AffineMap::get(dom.numLoopDims, /*symbolCount=*/0, results, ctx);
}

/// Rebuild the iterator types over `dom`. Both halves of a split dim inherit
/// the kind of the dim they came from: splitting a reduction gives two
/// reduction loops, splitting a parallel dim two parallel ones.
SmallVector<utils::IteratorType>
rebuildIterators(ArrayRef<utils::IteratorType> logicalIterators,
                 const LoopDomain &dom) {
  // Parallel is the default so that a broadcast loop gets it without being
  // singled out: replication copies a value to each lane, and copying never
  // accumulates, so a replication axis is parallel by what it is. Every logical
  // dim overwrites its own entries below, so the default reaches only the loops
  // the refinement did not cover — which are exactly the broadcast ones.
  SmallVector<utils::IteratorType> out(dom.numLoopDims,
                                      utils::IteratorType::parallel);
  for (unsigned d = 0, e = logicalIterators.size(); d < e; ++d) {
    out[dom.stickDim[d]] = logicalIterators[d];
    if (dom.isSplit(d))
      out[dom.elemDim[d]] = logicalIterators[d];
  }
  return out;
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct RewriteDescriptorLayoutGenericPass
    : public mlir::triton::ktdp::impl::RewriteDescriptorLayoutGenericBase<
          RewriteDescriptorLayoutGenericPass> {

  using RewriteDescriptorLayoutGenericBase::
      RewriteDescriptorLayoutGenericBase;

  /// True = "device" (physical row-major strides), false = "host" (derive the
  /// physical strides from the logical ones through the coord map).
  bool hwDataLayout = false;

  /// The layout every physicalized value carries. Phase 1 seeds it from the
  /// loads it retypes; Phase 2 reads it to decide what each generic operand's
  /// physical type must be, and extends it to the results it retypes.
  ///
  /// Absence is meaningful: a value with no entry is not on a physicalized
  /// chain, so it stays logical and its map keeps its unsplit dims. That is
  /// what leaves an unannotated kernel untouched.
  DenseMap<Value, CoordMap> layoutOf;

  /// Logical construct_memory_view ops superseded in Phase 1. They cannot be
  /// erased there: the marker's bridge cast still holds them, and that cast
  /// only dies in Phase 3.
  SmallVector<mlir::ktdp::ConstructMemoryViewOp> deadLogicalMemViews;

  //===--------------------------------------------------------------------===//
  // Phase 1 — physicalize one descriptor
  //===--------------------------------------------------------------------===//

  /// Physical sizes for a view, as static extents plus the dynamic values the
  /// kDynamic entries draw from, in order.
  struct PhysicalSizes {
    SmallVector<int64_t> staticSizes;
    SmallVector<Value> dynSizes;
  };

  /// Compute the physical sizes of `memViewOp` under `cm`, emitting a ceildiv
  /// for a floordiv dim whose logical extent is only known at runtime.
  FailureOr<PhysicalSizes>
  physicalSizes(const CoordMap &cm, mlir::ktdp::ConstructMemoryViewOp memViewOp,
                OpBuilder &b, triton::SpyreTensorLayoutOp marker) {
    ArrayRef<int64_t> logStatic = memViewOp.getStaticSizes();
    ValueRange logDyn = memViewOp.getSizes();
    Location loc = memViewOp.getLoc();

    // Position of each logical dim's runtime extent within `logDyn`; -1 when
    // the extent is static. This is the sentinel convention
    // construct_memory_view's own builder and verifier use.
    SmallVector<int> dynPos(logStatic.size(), -1);
    for (unsigned d = 0, n = 0; d < logStatic.size(); ++d)
      if (logStatic[d] == ShapedType::kDynamic)
        dynPos[d] = n++;

    PhysicalSizes out;
    for (unsigned p = 0, e = cm.physRank(); p < e; ++p) {
      int64_t d = cm.src[p];
      CoordOp op = cm.opAt(p);
      if (auto stat = applyStatic(logStatic[d], op, cm.arg[p])) {
        out.staticSizes.push_back(*stat);
        continue;
      }
      if (dynPos[d] < 0)
        return marker.emitError("spyre_tensor_layout: physical dim ")
               << p << " has no static extent and logical dim " << d
               << " supplies no dynamic one";
      out.staticSizes.push_back(ShapedType::kDynamic);
      Value logExtent = logDyn[dynPos[d]];
      // ceildiv, not floordiv: a partial boundary stick still needs a stick.
      out.dynSizes.push_back(
          op == CoordOp::FloorDiv
              ? arith::CeilDivSIOp::create(
                    b, loc, logExtent,
                    arith::ConstantOp::create(b, loc,
                                              b.getIndexAttr(cm.arg[p])))
                    .getResult()
              : logExtent);
    }
    return out;
  }

  /// Physical strides for a view under `cm`.
  ///
  /// In "device" mode the physical layout IS the memory layout, so strides are
  /// row-major over the physical sizes. In "host" mode the memory is still
  /// row-major over the LOGICAL shape, so each physical dim inherits its source
  /// logical dim's stride, scaled by the stick width where the dim counts
  /// sticks.
  FailureOr<PhysicalSizes>
  physicalStrides(const CoordMap &cm,
                  mlir::ktdp::ConstructMemoryViewOp memViewOp,
                  const PhysicalSizes &sizes, OpBuilder &b,
                  triton::SpyreTensorLayoutOp marker) {
    Location loc = memViewOp.getLoc();
    unsigned physRank = cm.physRank();
    PhysicalSizes out;

    auto asValue = [&](int64_t c) {
      return arith::ConstantOp::create(b, loc, b.getIndexAttr(c)).getResult();
    };

    if (hwDataLayout) {
      bool allStatic = llvm::none_of(sizes.staticSizes, ShapedType::isDynamic);
      if (allStatic) {
        out.staticSizes.assign(physRank, 1);
        for (int p = (int)physRank - 2; p >= 0; --p)
          out.staticSizes[p] =
              out.staticSizes[p + 1] * sizes.staticSizes[p + 1];
        return out;
      }
      // Any dynamic size makes every outer stride dynamic, so all of them are
      // built as SSA values and the whole static array is sentinels.
      SmallVector<Value> strides(physRank);
      strides[physRank - 1] = asValue(1);
      auto sizeValue = [&](unsigned p) -> Value {
        if (!ShapedType::isDynamic(sizes.staticSizes[p]))
          return asValue(sizes.staticSizes[p]);
        unsigned n = 0;
        for (unsigned q = 0; q < p; ++q)
          if (ShapedType::isDynamic(sizes.staticSizes[q]))
            ++n;
        return sizes.dynSizes[n];
      };
      for (int p = (int)physRank - 2; p >= 0; --p)
        strides[p] = arith::MulIOp::create(b, loc, strides[p + 1],
                                          sizeValue(p + 1))
                         .getResult();
      out.staticSizes.assign(physRank, ShapedType::kDynamic);
      out.dynSizes.assign(strides.begin(), strides.end());
      return out;
    }

    ArrayRef<int64_t> logStatic = memViewOp.getStaticStrides();
    ValueRange logDyn = memViewOp.getStrides();
    SmallVector<int> dynPos(logStatic.size(), -1);
    for (unsigned d = 0, n = 0; d < logStatic.size(); ++d)
      if (logStatic[d] == ShapedType::kDynamic)
        dynPos[d] = n++;

    for (unsigned p = 0; p < physRank; ++p) {
      int64_t d = cm.src[p];
      // Host memory is row-major over the LOGICAL shape, so it has no axis a
      // replication could stride along — a broadcast dim inherits no logical
      // stride and 0 would claim the buffer holds the replicated copies. Say so
      // rather than emit a stride that misdescribes the memory.
      if (cm.opAt(p) == CoordOp::Broadcast)
        return marker.emitError("spyre_tensor_layout: physical dim ")
               << p
               << " is a broadcast, which has no stride in a host row-major "
                  "buffer; broadcast dims need data-layout=device";
      // A stick index advances by a whole stick of the logical dim.
      int64_t scale = cm.opAt(p) == CoordOp::FloorDiv ? cm.arg[p] : 1;
      if (logStatic[d] != ShapedType::kDynamic) {
        out.staticSizes.push_back(logStatic[d] * scale);
        continue;
      }
      if (dynPos[d] < 0)
        return marker.emitError("spyre_tensor_layout: logical dim ")
               << d << " has neither a static nor a dynamic stride";
      out.staticSizes.push_back(ShapedType::kDynamic);
      Value logStride = logDyn[dynPos[d]];
      out.dynSizes.push_back(
          scale == 1 ? logStride
                     : arith::MulIOp::create(b, loc, logStride, asValue(scale))
                           .getResult());
    }
    return out;
  }

  /// Attributes this pass owns on a physicalized op — the ones whose value is a
  /// function of the shape, and so must be recomputed rather than carried.
  /// Everything NOT named here rides along on the clone, which is what makes a
  /// field added later carried by default.
  static bool isShapeOwnedAttr(StringRef name, Operation *op) {
    if (isa<mlir::ktdp::ConstructMemoryViewOp>(op))
      return name == "static_sizes" || name == "static_strides" ||
             name == "coordinate_set" || name == "operandSegmentSizes";
    if (isa<mlir::ktdp::ConstructAccessTilesOp>(op))
      return name == "base_map" || name == "access_tile_set" ||
             name == "access_tile_order" || name == "operandSegmentSizes";
    if (isa<linalg::GenericOp>(op))
      return name == "indexing_maps" || name == "iterator_types" ||
             name == "operandSegmentSizes";
    return false;
  }

  /// Assert that cloning carried everything this pass does not own.
  ///
  /// The bug this guards against is the opposite polarity: building a fresh op
  /// from an enumerated subset of the original's fields, so that a field nobody
  /// remembered is dropped silently. Here the clone carries everything and this
  /// checks the claim, naming any attribute that went missing.
  LogicalResult verifyNothingDropped(Operation *original, Operation *rewritten) {
    for (NamedAttribute attr : original->getAttrs()) {
      StringRef name = attr.getName().strref();
      if (isShapeOwnedAttr(name, original))
        continue;
      Attribute got = rewritten->getAttr(name);
      if (!got)
        return rewritten->emitError("rewrite-descriptor-layout-generic: "
                                    "physicalizing dropped attribute '")
               << name << "'; it is not one this pass owns";
      if (got != attr.getValue())
        return rewritten->emitError("rewrite-descriptor-layout-generic: "
                                    "physicalizing changed attribute '")
               << name << "'; it is not one this pass owns";
    }
    return success();
  }

  /// Physicalize the memory view behind `marker`, by cloning it and replacing
  /// only its shape: sizes, strides, result type and coordinate set.
  FailureOr<Value> physicalizeMemView(mlir::ktdp::ConstructMemoryViewOp memViewOp,
                                      const CoordMap &cm,
                                      triton::SpyreTensorLayoutOp marker) {
    OpBuilder b(memViewOp);
    Location loc = memViewOp.getLoc();

    // The physical view's coordinate set is the dense range of its own sizes,
    // recomputed below. A set that says anything more than that — the
    // partitioned or distributed views the attribute exists for — would be
    // overwritten, so reject it rather than lose it silently.
    if (memViewOp.getCoordinateSetAttr().getValue() !=
        buildRangeSetND(b.getContext(), memViewOp.getStaticSizes()))
      return memViewOp.emitError(
          "spyre_tensor_layout: coordinate_set must be the dense range of the "
          "view's sizes to physicalize it; a partitioned set would be "
          "overwritten");

    auto sizes = physicalSizes(cm, memViewOp, b, marker);
    if (failed(sizes))
      return failure();
    auto strides = physicalStrides(cm, memViewOp, *sizes, b, marker);
    if (failed(strides))
      return failure();

    // Clone rather than build: the offset operand, the memory space, the
    // element type and any attribute this pass has never heard of all come
    // across without being named.
    auto physOp = cast<mlir::ktdp::ConstructMemoryViewOp>(b.clone(*memViewOp));
    physOp.getSizesMutable().assign(sizes->dynSizes);
    physOp.getStridesMutable().assign(strides->dynSizes);
    physOp.setStaticSizes(sizes->staticSizes);
    physOp.setStaticStrides(strides->staticSizes);
    // The coordinate set enumerates the view's own index space, so it follows
    // the sizes; a stale logical set would misdescribe the physical view.
    physOp.setCoordinateSetAttr(IntegerSetAttr::get(
        buildRangeSetND(b.getContext(), sizes->staticSizes)));
    physOp.getResult().setType(
        MemRefType::get(sizes->staticSizes,
                        cast<MemRefType>(memViewOp.getResult().getType())
                            .getElementType()));

    if (failed(verifyNothingDropped(memViewOp, physOp)))
      return failure();
    (void)loc;
    return physOp.getResult();
  }

  /// Physicalize one access tile over the already-physical `physMemView`.
  ///
  /// The block shape becomes the physical one, and each physical dim's index is
  /// the coord op applied to its source logical index.
  LogicalResult physicalizeAccessTile(mlir::ktdp::ConstructAccessTilesOp tileOp,
                                      Value physMemView, const CoordMap &cm,
                                      triton::SpyreTensorLayoutOp marker) {
    OpBuilder b(tileOp);
    Location loc = tileOp.getLoc();

    ArrayRef<int64_t> logBlock = tileOp.getResult().getType().getShape();
    unsigned logRank = logBlock.size();
    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, cm.src, cm.op, cm.arg, physBlock))
      return tileOp.emitError("spyre_tensor_layout: cannot derive a static "
                              "physical block shape for this access tile");

    // A stick dim narrower than one stick has no physical meaning: the mod dim
    // would be wider than the data it indexes.
    for (unsigned p = 0, e = cm.physRank(); p < e; ++p) {
      if (cm.opAt(p) != CoordOp::Mod)
        continue;
      int64_t logExtent = logBlock[cm.src[p]];
      if (logExtent != ShapedType::kDynamic && logExtent < cm.arg[p])
        return tileOp.emitError(
                   "spyre_tensor_layout: block extent of stick dim (")
               << logExtent << ") is smaller than the stick size (" << cm.arg[p]
               << "); a stick dim cannot be sub-stick";
    }

    // Recover one index per logical dim. base_map may have fewer inputs than
    // results when the parser deduplicated identical SSA operands, so read the
    // per-dim value through the map rather than off the operand list.
    SmallVector<Value> raw(tileOp.getIndices().begin(),
                           tileOp.getIndices().end());
    AffineMap baseMap = tileOp.getBaseMap();
    SmallVector<Value> logIdx(logRank);
    if (baseMap.getNumResults() == logRank &&
        baseMap.getNumInputs() == raw.size()) {
      for (unsigned d = 0; d < logRank; ++d) {
        auto dimExpr = dyn_cast<AffineDimExpr>(baseMap.getResult(d));
        logIdx[d] = dimExpr ? raw[dimExpr.getPosition()] : raw[0];
      }
    } else {
      logIdx.assign(raw.begin(), raw.end());
    }

    // The rebuilt tile states its order and its set over the PHYSICAL dims, so
    // both are recomputed below. A non-identity order or a non-dense set on the
    // input carries information this pass would overwrite without noticing, so
    // reject it rather than normalize it away silently.
    if (!tileOp.getAccessTileOrder().isIdentity())
      return tileOp.emitError(
          "spyre_tensor_layout: access_tile_order must be the identity to "
          "physicalize this tile; a permuted order would be overwritten");
    if (tileOp.getAccessTileSetAttr().getValue() !=
        buildRangeSetND(b.getContext(), logBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: access_tile_set must be the dense range of the "
          "block shape to physicalize this tile; a non-dense set would be "
          "overwritten");

    SmallVector<Value> physIdx;
    for (unsigned p = 0, e = cm.physRank(); p < e; ++p) {
      // A broadcast dim is the replication axis, not a piece of the logical dim
      // it names, so its subscript is the axis's own origin rather than any
      // function of the logical index. The tile covers all `phys_arg` lanes, so
      // that origin is 0 — and this is the one physical dim whose index reads
      // nothing off logIdx.
      if (cm.opAt(p) == CoordOp::Broadcast) {
        physIdx.push_back(
            arith::ConstantOp::create(b, loc, b.getIndexAttr(0)).getResult());
        continue;
      }
      // The split is built here, so its input is lifted into `index` too: no
      // fixed-width arithmetic is left between a subscript and the values it
      // derives from.
      Value idx = rebuildInIndexDomain(b, loc, logIdx[cm.src[p]]);
      if (idx.getType() != b.getIndexType())
        idx = arith::IndexCastOp::create(b, loc, b.getIndexType(), idx)
                  .getResult();
      switch (cm.opAt(p)) {
      case CoordOp::Identity:
        break;
      case CoordOp::FloorDiv:
        idx = arith::DivSIOp::create(
                  b, loc, idx,
                  arith::ConstantOp::create(b, loc, b.getIndexAttr(cm.arg[p])))
                  .getResult();
        break;
      case CoordOp::Mod:
        idx = arith::RemSIOp::create(
                  b, loc, idx,
                  arith::ConstantOp::create(b, loc, b.getIndexAttr(cm.arg[p])))
                  .getResult();
        break;
      case CoordOp::Broadcast:
        llvm_unreachable("broadcast dims are handled before this switch");
      }
      physIdx.push_back(idx);
    }

    auto identity =
        AffineMap::getMultiDimIdentityMap(cm.physRank(), b.getContext());
    auto physTile = cast<mlir::ktdp::ConstructAccessTilesOp>(b.clone(*tileOp));
    physTile.getBaseMutable().assign(physMemView);
    physTile.getIndicesMutable().assign(physIdx);
    physTile.setBaseMapAttr(AffineMapAttr::get(identity));
    physTile.setAccessTileOrderAttr(AffineMapAttr::get(identity));
    physTile.setAccessTileSetAttr(
        IntegerSetAttr::get(buildRangeSetND(b.getContext(), physBlock)));
    physTile.getResult().setType(
        mlir::ktdp::AccessTileType::get(physBlock, b.getIndexType()));

    if (failed(verifyNothingDropped(tileOp, physTile)))
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "    access tile " << tileOp.getResult().getType() << " -> "
               << physTile.getResult().getType() << "\n");

    for (Operation *user :
         llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        // The load's result shape is the tile's, so retyping it in place is the
        // whole change; every attribute stays on the op that already had it.
        ld.getAccessTileMutable().assign(physTile.getResult());
        auto logResTy = cast<RankedTensorType>(ld.getResult().getType());
        ld.getResult().setType(
            RankedTensorType::get(physBlock, logResTy.getElementType()));
        layoutOf.try_emplace(ld.getResult(), cm);
      } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        // The data tile is not touched here: Phase 2 decides it, from the
        // physical access tile this store now points at.
        st.getAccessTileMutable().assign(physTile.getResult());
        layoutOf.try_emplace(st.getAccessTile(), cm);
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of an access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  /// Physicalize one INDIRECT access tile over the already-physical
  /// `physMemView`.
  ///
  /// Same shape of job as physicalizeAccessTile, different carrier. A direct
  /// tile carries one SSA index per base dim, so its physicalization is SSA
  /// arithmetic. An indirect tile carries one affine *subscript map* per base
  /// dim, stated over `(captured_variables..., intermediate_variables...)` —
  /// the canonical ordering the op's own verifier enforces — so its
  /// physicalization is a substitution on those maps.
  ///
  /// The substitution is the same one Phase 2 performs. Refining the variable
  /// space splits every logical variable a stick split touches into a (stick,
  /// elem) pair, and a subscript stated over the whole logical variable must
  /// recover it as `stick * width + elem`: composeStickSplit, shared with
  /// rebuildMap. Only the numbering differs — loop dims there, captured-then-
  /// intermediate variables here.
  LogicalResult
  physicalizeIndirectAccessTile(mlir::ktdp::ConstructIndirectAccessTilesOp tileOp,
                                Value physMemView, const CoordMap &cm) {
    OpBuilder b(tileOp);
    MLIRContext *ctx = b.getContext();

    ArrayRef<int64_t> logBlock = tileOp.getResult().getType().getShape();
    unsigned logRank = logBlock.size();
    unsigned physRank = cm.physRank();

    auto oldKinds = tileOp.getPerDimSubscriptKinds();
    auto oldMaps = tileOp.getPerDimSubscriptMaps();
    unsigned numCaptured = tileOp.getCapturedVariables().size();

    // The one gate that is a representational limit rather than missing work.
    //
    // An indirect subscript `ind(IDX[expr])` does not compute the base
    // coordinate — it computes an index INTO `IDX`, and the coordinate is the
    // value loaded from there. `floordiv`/`mod` of a stick split would have to
    // apply to that loaded value, and an affine subscript expression cannot
    // reference the result of a load. So the split has nowhere to be written:
    // splitting `expr` instead would split the wrong quantity, the position in
    // the index array rather than the coordinate it yields.
    //
    // Stated against the kinds rather than against dim 0, so it holds wherever
    // the indirect dims sit and however many there are.
    for (unsigned p = 0; p < physRank; ++p) {
      if (cm.opAt(p) == CoordOp::Identity)
        continue;
      int64_t logDim = cm.src[p];
      if (!cast<BoolAttr>(oldKinds[logDim]).getValue())
        continue;
      return tileOp.emitError(
                 "spyre_tensor_layout: logical dim ")
             << logDim
             << " is an indirect (gather) subscript, so it cannot be "
                "stick-split: the coordinate is loaded from the index memref "
                "and an affine subscript cannot do arithmetic on a loaded value";
    }

    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, cm.src, cm.op, cm.arg, physBlock))
      return tileOp.emitError("spyre_tensor_layout: cannot derive a static "
                              "physical block shape for this indirect access "
                              "tile");

    // The rebuilt tile states its order and its set over the PHYSICAL variable
    // space, so both are recomputed below. Anything the input said beyond "the
    // dense range, in order" would be overwritten, so reject it rather than lose
    // it silently — the same contract physicalizeAccessTile states for its own
    // order and set.
    if (!tileOp.getVariablesSpaceOrder().isIdentity())
      return tileOp.emitError(
          "spyre_tensor_layout: variables_space_order must be the identity to "
          "physicalize this tile; a permuted order would be overwritten");
    if (tileOp.getVariablesSpaceSetAttr().getValue() !=
        buildRangeSetND(ctx, logBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: variables_space_set must be the dense range of "
          "the tile shape to physicalize this tile; a non-dense set would be "
          "overwritten");

    // The physical variable space has one variable per physical dim, numbered
    // after the captured ones — the ordering the verifier requires.
    auto physVar = [&](unsigned p) {
      return getAffineDimExpr(numCaptured + p, ctx);
    };

    // Recover each logical variable from the physical ones. This is the same
    // per-physical-dim walk rebuildMap does, reading the same three coord ops
    // off the same marker; it accumulates into the logical variable instead of
    // emitting a result, because here the logical variables are what the old
    // subscripts are stated over.
    SmallVector<AffineExpr> logicalFromPhysical(logRank);
    SmallVector<int64_t> width(logRank, 0);
    SmallVector<AffineExpr> stickHalf(logRank), elemHalf(logRank);
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t logDim = cm.src[p];
      switch (cm.opAt(p)) {
      case CoordOp::Identity:
        logicalFromPhysical[logDim] = physVar(p);
        break;
      case CoordOp::FloorDiv:
        stickHalf[logDim] = physVar(p);
        break;
      case CoordOp::Mod:
        elemHalf[logDim] = physVar(p);
        width[logDim] = cm.arg[p];
        break;
      case CoordOp::Broadcast:
        // A broadcast axis addresses no element of the logical dim, so it
        // contributes nothing to recovering it — the identity dim beside it
        // does, and readCoordMap has already required that dim to exist.
        break;
      }
    }
    for (unsigned d = 0; d < logRank; ++d)
      if (width[d])
        logicalFromPhysical[d] =
            composeStickSplit(stickHalf[d], width[d], elemHalf[d]);

    // Substitute into every subscript map: captured variables keep their slots,
    // logical variables become the recovered composites.
    SmallVector<AffineExpr> oldToNew(numCaptured + logRank);
    for (unsigned c = 0; c < numCaptured; ++c)
      oldToNew[c] = getAffineDimExpr(c, ctx);
    for (unsigned d = 0; d < logRank; ++d) {
      if (!logicalFromPhysical[d])
        return tileOp.emitError("spyre_tensor_layout: logical dim ")
               << d << " is named by no physical dim, so its subscript cannot "
                       "be restated";
      oldToNew[numCaptured + d] = logicalFromPhysical[d];
    }

    unsigned newNumDims = numCaptured + physRank;
    SmallVector<Attribute> newKinds, newMaps;
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t logDim = cm.src[p];
      AffineMap oldMap = cast<AffineMapAttr>(oldMaps[logDim]).getValue();

      // An indirect subscript keeps every result (they index the index memref,
      // whose rank the verifier checks); a direct one has exactly one, and it is
      // the coordinate the coord op applies to. The gate above has already
      // established that only a direct dim reaches a non-identity coord op, so
      // this loop needs no per-kind branch beyond that.
      SmallVector<AffineExpr> results;
      for (AffineExpr r : oldMap.getResults()) {
        AffineExpr e = r.replaceDims(oldToNew);
        switch (cm.opAt(p)) {
        case CoordOp::Identity:
          break;
        case CoordOp::FloorDiv:
          e = e.floorDiv(cm.arg[p]);
          break;
        case CoordOp::Mod:
          e = e % cm.arg[p];
          break;
        case CoordOp::Broadcast:
          // The replication axis's own origin, not a function of the logical
          // subscript — the same answer physicalizeAccessTile gives by pushing a
          // constant 0.
          e = getAffineConstantExpr(0, ctx);
          break;
        }
        results.push_back(e);
      }

      newKinds.push_back(cast<BoolAttr>(oldKinds[logDim]));
      newMaps.push_back(AffineMapAttr::get(
          AffineMap::get(newNumDims, /*symbolCount=*/0, results, ctx)));
    }

    // An indirect memref is listed once per indirect dim, in dim order, so the
    // list has to be rebuilt in the PHYSICAL dim order the new kinds are stated
    // in — the verifier pairs the two positionally.
    SmallVector<Value> newIndirect;
    for (unsigned p = 0; p < physRank; ++p) {
      int64_t logDim = cm.src[p];
      if (!cast<BoolAttr>(oldKinds[logDim]).getValue())
        continue;
      unsigned oldIndirectIdx = 0;
      for (int64_t d = 0; d < logDim; ++d)
        if (cast<BoolAttr>(oldKinds[d]).getValue())
          ++oldIndirectIdx;
      newIndirect.push_back(tileOp.getIndirectMemrefs()[oldIndirectIdx]);
    }

    auto physTile = mlir::ktdp::ConstructIndirectAccessTilesOp::create(
        b, tileOp.getLoc(),
        mlir::ktdp::AccessTileType::get(physBlock, b.getIndexType()),
        physMemView, ArrayAttr::get(ctx, newKinds),
        ArrayAttr::get(ctx, newMaps), newIndirect,
        tileOp.getCapturedVariables(), tileOp.getSymbolOperands(),
        buildRangeSetND(ctx, physBlock),
        AffineMap::getMultiDimIdentityMap(physRank, ctx));

    // The subscript maps are what the substitution rewrote, and they are stated
    // over the refined variable space, so they are the output worth reading:
    // everything else about an indirect tile carries across unchanged.
    LLVM_DEBUG({
      llvm::dbgs() << "    indirect tile " << tileOp.getResult().getType()
                   << " -> " << physTile.getResult().getType() << ", "
                   << numCaptured << " captured + " << physRank
                   << " intermediate variable(s)\n";
      for (unsigned p = 0; p < physRank; ++p)
        llvm::dbgs() << "      phys dim " << p << " (from d" << cm.src[p] << ":"
                     << coordOpName(cm.opAt(p)) << ")"
                     << (cast<BoolAttr>(newKinds[p]).getValue() ? " ind" : "")
                     << " subscript "
                     << cast<AffineMapAttr>(newMaps[p]).getValue() << "\n";
    });

    for (Operation *user :
         llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        ld.getAccessTileMutable().assign(physTile.getResult());
        auto logResTy = cast<RankedTensorType>(ld.getResult().getType());
        ld.getResult().setType(
            RankedTensorType::get(physBlock, logResTy.getElementType()));
        layoutOf.try_emplace(ld.getResult(), cm);
      } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        st.getAccessTileMutable().assign(physTile.getResult());
        layoutOf.try_emplace(st.getAccessTile(), cm);
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of an indirect access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  LogicalResult physicalizeDescriptor(triton::SpyreTensorLayoutOp marker) {
    Value desc = marker.getDesc();
    if (!isLoweredDescriptor(desc))
      return marker.emitError(
          "spyre_tensor_layout: desc operand is not a lowered descriptor — "
          "pass must run after LowerDescriptorMemory");
    Value memView = getDescriptorMemView(desc);
    auto memViewOp = memView.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
    if (!memViewOp)
      return marker.emitError("spyre_tensor_layout: cannot locate "
                              "construct_memory_view behind the bridge cast");

    auto cm = readCoordMap(marker, memViewOp.getStaticSizes().size());
    if (failed(cm))
      return failure();

    LLVM_DEBUG({
      llvm::dbgs() << "  descriptor at " << marker.getLoc() << ": ";
      printCoordMap(llvm::dbgs(), *cm);
      llvm::dbgs() << "\n";
    });

    // Read the tiles before mutating anything: physicalizing the view does not
    // move them, but erasing one invalidates a walk over the users.
    SmallVector<mlir::ktdp::ConstructAccessTilesOp> tiles;
    SmallVector<mlir::ktdp::ConstructIndirectAccessTilesOp> indirectTiles;
    for (Operation *user : memView.getUsers())
      if (auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(user))
        tiles.push_back(tile);
      else if (auto indirect =
                   dyn_cast<mlir::ktdp::ConstructIndirectAccessTilesOp>(user))
        indirectTiles.push_back(indirect);

    auto physMemView = physicalizeMemView(memViewOp, *cm, marker);
    if (failed(physMemView))
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "    view " << memViewOp.getResult().getType() << " -> "
               << physMemView->getType() << ", " << tiles.size()
               << " direct tile(s), " << indirectTiles.size()
               << " indirect tile(s)\n");

    for (auto tile : tiles)
      if (failed(physicalizeAccessTile(tile, *physMemView, *cm, marker)))
        return failure();
    for (auto tile : indirectTiles)
      if (failed(physicalizeIndirectAccessTile(tile, *physMemView, *cm)))
        return failure();

    deadLogicalMemViews.push_back(memViewOp);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Phase 2 — the one rewrite
  //===--------------------------------------------------------------------===//

  /// The layout a value is physicalized under, or null when it carries none and
  /// therefore stays logical.
  const CoordMap *layoutFor(Value v) {
    auto it = layoutOf.find(v);
    return it == layoutOf.end() ? nullptr : &it->second;
  }

  /// Is `v` already at the physical rank its layout prescribes?
  ///
  /// Rank, not shape: the layout is stated over logical dims, so it is the rank
  /// that distinguishes a value still to be retyped from one already retyped.
  /// Testing the TYPE and not a map is the non-circular half of the guard — the
  /// markers are ground truth and the type comes from them, whereas a map is
  /// this pass's own arithmetic.
  bool atPhysicalRank(Value v) {
    const CoordMap *cm = layoutFor(v);
    if (!cm)
      return true; // no layout, so nothing to be at
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    return !ty || ty.getRank() == (int64_t)cm->physRank();
  }

  /// Why `op` is inconsistent, or an empty string when it is not — so this is
  /// both the consistency test and its explanation.
  ///
  /// One function rather than two so the two cannot drift: an op the guard
  /// rejects always has a reason to print, and a reason printed is always the
  /// one the guard acted on.
  std::string inconsistencyReason(linalg::GenericOp op) {
    std::string reason;
    llvm::raw_string_ostream os(reason);
    for (auto [i, v] : llvm::enumerate(op->getOperands()))
      if (!atPhysicalRank(v)) {
        os << "operand " << i << " is " << v.getType() << " but its layout "
           << "prescribes physical rank " << layoutFor(v)->physRank();
        return reason;
      }
    for (auto [i, v] : llvm::enumerate(op->getResults()))
      if (!atPhysicalRank(v)) {
        os << "result " << i << " is " << v.getType() << " but its layout "
           << "prescribes physical rank " << layoutFor(v)->physRank();
        return reason;
      }
    // Every map must have one result per dim of the operand it addresses, and
    // all of them must share one loop domain. A rank the operands have moved
    // past is exactly the state where an operand was retyped and this op has
    // not caught up.
    for (auto [i, map, operand] :
         llvm::enumerate(op.getIndexingMapsArray(), op->getOperands())) {
      auto ty = dyn_cast<RankedTensorType>(operand.getType());
      if (ty && map.getNumResults() != (unsigned)ty.getRank()) {
        os << "indexing map " << i << " (" << map << ") has "
           << map.getNumResults() << " result(s) but operand " << i << " is "
           << ty;
        return reason;
      }
    }
    return reason;
  }

  /// Is this op consistent, given its inputs and outputs?
  ///
  /// This is simultaneously the rewrite's guard and its postcondition, which is
  /// what makes termination structural: the rewrite's job is to make this true,
  /// so the driver cannot re-fire an op the rewrite just finished. It is a state
  /// predicate over the IR and the markers, not a record of what the pass has
  /// done, so nothing can get out of sync with the IR.
  ///
  /// An op with one physical and one still-logical operand is simply
  /// inconsistent, so it fires — which is the right answer and needs no
  /// separate notion of "partially retyped".
  /// The second conjunct is about RANK, not about map contents. Asking whether
  /// the maps are *right* would be asking whether this pass computed them
  /// correctly — circular, since the maps are derived from the markers by this
  /// very rewrite. Asking whether they are stated at the right rank compares
  /// them against the types, which come from the markers, so it is not.
  bool isConsistent(linalg::GenericOp op) {
    return inconsistencyReason(op).empty();
  }

  /// Give `v` the layout `cm`, and report whether that is new information.
  /// A value already carrying a layout keeps it: layouts come from markers, and
  /// a second opinion about one would mean two markers disagree.
  bool assignLayout(Value v, const CoordMap &cm) {
    return layoutOf.try_emplace(v, cm).second;
  }

  //===--------------------------------------------------------------------===//
  // The rewrite
  //===--------------------------------------------------------------------===//

  /// Rewrite one generic so that it is consistent: retype every operand and
  /// result to the physical type its own layout prescribes, and restate the
  /// indexing maps and iterator types over the rebuilt loop domain.
  ///
  /// Nothing here asks what the op *is*. The maps and iterators already
  /// discriminate elementwise from broadcast from transpose from reduce from
  /// contraction, and the rebuild reads them.
  LogicalResult rewriteGeneric(linalg::GenericOp op) {
    MLIRContext *ctx = op.getContext();
    unsigned numLoops = op.getNumLoops();

    // A generic's result takes its type from the `outs` operand that backs it,
    // so a layout on the result is a layout on that operand. This is how a
    // store's requirement reaches the accumulator it is written through: the
    // store records the layout against the result, and the operand that
    // supplies the result's shape inherits it here.
    for (auto [res, init] : llvm::zip_equal(op.getResults(), op.getDpsInits()))
      if (const CoordMap *cm = layoutFor(res))
        assignLayout(init, *cm);

    // The layouts as they stand. An operand whose type is already physical
    // states its layout over its physical dims, so its logical map is the one
    // on the op either way: the op's maps are never rewritten in place, only
    // replaced, so they stay logical until this op is rebuilt.
    SmallVector<const CoordMap *> layouts;
    for (Value v : op->getOperands())
      layouts.push_back(layoutFor(v));

    SmallVector<RebuildOperand> rebuildOperands;
    SmallVector<AffineMap> logicalMaps = op.getIndexingMapsArray();
    for (auto [i, m] : llvm::enumerate(logicalMaps))
      rebuildOperands.push_back(RebuildOperand{m, layouts[i]});

    // The `outs` operand backing result 0 is the loop frame, so it is read off
    // the op through the destination-style interface rather than assumed to be
    // last: `rebuildOperands` is built in operand order, and a positional guess
    // that ever went wrong would not fail the verifier -- the maps stay
    // internally consistent -- but would state the relation in a frame nothing
    // else uses, surfacing only as a shape mismatch inside ktir-cpu.
    unsigned resultIdx =
        op.getDpsInitOperand(0)->getOperandNumber();
    if (resultIdx >= rebuildOperands.size())
      return op.emitError("rewrite-descriptor-layout-generic: the outs operand "
                          "backing result 0 is outside the indexing maps");

    auto dom = buildLoopDomain(rebuildOperands, resultIdx, numLoops,
                               [&]() { return op.emitError(); });
    if (failed(dom))
      return failure();

    // Rebuild the maps over the new domain before touching any type: a map
    // states where an operand's dims live, so it must agree with the type the
    // operand is about to be given.
    SmallVector<AffineMap> physMaps;
    for (const RebuildOperand &o : rebuildOperands)
      physMaps.push_back(rebuildMap(o, *dom, ctx));

    SmallVector<utils::IteratorType> physIterators =
        rebuildIterators(op.getIteratorTypesArray(), *dom);

    // The rebuild is where a wrong answer would originate, so the trace shows
    // all three of its parts together: the domain each logical dim expanded
    // into, and every operand's map before and after. Read side by side they say
    // whether a map result landed on the loop dim the domain assigned it.
    LLVM_DEBUG({
      llvm::dbgs() << "    rebuilding at " << op.getLoc() << ": " << numLoops
                   << " logical loop dim(s) -> " << dom->numLoopDims << "\n";
      for (unsigned d = 0; d < numLoops; ++d) {
        llvm::dbgs() << "      logical d" << d << " -> ";
        if (dom->isSplit(d))
          llvm::dbgs() << "stick d" << dom->stickDim[d] << " + lane d"
                       << dom->elemDim[d] << " at width " << dom->width[d];
        else
          llvm::dbgs() << "whole d" << dom->stickDim[d];
        llvm::dbgs() << "\n";
      }
      for (auto [i, o] : llvm::enumerate(rebuildOperands)) {
        for (unsigned p = 0, e = o.broadcastDim.size(); p < e; ++p)
          if (o.broadcastDim[p] >= 0)
            llvm::dbgs() << "      operand " << i << " broadcast phys dim " << p
                         << " -> fresh loop d" << o.broadcastDim[p] << "\n";
        llvm::dbgs() << "      operand " << i << " " << o.logicalMap << " -> "
                     << physMaps[i];
        if (!o.layout)
          llvm::dbgs() << " (no layout, stays logical)";
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "      iterators [";
      llvm::interleaveComma(physIterators, llvm::dbgs(),
                            [&](utils::IteratorType t) {
                              llvm::dbgs() << utils::stringifyIteratorType(t);
                            });
      llvm::dbgs() << "]\n";
    });

    // Retype each operand that is not yet at its physical rank. A value with a
    // layout but no producer this pass can retype is a chain the rewrite cannot
    // complete, and says so.
    OpBuilder b(op);
    for (auto [i, operand] : llvm::enumerate(op->getOpOperands())) {
      if (!layouts[i] || atPhysicalRank(operand.get()))
        continue;
      auto physTy = physicalTensorType(
          *layouts[i], cast<RankedTensorType>(operand.get().getType()));
      if (failed(physTy))
        return op.emitError("rewrite-descriptor-layout-generic: operand ")
               << i << " has no static physical shape under its layout";
      if (failed(retypeToPhysical(operand.get(), *physTy, b)))
        return failure();
    }

    // Clone rather than build: the body, the attributes, the location and
    // anything added to linalg.generic later all ride along, and only the three
    // things this rewrite owns are replaced.
    auto physOp = cast<linalg::GenericOp>(b.clone(*op));
    physOp.setIndexingMapsAttr(b.getAffineMapArrayAttr(physMaps));
    physOp.setIteratorTypesAttr(
        b.getArrayAttr(llvm::to_vector(llvm::map_range(
            physIterators, [&](utils::IteratorType t) -> Attribute {
              return linalg::IteratorTypeAttr::get(ctx, t);
            }))));
    // A generic's results take their types from its `outs` operands, which the
    // loop above has already retyped.
    for (auto [res, out] :
         llvm::zip_equal(physOp.getResults(), physOp.getDpsInits()))
      res.setType(out.getType());
    for (auto [oldRes, newRes] :
         llvm::zip_equal(op.getResults(), physOp.getResults()))
      if (const CoordMap *cm = layoutFor(oldRes))
        assignLayout(newRes, *cm);

    if (failed(verifyNothingDropped(op, physOp)))
      return failure();

    op.getResults().replaceAllUsesWith(physOp.getResults());
    op.erase();
    return success();
  }

  /// Give `v` the physical type `physTy`, by retyping its producer.
  ///
  /// Only the producers whose result type is a pure function of their operands'
  /// are handled here — the ones a shape change propagates cleanly through. A
  /// generic is NOT one of them: it is retyped by the rewrite firing on it, not
  /// from a consumer, so it is left alone and the driver reaches it next round.
  LogicalResult retypeToPhysical(Value v, RankedTensorType physTy,
                                 OpBuilder &b) {
    Operation *def = v.getDefiningOp();
    if (isa_and_nonnull<linalg::GenericOp>(def))
      return success();
    if (auto empty = dyn_cast_or_null<tensor::EmptyOp>(def)) {
      // An uninitialised accumulator carries no values, so its physical form is
      // just the same op at the physical shape.
      if (!empty.getType().hasStaticShape())
        return empty.emitError("rewrite-descriptor-layout-generic: cannot "
                               "physicalize a dynamically shaped tensor.empty");
      empty.getResult().setType(physTy);
      return success();
    }
    if (auto cst = dyn_cast_or_null<arith::ConstantOp>(def)) {
      // A splat is the one constant whose physical form is a relabelling: every
      // element is the same, so no element moves. Any other constant would need
      // its elements permuted into stick order, which is a data rewrite this
      // pass does not do.
      auto splat = dyn_cast<SplatElementsAttr>(cst.getValue());
      if (!splat)
        return cst.emitError("rewrite-descriptor-layout-generic: cannot "
                             "physicalize a non-splat constant; its elements "
                             "would have to be reordered into stick layout");
      cst.setValueAttr(SplatElementsAttr::get(physTy,
                                              splat.getSplatValue<Attribute>()));
      cst.getResult().setType(physTy);
      return success();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "    decline: cannot restate ";
      if (Operation *d = v.getDefiningOp())
        llvm::dbgs() << d->getName() << " at " << d->getLoc();
      else
        llvm::dbgs() << "block argument " << v;
      llvm::dbgs() << " as " << physTy << "\n";
    });
    return v.getDefiningOp()
               ? v.getDefiningOp()->emitError(
                     "rewrite-descriptor-layout-generic: this op produces a "
                     "value on a physicalized chain but the rewrite cannot "
                     "restate it at physical shape")
               : failure();
  }

  /// Rewrite one store so that it is consistent.
  ///
  /// The store's access tile is already physical (Phase 1 did it), and
  /// ktdp.store requires its data tile to match the access tile's iteration
  /// shape. So the store knows the shape its data tile must have, and gives it
  /// that shape — which is how a requirement reaches a producer without anyone
  /// looking one up.
  LogicalResult rewriteStore(mlir::ktdp::StoreOp store, bool &progress) {
    const CoordMap *cm = layoutFor(store.getAccessTile());
    if (!cm)
      return success();
    auto dataTy = cast<RankedTensorType>(store.getDataTile().getType());
    ArrayRef<int64_t> want =
        cast<mlir::ktdp::AccessTileType>(store.getAccessTile().getType())
            .getShape();
    if (dataTy.getShape() == want)
      return success();

    auto physTy = physicalTensorType(*cm, dataTy);
    if (failed(physTy))
      return store.emitError("rewrite-descriptor-layout-generic: the store's "
                             "data tile has no static physical shape under the "
                             "destination's layout");
    if ((*physTy).getShape() != want)
      return store.emitError("rewrite-descriptor-layout-generic: the "
                             "destination's layout gives the data tile shape ")
             << *physTy << ", which does not match the access tile's iteration "
             << "shape";

    // The data tile's producer is what has to change; recording the layout is
    // what makes the producer inconsistent, and so acted on next round.
    // `progress` reports whether that record is new, since a store whose
    // producer has not caught up yet changes nothing else and the driver must
    // not read that as a fixpoint.
    progress = assignLayout(store.getDataTile(), *cm) || progress;
    LLVM_DEBUG(llvm::dbgs()
               << "    store at " << store.getLoc() << ": data tile " << dataTy
               << " -> " << *physTy
               << ", pulled from the access tile's iteration shape\n");
    OpBuilder b(store);
    return retypeToPhysical(store.getDataTile(), *physTy, b);
  }

  /// Apply the rewrite until nothing on a physicalized chain is inconsistent.
  ///
  /// Both directions matter and both come for free: loads push physical types
  /// forward as operands, stores pull them backward as results, and a generic
  /// in the middle is reached from either end. Each round re-examines every
  /// candidate, so a type that moved in one round is acted on in the next.
  LogicalResult runRewrite(ModuleOp module) {
    // One round per op on the longest chain is the worst case: each round moves
    // physical information at least one op further along. The bound is
    // generous, and exceeding it means the guard is not being falsified — a bug
    // in the rewrite, reported rather than silently accepted.
    unsigned numOps = 0;
    module.walk([&](Operation *) { ++numOps; });
    unsigned cap = numOps + 2;

    for (unsigned round = 0; round < cap; ++round) {
      SmallVector<mlir::ktdp::StoreOp> stores;
      SmallVector<linalg::GenericOp> generics;
      module.walk([&](Operation *op) {
        if (auto st = dyn_cast<mlir::ktdp::StoreOp>(op))
          stores.push_back(st);
        else if (auto g = dyn_cast<linalg::GenericOp>(op))
          generics.push_back(g);
      });

      bool changed = false;
      // Stores first: a store is the only op that can start the backward
      // direction, and doing it first saves a round on every chain that has
      // one.
      // Split out per store rather than accumulating into `changed` directly,
      // so the trace can say how many stores contributed rather than only that
      // one did. `changed` ends up the same disjunction either way.
      unsigned storesProgressed = 0, genericsFired = 0;
      for (auto st : stores) {
        bool storeProgress = false;
        if (failed(rewriteStore(st, storeProgress)))
          return failure();
        changed |= storeProgress;
        storesProgressed += storeProgress;
      }
      for (auto g : generics) {
        if (isConsistent(g))
          continue;
        LLVM_DEBUG(llvm::dbgs() << "  round " << round
                                << ": generic inconsistent at " << g.getLoc()
                                << ": " << inconsistencyReason(g) << "\n");
        if (failed(rewriteGeneric(g)))
          return failure();
        changed = true;
        ++genericsFired;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "  round " << round << ": " << storesProgressed << " of "
                 << stores.size() << " store(s) recorded new layout, "
                 << genericsFired << " of " << generics.size()
                 << " generic(s) rewritten\n");
      if (!changed)
        return checkAllConsistent(module);
    }
    // Naming the condition is not enough to act on: the culprit is whichever op
    // the guard still rejects after the cap, so list those before the error.
    LLVM_DEBUG({
      llvm::dbgs() << "  cap of " << cap << " round(s) exhausted; still "
                   << "inconsistent:\n";
      module.walk([&](linalg::GenericOp g) {
        std::string reason = inconsistencyReason(g);
        if (!reason.empty())
          llvm::dbgs() << "    " << g.getLoc() << ": " << reason << "\n";
      });
    });
    return module.emitError("rewrite-descriptor-layout-generic: the rewrite did "
                            "not reach a fixpoint; an op is not falsifying the "
                            "consistency guard");
  }

  /// Report every op the rewrite could not make consistent.
  ///
  /// A decline is not silence: an op left inconsistent on a physicalized chain
  /// is an error naming the op, because the surviving op is exactly what could
  /// not be reconciled. This is where a shape outside the agreed cases lands.
  /// Reject, before Phase 1 mutates anything, an op that will end up reading a
  /// physicalized value the rewrite cannot restate.
  ///
  /// This has to run first. Phase 1 retypes a load in place, so a consumer left
  /// at logical rank is invalid IR the moment that happens — and MLIR's own
  /// verifier reports it as a rank mismatch against an indexing map, naming
  /// neither this pass nor what it could not handle. A named linalg.matmul is
  /// the case that matters today: LowerComputeOps still emits one for tt.dot,
  /// and this pass rewrites only generics.
  ///
  /// TODO: this looks redundant against checkAllConsistent's non-generic branch —
  /// near-identical wording, and both reject a non-generic reading a
  /// physicalized value — but the two ask different questions and cannot simply
  /// be merged. This one is STRUCTURAL: is the consumer a kind the rewrite knows
  /// how to restate, which is answerable at any time. checkAllConsistent is
  /// about STATE: has this op been brought to physical rank, which is only
  /// meaningful once the rewrite has run, because before Phase 1 nothing is at
  /// physical rank and the state test therefore reports every ordinary kernel as
  /// an error. What is genuinely duplicated is the TRAVERSAL, not the predicate:
  /// this check needs its own marker -> memview -> tile -> load walk only because
  /// `layoutOf` is seeded during Phase 1's mutation and so is empty beforehand.
  /// Two ways to remove that, neither collapsing to one function: seed
  /// `layoutOf` in a pre-pass sharing physicalizeDescriptor's pure-analysis part
  /// (separable — its first mutation is physicalizeMemView) and let this check
  /// use layoutFor; or split checkAllConsistent into structural and state
  /// halves, calling the structural half at both points.
  LogicalResult checkConsumersAreRewritable(ModuleOp module) {
    LogicalResult result = success();
    module.walk([&](triton::SpyreTensorLayoutOp marker) {
      Value desc = marker.getDesc();
      if (!isLoweredDescriptor(desc))
        return;
      Value memView = getDescriptorMemView(desc);
      for (Operation *tile : memView.getUsers())
        for (Value tileRes : tile->getResults())
          for (Operation *user : tileRes.getUsers()) {
            auto ld = dyn_cast<mlir::ktdp::LoadOp>(user);
            if (!ld)
              continue;
            for (Operation *consumer : ld.getResult().getUsers())
              if (!isa<linalg::GenericOp, mlir::ktdp::StoreOp>(consumer)) {
                LLVM_DEBUG(llvm::dbgs()
                           << "  decline: " << consumer->getName() << " at "
                           << consumer->getLoc()
                           << " reads a load this marker would physicalize, and "
                           << "is not a linalg.generic\n");
                consumer->emitError(
                    "rewrite-descriptor-layout-generic: this op reads a value "
                    "on a physicalized chain, but the rewrite restates only "
                    "linalg.generic; spell this op as one");
                result = failure();
              }
          }
    });
    return result;
  }

  LogicalResult checkAllConsistent(ModuleOp module) {
    LogicalResult result = success();
    module.walk([&](Operation *op) {
      if (auto g = dyn_cast<linalg::GenericOp>(op)) {
        if (!isConsistent(g)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  decline: generic at " << g.getLoc() << ": "
                     << inconsistencyReason(g) << "\n");
          g.emitError("rewrite-descriptor-layout-generic: this op is on a "
                      "physicalized chain but could not be restated at "
                      "physical shape");
          result = failure();
        }
        return;
      }
      // Any other op reading a value the rewrite retyped is outside what one
      // generic rewrite covers, and saying so beats leaving IR whose only
      // complaint comes from a verifier that names neither this pass nor the op.
      //
      // The test is on the CONSUMER, not on the value: by the time this runs the
      // value has already been retyped, so asking whether it is at physical rank
      // would answer yes and let the op through.
      if (isa<mlir::ktdp::StoreOp, mlir::ktdp::LoadOp>(op))
        return;
      for (auto [i, v] : llvm::enumerate(op->getOperands()))
        if (layoutFor(v)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  decline: " << op->getName() << " at " << op->getLoc()
                     << " reads retyped operand " << i << " (" << v.getType()
                     << "), and is not a linalg.generic\n");
          op->emitError("rewrite-descriptor-layout-generic: this op reads a "
                        "value the rewrite retyped, but the rewrite restates "
                        "only linalg.generic, so this op still names the "
                        "logical type");
          result = failure();
          return;
        }
    });
    return result;
  }

  //===--------------------------------------------------------------------===//
  // Phase 3 — marker cleanup
  //===--------------------------------------------------------------------===//

  void eraseMarker(triton::SpyreTensorLayoutOp marker) {
    if (!marker->getBlock())
      return;
    auto castOp = marker.getDesc().getDefiningOp<UnrealizedConversionCastOp>();
    marker.erase();
    if (castOp && castOp.use_empty())
      castOp.erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The pass is invocable directly (spyre-triton-opt), bypassing the
    // frontend's own validation, so reject an unrecognized value rather than
    // falling through to one of the two branches.
    if (dataLayout != "device" && dataLayout != "host") {
      module.emitError("rewrite-descriptor-layout-generic: data-layout must be "
                       "'device' or 'host', got '")
          << dataLayout << "'";
      return signalPassFailure();
    }
    hwDataLayout = (dataLayout == "device");

    SmallVector<triton::SpyreTensorLayoutOp> markers;
    module.walk([&](triton::SpyreTensorLayoutOp op) { markers.push_back(op); });

    LLVM_DEBUG(llvm::dbgs()
               << "[rewrite-descriptor-layout-generic] " << markers.size()
               << " layout marker(s), data-layout=" << dataLayout << "\n");

    if (failed(checkConsumersAreRewritable(module)))
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "[rewrite-descriptor-layout-generic] Phase 1: physicalizing "
               << "descriptors\n");
    for (auto marker : markers)
      if (failed(physicalizeDescriptor(marker)))
        return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "[rewrite-descriptor-layout-generic] Phase 2: greedy "
               << "rewrite\n");
    if (failed(runRewrite(module)))
      return signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "[rewrite-descriptor-layout-generic] Phase 3: erasing "
               << markers.size() << " marker(s)\n");
    for (auto marker : markers)
      eraseMarker(marker);

    // With the bridge casts gone the superseded logical views have no users
    // left. Check rather than assume: a view whose tiles could not all be
    // re-pointed still has real ones.
    for (auto memViewOp : deadLogicalMemViews)
      if (memViewOp->getBlock() && memViewOp.use_empty())
        memViewOp.erase();
  }
};

} // namespace
