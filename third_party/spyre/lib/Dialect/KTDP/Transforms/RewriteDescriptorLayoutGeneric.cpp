//===- RewriteDescriptorLayoutGeneric.cpp ---------------------------------===//
//
// Rewrites logical tensor descriptors to their physical (stick-tiled) layout,
// driven by tt.spyre_tensor_layout markers, and retypes the compute ops on the
// annotated chain — which are all linalg.generic.
//
// The marker carries the physical layout as the OpSpec `device_coordinates`
// form, three i64 arrays with one entry per physical dim:
//   phys_src[p] : the logical dim physical dim p derives from
//   phys_op[p]  : 0 = identity, 1 = floordiv, 2 = mod, 3 = splat
//   phys_arg[p] : divisor (floordiv) / modulus (mod) / lane count (splat);
//                 ignored for identity
// e.g. [M,N] stick-on-N -> phys_src=[1,0,1] phys_op=[1,0,2] phys_arg=[64,0,64]
//   => physical size [ceil(N/64), M, 64].
// A splat dim replicates its source logical dim across `phys_arg` lanes
// instead of partitioning it, so [M] -> phys_src=[0,0] phys_op=[0,3]
// phys_arg=[0,64] gives physical size [M, 64].
//
//   physicalizeDescriptors   physicalize each annotated descriptor: memory view,
//            access tiles, loads. Stores have their access tile redirected. Each
//            physicalized view is recorded against the layout it was
//            physicalized under.
//   rewriteAdjacentGenerics  for each recorded view, find the generics that read
//            its loads or supply its stores, and restate each one over the
//            physical loop domain. Every generic the pass considers is reached
//            this way, so the recorded views are the scope; no fixpoint, no
//            pattern driver, and each generic is rewritten at most once.
//   eraseMarkers  erase the markers and their now-dead bridge casts.
//
// VOCABULARY. Two words below mean something narrower than they look, and one of
// them collides with an op.
//
//   LOOP always means a dimension of a linalg.generic's ITERATION SPACE — what
//   linalg itself counts with getNumLoops, one entry of indexing_maps' domain and
//   one entry of iterator_types. It never means an scf.for. The distinction
//   carries this pass's central claim: splitting a dim adds a REDUCTION LOOP DIM
//   to a contraction's iteration space and no loop op anywhere, which is why the
//   tests can assert two reduction loop dims and `CHECK-NOT: scf.for` in the same
//   breath.
//
//   SPLIT is a DELINEARIZATION of a logical dim over the basis
//   (ceildiv(N, W), W), where W is the stick width: exactly the coordinate pair
//   `affine.delinearize_index` computes, `x -> (x floordiv W, x mod W)`. The two
//   halves are the components of that multi-index, and they keep the hardware's
//   names — STICK for the first, LANE for the second — because that is what they
//   are on the device. W is a basis element.
//
//   The COMPOSITE an operand holding the dim whole has to carry is the inverse:
//   the LINEARIZATION `stick * W + lane` over the same basis, which is what
//   `affine.linearize_index` computes. See linearizeStickLane.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Utils/Utility.h"
#include "Utils/Utility.h"
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
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
// Local CoordOp, applyStatic, applyCoordMap
//===----------------------------------------------------------------------===//

// CoordOp: use Splat (value 3), NOT Broadcast
enum class CoordOp : int64_t { Identity = 0, FloorDiv = 1, Mod = 2, Splat = 3 };

/// Is `set` the dense range of `shape` — for every dim, the pair of constraints
/// that bounds it to [0, extent)?
///
/// Asked constraint by constraint through simplifyAffineExpr rather than by
/// comparing the whole IntegerSet against buildRangeSetND, because an
/// IntegerSet compares by uniqued identity and a dynamic extent's upper bound
/// is not identical across a round trip through IR text: the bound is built as
/// `(sym - 1) - d`, MLIR prints that as `-d + sym - 1`, and the parser reads
/// that text back as `((-d) + sym) - 1`. The two trees print the same and mean
/// the same but are not the same attribute, so an identity comparison rejects a
/// view whose set was parsed rather than built — every view in a hand-written
/// test, and every view in a module handed between two tool invocations. The
/// question here is what the set means, so ask it of the canonical form.
static bool isDenseRangeSet(IntegerSet set, MLIRContext *ctx,
                            ArrayRef<int64_t> shape) {
  IntegerSet want = buildRangeSetND(ctx, shape);
  if (set.getNumDims() != want.getNumDims() ||
      set.getNumSymbols() != want.getNumSymbols() ||
      set.getNumConstraints() != want.getNumConstraints())
    return false;
  for (unsigned i = 0, e = want.getNumConstraints(); i < e; ++i) {
    if (set.isEq(i) != want.isEq(i))
      return false;
    if (simplifyAffineExpr(set.getConstraint(i), set.getNumDims(),
                           set.getNumSymbols()) !=
        simplifyAffineExpr(want.getConstraint(i), want.getNumDims(),
                           want.getNumSymbols()))
      return false;
  }
  return true;
}

// applyStatic: apply one coord op to a static extent
inline std::optional<int64_t> applyStatic(int64_t logical, CoordOp op,
                                          int64_t arg) {
  switch (op) {
  case CoordOp::Identity:
    return (logical == mlir::ShapedType::kDynamic)
               ? std::nullopt
               : std::optional<int64_t>(logical);
  case CoordOp::FloorDiv:
    if (logical == mlir::ShapedType::kDynamic)
      return std::nullopt;
    return arg == 0 ? std::optional<int64_t>(std::nullopt)
                    : std::optional<int64_t>((logical + arg - 1) / arg);
  case CoordOp::Mod:
    return arg;
  case CoordOp::Splat:
    return arg;
  }
  return std::nullopt;
}

// applyCoordMap: compute physical extents from logical extents
inline bool applyCoordMap(ArrayRef<int64_t> logSizes, ArrayRef<int64_t> physSrc,
                          ArrayRef<int64_t> physOp, ArrayRef<int64_t> physArg,
                          SmallVectorImpl<int64_t> &out) {
  unsigned physRank = physSrc.size();
  out.resize(physRank);
  for (unsigned k = 0; k < physRank; ++k) {
    auto sz = applyStatic(logSizes[physSrc[k]], static_cast<CoordOp>(physOp[k]),
                          physArg[k]);
    if (!sz)
      return false;
    out[k] = *sz;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// IndexDomain lifting helpers
//
// Copies, and deliberately so. The originals are in IndexDomain.cpp under
// RewriteDescriptorLayout/, the named pass's own subdirectory, and that whole
// subdirectory goes away when the named pass does — the deletion note at the top
// of RewriteDescriptorLayout.cpp sequences it. Sharing these would make this
// pass, the one that survives, depend on a directory scheduled for deletion. So
// these are the copies that outlive it.
//
// Not the same judgement as buildRangeSetND, which this pass calls from
// Dialect/KTDP/Utils instead of copying: that one is shared infrastructure in a
// settled home. These only look like duplication because their current home is
// doomed.
//===----------------------------------------------------------------------===//

static bool isIdentityTracingIntCast(Operation *op) {
  return isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::TruncIOp,
             arith::ExtSIOp, arith::ExtUIOp>(op);
}

static bool isValuePreservingIntCast(Operation *op) {
  return isa<arith::IndexCastOp, arith::ExtSIOp>(op);
}

[[maybe_unused]] static BlockArgument traceToMLIRBlockArg(Value v) {
  while (true) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return ba;
    auto *op = v.getDefiningOp();
    if (!op)
      return nullptr;
    if (isIdentityTracingIntCast(op)) {
      v = op->getOperand(0);
      continue;
    }
    if (isa<arith::MulIOp, arith::DivSIOp, arith::RemSIOp, arith::AddIOp>(op)) {
      if (op->getNumOperands() == 2 &&
          triton::spyre::getConstantInt(op->getOperand(1))) {
        v = op->getOperand(0);
        continue;
      }
    }
    return nullptr;
  }
}

static bool isArithIntOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::AndIOp,
             arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
             arith::ShRUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
             arith::MinUIOp>(op);
}

static bool isRebuildableIntArith(Operation *op) {
  return isa<arith::MulIOp, arith::DivSIOp, arith::RemSIOp, arith::AddIOp,
             arith::SubIOp>(op);
}

static bool canRebuildInIndexDomain(Value v) {
  if (triton::spyre::getConstantInt(v))
    return true;
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (isa<mlir::ktdp::GetComputeTileIdOp, triton::GetProgramIdOp>(op))
    return true;
  if (isValuePreservingIntCast(op))
    return canRebuildInIndexDomain(op->getOperand(0));
  if (!isRebuildableIntArith(op))
    return false;
  if (op->getNumOperands() != 2)
    return false;
  return canRebuildInIndexDomain(op->getOperand(0)) &&
         canRebuildInIndexDomain(op->getOperand(1));
}

static bool hasFixedWidthIntArith(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (isValuePreservingIntCast(op))
    return hasFixedWidthIntArith(op->getOperand(0));
  if (!isArithIntOp(op))
    return false;
  if (!v.getType().isIndex())
    return true;
  if (!isRebuildableIntArith(op) || op->getNumOperands() != 2)
    return false;
  return hasFixedWidthIntArith(op->getOperand(0)) ||
         hasFixedWidthIntArith(op->getOperand(1));
}

static Value emitInIndexDomain(OpBuilder &b, Location loc, Value v) {
  if (auto cst = triton::spyre::getConstantInt(v)) {
    if (v.getType().isIndex())
      return v;
    return arith::ConstantOp::create(b, loc, b.getIndexAttr(*cst)).getResult();
  }

  Operation *op = v.getDefiningOp();

  if (isa<mlir::ktdp::GetComputeTileIdOp, triton::GetProgramIdOp>(op)) {
    if (v.getType().isIndex())
      return v;
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v).getResult();
  }

  if (isValuePreservingIntCast(op))
    return emitInIndexDomain(b, loc, op->getOperand(0));

  Value lhs = emitInIndexDomain(b, loc, op->getOperand(0));
  Value rhs = emitInIndexDomain(b, loc, op->getOperand(1));

  if (v.getType().isIndex() && lhs == op->getOperand(0) &&
      rhs == op->getOperand(1))
    return v;

  if (isa<arith::MulIOp>(op))
    return arith::MulIOp::create(b, loc, lhs, rhs).getResult();
  if (isa<arith::DivSIOp>(op))
    return arith::DivSIOp::create(b, loc, lhs, rhs).getResult();
  if (isa<arith::RemSIOp>(op))
    return arith::RemSIOp::create(b, loc, lhs, rhs).getResult();
  if (isa<arith::AddIOp>(op))
    return arith::AddIOp::create(b, loc, lhs, rhs).getResult();
  return arith::SubIOp::create(b, loc, lhs, rhs).getResult();
}

static Value rebuildInIndexDomain(OpBuilder &b, Location loc, Value v) {
  if (!hasFixedWidthIntArith(v) || !canRebuildInIndexDomain(v))
    return v;
  return emitInIndexDomain(b, loc, v);
}

//===----------------------------------------------------------------------===//
// The coordinate map, read off a marker
//===----------------------------------------------------------------------===//

/// One descriptor's physical layout: a copy of the marker's three arrays, plus
/// the logical rank they index into.
///
/// The marker is an *instruction* — it says how to split logical dims. It is
/// not a source of truth about the tensor: element type, memory space, base
/// offset, strides, coordinate set and dynamic extents all come from the ops
/// being rewritten, and where the two could disagree the op wins.
///
/// A copy, not a view of the marker's attribute storage: a CoordMap is recorded
/// against a physicalized view and read for the whole rewrite, which outlives
/// the marker it came from. Borrowing would make that lifetime a constraint on
/// the order the phases run in, and the ranks here are a handful of dims, so
/// there is nothing to save by it.
struct CoordMap {
  SmallVector<int64_t, 4> src, op, arg;
  unsigned logicalRank = 0;

  unsigned physRank() const { return src.size(); }
  CoordOp opAt(unsigned p) const { return static_cast<CoordOp>(op[p]); }

  /// Does this layout delinearize logical dim `d` into a (stick, lane) pair?
  ///
  /// A splat names `d` too, but it partitions nothing — it replicates `d`
  /// across a fresh axis — so it is not a delinearization and the dim stays
  /// whole. Asking for the floordiv half is therefore the question, not "is some
  /// dim of `d` non-identity".
  bool splits(int64_t d) const {
    return findPhys(d, CoordOp::FloorDiv) >= 0;
  }

  /// Is logical dim `d` named by any physical dim at all? Every dim must be,
  /// or the layout drops it — see readCoordMap.
  bool names(int64_t d) const { return llvm::is_contained(src, d); }

  /// The physical dim carrying `wanted` for logical dim `d`, or -1.
  int findPhys(int64_t d, CoordOp wanted) const {
    for (unsigned p = 0, e = physRank(); p < e; ++p)
      if (src[p] == d && opAt(p) == wanted)
        return p;
    return -1;
  }

  /// The stick width logical dim `d` is delinearized over — the second element
  /// of the basis (ceildiv(extent, W), W). Only meaningful when `splits(d)`;
  /// read off the mod dim, which is where the width is the extent.
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
  case CoordOp::Splat:
    return "splat";
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
///
/// Most of what this checks, SpyreTensorLayoutOp::verify() checks too; the
/// deletion note at the top of RewriteDescriptorLayout.cpp records that the
/// overlap needs an owner when the op becomes a tts attribute.
FailureOr<CoordMap> readCoordMap(triton::SpyreTensorLayoutOp marker,
                                 unsigned logicalRank) {
  CoordMap cm;
  ArrayRef<int64_t> physSrc = marker.getPhysSrc(), physOp = marker.getPhysOp(),
                    physArg = marker.getPhysArg();
  cm.src.assign(physSrc.begin(), physSrc.end());
  cm.op.assign(physOp.begin(), physOp.end());
  cm.arg.assign(physArg.begin(), physArg.end());
  cm.logicalRank = logicalRank;
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
                              "(splat)");
  }
  // A delinearization names the same logical dim twice, once floordiv and once
  // mod, the pair being the multi-index over (ceildiv(extent, W), W). One
  // component without the other would leave the rebuild unable to state where the
  // dim's elements live, so reject it here rather than emitting a map that cannot
  // address them.
  for (unsigned d = 0; d < logicalRank; ++d) {
    // Every logical dim has to be named by some physical dim. A dim named by
    // none loses its extent from the physical type altogether — data loss, and
    // silent, because what is left still verifies. It is also the one way a loop
    // dim could enter the rebuilt domain named by no operand's map, since
    // collectPieces emits a piece for logical position d only when some physical
    // dim has src[p] == d.
    //
    // Nothing upstream rejects it: SpyreTensorLayoutOp::verify() tallies the
    // physical dims per logical dim and would answer this in a line, but it only
    // constrains a dim named TWICE and lets a dim named zero times through.
    //
    // It is also what makes collectPieces' skips safe, and with them the claim
    // that buildLoopDomain needs no catch-all: because every logical dim is
    // sourced, that walk visits every logical position of a marked operand. Both
    // sites say so; keep the three in step.
    //
    // That verifier is arguably the better home even so — this is a property of
    // the marker alone, and the structural tallies are already there. It is here
    // to keep lib/Dialect/Triton/IR/Ops.cpp untouched, since that file is being
    // reverted to upstream when the op moves to the tts dialect; the deletion
    // note at the top of RewriteDescriptorLayout.cpp records the same ownership
    // question for the rest of this function's overlap.
    if (!cm.names(d))
      return marker.emitError("spyre_tensor_layout: logical dim ")
             << d
             << " is named by no phys_src entry, so its extent would be "
                "dropped from the physical layout";
    bool hasFloor = cm.findPhys(d, CoordOp::FloorDiv) >= 0;
    bool hasMod = cm.findPhys(d, CoordOp::Mod) >= 0;
    if (hasFloor != hasMod)
      return marker.emitError("spyre_tensor_layout: logical dim ")
             << d << " has a " << (hasFloor ? "floordiv" : "mod")
             << " physical dim without the matching "
             << (hasFloor ? "mod" : "floordiv") << " half";
    // A splat replicates the dim it names, so that dim must also be present
    // whole for the replication to have something to replicate. It cannot be
    // present as a split: the elements would then live in the floordiv/mod pair
    // and the splat axis would name a third copy of them.
    if (cm.findPhys(d, CoordOp::Splat) >= 0 &&
        cm.findPhys(d, CoordOp::Identity) < 0)
      return marker.emitError("spyre_tensor_layout: logical dim ")
             << d
             << " is splat but has no identity physical dim; a splat "
                "replicates a dim that is also carried whole";
  }
  return cm;
}

/// Recover logical dim `d`'s index from the two components the delinearization
/// gave it: the LINEARIZATION over the basis (ceildiv(extent, width), width).
/// The stick index counts whole sticks of `width` elements and the lane picks one
/// element out of the stick it lands in, so the index is `stick * width + lane` —
/// what `affine.linearize_index` computes over that basis, and the exact inverse
/// of the `affine.delinearize_index` the split states.
///
/// This is the *only* arithmetic either carrier introduces, and both introduce
/// the same one. rewriteAdjacentGenerics names the two components after dims of
/// the rebuilt iteration space; the indirect access tile names them after
/// intermediate variables of its own variable space. Different numbering,
/// identical algebra — so the algebra lives here once and each carrier passes in
/// the exprs it numbers the components with.
inline AffineExpr linearizeStickLane(AffineExpr stick, int64_t width,
                                     AffineExpr lane) {
  return stick * width + lane;
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
// The map rebuild, for rewriteAdjacentGenerics
//===----------------------------------------------------------------------===//

/// One value's place in the rebuild: its logical indexing map, and the layout
/// it is physicalized under. `layout` is null for a value that carries no
/// marker and therefore stays logical.
struct RebuildOperand {
  AffineMap logicalMap;
  const CoordMap *layout = nullptr;
  /// Loop dim assigned to each of this operand's SPLAT physical dims, keyed
  /// by physical dim; -1 for every dim that is not a splat. Filled by
  /// buildLoopDomain, because a splat axis is the one physical dim that no
  /// logical loop dim accounts for — see LoopDomain.
  SmallVector<int> splatDim;
};

/// One piece of the refined iteration space: a logical dim's stick index, its
/// lane (the element offset within a stick) — the two components of its
/// delinearization — or, when nothing delinearizes the dim, the whole dim, which
/// is spelled as the stick component.
///
/// A piece is the unit the numbering orders, because it is the unit an operand's
/// physical dim names: a physical dim carries exactly one of these, and that is
/// what lets an operand's physical order be read as an order on pieces.
///
/// Which is why this is a type and not just an index into the domain. The
/// rebuilt loop dims have to be put in SOME order, and the order is not free:
/// the result operand's rebuilt map has to be a projected permutation taking the
/// loop dims in the order the result's own physical type lays them out, or the
/// generic would be writing its result transposed and the store would need a
/// transpose this pass does not emit. So the ordering is derived from the
/// operands' physical orders (see buildLoopDomain), and a physical dim's
/// contribution to that is exactly one piece — hence the pair `(loop, lane)`,
/// which is what "the half of logical dim `loop`" needs to be said with.
struct DomainPiece {
  unsigned loop;
  /// True for the lane — the element offset within a stick — false for the
  /// stick index (or for an unsplit dim held whole).
  bool lane;

  bool operator==(const DomainPiece &o) const {
    return loop == o.loop && lane == o.lane;
  }
};

/// The rebuilt loop domain — a refinement of the generic's ITERATION SPACE, in
/// linalg's own term: how many loop dims it has, and where each logical dim's
/// pieces landed. No scf.for is involved anywhere in it; see the vocabulary note
/// at the top of the file.
///
/// This is the NUMBERING the whole rebuild agrees on. The logical generic has
/// some number of loop dims; physicalizing its operands delinearizes some of
/// those dims into two — a stick index and a lane within the stick — so the
/// rebuilt generic's iteration space has more dims than the logical one did, and
/// the components only mean anything if every operand names the same rebuilt loop
/// dim for the same component. stickDim[d] and laneDim[d] are that agreement, one
/// entry per LOGICAL dim; width[d] is the basis element an operand holding d
/// whole needs to linearize the two components back into the one index it
/// addresses d with (linearizeStickLane).
///
/// A logical dim is delinearized here if ANY operand delinearizes it, not only
/// if all do — which is why a carrier that does not needs the linearization at
/// all, and why width is part of the domain rather than of the operand that
/// delinearized the dim.
///
/// Splat physical dims are not in this numbering: a splat is an axis no logical
/// loop dim accounts for, so it gets its own loop dim per operand, allocated
/// after the pieces are ordered and recorded in RebuildOperand::splatDim.
struct LoopDomain {
  /// Loop dim carrying logical dim d's stick index, or its whole extent when
  /// the dim is unsplit.
  SmallVector<int> stickDim;
  /// Loop dim carrying logical dim d's lane — its element offset within a
  /// stick; -1 when the dim is unsplit.
  SmallVector<int> laneDim;
  /// The basis element to linearize over, per logical dim — the stick width; 0
  /// when the dim is not delinearized. Taken from the operands, which must agree
  /// — see buildLoopDomain.
  SmallVector<int64_t> width;
  unsigned numLoopDims = 0;

  bool isSplit(int64_t d) const { return laneDim[d] >= 0; }
};

/// The domain pieces one operand's physical dims name, in that operand's own
/// physical order.
///
/// This walk is the whole source of the domain's pieces — buildLoopDomain adds
/// none of its own — so the two dims it skips are what a reader has to be able to
/// rule out. Both skips are safe, and each says why below; readCoordMap's
/// completeness check is what makes the first one safe.
void collectPieces(const RebuildOperand &o,
                   SmallVectorImpl<DomainPiece> &pieces) {
  unsigned numDims =
      o.layout ? o.layout->physRank() : o.logicalMap.getNumResults();
  for (unsigned p = 0; p < numDims; ++p) {
    CoordOp coordOp = o.layout ? o.layout->opAt(p) : CoordOp::Identity;
    if (coordOp == CoordOp::Splat)
      // A splat partitions nothing, so it names no piece — it gets a loop of its
      // own once the pieces are ordered. Nothing is lost by skipping it: a splat
      // dim's logical dim is carried whole by an identity dim too (readCoordMap's
      // splat-companion rule), so that logical position is visited anyway, later
      // in this same walk.
      continue;
    int64_t logDim = o.layout ? o.layout->src[p] : p;
    auto dimExpr = dyn_cast<AffineDimExpr>(o.logicalMap.getResult(logDim));
    if (!dimExpr)
      // A constant (a folded broadcast) names no loop dim, so there is no piece
      // to make. Nothing is lost here either: buildLoopDomain's width scan skips
      // the same result, so the dim is not recorded as split, and any loop dim
      // this operand reaches only through a non-dim expression is named as a bare
      // dim by some other operand — or the generic was non-invertible and never
      // reached this pass. See buildLoopDomain.
      continue;
    pieces.push_back({dimExpr.getPosition(), coordOp == CoordOp::Mod});
  }
}

/// Build the loop domain over `logicalNumLoops` dims of the generic's iteration
/// space, delinearizing every logical dim that any operand delinearizes. Fails
/// when two operands delinearize the same logical dim over different bases.
///
/// Two separate decisions, in this order:
///
/// WHICH dims are delinearized, and over what basis — read off the operands'
/// layouts. Any one operand delinearizing a dim delinearizes it for the whole
/// domain; two operands doing it at different stick widths is the one way this
/// fails, since no single linearization would address the dim.
///
/// WHAT ORDER the resulting pieces are numbered in, which is the part that has a
/// constraint on it. The result operand's map has to take the loop dims in the
/// order its own physical type lays them out — anything else is a transposed
/// write the store cannot absorb — so:
///   1. seed the order from the result's physical order. Nothing below reorders
///      what this places, because the result is the one operand whose coordinate
///      order the generic does not get to choose.
///   2. merge every other operand's physical order in at a cursor: a piece the
///      result already placed just advances the cursor, and a piece it never
///      named is inserted there, so that operand's own dims stay in its order
///      relative to the ones already placed.
/// There is no third step: between them those two name every piece, for the
/// reasons set out at the assertion below.
///
/// Splat physical dims are outside all of this: they name no piece, so each one
/// gets a fresh loop dim appended after the ordering is fixed.
FailureOr<LoopDomain>
buildLoopDomain(MutableArrayRef<RebuildOperand> operands, unsigned resultIdx,
                unsigned logicalNumLoops,
                llvm::function_ref<InFlightDiagnostic()> emitError) {
  LoopDomain dom;
  dom.stickDim.assign(logicalNumLoops, -1);
  dom.laneDim.assign(logicalNumLoops, -1);
  dom.width.assign(logicalNumLoops, 0);

  // Which logical loop dims are split, and at what width.
  for (const RebuildOperand &o : operands) {
    if (!o.layout)
      continue;
    for (unsigned r = 0, e = o.logicalMap.getNumResults(); r < e; ++r) {
      auto dimExpr = dyn_cast<AffineDimExpr>(o.logicalMap.getResult(r));
      if (!dimExpr)
        continue; // a constant (a folded splat) names no loop dim
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
  // already placed only advances the cursor; a piece the result never named is
  // inserted AT the cursor.
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

  // The two merges above have named every piece there is, so there is no
  // catch-all step: `order` is exactly the pieces, and its size is #logical +
  // #split. The argument, in four parts:
  //
  //   - a marked operand sources every logical dim — readCoordMap rejects a
  //     marker that leaves one out — so collectPieces visits every one of its
  //     logical positions and emits a piece for each whose map result is a bare
  //     AffineDimExpr;
  //   - an unmarked operand's walk IS its map results, one per position, so
  //     likewise;
  //   - every loop dim is named by a bare AffineDimExpr in at least one
  //     operand's map, or the generic handed to this pass was already
  //     non-invertible and linalg rejected it before the pass ran. Linalg's
  //     verifier needs each loop dim to appear as a bare dim in some map result
  //     to invert the concatenated map — which is why a lone `(d0, d1) -> (d0 +
  //     d1)` fails it while a convolution's `(d0 + d2, d1 + d3)` passes,
  //     d2 and d3 being named bare by the kernel operand;
  //   - a split dim's two halves name the same logical dim — readCoordMap's
  //     pairing rule — so the stick piece and the lane piece are emitted
  //     together, and dom.width above is set from the same walk under the same
  //     skip, so no dim is marked split whose lane piece is absent.
  //
  // An assertion rather than a fallback, on the same footing as the
  // postconditions below: if this can be violated by input the pass accepts,
  // that is a bug in one of the four legs to fix, not a domain to patch up.
  assert(llvm::all_of(llvm::seq(0u, logicalNumLoops),
                      [&](unsigned d) {
                        return seen({d, /*lane=*/false}) &&
                               (!dom.width[d] || seen({d, /*lane=*/true}));
                      }) &&
         "a loop domain piece no operand's walk named");

  for (auto [n, pc] : llvm::enumerate(order))
    (pc.lane ? dom.laneDim : dom.stickDim)[pc.loop] = n;
  dom.numLoopDims = order.size();

  // One loop per splat physical dim, after the refinement.
  for (RebuildOperand &o : operands) {
    if (!o.layout)
      continue;
    o.splatDim.assign(o.layout->physRank(), -1);
    for (unsigned p = 0, e = o.layout->physRank(); p < e; ++p)
      if (o.layout->opAt(p) == CoordOp::Splat)
        o.splatDim[p] = dom.numLoopDims++;
  }
  return dom;
}

/// Rebuild one operand's indexing map over `dom`.
///
/// Mechanical, once the domain has decided the numbering and the order: walk this
/// operand's physical dims and, per dim, emit the loop dim the domain assigned to
/// the component that dim carries — the stick, the lane, or this operand's own
/// splat loop — and where the operand holds whole a dim the domain delinearized,
/// the linearization of that dim's two components.
AffineMap rebuildMap(const RebuildOperand &o, const LoopDomain &dom,
                     MLIRContext *ctx) {
  auto loopExpr = [&](int loopDim) { return getAffineDimExpr(loopDim, ctx); };

  unsigned numResults =
      o.layout ? o.layout->physRank() : o.logicalMap.getNumResults();

  SmallVector<AffineExpr> results;
  for (unsigned p = 0; p < numResults; ++p) {
    int64_t logDim = o.layout ? o.layout->src[p] : p;
    CoordOp coordOp = o.layout ? o.layout->opAt(p) : CoordOp::Identity;

    AffineExpr logResult = o.logicalMap.getResult(logDim);
    auto dimExpr = dyn_cast<AffineDimExpr>(logResult);
    if (!dimExpr) {
      // A constant survives every physical dim it is named by.
      results.push_back(logResult);
      continue;
    }
    unsigned loop = dimExpr.getPosition();

    switch (coordOp) {
    case CoordOp::FloorDiv:
      results.push_back(loopExpr(dom.stickDim[loop]));
      break;
    case CoordOp::Mod:
      results.push_back(loopExpr(dom.laneDim[loop]));
      break;
    case CoordOp::Splat:
      // The replication axis, named by the loop the domain allocated for it.
      results.push_back(loopExpr(o.splatDim[p]));
      break;
    case CoordOp::Identity:
      // This operand holds the dim whole. If the domain delinearized it, the two
      // components must be linearized back here.
      results.push_back(
          dom.isSplit(loop)
              ? linearizeStickLane(loopExpr(dom.stickDim[loop]),
                                  dom.width[loop], loopExpr(dom.laneDim[loop]))
              : loopExpr(dom.stickDim[loop]));
      break;
    }
  }
  return AffineMap::get(dom.numLoopDims, /*symbolCount=*/0, results, ctx);
}

//===----------------------------------------------------------------------===//
// Postconditions on the rebuilt maps
//
// Two properties of this pass's OWN OUTPUT, so they are assertions rather than
// diagnostics: valid input that violates either is a bug in the rebuild to fix,
// not an input to decline. Neither is therefore reachable from a negative lit
// test, which is the same character verifyAttributesCarried has — a self-check
// on the rewrite, checkable only by running the rewrite on input it accepts, so
// what covers them is the positive cases, all of them at once.
//===----------------------------------------------------------------------===//

/// The loop dims `e` names, left to right — a linearized composite reading as
/// its stick half then its lane half. Constants name none.
void collectNamedLoopDims(AffineExpr e, SmallVectorImpl<unsigned> &out) {
  if (auto dim = dyn_cast<AffineDimExpr>(e)) {
    out.push_back(dim.getPosition());
    return;
  }
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
    collectNamedLoopDims(bin.getLHS(), out);
    collectNamedLoopDims(bin.getRHS(), out);
  }
}

/// (i) Every loop dim of the rebuilt domain is named by at least one operand's
/// rebuilt map.
///
/// Linalg's own verifier would catch a violation — a loop dim named by no map
/// makes the concatenated map non-invertible — but it catches it as `invalid
/// indexing maps are non-invertible` several stages downstream, attributed to
/// nobody. This pass built the domain, so asserting it here attributes it here.
[[maybe_unused]] bool everyLoopDimIsNamed(ArrayRef<AffineMap> maps,
                                          unsigned numLoopDims) {
  llvm::SmallBitVector named(numLoopDims);
  SmallVector<unsigned> dims;
  for (AffineMap m : maps)
    for (AffineExpr r : m.getResults()) {
      dims.clear();
      collectNamedLoopDims(r, dims);
      for (unsigned d : dims)
        named.set(d);
    }
  return named.all();
}

/// (ii) The result operand's map names the loop dims in strictly increasing
/// order, reading each linearized composite left to right — EXCEPT for the
/// splat physical dims, which are excluded.
///
/// The ordering exists for this: the result's coordinate order is the one the
/// generic does not get to choose, so its map has to take the loop dims in the
/// order its own physical type lays them out, and buildLoopDomain seeds the
/// numbering from exactly that.
///
/// The splat dims are excluded because the code does not establish the property
/// for them and is right not to. A splat names no domain piece, so it takes no
/// part in the ordering and is given its loop AFTER the pieces are numbered —
/// the last loop dim, wherever the splat sits in the operand's physical order.
/// A result whose splat is not last therefore has a non-monotone map by
/// construction: rebuild-reduction.mlir case 3 pins `(d0, d1, d2, d3) -> (d3,
/// d1)`, which is correct and which linalg accepts, since a projected
/// permutation need not be monotone. Asserting monotonicity over the splat dims
/// too would be asserting something this pass never promised.
[[maybe_unused]] bool resultMapIsMonotone(const RebuildOperand &result,
                                          AffineMap map) {
  SmallVector<unsigned> dims;
  for (unsigned p = 0, e = map.getNumResults(); p < e; ++p) {
    if (result.layout && result.layout->opAt(p) == CoordOp::Splat)
      continue;
    collectNamedLoopDims(map.getResult(p), dims);
  }
  for (unsigned i = 1; i < dims.size(); ++i)
    if (dims[i - 1] >= dims[i])
      return false;
  return true;
}

/// Rebuild the iterator types over `dom`.
SmallVector<utils::IteratorType>
rebuildIterators(ArrayRef<utils::IteratorType> logicalIterators,
                 const LoopDomain &dom) {
  // Parallel is the default so that a splat loop gets it without being
  // singled out.
  SmallVector<utils::IteratorType> out(dom.numLoopDims,
                                      utils::IteratorType::parallel);
  for (unsigned d = 0, e = logicalIterators.size(); d < e; ++d) {
    out[dom.stickDim[d]] = logicalIterators[d];
    if (dom.isSplit(d))
      out[dom.laneDim[d]] = logicalIterators[d];
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

  /// Physicalized ConstructMemoryViewOp → its coord map. Keyed on the
  /// Operation* of the PHYSICALIZED (new) view op, set in physicalizeDescriptor
  /// after physicalizeMemView returns.
  ///
  /// This is what rewriteAdjacentGenerics iterates: the physicalized views ARE
  /// the scope of the rewrite, so the traversal starts here and reaches generics
  /// through the loads and stores over each view. It is also what the rewrite
  /// looks an operand up in, once it has a generic in hand.
  ///
  /// A MapVector, not a DenseMap, because the rewrite's order comes off this
  /// container and a DenseMap's iteration order is not the order the markers
  /// were physicalized in — the rewrite would be run in an order that varied
  /// with the pointer values.
  llvm::MapVector<Operation *, CoordMap> physViewOf;

  /// Logical construct_memory_view ops superseded by physicalizeDescriptors.
  /// They cannot be erased there: the marker's bridge cast still holds them, and
  /// that cast only dies in eraseMarkers.
  SmallVector<mlir::ktdp::ConstructMemoryViewOp> deadLogicalMemViews;

  //===--------------------------------------------------------------------===//
  // physicalizeDescriptors — one descriptor at a time
  //===--------------------------------------------------------------------===//

  /// Physical sizes for a view, as static extents plus the dynamic values the
  /// kDynamic entries draw from, in order.
  struct PhysicalSizes {
    SmallVector<int64_t> staticSizes;
    SmallVector<Value> dynSizes;
  };

  /// Compute the physical sizes of `memViewOp` under `cm`.
  FailureOr<PhysicalSizes>
  physicalSizes(const CoordMap &cm, mlir::ktdp::ConstructMemoryViewOp memViewOp,
                OpBuilder &b, triton::SpyreTensorLayoutOp marker) {
    ArrayRef<int64_t> logStatic = memViewOp.getStaticSizes();
    ValueRange logDyn = memViewOp.getSizes();
    Location loc = memViewOp.getLoc();

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

  /// Physical strides for a view under `cm`: row-major over the physical sizes.
  ///
  /// The view's own logical strides are not read. A physicalized view addresses
  /// stick-tiled device data, whose element order IS the physical shape's
  /// row-major order — so the logical strides describe a buffer this view no
  /// longer names, and there is nothing in them to derive from.
  ///
  /// Cannot fail: every stride is a function of the physical sizes, which the
  /// caller has already computed.
  PhysicalSizes physicalStrides(const CoordMap &cm,
                                mlir::ktdp::ConstructMemoryViewOp memViewOp,
                                const PhysicalSizes &sizes, OpBuilder &b) {
    Location loc = memViewOp.getLoc();
    unsigned physRank = cm.physRank();
    PhysicalSizes out;

    auto asValue = [&](int64_t c) {
      return arith::ConstantOp::create(b, loc, b.getIndexAttr(c)).getResult();
    };

    bool allStatic = llvm::none_of(sizes.staticSizes, ShapedType::isDynamic);
    if (allStatic) {
      out.staticSizes.assign(physRank, 1);
      for (int p = (int)physRank - 2; p >= 0; --p)
        out.staticSizes[p] = out.staticSizes[p + 1] * sizes.staticSizes[p + 1];
      return out;
    }
    // Any dynamic size makes every outer stride dynamic.
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
      strides[p] =
          arith::MulIOp::create(b, loc, strides[p + 1], sizeValue(p + 1))
              .getResult();
    out.staticSizes.assign(physRank, ShapedType::kDynamic);
    out.dynSizes.assign(strides.begin(), strides.end());
    return out;
  }

  /// Attributes this pass owns on a physicalized op — the ones whose value is a
  /// function of the shape, and so must be recomputed rather than carried.
  ///
  /// Per op, this is also the only place that says which attributes the rewrite
  /// of that op restates.
  static bool isShapeOwnedAttr(StringRef name, Operation *op) {
    if (isa<mlir::ktdp::ConstructMemoryViewOp>(op))
      return name == "static_sizes" || name == "static_strides" ||
             name == "coordinate_set" || name == "operandSegmentSizes";
    if (isa<mlir::ktdp::ConstructAccessTilesOp>(op))
      return name == "base_map" || name == "access_tile_set" ||
             name == "access_tile_order" || name == "operandSegmentSizes";
    if (isa<mlir::ktdp::ConstructIndirectAccessTilesOp>(op))
      return name == "per_dim_subscript_kinds" ||
             name == "per_dim_subscript_maps" ||
             name == "variables_space_set" ||
             name == "variables_space_order" || name == "operandSegmentSizes";
    if (isa<linalg::GenericOp>(op))
      return name == "indexing_maps" || name == "iterator_types" ||
             name == "operandSegmentSizes";
    return false;
  }

  /// Check that physicalizing an op carried every attribute this pass does not
  /// own, unchanged.
  ///
  /// Attributes only — not operands, not the location, not the region. That is
  /// the scope worth checking, because attribute loss is the loss that is
  /// invisible: each physicalize function clones its op and mutates the fields it
  /// owns, so an attribute nobody thought to carry is simply absent and the
  /// result still verifies. A dropped operand, by contrast, fails the op's own
  /// verifier on the spot.
  LogicalResult verifyAttributesCarried(Operation *original,
                                        Operation *rewritten) {
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

  /// Physicalize the memory view behind `marker`.
  FailureOr<Value>
  physicalizeMemView(mlir::ktdp::ConstructMemoryViewOp memViewOp,
                     const CoordMap &cm,
                     triton::SpyreTensorLayoutOp marker) {
    OpBuilder b(memViewOp);
    Location loc = memViewOp.getLoc();

    // The physical view's coordinate set is the dense range of its own sizes,
    // recomputed below. Reject a set that says more than that.
    if (!isDenseRangeSet(memViewOp.getCoordinateSetAttr().getValue(),
                         b.getContext(), memViewOp.getStaticSizes()))
      return memViewOp.emitError(
          "spyre_tensor_layout: coordinate_set must be the dense range of the "
          "view's sizes to physicalize it; a partitioned set would be "
          "overwritten");

    auto sizes = physicalSizes(cm, memViewOp, b, marker);
    if (failed(sizes))
      return failure();
    PhysicalSizes strides = physicalStrides(cm, memViewOp, *sizes, b);

    // Clone rather than build: the offset operand, the memory space, the
    // element type and any attribute this pass has never heard of all come
    // across without being named.
    auto physOp = cast<mlir::ktdp::ConstructMemoryViewOp>(b.clone(*memViewOp));
    physOp.getSizesMutable().assign(sizes->dynSizes);
    physOp.getStridesMutable().assign(strides.dynSizes);
    physOp.setStaticSizes(sizes->staticSizes);
    physOp.setStaticStrides(strides.staticSizes);
    physOp.setCoordinateSetAttr(IntegerSetAttr::get(
        buildRangeSetND(b.getContext(), sizes->staticSizes)));
    physOp.getResult().setType(
        MemRefType::get(sizes->staticSizes,
                        cast<MemRefType>(memViewOp.getResult().getType())
                            .getElementType()));

    if (failed(verifyAttributesCarried(memViewOp, physOp)))
      return failure();
    (void)loc;
    return physOp.getResult();
  }

  /// Physicalize one direct access tile over the already-physical `physMemView`.
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

    if (!tileOp.getAccessTileOrder().isIdentity())
      return tileOp.emitError(
          "spyre_tensor_layout: access_tile_order must be the identity to "
          "physicalize this tile; a permuted order would be overwritten");
    if (!isDenseRangeSet(tileOp.getAccessTileSetAttr().getValue(),
                         b.getContext(), logBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: access_tile_set must be the dense range of the "
          "block shape to physicalize this tile; a non-dense set would be "
          "overwritten");

    SmallVector<Value> physIdx;
    for (unsigned p = 0, e = cm.physRank(); p < e; ++p) {
      if (cm.opAt(p) == CoordOp::Splat) {
        physIdx.push_back(
            arith::ConstantOp::create(b, loc, b.getIndexAttr(0)).getResult());
        continue;
      }
      // The subscript arrives as `index`: construct_access_tile's indices are
      // Variadic<Index>, so the op's own definition guarantees it. And
      // rebuildInIndexDomain either hands the value back unchanged or re-emits
      // its arithmetic in that same domain. So the div/rem below is emitted at
      // the type the subscript already has, and there is no cast to insert.
      Value idx = rebuildInIndexDomain(b, loc, logIdx[cm.src[p]]);
      assert(idx.getType().isIndex() &&
             "an access tile subscript is index by the op's definition");
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
      case CoordOp::Splat:
        llvm_unreachable("splat dims are handled before this switch");
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

    if (failed(verifyAttributesCarried(tileOp, physTile)))
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "    access tile " << tileOp.getResult().getType() << " -> "
               << physTile.getResult().getType() << "\n");

    for (Operation *user :
         llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        ld.getAccessTileMutable().assign(physTile.getResult());
        auto logResTy = cast<RankedTensorType>(ld.getResult().getType());
        ld.getResult().setType(
            RankedTensorType::get(physBlock, logResTy.getElementType()));
      } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        st.getAccessTileMutable().assign(physTile.getResult());
      } else {
        return user->emitError(
            "spyre_tensor_layout: unexpected user of an access tile");
      }
    }

    tileOp.erase();
    return success();
  }

  /// Physicalize one indirect access tile over the already-physical
  /// `physMemView`.
  LogicalResult
  physicalizeIndirectAccessTile(
      mlir::ktdp::ConstructIndirectAccessTilesOp tileOp,
      Value physMemView, const CoordMap &cm) {
    OpBuilder b(tileOp);
    MLIRContext *ctx = b.getContext();

    ArrayRef<int64_t> logBlock = tileOp.getResult().getType().getShape();
    unsigned logRank = logBlock.size();
    unsigned physRank = cm.physRank();

    auto oldKinds = tileOp.getPerDimSubscriptKinds();
    auto oldMaps = tileOp.getPerDimSubscriptMaps();
    unsigned numCaptured = tileOp.getCapturedVariables().size();

    // An indirect subscript's coord is loaded from the index memref, so
    // floordiv/mod cannot apply to it — there is nowhere to write the split.
    for (unsigned p = 0; p < physRank; ++p) {
      if (cm.opAt(p) == CoordOp::Identity)
        continue;
      int64_t logDim = cm.src[p];
      if (!cast<BoolAttr>(oldKinds[logDim]).getValue())
        continue;
      return tileOp.emitError("spyre_tensor_layout: logical dim ")
             << logDim
             << " is an indirect (gather) subscript, so it cannot be "
                "stick-split";
    }

    SmallVector<int64_t> physBlock;
    if (!applyCoordMap(logBlock, cm.src, cm.op, cm.arg, physBlock))
      return tileOp.emitError("spyre_tensor_layout: cannot derive a static "
                              "physical block shape for this indirect access "
                              "tile");

    if (!tileOp.getVariablesSpaceOrder().isIdentity())
      return tileOp.emitError(
          "spyre_tensor_layout: variables_space_order must be the identity to "
          "physicalize this tile; a permuted order would be overwritten");
    if (!isDenseRangeSet(tileOp.getVariablesSpaceSetAttr().getValue(), ctx,
                         logBlock))
      return tileOp.emitError(
          "spyre_tensor_layout: variables_space_set must be the dense range of "
          "the tile shape to physicalize this tile; a non-dense set would be "
          "overwritten");

    auto physVar = [&](unsigned p) {
      return getAffineDimExpr(numCaptured + p, ctx);
    };

    // Recover each logical variable from the physical ones.
    SmallVector<AffineExpr> logicalFromPhysical(logRank);
    SmallVector<int64_t> width(logRank, 0);
    SmallVector<AffineExpr> stickHalf(logRank), laneHalf(logRank);
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
        laneHalf[logDim] = physVar(p);
        width[logDim] = cm.arg[p];
        break;
      case CoordOp::Splat:
        // A splat axis addresses no element of the logical dim, so it
        // contributes nothing to recovering it.
        break;
      }
    }
    for (unsigned d = 0; d < logRank; ++d)
      if (width[d])
        logicalFromPhysical[d] =
            linearizeStickLane(stickHalf[d], width[d], laneHalf[d]);

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
        case CoordOp::Splat:
          e = getAffineConstantExpr(0, ctx);
          break;
        }
        results.push_back(e);
      }

      newKinds.push_back(cast<BoolAttr>(oldKinds[logDim]));
      newMaps.push_back(AffineMapAttr::get(
          AffineMap::get(newNumDims, /*symbolCount=*/0, results, ctx)));
    }

    // Rebuild the indirect memref list in physical dim order.
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

    // Built rather than cloned: the physical tile's variable space has a
    // different number of intermediate variables, which are this op's region's
    // block arguments, so there is no clone-and-retype to do. Everything the
    // builder does not take is therefore carried across by hand — and then held
    // to the same standard as the cloned ops below.
    for (NamedAttribute attr : tileOp->getAttrs())
      if (!isShapeOwnedAttr(attr.getName().strref(), tileOp))
        physTile->setAttr(attr.getName(), attr.getValue());

    if (failed(verifyAttributesCarried(tileOp, physTile)))
      return failure();

    for (Operation *user :
         llvm::make_early_inc_range(tileOp.getResult().getUsers())) {
      if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
        ld.getAccessTileMutable().assign(physTile.getResult());
        auto logResTy = cast<RankedTensorType>(ld.getResult().getType());
        ld.getResult().setType(
            RankedTensorType::get(physBlock, logResTy.getElementType()));
      } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
        st.getAccessTileMutable().assign(physTile.getResult());
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
          "spyre_tensor_layout: desc operand is not a lowered descriptor");
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

    // Record the physicalized view → coord map, for the rewrite's tracing.
    if (Operation *physViewOp = (*physMemView).getDefiningOp())
      physViewOf.try_emplace(physViewOp, *cm);

    for (auto tile : tiles)
      if (failed(physicalizeAccessTile(tile, *physMemView, *cm, marker)))
        return failure();
    for (auto tile : indirectTiles)
      if (failed(physicalizeIndirectAccessTile(tile, *physMemView, *cm)))
        return failure();

    deadLogicalMemViews.push_back(memViewOp);
    return success();
  }

  /// Physicalize the descriptor behind every marker, recording each
  /// physicalized view against the layout it was physicalized under.
  LogicalResult
  physicalizeDescriptors(ArrayRef<triton::SpyreTensorLayoutOp> markers) {
    LLVM_DEBUG(llvm::dbgs()
               << "[rewrite-descriptor-layout-generic] physicalizing "
               << markers.size() << " descriptor(s)\n");
    for (auto marker : markers)
      if (failed(physicalizeDescriptor(marker)))
        return failure();
    return success();
  }

  //===--------------------------------------------------------------------===//
  // rewriteAdjacentGenerics — the one rewrite
  //===--------------------------------------------------------------------===//

  /// Trace a generic's input operand back to the physicalized view it was
  /// loaded from, returning its coord map, or null if not on a physicalized
  /// chain.
  const CoordMap *findLayoutForInput(mlir::Value ins) const {
    auto load = ins.getDefiningOp<mlir::ktdp::LoadOp>();
    if (!load)
      return nullptr;
    Operation *tileOp = load.getAccessTile().getDefiningOp();
    if (!tileOp)
      return nullptr;
    mlir::Value base;
    if (auto direct = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(tileOp))
      base = direct.getBase();
    else if (auto indirect =
                 dyn_cast<mlir::ktdp::ConstructIndirectAccessTilesOp>(tileOp))
      base = indirect.getBase();
    else
      return nullptr;
    auto view = base.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
    if (!view)
      return nullptr;
    auto it = physViewOf.find(view.getOperation());
    return it == physViewOf.end() ? nullptr : &it->second;
  }

  /// Trace a generic's result back to the physicalized view it will be stored
  /// into, returning its coord map, or null if not on a physicalized chain.
  const CoordMap *findLayoutForResult(mlir::Value result) const {
    for (Operation *user : result.getUsers()) {
      auto st = dyn_cast<mlir::ktdp::StoreOp>(user);
      if (!st)
        continue;
      Operation *tileOp = st.getAccessTile().getDefiningOp();
      if (!tileOp)
        continue;
      mlir::Value base;
      if (auto direct = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(tileOp))
        base = direct.getBase();
      else if (auto indirect =
                   dyn_cast<mlir::ktdp::ConstructIndirectAccessTilesOp>(tileOp))
        base = indirect.getBase();
      else
        continue;
      auto view = base.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>();
      if (!view)
        continue;
      auto it = physViewOf.find(view.getOperation());
      if (it != physViewOf.end())
        return &it->second;
    }
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // The rewrite
  //===--------------------------------------------------------------------===//

  LogicalResult rewriteGeneric(linalg::GenericOp op) {
    MLIRContext *ctx = op.getContext();
    unsigned numLoops = op.getNumLoops();

    // Collect layouts for all operands by tracing back to physicalized views.
    SmallVector<const CoordMap *> layouts;
    unsigned numIns = op.getNumDpsInputs();
    unsigned numOps = op->getNumOperands();
    bool anyPhys = false;
    for (unsigned i = 0; i < numOps; ++i) {
      const CoordMap *cm = nullptr;
      if (i < numIns)
        cm = findLayoutForInput(op->getOperand(i));
      else {
        unsigned outIdx = i - numIns;
        if (outIdx < op.getNumResults())
          cm = findLayoutForResult(op.getResult(outIdx));
      }
      layouts.push_back(cm);
      if (cm)
        anyPhys = true;
    }
    if (!anyPhys)
      return success();

    // Check that all consumers of physicalized results are rewritable generics
    // or stores. A non-generic non-store consumer (e.g. tensor.extract_slice)
    // would still name the logical type after the rewrite, silently producing
    // bad IR — diagnose it now with a pass-attributed message.
    for (unsigned i = numIns; i < numOps; ++i) {
      if (!layouts[i])
        continue;
      unsigned outIdx = i - numIns;
      for (Operation *user : op.getResult(outIdx).getUsers()) {
        if (isa<linalg::GenericOp, mlir::ktdp::StoreOp>(user))
          continue;
        return user->emitError(
            "rewrite-descriptor-layout-generic: this op reads a value the "
            "rewrite retyped, but the rewrite restates only linalg.generic, "
            "so this op still names the logical type");
      }
    }

    SmallVector<RebuildOperand> rebuildOperands;
    SmallVector<AffineMap> logicalMaps = op.getIndexingMapsArray();
    for (auto [i, m] : llvm::enumerate(logicalMaps))
      rebuildOperands.push_back(RebuildOperand{m, layouts[i]});

    unsigned resultIdx = op.getDpsInitOperand(0)->getOperandNumber();
    if (resultIdx >= rebuildOperands.size())
      return op.emitError("rewrite-descriptor-layout-generic: the outs operand "
                          "backing result 0 is outside the indexing maps");

    auto dom = buildLoopDomain(rebuildOperands, resultIdx, numLoops,
                               [&]() { return op.emitError(); });
    if (failed(dom))
      return failure();

    SmallVector<AffineMap> physMaps;
    for (const RebuildOperand &o : rebuildOperands)
      physMaps.push_back(rebuildMap(o, *dom, ctx));

    assert(everyLoopDimIsNamed(physMaps, dom->numLoopDims) &&
           "a rebuilt loop dim is named by no operand's map");
    assert(resultMapIsMonotone(rebuildOperands[resultIdx], physMaps[resultIdx]) &&
           "the result's map takes the loop dims out of order");

    SmallVector<utils::IteratorType> physIterators =
        rebuildIterators(op.getIteratorTypesArray(), *dom);

    LLVM_DEBUG({
      llvm::dbgs() << "    rebuilding at " << op.getLoc() << ": " << numLoops
                   << " logical loop dim(s) -> " << dom->numLoopDims << "\n";
      for (unsigned d = 0; d < numLoops; ++d) {
        llvm::dbgs() << "      logical d" << d << " -> ";
        if (dom->isSplit(d))
          llvm::dbgs() << "stick d" << dom->stickDim[d] << " + lane d"
                       << dom->laneDim[d] << " at width " << dom->width[d];
        else
          llvm::dbgs() << "whole d" << dom->stickDim[d];
        llvm::dbgs() << "\n";
      }
      for (auto [i, o] : llvm::enumerate(rebuildOperands)) {
        for (unsigned p = 0, e = o.splatDim.size(); p < e; ++p)
          if (o.splatDim[p] >= 0)
            llvm::dbgs() << "      operand " << i << " splat phys dim " << p
                         << " -> fresh loop d" << o.splatDim[p] << "\n";
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

    OpBuilder b(op);
    for (auto [i, operand] : llvm::enumerate(op->getOpOperands())) {
      if (!layouts[i])
        continue;
      auto ty = dyn_cast<RankedTensorType>(operand.get().getType());
      if (!ty || ty.getRank() == (int64_t)layouts[i]->physRank())
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
    // anything added to linalg.generic later all ride along.
    auto physOp = cast<linalg::GenericOp>(b.clone(*op));
    physOp.setIndexingMapsAttr(b.getAffineMapArrayAttr(physMaps));
    physOp.setIteratorTypesAttr(
        b.getArrayAttr(llvm::to_vector(llvm::map_range(
            physIterators, [&](utils::IteratorType t) -> Attribute {
              return linalg::IteratorTypeAttr::get(ctx, t);
            }))));
    for (auto [res, out] :
         llvm::zip_equal(physOp.getResults(), physOp.getDpsInits()))
      res.setType(out.getType());

    if (failed(verifyAttributesCarried(op, physOp)))
      return failure();

    op.getResults().replaceAllUsesWith(physOp.getResults());
    op.erase();
    return success();
  }

  /// Give `v` the physical type `physTy`, by retyping its producer.
  LogicalResult retypeToPhysical(Value v, RankedTensorType physTy,
                                 OpBuilder &b) {
    Operation *def = v.getDefiningOp();
    if (isa_and_nonnull<linalg::GenericOp>(def))
      return success();
    if (auto empty = dyn_cast_or_null<tensor::EmptyOp>(def)) {
      if (!empty.getType().hasStaticShape())
        return empty.emitError("rewrite-descriptor-layout-generic: cannot "
                               "physicalize a dynamically shaped tensor.empty");
      empty.getResult().setType(physTy);
      return success();
    }
    if (auto cst = dyn_cast_or_null<arith::ConstantOp>(def)) {
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
    // A linalg.fill writes one scalar into every element it is given, so it has
    // no element order to preserve and no shape of its own: restating it is
    // restating the tensor it writes into and letting its result follow. Its
    // indexing maps are implicit in the operand shapes, so nothing has to be
    // rebuilt for them.
    //
    // This is the init LowerComputeOps puts on every reduction's outs, and
    // DropReductionInitFill does not run until the spyrecode stage, so it is
    // present on every reduce this pass sees.
    if (auto fill = dyn_cast_or_null<linalg::FillOp>(def)) {
      if (failed(retypeToPhysical(fill.getDpsInitOperand(0)->get(), physTy, b)))
        return failure();
      fill.getResult(0).setType(physTy);
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

  /// Before physicalizeDescriptors mutates anything, check that every consumer
  /// of a to-be-physicalized load is a linalg.generic or a ktdp.store.
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

  /// The generics adjacent to a physicalized view: for each view
  /// physicalizeDescriptors recorded, every generic that reads one of its loads
  /// or supplies one of its stores. Listed once each, in the order the views
  /// were physicalized.
  ///
  /// This is the whole scope of the rewrite, and starting from the views is what
  /// makes that legible: a generic on no physicalized chain is never looked at,
  /// rather than looked at and found to have no layout on any operand. The
  /// collection is separate from the rewriting below because rewriteGeneric
  /// replaces the op it is given, which would invalidate a user iterator held
  /// across the call.
  void collectAdjacentGenerics(SmallVectorImpl<linalg::GenericOp> &out) {
    SmallPtrSet<Operation *, 8> seen;
    auto note = [&](Operation *op) {
      if (auto g = dyn_cast_or_null<linalg::GenericOp>(op))
        if (seen.insert(g).second)
          out.push_back(g);
    };
    for (const auto &entry : physViewOf) {
      for (Operation *tile : entry.first->getResult(0).getUsers()) {
        if (!isa<mlir::ktdp::ConstructAccessTilesOp,
                 mlir::ktdp::ConstructIndirectAccessTilesOp>(tile))
          continue;
        for (Operation *user : tile->getResult(0).getUsers()) {
          if (auto ld = dyn_cast<mlir::ktdp::LoadOp>(user)) {
            for (Operation *consumer : ld.getResult().getUsers())
              note(consumer);
          } else if (auto st = dyn_cast<mlir::ktdp::StoreOp>(user)) {
            note(st.getDataTile().getDefiningOp());
          }
        }
      }
    }
  }

  /// No ModuleOp parameter: the recorded views are the entry points now, so
  /// nothing here needs the module to walk.
  LogicalResult rewriteAdjacentGenerics() {
    SmallVector<linalg::GenericOp> generics;
    collectAdjacentGenerics(generics);
    LLVM_DEBUG(llvm::dbgs()
               << "[rewrite-descriptor-layout-generic] rewriting "
               << generics.size() << " generic(s) adjacent to "
               << physViewOf.size() << " physicalized view(s)\n");
    for (auto g : generics)
      if (failed(rewriteGeneric(g)))
        return failure();
    return success();
  }

  //===--------------------------------------------------------------------===//
  // eraseMarkers — the cleanup
  //===--------------------------------------------------------------------===//

  void eraseMarker(triton::SpyreTensorLayoutOp marker) {
    if (!marker->getBlock())
      return;
    auto castOp = marker.getDesc().getDefiningOp<UnrealizedConversionCastOp>();
    marker.erase();
    if (castOp && castOp.use_empty())
      castOp.erase();
  }

  /// Erase every marker, and with it the bridge cast that held the superseded
  /// logical view alive.
  void eraseMarkers(ArrayRef<triton::SpyreTensorLayoutOp> markers) {
    LLVM_DEBUG(llvm::dbgs() << "[rewrite-descriptor-layout-generic] erasing "
                            << markers.size() << " marker(s)\n");
    for (auto marker : markers)
      eraseMarker(marker);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<triton::SpyreTensorLayoutOp> markers;
    module.walk([&](triton::SpyreTensorLayoutOp op) { markers.push_back(op); });

    LLVM_DEBUG(llvm::dbgs()
               << "[rewrite-descriptor-layout-generic] " << markers.size()
               << " layout marker(s)\n");

    if (failed(checkConsumersAreRewritable(module)))
      return signalPassFailure();

    if (failed(physicalizeDescriptors(markers)))
      return signalPassFailure();
    if (failed(rewriteAdjacentGenerics()))
      return signalPassFailure();
    eraseMarkers(markers);

    for (auto memViewOp : deadLogicalMemViews)
      if (memViewOp->getBlock() && memViewOp.use_empty())
        memViewOp.erase();
  }
};

} // namespace
