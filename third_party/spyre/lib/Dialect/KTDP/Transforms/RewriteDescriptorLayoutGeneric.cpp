//===- RewriteDescriptorLayoutGeneric.cpp ---------------------------------===//
//
// Rewrites logical tensor descriptors to their physical (stick-tiled) layout,
// driven by tt.spyre_tensor_layout markers, and retypes the compute ops on the
// annotated chain — which are all linalg.generic.
//
// The marker carries the physical layout as the OpSpec `device_coordinates`
// form, three i64 arrays with one entry per physical dim:
//   phys_src[p] : the logical dim physical dim p derives from
//   phys_op[p]  : 0 = identity, 1 = floordiv, 2 = mod
//   phys_arg[p] : divisor (floordiv) / modulus (mod); ignored for identity
// e.g. [M,N] stick-on-N -> phys_src=[1,0,1] phys_op=[1,0,2] phys_arg=[64,0,64]
//   => physical size [ceil(N/64), M, 64].
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
  bool splits(int64_t d) const {
    for (unsigned p = 0, e = physRank(); p < e; ++p)
      if (src[p] == d && opAt(p) != CoordOp::Identity)
        return true;
    return false;
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
    if (cm.op[p] < 0 || cm.op[p] > 2)
      return marker.emitError("spyre_tensor_layout: phys_op must be 0 "
                              "(identity), 1 (floordiv) or 2 (mod)");
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
  }
  return cm;
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
};

/// The rebuilt loop domain: how many physical loop dims there are, and where
/// each logical dim's pieces landed.
///
/// A logical dim split by any operand contributes TWO loop dims — a stick
/// index and an element offset within the stick — and an operand that holds
/// that dim whole addresses it as `stick * width + elem`. A dim no operand
/// splits contributes one.
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

/// Build the loop domain over `logicalNumLoops` dims, splitting every logical
/// dim that any operand splits. Loop dims are numbered in logical order, a
/// split dim taking (stick, elem) adjacently, so the domain is a refinement of
/// the logical one and an unsplit program keeps its original numbering.
///
/// Fails when two operands split the same logical dim at different widths:
/// there is then no single `stick * width + elem` a third operand holding the
/// dim whole could use, and picking either width would silently address the
/// wrong elements.
FailureOr<LoopDomain>
buildLoopDomain(ArrayRef<RebuildOperand> operands, unsigned logicalNumLoops,
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

  for (unsigned d = 0; d < logicalNumLoops; ++d) {
    dom.stickDim[d] = dom.numLoopDims++;
    if (dom.width[d])
      dom.elemDim[d] = dom.numLoopDims++;
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
AffineMap rebuildMap(const RebuildOperand &o, const LoopDomain &dom,
                     MLIRContext *ctx) {
  auto loopExpr = [&](int loopDim) { return getAffineDimExpr(loopDim, ctx); };

  SmallVector<AffineExpr> results;
  // A physicalized operand's map is stated over its PHYSICAL dims, so it is
  // built by walking those; an unphysicalized one keeps its logical results.
  if (o.layout) {
    for (unsigned p = 0, e = o.layout->physRank(); p < e; ++p) {
      int64_t logDim = o.layout->src[p];
      AffineExpr logResult = o.logicalMap.getResult(logDim);
      auto dimExpr = dyn_cast<AffineDimExpr>(logResult);
      if (!dimExpr) {
        // A constant survives every physical dim it is named by: the operand
        // does not vary along this loop dim, whatever the layout does to it.
        results.push_back(logResult);
        continue;
      }
      unsigned loop = dimExpr.getPosition();
      switch (o.layout->opAt(p)) {
      case CoordOp::FloorDiv:
        results.push_back(loopExpr(dom.stickDim[loop]));
        break;
      case CoordOp::Mod:
        results.push_back(loopExpr(dom.elemDim[loop]));
        break;
      case CoordOp::Identity:
        // This operand holds the dim whole. If the domain split it — because
        // another operand does — the two halves have to be recombined here;
        // that composite is the rule's only arithmetic.
        results.push_back(dom.isSplit(loop)
                              ? loopExpr(dom.stickDim[loop]) * dom.width[loop] +
                                    loopExpr(dom.elemDim[loop])
                              : loopExpr(dom.stickDim[loop]));
        break;
      }
    }
  } else {
    for (unsigned r = 0, e = o.logicalMap.getNumResults(); r < e; ++r) {
      AffineExpr logResult = o.logicalMap.getResult(r);
      auto dimExpr = dyn_cast<AffineDimExpr>(logResult);
      if (!dimExpr) {
        results.push_back(logResult);
        continue;
      }
      unsigned loop = dimExpr.getPosition();
      results.push_back(dom.isSplit(loop)
                            ? loopExpr(dom.stickDim[loop]) * dom.width[loop] +
                                  loopExpr(dom.elemDim[loop])
                            : loopExpr(dom.stickDim[loop]));
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
  SmallVector<utils::IteratorType> out(dom.numLoopDims);
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

    // Read the tiles before mutating anything: physicalizing the view does not
    // move them, but erasing one invalidates a walk over the users.
    SmallVector<mlir::ktdp::ConstructAccessTilesOp> tiles;
    for (Operation *user : memView.getUsers())
      if (auto tile = dyn_cast<mlir::ktdp::ConstructAccessTilesOp>(user))
        tiles.push_back(tile);
      else if (isa<mlir::ktdp::ConstructIndirectAccessTilesOp>(user))
        return user->emitError(
            "spyre_tensor_layout: physicalizing an indirect access tile is not "
            "supported by rewrite-descriptor-layout-generic");

    auto physMemView = physicalizeMemView(memViewOp, *cm, marker);
    if (failed(physMemView))
      return failure();

    for (auto tile : tiles)
      if (failed(physicalizeAccessTile(tile, *physMemView, *cm, marker)))
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
    for (Value v : op->getOperands())
      if (!atPhysicalRank(v))
        return false;
    for (Value v : op->getResults())
      if (!atPhysicalRank(v))
        return false;
    // Every map must have one result per dim of the operand it addresses, and
    // all of them must share one loop domain. A rank the operands have moved
    // past is exactly the state where an operand was retyped and this op has
    // not caught up.
    for (auto [map, operand] :
         llvm::zip_equal(op.getIndexingMapsArray(), op->getOperands())) {
      auto ty = dyn_cast<RankedTensorType>(operand.getType());
      if (ty && map.getNumResults() != (unsigned)ty.getRank())
        return false;
    }
    return true;
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

    auto dom = buildLoopDomain(rebuildOperands, numLoops,
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
      for (auto st : stores)
        if (failed(rewriteStore(st, changed)))
          return failure();
      for (auto g : generics) {
        if (isConsistent(g))
          continue;
        if (failed(rewriteGeneric(g)))
          return failure();
        changed = true;
      }
      if (!changed)
        return checkAllConsistent(module);
    }
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

  LogicalResult checkAllConsistent(ModuleOp module) {
    LogicalResult result = success();
    module.walk([&](Operation *op) {
      if (auto g = dyn_cast<linalg::GenericOp>(op)) {
        if (!isConsistent(g)) {
          g.emitError("rewrite-descriptor-layout-generic: this op is on a "
                      "physicalized chain but could not be restated at "
                      "physical shape");
          result = failure();
        }
        return;
      }
      // Any other op reading or producing a physicalized value is outside what
      // one generic rewrite covers, and saying so beats emitting IR that only
      // fails later in the pipeline.
      for (Value v : op->getOperands())
        if (layoutFor(v) && !atPhysicalRank(v)) {
          op->emitError("rewrite-descriptor-layout-generic: this op consumes a "
                        "value on a physicalized chain but is not a "
                        "linalg.generic, so the rewrite cannot restate it");
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

    if (failed(checkConsumersAreRewritable(module)))
      return signalPassFailure();

    for (auto marker : markers)
      if (failed(physicalizeDescriptor(marker)))
        return signalPassFailure();

    if (failed(runRewrite(module)))
      return signalPassFailure();

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
