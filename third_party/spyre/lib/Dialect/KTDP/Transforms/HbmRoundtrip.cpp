//===- HbmRoundtrip.cpp - Break compute-to-compute edges through HBM ------===//
//
// Routes every compute-to-compute tensor value through HBM: a `ktdp.store` after
// the producer, a `ktdp.load` before each consumer, into a fresh spill buffer per
// edge that becomes a new `index` argument. The full contract -- scope, buffer
// geometry, what is reported on the module -- is in Passes.td; this file does not
// restate it.
//
// A temporary stand-in for a real spill/schedule decision.
//
// Two things a reader is likely to undo, so they are stated here as well:
//
//   - **CSE must not run after this pass.** The copies it gives each group are
//     Pure, so CSE merges them straight back. Any implementation would be equally
//     exposed -- the scheduler wants N structurally identical access tiles, and
//     merging structurally identical Pure ops is CSE's job. What contains it is
//     position: in `_make_spyrecode` (third_party/spyre/backend/compiler.py) this
//     pass is installed last on its branch, and that stage's only
//     canonicalize/CSE is on the other arm of the same conditional.
//
//   - **Scope is checked only after an edge is found** (:func:`roundtrip`), so a
//     function with nothing to route is left untouched whatever it contains. A
//     single-compute kernel whose one named op is out of scope compiles today and
//     must keep compiling; checking scope first breaks it.
//
// Tracing: `--debug-only=hbm-roundtrip`.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Dialect/KTDP/Transforms/Utility.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hbm-roundtrip"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_HBMROUNDTRIP
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// One compute-to-compute edge and the buffer that breaks it, 1:1.
struct Spill {
  Value value;
  /// The `linalg` ops reading `value`, in block order, deduplicated: one load per
  /// consumer op however many operands it reads `value` through.
  SmallVector<Operation *> consumers;
  /// The buffer's shape: the tile shape with dim 0 scaled by the grid.
  SmallVector<int64_t> viewShape;
  BlockArgument baseArg;
  Value view;
};

static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

struct HbmRoundtripPass
    : public mlir::triton::ktdp::impl::HbmRoundtripBase<HbmRoundtripPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<Attribute> reported;
    func::FuncOp reportedFor;
    bool rewrote = false;

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (!funcOp.isPublic() || funcOp.getBody().empty())
        continue;
      SmallVector<Attribute> buffers;
      FailureOr<bool> touched = roundtrip(funcOp, buffers);
      if (failed(touched))
        return signalPassFailure();
      rewrote |= *touched;
      if (buffers.empty())
        continue;
      // One list per module, consumed positionally. Concatenating two functions'
      // buffers would hand the second function's addresses to the first.
      if (reportedFor) {
        funcOp.emitError()
            << "HbmRoundtrip: both @" << reportedFor.getName() << " and @"
            << funcOp.getName()
            << " are public functions needing spill buffers, and the buffer "
               "list is reported per module; compile one entry function at a "
               "time";
        return signalPassFailure();
      }
      reportedFor = funcOp;
      reported = std::move(buffers);
    }

    if (!rewrote)
      return;
    if (!reported.empty()) {
      auto buffers = ArrayAttr::get(&getContext(), reported);
      module->setAttr("ktdp.hbm_roundtrip_buffers", buffers);
      LLVM_DEBUG(llvm::dbgs() << "hbm-roundtrip: reported " << buffers << "\n");
    }
    // Cloning left every original behind, unused. Swept because an operation
    // belonging to no compute group is one more thing for the scheduler to place.
    mlir::triton::ktdp::cleanupDeadOps(module);
  }

private:
  //===--------------------------------------------------------------------===//
  // Step 1: find the compute-to-compute edges, and check scope
  //===--------------------------------------------------------------------===//

  /// Whether `op` is a compute, i.e. becomes a compute group of its own. A
  /// `linalg.fill` is not: it is materialized into whichever group reads it, so
  /// routing one through HBM would put a store and a load around something that
  /// is not a schedule boundary.
  static bool isCompute(Operation *op) {
    return isa<linalg::LinalgOp>(op) && !isa<linalg::FillOp>(op);
  }

  /// The first `linalg` op outside this pass's op set, or null when there is none.
  ///
  /// The set is `generic`, `fill` and `reduce`, following the scheduler's own
  /// named-op allowlist for compute,
  /// `linalg.{add,mul,sub,max,min,reduce,generic,yield}`: `reduce` is in it, so a
  /// reduce is a compute like any other. `broadcast`, `transpose` and `matmul` are
  /// not in it, and so stay out of scope -- `broadcast` being the one that occurs,
  /// since LowerComputeOps emits it for `tt.broadcast` / `tt.expand_dims`.
  static Operation *firstOpOutOfScope(Block &entry) {
    for (Operation &op : entry)
      if (isa<linalg::LinalgOp>(&op) &&
          !isa<linalg::GenericOp, linalg::FillOp, linalg::ReduceOp>(&op))
        return &op;
    return nullptr;
  }

  static SmallVector<Spill> collectSpills(Block &entry) {
    SmallVector<Spill> spills;
    DenseMap<Value, unsigned> indexOf;

    for (Operation &op : entry) {
      if (!isCompute(&op))
        continue;
      // Every operand, not just `ins`: an `outs` fed by a compute is the same
      // unschedulable edge. In practice UnaliasLinalgOuts leaves a tensor.empty
      // there, so this finds `ins`.
      for (Value operand : op.getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer || producer->getBlock() != &entry || !isCompute(producer))
          continue;
        auto it = indexOf.find(operand);
        if (it == indexOf.end()) {
          indexOf[operand] = spills.size();
          spills.push_back(Spill{operand, {&op}, {}, nullptr, nullptr});
          continue;
        }
        auto &consumers = spills[it->second].consumers;
        if (!llvm::is_contained(consumers, &op))
          consumers.push_back(&op);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  " << spills.size() << " edge(s)\n";
      for (const Spill &spill : spills)
        llvm::dbgs() << "    " << spill.value.getType() << " to "
                     << spill.consumers.size() << " consumer(s)\n";
    });
    return spills;
  }

  /// Report a function that has an edge to route but cannot be handled. A
  /// diagnostic rather than a skip: such a kernel does not survive scheduling
  /// either, and this is the last point at which the limitation can be named.
  static LogicalResult checkInScope(func::FuncOp funcOp, Block &entry) {
    if (Operation *offender = firstOpOutOfScope(entry)) {
      LLVM_DEBUG(llvm::dbgs() << "  out of scope: " << offender->getName()
                              << "\n");
      return outOfScope(funcOp) << "it contains " << offender->getName();
    }
    auto grid = funcOp->getAttrOfType<ArrayAttr>("grid");
    if (grid && grid.size() > 1) {
      LLVM_DEBUG(llvm::dbgs() << "  out of scope: grid rank " << grid.size()
                              << "\n");
      return outOfScope(funcOp) << "its grid has rank " << grid.size();
    }
    return success();
  }

  /// The shared lead-in of both scope diagnostics: what the kernel needs, and
  /// what this pass handles. The caller appends the reason it cannot.
  static InFlightDiagnostic outOfScope(func::FuncOp funcOp) {
    return funcOp.emitError()
           << "HbmRoundtrip: this kernel hands a value from one compute to "
              "another, which has to be routed through HBM, but this pass is a "
              "temporary stand-in that handles linalg.generic, linalg.fill and "
              "linalg.reduce on a rank-1 grid only: ";
  }

  //===--------------------------------------------------------------------===//
  // Step 2: the compute-tile id
  //===--------------------------------------------------------------------===//

  /// The extent of the rank-1 `grid` attribute, or 1 when there is none. Higher
  /// rank never reaches here -- :func:`checkInScope` reported it.
  static int64_t gridSize(func::FuncOp funcOp) {
    auto grid = funcOp->getAttrOfType<ArrayAttr>("grid");
    if (!grid || grid.empty())
      return 1;
    return cast<IntegerAttr>(grid.getValue()[0]).getInt();
  }

  /// This tile's index in the grid, or null when the grid holds one tile and
  /// every slab offset is therefore zero.
  static FailureOr<Value> tileId(func::FuncOp funcOp, Block &entry,
                                 int64_t gridTotal) {
    if (gridTotal == 1)
      return Value();

    for (Operation &op : entry)
      if (auto tileIdOp = dyn_cast<mlir::ktdp::GetComputeTileIdOp>(&op))
        return tileIdOp->getResult(0);

    funcOp.emitError()
        << "HbmRoundtrip: grid has " << gridTotal
        << " compute tiles but the function never locates itself in it "
           "(no ktdp.get_compute_tile_id), so a per-tile spill slab cannot "
           "be addressed";
    return failure();
  }

  //===--------------------------------------------------------------------===//
  // Step 3: make each compute group self-contained
  //===--------------------------------------------------------------------===//

  /// Clone `value`'s defining op and, recursively, everything it reads, in front
  /// of the insertion point. Stops at block arguments: the base addresses, which
  /// every group legitimately shares. `cloned` memoizes within one cone.
  static Value cloneCone(OpBuilder &builder, Value value,
                         DenseMap<Value, Value> &cloned) {
    if (isa<BlockArgument>(value))
      return value;
    auto known = cloned.find(value);
    if (known != cloned.end())
      return known->second;

    Operation *def = value.getDefiningOp();
    IRMapping mapping;
    for (Value operand : def->getOperands())
      mapping.map(operand, cloneCone(builder, operand, cloned));
    Operation *clone = builder.clone(*def, mapping);
    for (auto [original, copy] :
         llvm::zip(def->getResults(), clone->getResults()))
      cloned[original] = copy;
    return cloned[value];
  }

  /// Give every compute group its own copy of everything it reads, so that no two
  /// groups share an operation. Each group is extracted into a schedule of its
  /// own, so an operation two groups read belongs to neither.
  ///
  /// Duplication is unconditional rather than conditional on being shared,
  /// because position matters too: groups are spans of the block, so a producer
  /// sitting inside the first group's span belongs to it even when only the
  /// second group reads it. Being over-generous costs only dead IR, which the
  /// sweep afterwards removes.
  ///
  /// Two halves, in order -- tensor operands, then the address operands of every
  /// memory operation including the ones the first half just cloned.
  static LogicalResult makeEachGroupSelfContained(Block &entry) {
    // 1. Tensor operands. A `linalg.fill` on `outs` is Pure, so the KTIR stage's
    //    closing CSE merges two identical fills into one; a `ktdp.load` of an
    //    input is one op with several readers because that is how the kernel was
    //    written. A `tensor.empty` needs nothing: UnaliasLinalgOuts makes one per
    //    `linalg` op, in position, after that CSE.
    int64_t clones = 0;
    for (Operation &op : llvm::make_early_inc_range(entry)) {
      // Computes only. A fill is excluded by the same predicate and privatized as
      // part of its reader's cone; visiting it in its own right would clone its
      // `tensor.empty` in front of the *fill*, a position that may belong to an
      // earlier group.
      if (!isCompute(&op))
        continue;
      for (OpOperand &operand : op.getOpOperands()) {
        Value value = operand.get();
        if (!isa<RankedTensorType>(value.getType()))
          continue;
        Operation *producer = value.getDefiningOp();
        if (!producer || producer->getBlock() != &entry ||
            isa<tensor::EmptyOp>(producer))
          continue;

        OpBuilder builder(&op);
        if (isa<linalg::FillOp>(producer)) {
          // The whole cone: a fill's `tensor.empty` and scalar are not address
          // operands, so loop 2 will not reach them.
          DenseMap<Value, Value> cloned;
          operand.set(cloneCone(builder, value, cloned));
        } else if (isa<mlir::ktdp::LoadOp>(producer)) {
          // The op alone is enough -- loop 2 rebuilds every memory operation's
          // address cone, this clone included.
          operand.set(builder.clone(*producer)->getResult(0));
        } else {
          // Every edge is a `ktdp.load` by now and the block holds no other
          // `linalg` op, so this is a tensor producer the pass cannot place -- a
          // reshape, say. Reported rather than cloned blindly: getting it wrong
          // surfaces much later, as a failure naming none of the caller's IR.
          return op.emitError()
                 << "HbmRoundtrip: " << producer->getName()
                 << " produces a tensor a compute reads, and this pass can only "
                    "place a tensor.empty, a linalg.fill or a ktdp.load into the "
                    "compute group that reads it";
        }
        ++clones;
      }
    }

    // 2. Address operands. Index arithmetic is not exempt from the rule above: a
    //    descriptor offset computed once and read by two access tiles is shared
    //    just as a tensor would be. This is what the hand-written reference
    //    chains do, re-reading the tile id and re-building their views per
    //    compute.
    SmallVector<Operation *> memoryOps;
    for (Operation &op : entry)
      if (isa<mlir::ktdp::LoadOp, mlir::ktdp::StoreOp>(&op))
        memoryOps.push_back(&op);

    for (Operation *memoryOp : memoryOps) {
      // The access tile only. A store's other operand is the tensor the compute
      // produced, which is the one thing that must NOT be duplicated.
      auto accessTile = isa<mlir::ktdp::LoadOp>(memoryOp)
                            ? cast<mlir::ktdp::LoadOp>(memoryOp).getAccessTile()
                            : cast<mlir::ktdp::StoreOp>(memoryOp).getAccessTile();
      OpBuilder builder(memoryOp);
      DenseMap<Value, Value> cloned;
      memoryOp->replaceUsesOfWith(accessTile,
                                  cloneCone(builder, accessTile, cloned));
    }

    LLVM_DEBUG(llvm::dbgs() << "  cloned " << clones << " tensor operand(s), "
                            << memoryOps.size() << " address cone(s)\n");
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Step 4: rewrite
  //===--------------------------------------------------------------------===//

  /// Returns whether the function was rewritten, so the caller knows whether the
  /// dead-op sweep has anything to do.
  FailureOr<bool> roundtrip(func::FuncOp funcOp,
                            SmallVectorImpl<Attribute> &reported) {
    Block &entry = funcOp.getBody().front();
    LLVM_DEBUG(llvm::dbgs() << "hbm-roundtrip: @" << funcOp.getName() << "\n");

    // Edges first, scope second -- see the header. No edge means no work, and a
    // function with no work is left byte-for-byte as it arrived.
    SmallVector<Spill> spills = collectSpills(entry);
    if (spills.empty())
      return false;
    if (failed(checkInScope(funcOp, entry)))
      return failure();

    int64_t gridTotal = gridSize(funcOp);
    for (Spill &spill : spills) {
      // A rank-0 spill has no dimension to slab per compute tile, and nothing in
      // this pipeline produces one (every reduce leaves at least a stick).
      auto tileType = dyn_cast<RankedTensorType>(spill.value.getType());
      if (!tileType || tileType.getRank() == 0 || !tileType.hasStaticShape()) {
        return spill.value.getDefiningOp()->emitError()
               << "HbmRoundtrip: cannot spill a value of type "
               << spill.value.getType()
               << " to HBM; a spill buffer needs a statically shaped tile of "
                  "rank >= 1";
      }
      spill.viewShape.assign(tileType.getShape().begin(),
                             tileType.getShape().end());
      spill.viewShape[0] *= gridTotal;
    }

    FailureOr<Value> gridId = tileId(funcOp, entry, gridTotal);
    if (failed(gridId))
      return failure();

    // Arguments first, then the views over them: adding an argument invalidates
    // nothing already built, whereas a view over an argument that does not exist
    // yet is impossible.
    Location loc = funcOp.getLoc();
    auto indexType = IndexType::get(funcOp.getContext());
    for (Spill &spill : spills) {
      spill.baseArg = entry.addArgument(indexType, loc);
      LLVM_DEBUG({
        llvm::dbgs() << "  buffer as argument " << spill.baseArg.getArgNumber()
                     << ", view shape ";
        llvm::interleave(spill.viewShape, llvm::dbgs(), "x");
        llvm::dbgs() << "x" << elementType(spill) << "\n";
      });
    }
    funcOp.setType(FunctionType::get(funcOp.getContext(),
                                     entry.getArgumentTypes(),
                                     funcOp.getFunctionType().getResults()));

    auto memorySpace = mlir::ktdp::MemorySpaceAttr::get(
        funcOp.getContext(), mlir::ktdp::MemorySpaceKind::global,
        /*ct_id=*/-1);
    OpBuilder builder(&entry, entry.begin());
    for (Spill &spill : spills)
      spill.view = mlir::triton::ktdp::buildMemoryView(
          builder, loc, spill.baseArg, spill.viewShape,
          rowMajorStrides(spill.viewShape), /*dynSizes=*/{},
          /*dynStrides=*/{}, elementType(spill), memorySpace);

    for (const Spill &spill : spills) {
      ArrayRef<int64_t> tileShape =
          cast<RankedTensorType>(spill.value.getType()).getShape();

      Operation *producer = spill.value.getDefiningOp();
      OpBuilder storeBuilder(producer);
      storeBuilder.setInsertionPointAfter(producer);
      Value storeTile =
          buildSlabTile(storeBuilder, spill.value.getLoc(), spill, tileShape,
                        *gridId, gridTotal);
      mlir::ktdp::StoreOp::create(storeBuilder, spill.value.getLoc(),
                                  spill.value, storeTile);

      for (Operation *consumer : spill.consumers) {
        OpBuilder loadBuilder(consumer);
        Value loadTile =
            buildSlabTile(loadBuilder, spill.value.getLoc(), spill, tileShape,
                          *gridId, gridTotal);
        Value loaded = mlir::ktdp::LoadOp::create(
            loadBuilder, spill.value.getLoc(), spill.value.getType(), loadTile);
        consumer->replaceUsesOfWith(spill.value, loaded);
      }
    }

    if (failed(makeEachGroupSelfContained(entry)))
      return failure();

    for (const Spill &spill : spills)
      reported.push_back(describe(funcOp.getContext(), spill));
    return true;
  }

  static Type elementType(const Spill &spill) {
    return cast<RankedTensorType>(spill.value.getType()).getElementType();
  }

  /// A fresh `ktdp.construct_access_tile` over this tile's slab of the buffer,
  /// anchored at `tileId * tileShape[0]` along dimension 0.
  static Value buildSlabTile(OpBuilder &builder, Location loc,
                             const Spill &spill, ArrayRef<int64_t> tileShape,
                             Value tileId, int64_t gridTotal) {
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    SmallVector<Value> indices(tileShape.size(), zero);
    if (gridTotal != 1) {
      Value slab = arith::ConstantIndexOp::create(builder, loc, tileShape[0]);
      indices[0] = arith::MulIOp::create(builder, loc, tileId, slab);
    }
    return mlir::triton::ktdp::buildAccessTile(builder, loc, spill.view,
                                               tileShape, indices);
  }

  /// One buffer as the launcher needs it. No element *width*: the backend already
  /// derives one from an element type's spelling for the kernel's own pointers
  /// (`_elem_bytes` in backend/compiler.py), and a second answer to the same
  /// question is a second thing to keep in step.
  static Attribute describe(MLIRContext *ctx, const Spill &spill) {
    Builder builder(ctx);
    return builder.getDictionaryAttr(
        {builder.getNamedAttr("shape",
                              builder.getDenseI64ArrayAttr(spill.viewShape)),
         builder.getNamedAttr("element_type",
                              TypeAttr::get(elementType(spill)))});
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createHbmRoundtripPass() {
  return std::make_unique<HbmRoundtripPass>();
}
} // namespace mlir::triton::ktdp
