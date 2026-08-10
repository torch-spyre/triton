//===- DistributeWork.cpp - Distribute work across Spyre cores -----------===//
//
// Replaces tt.get_program_id with ktdp.get_compute_tile_id, folds
// tt.get_num_programs to an arith.constant grid[axis], and stamps a
// grid attribute on the enclosing function. The pass trusts the kernel
// to distribute work itself (explicit per-core loop around the body);
// it does not synthesize a wrapping scf.for.
//
// Multi-axis programs are handled in one shot: ktdp.get_compute_tile_id
// is variadic and returns one index per grid dimension. All pid ops
// inside the same function share a single emitted op, each taking the
// result that matches its axis. tt.get_num_programs is folded against
// the same grid, so both ops share a single source of truth.
//
// Limitation: one grid per pass run. Every function in the module must
// share the same grid rank / shape (the pass takes a single
// ArrayRef<int64_t> grid option). This is fine for today's
// one-kernel-per-module compilation model but needs a per-function
// spyre.grid attribute before mixed-rank modules can be compiled.
//
// Before:                              After:
//   %px = tt.get_program_id x : i32      %px_i, %py_i = ktdp.get_compute_tile_id
//   %py = tt.get_program_id y : i32              : index, index
//                                        %px = arith.index_cast %px_i : i32
//                                        %py = arith.index_cast %py_i : i32
//   ... body ...                         ... body (unchanged) ...
//   return                               return
//
// The grid attribute lands on the enclosing op implementing
// FunctionOpInterface (tt.func before ConvertFunctions, func.func
// after). One op, one attribute, no dependency on pass ordering.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_DISTRIBUTEWORK
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

struct DistributeWorkPass
    : public mlir::triton::ktdp::impl::DistributeWorkBase<
          DistributeWorkPass> {

  using DistributeWorkBase::DistributeWorkBase;

  DistributeWorkPass(ArrayRef<int64_t> gridShape) {
    // ListOption inherits from llvm::cl::list which accepts ArrayRef
    // via operator=.
    grid = gridShape;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Default grid = [32] when nothing was passed (the common 1D-on-
    // full-hardware case). ListOption doesn't support a default in
    // tablegen, so we apply it here.
    if (grid.empty())
      grid = ArrayRef<int64_t>({32});

    // One walk over functions. Uses FunctionOpInterface so the pass
    // works on both tt.func (pre-ConvertFunctions) and func.func
    // (post) without a hidden pass-order dependency. Each function is
    // processed once: gather its pid ops, then either distribute (if
    // any) or stamp the single-program grid (if none).
    module.walk([&](FunctionOpInterface fn) {
      SmallVector<triton::GetProgramIdOp> pids;
      SmallVector<triton::GetNumProgramsOp> nprogs;
      fn.walk([&](Operation *op) {
        if (auto pid = dyn_cast<triton::GetProgramIdOp>(op))
          pids.push_back(pid);
        else if (auto np = dyn_cast<triton::GetNumProgramsOp>(op))
          nprogs.push_back(np);
      });

      if (pids.empty() && nprogs.empty()) {
        // Single-program kernel: no tl.program_id / tl.num_programs
        // reads. Stamp grid = [1] so downstream consumers don't have
        // to distinguish "no grid attr yet" from "grid explicitly
        // single-program".
        OpBuilder builder(fn.getContext());
        fn->setAttr("grid", builder.getI64ArrayAttr({1}));
        return;
      }
      distributeInFunction(fn.getOperation(), pids, nprogs);
    });
  }

private:
  // Rewrite every pid op in `pids` to read from a single shared
  // ktdp.get_compute_tile_id, cast to i32 for downstream consumers, and
  // stamp grid on `fnOp`.
  //
  // High-level flow
  // ---------------
  // Triton's `tl.program_id(axis)` lowers to one `tt.get_program_id`
  // op per source reference. A kernel that reads `tl.program_id(0)`
  // twice and `tl.program_id(1)` once ends up with **three** pid ops
  // in the IR (two on axis 0, one on axis 1) — they are not CSE'd.
  //
  // KTDP's side of the contract is different: `ktdp.get_compute_tile_id`
  // is variadic and returns a tuple of index values — one per grid
  // dimension — from a single op invocation. The natural lowering is
  // therefore:
  //
  //   1. Emit ONE `ktdp.get_compute_tile_id` whose result arity matches
  //      the kernel's grid dimensionality (= max axis read + 1). This
  //      is the "unwrap the tuple once" step.
  //
  //   2. Cast each index result to i32 — the type every downstream use
  //      of a pid expects — giving us one shared i32 value per axis.
  //
  //   3. For every pid op we collected: rewire all of its uses to the
  //      i32 shared value for its axis, then erase the pid. This
  //      collapses N pid ops down to 1 ktdp op + axis-many casts, no
  //      matter how many times or on how many axes the source code
  //      read `tl.program_id`.
  //
  // Example (2D kernel reading axis 0 twice, axis 1 once):
  //
  //   Before:
  //     %a = tt.get_program_id 0 : i32      // use U1
  //     %b = tt.get_program_id 0 : i32      // use U2
  //     %c = tt.get_program_id 1 : i32      // use U3
  //
  //   After:
  //     %p0_i, %p1_i = ktdp.get_compute_tile_id : index, index
  //     %p0 = arith.index_cast %p0_i : index to i32     // shared for axis 0
  //     %p1 = arith.index_cast %p1_i : index to i32     // shared for axis 1
  //     // U1 and U2 now use %p0; U3 now uses %p1.
  //
  // Finally, stamp a `grid` attribute on the enclosing function so
  // downstream passes / the runtime can see the partition shape.
  void distributeInFunction(Operation *fnOp,
                            ArrayRef<triton::GetProgramIdOp> pids,
                            ArrayRef<triton::GetNumProgramsOp> nprogs) {
    // --- Step 0: validate the kernel's grid contract -----------------
    // Triton's tt.get_program_id.axis and tt.get_num_programs.axis are
    // both i32 attributes in {0, 1, 2}. We enforce three invariants on
    // what the kernel reads, so silent mismatches become hard errors:
    //
    //   (a) nprogs without pids is rejected. A kernel that asks "how
    //       many cores?" but never "which one am I?" is almost
    //       certainly a bug — there is no per-core branch to act on
    //       the answer. Rejecting it catches the typo of reading
    //       num_programs when program_id was meant.
    //
    //   (b) The set of axes read must be {0, 1, ..., numDims-1} —
    //       dense from 0, no holes. A kernel that reads axis 0 and
    //       axis 2 but skips axis 1 would otherwise be accepted with
    //       grid.size() == 3 and the middle axis silently ignored.
    //
    //   (c) grid.size() == numDims. Caller's grid rank must match
    //       what the kernel actually reads. Any guess we make for a
    //       mismatched rank is wrong for some reachable kernel shape.
    llvm::SmallBitVector axesRead(/*size=*/4);
    for (auto pid : pids)
      axesRead.set(static_cast<unsigned>(pid.getAxisAsInt()));
    for (auto np : nprogs)
      axesRead.set(static_cast<unsigned>(np.getAxisAsInt()));
    unsigned numDims = axesRead.find_last() + 1;

    // (a) nprogs-only kernels are rejected.
    if (pids.empty() && !nprogs.empty()) {
      fnOp->emitError()
          << "DistributeWork: function reads tt.get_num_programs but never "
             "tt.get_program_id; a kernel that asks for the grid size "
             "without locating itself in the grid is almost certainly a "
             "bug";
      return signalPassFailure();
    }

    // (b) axes must be dense from 0.
    for (unsigned i = 0; i < numDims; ++i) {
      if (!axesRead.test(i)) {
        fnOp->emitError()
            << "DistributeWork: function reads grid axes non-densely "
               "(axis " << i << " is skipped; highest axis read is "
            << (numDims - 1) << ")";
        return signalPassFailure();
      }
    }

    // (c) caller's grid rank must match.
    if (grid.size() != numDims) {
      fnOp->emitError()
          << "DistributeWork: grid rank " << grid.size()
          << " does not match kernel's pid dimensionality " << numDims
          << " (function reads tt.get_program_id / tt.get_num_programs on "
             "axes 0.." << (numDims - 1) << ")";
      return signalPassFailure();
    }

    // --- Step 1: replace tt.get_program_id with ktdp.get_compute_tile_id
    // Invariant (a) guarantees pids is non-empty if we got here
    // (nprogs-only was rejected in step 0), so no guard is needed.
    //
    // Before (2D kernel, axis 0 read twice, axis 1 read once):
    //   %a = tt.get_program_id 0 : i32      // use U1
    //   %b = tt.get_program_id 0 : i32      // use U2
    //   %c = tt.get_program_id 1 : i32      // use U3
    //
    // After:
    //   %p0_i, %p1_i = ktdp.get_compute_tile_id : index, index
    //   %p0 = arith.index_cast %p0_i : index to i32    // shared axis 0
    //   %p1 = arith.index_cast %p1_i : index to i32    // shared axis 1
    //   // U1, U2 now use %p0; U3 now uses %p1.

    // 1a. Find the insertion point: the ktdp op + its casts must
    //     dominate every pid's use site. For today's kernels every
    //     pid is in the entry block, but we scan defensively in case
    //     a future kernel nests pids (e.g. inside scf.if) where walk
    //     order doesn't imply block order.
    triton::GetProgramIdOp firstPid = pids.front();
    for (auto pid : pids) {
      if (pid->isBeforeInBlock(firstPid))
        firstPid = pid;
    }

    OpBuilder builder(firstPid);
    Location loc = firstPid.getLoc();

    // 1b. Emit the single variadic ktdp op — produces `numDims`
    //     index results, one per grid axis (the "tuple unwrap").
    //       %p0_i, %p1_i = ktdp.get_compute_tile_id : index, index
    SmallVector<Type> idxTypes(numDims, builder.getIndexType());
    auto coreIds =
        mlir::ktdp::GetComputeTileIdOp::create(builder, loc, idxTypes);

    // 1c. Axis-keyed i32 casts: one per axis, shared across every
    //     pid reading that axis. i32Per[axis] is the single value
    //     all such pids will be rewritten to.
    //       %p0 = arith.index_cast %p0_i : index to i32
    //       %p1 = arith.index_cast %p1_i : index to i32
    SmallVector<Value> i32Per;
    i32Per.reserve(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      auto cast = arith::IndexCastOp::create(
          builder, loc, builder.getI32Type(), coreIds->getResult(i));
      i32Per.push_back(cast.getResult());
    }

    // 1d. Rewire every pid's uses to its axis's shared cast, then
    //     erase the pid. Multiple reads of the same axis collapse
    //     to one SSA def.
    for (auto pid : pids) {
      unsigned axis = static_cast<unsigned>(pid.getAxisAsInt());
      pid.getResult().replaceAllUsesWith(i32Per[axis]);
      pid.erase();
    }

    // --- Step 2: fold tt.get_num_programs to arith.constant ----------
    // Each num_programs read resolves to grid[axis] — a compile-time
    // constant given the pass option. We fold in place (constant
    // inserted just before each op); the canonicalize+CSE at the end
    // of the pipeline collapses duplicate constants across the
    // function.
    //
    // Before (1D kernel, grid = [32]):
    //   %n = tt.get_num_programs 0 : i32   // use U1
    //
    // After:
    //   %n = arith.constant 32 : i32       // U1 now uses %n
    for (auto np : nprogs) {
      unsigned axis = static_cast<unsigned>(np.getAxisAsInt());
      builder.setInsertionPoint(np);
      auto cst = arith::ConstantOp::create(
          builder, np.getLoc(), builder.getI32Type(),
          builder.getI32IntegerAttr(static_cast<int32_t>(grid[axis])));
      np.getResult().replaceAllUsesWith(cst.getResult());
      np.erase();
    }

    // --- Step 3: stamp grid ------------------------------------------
    // Verbatim from the pass option — the caller tells us exactly how
    // the hardware cores partition across the kernel's pid axes.
    SmallVector<int64_t> gridAttr(grid.begin(), grid.end());
    fnOp->setAttr("grid", builder.getI64ArrayAttr(gridAttr));
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>>
createDistributeWorkPass(ArrayRef<int64_t> grid) {
  return std::make_unique<DistributeWorkPass>(grid);
}
} // namespace mlir::triton::ktdp
