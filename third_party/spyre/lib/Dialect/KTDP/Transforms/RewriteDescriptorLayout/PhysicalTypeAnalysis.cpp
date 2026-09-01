//===- PhysicalTypeAnalysis.cpp - Phase 2A: decide types before rewriting -===//
//
// Analysis only. Walks forward from Phase 1's roots and computes, for every
// reachable value, the final physical type that value will carry once Phase 2
// has rewritten the IR. Creates no ops and mutates nothing.
//
// Phase 2 today answers "is this operand final?" from the IR it is itself
// concurrently mutating, which it can only do by enumerating ways a value
// might be unresolved (chainBlockedByPendingTranspose, the reshape/broadcast
// blocklist). This analysis answers the same question as a function of Phase
// 1's output alone. Step 4c-A lands the analysis and the assertions proving it
// agrees with those guards; step 4c-B is what replaces them.
//
//===----------------------------------------------------------------------===//

#include "RewriteDescriptorLayout/PhysicalTypeAnalysis.h"
#include "RewriteDescriptorLayout/ContractionSynthesis.h"
#include "RewriteDescriptorLayout/PermutationUtils.h"

#include "ktir/Dialect/KTDP/KTDP.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

//===----------------------------------------------------------------------===//
// Per-op behaviour: what an op does to physicality
//===----------------------------------------------------------------------===//

// Item 4 of the plan asks that each op answer for itself rather than appear in
// a central isa<> list. This classifier is that answer, deliberately as a free
// function over a behaviour enum rather than an MLIR op interface (.td):
//
//   - Every op that would implement the interface lives in a dialect this pass
//     does not own (linalg, arith, tensor, ktdp). An interface would therefore
//     have to be attached via external models registered by this pass, which
//     puts the per-op knowledge right back in one central registration list --
//     the same shape as the isa<> list, with a .td file's worth of machinery on
//     top and no new caller, since nothing outside this pass asks the question.
//   - 4c-A mutates nothing, so the classifier has exactly one consumer.
//
// What replaces the interface's "a new op implements it or inherits a default"
// obligation: the switch below has NO silent default. An op this pass has not
// been taught falls to PhysicalPropagation::Unknown, which the analysis treats
// as Stop *and reports* it (the Unknown case in getPhysicalizedType logs the
// op name under -debug-only=rewrite-descriptor-layout). So the
// obligation a future op author inherits is: if your op can appear on a
// physicalized chain, name its behaviour here. Failing to do so is visible
// (an Unknown on a reachable chain), not silent.
//
// If 4c-B or a later step gives the question a second caller outside this pass,
// promoting this to a real op interface becomes worth its cost; until then it
// is not.
PhysicalPropagation classifyPropagation(Operation *op) {
  // --- Absorb: contributes a permutation and no new physical value. ---
  if (isa<linalg::TransposeOp>(op))
    return PhysicalPropagation::Absorb;

  // --- Stop: consumes physical, produces logical (or nothing). ---
  //
  // matmul / batch_matmul: contract exactly one K axis, so a stick-split K
  // needs real cross-stick accumulation; the emitted accumulator is at
  // op-tile (logical) shape.
  //
  // linalg.reduce: also stops. `dimensions` being a strictly-sorted list means
  // a reduce can *absorb* an arbitrary physical reduce-axis set into one op
  // (that is step 3's absorbLoopDims), but absorbing the input axes is not the
  // same as producing a physical result: emitNarrowStage builds the emitted
  // reduce's result type from accDims, the union of (outputAxis, extent) pairs
  // over opTileDims, which is the op-tile/logical output shape. A reduce
  // result is therefore never physical.
  //
  // Verified empirically as well as by reading, because under-claiming here is
  // the silent direction: instrumenting the single writer that could make a
  // reduce result physical (RewriteElementwisePattern) and running all 33
  // Conversion fixtures plus the whole pytest suite recorded 4 + 23 retypes,
  // every one of them an arith.negf/arith.addf directly on a ktdp.load result,
  // and zero on a linalg.reduce. That agrees with the code: linalg.reduce has
  // two tensor operands (ins + outs) whose shapes differ by construction, and
  // RewriteElementwisePattern requires every tensor operand to agree on a
  // shape before it retypes anything -- so it can never fire on a reduce.
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ReduceOp,
          mlir::ktdp::StoreOp>(op))
    return PhysicalPropagation::Stop;

  // --- PassThrough: rank-agnostic, the physical type propagates unchanged. ---
  //
  // The same purely local rule RewriteElementwisePattern uses: one result, a
  // RankedTensorType, and every tensor operand agreeing on a shape. It is
  // local because reachability is supplied separately, by the seeded walk --
  // an op on an unannotated path is never asked. That is what keeps this from
  // mis-firing on tt.expand_dims and its seven rank-changing siblings, which
  // satisfy the shape rule but are never reachable from a root.
  if (op->getNumResults() == 1 &&
      isa<RankedTensorType>(op->getResult(0).getType())) {
    ArrayRef<int64_t> commonShape;
    bool sawTensorOperand = false;
    bool agree = true;
    for (Value o : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(o.getType());
      if (!t)
        continue;
      if (!sawTensorOperand) {
        commonShape = t.getShape();
        sawTensorOperand = true;
      } else if (t.getShape() != commonShape) {
        agree = false;
      }
    }
    if (sawTensorOperand && agree)
      return PhysicalPropagation::PassThrough;
  }

  // Everything else. Notably a reshape/expand_shape/collapse_shape/broadcast
  // reaches here: its element-to-index mapping is neither a physical tensor's
  // nor a plain logical one's, so there is no propagation rule to state and
  // guessing one would silently compute the wrong slice (#91/#92). Phase 2's
  // blocklist hard-errors on exactly these; the analysis reports Unknown,
  // which carries the same "this pass has no answer" meaning without having to
  // enumerate the op list.
  return PhysicalPropagation::Unknown;
}

/// The tensor operand a PassThrough/Absorb op inherits its physicality from:
/// the first tensor operand already known to be physical.
Value findPhysicalTensorOperand(Operation *op, const PhysicalTypeMap &roots,
                                const PhysicalTypeMap &resolved) {
  for (Value o : op->getOperands()) {
    if (!isa<RankedTensorType>(o.getType()))
      continue;
    if (roots.contains(o) || resolved.contains(o))
      return o;
  }
  return {};
}

} // namespace

//===----------------------------------------------------------------------===//
// getPhysicalizedType
//===----------------------------------------------------------------------===//

llvm::FailureOr<PhysicalTypeInfo>
getPhysicalizedType(Value value, const PhysicalTypeMap &roots,
                    PhysicalTypeMap &resolved,
                    llvm::SmallVector<Value> &invocationStack) {
  // Already decided.
  if (auto it = resolved.find(value); it != resolved.end())
    return it->second;
  if (auto it = roots.find(value); it != roots.end())
    return it->second;

  // Cycle detection, mirroring BufferizableOpInterface::getBufferType's
  // invocationStack contract: the stack holds every value whose computation is
  // in progress, and finding `value` on it means the value graph closed a loop
  // (a tensor through scf.for iter_args does exactly that even though the op
  // graph is a DAG). Return failure rather than recurse forever -- a cycle has
  // no fixpoint this analysis is asked to compute, so the value is treated as
  // not resolving to a physical type.
  if (llvm::is_contained(invocationStack, value)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  [2A] cycle on value, not physicalizing: " << value << "\n");
    return failure();
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    // A BlockArgument. This is the region boundary, and 4c-A's stated decision
    // is to REPRODUCE Phase 2's current multi-result exclusion deliberately
    // rather than handle it:
    //
    //   Phase 2's candidate list has a named-op isa<> branch and a structural
    //   catch-all requiring exactly one result. scf.for has as many results as
    //   iter_args, so a pre-existing scf.for carrying a tensor is never a
    //   candidate and its iter_arg is never retyped today. An analysis that
    //   resolved through iter_args would then claim a physical type for a
    //   value the rewrite leaves logical -- a disagreement introduced by the
    //   analysis, not measured by it, which is the opposite of what 4c-A is
    //   for.
    //
    // 4c-B is where the exclusion is either kept or lifted, together with the
    // rewrite that would honour it. Measured across all 33 Conversion fixtures
    // and the pytest kernels: the one pre-existing tensor iter_args on a
    // marker-carrying function (attn_pv in
    // rewrite-descriptor-layout-parallel-scatter-iter-arg.mlir) carries the
    // matmul accumulator, which is a Stop result and never physical -- so no
    // fixture reaches this boundary with a physical value today.
    return failure();
  }

  // Resolving this value requires resolving its producer's operand.
  invocationStack.push_back(value);
  llvm::scope_exit popStack([&] { invocationStack.pop_back(); });

  PhysicalPropagation behaviour = classifyPropagation(defOp);
  switch (behaviour) {
  case PhysicalPropagation::Stop:
    // The op consumes physical and produces logical: the chain ends here.
    return failure();
  case PhysicalPropagation::Unknown:
    LLVM_DEBUG(llvm::dbgs()
               << "  [2A] no propagation rule for " << defOp->getName()
               << "; treating its result as logical\n");
    return failure();
  case PhysicalPropagation::Absorb:
  case PhysicalPropagation::PassThrough:
    break;
  }

  Value src = findPhysicalTensorOperand(defOp, roots, resolved);
  if (!src) {
    // No operand is known physical yet; try to resolve one transitively.
    for (Value o : defOp->getOperands()) {
      if (!isa<RankedTensorType>(o.getType()))
        continue;
      auto sub = getPhysicalizedType(o, roots, resolved, invocationStack);
      if (succeeded(sub)) {
        src = o;
        break;
      }
    }
  }
  if (!src)
    return failure();

  auto srcInfo = getPhysicalizedType(src, roots, resolved, invocationStack);
  if (failed(srcInfo))
    return failure();

  PhysicalTypeInfo info = *srcInfo;

  if (behaviour == PhysicalPropagation::Absorb) {
    // linalg.transpose: the result is the same physical data in a permuted
    // dim order, so the physical type follows the permutation and the
    // permutation itself composes onto the chain -- exactly what
    // RewriteTransposePattern records against the transpose's input as it
    // erases the op, and what dispatchSource later folds into canonicalAxes.
    auto tr = cast<linalg::TransposeOp>(defOp);
    auto perm = llvm::SmallVector<int64_t>(tr.getPermutation());
    info.transposePerm = info.transposePerm.empty()
                             ? perm
                             : composePerm(info.transposePerm, perm);
    // The erasure replaces the transpose with its input, so the value that
    // survives carries the INPUT's physical type, not a permuted one.
    info.type = cast<RankedTensorType>(src.getType());
  } else {
    // PassThrough: the result takes the operand's physical shape verbatim.
    auto resTy = cast<RankedTensorType>(value.getType());
    auto srcTy = cast<RankedTensorType>(info.type);
    info.type = RankedTensorType::get(srcTy.getShape(), resTy.getElementType());
  }

  resolved[value] = info;
  return info;
}

//===----------------------------------------------------------------------===//
// runPhysicalTypeAnalysis
//===----------------------------------------------------------------------===//

PhysicalTypeMap runPhysicalTypeAnalysis(ModuleOp module,
                                        const PassContext &ctx) {
  // Phase 1's roots. retypeLoad is Phase 1's only writer to
  // ctx.physicalValues, so at this point the map holds exactly one entry per
  // physicalized ktdp.load result, and no pattern creates a ktdp.load -- the
  // seed set is closed.
  PhysicalTypeMap roots;
  for (auto &[value, phys] : ctx.physicalValues) {
    roots[value] = PhysicalTypeInfo{value.getType(), phys.marker,
                                    phys.transposePerm};
  }

  PhysicalTypeMap resolved;

  // Forward walk. Each root's users are asked to resolve themselves, which
  // recurses back through getPhysicalizedType -- so the walk order does not
  // matter (unlike Phase 2's, which is what 4c-B exists to fix): a value's
  // answer is a function of the seed set and the op graph, not of how far the
  // walk has got.
  llvm::SmallVector<Value> worklist;
  for (auto &[value, info] : roots)
    worklist.push_back(value);

  llvm::DenseSet<Value> enqueued;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      for (Value result : user->getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        if (resolved.contains(result) || roots.contains(result))
          continue;
        llvm::SmallVector<Value> invocationStack;
        if (failed(getPhysicalizedType(result, roots, resolved,
                                       invocationStack)))
          continue;
        if (enqueued.insert(result).second)
          worklist.push_back(result);
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[rewrite-descriptor-layout] Phase 2A resolved "
                 << resolved.size() << " value(s) beyond " << roots.size()
                 << " root(s)\n";
  });

  // The analysis map is roots plus everything reachable from them: 4c-B reads
  // one map, so return one map.
  PhysicalTypeMap all = std::move(roots);
  for (auto &[value, info] : resolved)
    all[value] = info;
  return all;
}

//===----------------------------------------------------------------------===//
// verifyPhysicalTypeAgreement
//===----------------------------------------------------------------------===//

void verifyPhysicalTypeAgreement(ModuleOp module, const PassContext &ctx,
                                 const PhysicalTypeMap &analysis,
                                 llvm::StringRef when) {
#ifndef NDEBUG
  // Invariant 3 of the three the plan asks for (the other two are checked at
  // the point the guards fire, inside dispatchSource -- see
  // assertAnalysisAgreesOnOperand there, since a guard's conclusion is only
  // observable when the guard runs).
  //
  // At end of Phase 2, ctx.physicalValues must be a SUBSET of the analysis
  // map: every value Phase 2 discovered to be physical, the analysis predicted
  // up front. The converse is not required -- the analysis also predicts
  // types for values Phase 2 rewrites away before it would record them (an
  // erased transpose's result, for instance), so the analysis is allowed to be
  // strictly larger.
  for (auto &[value, phys] : ctx.physicalValues) {
    auto it = analysis.find(value);
    assert(it != analysis.end() &&
           "Phase 2A disagreement: a value Phase 2 made physical is absent "
           "from the analysis map -- the analysis under-claims, which is the "
           "silent direction 4c-B would build a wrong decision on");
    (void)it;
  }
  LLVM_DEBUG(llvm::dbgs()
             << "[rewrite-descriptor-layout] Phase 2A agreement (" << when
             << "): physicalValues (" << ctx.physicalValues.size()
             << ") is a subset of the analysis map (" << analysis.size()
             << ")\n");
#else
  (void)module;
  (void)ctx;
  (void)analysis;
  (void)when;
#endif
}

} // namespace mlir::triton::ktdp
