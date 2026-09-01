//===- PhysicalTypeAnalysis.cpp - Phase 2A: decide types before rewriting -===//
//
// Analysis only. Walks forward from Phase 1's roots and computes, for every
// reachable value, the final physical type that value will carry once Phase 2
// has rewritten the IR. Creates no ops and mutates nothing.
//
// Phase 2 used to answer "is this operand final?" from the IR it was itself
// concurrently mutating, which it could only do by enumerating ways a value
// might be unresolved. This analysis answers the same question as a function of
// Phase 1's output alone, and dispatchSource now reads it: an operand present
// here will be physical, so a miss on ctx.physicalValues means "not yet",
// while absence here means "genuinely logical".
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
// Propagation patterns: one per op, each computing its own result type
//===----------------------------------------------------------------------===//

/// Elementwise: the result takes the operand's physical shape verbatim, keeping
/// its own element type. Matched by the same purely local rule
/// RewriteElementwisePattern uses -- one result, a RankedTensorType, every
/// tensor operand agreeing on a shape. Local is safe because reachability comes
/// from the seeded walk, so an op on an unannotated path is never asked; that
/// is what keeps tt.expand_dims and its rank-changing siblings, which satisfy
/// the shape rule, from being matched here.
struct ElementwisePropagation : PhysicalPropagationPattern {
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

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const override {
    PhysicalTypeInfo info = srcInfo;
    auto resTy = cast<RankedTensorType>(result.getType());
    auto srcTy = cast<RankedTensorType>(srcInfo.type);
    info.type = RankedTensorType::get(srcTy.getShape(), resTy.getElementType());
    return info;
  }
};

/// linalg.transpose: the result is the same physical data in a permuted dim
/// order, so the permutation composes onto the chain -- exactly what
/// RewriteTransposePattern records against the transpose's input as it erases
/// the op, and what dispatchSource later folds into canonicalAxes. The erasure
/// replaces the transpose with its input, so the surviving value carries the
/// INPUT's physical type, not a permuted one.
struct TransposePropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<linalg::TransposeOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const override {
    PhysicalTypeInfo info = srcInfo;
    auto tr = cast<linalg::TransposeOp>(op);
    auto perm = llvm::SmallVector<int64_t>(tr.getPermutation());
    info.transposePerm = info.transposePerm.empty()
                             ? perm
                             : composePerm(info.transposePerm, perm);
    info.type = cast<RankedTensorType>(src.getType());
    return info;
  }
};

/// linalg.matmul / linalg.batch_matmul: they contract exactly one K axis, so a
/// stick-split K needs real cross-stick accumulation and the emitted
/// accumulator is at op-tile (logical) shape. No physical result.
struct MatmulPropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const override {
    return failure();
  }
};

/// ktdp.store: a sink with no result at all, so nothing to propagate.
struct StorePropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<mlir::ktdp::StoreOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const override {
    return failure();
  }
};

/// linalg.reduce: no physical result.
///
/// The reduce's result shape IS its logical shape -- reducing logical dim `d`
/// off a logical input leaves the remaining logical dims, and that is what the
/// emitted op produces. Surviving stick dims on the input do not make the
/// result physical: they belong to the *operand's* layout, and a result is
/// physical only under a layout of its own, which a reduce result acquires
/// only when the output descriptor is annotated. Any physical form is built
/// afterwards, by the store's widen stage.
///
/// So this is stated as a rule, not measured from the operand's marker: no
/// reduce produces a physical result today, whatever its input layout does.
struct ReducePropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<linalg::ReduceOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const override {
    return failure();
  }
};

/// tensor.expand_shape / collapse_shape / reshape: no physical result.
///
/// Not because the result SHAPE is unknown -- it is derivable. A
/// tensor.collapse_shape carries its reassociation map in the op, so the result
/// extents follow from the operand's shape plus that map.
///
/// The obstacle is the coordinate map, not the shape. PhysicalTypeInfo pairs a
/// type with a marker, and the marker's phys_src/phys_op/phys_arg arrays are
/// indexed per PHYSICAL DIM. Collapsing physical [1, 64, 64] under
/// phys_op = [floor, id, mod] via [[0, 1], [2]] fuses the stick index with the
/// row, leaving a 2-dim result while the marker still describes 3 dims. The
/// type and the marker would no longer agree, so the pair is internally
/// inconsistent: the shape is knowable, the layout is not.
///
/// Declining is therefore a positive statement of these ops' own rule rather
/// than a hole in the structural elementwise predicate -- which would
/// otherwise claim them, since "every tensor operand agrees on a shape" is
/// satisfied trivially by an op with one tensor operand, and propagate() would
/// then hand the result the OPERAND's shape, contradicting the op's own result
/// type.
///
/// An operand produced by one of these is rejected downstream as a legality
/// question (dispatchSource's reshape/broadcast check), which is separate from
/// asking whether the operand is physical. If someone later works out how to
/// rewrite a marker across a reassociation map, this is the one place that
/// changes.
struct ReshapePropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
               tensor::ReshapeOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const override {
    return failure();
  }
};

/// linalg.broadcast: no physical result -- but for a different reason than the
/// reshape family above, which is why it is stated separately.
///
/// A broadcast's result shape is not derived from its operand at all: it is a
/// target shape fixed when the op was built, and `dimensions` names the axes
/// added to reach it. LowerComputeOps builds it from tt.broadcast against the
/// operand's LOGICAL rank. Given a physical operand, that target shape is
/// simply stale -- it describes a tensor of the wrong rank, and a consuming
/// elementwise op then fails its same-type constraint.
///
/// Unlike a reshape's coordinate map, this is recoverable in principle: the
/// target shape and `dimensions` could be recomputed against the physical rank,
/// which would let the op work on physical operands directly. That capability
/// does not exist yet, and it is what a reduce -> broadcast -> elementwise chain
/// needs -- the shape softmax and layernorm are written in. Until it does,
/// declining here is the safe answer, and an operand produced by a broadcast is
/// rejected downstream rather than guessed at.
struct BroadcastPropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<linalg::BroadcastOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo) const override {
    return failure();
  }
};

/// The tensor operand an op inherits its physicality from: the first tensor
/// operand already known to be physical.
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

/// The pattern whose rule covers `op`, or null when this pass has not been
/// taught the op.
const PhysicalPropagationPattern *
lookupPattern(Operation *op, const PhysicalPropagationPatternSet &patterns) {
  for (const auto &p : patterns)
    if (p->match(op))
      return p.get();
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// populatePhysicalPropagationPatterns
//===----------------------------------------------------------------------===//

void populatePhysicalPropagationPatterns(
    PhysicalPropagationPatternSet &patterns) {
  // Order matters only where two patterns could match the same op. Every
  // named-op pattern must be asked before the structural elementwise rule,
  // which a linalg op with uniformly shaped operands could otherwise satisfy.
  // The reshape family and linalg.broadcast are the load-bearing cases: those
  // ops have a single tensor operand, so the elementwise rule's "every tensor
  // operand agrees on a shape" test is satisfied trivially and it WOULD claim
  // them if asked first -- recording the operand's shape against a result of a
  // different one. Both are registered ahead of it for exactly that reason.
  patterns.push_back(std::make_unique<TransposePropagation>());
  patterns.push_back(std::make_unique<MatmulPropagation>());
  patterns.push_back(std::make_unique<StorePropagation>());
  patterns.push_back(std::make_unique<ReducePropagation>());
  patterns.push_back(std::make_unique<ReshapePropagation>());
  patterns.push_back(std::make_unique<BroadcastPropagation>());
  patterns.push_back(std::make_unique<ElementwisePropagation>());
}

//===----------------------------------------------------------------------===//
// getPhysicalizedType
//===----------------------------------------------------------------------===//

llvm::FailureOr<PhysicalTypeInfo>
getPhysicalizedType(Value value, const PhysicalTypeMap &roots,
                    PhysicalTypeMap &resolved,
                    const PhysicalPropagationPatternSet &patterns,
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
  // no fixpoint this analysis is asked to compute.
  if (llvm::is_contained(invocationStack, value)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  [2A] cycle on value, not physicalizing: " << value << "\n");
    return failure();
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    // A BlockArgument: the region boundary. Phase 2's candidate list requires
    // exactly one result outside its named-op branch, and scf.for has as many
    // results as iter_args -- so a pre-existing scf.for carrying a tensor is
    // never a candidate and its iter_arg is never retyped. Resolving through
    // iter_args here would claim a physical type for a value the rewrite
    // leaves logical, so this boundary deliberately stops instead.
    return failure();
  }

  const PhysicalPropagationPattern *pattern = lookupPattern(defOp, patterns);
  if (!pattern) {
    // No pattern is this op's rule. Notably a reshape/expand_shape/
    // collapse_shape/broadcast lands here: its element-to-index mapping is
    // neither a physical tensor's nor a plain logical one's, so there is no
    // propagation rule to state and guessing one would silently compute the
    // wrong slice. An untaught op must stay visible here rather than acquire a
    // default: if an op can appear on a physicalized chain, it needs a pattern.
    LLVM_DEBUG(llvm::dbgs()
               << "  [2A] no propagation rule for " << defOp->getName()
               << "; treating its result as logical\n");
    return failure();
  }

  // Resolving this value requires resolving its producer's operand.
  invocationStack.push_back(value);
  llvm::scope_exit popStack([&] { invocationStack.pop_back(); });

  Value src = findPhysicalTensorOperand(defOp, roots, resolved);
  if (!src) {
    // No operand is known physical yet; try to resolve one transitively.
    for (Value o : defOp->getOperands()) {
      if (!isa<RankedTensorType>(o.getType()))
        continue;
      auto sub =
          getPhysicalizedType(o, roots, resolved, patterns, invocationStack);
      if (succeeded(sub)) {
        src = o;
        break;
      }
    }
  }
  if (!src)
    return failure();

  auto srcInfo =
      getPhysicalizedType(src, roots, resolved, patterns, invocationStack);
  if (failed(srcInfo))
    return failure();

  auto info = pattern->propagate(defOp, value, src, *srcInfo);
  if (failed(info))
    return failure();

  resolved[value] = *info;
  return *info;
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

  PhysicalPropagationPatternSet patterns;
  populatePhysicalPropagationPatterns(patterns);

  PhysicalTypeMap resolved;

  // Forward walk. Each root's users are asked to resolve themselves, which
  // recurses back through getPhysicalizedType -- so the walk order does not
  // matter: a value's answer is a function of the seed set and the op graph,
  // not of how far the walk has got.
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
        if (failed(getPhysicalizedType(result, roots, resolved, patterns,
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

  // One map: roots plus everything reachable from them.
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
  // The surviving agreement invariant. The two that lived in dispatchSource
  // compared the analysis against the guards it has now replaced, so they went
  // with them; this one does not mention a guard, it relates the analysis to
  // what Phase 2 actually discovered, and stays meaningful. A disagreement is a
  // defect in the analysis, not an input the user can fix, so it is an
  // assertion rather than a diagnostic.

  // ctx.physicalValues must be a SUBSET of the analysis map: every value
  // Phase 2 discovered to be physical, the analysis predicted up front. The
  // converse is not required -- the analysis also predicts types for values
  // Phase 2 rewrites away before it would record them (an erased transpose's
  // result, for instance), so it is allowed to be strictly larger.
  for (auto &[value, phys] : ctx.physicalValues) {
    auto it = analysis.find(value);
    assert(it != analysis.end() &&
           "Phase 2A disagreement: a value Phase 2 made physical is absent "
           "from the analysis map -- the analysis under-claims");
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
