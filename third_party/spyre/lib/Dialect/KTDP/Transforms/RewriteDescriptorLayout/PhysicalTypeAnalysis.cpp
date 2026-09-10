//===- PhysicalTypeAnalysis.cpp - Phase 2A: decide types before rewriting -===//
//
// Analysis only. Walks forward from Phase 1's roots and computes, for every
// reachable value, the final physical type that value will carry once Phase 2
// has rewritten the IR. Creates no ops and mutates nothing.
//
// The answer is a function of Phase 1's output alone, and dispatchSource reads
// it: an operand present here will be physical, so a miss on
// ctx.physicalValues means "not yet", while absence here means "genuinely
// logical".
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
            const PhysicalTypeInfo &srcInfo,
            const LayoutRequirement *want) const override {
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
            const PhysicalTypeInfo &srcInfo,
            const LayoutRequirement *want) const override {
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
            const PhysicalTypeInfo &srcInfo,
            const LayoutRequirement *want) const override {
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
            const PhysicalTypeInfo &srcInfo,
            const LayoutRequirement *want) const override {
    return failure();
  }
};

/// linalg.reduce: Physical exactly when the layout it is stored under is the one
/// its operand's surviving stick structure induces. Three conditions, and failing
/// any of them is not an error -- it is the Logical answer, where the store's
/// widen stage builds the physical form afterwards.
///
///   1. One input and one init, and the input is the value that resolved
///      physical. A multi-operand reduce (argmax) would need every input to
///      induce the same layout; nothing asks for that yet.
///   2. The induced layout equals the one WANTED of this result -- the backward
///      analysis's requirement at the result. Reducing a logical dim away
///      deletes the physical dims sourced from it, so the survivors in order --
///      same coord op and arg, axes renumbered -- are the layout the result
///      would carry. Compared against the requirement on all three coordinate
///      arrays AND its physical extents: the arrays fix the order, the extents
///      fix the sizes.
///   3. The init is rebuildable at the physical shape, asked here so this
///      decision and the emission cannot disagree (canRebuildPhysicalInit).
///
/// Why a reduce has to be asked at all, and why the answer belongs to the store
/// rather than the input: "Which physical shape" in docs/spyre-tensor-layouts.md.
struct ReducePropagation : PhysicalPropagationPattern {

  bool match(Operation *op) const override {
    return isa<linalg::ReduceOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo,
            const LayoutRequirement *want) const override {
    auto rd = cast<linalg::ReduceOp>(op);
    // Condition 1.
    if (rd.getNumDpsInputs() != 1 || rd.getNumDpsInits() != 1 ||
        src != rd.getInputs()[0])
      return failure();
    auto marker = srcInfo.marker;
    if (!marker)
      return failure();
    // An erased transpose means the marker's dim order and the tile's differ,
    // so "the surviving physical dims in the marker's order" is not the order
    // the emitted op sees and the induced layout would be a fiction. The
    // Logical space handles that case as it always did.
    if (!srcInfo.transposePerm.empty())
      return failure();

    auto inTy = dyn_cast<RankedTensorType>(srcInfo.type);
    auto resTy = dyn_cast<RankedTensorType>(result.getType());
    if (!inTy || !resTy)
      return failure();
    auto physSrc = marker.getPhysSrc();
    auto physOp = marker.getPhysOp();
    auto physArg = marker.getPhysArg();
    if (inTy.getRank() != (int64_t)physSrc.size())
      return failure();

    unsigned logicalRank = 0;
    for (int64_t s : physSrc)
      logicalRank = std::max(logicalRank, (unsigned)(s + 1));
    // `dimensions` must be interpretable as logical dims of this marker --
    // the same precondition RewriteReducePattern states before it builds
    // canonicalAxes from them.
    if (llvm::any_of(rd.getDimensions(),
                     [&](int64_t d) { return d >= (int64_t)logicalRank; }))
      return failure();

    llvm::SmallVector<int64_t> canonicalAxes(logicalRank, -1);
    int64_t outAxis = 0;
    for (unsigned d = 0; d < logicalRank; ++d)
      if (!llvm::is_contained(rd.getDimensions(), (int64_t)d))
        canonicalAxes[d] = outAxis++;

    // The induced output layout: the surviving physical dims, in order.
    llvm::SmallVector<int64_t> indSrc, indOp, indArg, indShape;
    for (unsigned p = 0; p < physSrc.size(); ++p) {
      int64_t role = canonicalAxes[physSrc[p]];
      if (role < 0)
        continue;
      indSrc.push_back(role);
      indOp.push_back(physOp[p]);
      indArg.push_back(physArg[p]);
      indShape.push_back(inTy.getDimSize(p));
    }

    if (!canRebuildPhysicalInit(rd.getDpsInits()[0]))
      return failure();

    // Nothing wants a layout of this result, so it carries none.
    if (!want)
      return failure();
    const LayoutRequirement &req = *want;
    if (llvm::ArrayRef<int64_t>(indSrc) !=
            llvm::ArrayRef<int64_t>(req.physSrc) ||
        llvm::ArrayRef<int64_t>(indOp) != llvm::ArrayRef<int64_t>(req.physOp) ||
        llvm::ArrayRef<int64_t>(indArg) !=
            llvm::ArrayRef<int64_t>(req.physArg) ||
        llvm::ArrayRef<int64_t>(indShape) !=
            llvm::ArrayRef<int64_t>(req.physExtents))
      return failure();

    // The marker recorded is the OUTPUT's, not the operand's: it is the layout
    // this value carries, and it is what the emission registers against the
    // value it mints. It comes from the requirement, so it is the store's marker
    // -- the same one the forward walk used to fetch.
    return PhysicalTypeInfo{
        RankedTensorType::get(indShape, resTy.getElementType()), req.marker, {}};
  }
};

/// tensor.expand_shape / collapse_shape / reshape: no physical result.
///
/// The obstacle is the coordinate map, not the shape -- the result extents are
/// derivable from the reassociation map the op carries. PhysicalTypeInfo pairs a
/// type with a marker whose phys_src/phys_op/phys_arg arrays are indexed per
/// PHYSICAL DIM. Collapsing physical [1, 64, 64] under
/// phys_op = [floor, id, mod] via [[0, 1], [2]] fuses the stick index with the
/// row, leaving a 2-dim result while the marker still describes 3 dims: the pair
/// would be internally inconsistent. If someone later works out how to rewrite a
/// marker across a reassociation map, this is the one place that changes.
///
/// Declining is a positive statement of these ops' own rule, and it must be
/// registered ahead of the structural elementwise rule -- see
/// populatePhysicalPropagationPatterns.
struct ReshapePropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
               tensor::ReshapeOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo,
            const LayoutRequirement *want) const override {
    return failure();
  }
};

/// linalg.broadcast: no physical result, for a different reason than the reshape
/// family above.
///
/// A broadcast's result shape is not derived from its operand at all: it is a
/// target shape fixed when the op was built, which LowerComputeOps builds from
/// tt.broadcast against the operand's LOGICAL rank. Given a physical operand
/// that target shape is simply stale -- it describes a tensor of the wrong rank,
/// and a consuming elementwise op then fails its same-type constraint.
///
/// Unlike a reshape's coordinate map, this is recoverable in principle:
/// recomputing the target shape and `dimensions` against the physical rank would
/// let the op work on physical operands directly, which is what a reduce ->
/// broadcast -> elementwise chain needs -- the shape softmax and layernorm are
/// written in. Until that exists, declining is the safe answer and such an
/// operand is rejected downstream rather than guessed at.
struct BroadcastPropagation : PhysicalPropagationPattern {
  bool match(Operation *op) const override {
    return isa<linalg::BroadcastOp>(op);
  }

  llvm::FailureOr<PhysicalTypeInfo>
  propagate(Operation *op, Value result, Value src,
            const PhysicalTypeInfo &srcInfo,
            const LayoutRequirement *want) const override {
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
  // The reshape family and linalg.broadcast are the load-bearing cases: they
  // have a single tensor operand, so the elementwise rule's "every tensor
  // operand agrees on a shape" test is satisfied trivially and it WOULD claim
  // them if asked first -- recording the operand's shape against a result of a
  // different one.
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
                    const RequirementMap &requirements,
                    llvm::SmallVector<Value> &invocationStack) {
  // Already decided.
  if (auto it = resolved.find(value); it != resolved.end())
    return it->second;
  if (auto it = roots.find(value); it != roots.end())
    return it->second;

  // Cycle detection; the invocationStack contract is stated on the declaration.
  // A cycle has no fixpoint this analysis is asked to compute, so fail rather
  // than recurse forever.
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
    // No pattern is this op's rule. An untaught op stays visible here rather
    // than acquiring a default: if an op can appear on a physicalized chain it
    // needs a pattern, and guessing one would silently compute the wrong slice.
    // (The reshape family and linalg.broadcast used to be the example here; they
    // have patterns of their own now and no longer reach this branch.)
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
          getPhysicalizedType(o, roots, resolved, patterns, requirements,
                              invocationStack);
      if (succeeded(sub)) {
        src = o;
        break;
      }
    }
  }
  if (!src)
    return failure();

  auto srcInfo =
      getPhysicalizedType(src, roots, resolved, patterns, requirements,
                          invocationStack);
  if (failed(srcInfo))
    return failure();

  // The one lookup, done here rather than in the pattern: a rule is asked about
  // `value`, so it is handed `value`'s requirement and nothing else.
  auto reqIt = requirements.find(value);
  const LayoutRequirement *want =
      reqIt != requirements.end() ? &reqIt->second : nullptr;
  auto info = pattern->propagate(defOp, value, src, *srcInfo, want);
  if (failed(info))
    return failure();

  resolved[value] = *info;
  return *info;
}

//===----------------------------------------------------------------------===//
// runPhysicalTypeAnalysis
//===----------------------------------------------------------------------===//

PhysicalTypeMap runPhysicalTypeAnalysis(ModuleOp module, const PassContext &ctx,
                                        const RequirementMap &requirements) {
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
                                      requirements,
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
// PhysicalTypeCarryForward
//===----------------------------------------------------------------------===//

PhysicalTypeCarryForward::PhysicalTypeCarryForward(PhysicalTypeMap &analysis)
    : analysis(&analysis) {}

void PhysicalTypeCarryForward::carryForward(Value replaced,
                                            Value replacement) const {
  assert(analysis &&
         "carryForward on an inert handle: the decision being carried forward "
         "is one only Phase 2A can have made, so there must be a map to carry "
         "it in");
  auto it = analysis->find(replaced);
  assert(it != analysis->end() &&
         "carryForward: the replaced value has no analysis entry -- a pattern "
         "minted a replacement without the analysis having decided the "
         "original is physical, which is the condition that puts it on that "
         "path in the first place");
  if (it == analysis->end())
    return;
  // Copy out before inserting: insertion may rehash and invalidate `it`.
  PhysicalTypeInfo decided = it->second;
  (*analysis)[replacement] = decided;
}

//===----------------------------------------------------------------------===//
// verifyPhysicalTypeAgreement
//===----------------------------------------------------------------------===//

void verifyPhysicalTypeAgreement(ModuleOp module, const PassContext &ctx,
                                 const PhysicalTypeMap &analysis,
                                 llvm::StringRef when) {
// The only #ifndef NDEBUG in this library, against two dozen LLVM_DEBUG sites,
// and deliberately not one of them. LLVM_DEBUG is gated on a runtime flag, so a
// check inside it runs only when someone thinks to ask -- this one has to run
// unasked, in CI, on every build. A plain assert() would compile the condition
// away but not the walk over physicalValues that feeds it. Note the default
// build is TritonRelBuildWithAsserts, so this is live in every build we
// actually use; it is debug-only in name, not in practice.
#ifndef NDEBUG
  // A disagreement is a defect in the analysis, not an input the user can fix,
  // so it is an assertion rather than a diagnostic.

  // ctx.physicalValues must be a SUBSET of the analysis map: every value
  // Phase 2 discovered to be physical, the analysis predicted up front. The
  // converse is not required -- the analysis also predicts types for values
  // Phase 2 rewrites away before it would record them (an erased transpose's
  // result, for instance), so it is allowed to be strictly larger.
  //
  // No exemptions. A value Phase 2 *mints* (a source pattern's replacement)
  // postdates the analysis and so could not be in the map on its own, but the
  // pattern that minted it also carried the replaced value's entry forward onto
  // it (PhysicalTypeCarryForward), so containment holds for it by construction
  // rather than by being excused.
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
