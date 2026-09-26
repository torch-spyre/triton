//===- Ops.cpp - The tts dialect's ops ------------------------------------===//
//
// Two ops. `tensor_layout`'s verifier delegates rather than restates -- see the
// note on `tts::verifyTensorLayoutArrays` for why the rules have a single owner.
// `pin`'s rules are its own, and they are all STRUCTURAL: what shape an address
// expression may have, not what numbers it may hold. The numeric rules --
// scratchpad capacity, stick alignment -- need the device description, which this
// tree deliberately does not read (see `SpyreUtils.get_device_properties`), so
// they belong to `PlacePinnedValues`, which takes them as pass options.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Dialect.h"

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#define GET_OP_CLASSES
#include "Dialect/TTS/IR/Ops.cpp.inc"

namespace mlir::triton::tts {

LogicalResult TensorLayoutOp::verify() {
  // The logical rank comes from the descriptor's BLOCK type. The attribute form
  // reads it from the memory view's memref instead; the extents differ between
  // the two, the rank does not, so both measure the same bound.
  unsigned logicalRank = getDesc().getType().getBlockType().getRank();

  // `emitError` and not `emitOpError`: every message the shared checker
  // produces already names `tts.tensor_layout`, so the op-error prefix would
  // say it twice — and identical text either side of the lowering is what makes
  // the op and the attribute diagnosable as one contract.
  auto emitError = [&]() { return this->emitError(); };
  return verifyTensorLayoutArrays(getPhysSrc(), getPhysOp(), getPhysArg(),
                                  logicalRank, emitError);
}

//===----------------------------------------------------------------------===//
// tts.pin
//===----------------------------------------------------------------------===//

namespace {

/// The integer a value is, if it is a constant.
///
/// `m_ConstantInt` rather than `arith::ConstantOp`, so this admits any
/// ConstantLike op folding to an integer attribute. A pinned address written as
/// `BASE + pid * STRIDE` with `tl.constexpr` coefficients reaches us as
/// `arith.constant`s today; nothing here depends on that staying true.
std::optional<int64_t> matchConstantInt(Value v) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return c.getSExtValue();
  return std::nullopt;
}

/// The coefficient of a `tl.program_id(0)` term: `pid`, `pid * k` or `k * pid`.
///
/// Axis X only. The launch grid this backend distributes over is
/// one-dimensional -- `TTIRToKTIRPipelineOptions.grid` is what DistributeWork
/// divides and what `wk_slice_coord` indexes -- so X is the only axis whose
/// address set is a finite enumerable `{base + i*stride : i < grid}`. A Y or Z
/// term would have no bound to enumerate against, which is the whole reason the
/// form is restricted.
std::optional<int64_t> matchProgramIdTerm(Value v) {
  auto isPid = [](Value v) {
    auto pid = v.getDefiningOp<triton::GetProgramIdOp>();
    return pid && pid.getAxis() == triton::ProgramIDDim::X;
  };

  if (isPid(v))
    return 1;
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    if (isPid(mul.getLhs()))
      return matchConstantInt(mul.getRhs());
    if (isPid(mul.getRhs()))
      return matchConstantInt(mul.getLhs());
  }
  return std::nullopt;
}

} // namespace

/// Addition is commutative and so is the multiply inside the program-id term, so
/// all of `B + pid*S`, `pid*S + B`, `S*pid + B` and a bare `pid` match. A missing
/// base is 0 and a missing stride is 0, which is what makes `(base, stride)` a
/// description of the address SET rather than of the syntax that spelled it.
///
/// CONSERVATIVE, in one direction worth naming. An expression with two
/// program-id terms -- `pid*256 + pid*512` -- describes an admissible set,
/// `pid*768`, and is refused here only because this does not sum strides. The
/// hazard the refusal avoids is the opposite mistake, matching one term and
/// silently ignoring the other, which would leave the pin occupying a range
/// nobody computed; summing would avoid it too. The frontend folds `tl.constexpr`
/// coefficients in Python, so `BASE + pid*STRIDE` arrives as exactly one
/// multiply and one add and nothing in tree spells two terms -- which is why the
/// cheaper of the two is the one implemented.
LogicalResult matchPinAddress(Value addr, int64_t &base, int64_t &stride) {
  if (!addr)
    return failure();

  if (auto c = matchConstantInt(addr)) {
    base = *c;
    stride = 0;
    return success();
  }
  if (auto s = matchProgramIdTerm(addr)) {
    base = 0;
    stride = *s;
    return success();
  }
  if (auto add = addr.getDefiningOp<arith::AddIOp>()) {
    Value lhs = add.getLhs(), rhs = add.getRhs();
    for (auto [constSide, pidSide] : {std::pair<Value, Value>{lhs, rhs},
                                      std::pair<Value, Value>{rhs, lhs}}) {
      auto c = matchConstantInt(constSide);
      if (!c)
        continue;
      if (auto s = matchProgramIdTerm(pidSide)) {
        base = *c;
        stride = *s;
        return success();
      }
    }
  }
  return failure();
}

LogicalResult PinOp::verify() {
  // (1) The memory space, spelled as a string because this dialect defines no
  // attribute type. Checking it against ktdp's enum is what makes the string as
  // safe as the attribute would have been, and the lowering builds the real
  // `#ktdp.memory_space` from the same symbolization -- but only ONE of the two
  // kinds is admitted.
  StringRef space = getMemorySpace();
  auto kind = mlir::ktdp::symbolizeMemorySpaceKind(space);
  if (!kind)
    return emitOpError() << "unknown memory space '" << space
                         << "': expected 'ct_local'";

  // (2) `global` is refused rather than admitted and then found unplaceable.
  // lx-placement.md's first assumption puts HBM intermediates outside a pin
  // altogether: an author who wants one "writes a tl.make_tensor_descriptor for
  // it, with a tl.spyre_tensor_layout, and an explicit store and load", which
  // reaches a binary today. Nothing in this tree allocates an anonymous global
  // buffer, so admitting the kind here would only make a surface that never
  // compiles -- and would say so with a diagnostic about the missing ADDRESS,
  // which is not what such an author got wrong.
  //
  // The parameter stays a memory space rather than collapsing into the op's
  // name, so that the surface does not change shape if `global` ever becomes
  // placeable.
  if (*kind != mlir::ktdp::MemorySpaceKind::ct_local)
    return emitOpError()
           << "memory space '" << space
           << "' cannot be pinned: only 'ct_local' is, since an intermediate in "
              "HBM is written as a descriptor with an explicit store and load";

  // A block argument is deliberately NOT refused. A `tensor` value is a value
  // whether an op or a block argument defines it -- the same state a
  // `ktdp.load` result is in -- so pinning one is the same request, and nothing
  // here reads the producing op (the store goes at the pin, and dominance is
  // measured against it). The "only a value the author named" rule the design
  // states is enforced by the surface, where an anonymous subexpression simply
  // has no name to pass, and needs no restatement as an op rule.

  // (3) The address expression's shape. Rejected here rather than in the
  // frontend because this is where the expression is: the frontend hands over a
  // value it has already built, and a module parsed from text never went through
  // the frontend at all.
  if (getAddress()) {
    int64_t base = 0, stride = 0;
    if (failed(matchPinAddress(getAddress(), base, stride)))
      return emitOpError()
             << "address must be a constant or `base + tl.program_id(0) * "
                "stride` with constant coefficients; anything else leaves the "
                "pinned range unenumerable, so capacity and disjointness "
                "cannot be checked";
  }

  return success();
}

} // namespace mlir::triton::tts
