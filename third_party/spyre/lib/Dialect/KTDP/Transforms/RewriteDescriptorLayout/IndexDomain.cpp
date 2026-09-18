//===- IndexDomain.cpp ----------------------------------------------------===//
//
// Re-emitting a descriptor subscript with every step performed in `index`.
//
// Triton computes scalar offsets in i32; the KTDP coordinate split is built
// natively in `index`. The two meet at a cast, so `pid * 64` arrives as
//
//     %0 = tt.get_program_id x            : i32
//     %1 = arith.muli %0, %c64_i32        : i32      <-- multiply in i32
//     %2 = arith.index_cast %1 : i32 to index        <-- opaque to the scheduler
//     %3 = arith.divsi %2, %c64           : index    <-- split already native
//
// The scheduler's symbolic start-address analysis reads a closed set of binary
// `index` arithmetic and treats the cast as opaque, so it cannot see that %2 is
// a grid coordinate times a tile size and rejects the address as run-time
// varying. Re-emitting the multiply in `index` removes the cast from between
// the split and the values it derives from:
//
//     %0 = ktdp.get_compute_tile_id       : index
//     %1 = arith.muli %0, %c64            : index
//     %2 = arith.divsi %1, %c64           : index
//
//===----------------------------------------------------------------------===//

#include "RewriteDescriptorLayout/IndexDomain.h"
#include "Dialect/KTDP/Utils/Utility.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::triton::ktdp {

bool isIdentityTracingIntCast(Operation *op) {
  return isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::TruncIOp,
             arith::ExtSIOp, arith::ExtUIOp>(op);
}

// Narrower than isIdentityTracingIntCast, on two counts that are the whole
// reason these are two functions rather than one.
//
// arith.trunci is excluded. It drops high bits by definition, so walking
// through it would let the rebuild discard the truncation itself:
//
//     %1 = arith.muli %pid64, %c64_i64 : i64   // may exceed i32
//     %2 = arith.trunci %1 : i64 to i32        // caller asked for the low bits
//
// rebuilt through the trunc gives `muli(%pid, 64) : index` -- the untruncated
// product, which selects a different tile than %2 named.
//
// arith.extui / arith.index_castui are excluded because the arithmetic
// re-emitted around them is signed. A zero-extended negative i32 is a large
// positive i64, so the two readings disagree:
//
//     %1 = arith.muli %pid, %c-3_i32 : i32     // negative
//     %2 = arith.extui %1 : i32 to i64         // large positive
//     %3 = arith.divsi %2, %c64_i64 : i64      // ~0x3FFFFFFF
//
// rebuilt as `divsi(muli(%pid, -3), 64) : index`, which stays negative and
// divides to a small negative instead.
bool isValuePreservingIntCast(Operation *op) {
  return isa<arith::IndexCastOp, arith::ExtSIOp>(op);
}

BlockArgument traceToMLIRBlockArg(Value v) {
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
      if (op->getNumOperands() == 2 && getConstantInt(op->getOperand(1)))
        { v = op->getOperand(0); continue; }
    }
    return nullptr;
  }
}

namespace {

/// Any integer arithmetic in the arith dialect, whether or not this file can
/// re-emit it. Used to tell "arithmetic happened at a fixed width" from "this
/// is a bare root or cast" -- a separate question from whether lifting is safe.
bool isArithIntOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::AndIOp,
             arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
             arith::ShRUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
             arith::MinUIOp>(op);
}

/// The arithmetic emitInIndexDomain knows how to re-emit. Deliberately not
/// shared with traceToMLIRBlockArg's list: that walk drops a constant RHS
/// assuming scaleDownIVMuls compensates, so widening its op set would change
/// which loops get rescaled.
bool isRebuildableIntArith(Operation *op) {
  return isa<arith::MulIOp, arith::DivSIOp, arith::RemSIOp, arith::AddIOp,
             arith::SubIOp>(op);
}

/// True if `v`'s expression can be re-emitted in `index` without changing the
/// number it denotes: every leaf an integer constant or a grid-coordinate
/// query, joined only by value-preserving casts and rebuildable arithmetic.
///
/// This is the safety argument. `index` is 64-bit, so re-emitting an i32
/// expression there drops the modular reduction i32 would have applied:
///
///     pid = 0x0400'0000, BLOCK = 64
///     i32:   pid * 64  ==  0            (0x1'0000'0000 truncated to 32 bits)
///     index: pid * 64  ==  0x1'0000'0000
///
/// Those name different tiles. Which one the kernel meant is only decidable
/// from provenance: for a grid coordinate scaled by a tile size the wide
/// reading is the intended one, because the product is bounded by the grid and
/// the wrap was never wanted. For a genuine run-time i32 -- a kernel argument,
/// or a dimension read from memory as in the *_scalar_dim fixtures -- the i32
/// reading is what the program specified and must be preserved.
///
/// Note this is not the question "does the value fit in i32". A run-time i32
/// argument fits trivially, and lifting it is exactly what must not happen.
bool canRebuildInIndexDomain(Value v) {
  if (getConstantInt(v))
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
  // Every rebuildable op is binary today and emitInIndexDomain rebuilds them as
  // binary. Bail rather than read past the end if that list grows a unary or
  // variadic member -- traceToMLIRBlockArg guards this the same way.
  if (op->getNumOperands() != 2)
    return false;
  return canRebuildInIndexDomain(op->getOperand(0)) &&
         canRebuildInIndexDomain(op->getOperand(1));
}

/// True if `v` is computed by arithmetic performed outside `index`, i.e. there
/// is something for the rebuild to remove. A cast with no arithmetic under it
/// does not count: the scheduler already reads through a cast of a constant,
/// and through a cast pair with nothing between them, so rebuilding those
/// churns fixtures for no benefit.
bool hasFixedWidthIntArith(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return false;
  if (isValuePreservingIntCast(op))
    return hasFixedWidthIntArith(op->getOperand(0));
  // Any fixed-width integer arithmetic counts here, including ops this file
  // cannot re-emit -- arith.shli, which is what `pid * BLOCK` becomes after
  // strength reduction. Testing rebuildability first would answer "no
  // fixed-width arithmetic" for those and skip a subscript that does need
  // lifting; canRebuildInIndexDomain is what decides whether lifting is safe,
  // and it still rejects them. A non-arithmetic root must answer false: that
  // is the bare-cast case above.
  if (!isArithIntOp(op))
    return false;
  if (!v.getType().isIndex())
    return true;
  if (!isRebuildableIntArith(op) || op->getNumOperands() != 2)
    return false;
  return hasFixedWidthIntArith(op->getOperand(0)) ||
         hasFixedWidthIntArith(op->getOperand(1));
}

/// Emit `v`'s expression with every step performed in `index`.
/// Requires canRebuildInIndexDomain(v).
Value emitInIndexDomain(OpBuilder &b, Location loc, Value v) {
  if (auto cst = getConstantInt(v)) {
    if (v.getType().isIndex())
      return v;
    return arith::ConstantOp::create(b, loc, b.getIndexAttr(*cst)).getResult();
  }

  Operation *op = v.getDefiningOp();

  // A grid coordinate is index-valued at the source; tt.get_program_id only
  // spells it as i32 because Triton has no index type. DistributeWork has not
  // run yet here, so both spellings occur. Casting the i32 form back leaves an
  // adjacent cast pair once DistributeWork rewrites the query to
  // index_cast(ktdp.get_compute_tile_id), which the following canonicalizer
  // folds away.
  if (isa<mlir::ktdp::GetComputeTileIdOp, triton::GetProgramIdOp>(op)) {
    if (v.getType().isIndex())
      return v;
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v).getResult();
  }

  if (isValuePreservingIntCast(op))
    return emitInIndexDomain(b, loc, op->getOperand(0));

  Value lhs = emitInIndexDomain(b, loc, op->getOperand(0));
  Value rhs = emitInIndexDomain(b, loc, op->getOperand(1));

  // Already native `index` arithmetic over the same operands -- reuse it
  // rather than emitting a duplicate chain.
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

} // namespace

// Both predicates run before anything is emitted. An earlier version recursed
// and gave up partway, leaving dead arith.constant ops behind when it
// materialised one operand before finding the other was a run-time i32. Now a
// kernel this cannot help gets byte-identical IR.
Value rebuildInIndexDomain(OpBuilder &b, Location loc, Value v) {
  if (!hasFixedWidthIntArith(v) || !canRebuildInIndexDomain(v))
    return v;
  return emitInIndexDomain(b, loc, v);
}

} // namespace mlir::triton::ktdp
