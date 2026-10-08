//===- Intrinsics.h - The spyreop intrinsics tl.spyre_op may name ---------===//
//
// The one table of the intrinsics a `tl.spyre_op` call site may name. Read by
// the `tts.spyre_op` region verifier, the `tts.spyreop_hint` attribute
// verifier, LowerSpyreOps, and -- through the `spyre.intrinsics` pybind module
// -- the frontend, which pairs each name with its fallback.
//
// Per entry: the operand count, the element types every operand may have, the
// result rule, and the builder that selects the intrinsic. The element types are
// the spyreop op's operand constraints restricted to the builtin types this tree
// produces; DF16 is spyreop's own type, and nothing upstream of LowerSpyreOps
// makes one.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SPYRE_DIALECT_TTS_IR_INTRINSICS_H
#define TRITON_SPYRE_DIALECT_TTS_IR_INTRINSICS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::tts {

/// An element type an intrinsic operand may have, as a bit of
/// `SpyreopIntrinsic::elementTypes`.
enum IntrinsicElementType : unsigned {
  kIntrinsicF16 = 1u << 0,
  kIntrinsicF32 = 1u << 1,
};

struct SpyreopIntrinsic {
  /// The name a `tl.spyre_op` call site names, and the hint's `name`.
  llvm::StringLiteral name;
  unsigned numOperands;
  /// The `IntrinsicElementType`s every operand may have.
  unsigned elementTypes;
  /// The result rule, one entry per result: result `i` has the type of operand
  /// `resultTypeOperands[i]`. Its size is the result count.
  llvm::ArrayRef<unsigned> resultTypeOperands;
  /// Builds the intrinsic over `operands`, which are scalars of an element
  /// type the entry takes.
  Operation *(*build)(OpBuilder &, Location, ValueRange operands);

  /// Whether `elementType` is one the operands may have.
  bool takesElementType(Type elementType) const;

  /// The result types the rule gives `operandTypes`, which must number
  /// `numOperands`.
  SmallVector<Type> resultTypes(TypeRange operandTypes) const;
};

/// Every entry, in table order. Names are unique: the table is checked where it
/// is defined.
llvm::ArrayRef<SpyreopIntrinsic> getSpyreopIntrinsics();

/// The entry named `name`, or null.
const SpyreopIntrinsic *lookupSpyreopIntrinsic(llvm::StringRef name);

} // namespace mlir::triton::tts

#endif // TRITON_SPYRE_DIALECT_TTS_IR_INTRINSICS_H
