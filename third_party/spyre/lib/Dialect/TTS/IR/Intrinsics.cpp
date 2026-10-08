//===- Intrinsics.cpp - The spyreop intrinsics tl.spyre_op may name -------===//
//
// The table itself. Adding an intrinsic is one entry here and one fallback in
// backend/intrinsics.py, which checks at import that its names are this table's.
//
//===----------------------------------------------------------------------===//

#include "Dialect/TTS/IR/Intrinsics.h"

#include "ktir/Dialect/SpyreOp/SpyreOp.h"

#include "mlir/IR/BuiltinTypes.h"

#include <string_view>

namespace mlir::triton::tts {

namespace {

/// A unary intrinsic whose result has its operand's type.
template <typename Intrinsic>
Operation *buildUnary(OpBuilder &builder, Location loc, ValueRange operands) {
  return Intrinsic::create(builder, loc, operands[0].getType(), operands[0])
      .getOperation();
}

constexpr unsigned kResultIsOperand0[] = {0};

constexpr SpyreopIntrinsic kIntrinsics[] = {
    {"gelu", 1, kIntrinsicF16, kResultIsOperand0, /*testOnly=*/false,
     buildUnary<spyreop::GeLU>},
    {"silu", 1, kIntrinsicF16 | kIntrinsicF32, kResultIsOperand0,
     /*testOnly=*/false, buildUnary<spyreop::SiLU>},
    {"sigmoid", 1, kIntrinsicF16 | kIntrinsicF32, kResultIsOperand0,
     /*testOnly=*/false, buildUnary<spyreop::Sigmoid>},
    // A unary intrinsic for test_spyre_op.py, which binds its own fallbacks to
    // it, so that test names no real intrinsic. Selects an existing spyreop op,
    // so the spyreop dialect needs nothing for it.
    {"test_mock", 1, kIntrinsicF16 | kIntrinsicF32, kResultIsOperand0,
     /*testOnly=*/true, buildUnary<spyreop::Sigmoid>},
};

constexpr bool namesAreUnique() {
  constexpr size_t n = std::size(kIntrinsics);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j)
      if (std::string_view(kIntrinsics[i].name.data(),
                           kIntrinsics[i].name.size()) ==
          std::string_view(kIntrinsics[j].name.data(),
                           kIntrinsics[j].name.size()))
        return false;
  return true;
}
static_assert(namesAreUnique(), "two spyreop intrinsics share a name");

} // namespace

bool SpyreopIntrinsic::takesElementType(Type elementType) const {
  if (isa<Float16Type>(elementType))
    return elementTypes & kIntrinsicF16;
  if (isa<Float32Type>(elementType))
    return elementTypes & kIntrinsicF32;
  return false;
}

SmallVector<Type> SpyreopIntrinsic::resultTypes(TypeRange operandTypes) const {
  SmallVector<Type> types;
  for (unsigned operand : resultTypeOperands)
    types.push_back(operandTypes[operand]);
  return types;
}

ArrayRef<SpyreopIntrinsic> getSpyreopIntrinsics() { return kIntrinsics; }

const SpyreopIntrinsic *lookupSpyreopIntrinsic(StringRef name) {
  for (const SpyreopIntrinsic &entry : kIntrinsics)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

} // namespace mlir::triton::tts
