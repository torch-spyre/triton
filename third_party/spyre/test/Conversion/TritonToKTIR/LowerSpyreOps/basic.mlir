// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s

// What this pass converts, and what it deliberately leaves alone.
//
// CONTRACT CHANGE, recorded because two cases in this file assert the opposite of
// what they used to. This pass now runs BEFORE ConvertElementwiseToLinalg, so it
// matches TENSOR-level math/arith and emits each intrinsic wrapped in a
// linalg.generic it builds itself. It previously ran last and matched SCALARS
// inside generic bodies. So:
//
//   then: a tensor op passed through untouched, a scalar op converted
//   now:  a tensor op converts, a scalar op passes through untouched
//
// `sqrt_tensor_survives` and the scalar conversion cases therefore invert. The
// reason for the move is that absorbing a mask cast into its comparison is only
// possible while the two are adjacent ops in one block; after scalarization they
// sit in separate generic bodies and the producer is a block argument with no
// defining op.
//
// The one case that does NOT change is `addi_i32_outside_generic_survives`. It
// asserted that scalar address arithmetic is left alone, and it still does -- the
// gate that protects it moved from "is this inside a generic?" to "is this a
// tensor?", and both answer the same question, because address math is never
// tensor-typed. That case is the most important assertion in this file.

// math.sqrt on a tensor of f32 -> spyreop.sqrt, inside a generic this pass builds.
// CHECK-LABEL:   tt.func @sqrt_f32(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {{.*}} ins(%[[VAL_0]] : tensor<4xf32>) outs(%[[VAL_1]] : tensor<4xf32>)
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = spyreop.sqrt %[[VAL_3]] : f32
// CHECK:             linalg.yield %[[VAL_5]] : f32
// CHECK:           tt.return %[[VAL_2]] : tensor<4xf32>
tt.func @sqrt_f32(%t: tensor<4xf32>) -> tensor<4xf32> {
  %0 = math.sqrt %t : tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// Same at f16: the float patterns do not branch on width.
// CHECK-LABEL:   tt.func @sqrt_f16(
// CHECK:           spyreop.sqrt {{.*}} : f16
tt.func @sqrt_f16(%t: tensor<4xf16>) -> tensor<4xf16> {
  %0 = math.sqrt %t : tensor<4xf16>
  tt.return %0 : tensor<4xf16>
}

// -----

// INVERTED from the previous contract. A SCALAR math.sqrt now survives: at this
// point in the pipeline nothing has scalarized, so a scalar float op is not
// elementwise compute and is not this pass's business. This used to be the case
// that asserted a TENSOR sqrt survives.
// CHECK-LABEL:   tt.func @sqrt_scalar_survives(
// CHECK:           math.sqrt {{.*}} : f32
// CHECK-NOT:       spyreop.sqrt
tt.func @sqrt_scalar_survives(%s: f32) -> f32 {
  %0 = math.sqrt %s : f32
  tt.return %0 : f32
}

// -----

// CHECK-LABEL:   tt.func @exp_f32(
// CHECK:           spyreop.exp {{.*}} : f32
tt.func @exp_f32(%t: tensor<4xf32>) -> tensor<4xf32> {
  %0 = math.exp %t : tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL:   tt.func @rsqrt_f32(
// CHECK:           spyreop.rsqrt {{.*}} : f32
tt.func @rsqrt_f32(%t: tensor<4xf32>) -> tensor<4xf32> {
  %0 = math.rsqrt %t : tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// arith.divf with a non-one numerator -> the binary intrinsic.
// CHECK-LABEL:   tt.func @divf_f32(
// CHECK:           spyreop.realdiv {{.*}} : f32
// CHECK-NOT:       spyreop.reciprocal
tt.func @divf_f32(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.divf %a, %b : tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// arith.divf has a SECOND target, and this file covers both so that a reader
// checking "what does this pass do" sees the split without opening another file.
// A numerator of one takes the unary path: spyreop.reciprocal, with the splat
// constant erased because the divide was its only reader.
//
// reciprocal.mlir holds the full rule -- both float widths, rank 2, a non-one
// constant numerator, a non-constant numerator, and a constant DENOMINATOR, which
// is the case that proves the match is on the numerator specifically rather than
// on "an operand is constant". This case is the smoke test for the same path.
// CHECK-LABEL:   tt.func @divf_reciprocal_f16(
// CHECK-NOT:       arith.constant
// CHECK-NOT:       spyreop.realdiv
// CHECK:           spyreop.reciprocal {{.*}} : f16
tt.func @divf_reciprocal_f16(%x: tensor<4xf16>) -> tensor<4xf16> {
  %one = arith.constant dense<1.0> : tensor<4xf16>
  %0 = arith.divf %one, %x : tensor<4xf16>
  tt.return %0 : tensor<4xf16>
}

// -----

// arith.addi needs more than a supported width: being TENSOR-typed is the signal
// that this is elementwise compute rather than address arithmetic. i32 ->
// spyreop.addi32toi32.
// CHECK-LABEL:   tt.func @addi_i32_tensor(
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           linalg.generic
// CHECK:             spyreop.addi32toi32
// CHECK:           tt.return
tt.func @addi_i32_tensor(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = arith.addi %a, %b : tensor<4xi32>
  tt.return %0 : tensor<4xi32>
}

// -----

// i64 -> spyreop.addi64toi64. addi has two widths; muli has one.
// CHECK-LABEL:   tt.func @addi_i64_tensor(
// CHECK:           spyreop.addi64toi64
tt.func @addi_i64_tensor(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<4xi64> {
  %0 = arith.addi %a, %b : tensor<4xi64>
  tt.return %0 : tensor<4xi64>
}

// -----

// An unsupported width survives rather than being reported. Unlike the float ops,
// where an unsupported element type is an error (see invalid.mlir), an integer add
// at a width spyreop has no intrinsic for is left legal -- there is no claim that
// every integer add was meant to reach the device.
// CHECK-LABEL:   tt.func @addi_i16_tensor_survives(
// CHECK:           arith.addi {{.*}} : tensor<4xi16>
// CHECK-NOT:       spyreop
tt.func @addi_i16_tensor_survives(%a: tensor<4xi16>, %b: tensor<4xi16>) -> tensor<4xi16> {
  %0 = arith.addi %a, %b : tensor<4xi16>
  tt.return %0 : tensor<4xi16>
}

// -----

// UNCHANGED, and the most important case in this file. A scalar i32 add is a loop
// index or a tile offset -- `pid * BLOCK_SIZE` and friends -- and converting it
// would turn address arithmetic into a compute intrinsic.
//
// The gate that protects this moved from "is this inside a linalg.generic?" to "is
// this a tensor?" when the pass moved. Both answer the same question and this case
// is what proves the substitution holds: address math is never tensor-typed, so a
// type test separates compute from addressing exactly as the position test did.
// CHECK-LABEL:   tt.func @addi_i32_outside_generic_survives(
// CHECK:           arith.addi {{.*}} : i32
// CHECK-NOT:       spyreop
tt.func @addi_i32_outside_generic_survives(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  tt.return %0 : i32
}

// -----

// CHECK-LABEL:   tt.func @muli_i32_tensor(
// CHECK:           spyreop.muli32toi32
tt.func @muli_i32_tensor(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
  %0 = arith.muli %a, %b : tensor<4xi32>
  tt.return %0 : tensor<4xi32>
}

// -----

// muli has an i32 intrinsic and no i64 one, so i64 survives. The asymmetry with
// addi above is spyreop's, not this pass's.
// CHECK-LABEL:   tt.func @muli_i64_tensor_survives(
// CHECK:           arith.muli {{.*}} : tensor<4xi64>
// CHECK-NOT:       spyreop
tt.func @muli_i64_tensor_survives(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<4xi64> {
  %0 = arith.muli %a, %b : tensor<4xi64>
  tt.return %0 : tensor<4xi64>
}

// -----

// The muli counterpart of addi_i32_outside_generic_survives. Scalar multiply is
// how every tile offset in a kernel is computed, so this is the other half of the
// address-arithmetic guard.
// CHECK-LABEL:   tt.func @muli_i32_outside_generic_survives(
// CHECK:           arith.muli {{.*}} : i32
// CHECK-NOT:       spyreop
tt.func @muli_i32_outside_generic_survives(%a: i32, %b: i32) -> i32 {
  %0 = arith.muli %a, %b : i32
  tt.return %0 : i32
}
