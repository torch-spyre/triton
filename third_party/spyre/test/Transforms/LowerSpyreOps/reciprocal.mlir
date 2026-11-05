// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s

// `arith.divf` with a constant 1.0 numerator -> `spyreop.reciprocal`; any other
// numerator -> `spyreop.realdiv`. One rule, SelectArithDivF, chooses between them.
//
// The case this file exists for is the FIRST one: the `1.0` is a splat `ins`
// operand and the numerator in the body is a BLOCK ARGUMENT. Reading the operand
// through the body is what sees it, and upstream's unused-operand erasure is what
// takes the operand, its block argument and its indexing map away afterwards. So
// this rule needs nothing to have folded the constant in first -- unlike the
// compare rule, which does need its group brought into one body. This whole file
// therefore runs --lower-spyre-ops alone.
//
// `func.func`, not `tt.func`, because that is what this pass sees: it runs in the
// `spyrecode` stage, long after ConvertFunctions.

// THE SPLAT `ins` FORM -- `tl.full([M, S], 1.0)` as it actually arrives. Two `ins`
// go in and one comes out, the constant gone entirely.
// CHECK-LABEL:   func.func @recip_splat_ins(
// CHECK-SAME:  %[[T:.*]]: tensor<4x1xf16>) -> tensor<4x1xf16> {
// CHECK-NOT:       arith.constant
// CHECK-NOT:       arith.divf
// CHECK:           %[[E:.*]] = tensor.empty() : tensor<4x1xf16>
// CHECK:           linalg.generic {{.*}} ins(%[[T]] : tensor<4x1xf16>) outs(%[[E]] : tensor<4x1xf16>)
// CHECK:           ^bb0(%[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:             %[[R:.*]] = spyreop.reciprocal %[[IN]] : f16
// CHECK:             linalg.yield %[[R]] : f16
func.func @recip_splat_ins(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %splat = arith.constant dense<1.0> : tensor<4x1xf16>
  %init = tensor.empty() : tensor<4x1xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%splat, %t : tensor<4x1xf16>, tensor<4x1xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%one: f16, %in: f16, %out: f16):
    %1 = arith.divf %one, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// THE HOISTED-SCALAR FORM -- the same kernel once something has folded the splat
// into the body. Both forms reach the same result, which is what "reads the
// operand through the body" buys: there is no operand to erase here, and the rule
// does not care which case it is in.
// CHECK-LABEL:   func.func @recip_hoisted_scalar(
// CHECK-SAME:  %[[T:.*]]: tensor<4x1xf16>) -> tensor<4x1xf16> {
// CHECK-NOT:       arith.constant
// CHECK-NOT:       arith.divf
// CHECK:           ^bb0(%[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:             %[[R:.*]] = spyreop.reciprocal %[[IN]] : f16
func.func @recip_hoisted_scalar(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %one = arith.constant 1.0 : f16
  %init = tensor.empty() : tensor<4x1xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%t : tensor<4x1xf16>) outs(%init : tensor<4x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %1 = arith.divf %one, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// -----

// f32 as well as f16: the rule does not branch on the float width.
// CHECK-LABEL:   func.func @recip_f32(
// CHECK:             spyreop.reciprocal
func.func @recip_f32(%t: tensor<4xf32>) -> tensor<4xf32> {
  %splat = arith.constant dense<1.0> : tensor<4xf32>
  %init = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%splat, %t : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%one: f32, %in: f32, %out: f32):
    %1 = arith.divf %one, %in : f32
    linalg.yield %1 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A shared numerator is rewritten the same, and the constant stays because
// something still reads it. Nothing in the rule asks about use counts: the
// numerator goes, or does not go, by ordinary dead-op elimination.
// CHECK-LABEL:   func.func @recip_shared_numerator(
// CHECK-NOT:       arith.divf
// CHECK:           %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:             %[[R:.*]] = spyreop.reciprocal %[[IN:.*]] : f16
// CHECK:             arith.addf
func.func @recip_shared_numerator(%t: tensor<4xf16>) -> tensor<4xf16> {
  %one = arith.constant 1.0 : f16
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%t : tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%in: f16, %out: f16):
    %1 = arith.divf %one, %in : f16
    %2 = arith.addf %1, %one : f16
    linalg.yield %2 : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// The same divide outside any generic body is a reciprocal too: the numerator is
// read directly when there is no body to read it through.
// CHECK-LABEL:   func.func @recip_outside_a_body(
// CHECK-NOT:       arith.constant
// CHECK-NOT:       spyreop.realdiv
// CHECK:           %[[R:.*]] = spyreop.reciprocal %{{.*}} : f16
func.func @recip_outside_a_body(%x: f16) -> f16 {
  %one = arith.constant 1.0 : f16
  %0 = arith.divf %one, %x : f16
  return %0 : f16
}

// -----

// A numerator that is not one gets the BINARY intrinsic, constant and all: the
// rule matches on the VALUE, not on "there is a constant on the left".
// CHECK-LABEL:   func.func @divf_two_over_x(
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           spyreop.realdiv
func.func @divf_two_over_x(%t: tensor<4xf16>) -> tensor<4xf16> {
  %splat = arith.constant dense<2.0> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%splat, %t : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%two: f16, %in: f16, %out: f16):
    %1 = arith.divf %two, %in : f16
    linalg.yield %1 : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// The numerator specifically, not "an operand is constant": a constant
// DENOMINATOR still needs the binary op.
//
// The constant is 2.0 and not 1.0, which would have been the sharper test of
// position: with the splat folded into the body `x / 1.0` folds to `x` by arith's
// own folder before any rule sees it, and the case would pass for the wrong reason.
// CHECK-LABEL:   func.func @divf_x_over_two(
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           spyreop.realdiv
func.func @divf_x_over_two(%t: tensor<4xf16>) -> tensor<4xf16> {
  %splat = arith.constant dense<2.0> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%t, %splat : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%in: f16, %one: f16, %out: f16):
    %1 = arith.divf %in, %one : f16
    linalg.yield %1 : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// A NON-SPLAT constant tensor resolves through the body too, and correctly
// matches nothing: reading an operand through the body is sound only for a value
// uniform across it, and `m_OneFloat` asking about a splat is what keeps it so.
// Without that, element 0 being 1.0 would have been taken for the whole tensor.
// CHECK-LABEL:   func.func @divf_nonsplat_numerator(
// CHECK-NOT:       spyreop.reciprocal
// CHECK:           spyreop.realdiv
func.func @divf_nonsplat_numerator(%t: tensor<2xf16>) -> tensor<2xf16> {
  %mixed = arith.constant dense<[1.0, 3.0]> : tensor<2xf16>
  %init = tensor.empty() : tensor<2xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%mixed, %t : tensor<2xf16>, tensor<2xf16>) outs(%init : tensor<2xf16>) {
  ^bb0(%n: f16, %in: f16, %out: f16):
    %1 = arith.divf %n, %in : f16
    linalg.yield %1 : f16
  } -> tensor<2xf16>
  return %0 : tensor<2xf16>
}

// -----

// An unsupported float width has no intrinsic, so BOTH rules decline and the
// divide flows through to the backend as arith. The only case in this file where
// no spyreop op appears at all -- see pass-through.mlir for the rest.
// CHECK-LABEL:   func.func @divf_f64_declined(
// CHECK-NOT:       spyreop
// CHECK:           arith.divf
func.func @divf_f64_declined(%t: tensor<4xf64>) -> tensor<4xf64> {
  %splat = arith.constant dense<1.0> : tensor<4xf64>
  %init = tensor.empty() : tensor<4xf64>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%splat, %t : tensor<4xf64>, tensor<4xf64>) outs(%init : tensor<4xf64>) {
  ^bb0(%one: f64, %in: f64, %out: f64):
    %1 = arith.divf %one, %in : f64
    linalg.yield %1 : f64
  } -> tensor<4xf64>
  return %0 : tensor<4xf64>
}
