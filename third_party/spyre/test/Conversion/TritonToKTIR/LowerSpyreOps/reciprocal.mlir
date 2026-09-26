// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s

// `arith.divf` has two targets, and which one it gets turns on the NUMERATOR.
// A numerator of one becomes the UNARY spyreop.reciprocal, taking its float
// immediate out of the program entirely; every other numerator keeps the binary
// spyreop.realdiv. The match is by value, not by "there is a constant on the
// left", and the rest of this file holds that line.
//
// The inputs are TENSORS because this pass now runs before
// ConvertElementwiseToLinalg, so a divide reaches it tensor-typed. Each case
// therefore checks two things at once: that the right intrinsic was chosen, and
// that it came out wrapped in a linalg.generic whose body holds the scalar op --
// which is the only shape a spyreop intrinsic can take, its operands being
// declared scalar-only.
//
// The numerator match is unchanged by that move: m_OneFloat accepts a splat as
// well as a scalar constant, so `dense<1.0>` matches exactly as `1.0` did.

// A splat `1.0 / x` at f16 -> spyreop.reciprocal, and the splat goes with it: the
// divide was its only reader, so nothing dead is left behind.
// CHECK-LABEL:   tt.func @recip_f16(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK-NOT:       arith.constant
// CHECK-NOT:       spyreop.realdiv
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {{.*}} ins(%[[VAL_0]] : tensor<4xf16>) outs(%[[VAL_1]] : tensor<4xf16>)
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             %[[VAL_5:.*]] = spyreop.reciprocal %[[VAL_3]] : f16
// CHECK:             linalg.yield %[[VAL_5]] : f16
// CHECK:           tt.return %[[VAL_2]] : tensor<4xf16>
tt.func @recip_f16(%x: tensor<4xf16>) -> tensor<4xf16> {
  %one = arith.constant dense<1.0> : tensor<4xf16>
  %0 = arith.divf %one, %x : tensor<4xf16>
  tt.return %0 : tensor<4xf16>
}

// -----

// Same at f32: the rewrite does not branch on the float width.
// CHECK-LABEL:   tt.func @recip_f32(
// CHECK-NOT:       arith.constant
// CHECK-NOT:       spyreop.realdiv
// CHECK:           spyreop.reciprocal {{.*}} : f32
tt.func @recip_f32(%x: tensor<4xf32>) -> tensor<4xf32> {
  %one = arith.constant dense<1.0> : tensor<4xf32>
  %0 = arith.divf %one, %x : tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// Rank 2, to hold the line that nothing about the shape is assumed. The wrapper
// derives the loop count from the operand's rank, so a 2D divide gets two
// iterator types and rank-2 identity maps rather than the rank-1 form above.
// CHECK-LABEL:   tt.func @recip_f16_rank2(
// CHECK-NOT:       arith.constant
// CHECK-NOT:       spyreop.realdiv
// CHECK:           tensor.empty() : tensor<4x1xf16>
// CHECK:           linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
// CHECK:             spyreop.reciprocal {{.*}} : f16
tt.func @recip_f16_rank2(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
  %one = arith.constant dense<1.0> : tensor<4x1xf16>
  %0 = arith.divf %one, %t : tensor<4x1xf16>
  tt.return %0 : tensor<4x1xf16>
}

// -----

// A SCALAR divide inside a hand-written linalg.generic is NOT converted, and that
// is a deliberate narrowing recorded here rather than a gap left silent.
//
// This pass used to convert exactly this shape, because it ran after
// ConvertElementwiseToLinalg and every op it saw was a scalar in a generic body.
// Running before that pass, it matches tensors instead, so a scalar op inside a
// generic no longer matches anything.
//
// Why that is acceptable: nothing in the pipeline produces this shape at this
// point. Measured across all 34 device fixtures' `ktir` artifacts -- the exact IR
// this pass now receives -- ZERO contain any linalg.generic, because every generic
// in the pipeline is created by ConvertElementwiseToLinalg, which now runs later.
//
// The case is kept rather than deleted so that the narrowing is pinned: if some
// future pass starts handing this one a generic, this test says the gap is known
// and where to look.
// CHECK-LABEL:   tt.func @recip_scalar_in_generic_survives(
// CHECK:           arith.constant
// CHECK:           arith.divf {{.*}} : f16
// CHECK-NOT:       spyreop.reciprocal
tt.func @recip_scalar_in_generic_survives(%t: tensor<4x1xf16>) -> tensor<4x1xf16> {
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
  tt.return %0 : tensor<4x1xf16>
}

// -----

// A constant numerator that is NOT one keeps spyreop.realdiv, splat and all.
// CHECK-LABEL:   tt.func @realdiv_two_over_x_f16(
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf16>
// CHECK:           spyreop.realdiv {{.*}} : f16
// CHECK-NOT:       spyreop.reciprocal
tt.func @realdiv_two_over_x_f16(%x: tensor<4xf16>) -> tensor<4xf16> {
  %two = arith.constant dense<2.0> : tensor<4xf16>
  %0 = arith.divf %two, %x : tensor<4xf16>
  tt.return %0 : tensor<4xf16>
}

// -----

// A non-constant numerator keeps spyreop.realdiv too. Nothing to match, so
// nothing to remove.
// CHECK-LABEL:   tt.func @realdiv_var_over_x_f16(
// CHECK:           spyreop.realdiv {{.*}} : f16
// CHECK-NOT:       spyreop.reciprocal
tt.func @realdiv_var_over_x_f16(%a: tensor<4xf16>, %b: tensor<4xf16>) -> tensor<4xf16> {
  %0 = arith.divf %a, %b : tensor<4xf16>
  tt.return %0 : tensor<4xf16>
}

// -----

// The match is on the numerator specifically, not on "an operand is constant":
// a constant DENOMINATOR still needs a binary op, and gets one. (`x / 1.0` is
// not the test for this -- arith's own folder rewrites that to `x` before any
// pattern here sees it -- so the constant is 2.0.)
// CHECK-LABEL:   tt.func @realdiv_x_over_two_f16(
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf16>
// CHECK:           spyreop.realdiv {{.*}} : f16
// CHECK-NOT:       spyreop.reciprocal
tt.func @realdiv_x_over_two_f16(%x: tensor<4xf16>) -> tensor<4xf16> {
  %two = arith.constant dense<2.0> : tensor<4xf16>
  %0 = arith.divf %x, %two : tensor<4xf16>
  tt.return %0 : tensor<4xf16>
}
