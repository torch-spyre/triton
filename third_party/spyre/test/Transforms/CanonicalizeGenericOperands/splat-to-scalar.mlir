// RUN: spyre-triton-opt %s --canonicalize-generic-operands -split-input-file | FileCheck %s

// SplatOperandToScalar: an ins defined by a splat dense constant becomes a
// scalar arith.constant read in the body, and the operand, its map and its
// block argument go.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @splat(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<8xf16>) outs(%[[VAL_2]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = arith.divf %[[VAL_1]], %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_3]] : tensor<8xf16>
// CHECK:         }
func.func @splat(%x: tensor<8xf16>) -> tensor<8xf16> {
  %one = arith.constant dense<1.0> : tensor<8xf16>
  %e = tensor.empty() : tensor<8xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%one, %x : tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.divf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>

// Two splats on one generic, an integer and a float, both folded.
module {
// CHECK-LABEL:   func.func @two_splats(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf32>) -> tensor<8xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 3.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<8xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<8xf32>) outs(%[[VAL_3]] : tensor<8xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_5]], %[[VAL_2]] : f32
// CHECK:             %[[VAL_8:.*]] = arith.addf %[[VAL_7]], %[[VAL_1]] : f32
// CHECK:             linalg.yield %[[VAL_8]] : f32
// CHECK:           } -> tensor<8xf32>
// CHECK:           return %[[VAL_4]] : tensor<8xf32>
// CHECK:         }
func.func @two_splats(%x: tensor<8xf32>) -> tensor<8xf32> {
  %a = arith.constant dense<2.0> : tensor<8xf32>
  %b = arith.constant dense<3.0> : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %a, %b : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%p: f32, %q: f32, %s: f32, %out: f32):
    %m = arith.mulf %p, %q : f32
    %v = arith.addf %m, %s : f32
    linalg.yield %v : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}
}

// -----

// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0)>

// Declined because a loop dim would be named by no operand map: the splat is
// the only operand naming d1 (the outs is a reduced rank-1 accumulator), so
// dropping it would leave d1 named by no map.
module {
// CHECK-LABEL:   func.func @d3_decline(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<1.000000e+00> : tensor<8x4xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_1]] : tensor<8x4xf16>) outs(%[[VAL_0]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_5]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_2]] : tensor<8xf16>
// CHECK:         }
func.func @d3_decline(%x: tensor<8xf16>) -> tensor<8xf16> {
  %c = arith.constant dense<1.0> : tensor<8x4xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%c : tensor<8x4xf16>) outs(%x : tensor<8xf16>) {
  ^bb0(%a: f16, %out: f16):
    %v = arith.addf %a, %out : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0) -> ()>

// Left alone: a non-splat constant, and a scalar constant passed as an ins.
module {
// CHECK-LABEL:   func.func @non_splat_and_scalar(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<2xf16>) -> tensor<2xf16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf16>
// CHECK:           %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f16
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<2xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_5]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : tensor<2xf16>, tensor<2xf16>, f16) outs(%[[VAL_3]] : tensor<2xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK:             %[[VAL_9:.*]] = arith.mulf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             %[[VAL_10:.*]] = arith.addf %[[VAL_9]], %[[VAL_7]] : f16
// CHECK:             linalg.yield %[[VAL_10]] : f16
// CHECK:           } -> tensor<2xf16>
// CHECK:           return %[[VAL_4]] : tensor<2xf16>
// CHECK:         }
func.func @non_splat_and_scalar(%x: tensor<2xf16>) -> tensor<2xf16> {
  %c = arith.constant dense<[1.0, 2.0]> : tensor<2xf16>
  %s = arith.constant 3.0 : f16
  %e = tensor.empty() : tensor<2xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %c, %s : tensor<2xf16>, tensor<2xf16>, f16) outs(%e : tensor<2xf16>) {
  ^bb0(%a: f16, %b: f16, %k: f16, %out: f16):
    %m = arith.mulf %a, %b : f16
    %v = arith.addf %m, %k : f16
    linalg.yield %v : f16
  } -> tensor<2xf16>
  return %r : tensor<2xf16>
}
}

// -----

// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0) -> (d0, 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0) -> (d0)>

// Not this pattern's producers: a forwarding generic and a reshape operand on
// the same consumer are not dropped. The reshape becomes a generic (the other
// pattern); the consumer keeps both operands.
module {
// CHECK-LABEL:   func.func @other_producers(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8x1xf16>, %[[VAL_1:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_7]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<8x1xf16>) outs(%[[VAL_2]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_4]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_7]], #[[$ATTR_7]]], iterator_types = ["parallel"]} ins(%[[VAL_1]] : tensor<8xf16>) outs(%[[VAL_6]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           %[[VAL_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_7]], #[[$ATTR_7]], #[[$ATTR_7]]], iterator_types = ["parallel"]} ins(%[[VAL_7]], %[[VAL_3]] : tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_6]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: f16, %[[VAL_12:.*]]: f16, %[[VAL_13:.*]]: f16):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : f16
// CHECK:             linalg.yield %[[VAL_14]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_10]] : tensor<8xf16>
// CHECK:         }
func.func @other_producers(%s: tensor<8x1xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<8x1xf16> into tensor<8xf16>
  %e = tensor.empty() : tensor<8xf16>
  %f = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%y : tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<8xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%f, %c : tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}
