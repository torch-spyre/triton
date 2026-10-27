// RUN: spyre-triton-opt %s --fuse-generics -split-input-file | FileCheck %s

// GroupComputes: a consumer ins whose producer generic yields an op result, the
// anchor, is fused when every reader of it in the consumer forms, with the
// anchor, a group LowerSpyreOps lowers as one intrinsic. Today that is an
// arith.cmpf read by arith.uitofp (SelectCompare) or by arith.select as its
// condition (SelectWhere). A producer with another user is fused only when its
// body is the anchor alone. The edges this pattern declines leave an i1 between
// generics; those are in boolean-between-generics.mlir.

// -----

// cmpf then uitofp: one body, for SelectCompare.
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @cmpf_uitofp(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>, %[[VAL_1:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_2]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.cmpf ogt, %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             %[[VAL_8:.*]] = arith.uitofp %[[VAL_7]] : i1 to f16
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_3]] : tensor<8xf16>
// CHECK:         }
func.func @cmpf_uitofp(%x: tensor<8xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %v = arith.cmpf ogt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<8xi1>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %out: f16):
    %f = arith.uitofp %m : i1 to f16
    linalg.yield %f : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>

// cmpf then select: one body, for SelectWhere.
module {
// CHECK-LABEL:   func.func @cmpf_select(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>, %[[VAL_1:.*]]: tensor<8xf16>, %[[VAL_2:.*]]: tensor<8xf16>, %[[VAL_3:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<8xf16>, tensor<8xf16>, tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_4]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK:             %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_6]], %[[VAL_7]] : f16
// CHECK:             %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_9]] : f16
// CHECK:             linalg.yield %[[VAL_12]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_5]] : tensor<8xf16>
// CHECK:         }
func.func @cmpf_select(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %v = arith.cmpf olt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p, %q : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>

// A zero-test select: the splat zero is dropped and the compare grouped with
// its select.
#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @zero_test_select(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>, %[[VAL_1:.*]]: tensor<8xf16>, %[[VAL_2:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : tensor<8xf16>, tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_4]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK:             %[[VAL_10:.*]] = arith.cmpf une, %[[VAL_6]], %[[VAL_3]] : f16
// CHECK:             %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_7]], %[[VAL_8]] : f16
// CHECK:             linalg.yield %[[VAL_11]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_5]] : tensor<8xf16>
// CHECK:         }
  func.func @zero_test_select(%arg0: tensor<8xf16>, %arg1: tensor<8xf16>, %arg2: tensor<8xf16>) -> tensor<8xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<8xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<8xf16>) outs(%0 : tensor<8xi1>) {
    ^bb0(%in: f16, %out: i1):
      %4 = arith.cmpf une, %in, %cst : f16
      linalg.yield %4 : i1
    } -> tensor<8xi1>
    %2 = tensor.empty() : tensor<8xf16>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%1, %arg1, %arg2 : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%2 : tensor<8xf16>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %4 = arith.select %in, %in_0, %in_1 : f16
      linalg.yield %4 : f16
    } -> tensor<8xf16>
    return %3 : tensor<8xf16>
  }
}

// -----

// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (d0)>

// A compare-alone producer with two readers is duplicated into both, and no
// fused op keeps an extra result.
module {
// CHECK-LABEL:   func.func @lone_compare_two_readers(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>, %[[VAL_1:.*]]: tensor<8xf16>, %[[VAL_2:.*]]: tensor<8xf16>, %[[VAL_3:.*]]: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_3]], #[[$ATTR_3]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_4]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK:             %[[VAL_9:.*]] = arith.cmpf oge, %[[VAL_6]], %[[VAL_7]] : f16
// CHECK:             %[[VAL_10:.*]] = arith.uitofp %[[VAL_9]] : i1 to f16
// CHECK:             linalg.yield %[[VAL_10]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_3]], #[[$ATTR_3]], #[[$ATTR_3]], #[[$ATTR_3]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<8xf16>, tensor<8xf16>, tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_4]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: f16, %[[VAL_13:.*]]: f16, %[[VAL_14:.*]]: f16, %[[VAL_15:.*]]: f16, %[[VAL_16:.*]]: f16):
// CHECK:             %[[VAL_17:.*]] = arith.cmpf oge, %[[VAL_12]], %[[VAL_13]] : f16
// CHECK:             %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_14]], %[[VAL_15]] : f16
// CHECK:             linalg.yield %[[VAL_18]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_5]], %[[VAL_11]] : tensor<8xf16>, tensor<8xf16>
// CHECK:         }
func.func @lone_compare_two_readers(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %v = arith.cmpf oge, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  %r1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<8xi1>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %out: f16):
    %f = arith.uitofp %m : i1 to f16
    linalg.yield %f : f16
  } -> tensor<8xf16>
  %r2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p, %q : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r1, %r2 : tensor<8xf16>, tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>

// The general where: an ogt compare read by a select as its condition is a
// group (SelectWhere lowers it to spyreop.compare feeding spyreop.select), so
// it is fused even though the compare is not a zero test.
// CHECK-LABEL:   func.func @general_where_two_readers(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>, %[[VAL_1:.*]]: tensor<8xf16>, %[[VAL_2:.*]]: tensor<8xf16>, %[[VAL_3:.*]]: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<8xf16>, tensor<8xf16>, tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_4]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK:             %[[VAL_11:.*]] = arith.cmpf ogt, %[[VAL_6]], %[[VAL_7]] : f16
// CHECK:             %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_9]] : f16
// CHECK:             linalg.yield %[[VAL_12]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           %[[VAL_13:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_4]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: f16, %[[VAL_15:.*]]: f16, %[[VAL_16:.*]]: f16):
// CHECK:             %[[VAL_17:.*]] = arith.cmpf ogt, %[[VAL_14]], %[[VAL_15]] : f16
// CHECK:             %[[VAL_18:.*]] = arith.uitofp %[[VAL_17]] : i1 to f16
// CHECK:             linalg.yield %[[VAL_18]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_5]], %[[VAL_13]] : tensor<8xf16>, tensor<8xf16>
// CHECK:         }
func.func @general_where_two_readers(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %v = arith.cmpf ogt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p, %q : tensor<8xi1>, tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  %f = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c : tensor<8xi1>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %out: f16):
    %u = arith.uitofp %m : i1 to f16
    linalg.yield %u : f16
  } -> tensor<8xf16>
  return %r, %f : tensor<8xf16>, tensor<8xf16>
}

// -----

// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Not this pattern's producer: a generic that forwards an i1 (a broadcast of a
// mask) is absorbed, not grouped, and the compare behind it is then grouped
// with the select.
module {
// CHECK-LABEL:   func.func @i1_through_broadcast(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf16>, %[[VAL_1:.*]]: tensor<8xf16>, %[[VAL_2:.*]]: tensor<8x4xf16>, %[[VAL_3:.*]]: tensor<8x4xf16>) -> tensor<8x4xf16> {
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<8x4xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]], #[[$ATTR_6]], #[[$ATTR_6]], #[[$ATTR_6]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : tensor<8xf16>, tensor<8xf16>, tensor<8x4xf16>, tensor<8x4xf16>) outs(%[[VAL_4]] : tensor<8x4xf16>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK:             %[[VAL_11:.*]] = arith.cmpf ogt, %[[VAL_6]], %[[VAL_7]] : f16
// CHECK:             %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_9]] : f16
// CHECK:             linalg.yield %[[VAL_12]] : f16
// CHECK:           } -> tensor<8x4xf16>
// CHECK:           return %[[VAL_5]] : tensor<8x4xf16>
// CHECK:         }
func.func @i1_through_broadcast(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8x4xf16>, %q: tensor<8x4xf16>) -> tensor<8x4xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %v = arith.cmpf ogt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e2 = tensor.empty() : tensor<8x4xi1>
  %bc = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%c : tensor<8xi1>) outs(%e2 : tensor<8x4xi1>) {
  ^bb0(%in: i1, %out: i1):
    linalg.yield %in : i1
  } -> tensor<8x4xi1>
  %e = tensor.empty() : tensor<8x4xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%bc, %p, %q : tensor<8x4xi1>, tensor<8x4xf16>, tensor<8x4xf16>) outs(%e : tensor<8x4xf16>) {
  ^bb0(%m: i1, %a: f16, %b: f16, %out: f16):
    %s = arith.select %m, %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<8x4xf16>
  return %r : tensor<8x4xf16>
}
}
