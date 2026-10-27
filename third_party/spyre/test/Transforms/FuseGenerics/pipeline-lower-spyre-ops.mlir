// RUN: spyre-triton-opt %s --canonicalize-generic-operands --fuse-generics --lower-spyre-ops -split-input-file | FileCheck %s

// After FuseGenerics, LowerSpyreOps selects each compare with the reader
// GroupComputes grouped it with (SelectCompare, SelectWhere), and no i1 is
// left.

// A `where` with a `!= 0` mask comes out as one generic holding spyreop.select
// alone (SelectWhere).
//
// CHECK-LABEL: func.func @where_zero_test
// CHECK-NOT:     arith.cmpf
// CHECK:         linalg.generic
// CHECK:           spyreop.select
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     i1
module {
func.func @where_zero_test(%x: tensor<8xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> tensor<8xf16> {
  %z = arith.constant dense<0.0> : tensor<8xf16>
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %z : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %v = arith.cmpf une, %a, %b : f16
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

// cmpf + uitofp comes out as spyreop.compare (SelectCompare).
//
// CHECK-LABEL: func.func @mask_compare
// CHECK:         linalg.generic
// CHECK:           spyreop.compare <greaterthan>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     i1
module {
func.func @mask_compare(%x: tensor<8xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
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

// One consumer reads a compare twice, once as a select condition and once
// through uitofp, so both rules fire on one fused body.
//
// CHECK-LABEL: func.func @both_readers
// CHECK:         linalg.generic
// CHECK-DAG:       spyreop.compare <lesserthan>
// CHECK-DAG:       spyreop.select
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     i1
module {
func.func @both_readers(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8xf16>) -> tensor<8xf16> {
  %e1 = tensor.empty() : tensor<8xi1>
  %c = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x, %y : tensor<8xf16>, tensor<8xf16>) outs(%e1 : tensor<8xi1>) {
  ^bb0(%a: f16, %b: f16, %out: i1):
    %v = arith.cmpf olt, %a, %b : f16
    linalg.yield %v : i1
  } -> tensor<8xi1>
  %e = tensor.empty() : tensor<8xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%c, %p : tensor<8xi1>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%m: i1, %a: f16, %out: f16):
    %f = arith.uitofp %m : i1 to f16
    %s = arith.select %m, %a, %f : f16
    linalg.yield %s : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}

// -----

// The general where: a compare that is not a zero test, grouped with its
// select, is lowered to spyreop.compare feeding spyreop.select in one body.
//
// CHECK-LABEL: func.func @general_where
// CHECK-NOT:     arith.cmpf
// CHECK:         linalg.generic
// CHECK:           %[[C:.*]] = spyreop.compare <greaterthan>
// CHECK:           spyreop.select %[[C]]
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     i1
module {
func.func @general_where(%x: tensor<8xf16>, %y: tensor<8xf16>, %p: tensor<8xf16>, %q: tensor<8xf16>) -> tensor<8xf16> {
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
  return %r : tensor<8xf16>
}
}
