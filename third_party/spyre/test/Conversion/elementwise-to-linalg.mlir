// RUN: spyre-triton-opt %s --convert-elementwise-to-linalg -split-input-file | FileCheck %s

// Tests for upstream MLIR's --convert-elementwise-to-linalg, which the Spyre
// TTIR->KTDP pipeline appends after LowerComputeOps. The pass rewrites every op
// carrying the ElementwiseMappable trait (all of arith and math) that has at
// least one ranked-tensor operand into a linalg.generic with identity indexing
// maps and the original op, now scalar, in the body.
//
// The pass itself is upstream code and is not re-tested here. These tests pin
// the two carve-outs the Spyre backend depends on: scalar-only ops are left
// alone (linalg.reduce combiner regions are built from scalar arith and would
// become malformed otherwise), and dense arith.constant is left alone (it does
// not carry the trait; rewriting it to tensor.empty + linalg.fill is deferred).

// A tensor-typed binary op becomes a linalg.generic with the op in its body.
// Tile width 7 divides no natural vector width, so a shape-assuming
// implementation would break here.

// CHECK-LABEL: tt.func @binary_f32
// CHECK-NOT:     arith.mulf %{{.*}} : tensor
// CHECK:         %[[G:.*]] = linalg.generic
// CHECK-SAME:      ins(%arg0, %arg1 : tensor<7xf32>, tensor<7xf32>)
// CHECK:         ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[R:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK:           linalg.yield %[[R]] : f32
// CHECK:         tt.return %[[G]] : tensor<7xf32>
tt.func @binary_f32(%a: tensor<7xf32>, %b: tensor<7xf32>) -> tensor<7xf32> {
  %r = arith.mulf %a, %b : tensor<7xf32>
  tt.return %r : tensor<7xf32>
}

// -----

// The math dialect is a separate dialect from arith, so it needs its own
// coverage even though the same trait drives the conversion. A width of 1 is
// the smallest legal tensor extent.

// CHECK-LABEL: tt.func @unary_math
// CHECK-NOT:     math.exp %{{.*}} : tensor
// CHECK:         linalg.generic
// CHECK:           math.exp %{{.*}} : f32
// CHECK:           linalg.yield
tt.func @unary_math(%a: tensor<1xf32>) -> tensor<1xf32> {
  %r = math.exp %a : tensor<1xf32>
  tt.return %r : tensor<1xf32>
}

// -----

// Integer ops convert too. MLIR's i32 is signless, so signed and unsigned
// division are distinct ops; andi has no named linalg equivalent, which is
// exactly why linalg.generic is the right target rather than a named op.

// CHECK-LABEL: tt.func @integer_ops
// CHECK-NOT:     arith.divsi %{{.*}} : tensor
// CHECK-NOT:     arith.divui %{{.*}} : tensor
// CHECK-NOT:     arith.andi %{{.*}} : tensor
// CHECK:         linalg.generic
// CHECK:           arith.divsi %{{.*}} : i32
// CHECK:         linalg.generic
// CHECK:           arith.divui %{{.*}} : i32
// CHECK:         linalg.generic
// CHECK:           arith.andi %{{.*}} : i32
tt.func @integer_ops(%a: tensor<3xi32>, %b: tensor<3xi32>) -> tensor<3xi32> {
  %s = arith.divsi %a, %b : tensor<3xi32>
  %u = arith.divui %s, %b : tensor<3xi32>
  %r = arith.andi %u, %b : tensor<3xi32>
  tt.return %r : tensor<3xi32>
}

// -----

// A comparison on tensor<Nxf32> yields tensor<Nxi1>. Because the result element
// type differs from the operands', the conversion cannot reuse an operand as the
// destination tensor and must create a fresh tensor.empty -- a path the
// same-element-type cases never exercise.

// CHECK-LABEL: tt.func @compare_result_type_differs
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<7xi1>
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%[[E]] : tensor<7xi1>)
// CHECK:           arith.cmpf ogt, %{{.*}}, %{{.*}} : f32
// CHECK:           linalg.yield %{{.*}} : i1
tt.func @compare_result_type_differs(%a: tensor<7xf32>, %b: tensor<7xf32>) -> tensor<7xi1> {
  %r = arith.cmpf ogt, %a, %b : tensor<7xf32>
  tt.return %r : tensor<7xi1>
}

// -----

// Carve-out 1: scalar-only ops are left at function level even when tensor
// arith converts alongside them. If the conversion keyed on the dialect instead
// of the operand types, the scalar addf would become a rank-0 linalg.generic and
// every linalg.reduce combiner region in the backend would be malformed.

// CHECK-LABEL: tt.func @scalar_operands_untouched
// CHECK:         %[[X:.*]] = arith.addf %arg0, %arg0 : f32
// CHECK-NOT:     linalg.yield %[[X]]
// CHECK:         %[[Y:.*]] = linalg.generic
// CHECK:           arith.mulf %{{.*}} : f32
// CHECK:         tt.return %[[X]], %[[Y]]
tt.func @scalar_operands_untouched(%s: f32, %t: tensor<7xf32>) -> (f32, tensor<7xf32>) {
  %x = arith.addf %s, %s : f32
  %y = arith.mulf %t, %t : tensor<7xf32>
  tt.return %x, %y : f32, tensor<7xf32>
}

// -----

// Carve-out 2: a dense arith.constant survives as a tensor-typed constant and
// is read as an input by the generic that replaced its consumer. Rewriting it
// into tensor.empty + linalg.fill is deferred; when that lands this test fails
// loudly, which marks the boundary so it cannot move silently.

// CHECK-LABEL: tt.func @dense_constant_survives
// CHECK:         %[[C:.*]] = arith.constant dense<2.000000e+00> : tensor<7xf32>
// CHECK-NOT:     linalg.fill
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg0, %[[C]] : tensor<7xf32>, tensor<7xf32>)
// CHECK:           arith.mulf %{{.*}} : f32
tt.func @dense_constant_survives(%t: tensor<7xf32>) -> tensor<7xf32> {
  %c = arith.constant dense<2.0> : tensor<7xf32>
  %r = arith.mulf %t, %c : tensor<7xf32>
  tt.return %r : tensor<7xf32>
}

// -----

// A non-splat dense constant cannot be expressed as a linalg.fill, which takes
// a single scalar. Today it passes through untouched and is not rejected.
// Guarded separately from the splat case because the eventual implementation
// will treat the two differently: splats become linalg.fill, non-splats need a
// diagnostic.

// CHECK-LABEL: tt.func @non_splat_constant_survives
// CHECK:         arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     linalg.generic
tt.func @non_splat_constant_survives() -> tensor<3xf32> {
  %c = arith.constant dense<[0.0, 1.0, 2.0]> : tensor<3xf32>
  tt.return %c : tensor<3xf32>
}

// -----

// An op with one scalar and one tensor operand still converts. tt.splat
// produces exactly this shape upstream of the conversion, so this is not a
// synthetic case.

// CHECK-LABEL: tt.func @mixed_scalar_and_tensor
// CHECK:         %[[S:.*]] = tt.splat %arg0
// CHECK-NOT:     arith.addf %{{.*}} : tensor
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%arg1, %[[S]] : tensor<64xf32>, tensor<64xf32>)
// CHECK:           arith.addf %{{.*}} : f32
tt.func @mixed_scalar_and_tensor(%s: f32, %t: tensor<64xf32>) -> tensor<64xf32> {
  %z = tt.splat %s : f32 -> tensor<64xf32>
  %r = arith.addf %t, %z : tensor<64xf32>
  tt.return %r : tensor<64xf32>
}
