// RUN: spyre-triton-opt %s --normalize-float-min-max -split-input-file | FileCheck %s

// NormalizeFloatMinMax rewrites arith.maxnumf -> arith.maximumf and
// arith.minnumf -> arith.minimumf, module-wide.
//
// The rewrite is a substitution of the op name; operands and the fastmath
// attribute ride along. What makes it worth a pass is where the ops turn up:
// Triton emits the `numf` spelling from tl.max/tl.min (a reduce combiner) and
// from tl.maximum/tl.minimum (elementwise), and the scheduler mis-lowers it in
// both places -- maxnumf to vectorchain `abs_max`, which compares magnitudes, and
// minnumf to nothing at all, since its neutral-element table has no entry. So the
// two shapes below are the two the pipeline actually produces, not variations for
// their own sake.
//
// It is a NaN behaviour change, not a spelling fix; see the pass description.

// Test 1: a reduce combiner -- what tl.max lowers to. The neutral element on the
// init is left exactly as it was: this pass has no opinion about it, and
// DropReductionInitFill runs next and does.
module {
// CHECK-LABEL:   func.func @max_reduce_combiner(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<2x64x64xf32>) -> tensor<2x64xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<2x64xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : f32) outs(%[[VAL_2]] : tensor<2x64xf32>) -> tensor<2x64xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_0]] : tensor<2x64x64xf32>) outs(%[[VAL_3]] : tensor<2x64xf32>) dimensions = [1]
// CHECK:             (%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32) {
// CHECK:               %[[VAL_7:.*]] = arith.maximumf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:               linalg.yield %[[VAL_7]] : f32
// CHECK:             }
// CHECK:           return %[[VAL_4]] : tensor<2x64xf32>
// CHECK:         }
func.func @max_reduce_combiner(%a: tensor<2x64x64xf32>) -> tensor<2x64xf32> {
  %neg_inf = arith.constant 0xFF800000 : f32
  %empty = tensor.empty() : tensor<2x64xf32>
  %init = linalg.fill ins(%neg_inf : f32) outs(%empty : tensor<2x64xf32>) -> tensor<2x64xf32>
  %r = linalg.reduce ins(%a : tensor<2x64x64xf32>) outs(%init : tensor<2x64xf32>) dimensions = [1]
    (%in: f32, %acc: f32) {
      %m = arith.maxnumf %in, %acc : f32
      linalg.yield %m : f32
    }
  return %r : tensor<2x64xf32>
}
}

// -----

// Test 2: an ELEMENTWISE min inside a linalg.generic body -- what tl.minimum
// lowers to, and the case a fix confined to the reduce path would miss. The
// fastmath attribute is carried across, which is why this one states one.
module {
// CHECK-LABEL:   func.func @min_elementwise_with_fastmath(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x64xf32>, %[[VAL_1:.*]]: tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x64xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.generic
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.minimumf %[[VAL_4]], %[[VAL_5]] fastmath<fast> : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           return %[[VAL_3]] : tensor<64x64xf32>
// CHECK:         }
func.func @min_elementwise_with_fastmath(%x: tensor<64x64xf32>, %y: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %e = tensor.empty() : tensor<64x64xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x, %y : tensor<64x64xf32>, tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %m = arith.minnumf %a, %b fastmath<fast> : f32
    linalg.yield %m : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}
}

// -----

// Test 3: the ops that must NOT move. arith.maximumf/minimumf are already the
// target spelling and are left as they are (the pass must not oscillate, since it
// runs once inside a pipeline and idempotence is what makes that safe), and the
// INTEGER min/max ops are a different family entirely -- there is no NaN and
// nothing to normalise.
module {
// CHECK-LABEL:   func.func @already_normalized_and_integers(
// CHECK:           arith.maximumf
// CHECK:           arith.minimumf
// CHECK:           arith.maxsi
// CHECK:           arith.minui
// CHECK-NOT:       arith.maxnumf
// CHECK-NOT:       arith.minnumf
func.func @already_normalized_and_integers(%f0: f32, %f1: f32, %i0: i32, %i1: i32)
    -> (f32, f32, i32, i32) {
  %a = arith.maximumf %f0, %f1 : f32
  %b = arith.minimumf %f0, %f1 : f32
  %c = arith.maxsi %i0, %i1 : i32
  %d = arith.minui %i0, %i1 : i32
  return %a, %b, %c, %d : f32, f32, i32, i32
}
}
