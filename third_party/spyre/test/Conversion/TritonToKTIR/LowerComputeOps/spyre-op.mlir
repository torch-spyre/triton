// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops | FileCheck %s

// Tests for --lower-compute-ops on tts.spyre_op -> linalg.generic { spyreop.* }.
// See spyreop-frontend-api.md for the design this implements directly, with no
// decomposition and no later re-matching pass: the spyreop.* op this produces is
// already final.
//
// Two shapes, one case each:
//
//   D1 (elementwise) -- gelu_elementwise. Identity indexing maps on both the
//   one input and the one output; the real op is the whole generic body.
//   softplus's attrs and layernormnorm's five operands are the same pattern
//   with more operands/attrs on the generic's ins; nothing about the shape
//   changes, so this one case covers that pattern too.
//
//   D2 (reduction-shaped, "exx2" only) -- exx2_reduction. A reduction
//   iterator over axis, TWO results, and -- the one property worth a
//   dedicated negative check below -- NO linalg.fill on either init. This is
//   the direct test for the DropReductionInitFill shape-mismatch fix: see
//   LowerComputeOps.cpp's Group D header comment for why a linalg.fill-built
//   init here would be silently left behind by that pass and then rejected
//   downstream.

// -----

// CHECK-LABEL:   tt.func @gelu_elementwise(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<128xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0:.+]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<128xf16>) outs(%[[VAL_1]] : tensor<128xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             %[[VAL_5:.*]] = spyreop.gelu %[[VAL_3]] : f16
// CHECK:             linalg.yield %[[VAL_5]] : f16
// CHECK:           } -> tensor<128xf16>
// CHECK-NOT:       tts.spyre_op
// CHECK:           tt.return %[[VAL_2]] : tensor<128xf16>
// CHECK:         }
tt.func @gelu_elementwise(%x: tensor<128xf16>) -> tensor<128xf16> {
  %y = tts.spyre_op {name = "gelu"} (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}

// -----

// CHECK-LABEL:   tt.func @exx2_reduction(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>) {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4xf16>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4xf16>
// CHECK:           %[[VAL_3:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_1:.+]], #[[$ATTR_2:.+]], #[[$ATTR_2]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_0]] : tensor<4x8xf16>) outs(%[[VAL_1]], %[[VAL_2]] : tensor<4xf16>, tensor<4xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]], %[[VAL_8:.*]] = spyreop.exx2 %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_7]], %[[VAL_8]] : f16, f16
// CHECK:           } -> (tensor<4xf16>, tensor<4xf16>)
// CHECK-NOT:       linalg.fill
// CHECK-NOT:       tts.spyre_op
// CHECK:           tt.return %[[VAL_3]]#0, %[[VAL_3]]#1 : tensor<4xf16>, tensor<4xf16>
// CHECK:         }
tt.func @exx2_reduction(%x: tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>) {
  %mean, %msq = tts.spyre_op {name = "exx2", axis = 1 : i32} (%x)
      : (tensor<4x8xf16>) -> (tensor<4xf16>, tensor<4xf16>)
  tt.return %mean, %msq : tensor<4xf16>, tensor<4xf16>
}
