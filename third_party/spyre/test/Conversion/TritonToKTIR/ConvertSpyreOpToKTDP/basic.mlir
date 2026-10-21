// RUN: spyre-triton-opt %s --convert-spyre-op-to-ktdp | FileCheck %s

// tts.spyre_op "gelu" -> ktdp.gelu. A 1:1 rename -- no region, no fallback
// body to inline, nothing to tag.
module {
// CHECK-LABEL:   tt.func @gelu(
// CHECK-SAME:                  %[[VAL_0:.*]]: tensor<128xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_1:.*]] = ktdp.gelu %[[VAL_0]] : tensor<128xf16>
// CHECK:           tt.return %[[VAL_1]] : tensor<128xf16>
// CHECK:         }
// CHECK-NOT:     tts.spyre_op
tt.func @gelu(%x: tensor<128xf16>) -> tensor<128xf16> {
  %y = tts.spyre_op "gelu"(%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return %y : tensor<128xf16>
}
}
