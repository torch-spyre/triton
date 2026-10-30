// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for the tts.spyre_op OP.
//
// Unlike tts.tensor_layout/tts.pin, this op has a real result -- it is a
// request, not a marker -- so its assembly format names the operand and
// result explicitly rather than routing everything through attr-dict.

// CHECK-LABEL: tt.func @gelu(
// CHECK: tts.spyre_op "gelu"(%{{.*}}) : (tensor<128xf16>) -> tensor<128xf16>
tt.func @gelu(%x: tensor<128xf16>) {
  %y = tts.spyre_op "gelu" (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return
}
