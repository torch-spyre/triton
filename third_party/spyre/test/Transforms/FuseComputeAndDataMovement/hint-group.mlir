// RUN: spyre-triton-opt %s -split-input-file --convert-elementwise-to-linalg --fuse-compute-and-data-movement | FileCheck %s

// The THIRD fusion clause: two generics whose bodies carry one `tts.hint` tag
// are one intrinsic request, and are fused into one body for LowerSpyreOps to
// replace whole.
//
// The input is tensor-level, as LowerTTSMarkers leaves a request, and the RUN
// line converts it first. That conversion is what puts each tagged tensor op's
// tag onto the scalar op in its generic's body, which is where this clause
// reads it -- so this file also pins that the tag survives that conversion and
// the fusion itself.

// The f16 sigmoid fallback, seven tensor ops: one generic, every compute op
// tagged, the splat constant folded in as an untagged scalar -- a constant does
// not count toward a body's tag, which is what lets this one fuse at all.
// CHECK-LABEL: func.func @one_request(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic
// CHECK:           arith.extf {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           arith.negf {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           math.exp {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           arith.addf {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           arith.divf {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           arith.truncf {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           linalg.yield
// CHECK-NOT:     linalg.generic
func.func @one_request(%x: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %one = arith.constant {tts.hint = {group = 0 : i64, hint = "sigmoid"}} dense<1.0> : tensor<64x128xf32>
  %xf = arith.extf %x {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64x128xf16> to tensor<64x128xf32>
  %n = arith.negf %xf {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64x128xf32>
  %e = math.exp %n {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64x128xf32>
  %d = arith.addf %one, %e {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64x128xf32>
  %q = arith.divf %one, %d {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64x128xf32>
  %y = arith.truncf %q {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64x128xf32> to tensor<64x128xf16>
  return %y : tensor<64x128xf16>
}

// -----

// A producer read twice by one request -- silu's `x * sigmoid(x)` reads the
// widened input in two places -- is fused into the one body, and the widening
// may be duplicated there; nothing of the request is left outside.
// CHECK-LABEL: func.func @multi_use_within_request(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic
// CHECK:           arith.divf {{.*}} {tts.hint = {group = 3 : i64, hint = "silu"}}
// CHECK:           linalg.yield
// CHECK-NOT:     linalg.generic
func.func @multi_use_within_request(%x: tensor<64xf16>) -> tensor<64xf16> {
  %one = arith.constant {tts.hint = {group = 3 : i64, hint = "silu"}} dense<1.0> : tensor<64xf32>
  %xf = arith.extf %x {tts.hint = {group = 3 : i64, hint = "silu"}} : tensor<64xf16> to tensor<64xf32>
  %n = arith.negf %xf {tts.hint = {group = 3 : i64, hint = "silu"}} : tensor<64xf32>
  %e = math.exp %n {tts.hint = {group = 3 : i64, hint = "silu"}} : tensor<64xf32>
  %d = arith.addf %one, %e {tts.hint = {group = 3 : i64, hint = "silu"}} : tensor<64xf32>
  %q = arith.divf %xf, %d {tts.hint = {group = 3 : i64, hint = "silu"}} : tensor<64xf32>
  %y = arith.truncf %q {tts.hint = {group = 3 : i64, hint = "silu"}} : tensor<64xf32> to tensor<64xf16>
  return %y : tensor<64xf16>
}

// -----

// The negatives. Two requests one after the other stay two bodies, even for the
// same intrinsic: the clause compares the WHOLE tag, group included. And an
// untagged compute between or after them fuses with neither -- two computes
// that are not one request never merge.
// CHECK-LABEL: func.func @distinct_requests(
// CHECK:         linalg.generic
// CHECK:           math.exp {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           arith.negf {{.*}} {tts.hint = {group = 0 : i64, hint = "sigmoid"}}
// CHECK:           linalg.yield
// CHECK:         linalg.generic
// CHECK:           math.exp {{.*}} {tts.hint = {group = 1 : i64, hint = "sigmoid"}}
// CHECK:           linalg.yield
// CHECK:         linalg.generic
// CHECK:           arith.mulf
// CHECK-NOT:       tts.hint
// CHECK:           linalg.yield
func.func @distinct_requests(%x: tensor<64xf32>) -> tensor<64xf32> {
  %a = math.exp %x {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64xf32>
  %b = arith.negf %a {tts.hint = {group = 0 : i64, hint = "sigmoid"}} : tensor<64xf32>
  %c = math.exp %b {tts.hint = {group = 1 : i64, hint = "sigmoid"}} : tensor<64xf32>
  %d = arith.mulf %c, %c : tensor<64xf32>
  return %d : tensor<64xf32>
}
