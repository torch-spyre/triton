// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s

// THE REQUEST RULE: a compute body whose ops, constants aside, all carry one
// `tts.spyreop_hint` tag is one `tl.spyre_op` request's whole fallback, and becomes the
// intrinsic the tag names applied to the body's input.
//
// Each body below is the fallback the backend registers for that intrinsic, as
// FuseComputeAndDataMovement leaves it: one generic, its scalar constants
// captured from above and untagged. The fallback is replaced WHOLE -- the casts
// around an f16 request included -- so what survives is the intrinsic at the
// type the body reads and yields.
//
// It also shows the 1:1 rules declining a claimed op: each body holds a
// `math.exp` or an `arith.divf`, which would otherwise become spyreop.exp or
// spyreop.realdiv. None is left, untagged or otherwise.

#map = affine_map<(d0) -> (d0)>

// gelu, f16: 0.5 * x * (1 + erf(x / sqrt(2))), computed in f32.
// CHECK-LABEL: func.func @gelu_f16(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:      %[[R:.*]] = spyreop.gelu %[[IN]] : f16
// CHECK-NEXT:      linalg.yield %[[R]] : f16
// CHECK-NOT:     tts.spyreop_hint
func.func @gelu_f16(%x: tensor<64xf16>) -> tensor<64xf16> {
  %half = arith.constant 5.000000e-01 : f32
  %one = arith.constant 1.000000e+00 : f32
  %rsqrt2 = arith.constant 0.707106769 : f32
  %e = tensor.empty() : tensor<64xf16>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf16>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %xf = arith.extf %in {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f16 to f32
    %h = arith.mulf %xf, %half {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32
    %s = arith.mulf %xf, %rsqrt2 {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32
    %er = math.erf %s {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32
    %p = arith.addf %er, %one {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32
    %m = arith.mulf %h, %p {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32
    %t = arith.truncf %m {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

#map = affine_map<(d0) -> (d0)>

// silu, f32: x / (1 + exp(-x)). No casts at f32; the input is read twice.
// CHECK-LABEL: func.func @silu_f32(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.silu %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @silu_f32(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in {tts.spyreop_hint = {id = 4 : i64, name = "silu"}} : f32
    %ex = math.exp %n {tts.spyreop_hint = {id = 4 : i64, name = "silu"}} : f32
    %d = arith.addf %ex, %one {tts.spyreop_hint = {id = 4 : i64, name = "silu"}} : f32
    %q = arith.divf %in, %d {tts.spyreop_hint = {id = 4 : i64, name = "silu"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// silu, f16: the same fallback widened, with the widening duplicated by fusion.
// CHECK-LABEL: func.func @silu_f16(
// CHECK:         spyreop.silu %{{.*}} : f16
// CHECK-NOT:     arith.extf
func.func @silu_f16(%x: tensor<64xf16>) -> tensor<64xf16> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf16>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf16>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %xf = arith.extf %in {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f16 to f32
    %n = arith.negf %xf {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32
    %ex = math.exp %n {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32
    %d = arith.addf %ex, %one {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32
    %xf2 = arith.extf %in {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f16 to f32
    %q = arith.divf %xf2, %d {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32
    %t = arith.truncf %q {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

#map = affine_map<(d0) -> (d0)>

// sigmoid, f32: 1 / (1 + exp(-x)). The divide's numerator is a one, which the
// reciprocal rule would otherwise take.
// CHECK-LABEL: func.func @sigmoid_f32(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.sigmoid %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @sigmoid_f32(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %ex = math.exp %n {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %d = arith.addf %ex, %one {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %q = arith.divf %one, %d {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// sigmoid, f16, with its constant still a splat `ins` operand -- the form
// before the fusion folds it. The block argument reading a constant is neutral
// like the constant, so the body still reads one value; the unused operand is
// then erased.
// CHECK-LABEL: func.func @sigmoid_f16_splat_operand(
// CHECK:         arith.constant dense<1.000000e+00> : tensor<64xf32>
// CHECK:         linalg.generic {{.*}} ins(%{{.*}} : tensor<64xf16>)
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:      %[[R:.*]] = spyreop.sigmoid %[[IN]] : f16
// CHECK-NEXT:      linalg.yield %[[R]] : f16
// CHECK-NOT:     tts.spyreop_hint
func.func @sigmoid_f16_splat_operand(%x: tensor<64xf16>) -> (tensor<64xf16>, tensor<64xf32>) {
  %one = arith.constant dense<1.0> : tensor<64xf32>
  %e = tensor.empty() : tensor<64xf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%x, %one : tensor<64xf16>, tensor<64xf32>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %c: f32, %out: f16):
    %xf = arith.extf %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f16 to f32
    %n = arith.negf %xf {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %ex = math.exp %n {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %d = arith.addf %ex, %c {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %q = arith.divf %c, %d {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %t = arith.truncf %q {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r, %one : tensor<64xf16>, tensor<64xf32>
}
