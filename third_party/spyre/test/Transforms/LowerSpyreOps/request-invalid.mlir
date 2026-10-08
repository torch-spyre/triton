// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics

// A REQUEST LEFT UNSELECTED IS REFUSED. The author asked for the intrinsic by
// name with `tl.spyre_op`, so passing its fallback through to the backend would
// ignore the request without saying so. Every case is reported once, at the
// first op of the request, naming the intrinsic and why.

#map = affine_map<(d0) -> (d0)>

// SPLIT: the request spread over two bodies, which is what a fusion that did not
// happen leaves. Each half alone reads one value and yields one, so without the
// count the first half would be replaced by the whole intrinsic.
func.func @split(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_op("sigmoid") was not selected: the request's ops are spread over 2 compute bodies}}
    %n = arith.negf %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %ex = math.exp %n {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %ex : f32
  } -> tensor<64xf32>
  %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %d = arith.addf %in, %one {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    %q = arith.divf %one, %d {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %b : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A TYPE THE INTRINSIC DOES NOT TAKE: spyreop.gelu is f16 only.
func.func @gelu_f32(%x: tensor<64xf32>) -> tensor<64xf32> {
  %half = arith.constant 5.000000e-01 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_op("gelu") was not selected: spyreop.gelu does not take f32}}
    %er = math.erf %in {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32
    %m = arith.mulf %er, %half {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}} : f32
    linalg.yield %m : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A TYPE MISMATCH: the request reads f16 and yields f32, and every intrinsic
// here has one type for both.
func.func @type_mismatch(%x: tensor<64xf16>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf16>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    // expected-error @+1 {{tl.spyre_op("sigmoid") was not selected: the request takes and yields different types (f16 in, f32 out)}}
    %xf = arith.extf %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f16 to f32
    %ex = math.exp %xf {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %ex : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A PARTIAL REQUEST: the body also holds an untagged compute op, the shape a
// fold leaves when it merges an op of the request with one outside it.
func.func @partial(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_op("silu") was not selected: its body also holds 'arith.mulf', which is not part of the request}}
    %ex = math.exp %in {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32
    %m = arith.mulf %ex, %ex : f32
    linalg.yield %m : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// TWO VALUES READ: the intrinsic is unary.
func.func @binary(%x: tensor<64xf32>, %y: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%x, %y : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_op("sigmoid") was not selected: the body reads 2 block arguments, and the intrinsic takes one input}}
    %s = arith.addf %a, %b {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A NAME WITH NO INTRINSIC. Tracing refuses one, so only hand-written IR gets
// here.
func.func @unknown(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_op("softplus") was not selected: there is no spyreop intrinsic named 'softplus'}}
    %ex = math.exp %in {tts.spyreop_hint = {id = 0 : i64, name = "softplus"}} : f32
    linalg.yield %ex : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

// A CLAIMED OP OUTSIDE ANY BODY, which is also the 1:1 rules' decline made
// visible: a scalar math.exp is something SelectUnaryFloat selects wherever it
// is. Were it not declined, it would become an untagged spyreop.exp and nothing
// would be reported.
func.func @outside_body(%x: f32) -> f32 {
  // expected-error @+1 {{tl.spyre_op("sigmoid") was not selected: an op of it is outside any compute body}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
  return %e : f32
}
