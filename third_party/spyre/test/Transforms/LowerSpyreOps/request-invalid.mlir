// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics

// A HINT SURVIVING SELECTION IS REFUSED. The author asked for the intrinsic by
// name with `tl.spyre_op`, so passing its fallback through to the backend would
// ignore the request without saying so. Whatever kept the call-site rule from
// firing, the error is the same, once per call site, naming the intrinsic.
//
// What makes a call site selectable is checked before this pass: its body by
// the `tts.spyre_op` verifier, its fusion into one body by
// FuseComputeAndDataMovement. So both cases are hand-written.

#map = affine_map<(d0) -> (d0)>

// TWO VALUES READ: the intrinsic is unary.
func.func @binary(%x: tensor<64xf32>, %y: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%x, %y : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_op("sigmoid") was not selected. The request is explicit, so its fallback is not used instead}}
    %s = arith.addf %a, %b {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

// A HINTED OP OUTSIDE ANY BODY, which is also the 1:1 rules' decline made
// visible: a scalar math.exp is something SelectUnaryFloat selects wherever it
// is. Were it not declined, it would become an unhinted spyreop.exp and nothing
// would be reported.
func.func @outside_body(%x: f32) -> f32 {
  // expected-error @+1 {{tl.spyre_op("sigmoid") was not selected}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
  return %e : f32
}
