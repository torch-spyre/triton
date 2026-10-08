// RUN: spyre-triton-opt %s -split-input-file --fuse-compute-and-data-movement -verify-diagnostics

// THE THIRD CLAUSE'S POST-CONDITION: after the fixpoint every `tl.spyre_op` call
// site is one compute body holding nothing else. The `tts.spyre_op` verifier
// guarantees that its body can fuse, so each case below is hand-written IR that
// no traced kernel produces, and the error calls it a compiler bug.

#map = affine_map<(d0) -> (d0)>

// SPLIT: two bodies of one call site, neither the other's producer, so the
// clause has nothing to fuse.
func.func @split(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  // expected-note @+1 {{a body holding ops of it}}
  %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_op("sigmoid") call site 0 did not fuse into one compute body, which its verified body guarantees -- a compiler bug: its ops are spread over 2 compute bodies}}
    %n = arith.negf %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %n : f32
  } -> tensor<64xf32>
  // expected-note @+1 {{a body holding ops of it}}
  %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %ex = math.exp %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    linalg.yield %ex : f32
  } -> tensor<64xf32>
  %s = arith.addf %a, %b : tensor<64xf32>
  return %s : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// TWO IDS IN ONE BODY: the clause never fuses two call sites, so only a body
// written that way holds both.
func.func @two_ids(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    // expected-error @+1 {{call site 0 did not fuse into one compute body, which its verified body guarantees -- a compiler bug: its body also holds ops of call site 1, silu}}
    %ex = math.exp %n {tts.spyreop_hint = {id = 1 : i64, name = "silu"}} : f32
    linalg.yield %ex : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// AN UNHINTED OP IN THE BODY.
func.func @foreign_op(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %ex = math.exp %in {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32
    // expected-error @+1 {{its body also holds 'arith.mulf', which is not part of it}}
    %m = arith.mulf %ex, %ex : f32
    linalg.yield %m : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

// OUTSIDE ANY BODY: a hinted op left at tensor level.
func.func @outside_body(%x: tensor<64xf32>) -> tensor<64xf32> {
  // expected-error @+1 {{'math.exp' is outside any compute body}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : tensor<64xf32>
  return %e : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// ONE ID, TWO NAMES.
func.func @two_names(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : f32
    // expected-error @+1 {{tl.spyre_op("sigmoid") call site 0 did not fuse into one compute body, which its verified body guarantees -- a compiler bug: an op of it names 'silu'}}
    %ex = math.exp %n {tts.spyreop_hint = {id = 0 : i64, name = "silu"}} : f32
    linalg.yield %ex : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}
