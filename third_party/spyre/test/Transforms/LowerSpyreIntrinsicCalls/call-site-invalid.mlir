// RUN: spyre-triton-opt %s --lower-spyre-intrinsic-calls -split-input-file -verify-diagnostics

// A HINT SURVIVING SELECTION IS REFUSED. The author asked for the intrinsic by
// name with `tl.spyre_intrinsic`, so passing its fallback through to the backend would
// ignore the request without saying so. Whatever kept the call-site rule from
// firing, the error is the same, once per call site, naming the intrinsic.
//
// What makes a call site selectable is checked before this pass: its body by
// the `tts.spyre_intrinsic` verifier, its fusion into one body by
// FuseComputeAndDataMovement. So every case is hand-written. Where the reason
// a call site cannot be selected is known -- its `operands` bindings -- it is
// reported first, naming the call operand.

#map = affine_map<(d0) -> (d0)>

// A CALL OPERAND NO OP BINDS: the hint says the call site has two operands,
// and nothing in the body reads the second, so the target has nothing to take
// it from. The reason is reported, and the request then as unselected.
func.func @unbound_operand(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  // expected-error @+1 {{lower-spyre-intrinsic-calls: tl.spyre_intrinsic("spyreop.realdiv"): call operand 1 is read by no op left of the fallback, so there is no input to pass the target for it}}
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_intrinsic("spyreop.realdiv") was not selected. The request is explicit, so its fallback is not used instead}}
    %s = math.exp %a {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.realdiv"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// ONE CALL OPERAND, TWO INPUTS: two ops say call operand 0 is block arguments
// reading different values, so which one the target takes is not known.
func.func @conflicting_binding(%x: tensor<64xf32>, %y: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%x, %y : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_intrinsic("spyreop.realdiv") was not selected}}
    %n = math.exp %a {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.realdiv"}} : f32
    // expected-error @+1 {{lower-spyre-intrinsic-calls: tl.spyre_intrinsic("spyreop.realdiv"): call operand 0 is read from two different inputs of its compute body, block arguments 0 and 1}}
    %s = arith.addf %n, %b {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.realdiv"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A BINDING TO SOMETHING NOT AN `ins` BLOCK ARGUMENT: here a value computed in
// the body, which no pass between the inlining and selection produces.
func.func @binding_not_an_input(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %out: f32):
    // expected-error @+1 {{tl.spyre_intrinsic("spyreop.sigmoid") was not selected}}
    %n = arith.negf %a {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}} : f32
    // expected-error @+1 {{lower-spyre-intrinsic-calls: tl.spyre_intrinsic("spyreop.sigmoid"): call operand 0 is operand 0 of 'math.exp', which is not an input of its compute body -- a compiler bug}}
    %s = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

// A HINTED OP OUTSIDE ANY BODY. No body holds it, so no call site is selected,
// and the attribute left on it is reported. The pass fails, so the pipeline
// stops before LowerSpyreOps, which would select a scalar math.exp wherever it
// is, as an unhinted spyreop.exp.
func.func @outside_body(%x: f32) -> f32 {
  // expected-error @+1 {{tl.spyre_intrinsic("spyreop.sigmoid") was not selected}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
  return %e : f32
}

// -----

#map = affine_map<(d0) -> (d0)>

// A TARGET THAT REFUSES THE OPERAND. The target is built by name, so its own
// verifier is what knows spyreop.gelu takes no f32; its diagnostic is reported,
// the body is left as it was, and the hint then reports the request.
func.func @target_refuses_type(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // expected-error @+2 {{'spyreop.gelu' op}}
    // expected-error @+1 {{tl.spyre_intrinsic("spyreop.gelu") was not selected}}
    %g = math.erf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.gelu"}} : f32
    linalg.yield %g : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A NON-COMMUTATIVE MEMBER WHOSE OPERANDS ARE NOT WHERE ITS BINDING SAYS. Its
// positions are exact, so call operand 0 is its operand 1, a constant.
func.func @non_commutative_swapped(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %out: f32):
    // expected-error @+2 {{tl.spyre_intrinsic("spyreop.sigmoid") was not selected}}
    // expected-error @+1 {{lower-spyre-intrinsic-calls: tl.spyre_intrinsic("spyreop.sigmoid"): call operand 0 is operand 1 of 'arith.subf', which is not an input of its compute body -- a compiler bug}}
    %s = arith.subf %a, %one {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.sigmoid"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A COMMUTATIVE MEMBER READ AS A SET, BINDING MORE CALL OPERANDS THAN IT READS
// INPUTS: two listed, one input, so which it is cannot be told.
func.func @commutative_count_mismatch(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%a: f32, %out: f32):
    // expected-error @+2 {{tl.spyre_intrinsic("spyreop.realdiv") was not selected}}
    // expected-error @+1 {{lower-spyre-intrinsic-calls: tl.spyre_intrinsic("spyreop.realdiv"): call operand 1 is bound on commutative 'arith.addf', which binds 2 call operand(s) and reads 1 input(s) of its compute body}}
    %s = arith.addf %one, %a {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1, 0>, target = "spyreop.realdiv"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}
