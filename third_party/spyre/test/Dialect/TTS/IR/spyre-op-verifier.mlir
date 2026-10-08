// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// What tts.spyre_op's region verifier refuses. Each case is one rule.

// A memory op in the body. The request names an intrinsic that computes on
// values, so a load inside the fallback has nothing to stand for.
tt.func @load_in_body(%x: tensor<128xf32>, %p: tensor<128x!tt.ptr<f32>>) -> tensor<128xf32> {
  // expected-error @+1 {{body must not access memory}}
  %r = tts.spyre_op "sigmoid" (%x, %p) : (tensor<128xf32>, tensor<128x!tt.ptr<f32>>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>, %q: tensor<128x!tt.ptr<f32>>):
    // expected-note @+1 {{the op is here}}
    %l = tt.load %q : tensor<128x!tt.ptr<f32>>
    %s = arith.addf %a, %l : tensor<128xf32>
    tts.spyreop_yield %s : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A body ending in some other terminator. tt.return refuses this parent itself,
// so the test uses a terminator with no parent constraint.
tt.func @bad_terminator(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must end in tts.spyreop_yield, not 'ub.unreachable'}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    ub.unreachable
  }
  tt.return %x : tensor<128xf32>
}

// -----

// The yield disagreeing with the declared result types.
tt.func @yield_type(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{body yields 'tensor<128xf32>' but the op's results are 'tensor<128xf16>'}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf16>) -> tensor<128xf16> {
  ^bb0(%a: tensor<128xf16>):
    %f = arith.extf %a : tensor<128xf16> to tensor<128xf32>
    tts.spyreop_yield %f : tensor<128xf32>
  }
  tt.return %r : tensor<128xf16>
}

// -----

// Block arguments that are not the operands' types.
tt.func @block_args(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{body block arguments must have the operand types}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf16>) -> tensor<128xf16> {
  ^bb0(%a: tensor<128xf32>):
    %t = arith.truncf %a : tensor<128xf32> to tensor<128xf16>
    tts.spyreop_yield %t : tensor<128xf16>
  }
  tt.return %r : tensor<128xf16>
}

// -----

// IsolatedFromAbove: the body cannot read a value computed outside it.
tt.func @captures(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-note @+1 {{required by region isolation constraints}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-error @+1 {{using value defined outside the region}}
    %s = arith.addf %a, %x : tensor<128xf32>
    tts.spyreop_yield %s : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// The tag's spelling, which the dialect's attribute verifier owns.
tt.func @bad_tag(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'name' and an i64 'id'}}
  %e = math.exp %x {tts.spyreop_hint = {name = "sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}
