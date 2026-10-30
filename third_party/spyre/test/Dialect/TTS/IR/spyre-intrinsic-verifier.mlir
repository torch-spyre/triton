// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// What tts.spyre_intrinsic's region verifier refuses. Each case is one rule.

// A target that is not a registered op. Which targets exist, and what each
// takes, is the frontend's registration; the IR is held only to the target
// existing, and LowerSpyreIntrinsicCalls asks the target's own verifier about
// the operands.
tt.func @unknown_target(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{targets 'spyreop.no_such_op', which is not a registered op}}
  %r = tts.spyre_intrinsic "spyreop.no_such_op" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    tts.spyre_intrinsic_yield %a : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A memory op in the body. The request names an intrinsic that computes on
// values, so a print inside the fallback has nothing to stand for.
tt.func @memory_op(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must not access memory}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-note @+1 {{the op is here}}
    tt.print " a: " {hex = false, isSigned = array<i32: 0>} : %a : tensor<128xf32>
    tts.spyre_intrinsic_yield %a : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A reduction in the body: not elementwise, so the body cannot be one generic.
tt.func @reduce(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must be elementwise, so it can fuse into one generic, and 'tt.reduce' is not}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-note @+1 {{the op is here}}
    %s = "tt.reduce"(%a) <{axis = 0 : i32}> ({
    ^bb0(%l: f32, %r: f32):
      %t = arith.addf %l, %r : f32
      tt.reduce.return %t : f32
    }) : (tensor<128xf32>) -> f32
    %b = tt.splat %s : f32 -> tensor<128xf32>
    tts.spyre_intrinsic_yield %b : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A broadcast in the body: it changes the shape, so not elementwise either.
tt.func @broadcast(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must be elementwise, so it can fuse into one generic, and 'tt.broadcast' is not}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    %c = arith.constant dense<1.0> : tensor<1xf32>
    // expected-note @+1 {{the op is here}}
    %b = tt.broadcast %c : tensor<1xf32> -> tensor<128xf32>
    %s = arith.addf %a, %b : tensor<128xf32>
    tts.spyre_intrinsic_yield %s : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A call is admitted until the inliner replaces it, and its results are still
// held to the operands' shape.
tt.func private @widen(%x: tensor<128xf32>) -> tensor<2x128xf32>
tt.func @call_shape(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must keep the operands' shape, so it can fuse into one generic, and 'tt.call' gives 'tensor<2x128xf32>'}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-note @+1 {{the op is here}}
    %w = tt.call @widen(%a) : (tensor<128xf32>) -> tensor<2x128xf32>
    tts.spyre_intrinsic_yield %a : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A body ending in some other terminator. tt.return refuses this parent itself,
// so the test uses a terminator with no parent constraint.
tt.func @bad_terminator(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+2 {{expects regions to end with 'tts.spyre_intrinsic_yield', found 'ub.unreachable'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'tts.spyre_intrinsic_yield'}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    ub.unreachable
  }
  tt.return %x : tensor<128xf32>
}

// -----

// The yield disagreeing with the declared result types.
tt.func @yield_type(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{body yields 'tensor<128xf32>' but the op's results are 'tensor<128xf16>'}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf16>) -> tensor<128xf16> {
  ^bb0(%a: tensor<128xf16>):
    %f = arith.extf %a : tensor<128xf16> to tensor<128xf32>
    tts.spyre_intrinsic_yield %f : tensor<128xf32>
  }
  tt.return %r : tensor<128xf16>
}

// -----

// Block arguments that are not the operands' types.
tt.func @block_args(%x: tensor<128xf16>) -> tensor<128xf16> {
  // expected-error @+1 {{body block arguments must have the operand types}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf16>) -> tensor<128xf16> {
  ^bb0(%a: tensor<128xf32>):
    %t = arith.truncf %a : tensor<128xf32> to tensor<128xf16>
    tts.spyre_intrinsic_yield %t : tensor<128xf16>
  }
  tt.return %r : tensor<128xf16>
}

// -----

// IsolatedFromAbove: the body cannot read a value computed outside it.
tt.func @captures(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-note @+1 {{required by region isolation constraints}}
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-error @+1 {{using value defined outside the region}}
    %s = arith.addf %a, %x : tensor<128xf32>
    tts.spyre_intrinsic_yield %s : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

