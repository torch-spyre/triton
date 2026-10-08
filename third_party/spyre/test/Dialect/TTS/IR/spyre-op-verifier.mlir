// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// What tts.spyre_op's region verifier refuses. Each case is one rule.

// A name the intrinsic table does not have.
tt.func @unknown_name(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{names no spyreop intrinsic: 'softplus'; the intrinsic table has 'gelu' 'silu' 'sigmoid'}}
  %r = tts.spyre_op "softplus" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    tts.spyreop_yield %a : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// An operand count the table does not declare.
tt.func @operand_count(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'sigmoid' takes 1 operand(s), got 2}}
  %r = tts.spyre_op "sigmoid" (%x, %x) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>, %b: tensor<128xf32>):
    %s = arith.addf %a, %b : tensor<128xf32>
    tts.spyreop_yield %s : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A dtype the intrinsic does not take: spyreop.gelu is f16 only.
tt.func @gelu_f32(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'gelu' does not take operand 0 of element type 'f32'}}
  %r = tts.spyre_op "gelu" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    %e = math.erf %a : tensor<128xf32>
    tts.spyreop_yield %e : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// Result types other than the table's rule gives: sigmoid's result has its
// operand's type.
tt.func @result_type(%x: tensor<128xf16>) -> tensor<128xf32> {
  // expected-error @+1 {{'sigmoid' gives results 'tensor<128xf16>' for these operands, but the op declares 'tensor<128xf32>'}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf16>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf16>):
    %f = arith.extf %a : tensor<128xf16> to tensor<128xf32>
    tts.spyreop_yield %f : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A memory op in the body. The request names an intrinsic that computes on
// values, so a print inside the fallback has nothing to stand for.
tt.func @memory_op(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must not access memory}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-note @+1 {{the op is here}}
    tt.print " a: " {hex = false, isSigned = array<i32: 0>} : %a : tensor<128xf32>
    tts.spyreop_yield %a : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A reduction in the body: not elementwise, so the body cannot be one generic.
tt.func @reduce(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must be elementwise, so it can fuse into one generic, and 'tt.reduce' is not}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-note @+1 {{the op is here}}
    %s = "tt.reduce"(%a) <{axis = 0 : i32}> ({
    ^bb0(%l: f32, %r: f32):
      %t = arith.addf %l, %r : f32
      tt.reduce.return %t : f32
    }) : (tensor<128xf32>) -> f32
    %b = tt.splat %s : f32 -> tensor<128xf32>
    tts.spyreop_yield %b : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A broadcast in the body: it changes the shape, so not elementwise either.
tt.func @broadcast(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must be elementwise, so it can fuse into one generic, and 'tt.broadcast' is not}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    %c = arith.constant dense<1.0> : tensor<1xf32>
    // expected-note @+1 {{the op is here}}
    %b = tt.broadcast %c : tensor<1xf32> -> tensor<128xf32>
    %s = arith.addf %a, %b : tensor<128xf32>
    tts.spyreop_yield %s : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A call is admitted until the inliner replaces it, and its results are still
// held to the operands' shape.
tt.func private @widen(%x: tensor<128xf32>) -> tensor<2x128xf32>
tt.func @call_shape(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{body must keep the operands' shape, so it can fuse into one generic, and 'tt.call' gives 'tensor<2x128xf32>'}}
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    // expected-note @+1 {{the op is here}}
    %w = tt.call @widen(%a) : (tensor<128xf32>) -> tensor<2x128xf32>
    tts.spyreop_yield %a : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// -----

// A body ending in some other terminator. tt.return refuses this parent itself,
// so the test uses a terminator with no parent constraint.
tt.func @bad_terminator(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+2 {{expects regions to end with 'tts.spyreop_yield', found 'ub.unreachable'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'tts.spyreop_yield'}}
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
