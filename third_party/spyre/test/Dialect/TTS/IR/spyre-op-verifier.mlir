// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// The tts.spyre_op OP's verifier. No pass runs here.
//
// Hard-coded to the one registered intrinsic today ("gelu"), matching
// semantic.py's own check at the kernel line and the frontend's hard-coded
// dtype requirement (f16, matching spyreop.gelu's own real constraint
// exactly -- no f32, no DF16). Both generalize into a real name-to-op table
// once a second intrinsic exists; see TTSOps.td.

// An unregistered name. Refused here and also, redundantly, at the kernel
// line in semantic.py -- the same split tts.pin's memory_space check uses,
// and for the same reason: a typo should be a traceback at the kernel line,
// not a verifier failure after the whole function has traced.
tt.func @unregistered_name(%x: tensor<128xf16>) {
  // expected-error @+1 {{'silu' names no registered intrinsic; the only one registered today is 'gelu'}}
  %y = tts.spyre_op "silu" (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return
}

// -----
// gelu's real hardware op (spyreop.gelu) is f16-only.
tt.func @wrong_element_type(%x: tensor<128xf32>) {
  // expected-error @+1 {{operand must be a tensor of f16, got 'tensor<128xf32>'}}
  %y = tts.spyre_op "gelu" (%x) : (tensor<128xf32>) -> tensor<128xf32>
  tt.return
}

// -----
// Result type must match the operand type -- this op is a request, not a
// cast; nothing about it changes shape or element type.
tt.func @result_type_mismatch(%x: tensor<128xf16>) {
  // expected-error @+1 {{result type 'tensor<64xf16>' must match the operand type 'tensor<128xf16>'}}
  %y = tts.spyre_op "gelu" (%x) : (tensor<128xf16>) -> tensor<64xf16>
  tt.return
}

// -----
// The form that DOES verify, so the rejections above are read as rules and
// not as the op being hard to satisfy.
tt.func @accepted_form(%x: tensor<128xf16>) {
  %y = tts.spyre_op "gelu" (%x) : (tensor<128xf16>) -> tensor<128xf16>
  tt.return
}
