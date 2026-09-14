// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics

// spyreop's math intrinsics only exist for f16/f32. math.sqrt/exp/rsqrt and
// arith.divf are matched unconditionally on scalar type (unlike arith.addi/
// arith.muli, which also require being inside a linalg.generic body), so an
// f64 operand is not silently left alone -- it is reported, because a
// caller relying on this pass to reach spyreop should not have an
// unsupported dtype pass through undetected.

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @sqrt_f64(%s: f64) -> f64 {
  // expected-error @below {{failed to legalize operation 'math.sqrt' that was explicitly marked illegal}}
  %0 = math.sqrt %s : f64
  tt.return %0 : f64
}
}

// -----

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @exp_f64(%s: f64) -> f64 {
  // expected-error @below {{failed to legalize operation 'math.exp' that was explicitly marked illegal}}
  %0 = math.exp %s : f64
  tt.return %0 : f64
}
}

// -----

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @rsqrt_f64(%s: f64) -> f64 {
  // expected-error @below {{failed to legalize operation 'math.rsqrt' that was explicitly marked illegal}}
  %0 = math.rsqrt %s : f64
  tt.return %0 : f64
}
}

// -----

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @divf_f64(%a: f64, %b: f64) -> f64 {
  // expected-error @below {{failed to legalize operation 'arith.divf' that was explicitly marked illegal}}
  %0 = arith.divf %a, %b : f64
  tt.return %0 : f64
}
}
