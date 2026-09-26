// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics | FileCheck %s

// FileCheck is piped as well as -verify-diagnostics, because the last case asserts
// what the pass LEAVES BEHIND rather than what it reports. Without the pipe the
// CHECK lines below would be inert text that reads as verified.

// spyreop's math intrinsics only exist for f16/f32. math.sqrt/exp/rsqrt and
// arith.divf are matched on any TENSOR of a float type, so an f64 tensor is not
// silently left alone -- it is reported, because a caller relying on this pass to
// reach spyreop should not have an unsupported dtype pass through undetected.
//
// That requirement is unchanged from when this pass ran after scalarization and
// matched scalars; only the shape it applies to moved. What makes it hold is that
// the legality callback marks a float op on a TENSOR illegal whatever its element
// type, while the pattern converts only f16/f32 -- so an f64 tensor is a decline
// against an illegal op, which is a diagnostic rather than a silent pass-through.
// Marking only the SUPPORTED element types illegal would reintroduce exactly the
// hole this file exists to catch.
//
// The scalar cases at the end are the other half of the contract: a scalar float
// op is legal and left alone, because at this point in the pipeline nothing has
// scalarized and a scalar float op is not elementwise compute. They are asserted
// rather than assumed, so that "scalars are ignored" stays pinned.

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @sqrt_f64_tensor(%t: tensor<4xf64>) -> tensor<4xf64> {
  // expected-error @below {{failed to legalize operation 'math.sqrt' that was explicitly marked illegal}}
  %0 = math.sqrt %t : tensor<4xf64>
  tt.return %0 : tensor<4xf64>
}
}

// -----

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @exp_f64_tensor(%t: tensor<4xf64>) -> tensor<4xf64> {
  // expected-error @below {{failed to legalize operation 'math.exp' that was explicitly marked illegal}}
  %0 = math.exp %t : tensor<4xf64>
  tt.return %0 : tensor<4xf64>
}
}

// -----

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @rsqrt_f64_tensor(%t: tensor<4xf64>) -> tensor<4xf64> {
  // expected-error @below {{failed to legalize operation 'math.rsqrt' that was explicitly marked illegal}}
  %0 = math.rsqrt %t : tensor<4xf64>
  tt.return %0 : tensor<4xf64>
}
}

// -----

// expected-error @below {{LowerSpyreOps: failed to convert math ops}}
module {
tt.func @divf_f64_tensor(%a: tensor<4xf64>, %b: tensor<4xf64>) -> tensor<4xf64> {
  // expected-error @below {{failed to legalize operation 'arith.divf' that was explicitly marked illegal}}
  %0 = arith.divf %a, %b : tensor<4xf64>
  tt.return %0 : tensor<4xf64>
}
}

// -----

// A SCALAR float op is left alone, at any element type -- no diagnostic, no
// conversion. Four ops in one module, because the claim is about the whole family
// and a per-op split would not say anything the first case does not.
//
// No expected-error anywhere in this case: -verify-diagnostics fails the test if
// the pass emits one, so the absence of an annotation IS the assertion.

// CHECK-LABEL:   tt.func @scalar_float_ops_survive(
// CHECK:           math.sqrt {{.*}} : f32
// CHECK:           math.exp {{.*}} : f16
// CHECK:           math.rsqrt {{.*}} : f64
// CHECK:           arith.divf {{.*}} : f32
// CHECK-NOT:       spyreop
module {
tt.func @scalar_float_ops_survive(%f32: f32, %f16: f16, %f64: f64) -> f32 {
  %0 = math.sqrt %f32 : f32
  %1 = math.exp %f16 : f16
  %2 = math.rsqrt %f64 : f64
  %3 = arith.divf %0, %f32 : f32
  tt.return %3 : f32
}
}
