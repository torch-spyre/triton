// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.spyreop_hint ATTRIBUTE, checked by the dialect's attribute verifier on
// whichever op carries it. Only what one op can show: the two fields and their
// types, a name in the intrinsic table, and an op that is not a constant.

// The spelling LowerTTSMarkers writes verifies and round-trips.
// CHECK-LABEL: tt.func @verifies
// CHECK: math.exp {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
tt.func @verifies(%x: tensor<128xf32>) -> tensor<128xf32> {
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// Not a dictionary.
tt.func @not_a_dictionary(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'name' and an i64 'id', got "sigmoid"}}
  %e = math.exp %x {tts.spyreop_hint = "sigmoid"} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A missing field.
tt.func @missing_id(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'name' and an i64 'id', got {name = "sigmoid"}}}
  %e = math.exp %x {tts.spyreop_hint = {name = "sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A field of the wrong type.
tt.func @i32_id(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'name' and an i64 'id'}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i32, name = "sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A field LowerTTSMarkers does not write.
tt.func @extra_field(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' must be a dictionary of exactly a string 'name' and an i64 'id'}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid", size = 4 : i64}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A name the intrinsic table does not have.
tt.func @unknown_name(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' names no spyreop intrinsic: "softplus"}}
  %e = math.exp %x {tts.spyreop_hint = {id = 0 : i64, name = "softplus"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A constant, which LowerTTSMarkers never hints.
tt.func @constant() -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyreop_hint' is not set on a constant}}
  %c = arith.constant {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}} dense<1.0> : tensor<128xf32>
  tt.return %c : tensor<128xf32>
}
