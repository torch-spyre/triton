// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.spyre_intrinsic ATTRIBUTE, checked by the dialect's attribute verifier on
// whichever op carries it. Only what one op can show: the three required fields
// and their types, the optional `operands` against the op's own operand count
// and the hint's `arity`, a target that is a registered op, and an op that is
// not a constant.

// The spellings LowerTTSMarkers writes verify and round-trip: with `operands`
// on an op reading call operands, and without it on one reading none.
// CHECK-LABEL: tt.func @verifies
// CHECK: arith.divf {{.*}} {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1, -1>, target = "spyreop.realdiv"}}
// CHECK: math.exp {{.*}} {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, target = "spyreop.realdiv"}}
tt.func @verifies(%x: tensor<128xf32>, %y: tensor<128xf32>) -> tensor<128xf32> {
  %d = arith.divf %y, %x {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1, -1>, target = "spyreop.realdiv"}} : tensor<128xf32>
  %e = math.exp %d {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, target = "spyreop.realdiv"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// Not a dictionary.
tt.func @not_a_dictionary(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' must be a dictionary of a string 'target', an i64 'id', an i64 'arity' and optionally an array<i64> 'operands', and nothing else, got "sigmoid"}}
  %e = math.exp %x {tts.spyre_intrinsic = "sigmoid"} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A missing field.
tt.func @missing_id(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' must be a dictionary of a string 'target', an i64 'id', an i64 'arity' and optionally an array<i64> 'operands', and nothing else, got {arity = 1 : i64, target = "spyreop.sigmoid"}}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// No arity, the spelling before call operands were bound.
tt.func @missing_arity(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' must be a dictionary of a string 'target', an i64 'id', an i64 'arity'}}
  %e = math.exp %x {tts.spyre_intrinsic = {id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A field of the wrong type.
tt.func @i32_id(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' must be a dictionary of a string 'target', an i64 'id', an i64 'arity'}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i32, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

tt.func @i32_arity(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' must be a dictionary of a string 'target', an i64 'id', an i64 'arity'}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i32, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// `operands` that is not an i64 array.
tt.func @operands_not_an_array(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' must be a dictionary of a string 'target', an i64 'id', an i64 'arity'}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = [0 : i64], target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A field LowerTTSMarkers does not write.
tt.func @extra_field(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' must be a dictionary of a string 'target', an i64 'id', an i64 'arity'}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid", size = 4 : i64}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// An arity of no operands.
tt.func @zero_arity(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' has arity 0, and a call site has at least one operand}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 0 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// `operands` of another length than the op's operand count.
tt.func @operands_wrong_length(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' has 2 'operands' entries, one per operand of an op with 1}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0, -1>, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A call operand index at the arity, past the last.
tt.func @operand_past_arity(%x: tensor<128xf32>, %y: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' binds operand 1 to call operand 2, and a call site of arity 2 has operands 0 to 1, or -1 for none}}
  %d = arith.divf %y, %x {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 0, 2>, target = "spyreop.realdiv"}} : tensor<128xf32>
  tt.return %d : tensor<128xf32>
}

// -----

// Below -1, the one negative entry that means "not a call operand".
tt.func @operand_below_minus_one(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' binds operand 0 to call operand -2, and a call site of arity 1 has operands 0 to 0, or -1 for none}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: -2>, target = "spyreop.sigmoid"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A target that is not a registered op.
tt.func @unknown_target(%x: tensor<128xf32>) -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' targets "spyreop.no_such_op", which is not a registered op}}
  %e = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.no_such_op"}} : tensor<128xf32>
  tt.return %e : tensor<128xf32>
}

// -----

// A constant, which LowerTTSMarkers never hints.
tt.func @constant() -> tensor<128xf32> {
  // expected-error @+1 {{'tts.spyre_intrinsic' is not set on a constant}}
  %c = arith.constant {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} dense<1.0> : tensor<128xf32>
  tt.return %c : tensor<128xf32>
}
