// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.pin ATTRIBUTE's verifier, on its own -- no pass runs here.
//
// That is the point of the file, and it is the same point tensor-layout-verifier
// makes for the other marker: `tts.pin` is a DISCARDABLE attribute, and the usual
// expectation is that MLIR does not check one at all. It does when the name
// carries a dialect prefix, because the verifier looks the name's dialect up and
// calls its verifyOperationAttribute, on any op, at every verification point. The
// prefix is ours, so these checks are ours -- and they fire on a bare parse, ahead
// of any pass.
//
// This form matters MORE than the op's, not less. `tts.pin` the op is consumed in
// the `ktir` stage; `tts.pin` the attribute is what crosses into the artifact, so
// it is the spelling a consumer reads and the one a hand-written or
// pass-corrupted module can present.
//
// The FIELD rules -- ct_local only, no ct_id, an address, a non-empty array -- are
// tts::verifyPinFields, shared with the op's verifier, so pin-op-verifier.mlir
// covers them through the other caller and only one case repeats here to show the
// shared path is reached. What is unique to this file is everything about the
// attribute's SPELLING, which the op gets from ODS for free, and the rule about
// what may CARRY it.

// Case 1 -- the accepted form, which nothing else here would distinguish a
// silently-unchecked attribute from. It survives the round trip unchanged, with
// the entries in the printer's sorted order.
// CHECK-LABEL: tt.func @accepted
// CHECK: math.exp {{.*}} {tts.pin = {address = 4096 : i32, memory_space = #ktdp.memory_space<ct_local>}}
tt.func @accepted(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = #ktdp.memory_space<ct_local>,
                               address = 4096 : i32}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// The per-core spelling, also accepted, so the array is known to survive as an
// attribute entry and not only as the op's own field.
// CHECK-LABEL: tt.func @accepted_array
// CHECK: math.exp {{.*}} {tts.pin = {address = array<i32: 4096, 6144>, memory_space = #ktdp.memory_space<ct_local>}}
tt.func @accepted_array(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = #ktdp.memory_space<ct_local>,
                               address = array<i32: 4096, 6144>}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// WHAT MAY CARRY IT, and this is the attribute form's own rule rather than a
// shared one. A pin is about one value, and the annotation names no result index,
// so an op with two results could not say which result it is about -- and picking
// the first would be silent. Note a block argument cannot arise here at all: an
// attribute on an op means there is an op, which is why the op's verifier carries
// that half instead.
// `tt.split` is the multi-result op this tree actually has, so the rule is shown
// on IR that could occur rather than on a stub from an unregistered dialect.
tt.func @two_results(%t: tensor<8x2xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  // expected-error @+1 {{'tts.pin' is about one value, so it needs an op with exactly one result to name; this op has 2}}
  %0, %1 = tt.split %t {tts.pin = {memory_space = #ktdp.memory_space<ct_local>,
                                   address = 4096 : i32}}
      : tensor<8x2xf32> -> tensor<8xf32>
  tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----
// A result that is not a tensor. A pin annotates a buffer for a tensor, so there
// is nothing for it to mean on a scalar -- and it would be inert rather than
// wrong, which is the failure worth catching.
tt.func @non_tensor_result() {
  // expected-error @+1 {{'tts.pin' annotates a buffer for a tensor, but this op's result is 'i32'}}
  %c = arith.constant {tts.pin = {memory_space = #ktdp.memory_space<ct_local>,
                                  address = 4096 : i32}} 7 : i32
  tt.return
}

// -----
// Not a dictionary. The op gets its two fields typed by ODS; the attribute is
// whatever was written, so its shape is checked here or nowhere.
tt.func @not_a_dictionary(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{tts.pin: expected a dictionary attribute}}
  %e = math.exp %x {tts.pin = 4096 : i32} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// The memory space missing altogether. Required, unlike the address, because a
// pin with no space names nothing at all -- there is no default memory to fall
// back to.
tt.func @missing_memory_space(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{tts.pin: missing 'memory_space' entry}}
  %e = math.exp %x {tts.pin = {address = 4096 : i32}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// The memory space present but a string, which is what this attribute used to
// hold. Refused, so an artifact written by an older tree is a diagnostic rather
// than a silently different meaning.
tt.func @memory_space_as_string(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{tts.pin: 'memory_space' must be a #ktdp.memory_space}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local",
                               address = 4096 : i32}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// An entry nobody reads. Counted rather than ignored, so a typo'd name is an
// error instead of a field that silently does nothing -- which for a placement
// annotation means a buffer somewhere the author did not ask for.
tt.func @extra_entry(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{tts.pin: expected the entries memory_space and optionally address, got 3 entries}}
  %e = math.exp %x {tts.pin = {memory_space = #ktdp.memory_space<ct_local>,
                               address = 4096 : i32,
                               aligment = 128 : i32}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// One FIELD rule, to show verifyPinFields is reached through this caller too and
// not only through the op's verifier. The rest of them are covered there.
tt.func @field_rules_are_reached(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{tts.pin: memory space 'global' cannot be pinned}}
  %e = math.exp %x {tts.pin = {memory_space = #ktdp.memory_space<global>,
                               address = 4096 : i32}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// An attribute name this dialect does not define. The prefix routes every `tts.`
// name here, so one nobody implements is an error rather than an annotation that
// rides along unchecked.
tt.func @unknown_tts_attribute(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{attribute 'tts.pinned' is not one the tts dialect defines}}
  %e = math.exp %x {tts.pinned = {memory_space = #ktdp.memory_space<ct_local>}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}
