// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.pin ATTRIBUTE, and how little the dialect checks about it.
//
// `tts.pin` is a discardable attribute, and the usual expectation is that MLIR
// does not check one at all. It does when the name carries a dialect prefix: the
// verifier looks the name's dialect up and calls its verifyOperationAttribute, on
// any op, at every verification point. The prefix is ours.
//
// What ours says is only that the name is ours. Everything about a pin's content
// is settled BEFORE the attribute exists -- the op's verifier checks the fields an
// author wrote (pin-op-verifier.mlir) and LowerTTSMarkers checks the move
// (Transforms/invalid.mlir) -- so re-deriving those rules out of the dictionary
// here would establish nothing and would state them a second place to drift.
//
// So this file pins two things, and neither is about a pin's fields.

// The attribute VERIFIES and round-trips, which is the load-bearing case: an
// unhandled `tts.` name falls through to the refusal below, so a dialect that
// forgot this branch would fail every module the lowering produced.
// CHECK-LABEL: tt.func @attribute_verifies
// CHECK: math.exp {{.*}} {tts.pin = {address = 4096 : i32, memory_space = #ktdp.memory_space<ct_local>}}
tt.func @attribute_verifies(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = #ktdp.memory_space<ct_local>,
                               address = 4096 : i32}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// And it verifies without the dialect asking anything about the content: a
// memory space a pin may NOT name passes here, because the op that wrote it was
// what refused it. Stated as a test so the division is visible rather than
// inferred -- if someone adds field checks to the hook, this case is what says
// they chose to.
// CHECK-LABEL: tt.func @content_is_not_the_hooks_business
// CHECK: math.exp {{.*}} {tts.pin = {memory_space = #ktdp.memory_space<global>}}
tt.func @content_is_not_the_hooks_business(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = #ktdp.memory_space<global>}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// A `tts.` name the dialect does not define. The prefix routes every such name
// here, so one nobody implements is an error rather than an annotation that rides
// along unchecked -- which is the whole reason the dialect exists.
tt.func @unknown_tts_attribute(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{attribute 'tts.pinned' is not one the tts dialect defines}}
  %e = math.exp %x {tts.pinned = {memory_space = #ktdp.memory_space<ct_local>}} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}
