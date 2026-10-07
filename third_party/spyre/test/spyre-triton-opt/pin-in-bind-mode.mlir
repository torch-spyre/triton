// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir --spyre-prepare-spyrecode="bind-base-addresses" -verify-diagnostics

// `bind-base-addresses` is the one arm of this stage that runs a CSE, and a pin cannot
// survive it: the store's `ktdp.construct_access_tile` and each load's are identical --
// same view, same zero indices, same block, both `Pure` -- so CSE merges them and
// dbo-opt's compute-group extraction then aborts. See issue #161, and the stage's own
// canonicalize, which has no CSE beside it for exactly this reason.
//
// Refused in `MaterializePinnedBuffers` rather than left to that abort, which is what
// it used to be: the IR compiled, and the failure arrived from dbo-opt with nothing
// naming its cause. #199 removes the mode, and the refusal leaves with it.
//
// `-verify-diagnostics` rather than FileCheck, and its own file rather than an arm of
// pin-survives-the-stage.mlir, because a run that refuses and a run that places cannot
// share one body: the expected-error below would go unmatched under the symbolic arm.
//
// No base addresses are supplied, and none can be: MaterializeBaseAddresses refuses a
// list longer than the function's index arguments, and a pinned kernel here takes a
// tensor rather than a `!tt.ptr`. The bare option is what puts the CSE in the pipeline,
// which is this file's whole subject.

// A pin with NO use is EXEMPT, and this chunk is that: one access tile, so there is
// nothing for CSE to merge and nothing to refuse. It expects no diagnostic, which is
// the assertion -- that the refusal is about the hazard rather than about the mode.
tt.func public @no_use_is_exempt(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// A CONSUMER is what makes the hazard real, and so it is what is refused.
tt.func public @a_consumer_is_refused(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{a pin is not supported with bind-base-addresses: the stage's CSE merges the pin's store and load access tiles, which aborts the scheduler}}
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
