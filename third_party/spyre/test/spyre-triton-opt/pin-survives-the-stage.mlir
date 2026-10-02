// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PLACED
// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir --spyre-prepare-spyrecode="bind-base-addresses" | FileCheck %s --check-prefix=BOUND

// A pin surviving BOTH stages: annotated in `ktir`, honoured at the head of
// `spyrecode`. That the attribute reaches the end of the first stage at all is
// pin-survives-the-ktir-stage.mlir's subject, and it is the harder half -- the
// canonicalize there would delete a pinned value whose only use was the marker. This
// file starts where that one ends and asks the other question: the request is in the
// module, and the next stage acts on it.
//
// Both halves are needed because they fail differently. Losing the value in `ktir`
// leaves no attribute to honour; keeping the attribute and not honouring it leaves a
// request in a binary.

// PLACED-LABEL: func.func @pinned_value_with_no_other_use
// The buffer, the store, and no request left outstanding.
// PLACED: ktdp.construct_memory_view
// PLACED-SAME: memory_space = #ktdp.memory_space<ct_local>
// PLACED: ktdp.store
// PLACED-NOT: tts.pin

tt.func public @pinned_value_with_no_other_use(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// The SECOND run's subject, and it is a tracking case rather than an assertion of
// something desirable.
//
// `bind-base-addresses` is the one arm of this stage that runs a CSE
// (Pipeline.cpp:271), and a pin builds a `ktdp.construct_access_tile` for its store
// and an identical one for each load: same view, same zero indices, both `Pure`, same
// block. Pipeline.cpp:250 names exactly that shape as fatal -- CSE merges the two and
// dbo-opt's compute-group extraction aborts (issue #161), which is why the stage's own
// canonicalize has no CSE beside it. So under this arm the load reads the store's tile
// rather than its own.
//
// Not worked around in the pass, deliberately: `bindBaseAddresses` is slated for
// deprecation (#199), and a pin exists to serve the symbolic path that the first run
// covers. This case is here so that the interaction is recorded, and so that whatever
// replaces the bind arm has to decide about it rather than inherit it.
//
// A CONSUMER is what makes it visible -- the run above pins a value with no use, which
// has a store and no load and therefore only one tile.
//
// No base addresses are supplied, and none can be: MaterializeBaseAddresses refuses a
// list longer than the function's index arguments, and a pinned kernel here takes a
// tensor rather than a `!tt.ptr`. The bare option is what puts the CSE in the pipeline,
// which is this case's whole subject.
// BOUND-LABEL: func.func @pinned_value_with_a_consumer
// BOUND: ktdp.construct_memory_view
// BOUND-SAME: memory_space = #ktdp.memory_space<ct_local>
// BOUND: ktdp.store
// The merged tile: one construct_access_tile serves the store and the load.
// BOUND-NOT: ktdp.construct_access_tile
// BOUND: ktdp.load
// BOUND-NOT: tts.pin

tt.func public @pinned_value_with_a_consumer(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
