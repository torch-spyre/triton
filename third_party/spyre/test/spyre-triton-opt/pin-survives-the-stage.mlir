// RUN: spyre-triton-opt %s -split-input-file --spyre-ttir-to-ktir --spyre-prepare-spyrecode | FileCheck %s --check-prefix=PLACED

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
// A pin with a CONSUMER, which is the shape the case above cannot show: a pinned value
// with no use has a store and no load, so it has one access tile. With a consumer there
// are TWO, built separately, and that is what this asserts -- the store's and the
// load's, over the same view and the same block.
//
// Two rather than one is load-bearing rather than incidental. Both are `Pure` over the
// same block, so a CSE would merge them and dbo-opt's compute-group extraction then
// aborts (issue #161), which is why the stage's canonicalize has no CSE beside it. The
// one arm of this pipeline that does run a CSE is `bind-base-addresses`, and a pin is
// refused there rather than quietly merged -- pin-in-bind-mode.mlir is that.
// PLACED-LABEL: func.func @pinned_value_with_a_consumer
// PLACED: ktdp.construct_memory_view
// PLACED-SAME: memory_space = #ktdp.memory_space<ct_local>
// PLACED: ktdp.construct_access_tile
// PLACED: ktdp.store
// PLACED: ktdp.construct_access_tile
// PLACED: ktdp.load
// PLACED-NOT: tts.pin

tt.func public @pinned_value_with_a_consumer(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
