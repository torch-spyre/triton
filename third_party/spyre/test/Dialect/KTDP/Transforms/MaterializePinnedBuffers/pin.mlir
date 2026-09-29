// RUN: spyre-triton-opt %s --materialize-pinned-buffers -split-input-file | FileCheck %s

// The shape of the whole thing, once: a view in the pin's space at the pin's
// offset, a store of the value through a whole-buffer access tile, and the
// consumer reading a load instead of the register. The annotation is gone, which
// is what keeps the pass idempotent -- @already_materialized drives that.
//
// The view is built ONCE and shared by the store and the load, with a fresh access
// tile per access -- the shape LowerDescriptorMemory uses for a descriptor, which
// is the one that runs on hardware.
//
// Note the two spellings of the memory space, which are the point rather than an
// inconsistency: the ANNOTATION holds the name `"ct_local"`, and the view this pass
// builds holds `#ktdp.memory_space<ct_local>`. Symbolizing the one into the other is
// exactly what this pass is for -- `tts` cannot construct that attribute, because
// the marker op it comes from is built during tracing, where loading ktdp would load
// `func` and abort the `ttir` stage's Inliner.
// CHECK-LABEL: tt.func @one_consumer
// CHECK: %[[E:.*]] = math.exp
// CHECK-NOT: tts.pin
// CHECK: %[[OFF:.*]] = arith.constant 4096 : index
// CHECK: %[[V:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [4, 64], strides: [64, 1] {{.*}}memory_space = #ktdp.memory_space<ct_local>
// CHECK: %[[ST:.*]] = ktdp.construct_access_tile %[[V]]
// CHECK: ktdp.store %[[E]], %[[ST]]
// CHECK: %[[LT:.*]] = ktdp.construct_access_tile %[[V]]
// CHECK: %[[L:.*]] = ktdp.load %[[LT]]
// CHECK: math.sqrt %[[L]]
tt.func @one_consumer(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 4096 : i32}} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// Two consumers, one buffer. Each gets its own access tile and its own load, and
// both read the SAME view -- a pin is one buffer for one value, not one per reader.
// CHECK-LABEL: tt.func @two_consumers
// CHECK: %[[V:.*]] = ktdp.construct_memory_view
// CHECK: ktdp.store
// CHECK: ktdp.construct_access_tile %[[V]]
// CHECK: %[[L1:.*]] = ktdp.load
// CHECK: math.sqrt %[[L1]]
// CHECK: ktdp.construct_access_tile %[[V]]
// CHECK: %[[L2:.*]] = ktdp.load
// CHECK: math.log %[[L2]]
tt.func @two_consumers(%x: tensor<4x64xf16>) -> (tensor<4x64xf16>, tensor<4x64xf16>) {
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  %a = math.sqrt %e : tensor<4x64xf16>
  %b = math.log %e : tensor<4x64xf16>
  tt.return %a, %b : tensor<4x64xf16>, tensor<4x64xf16>
}

// -----
// NO consumer. The buffer and the store are still built: the value was pinned, and
// whether anything reads it back is not this pass's question -- a pinned value with
// no reader in this function is what a value handed to a later local schedule looks
// like before anything else has run.
// CHECK-LABEL: tt.func @no_consumer
// CHECK: ktdp.construct_memory_view
// CHECK: ktdp.store
// CHECK-NOT: ktdp.load
tt.func @no_consumer(%x: tensor<4x64xf16>) {
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  tt.return
}

// -----
// RANK 0. A scalar has a buffer like any other value: sizes and strides are empty,
// the access tile takes no index, and the coordinate set is the always-true one an
// IntegerSet needs when it has no dimension to constrain.
// CHECK-LABEL: tt.func @rank_zero
// CHECK: ktdp.construct_memory_view {{.*}} : memref<f16>
// CHECK: ktdp.store
// CHECK: ktdp.load
tt.func @rank_zero(%x: tensor<f16>) -> tensor<f16> {
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<f16>
  %y = math.sqrt %e : tensor<f16>
  tt.return %y : tensor<f16>
}

// -----
// A pin on a value that is ALREADY a ktdp.load -- an HBM read the author wanted
// kept on chip. Nothing about the producing op is read, so this needs no case in
// the pass; it is here because it is the shape #207's composition consumes, where
// a share arrives from memory and is then redistributed.
// CHECK-LABEL: tt.func @pinned_load_result
// CHECK: %[[HBM:.*]] = ktdp.load
// CHECK-NOT: tts.pin
// CHECK: %[[V:.*]] = ktdp.construct_memory_view {{.*}}memory_space = #ktdp.memory_space<ct_local>
// CHECK: ktdp.store %[[HBM]]
tt.func @pinned_load_result(%base: index) -> tensor<4x64xf16> {
  %c0 = arith.constant 0 : index
  %v = ktdp.construct_memory_view %base, sizes: [4, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<4x64xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %x = ktdp.load %t {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : <4x64xindex> -> tensor<4x64xf16>
  tt.return %x : tensor<4x64xf16>
}

// -----
// Two pins that do NOT overlap, at 4 sticks apart. Both are placed, which is what
// makes the refusals in invalid.mlir rules about intersection rather than about
// having two pins at all.
// CHECK-LABEL: tt.func @two_disjoint_pins
// CHECK-COUNT-2: ktdp.construct_memory_view
// CHECK-NOT: tts.pin
tt.func @two_disjoint_pins(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %a = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  %b = math.sqrt %x {tts.pin = {memory_space = "ct_local", offset = 256 : i32}} : tensor<4x64xf16>
  %y = arith.addf %a, %b : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// Two FUNCTIONS, each pinning at offset 0. Both are placed, and that is the point:
// the overlap rule is that two pins of one kernel share one base, so two kernels
// share nothing and have nothing to compare. Collected module-wide, these were
// reported as overlapping -- a real bug rather than a hand-written-IR one, since a
// module with several functions is an ordinary shape here.
//
// One chunk rather than two, because splitting them would pass whatever the grouping
// did: it is the two being in ONE module that used to refuse.
// CHECK-LABEL: tt.func @first_kernel
// CHECK: ktdp.construct_memory_view
// CHECK-NOT: tts.pin
// CHECK-LABEL: tt.func @second_kernel
// CHECK: ktdp.construct_memory_view
// CHECK-NOT: tts.pin
tt.func @first_kernel(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
tt.func @second_kernel(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// NO pin anywhere. The pass is installed in every `spyrecode` compile and most
// kernels pin nothing, so the no-op path is the common one and is driven here.
// CHECK-LABEL: tt.func @no_pin
// CHECK-NOT: ktdp.construct_memory_view
// CHECK: math.sqrt
tt.func @no_pin(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// A LINALG carrier, which is the case this directory was missing and the one the
// pass's whole placement argument is about. A pinned `tl.sum` is a `linalg.reduce` by
// the time the annotation is written, and the reason this pass runs FIRST in
// `spyrecode` is that DropReductionInitFill and LinalgGeneralizeNamedOps -- both
// below it -- replace such an op and would take the annotation with them in silence.
// Nothing in the pass reads the carrier's identity, which is what this case pins.
// CHECK-LABEL: tt.func @linalg_carrier
// CHECK: %[[R:.*]] = linalg.reduce
// CHECK-NOT: tts.pin
// CHECK: %[[V:.*]] = ktdp.construct_memory_view {{.*}} : memref<4xf32>
// CHECK: ktdp.store %[[R]]
// CHECK: ktdp.load
// CHECK: math.sqrt
tt.func @linalg_carrier(%x: tensor<4x64xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.reduce { arith.addf } ins(%x : tensor<4x64xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
      {tts.pin = {memory_space = "ct_local", offset = 0 : i32}}
  %y = math.sqrt %r : tensor<4xf32>
  tt.return %y : tensor<4xf32>
}

// -----
// IDEMPOTENCY, which Passes.td claims as a property and nothing drove. The pass
// removes the annotation once it has honoured it, so a module that has already been
// through it has no request left and running it again changes nothing. That is what
// keeps a second consumer from reading a pin as still outstanding.
//
// Expressed as input that has already been materialized: one view, one store, one
// load, and no `tts.pin`. A second view would mean the pass had rooted on something
// other than the annotation.
// CHECK-LABEL: tt.func @already_materialized
// CHECK-COUNT-1: ktdp.construct_memory_view
// CHECK-NOT: ktdp.construct_memory_view
// CHECK-NOT: tts.pin
tt.func @already_materialized(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %c0 = arith.constant 0 : index
  %e = math.exp %x : tensor<4x64xf16>
  %v = ktdp.construct_memory_view %c0, sizes: [4, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<4x64xf16>
  %st = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  ktdp.store %e, %st : tensor<4x64xf16>, <4x64xindex>
  %lt = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %l = ktdp.load %lt : <4x64xindex> -> tensor<4x64xf16>
  %y = math.sqrt %l : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// A pin consumed INSIDE an scf.for body, which is undocumented behaviour rather than
// a rule -- so it is written down here instead of being left to be discovered.
//
// The access tile and the load are built immediately before each USE, and a use
// inside a loop body is reached every iteration. So the scratchpad is re-read per
// iteration rather than once: the store stays outside, hoisted where the producer is,
// and only the read is repeated. That is correct -- a load from a buffer nothing
// writes in the loop yields the same value -- and it is a cost nobody chose.
//
// TODO: hoist the tile and load to the value's own position when every use is
// dominated by it, which would make this one read. Recorded rather than fixed,
// because a pin inside a loop has no consumer in this tree yet.
// CHECK-LABEL: tt.func @consumed_in_a_loop
// CHECK: %[[V:.*]] = ktdp.construct_memory_view
// CHECK: ktdp.store
// CHECK: scf.for
// The read is INSIDE the loop, which is the observation.
// CHECK: ktdp.construct_access_tile %[[V]]
// CHECK: ktdp.load
// CHECK: arith.addf
tt.func @consumed_in_a_loop(%x: tensor<4x64xf16>, %lb: index, %ub: index, %step: index)
    -> tensor<4x64xf16> {
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  %acc = scf.for %i = %lb to %ub step %step iter_args(%a = %x) -> tensor<4x64xf16> {
    %s = arith.addf %a, %e : tensor<4x64xf16>
    scf.yield %s : tensor<4x64xf16>
  }
  tt.return %acc : tensor<4x64xf16>
}

// -----
// A ZERO-ELEMENT value, which occupies no bytes and is therefore exempt from the
// overlap rule: it cannot alias anything, so a pin at the same offset as a real one
// is not a collision. Both are placed.
//
// This is the case that made splitting `empty()` necessary. It used to share one
// guard with "the byte size is not known", which a sub-byte element type produced --
// and that meaning is gone, refused in checkOffset instead, so `hi == lo` now says
// only this. Skipping the comparison is a conclusion about a value with no elements
// rather than a gap.
// CHECK-LABEL: tt.func @zero_element_pin
// CHECK-COUNT-2: ktdp.construct_memory_view
// CHECK-NOT: tts.pin
tt.func @zero_element_pin(%z: tensor<0x64xf16>, %x: tensor<4x64xf16>) {
  %a = math.exp %z {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<0x64xf16>
  %b = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  tt.return
}
