// RUN: spyre-triton-opt %s --materialize-pinned-buffers="lx-capacity-bytes=8192" -split-input-file -verify-diagnostics

// `not` is not a lit substitution in this tree, so a refusing run is expressed with
// -verify-diagnostics rather than by expecting a non-zero exit.
//
// The capacity here is DRIVEN rather than defaulted, because there is no default:
// 8192 bytes is this file's device. What a budget buys, and what stating none means,
// is options.mlir's -- that is the pass's one option; what is here are the refusals
// that hold under any settings.

// A NEGATIVE offset. The attribute is an i32, which admits one, and no allocation
// starts before its own base: checked here rather than in the op's verifier because
// what makes it wrong is a fact about memory rather than about the field.
tt.func @negative_offset(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{pinned offset is negative: -64}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = -64 : i32}} : tensor<4x64xf16>
  tt.return
}

// -----
// Two pins whose ranges INTERSECT. Refused for now although a region reused by two
// values whose live ranges do not overlap is a legitimate thing to want -- see the
// TODO in the pass. Relaxing that rule is expected to change this case.
tt.func @overlapping_pins(%x: tensor<4x64xf16>) {
  // expected-note @+1 {{the other pin is here}}
  %a = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  // expected-error @+1 {{pinned range [128, 640) bytes overlaps another pin's [0, 512)}}
  %b = math.sqrt %x {tts.pin = {memory_space = "ct_local", offset = 64 : i32}} : tensor<4x64xf16>
  tt.return
}

// -----
// The same intersection, across two DIFFERENT element types, whose element ranges
// do NOT intersect: [0, 64) of an f32 against [64, 128) of an f16. As indices those
// miss; as memory they are bytes [0, 256) against [128, 256), and they alias. This
// is why the comparison is done in bytes, and it is the case an element-index
// comparison passed silently.
tt.func @overlap_only_in_bytes(%a: tensor<64xf32>, %b: tensor<64xf16>) {
  // expected-note @+1 {{the other pin is here}}
  %p = math.exp %a {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<64xf32>
  // expected-error @+1 {{pinned range [128, 256) bytes overlaps another pin's [0, 256)}}
  %q = math.exp %b {tts.pin = {memory_space = "ct_local", offset = 64 : i32}} : tensor<64xf16>
  tt.return
}

// -----
// An annotation with NO offset. Refused by `tts.pin`'s own verifier for an authored
// pin, and again here because an attribute written by hand never passed that: this
// pass runs on hand-written IR in exactly these tests.
tt.func @no_offset(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{pinned value has no offset}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local"}} : tensor<4x64xf16>
  tt.return
}

// -----
// An offset that is an integer of the WRONG KIND. `tts.pin` declares the field as an
// `I32Attr`, so signless i32 is the only spelling either producer writes -- and the
// three cases below are not merely wrong answers. An unsigned attribute reaches
// `IntegerAttr::getInt()`, whose first assertion is that the type is signless, so the
// pass used to CRASH on hand-written IR rather than diagnose it.
tt.func @offset_unsigned(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{a pinned offset must be a signless i32}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : ui32}} : tensor<4x64xf16>
  tt.return
}

// -----
// An offset too WIDE for the value to be read out of. The second assertion inside
// `getInt()` is that the value fits an int64_t, so this crashed too; 2^63 is the
// smallest value that reaches it.
tt.func @offset_too_wide(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{a pinned offset must be a signless i32}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 9223372036854775808 : i128}} : tensor<4x64xf16>
  tt.return
}

// -----
// An offset that is not an INTEGER at all. This one did not crash and was worse for
// it: the entry was read with a dyn_cast to IntegerAttr and a failed cast was taken
// for an absent entry, so the author was told the pin had no offset. It has one.
tt.func @offset_not_an_integer(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{a pinned offset must be a signless i32}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 1.0 : f32}} : tensor<4x64xf16>
  tt.return
}

// -----
// An annotation with no memory space. Same reason: the dictionary is malformed, and
// a malformed one is diagnosed rather than asserted on.
tt.func @no_memory_space(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{has no 'memory_space' entry naming a memory space}}
  %e = math.exp %x {tts.pin = {offset = 0 : i32}} : tensor<4x64xf16>
  tt.return
}

// -----
// An annotation on an op with more than one result. `LowerTTSMarkers` refuses to
// write one, because a single dictionary cannot say which result it is for; said
// again here for the same reason as the two above.
tt.func @multi_result(%x: tensor<4x64xf16>, %lb: index, %ub: index, %step: index) {
  // expected-error @+1 {{a pin annotation is one buffer for one value, and this op has 2 results}}
  %r:2 = scf.for %i = %lb to %ub step %step iter_args(%a = %x, %b = %x)
      -> (tensor<4x64xf16>, tensor<4x64xf16>) {
    scf.yield %a, %b : tensor<4x64xf16>, tensor<4x64xf16>
  } {tts.pin = {memory_space = "ct_local", offset = 0 : i32}}
  tt.return
}

// -----
// An annotation on an op whose result is not a tensor. There is no shape to build a
// buffer over, so there is no buffer to build.
tt.func @not_a_tensor(%a: index, %b: index) {
  // expected-error @+1 {{a pinned value must be a statically shaped tensor, and this one is 'index'}}
  %s = arith.addi %a, %b {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : index
  tt.return
}

// -----
// A memory space that IS a ktdp kind and is still not pinnable. Two separate checks
// for a reason: symbolizing the name answers whether ktdp defines it, and `global`
// passes that -- it is a real kind, and the design's prose names it. What it cannot
// be is pinned, because an intermediate in HBM is written as a
// tl.make_tensor_descriptor with an explicit store and load and nothing here
// allocates an anonymous device buffer.
//
// `tts.pin`'s verifier says this too. Re-checked here for the reason every other rule
// in readPin is re-checked: an attribute reaching this pass need never have been an
// op, and this file is that input.
tt.func @not_ct_local(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{pinned memory space 'global' cannot be pinned: only 'ct_local' is}}
  %e = math.exp %x {tts.pin = {memory_space = "global", offset = 0 : i32}} : tensor<4x64xf16>
  tt.return
}

// -----
// A SUB-BYTE element type. Reachable from Python rather than hypothetical:
// `arith.cmpf` on tensors yields `tensor<...xi1>`, which is every comparison and
// every tl.where mask, and `tl.spyre_pin(mask, "ct_local", ...)` reaches here.
//
// It used to be waved through, and silently: `bitWidth / 8` was 0, which returned
// success with an empty range, and an empty range was also what the overlap loop
// skipped -- so two i1 pins at the same offset produced two buffers, past any
// capacity, with no diagnostic. Refused now, because an offset in ELEMENTS cannot
// address a value narrower than a byte and what stride a packed i1 buffer has is a
// question about the device's packing.
tt.func @sub_byte_element(%a: tensor<4x64xf16>, %b: tensor<4x64xf16>) {
  %m = arith.cmpf ogt, %a, %b : tensor<4x64xf16>
  // expected-error @+1 {{a pinned value's element type must occupy whole bytes, and 'i1' is 1 bits; an offset counts elements, which cannot address a value narrower than a byte}}
  %n = arith.andi %m, %m {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xi1>
  tt.return
}

// -----
// Two neighbours stating DIFFERENT layouts, which is the one refusal the layout
// search has. A buffer has one layout and nothing picks between two, so this is
// rejected rather than resolved -- and the reason it cannot be resolved by falling
// back to the logical form is that a buffer's extent is what the author's offset is
// denominated in, so a third answer would make their footprint depend on which
// neighbour the walk happened to reach.
//
// Hand-written because it needs two annotated views and the pinned value between
// them: the source is stick-tiled on dim 1 at the 64-element stick, the destination
// is untiled -- every dim named whole -- and no layout is a layout for both.
//
// The error is on the pin's CARRIER and each note on the access that stated a
// layout, which is what lets an author see the two sides rather than be told the
// answer is ambiguous.
tt.func @neighbours_disagree(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %a = ktdp.construct_memory_view %src, sizes: [4, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<4x64xf16>
  %ta = ktdp.construct_access_tile %a[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  // expected-note @+1 {{one of them is stated here}}
  %x = ktdp.load %ta : <4x64xindex> -> tensor<4x64xf16>
  // expected-error @+1 {{a pinned value's neighbours state different layouts, so the layout of its buffer is not determined}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : tensor<4x64xf16>
  %b = ktdp.construct_memory_view %dst, sizes: [4, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 0, 0>, phys_op = array<i64: 0, 0>, phys_src = array<i64: 0, 1>}} : memref<4x64xf16>
  %tb = ktdp.construct_access_tile %b[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  // expected-note @+1 {{one of them is stated here}}
  ktdp.store %e, %tb : tensor<4x64xf16>, <4x64xindex>
  tt.return
}

// -----
// A `tts.pin` that is not a DICTIONARY at all. `LowerTTSMarkers` writes one, so this
// is only reachable by hand -- which is exactly what this file is, and why readPin
// diagnoses it instead of asserting.
tt.func @not_a_dictionary(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{'tts.pin' is not a dictionary; LowerTTSMarkers writes one}}
  %e = math.exp %x {tts.pin = 42 : i32} : tensor<4x64xf16>
  tt.return
}
