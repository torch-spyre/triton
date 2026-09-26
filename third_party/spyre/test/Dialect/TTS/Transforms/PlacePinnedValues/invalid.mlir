// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics --place-pinned-values

// What PlacePinnedValues refuses, which is every rule about a pin that an op
// cannot answer on its own.
//
// Three kinds, and the split from the op's verifier is the reason each is here:
//
//   * the NUMERIC rules -- stick alignment, scratchpad capacity, and no two
//     pinned ranges intersecting. All three read a hardware number this tree does
//     not otherwise hold, or the launch grid, and the pass takes both as options
//     while an op can see neither;
//   * a pin with NO address, which the op admits because the design does and the
//     pass declines because nothing here allocates;
//   * the two rules about a pin's relation to the rest of the function -- one
//     buffer per value, and a use the pin can actually reach.
//
// Note there are no rules of dashes anywhere in this file, and no comment quotes
// the split marker: -split-input-file matches it as a substring.

// Stick alignment, on the BASE. 100 elements at fp16 is byte 200, and a stick is
// 128 bytes. Checked on the coefficients rather than on the grid's worth of
// addresses, since every member is aligned exactly when base and stride are.
tt.func @misaligned_base(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %a = arith.constant 100 : i32
  // expected-error @+1 {{pinned address base 100 is 200 bytes, which is not a multiple of 128 (one stick)}}
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// And on the STRIDE, with an aligned base -- so the message has to name which
// coefficient failed rather than just the address.
tt.func @misaligned_stride(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %stride = arith.constant 100 : i32
  %base = arith.constant 4096 : i32
  %m = arith.muli %pid, %stride : i32
  %a = arith.addi %base, %m : i32
  // expected-error @+1 {{pinned address stride 100 is 200 bytes, which is not a multiple of 128}}
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// Capacity: 1024x2048 at fp16 is 4 MiB against the 2 MiB a core's scratchpad
// holds.
tt.func @past_capacity(%x: tensor<1024x2048xf16>) {
  %e = math.exp %x : tensor<1024x2048xf16>
  %a = arith.constant 0 : i32
  // expected-error @+1 {{pinned range reaches 4194304 bytes, past the 2097152 a core's scratchpad holds}}
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<1024x2048xf16>
  tt.return
}

// -----
// Capacity again, and the case that shows why an affine address is not free: the
// value itself is 512 bytes, and the pin still overruns, because a `program_id`
// term moves each core's own value within that core's own scratchpad and no core
// knows at compile time which member it will use. So every core reserves the
// whole set: 32 cores times a 65536-element stride.
tt.func @affine_past_capacity(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %stride = arith.constant 65536 : i32
  %m = arith.muli %pid, %stride : i32
  // expected-error @+1 {{makes every core reserve the whole set}}
  tts.pin %e, address %m {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// Two pinned ranges that intersect. The second pin is the error and the first
// gets a remark, since which of the two is wrong is the author's to decide.
//
// This one is a TEMPORARY restriction and not a rule of the design: reusing one
// scratchpad region for two values whose live ranges do not overlap is
// legitimate, and refusing every intersection forbids it. The case is here to pin
// today's behaviour, so relaxing the rule is expected to change it.
tt.func @overlapping_pins(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %a0 = arith.constant 0 : i32
  // expected-remark @+1 {{the other pin is here}}
  tts.pin %e, address %a0 {memory_space = "ct_local"} : tensor<4x64xf16>
  %s = math.sqrt %e : tensor<4x64xf16>
  %a1 = arith.constant 128 : i32
  // expected-error @+1 {{pinned range [128, 384) overlaps another pin's [0, 256)}}
  tts.pin %s, address %a1 {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// One buffer per value. Two pins on one value name two buffers and nothing
// decides which the uses read, so this is refused rather than resolved by walk
// order.
tt.func @pinned_twice(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %a0 = arith.constant 0 : i32
  // expected-remark @+1 {{first pinned here}}
  tts.pin %e, address %a0 {memory_space = "ct_local"} : tensor<4x64xf16>
  %a1 = arith.constant 4096 : i32
  // expected-error @+1 {{value is pinned twice; a value has one buffer}}
  tts.pin %e, address %a1 {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// -----
// A use ABOVE the pin. Ordinary Python reaches this -- `e = tl.exp(x)`,
// `y = tl.sqrt(e)`, then `tl.spyre_pin(e, ...)` traces to exactly the module
// below -- so it is an author's mistake to diagnose and not a hand-written-IR
// curiosity.
//
// The alternative is a positional reading: uses after the pin read the buffer,
// uses before it read the register. Coherent, and declined, because it splits one
// value across a register path and a memory path and so makes the pin's
// observable effect -- severing the producer from its consumers -- hold for only
// some of them.
tt.func @use_above_the_pin(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  %a = arith.constant 4096 : i32
  // expected-error @+1 {{a use of the pinned value is not dominated by this pin}}
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// No address. The op admits it -- that is the design's baseline, in which the
// compiler places every intermediate -- and this pass declines it, because
// nothing in this tree allocates a buffer that is not a kernel argument. Offset 0
// would be a fabricated address rather than an allocation, and would collide with
// the scheduler's own pool.
//
// This is the whole of what an unaddressed pin means today, and the op's side of
// the same question is `global`, which the VERIFIER refuses outright -- see
// Dialect/TTS/IR/pin-op-verifier.mlir. The two halves are deliberately in
// different files: one is answerable from the op and one needs an allocator.
tt.func @no_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{a pin with no address cannot be placed}}
  tts.pin %e {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}
