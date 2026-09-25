// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// The tts.pin OP's verifier. No pass runs here.
//
// The address is an ATTRIBUTE, and that removes most of what this file used to
// test rather than relocating it. A shape rule needs an expression to have a
// shape: when the address was an SSA operand, a run-time value, a non-constant
// coefficient, a `program_id` on the wrong axis and a sum of two `program_id`
// terms were all well-formed arith that had to be refused one at a time. None of
// them is spellable in an attribute, so none of them needs a rule.
//
// What is left is what a type constraint cannot say. ODS already narrows the
// address to an i32 or an i32 array; the verifier adds the memory space, checked
// against ktdp's enum, and the one array shape that parses but names nothing.
//
// The numeric rules -- a stick-aligned offset, a range that fits the scratchpad,
// ranges that do not overlap -- are arithmetic on those numbers and belong to
// whatever consumes the attribute. Nothing does yet.

// The memory-space vocabulary is ktdp's MemorySpaceKind, reached through
// `symbolizeMemorySpaceKind` so the two names are never restated in C++.
tt.func @unknown_memory_space(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{unknown memory space 'lx': expected 'ct_local'}}
  tts.pin %e {memory_space = "lx"} : tensor<4x64xf16>
  tt.return
}

// -----
// Case matters, since the enum's own spelling is lower case.
tt.func @wrong_case(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{unknown memory space 'CT_LOCAL'}}
  tts.pin %e {memory_space = "CT_LOCAL"} : tensor<4x64xf16>
  tt.return
}

// -----
// `global` is a KNOWN kind and still not pinnable, so it gets its own message
// rather than being reported as a misspelling. lx-placement.md's first
// assumption puts HBM intermediates outside a pin: one is written as a
// tl.make_tensor_descriptor with an explicit store and load, and nothing here
// allocates an anonymous device buffer.
tt.func @global_memory_space(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{memory space 'global' cannot be pinned: only 'ct_local' is}}
  tts.pin %e {memory_space = "global"} : tensor<4x64xf16>
  tt.return
}

// -----
// And with an address too, so the message is the space's either way rather than
// changing depending on what else the pin carries.
tt.func @global_with_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{memory space 'global' cannot be pinned}}
  tts.pin %e {memory_space = "global", address = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// An EMPTY address array parses and names no address for any core, so it is
// neither spelling: a consumer indexing it by the program id reads out of bounds
// on the first core. The only address rule a type constraint cannot express.
tt.func @empty_address_array(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{address array is empty}}
  tts.pin %e {memory_space = "ct_local", address = array<i32>} : tensor<4x64xf16>
  tt.return
}

// -----
// A dynamic extent cannot be placed: there is no buffer size, so no capacity
// answer and nothing to build a view over. Refused by the operand's type
// constraint rather than by a hand-written rule.
tt.func @dynamic_shape(%x: tensor<?x64xf16>) {
  %e = math.exp %x : tensor<?x64xf16>
  // expected-error @+1 {{operand #0 must be statically shaped tensor of any type values}}
  tts.pin %e {memory_space = "ct_local", address = 4096 : i32} : tensor<?x64xf16>
  tt.return
}

// The forms that DO verify, so the rejections above are read as rules and not as
// the op being hard to satisfy.
//
// Note there are no rules of dashes anywhere in this file, and no comment quotes
// the split marker either: -split-input-file matches the marker as a substring,
// so both would start a new chunk and the text after them would be parsed as IR.

// -----
tt.func @accepted_forms(%x: tensor<4x64xf16>) {
  %e0 = math.exp %x : tensor<4x64xf16>
  %e1 = math.exp %x : tensor<4x64xf16>
  %e2 = math.exp %x : tensor<4x64xf16>

  // One i32: the same address on every core.
  tts.pin %e0 {memory_space = "ct_local", address = 4096 : i32} : tensor<4x64xf16>

  // One per program id, positionally. Nothing here requires them to be
  // increasing, evenly spaced, or distinct -- those are a consumer's questions.
  tts.pin %e1 {memory_space = "ct_local", address = array<i32: 4096, 6144, 8192>} : tensor<4x64xf16>

  // No address at all: the design's baseline, where the compiler places every
  // intermediate and a pin only names the space.
  tts.pin %e2 {memory_space = "ct_local"} : tensor<4x64xf16>

  tt.return
}
