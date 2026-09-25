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
// What is left is what a type constraint cannot say. ODS narrows the address to
// an i32 or an i32 array, and the memory space to a `#ktdp.memory_space`; the
// verifier adds the rules that are about MEANING rather than shape: what may be
// pinned at all, which kind a pin may name, that it names no core, that it states
// an address, and the one array shape that parses but addresses nothing.
//
// Note what is NOT here any more: a misspelled memory space. It used to be a
// string checked against ktdp's enum, so `"lx"` and `"CT_LOCAL"` were verifier
// errors; typed as the attribute they are parse errors, caught before this
// verifier runs and not worth a case of their own. The first case below is what
// remains of them.
//
// The numeric rules -- a stick-aligned offset, a range that fits the scratchpad,
// ranges that do not overlap -- are arithmetic on those numbers and belong to
// whatever consumes the attribute. Nothing does yet.

// Not a `#ktdp.memory_space` at all. Refused by the ODS constraint, which is the
// whole of what used to be two hand-written cases about misspelling.
tt.func @not_a_memory_space(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{attribute 'memory_space' failed to satisfy constraint}}
  tts.pin %e {memory_space = "ct_local"} : tensor<4x64xf16>
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
  tts.pin %e {memory_space = #ktdp.memory_space<global>} : tensor<4x64xf16>
  tt.return
}

// -----
// And with an address too, so the message is the space's either way rather than
// changing depending on what else the pin carries.
tt.func @global_with_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{memory space 'global' cannot be pinned}}
  tts.pin %e {memory_space = #ktdp.memory_space<global>, address = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// A ct_id, which the attribute can carry and a pin may not use. `ct_local` alone
// means the scratchpad of whichever core is running; naming core 7 is a different
// request, and one nothing here honours -- so it is refused rather than accepted
// and then ignored, which would put the buffer somewhere the author did not ask
// for with nothing saying so. This rule is only expressible because the memory
// space is the typed attribute: a string could not carry a ct_id to reject.
tt.func @ct_id_specified(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{memory space names ct_id 7; a pin is always the running core's own scratchpad}}
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local, ct_id = 7>, address = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// An EMPTY address array parses and names no address for any core, so it is
// neither spelling: a consumer indexing it by the program id reads out of bounds
// on the first core. The only address rule a type constraint cannot express.
tt.func @empty_address_array(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{address array is empty}}
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = array<i32>} : tensor<4x64xf16>
  tt.return
}

// -----
// A dynamic extent cannot be placed: there is no buffer size, so no capacity
// answer and nothing to build a view over. Refused by the operand's type
// constraint rather than by a hand-written rule.
tt.func @dynamic_shape(%x: tensor<?x64xf16>) {
  %e = math.exp %x : tensor<?x64xf16>
  // expected-error @+1 {{operand #0 must be statically shaped tensor of any type values}}
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<?x64xf16>
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

  // One i32: the same address on every core.
  tts.pin %e0 {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<4x64xf16>

  // One per program id, positionally. Nothing here requires them to be
  // increasing, evenly spaced, or distinct -- those are a consumer's questions.
  tts.pin %e1 {memory_space = #ktdp.memory_space<ct_local>, address = array<i32: 4096, 6144, 8192>} : tensor<4x64xf16>

  tt.return
}

// -----
// A BLOCK ARGUMENT. Refused because no op produces it, so nothing can carry the
// annotation -- and the message says only that, because there are two kinds of
// block argument and they are unalike:
//
//   an ENTRY input        is global, at the address its base pointer supplies.
//                         Pinning it asks to relocate a kernel argument.
//   a LOOP-CARRIED value  is an `scf.for` iter_arg -- an attention accumulator is
//                         one, see RewriteDescriptorLayout/parallel-scatter-iter-arg.mlir
//                         -- and is NOT global. It is a live intermediate and a
//                         plausible thing to pin; it is refused because nothing
//                         has decided what pinning one means, one buffer reused
//                         each iteration or one per iteration. Its RESULT, outside
//                         the loop, has a producer and is pinnable.
//
// Asserting "entry input" would therefore be false half the time, which is why the
// diagnostic names the shared fact and points at the loop-result workaround.
tt.func @block_argument(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{names a block argument, which no op produces, so there is nothing to carry the annotation}}
  tts.pin %x {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// No address. The field is OPTIONAL in ODS and required here, which is the honest
// statement of what exists: the unaddressed form is the design's baseline -- the
// compiler places every intermediate and a pin only overrides where -- but nothing
// in this tree can act on one, so accepting it would put an annotation in the
// artifact that no consumer could honour. Offset 0 is not the fallback; it is a
// real address that would collide with the scheduler's own pool.
//
// Keeping the field optional is what lets this refusal be lifted without the
// surface changing shape.
tt.func @no_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{no address, and nothing here can choose one}}
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
  tt.return
}
