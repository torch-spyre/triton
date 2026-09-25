// RUN: spyre-triton-opt %s -split-input-file --lower-tts-markers -verify-diagnostics

// A marker whose operand does not resolve to anything that can carry its
// attribute.
//
// This is the per-marker half of the pass stated as a refusal: `tts.tensor_layout`
// admits exactly one resolution -- the lowered-descriptor bridge
// LowerDescriptorMemory leaves, and the ktdp.construct_memory_view behind it --
// and anything else is an error rather than a skip. Silently skipping would take
// the kernel's layout with the marker and leave a logical artifact that looks
// correct; asserting would be wrong for a pass that is invocable on hand-written
// IR, which is what these two cases are.

// A descriptor that was never lowered: the operand is a function argument of
// !tt.tensordesc type, so there is no bridge cast at all. This is also what
// reaching the pass out of order looks like -- before LowerDescriptorMemory, or
// on a descriptor that pass declined because it could not recover the shape.
tt.func @not_lowered(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @+1 {{tts.tensor_layout does not annotate a lowered tensor descriptor: expected its operand to be the builtin.unrealized_conversion_cast that lower-descriptor-memory leaves bridging a memref back to !tt.tensordesc}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// A bridge cast of the right shape over the wrong memref: the operand really is
// an unrealized_conversion_cast from a memref, so the first check passes, but the
// memref is a bare allocation and no memory view defines it. The layout has
// nothing to land on, and the two checks are separate for exactly this reason --
// the first says "is this a bridge", the second says "does it bridge a view".
tt.func @no_memory_view() {
  %alloc = memref.alloc() : memref<64x64xf32>
  %desc = builtin.unrealized_conversion_cast %alloc
      : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
  // expected-error @+1 {{tts.tensor_layout does not annotate a lowered tensor descriptor: its operand bridges a memref that no ktdp.construct_memory_view defines, so there is no memory view to carry the layout}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Two markers reaching one subject. `setAttr` would resolve this by overwriting,
// so the later marker would win and the earlier would be gone with nothing said --
// and which one survived would be a fact about the driver's loop order rather than
// about anything the author wrote.
//
// Two pins on one VALUE is the reachable shape of it: they share a producer, and
// the producer is the carrier. The generic driver holds the check rather than the
// pin, because the hazard is one subject reachable from two markers, which
// `tts.tensor_layout` has its own spelling of -- two layouts on one descriptor
// share a memory view.
tt.func @two_pins_on_one_value(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-note @+1 {{the op both resolve to is here}}
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = 0 : i32} : tensor<4x64xf16>
  // expected-error @+1 {{second tts.pin resolving to the same op, which can carry only one; the first states {address = 0 : i32, memory_space = #ktdp.memory_space<ct_local>}}}
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = 512 : i32} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----

// A producer with several results. An attribute attaches to an OP and not to a
// value, so on a multi-result producer it could not say which result is pinned,
// and taking it to mean the first would be silent. `tt.split` is the multi-result
// op this tree has.
//
// The pinned VALUE is well formed, which is why this is the lowering's rule and
// not the op's: the op holds its value as an operand and can check that: only
// resolution knows which op is about to carry the annotation.
tt.func @multi_result_producer(%t: tensor<8x2xf32>) -> tensor<8xf32> {
  // expected-note @+1 {{the producer is here}}
  %0, %1 = tt.split %t : tensor<8x2xf32> -> tensor<8xf32>
  // expected-error @+1 {{tts.pin names a value whose producer has 2 results, so an attribute on it could not say which one is pinned}}
  tts.pin %0 {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<8xf32>
  tt.return %0 : tensor<8xf32>
}
