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
