// RUN: spyre-triton-opt %s --materialize-pinned-buffers | FileCheck %s
// RUN: spyre-triton-opt %s --materialize-pinned-buffers="lx-capacity-bytes=8192" -verify-diagnostics

// The capacity, driven rather than described, and the point is the FIRST run.
//
// There is no capacity default to fall back on: how big the scratchpad is belongs
// to the device description, and how much of it a kernel may spend on pins is the
// caller's budget, so an unstated one means the question is not asked -- NOT that it
// is answered with a figure the compiler chose. The same pin is therefore placed
// under no budget and refused under a stated one, which is one pin and two answers
// with nothing about the pin differing between them.
//
// It is the pass's ONLY option, and the only numeric rule it takes from a caller.
// Alignment used to be a second one and is gone rather than defaulted: an offset
// counts from a base the scratchpad allocator assigns, so aligning that base is the
// allocator's rule and a stick multiple here would be neither necessary nor
// sufficient.

// CHECK-LABEL: tt.func @past_a_stated_budget
// CHECK: ktdp.construct_memory_view
// CHECK-NOT: tts.pin
tt.func @past_a_stated_budget(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-error @+1 {{pinned range reaches byte 8704 from the base of this kernel's allocation, past the 8192 a core's scratchpad holds in total}}
  %e = math.exp %x {tts.pin = {memory_space = "ct_local", offset = 4096 : i32}} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
