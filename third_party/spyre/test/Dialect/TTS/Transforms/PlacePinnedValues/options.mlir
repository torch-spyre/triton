// RUN: spyre-triton-opt %s --place-pinned-values=grid=1 | FileCheck %s
// RUN: spyre-triton-opt %s --place-pinned-values=lx-capacity-bytes=0 | FileCheck %s

// The pass's options, driven rather than described.
//
// One pin, and it is PLACED under either option below while being refused under
// the defaults -- see `affine_past_capacity` in invalid.mlir, which is the same
// pin. Nothing about the pin changes between the three, which is the point: the
// capacity of a core's scratchpad and the size of a stick are facts about the
// DEVICE, and this tree deliberately does not hold them.
// `SpyreUtils.get_device_properties` states that the LX scratchpad "is sized by
// the device description and allocated by the scheduler", so they arrive as
// options, and these RUN lines are what keeps that true rather than decorative.
//
// The pin asks for a 512-byte value at `program_id * 65536` elements.
//
//   defaults (grid 32)   32 members, the last at element 2031616, so every core
//                        must reserve 4063744 bytes against 2097152. Refused, in
//                        invalid.mlir. This is the case that shows an affine
//                        address is not free.
//   grid=1               one member, at 0. Fits.
//   lx-capacity-bytes=0  the check is DISABLED, not passed: 0 means "this build
//                        does not know the capacity", which has to be
//                        distinguishable from "the capacity is zero".
//
// No -split-input-file, so no need to avoid rules of dashes here; there are none
// anyway, for consistency with the two files beside this one.

// CHECK-LABEL: tt.func @affine
// CHECK: ktdp.construct_memory_view {{.*}} memory_space = #ktdp.memory_space<ct_local>
// CHECK: ktdp.store
// CHECK: ktdp.load
// CHECK-NOT: tts.pin
tt.func @affine(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %stride = arith.constant 65536 : i32
  %m = arith.muli %pid, %stride : i32
  tts.pin %e, address %m {memory_space = "ct_local"} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
