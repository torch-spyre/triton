// RUN: spyre-triton-opt %s --lower-descriptor-memory | FileCheck %s

// The two layout spellings in one module, side by side, to pin that they do not
// interfere.
//
// There is no flag and no pass option selecting between them: the discriminator
// is which op annotates the descriptor, and it is read per descriptor. So one
// module can carry one of each, and each behaves as if it were alone:
//
//   tt.spyre_tensor_layout  op and bridge cast both SURVIVE, for the named
//                           --rewrite-descriptor-layout to consume later
//   tts.tensor_layout       becomes the attribute on the memory view; op and
//                           bridge cast both ERASED
//
// Read together with tensor-layout-attribute.mlir, this is also what makes that
// file's absence guards meaningful: the ops they scan for are emitted in exactly
// the gap they cover, as the old-path descriptor here demonstrates.
//
// Not -split-input-file deliberately -- one module is the claim.

tt.func @both_paths(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>)
    -> (tensor<32x64xf16>, tensor<32x64xf16>) {
  %m = arith.constant 32 : i32
  %n = arith.constant 128 : i32
  %sr = arith.constant 128 : i64
  %sc = arith.constant 1 : i64
  %c0 = arith.constant 0 : i32
  %old = tt.make_tensor_descriptor %a, [%m, %n], [%sr, %sc]
      : !tt.ptr<f16>, !tt.tensordesc<32x64xf16>
  tt.spyre_tensor_layout %old
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x64xf16>
  %new = tt.make_tensor_descriptor %b, [%m, %n], [%sr, %sc]
      : !tt.ptr<f16>, !tt.tensordesc<32x64xf16>
  tts.tensor_layout %new
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x64xf16>
  %d0 = tt.descriptor_load %old[%c0, %c0] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
  %d1 = tt.descriptor_load %new[%c0, %c0] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
  tt.return %d0, %d1 : tensor<32x64xf16>, tensor<32x64xf16>
}

// CHECK-LABEL: tt.func @both_paths

// The old path, unchanged: no attribute on the view, and the bridge cast is
// still there with the marker still using it.
// CHECK: %[[OLDVIEW:.*]] = ktdp.construct_memory_view {{.*}} {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>} : memref<32x128xf16>
// CHECK-NEXT: %[[BRIDGE:.*]] = builtin.unrealized_conversion_cast %[[OLDVIEW]] : memref<32x128xf16> to !tt.tensordesc<32x64xf16>
// CHECK-NEXT: tt.spyre_tensor_layout %[[BRIDGE]] {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <32x64xf16>

// The new path, in the same function: the attribute is on the view, and nothing
// else of the annotation is left. The absence guards are anchored here rather
// than at the end of the module, since the old path's cast above proves the gap
// is where a survivor appears.
// CHECK: %[[NEWVIEW:.*]] = ktdp.construct_memory_view {{.*}} {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<32x128xf16>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc

// Both loads read through their own view, so the surviving cast really is dead
// plumbing for the marker rather than something either load goes through.
// CHECK: %[[OLDTILE:.*]] = ktdp.construct_access_tile %[[OLDVIEW]]
// CHECK: ktdp.load %[[OLDTILE]]
// CHECK: %[[NEWTILE:.*]] = ktdp.construct_access_tile %[[NEWVIEW]]
// CHECK: ktdp.load %[[NEWTILE]]
// CHECK: tt.return
