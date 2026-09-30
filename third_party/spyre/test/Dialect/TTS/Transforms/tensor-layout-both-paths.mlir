// RUN: spyre-triton-opt %s --lower-tts-markers | FileCheck %s

// Two markers naming the same value, to pin the `use_empty()` guard on the
// bridge-cast cleanup.
//
// A marker's operand is reached through a builtin.unrealized_conversion_cast
// that exists for the marker's sake, and the driver erases that cast once the
// marker is gone -- but only when nothing else holds it. Two markers on one cast
// is that "something else": lowering the first leaves the cast alive for the
// second, and only the second's lowering takes it. So the erase order cannot
// drop a cast another marker still names, and the dead-cast question stops being
// anyone's to manage.
//
// The second descriptor is on a cast of its own with one marker, so the two
// halves of the guard -- a cast with another user and a cast without -- are both
// in one module. Which annotation a descriptor carries decides its fate; there
// is no flag and no pass option.
//
// The surviving-then-erased pair is also what makes the absence guards in
// tensor-layout-attribute.mlir meaningful: the ops they scan for are emitted in
// exactly the gap they cover.
//
// Not -split-input-file deliberately -- one module is the claim.
//
// Hand-written post-LowerDescriptorMemory KTIR, one pass in the RUN line.

#id = affine_map<(d0, d1) -> (d0, d1)>
#view = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#tile = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>

tt.func @both_paths(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>)
    -> (tensor<32x64xf16>, tensor<32x64xf16>) {
  %c0 = arith.constant 0 : index

  // Two markers on one bridge cast.
  %abase = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  %sharedview = ktdp.construct_memory_view %abase, sizes: [32, 128], strides: [128, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<32x128xf16>
  %shared = builtin.unrealized_conversion_cast %sharedview
      : memref<32x128xf16> to !tt.tensordesc<32x64xf16>
  tts.tensor_layout %shared
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x64xf16>
  tts.tensor_layout %shared
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x64xf16>

  // One marker on its own bridge cast.
  %bbase = builtin.unrealized_conversion_cast %b : !tt.ptr<f16> to index
  %loneview = ktdp.construct_memory_view %bbase, sizes: [32, 128], strides: [128, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<32x128xf16>
  %lone = builtin.unrealized_conversion_cast %loneview
      : memref<32x128xf16> to !tt.tensordesc<32x64xf16>
  tts.tensor_layout %lone
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x64xf16>

  %sharedtile = ktdp.construct_access_tile %sharedview[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #tile}
      : memref<32x128xf16> -> !ktdp.access_tile<32x64xindex>
  %d0 = ktdp.load %sharedtile : <32x64xindex> -> tensor<32x64xf16>
  %lonetile = ktdp.construct_access_tile %loneview[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #tile}
      : memref<32x128xf16> -> !ktdp.access_tile<32x64xindex>
  %d1 = ktdp.load %lonetile : <32x64xindex> -> tensor<32x64xf16>
  tt.return %d0, %d1 : tensor<32x64xf16>, tensor<32x64xf16>
}

// CHECK-LABEL: tt.func @both_paths

// Both views carry the attribute, and neither cast is left: the shared one
// survived its first marker's lowering and went with its second.
// CHECK: %[[SHAREDVIEW:.*]] = ktdp.construct_memory_view {{.*}} {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<32x128xf16>
// CHECK: %[[LONEVIEW:.*]] = ktdp.construct_memory_view {{.*}} {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<32x128xf16>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc

// Each load reads through its own view, so the casts really were plumbing for
// the markers rather than something either load goes through.
// CHECK: %[[SHAREDTILE:.*]] = ktdp.construct_access_tile %[[SHAREDVIEW]]
// CHECK: ktdp.load %[[SHAREDTILE]]
// CHECK: %[[LONETILE:.*]] = ktdp.construct_access_tile %[[LONEVIEW]]
// CHECK: ktdp.load %[[LONETILE]]
// CHECK: tt.return
