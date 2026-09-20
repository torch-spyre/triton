// RUN: spyre-triton-opt %s --lower-tts-markers | FileCheck %s

// The two layout spellings in one module, side by side, to pin that they do not
// interfere.
//
// There is no flag and no pass option selecting between them: this pass walks
// `tts` markers and nothing else, so which annotation a descriptor carries is
// what decides its fate. One module can therefore carry one of each, and each
// behaves as if it were alone:
//
//   tt.spyre_tensor_layout  op and bridge cast both SURVIVE, for the named
//                           --rewrite-descriptor-layout to consume later
//   tts.tensor_layout       becomes the attribute on the memory view; op and
//                           bridge cast both ERASED
//
// The surviving pair is also what makes the absence guards in
// tensor-layout-attribute.mlir meaningful: the ops they scan for are emitted in
// exactly the gap they cover, as the old-path descriptor here demonstrates. And
// the old path's cast surviving is the case the `use_empty()` guard in the
// driver's bridge-cast cleanup is there for -- it is a cast this pass never
// looked at, because no `tts` marker names it.
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

  // The old path.
  %abase = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  %oldview = ktdp.construct_memory_view %abase, sizes: [32, 128], strides: [128, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<32x128xf16>
  %old = builtin.unrealized_conversion_cast %oldview
      : memref<32x128xf16> to !tt.tensordesc<32x64xf16>
  tt.spyre_tensor_layout %old
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x64xf16>

  // The new path.
  %bbase = builtin.unrealized_conversion_cast %b : !tt.ptr<f16> to index
  %newview = ktdp.construct_memory_view %bbase, sizes: [32, 128], strides: [128, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<32x128xf16>
  %new = builtin.unrealized_conversion_cast %newview
      : memref<32x128xf16> to !tt.tensordesc<32x64xf16>
  tts.tensor_layout %new
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x64xf16>

  %oldtile = ktdp.construct_access_tile %oldview[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #tile}
      : memref<32x128xf16> -> !ktdp.access_tile<32x64xindex>
  %d0 = ktdp.load %oldtile : <32x64xindex> -> tensor<32x64xf16>
  %newtile = ktdp.construct_access_tile %newview[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #tile}
      : memref<32x128xf16> -> !ktdp.access_tile<32x64xindex>
  %d1 = ktdp.load %newtile : <32x64xindex> -> tensor<32x64xf16>
  tt.return %d0, %d1 : tensor<32x64xf16>, tensor<32x64xf16>
}

// CHECK-LABEL: tt.func @both_paths

// The old path, untouched: no attribute on the view, and the bridge cast is
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
// plumbing for the old marker rather than something either load goes through.
// CHECK: %[[OLDTILE:.*]] = ktdp.construct_access_tile %[[OLDVIEW]]
// CHECK: ktdp.load %[[OLDTILE]]
// CHECK: %[[NEWTILE:.*]] = ktdp.construct_access_tile %[[NEWVIEW]]
// CHECK: ktdp.load %[[NEWTILE]]
// CHECK: tt.return
