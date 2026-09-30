// RUN: spyre-triton-opt %s -split-input-file --lower-tts-markers | FileCheck %s

// The `tts.tensor_layout` marker becoming the `tts.tensor_layout` attribute.
//
// Three things happen to an annotated descriptor, and all three are the point:
//
//   1. the marker's three coordinate arrays land on the
//      ktdp.construct_memory_view its operand resolves to, as the
//      `tts.tensor_layout` attribute;
//   2. the marker op is erased;
//   3. so is the builtin.unrealized_conversion_cast bridging the view's memref
//      back to !tt.tensordesc, which the marker was the last user of.
//
// (3) is the one worth asserting negatively, and every case below does. That
// cast exists only to keep a !tt.tensordesc operand alive, so nothing wants it
// once the marker that needed the type is gone -- but a pass that annotated the
// view and left the bridge standing would look correct in every positive CHECK
// here. The absence guards sit between the memory view and the tt.return, inside
// the function body, so they are not in the module's empty tail where they could
// never fire; tensor-layout-both-paths.mlir is where a survivor is shown
// appearing in exactly that gap.
//
// The surviving `!tt.ptr -> index` cast is a different cast with a different
// lifetime: LowerDescriptorMemory builds it for the base pointer and
// ConvertFunctions consumes it much later. So the guards name the tensordesc
// direction specifically rather than banning unrealized_conversion_cast outright.
//
// Input is hand-written post-LowerDescriptorMemory KTIR with one pass in the RUN
// line, so what the pass under test receives is in the file. Access-tile
// subscripts are `index` constants, which is what the op's own definition calls
// for; the arith.index_cast chain the pass ahead would have produced is an
// artifact of that pass and not of this one.
//
// The attribute's entries come back in the printer's sorted order (phys_arg,
// phys_op, phys_src) rather than the order the marker spells them.

// Static shapes. The view's sizes are the FULL tensor extent (128x256) while the
// descriptor's block type is the tile (128x64) -- the layout describes the
// former, and the attribute verifier measures phys_src against the view's rank.
#id = affine_map<(d0, d1) -> (d0, d1)>
#view = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#tile = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>

tt.func @static_2d(%ptr: !tt.ptr<f16>) -> tensor<128x64xf16> {
  %c0 = arith.constant 0 : index
  %base = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  %view = ktdp.construct_memory_view %base, sizes: [128, 256], strides: [256, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<128x256xf16>
  %desc = builtin.unrealized_conversion_cast %view
      : memref<128x256xf16> to !tt.tensordesc<128x64xf16>
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf16>
  %t = ktdp.construct_access_tile %view[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #tile}
      : memref<128x256xf16> -> !ktdp.access_tile<128x64xindex>
  %d = ktdp.load %t : <128x64xindex> -> tensor<128x64xf16>
  tt.return %d : tensor<128x64xf16>
}

// CHECK-LABEL: tt.func @static_2d
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f16> to index
// CHECK: %[[VIEW:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [128, 256], strides: [256, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<128x256xf16>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc
// CHECK: ktdp.construct_access_tile %[[VIEW]]
// CHECK: ktdp.load
// CHECK: tt.return

// -----
// Dynamic shapes. Same three outcomes; the layout is independent of whether the
// extents are constants, since it names logical DIMS and not their sizes.
#id = affine_map<(d0, d1) -> (d0, d1)>
#view = affine_set<(d0, d1)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)>
#tile = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>

tt.func @dynamic_2d(%ptr: !tt.ptr<f16>, %M: index, %N: index, %sr: index, %sc: index)
    -> tensor<128x64xf16> {
  %c0 = arith.constant 0 : index
  %base = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  %view = ktdp.construct_memory_view %base, sizes: [%M, %N], strides: [%sr, %sc]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<?x?xf16>
  %desc = builtin.unrealized_conversion_cast %view
      : memref<?x?xf16> to !tt.tensordesc<128x64xf16>
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf16>
  %t = ktdp.construct_access_tile %view[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #tile}
      : memref<?x?xf16> -> !ktdp.access_tile<128x64xindex>
  %d = ktdp.load %t : <128x64xindex> -> tensor<128x64xf16>
  tt.return %d : tensor<128x64xf16>
}

// CHECK-LABEL: tt.func @dynamic_2d
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f16> to index
// CHECK: %[[VIEW:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: {{\[}}%{{.*}}, %{{.*}}], strides: {{\[}}%{{.*}}, %{{.*}}] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<?x?xf16>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc
// CHECK: ktdp.construct_access_tile %[[VIEW]]
// CHECK: ktdp.load
// CHECK: tt.return

// -----
// Two annotated descriptors in one kernel, with DIFFERENT layouts: the input is
// stick-tiled on dim 1, the output is identity. Each attribute has to land on
// its own view, which a pass keying off "the kernel's layout" rather than off
// the marked value would get wrong -- and would get wrong invisibly, since both
// views would still verify.
#id = affine_map<(d0, d1) -> (d0, d1)>
#view = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>

tt.func @two_descriptors(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %inbase = builtin.unrealized_conversion_cast %in : !tt.ptr<f32> to index
  %inview = ktdp.construct_memory_view %inbase, sizes: [64, 128], strides: [128, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<64x128xf32>
  %src = builtin.unrealized_conversion_cast %inview
      : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tts.tensor_layout %src
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %outbase = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %outview = ktdp.construct_memory_view %outbase, sizes: [64, 128], strides: [128, 1]
      {coordinate_set = #view, memory_space = #ktdp.memory_space<global>}
      : memref<64x128xf32>
  %dst = builtin.unrealized_conversion_cast %outview
      : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tts.tensor_layout %dst
    {phys_src = array<i64: 0, 1>,
     phys_op = array<i64: 0, 0>,
     phys_arg = array<i64: 0, 0>} : !tt.tensordesc<64x128xf32>
  %intile = ktdp.construct_access_tile %inview[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #view}
      : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %d = ktdp.load %intile : <64x128xindex> -> tensor<64x128xf32>
  %outtile = ktdp.construct_access_tile %outview[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #view}
      : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %d, %outtile : tensor<64x128xf32>, <64x128xindex>
  tt.return
}

// CHECK-LABEL: tt.func @two_descriptors
// CHECK: %[[INBASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// CHECK: %[[INVIEW:.*]] = ktdp.construct_memory_view %[[INBASE]], sizes: [64, 128], strides: [128, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf32>
// CHECK: %[[OUTBASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// CHECK: %[[OUTVIEW:.*]] = ktdp.construct_memory_view %[[OUTBASE]], sizes: [64, 128], strides: [128, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 0, 0>, phys_op = array<i64: 0, 0>, phys_src = array<i64: 0, 1>}} : memref<64x128xf32>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile %[[INVIEW]]
// CHECK: %[[VAL:.*]] = ktdp.load %[[TILE]]
// CHECK: %[[OUTTILE:.*]] = ktdp.construct_access_tile %[[OUTVIEW]]
// CHECK: ktdp.store %[[VAL]], %[[OUTTILE]]
// CHECK: tt.return
