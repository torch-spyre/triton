// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// Where the physical memory view's strides come from.
//
// Physicalizing a ktdp.construct_memory_view restates its sizes from the marker
// and lays the physical shape out row-major. It ignores the view's logical
// strides entirely: a physicalized view addresses stick-tiled device data, whose
// element order is the physical shape's own, so the logical strides describe a
// buffer the view no longer names. Each case below states that by making the
// logical strides visibly not the answer -- [128, 1] giving [8192, 64, 1], and a
// dynamic %stride operand that the physical view never mentions.
//
// The two cases are the same rule at a static shape, then at a dynamic size,
// where the row-major strides are no longer constants and the arithmetic over
// the dynamic extent has to be emitted.
//
// Checks are hand-written and minimal on purpose: the claim is the strides on the
// physical view, not the whole module.

// Case 1 -- static shape.
//
// [M=128, N=128] stick-on-N(64) -> physical [N/64, M, N%64] = [2, 128, 64],
// laid out row-major: [128*64, 64, 1] = [8192, 64, 1].

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL:   tt.func @pointwise_static(
// CHECK:           %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// CHECK:           %[[PHYS:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [2, 128, 64], strides: [8192, 64, 1]
// CHECK-SAME:       memref<2x128x64xf32>
// CHECK:           ktdp.load
// CHECK:           ktdp.store
tt.func @pointwise_static(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [128, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<128x128xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<128x128xf32> to !tt.tensordesc<128x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <128x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
  %al = ktdp.load %at : <128x128xindex> -> tensor<128x128xf32>

  // Store to a second annotated descriptor, to keep the loaded value alive.
  %oi = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [128, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<128x128xf32>
  %od = builtin.unrealized_conversion_cast %ov : memref<128x128xf32> to !tt.tensordesc<128x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <128x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
  ktdp.store %al, %ot : tensor<128x128xf32>, <128x128xindex>
  tt.return
}
}

// -----

// Case 2 -- a dynamic size, and a dynamic logical stride that goes unread.
//
// [%m, 64] stick-on-N(64) gives physical [1, %m, 64]. One dynamic extent makes
// every outer stride dynamic, so the row-major computation has to be *emitted*
// as a muli chain over %m. The logical stride operand %[[STRIDE]] appears in the
// signature and nowhere in the strides.

#id = affine_map<(d0, d1) -> (d0, d1)>
#sdyn = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sblock = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @dynamic_strides(
// CHECK-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M:.*]]: index, %[[STRIDE:.*]]: index)
// CHECK:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
//                Row-major stride computation: c1=1, c64=64, s1=c1*c64=64, s0=s1*M
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C64:.*]] = arith.constant 64 : index
// CHECK:         %[[S1:.*]] = arith.muli %[[C1]], %[[C64]] : index
// CHECK:         %[[S0:.*]] = arith.muli %[[S1]], %[[M]] : index
// CHECK:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: {{\[}}%[[S0]], %[[S1]], %[[C1]]]
// CHECK-SAME:      memref<1x?x64xf32>
// CHECK:         ktdp.load
// CHECK:         ktdp.store
tt.func @dynamic_strides(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>, %m: index, %stride: index) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [%m, 64], strides: [%stride, 1] {coordinate_set = #sdyn, memory_space = #ktdp.memory_space<global>} : memref<?x64xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<?x64xf32> to !tt.tensordesc<128x64xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <128x64xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #sblock} : memref<?x64xf32> -> !ktdp.access_tile<128x64xindex>
  %al = ktdp.load %at : <128x64xindex> -> tensor<128x64xf32>

  %oi = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [%m, 64], strides: [%stride, 1] {coordinate_set = #sdyn, memory_space = #ktdp.memory_space<global>} : memref<?x64xf32>
  %od = builtin.unrealized_conversion_cast %ov : memref<?x64xf32> to !tt.tensordesc<128x64xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <128x64xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #sblock} : memref<?x64xf32> -> !ktdp.access_tile<128x64xindex>
  ktdp.store %al, %ot : tensor<128x64xf32>, <128x64xindex>
  tt.return
}
}
