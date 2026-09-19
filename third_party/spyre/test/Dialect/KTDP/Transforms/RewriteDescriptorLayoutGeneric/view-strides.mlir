// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic=data-layout=host -split-input-file | FileCheck %s --check-prefix=HOST
// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s --check-prefix=DEVICE

// Where the physical memory view's strides come from.
//
// Physicalizing a ktdp.construct_memory_view restates its sizes from the marker,
// and the data-layout option decides what to do about the strides:
// data-layout=host derives each physical stride from the view's LOGICAL strides
// through the coordinate map, so the physical view still addresses host-order
// data; data-layout=device (the default) ignores the logical strides entirely
// and lays the physical shape out row-major.
//
// Every module here is checked under BOTH prefixes, so neither mode's checks can
// pass by matching the other's output -- which is the only way to state a
// difference between two whole-view stride computations. The three cases are the
// same rule at a static shape, then at a dynamic size and a dynamic logical
// stride, where device mode has to emit stride arithmetic that host mode can read
// straight off the input.
//
// Checks are hand-written and minimal on purpose: the claim is the strides on the
// physical view, not the whole module.

// Case 1 -- static shape.
//
// [M=128, N=128] stick-on-N(64) -> physical [N/64, M, N%64] = [2, 128, 64]:
//   host:   dim0 = logStride[1]*64 = 64, dim1 = logStride[0] = 128, dim2 = 1
//   device: row-major over [2, 128, 64] = [8192, 64, 1]

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// HOST-LABEL:   tt.func @pointwise_static(
// HOST:           %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// HOST:           %[[PHYS:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [2, 128, 64], strides: [64, 128, 1]
// HOST-SAME:        memref<2x128x64xf32>
// HOST:           ktdp.load
// HOST:           ktdp.store
//
// DEVICE-LABEL:   tt.func @pointwise_static(
// DEVICE:           %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// DEVICE:           %[[PHYS:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [2, 128, 64], strides: [8192, 64, 1]
// DEVICE-SAME:       memref<2x128x64xf32>
// DEVICE:           ktdp.load
// DEVICE:           ktdp.store
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

// Case 2 -- a dynamic size and a dynamic logical stride, checked with the device
// prefix listed first.
//
// [%m, 64] stick-on-N(64) gives physical [1, %m, 64]. Device mode has to *emit*
// the row-major stride arithmetic (a muli chain over %m), while host mode threads
// the dynamic logical stride straight into dim 1.

#id = affine_map<(d0, d1) -> (d0, d1)>
#sdyn = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sblock = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// DEVICE-LABEL: tt.func @dynamic_strides_device_first(
// DEVICE-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M:.*]]: index, %[[STRIDE:.*]]: index)
// DEVICE:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
//                 Row-major stride computation: c1=1, c64=64, s1=c1*c64=64, s0=s1*M
// DEVICE:         %[[C1:.*]] = arith.constant 1 : index
// DEVICE:         %[[C64:.*]] = arith.constant 64 : index
// DEVICE:         %[[S1:.*]] = arith.muli %[[C1]], %[[C64]] : index
// DEVICE:         %[[S0:.*]] = arith.muli %[[S1]], %[[M]] : index
// DEVICE:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: {{\[}}%[[S0]], %[[S1]], %[[C1]]]
// DEVICE-SAME:      memref<1x?x64xf32>
// DEVICE:         ktdp.load
// DEVICE:         ktdp.store
//
// HOST-LABEL: tt.func @dynamic_strides_device_first(
// HOST-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M:.*]]: index, %[[STRIDE:.*]]: index)
// HOST:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
// HOST:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: [64, %[[STRIDE]], 1]
// HOST-SAME:      memref<1x?x64xf32>
// HOST:         ktdp.load
// HOST:         ktdp.store
tt.func @dynamic_strides_device_first(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>, %m: index, %stride: index) {
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

// -----

// Case 3 -- the same descriptor as case 2, checked with the host prefix listed
// first.

#id = affine_map<(d0, d1) -> (d0, d1)>
#sdyn = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sblock = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// HOST-LABEL: tt.func @dynamic_strides_host_first(
// HOST-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M:.*]]: index, %[[STRIDE:.*]]: index)
// HOST:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
// HOST:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: [64, %[[STRIDE]], 1]
// HOST-SAME:      memref<1x?x64xf32>
// HOST:         ktdp.load
// HOST:         ktdp.store
//
// DEVICE-LABEL: tt.func @dynamic_strides_host_first(
// DEVICE-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M:.*]]: index, %[[STRIDE:.*]]: index)
// DEVICE:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
// DEVICE:         %[[C1:.*]] = arith.constant 1 : index
// DEVICE:         %[[C64:.*]] = arith.constant 64 : index
// DEVICE:         %[[S1:.*]] = arith.muli %[[C1]], %[[C64]] : index
// DEVICE:         %[[S0:.*]] = arith.muli %[[S1]], %[[M]] : index
// DEVICE:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: {{\[}}%[[S0]], %[[S1]], %[[C1]]]
// DEVICE-SAME:      memref<1x?x64xf32>
// DEVICE:         ktdp.load
// DEVICE:         ktdp.store
tt.func @dynamic_strides_host_first(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>, %m: index, %stride: index) {
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
