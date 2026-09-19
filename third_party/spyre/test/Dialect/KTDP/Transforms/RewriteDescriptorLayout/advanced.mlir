// RUN: spyre-triton-opt %s --rewrite-descriptor-layout=data-layout=host -split-input-file | FileCheck %s --check-prefix=HOST
// RUN: spyre-triton-opt %s --rewrite-descriptor-layout -split-input-file | FileCheck %s --check-prefix=DEVICE

// -----

// Test: data-layout=host uses logical strides (derived from logical strides via coord map)
// instead of row-major physical strides.
// For [M=128, N=128] stick-on-N with stick_size=64 -> physical [N/64, M, N%64] = [2, 128, 64]:
//   Logical strides: [128, 1]
//   Host-mode physical strides:
//     dim 0 (src=1, floordiv 64): logStride[1]*64 = 1*64 = 64
//     dim 1 (src=0, identity):    logStride[0] = 128
//     dim 2 (src=1, mod 64):      logStride[1] = 1
//   => strides: [64, 128, 1]
//   Device-mode row-major strides: [2*128*64/(2)=8192, 64, 1] = [8192, 64, 1]

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// HOST-LABEL:   tt.func @pointwise_host_layout(
// HOST:           %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// HOST:           %[[PHYS:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [2, 128, 64], strides: [64, 128, 1]
// HOST-SAME:        memref<2x128x64xf32>
// HOST:           ktdp.load
// HOST:           ktdp.store
//
// DEVICE-LABEL:   tt.func @pointwise_host_layout(
// DEVICE:           %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// DEVICE:           %[[PHYS:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [2, 128, 64], strides: [8192, 64, 1]
// DEVICE-SAME:       memref<2x128x64xf32>
// DEVICE:           ktdp.load
// DEVICE:           ktdp.store
  tt.func @pointwise_host_layout(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    // [M=128, N=128] stick-on-N with stick_size=64
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [128, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<128x128xf32> to !tt.tensordesc<128x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
    %4 = ktdp.load %3 : <128x128xindex> -> tensor<128x128xf32>
    // Store to physical-annotated output to keep alive.
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [128, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x128xf32>
    %7 = builtin.unrealized_conversion_cast %6 : memref<128x128xf32> to !tt.tensordesc<128x128xf32>
    tt.spyre_tensor_layout %7 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x128xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
    ktdp.store %4, %8 : tensor<128x128xf32>, <128x128xindex>
    tt.return
  }
}

// -----

// Test: dynamic M size and dynamic stride with stick-on-N(64), data-layout=device (default).
// Physical shape: [N/64, M, N%64] = [1, %M, 64] (N=64 so N/64=1).
// Device-mode computes row-major strides dynamically:
//   dim2 = 1, dim1 = 1*64 = 64, dim0 = 64 * %M
// Host-mode derives strides from logical stride:
//   dim 0 (src=1, floordiv 64): logStride[1]*64 = 1*64 = 64
//   dim 1 (src=0, identity):    logStride[0] = %stride_m (passed through)
//   dim 2 (src=1, mod 64):      logStride[1] = 1
//   => strides: [64, %stride_m, 1]

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// DEVICE-LABEL: tt.func @dynamic_strides_device(
// DEVICE-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M_I32:.*]]: i32, %[[STRIDE_I64:.*]]: i64)
// DEVICE:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
// DEVICE:         %[[M:.*]] = arith.index_cast %[[M_I32]] : i32 to index
//                 Row-major stride computation: c1=1, c64=64, s1=c1*c64=64, s0=s1*M
// DEVICE:         %[[C1:.*]] = arith.constant 1 : index
// DEVICE:         %[[C64:.*]] = arith.constant 64 : index
// DEVICE:         %[[S1:.*]] = arith.muli %[[C1]], %[[C64]] : index
// DEVICE:         %[[S0:.*]] = arith.muli %[[S1]], %[[M]] : index
// DEVICE:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: [%[[S0]], %[[S1]], %[[C1]]]
// DEVICE-SAME:      memref<1x?x64xf32>
// DEVICE:         ktdp.load
// DEVICE:         ktdp.store
//
// HOST-LABEL: tt.func @dynamic_strides_device(
// HOST-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M_I32:.*]]: i32, %[[STRIDE_I64:.*]]: i64)
// HOST:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
// HOST:         %[[M:.*]] = arith.index_cast %[[M_I32]] : i32 to index
// HOST:         %[[STRIDE:.*]] = arith.index_cast %[[STRIDE_I64]] : i64 to index
// HOST:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: [64, %[[STRIDE]], 1]
// HOST-SAME:      memref<1x?x64xf32>
// HOST:         ktdp.load
// HOST:         ktdp.store
  tt.func @dynamic_strides_device(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i64) {
    %c0 = arith.constant 0 : index
    // [M, 64] with dynamic M and dynamic stride_m, stick-on-N(64)
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg3 : i64 to index
    %3 = ktdp.construct_memory_view %0, sizes: [%1, 64], strides: [%2, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<?x64xf32>
    %4 = builtin.unrealized_conversion_cast %3 : memref<?x64xf32> to !tt.tensordesc<128x64xf32>
    tt.spyre_tensor_layout %4 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x64xf32>
    %5 = ktdp.construct_access_tile %3[%c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<?x64xf32> -> !ktdp.access_tile<128x64xindex>
    %6 = ktdp.load %5 : <128x64xindex> -> tensor<128x64xf32>
    // Store to keep alive
    %7 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %arg3 : i64 to index
    %10 = ktdp.construct_memory_view %7, sizes: [%8, 64], strides: [%9, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<?x64xf32>
    %11 = builtin.unrealized_conversion_cast %10 : memref<?x64xf32> to !tt.tensordesc<128x64xf32>
    tt.spyre_tensor_layout %11 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x64xf32>
    %12 = ktdp.construct_access_tile %10[%c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<?x64xf32> -> !ktdp.access_tile<128x64xindex>
    ktdp.store %6, %12 : tensor<128x64xf32>, <128x64xindex>
    tt.return
  }
}

// -----

// Test: dynamic M size and dynamic stride with stick-on-N(64), data-layout=host.
// Same descriptor as above but named differently to verify host-mode strides independently.
// Host-mode strides: [64, %stride_m, 1]
// Device-mode strides: [64*%M, 64, 1] (row-major)

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// HOST-LABEL: tt.func @dynamic_strides_host(
// HOST-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M_I32:.*]]: i32, %[[STRIDE_I64:.*]]: i64)
// HOST:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
// HOST:         %[[M:.*]] = arith.index_cast %[[M_I32]] : i32 to index
// HOST:         %[[STRIDE:.*]] = arith.index_cast %[[STRIDE_I64]] : i64 to index
// HOST:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: [64, %[[STRIDE]], 1]
// HOST-SAME:      memref<1x?x64xf32>
// HOST:         ktdp.load
// HOST:         ktdp.store
//
// DEVICE-LABEL: tt.func @dynamic_strides_host(
// DEVICE-SAME:    %[[PTR:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>, %[[M_I32:.*]]: i32, %[[STRIDE_I64:.*]]: i64)
// DEVICE:         %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[PTR]] : !tt.ptr<f32> to index
// DEVICE:         %[[M:.*]] = arith.index_cast %[[M_I32]] : i32 to index
// DEVICE:         %[[C1:.*]] = arith.constant 1 : index
// DEVICE:         %[[C64:.*]] = arith.constant 64 : index
// DEVICE:         %[[S1:.*]] = arith.muli %[[C1]], %[[C64]] : index
// DEVICE:         %[[S0:.*]] = arith.muli %[[S1]], %[[M]] : index
// DEVICE:         ktdp.construct_memory_view %[[BASE]], sizes: [1, %[[M]], 64], strides: [%[[S0]], %[[S1]], %[[C1]]]
// DEVICE-SAME:      memref<1x?x64xf32>
// DEVICE:         ktdp.load
// DEVICE:         ktdp.store
  tt.func @dynamic_strides_host(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i64) {
    %c0 = arith.constant 0 : index
    // [M, 64] with dynamic M and dynamic stride_m, stick-on-N(64)
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg3 : i64 to index
    %3 = ktdp.construct_memory_view %0, sizes: [%1, 64], strides: [%2, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<?x64xf32>
    %4 = builtin.unrealized_conversion_cast %3 : memref<?x64xf32> to !tt.tensordesc<128x64xf32>
    tt.spyre_tensor_layout %4 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x64xf32>
    %5 = ktdp.construct_access_tile %3[%c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<?x64xf32> -> !ktdp.access_tile<128x64xindex>
    %6 = ktdp.load %5 : <128x64xindex> -> tensor<128x64xf32>
    // Store to keep alive
    %7 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %arg3 : i64 to index
    %10 = ktdp.construct_memory_view %7, sizes: [%8, 64], strides: [%9, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<?x64xf32>
    %11 = builtin.unrealized_conversion_cast %10 : memref<?x64xf32> to !tt.tensordesc<128x64xf32>
    tt.spyre_tensor_layout %11 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x64xf32>
    %12 = ktdp.construct_access_tile %10[%c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<?x64xf32> -> !ktdp.access_tile<128x64xindex>
    ktdp.store %6, %12 : tensor<128x64xf32>, <128x64xindex>
    tt.return
  }
}
