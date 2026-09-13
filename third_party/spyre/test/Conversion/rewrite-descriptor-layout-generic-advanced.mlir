// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout-generic=data-layout=host -split-input-file | FileCheck %s --check-prefix=HOST
// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s --check-prefix=DEVICE

// -----

// data-layout=host derives physical strides from the descriptor's logical
// strides through the coordinate map; device mode ignores them and lays the
// physical shape out row-major.
// [M=128, N=128] stick-on-N(64) -> physical [N/64, M, N%64] = [2, 128, 64]:
//   host:   dim0 = logStride[1]*64 = 64, dim1 = logStride[0] = 128, dim2 = 1
//   device: row-major over [2, 128, 64] = [8192, 64, 1]

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
tt.func @pointwise_host_layout(%ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  // [M=128, N=128] stick-on-N with stick_size=64
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x128xf32>
  %d = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
  // Store to physical-annotated output to keep alive.
  %out_desc = tt.make_tensor_descriptor %out_ptr, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
  tt.spyre_tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x128xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<128x128xf32>, tensor<128x128xf32>
  tt.return
}
}

// -----

// A dynamic size and a dynamic logical stride, stick-on-N(64): physical shape
// [1, %M, 64].  Device mode has to *emit* the row-major stride arithmetic
// (muli chain over %M), while host mode threads the dynamic logical stride
// straight into dim 1.

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
tt.func @dynamic_strides_device(%ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %M: i32, %stride_m: i64) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  // [M, 64] with dynamic M and dynamic stride_m, stick-on-N(64)
  %desc = tt.make_tensor_descriptor %ptr, [%M, %c64_i32], [%stride_m, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x64xf32>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf32>
  %d = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf32> -> tensor<128x64xf32>
  // Store to keep alive
  %out_desc = tt.make_tensor_descriptor %out_ptr, [%M, %c64_i32], [%stride_m, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x64xf32>
  tt.spyre_tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<128x64xf32>, tensor<128x64xf32>
  tt.return
}
}

// -----

// Same descriptor as the previous module, checked with the host prefix listed
// first, so neither mode's checks can pass by matching the other's output.

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
tt.func @dynamic_strides_host(%ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %M: i32, %stride_m: i64) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  // [M, 64] with dynamic M and dynamic stride_m, stick-on-N(64)
  %desc = tt.make_tensor_descriptor %ptr, [%M, %c64_i32], [%stride_m, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x64xf32>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf32>
  %d = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf32> -> tensor<128x64xf32>
  // Store to keep alive
  %out_desc = tt.make_tensor_descriptor %out_ptr, [%M, %c64_i32], [%stride_m, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x64xf32>
  tt.spyre_tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<128x64xf32>, tensor<128x64xf32>
  tt.return
}
}
