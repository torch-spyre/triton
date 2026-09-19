// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout=data-layout=host -split-input-file | FileCheck %s --check-prefix=HOST
// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file | FileCheck %s --check-prefix=DEVICE

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

// Test: dynamic M size and dynamic stride with stick-on-N(64), data-layout=device (default).
// Physical shape: [N/64, M, N%64] = [1, %M, 64] (N=64 so N/64=1).
// Device-mode computes row-major strides dynamically:
//   dim2 = 1, dim1 = 1*64 = 64, dim0 = 64 * %M
// Host-mode derives strides from logical stride:
//   dim 0 (src=1, floordiv 64): logStride[1]*64 = 1*64 = 64
//   dim 1 (src=0, identity):    logStride[0] = %stride_m (passed through)
//   dim 2 (src=1, mod 64):      logStride[1] = 1
//   => strides: [64, %stride_m, 1]

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

// Test: dynamic M size and dynamic stride with stick-on-N(64), data-layout=host.
// Same descriptor as above but named differently to verify host-mode strides independently.
// Host-mode strides: [64, %stride_m, 1]
// Device-mode strides: [64*%M, 64, 1] (row-major)

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
