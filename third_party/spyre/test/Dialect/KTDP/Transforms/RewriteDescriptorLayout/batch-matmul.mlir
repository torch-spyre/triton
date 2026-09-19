// RUN: spyre-triton-opt %s --rewrite-descriptor-layout | FileCheck %s

// Test: Batched matmul with stick layouts — A[4x64x128] @ B[4x128x64] = C[4x64x64].
// Layout: stick-on-K for A and B (K=128, stick=64 => 2 K-sticks => loop of 2).
// Expected: scf.for over 2 sticks, extract_slice of each operand, linalg.batch_matmul.

// CHECK-LABEL: tt.func public @bmm_matmul_kernel
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<4x64x64xf32>
// CHECK:         scf.for
// CHECK:           %[[A_PHYS:.*]] = ktdp.load {{.*}} -> tensor<2x4x64x64xf16>
// CHECK:           %[[B_PHYS:.*]] = ktdp.load {{.*}} -> tensor<1x4x128x64xf16>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[STICK_LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[CST]]) -> (tensor<4x64x64xf32>)
// CHECK:             %[[A_SLICE:.*]] = tensor.extract_slice %[[A_PHYS]][%[[IV]], 0, 0, 0] [1, 4, 64, 64]
// CHECK:             %[[B_SLICE:.*]] = tensor.extract_slice %[[B_PHYS]][0, 0, {{.*}}, 0] [1, 4, 64, 64]
// CHECK:             %[[BMM:.*]] = linalg.batch_matmul ins(%[[A_SLICE]], %[[B_SLICE]] : tensor<4x64x64xf16>, tensor<4x64x64xf16>) outs(%[[ACC]] : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
// CHECK:             scf.yield %[[BMM]]
// CHECK:           ktdp.store {{.*}} : tensor<1x4x64x64xf16>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  tt.func public @bmm_matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64x64xf32>
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.divsi %1, %1 : i32
    %3 = arith.muli %0, %2 : i32
    %4 = arith.addi %3, %2 : i32
    %5 = arith.minsi %4, %c1_i32 : i32
    %6 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f16> to index
    %7 = ktdp.construct_memory_view %6, sizes: [4, 64, 128], strides: [8192, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x64x128xf16>
    %8 = builtin.unrealized_conversion_cast %7 : memref<4x64x128xf16> to !tt.tensordesc<4x64x128xf16>
    %9 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f16> to index
    %10 = ktdp.construct_memory_view %9, sizes: [4, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<4x128x64xf16>
    %11 = builtin.unrealized_conversion_cast %10 : memref<4x128x64xf16> to !tt.tensordesc<4x128x64xf16>
    %12 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to index
    %13 = ktdp.construct_memory_view %12, sizes: [4, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<4x64x64xf16>
    %14 = builtin.unrealized_conversion_cast %13 : memref<4x64x64xf16> to !tt.tensordesc<4x64x64xf16>
    tt.spyre_tensor_layout %8 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x64x128xf16>
    tt.spyre_tensor_layout %11 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x128x64xf16>
    tt.spyre_tensor_layout %14 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x64x64xf16>
    scf.for %arg3 = %3 to %5 step %c1_i32  : i32 {
      %15 = arith.muli %arg3, %c4_i32 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = ktdp.construct_access_tile %7[%16, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x64x128xf16> -> !ktdp.access_tile<4x64x128xindex>
      %18 = ktdp.load %17 : <4x64x128xindex> -> tensor<4x64x128xf16>
      %19 = arith.index_cast %15 : i32 to index
      %20 = ktdp.construct_access_tile %10[%19, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<4x128x64xf16> -> !ktdp.access_tile<4x128x64xindex>
      %21 = ktdp.load %20 : <4x128x64xindex> -> tensor<4x128x64xf16>
      %22 = linalg.batch_matmul ins(%18, %21 : tensor<4x64x128xf16>, tensor<4x128x64xf16>) outs(%cst : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
      %23 = arith.truncf %22 : tensor<4x64x64xf32> to tensor<4x64x64xf16>
      %24 = arith.index_cast %15 : i32 to index
      %25 = ktdp.construct_access_tile %13[%24, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x64x64xf16> -> !ktdp.access_tile<4x64x64xindex>
      ktdp.store %23, %25 : tensor<4x64x64xf16>, <4x64x64xindex>
    }
    tt.return
  }
}
