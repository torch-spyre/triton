// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout | FileCheck %s

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
module {
  tt.func public @bmm_matmul_kernel(%a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c_desc = arith.constant 4096 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %a_desc = arith.constant 128 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32

    %pid = tt.get_program_id x : i32
    %num_cores = tt.get_num_programs x : i32
    %bm_per_core = arith.divsi %num_cores, %num_cores : i32
    %bm_start = arith.muli %pid, %bm_per_core : i32
    %bm_end = arith.addi %bm_start, %bm_per_core : i32
    %bm_end_0 = arith.minsi %bm_end, %c1_i32 : i32

    %a_desc_1 = tt.make_tensor_descriptor %a_ptr, [%c4_i32, %c64_i32, %c128_i32], [%c8192_i64, %a_desc, %c1_i64] : <f16>, <4x64x128xf16>
    %b_desc = tt.make_tensor_descriptor %b_ptr, [%c4_i32, %c128_i32, %c64_i32], [%c8192_i64, %c64_i64, %c1_i64] : <f16>, <4x128x64xf16>
    %c_desc_2 = tt.make_tensor_descriptor %c_ptr, [%c4_i32, %c64_i32, %c64_i32], [%c_desc, %c64_i64, %c1_i64] : <f16>, <4x64x64xf16>
    tt.spyre_tensor_layout %a_desc_1 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x64x128xf16>
    tt.spyre_tensor_layout %b_desc {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x128x64xf16>
    tt.spyre_tensor_layout %c_desc_2 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x64x64xf16>

    scf.for %arg3 = %bm_start to %bm_end_0 step %c1_i32  : i32 {
      %a_tile = arith.muli %arg3, %c4_i32 : i32
      %a_tile_3 = tt.descriptor_load %a_desc_1[%a_tile, %c0_i32, %c0_i32] : !tt.tensordesc<4x64x128xf16> -> tensor<4x64x128xf16>
      %b_tile = tt.descriptor_load %b_desc[%a_tile, %c0_i32, %c0_i32] : !tt.tensordesc<4x128x64xf16> -> tensor<4x128x64xf16>
      %acc = tt.dot %a_tile_3, %b_tile, %cst : tensor<4x64x128xf16> * tensor<4x128x64xf16> -> tensor<4x64x64xf32>
      %0 = arith.truncf %acc : tensor<4x64x64xf32> to tensor<4x64x64xf16>
      tt.descriptor_store %c_desc_2[%a_tile, %c0_i32, %c0_i32], %0 : !tt.tensordesc<4x64x64xf16>, tensor<4x64x64xf16>
    }

    tt.return
  }
}
