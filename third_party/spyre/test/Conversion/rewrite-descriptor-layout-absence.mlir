// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file | FileCheck %s

// What RewriteDescriptorLayout must leave absent. The first four cases are
// case 1 of the shape check in docs/spyre-tensor-layouts.md: layouts that
// physicalize with no synthesized loop. The last two only assert marker
// erasure, which every case checks.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.

// A stick-on-K(64) with K=64, B stick-on-N(64) with N=64: one stick each
// side, nothing to accumulate across.
module {
// CHECK-LABEL: tt.func @matmul_single_stick
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.matmul
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @matmul_single_stick(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64

  // A[64, 64] stick-on-K(64): phys [1, 64, 64] = [K/64, M, K%64]
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  // B[64, 64] stick-on-N(64): phys [1, 64, 64] = [N/64, K, N%64]
  %b_desc = tt.make_tensor_descriptor %b_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  tt.spyre_tensor_layout %b_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  // C[64, 64] — unannotated accumulator
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  %c = tt.descriptor_load %c_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  %d = tt.dot %a, %b, %c : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>

  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<64x64xf32>, tensor<64x64xf32>
  tt.return
}
}

// -----

// Both operands stick-split on a parallel axis (M for A, N for B), each with
// trip count 1.
module {
// CHECK-LABEL: tt.func @matmul_both_parallel_trip1
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.matmul
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @matmul_both_parallel_trip1(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64

  // A[M=64, K=128] stick-on-M(64): phys [M/64, K, M%64] = [1, 128, 64]
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>

  // B[K=128, N=64] stick-on-N(64): phys [N/64, K, N%64] = [1, 128, 64]
  %b_desc = tt.make_tensor_descriptor %b_ptr, [%c128_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x64xf32>
  tt.spyre_tensor_layout %b_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf32>
  %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf32> -> tensor<128x64xf32>

  // C[64, 64] — unannotated accumulator
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  %c = tt.descriptor_load %c_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  %d = tt.dot %a, %b, %c : tensor<64x128xf32> * tensor<128x64xf32> -> tensor<64x64xf32>

  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<64x64xf32>, tensor<64x64xf32>
  tt.return
}
}

// -----

// Reduce with an identity layout (phys_op = [0, 0]): no dim is stick-split.
module {
// CHECK-LABEL: tt.func @reduce_identity_layout
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @reduce_identity_layout(%a_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64

  // A[64, 64] identity layout: phys_src=[0,1] phys_op=[0,0] phys_arg=[0,0]
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x64xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 0, 1>, phys_op = array<i64: 0, 0>, phys_arg = array<i64: 0, 0>} : !tt.tensordesc<64x64xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>

  // C[64] — unannotated output
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32], [%c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64xf32>
  %c = tt.descriptor_load %c_desc[%c0_i32] : !tt.tensordesc<64xf32> -> tensor<64xf32>

  %r = "tt.reduce"(%a) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<64x64xf32>) -> tensor<64xf32>

  tt.descriptor_store %c_desc[%c0_i32], %r : !tt.tensordesc<64xf32>, tensor<64xf32>
  tt.return
}
}

// -----

// Reduce whose stick-split floor dim has extent 1: A[2,64,64] stick-on-dim2
// with dim2 == 64, so the floor dim exists but its trip count is 1.
module {
// CHECK-LABEL: tt.func @reduce_floor_dim_extent_one
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @reduce_floor_dim_extent_one(%a_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c64_i32 = arith.constant 64 : i32
  %c64_i64 = arith.constant 64 : i64
  %c4096_i64 = arith.constant 4096 : i64
  %c1_i64 = arith.constant 1 : i64

  // A[2, 64, 64] stick-on-dim2(64): phys [dim2/64, dim0, dim1, dim2%64] = [1, 2, 64, 64]
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c2_i32, %c64_i32, %c64_i32], [%c4096_i64, %c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<2x64x64xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 2, 0, 1, 2>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 64, 0, 0, 64>} : !tt.tensordesc<2x64x64xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<2x64x64xf32> -> tensor<2x64x64xf32>

  // C[2, 64] — unannotated output
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c2_i32, %c64_i32], [%c64_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<2x64xf32>
  %c = tt.descriptor_load %c_desc[%c0_i32, %c0_i32] : !tt.tensordesc<2x64xf32> -> tensor<2x64xf32>

  %r = "tt.reduce"(%a) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 2 : i32} : (tensor<2x64x64xf32>) -> tensor<2x64xf32>

  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %r : !tt.tensordesc<2x64xf32>, tensor<2x64xf32>
  tt.return
}
}

// -----

// RewriteDescriptorLayout erases every tt.spyre_tensor_layout marker in its
// Phase 3 -- see docs/spyre-tensor-layouts.md. Each case below carries a
// marker on input, so the CHECK-NOT proves erasure rather than passing on an
// input that never had one.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.

// Elementwise op with a marker on both input and output descriptors: the
// whole chain is absorbed by the forward retype, so nothing is synthesized.
module {
// CHECK-LABEL: tt.func @pointwise
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @pointwise(%ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  // [M=128, N=128] stick-on-N with stick_size=64 -> physical [N/64, M, 64] = [2, 128, 64]
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x128xf32>
  %d = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<128x128xf32> -> tensor<128x128xf32>
  // Output with same layout
  %out_desc = tt.make_tensor_descriptor %out_ptr, [%c128_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<128x128xf32>
  tt.spyre_tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x128xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<128x128xf32>, tensor<128x128xf32>
  tt.return
}
}

// -----

// Reduce with markers on both the input and the output descriptor, so erasure
// is proven on the sink side too. This case does emit a loop -- its input is
// stick-split on the reduced axis -- so it asserts marker absence only.
module {
// CHECK-LABEL: tt.func @reduce_annotated_output
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @reduce_annotated_output(%a_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64

  // A[64, 128] stick-on-N(64): phys [N/64, M, N%64] = [2, 64, 64]
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>

  // C[64] stick-on-M(64): phys [M/64, M%64] = [1, 64]
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32], [%c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64xf32>
  tt.spyre_tensor_layout %c_desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf32>
  %c = tt.descriptor_load %c_desc[%c0_i32] : !tt.tensordesc<64xf32> -> tensor<64xf32>

  %r = "tt.reduce"(%a) ({
  ^bb0(%arg0: f32, %arg1: f32):
    %add = arith.addf %arg0, %arg1 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<64x128xf32>) -> tensor<64xf32>

  tt.descriptor_store %c_desc[%c0_i32], %r : !tt.tensordesc<64xf32>, tensor<64xf32>
  tt.return
}
}
