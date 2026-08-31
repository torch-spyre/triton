// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout | FileCheck %s

// Two annotated inputs feeding an annotated store through arith.addf: the
// multi-tensor-operand case RewriteElementwisePattern's local shape rule
// exists to cover (item 3 of Step 2b). Both operands physicalize to the
// same stick shape, addf's result is retyped to match, and the store's
// destination marker lets it physicalize too -- so the whole chain stays
// physical with no bridging loop, unlike the unannotated-store sibling
// fixture in this directory.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.
module {
// CHECK-LABEL: tt.func @elementwise_annotated_addf
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       arith.addf {{.*}} : tensor<2x64x64xf32>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK-NOT:   scf.for
// CHECK:       ktdp.store {{.*}} : tensor<2x64x64xf32>, <2x64x64xindex>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @elementwise_annotated_addf(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %b_desc = tt.make_tensor_descriptor %b_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %b_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %b = tt.descriptor_load %b_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %c_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %s = arith.addf %a, %b : tensor<64x128xf32>
  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %s : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}
