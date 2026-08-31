// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout | FileCheck %s

// Annotated input feeding an unannotated store through arith.negf.
// Before Step 2b (retypeChain): the negf's physical retype propagated
// forward into the store's data tile while the store's access tile stayed
// logical, producing invalid IR ('ktdp.store' op data tile shape must match
// access tile shape). After Step 2b: Phase 2 decides the store itself, sees
// there is no destination marker to physicalize into, and emits a bridging
// loop (emitBridgeToLogical) that reassembles the logical shape from the
// physical data tile's stick slices before the store.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.
module {
// CHECK-LABEL: tt.func @elementwise_unannotated_store
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       arith.negf {{.*}} : tensor<2x64x64xf32>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tensor.empty() : tensor<64x128xf32>
// CHECK:       scf.for
// CHECK:         tensor.extract_slice
// CHECK:         tensor.insert_slice
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.store {{.*}} : tensor<64x128xf32>, <64x128xindex>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
tt.func @elementwise_unannotated_store(%a_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  %n = arith.negf %a : tensor<64x128xf32>
  tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %n : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}
