// RUN: spyre-triton-opt %s --lower-descriptor-memory | FileCheck %s

// Regression test: an integer descriptor whose *type* carries a signed
// element (`si32`) must still get a signless (`i32`) backing memory view,
// matching the signless tensor values loaded/stored through it. Triton
// stamps signedness onto an integer descriptor's type (`*i32` -> block
// type `si32`, see `create_tensor_descriptor_type` in `python/src/ir.cc`)
// but keeps every tensor VALUE signless (`getSignlessBlockType()` in
// `create_descriptor_load`; enforced on stores too by
// `verifyDescriptorLoadStoreOp` in `Ops.cpp`). Both directions are
// checked since load and store share the `buildBaseMemoryView` helper.

// CHECK-LABEL:   tt.func @store_signed_i32(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<i32>, %[[VAL_1:.*]]: tensor<64xi32>) {
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<i32> to index
// CHECK:           %[[VAL_6:.*]] = ktdp.construct_memory_view %[[VAL_5]], sizes: [64], strides: [1] {{.*}} : memref<64xi32>
// CHECK-NOT:       memref<64xsi32>
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_6]]{{\[}}%{{.*}}] {{.*}} : memref<64xi32> -> !ktdp.access_tile<64xindex>
// CHECK:           ktdp.store %[[VAL_1]], %[[VAL_9]] : tensor<64xi32>, <64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @store_signed_i32(%out_ptr: !tt.ptr<i32>, %val: tensor<64xi32>) {
  %n = arith.constant 64 : i32
  %stride = arith.constant 1 : i64
  %off = arith.constant 0 : i32
  %desc = tt.make_tensor_descriptor %out_ptr, [%n], [%stride]
      : <i32>, <64xsi32>
  tt.descriptor_store %desc[%off], %val
      : !tt.tensordesc<64xsi32>, tensor<64xi32>
  tt.return
}

// CHECK-LABEL:   tt.func @load_signed_i32(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<i32>) -> tensor<64xi32> {
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<i32> to index
// CHECK:           %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [64], strides: [1] {{.*}} : memref<64xi32>
// CHECK-NOT:       memref<64xsi32>
// CHECK:           %[[VAL_8:.*]] = ktdp.construct_access_tile %[[VAL_5]]{{\[}}%{{.*}}] {{.*}} : memref<64xi32> -> !ktdp.access_tile<64xindex>
// CHECK:           %[[VAL_9:.*]] = ktdp.load %[[VAL_8]] : <64xindex> -> tensor<64xi32>
// CHECK:           tt.return %[[VAL_9]] : tensor<64xi32>
// CHECK:         }
tt.func @load_signed_i32(%in_ptr: !tt.ptr<i32>) -> tensor<64xi32> {
  %n = arith.constant 64 : i32
  %stride = arith.constant 1 : i64
  %off = arith.constant 0 : i32
  %desc = tt.make_tensor_descriptor %in_ptr, [%n], [%stride]
      : <i32>, <64xsi32>
  %data = tt.descriptor_load %desc[%off]
      : !tt.tensordesc<64xsi32> -> tensor<64xi32>
  tt.return %data : tensor<64xi32>
}
