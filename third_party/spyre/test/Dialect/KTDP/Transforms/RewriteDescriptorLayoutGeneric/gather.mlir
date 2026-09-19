// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout-generic | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 63 >= 0)>

// An annotated gather: the indirect access tile is physicalized.
//
// The tile carries one affine SUBSCRIPT MAP per base dim rather than one SSA
// index, so physicalizing it is a substitution on those maps rather than
// arith.divsi/remsi. Refining the variable space splits logical dim 1 into a
// (stick, elem) pair, and the subscript that named the whole dim recovers it as
// `stick * 64 + elem` -- the same composite the generic rewrite emits when one
// operand holds a dim some other operand splits.
//
// The emitted tile is bit-identical to the one RewriteDescriptorLayout produces
// for this input; see rewrite-descriptor-layout-gather.mlir.

module {
// CHECK-LABEL:   tt.func @gather_with_layout(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<i32>, %[[VAL_2:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<i32> to index
// CHECK:           %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [32], strides: [1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<32xi32>
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_7:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_8:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: [2, 512, 64], strides: [32768, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<2x512x64xf32>
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]]) %[[VAL_8]][((%[[VAL_9]] + %[[VAL_11]] * 64 + %[[VAL_13]]) floordiv 64), ind(%[[VAL_5]]{{\[}}%[[VAL_6]] + %[[VAL_12]]]), ((%[[VAL_9]] + %[[VAL_11]] * 64 + %[[VAL_13]]) mod 64)] {variables_space_order = #[[$ATTR_0]], variables_space_set = #[[$ATTR_3]]} : memref<2x512x64xf32>, memref<32xi32> -> !ktdp.access_tile<2x32x64xindex>
// CHECK:           %[[VAL_14:.*]] = ktdp.load %[[VAL_10]] : <2x32x64xindex> -> tensor<2x32x64xf32>
// CHECK:           %[[VAL_15:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_16:.*]] = ktdp.construct_memory_view %[[VAL_15]], sizes: [2, 32, 64], strides: [2048, 64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<2x32x64xf32>
// CHECK:           %[[VAL_17:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_20:.*]] = arith.divsi %[[VAL_18]], %[[VAL_19]] : index
// CHECK:           %[[VAL_21:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_22:.*]] = arith.remsi %[[VAL_18]], %[[VAL_21]] : index
// CHECK:           %[[VAL_23:.*]] = ktdp.construct_access_tile %[[VAL_16]]{{\[}}%[[VAL_20]], %[[VAL_17]], %[[VAL_22]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_3]]} : memref<2x32x64xf32> -> !ktdp.access_tile<2x32x64xindex>
// CHECK:           ktdp.store %[[VAL_14]], %[[VAL_23]] : tensor<2x32x64xf32>, <2x32x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @gather_with_layout(%data_ptr: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c512_i32 = arith.constant 512 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c32_i64 = arith.constant 32 : i64

  // Index descriptor: 32-element 1-D tensor holding row indices.
  %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%c32_i32], [%c1_i64]
      : !tt.ptr<i32>, !tt.tensordesc<32xi32>
  %x_offsets = tt.descriptor_load %idx_desc[%c0_i32] : !tt.tensordesc<32xi32> -> tensor<32xi32>

  // Data descriptor: [512, 128] with stick-on-N layout (stick=64).
  //   phys_src=[1, 0, 1] phys_op=[1, 0, 2] phys_arg=[64, 0, 64]
  //   => physical shape [N/64, M, N%64] = [2, 512, 64]
  %data_desc = tt.make_tensor_descriptor %data_ptr, [%c512_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<1x128xf32>
  tt.spyre_tensor_layout %data_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<1x128xf32>

  // Gather 32 non-contiguous rows.
  %gathered = tt.descriptor_gather %data_desc[%x_offsets, %c0_i32]
      : (!tt.tensordesc<1x128xf32>, tensor<32xi32>, i32) -> tensor<32x128xf32>

  // Store the gathered result to a physical-annotated output.
  %out_desc = tt.make_tensor_descriptor %out_ptr, [%c32_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<32x128xf32>
  tt.spyre_tensor_layout %out_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<32x128xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %gathered : !tt.tensordesc<32x128xf32>, tensor<32x128xf32>

  tt.return
}
}
