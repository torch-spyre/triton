// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic | FileCheck %s

// An annotated gather: the indirect access tile is physicalized.
//
// The tile carries one affine SUBSCRIPT MAP per base dim rather than one SSA
// index, so physicalizing it is a substitution on those maps rather than
// arith.divsi/remsi. Refining the variable space splits logical dim 1 into a
// (stick, elem) pair, and the subscript that named the whole dim recovers it as
// `stick * 64 + elem` -- the same composite the generic rewrite emits when one
// operand holds a dim some other operand splits.
//
// The data descriptor is [512, 128] stick-on-N(64):
//   phys_src=[1, 0, 1] phys_op=[1, 0, 2] phys_arg=[64, 0, 64]
//   => physical shape [N/64, M, N%64] = [2, 512, 64]
// and the gathered tile 32x128 becomes 2x32x64.

#varorder = affine_map<(d0, d1) -> (d0, d1)>
#sidx = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#sdata = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#stile = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL:   tt.func @gather_with_layout(
// CHECK-SAME:      %[[DATA:.*]]: !tt.ptr<f32>, %[[IDX:.*]]: !tt.ptr<i32>, %[[OUT:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[IDXV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [32], strides: [1]
// CHECK-SAME:        : memref<32xi32>
//
// The data view is physical, and so is the tile over it. The gathered dim 1 is
// the indirect subscript and stays whole; the contiguous dim 1 of the base is
// what splits, and its subscript is recovered as the composite.
// CHECK:           %[[DATAV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 512, 64], strides: [32768, 64, 1]
// CHECK-SAME:        : memref<2x512x64xf32>
// CHECK:           %[[TILE:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[V0:.*]], %[[V1:.*]], %[[V2:.*]]) %[[DATAV]]{{\[}}((%[[C0]] + %[[V0]] * 64 + %[[V2]]) floordiv 64), ind(%[[IDXV]]{{\[}}%[[C0]] + %[[V1]]]), ((%[[C0]] + %[[V0]] * 64 + %[[V2]]) mod 64)]
// CHECK-SAME:        -> !ktdp.access_tile<2x32x64xindex>
// CHECK:           %[[LOAD:.*]] = ktdp.load %[[TILE]] : <2x32x64xindex> -> tensor<2x32x64xf32>
//
// The output descriptor carries the same layout, so its direct tile is split by
// arith.divsi/remsi on the subscript rather than by a substitution.
// CHECK:           %[[OUTV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 32, 64], strides: [2048, 64, 1]
// CHECK-SAME:        : memref<2x32x64xf32>
// CHECK:           %[[W:.*]] = arith.constant 64 : index
// CHECK:           %[[STICK:.*]] = arith.divsi %[[C0]], %[[W]] : index
// CHECK:           %[[W2:.*]] = arith.constant 64 : index
// CHECK:           %[[LANE:.*]] = arith.remsi %[[C0]], %[[W2]] : index
// CHECK:           %[[OUTT:.*]] = ktdp.construct_access_tile %[[OUTV]]{{\[}}%[[STICK]], %[[C0]], %[[LANE]]]
// CHECK-SAME:        -> !ktdp.access_tile<2x32x64xindex>
// CHECK:           ktdp.store %[[LOAD]], %[[OUTT]] : tensor<2x32x64xf32>, <2x32x64xindex>
//
// No marker survives.
// CHECK-NOT:       tt.spyre_tensor_layout
tt.func @gather_with_layout(%data: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index

  // Index tensor: 32 row indices, unannotated, so it stays logical.
  %ii = builtin.unrealized_conversion_cast %idx : !tt.ptr<i32> to index
  %iv = ktdp.construct_memory_view %ii, sizes: [32], strides: [1] {coordinate_set = #sidx, memory_space = #ktdp.memory_space<global>} : memref<32xi32>

  // Data: [512, 128] stick-on-N(64).
  %di = builtin.unrealized_conversion_cast %data : !tt.ptr<f32> to index
  %dv = ktdp.construct_memory_view %di, sizes: [512, 128], strides: [128, 1] {coordinate_set = #sdata, memory_space = #ktdp.memory_space<global>} : memref<512x128xf32>
  %dd = builtin.unrealized_conversion_cast %dv : memref<512x128xf32> to !tt.tensordesc<512x128xf32>
  tt.spyre_tensor_layout %dd {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <512x128xf32>
  %dt = ktdp.construct_indirect_access_tile intermediate_variables(%v0, %v1) %dv[ind(%iv[%c0 + %v0]), (%c0 + %v1)] {variables_space_order = #varorder, variables_space_set = #stile} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
  %dl = ktdp.load %dt : <32x128xindex> -> tensor<32x128xf32>

  // Output: [32, 128] under the same layout, reached by a direct tile.
  %oi = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [32, 128], strides: [128, 1] {coordinate_set = #stile, memory_space = #ktdp.memory_space<global>} : memref<32x128xf32>
  %od = builtin.unrealized_conversion_cast %ov : memref<32x128xf32> to !tt.tensordesc<32x128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <32x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #varorder, access_tile_set = #stile} : memref<32x128xf32> -> !ktdp.access_tile<32x128xindex>
  ktdp.store %dl, %ot : tensor<32x128xf32>, <32x128xindex>
  tt.return
}
}
