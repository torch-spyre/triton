// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// A direct access tile's subscripts are split arithmetically.
//
// ktdp.construct_access_tile names one `index` subscript per logical dim. When a
// marker stick-splits a logical dim, that dim becomes two physical dims, and the
// one subscript that named it has to become two: the pass emits an arith.divsi
// for the stick half and an arith.remsi for the lane half, both over the stick
// width read from phys_arg. Every other physical dim keeps the subscript it
// arrived with.
//
// That plain shape -- a distinct SSA value per logical dim, split in place -- is
// what every positive test in this directory shows. The cases here are the two
// ways a subscript can arrive without being a plain value of its own: shared
// between two logical dims, and defined by an enclosing loop. The split is
// emitted the same way in both, which is the claim.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py, which numbers captures globally across a file and
// would renumber every section on any edit.

// Case 1 -- one SSA value standing for two logical dims.
//
// Two logical index operands of construct_access_tile are the same SSA value,
// which the op's custom parser deduplicates via base_map (one operand plus
// base_map (d0) -> (d0, d0)). Physicalization must expand through base_map to
// recover a value per logical dim, or the split subscripts come out wrong.

// CHECK: #[[$SHARED_ORDER:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$SHARED_SET:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>

module {
// CHECK-LABEL:   tt.func @shared_index_value(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_3:.*]] = ktdp.construct_memory_view %[[VAL_2]], sizes: [2, 128, 64], strides: [8192, 64, 1] {coordinate_set = #[[$SHARED_SET]], memory_space = #ktdp.memory_space<global>} : memref<2x128x64xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_5:.*]] = arith.divsi %[[VAL_1]], %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_7:.*]] = arith.remsi %[[VAL_1]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = ktdp.construct_access_tile %[[VAL_3]]{{\[}}%[[VAL_5]], %[[VAL_1]], %[[VAL_7]]] {access_tile_order = #[[$SHARED_ORDER]], access_tile_set = #[[$SHARED_SET]]} : memref<2x128x64xf32> -> !ktdp.access_tile<2x128x64xindex>
// CHECK:           %[[VAL_9:.*]] = ktdp.load %[[VAL_8]] : <2x128x64xindex> -> tensor<2x128x64xf32>
// CHECK:           tt.return
// CHECK:         }
tt.func @shared_index_value(%arg0: !tt.ptr<f32>) {
  // Use the SAME value for both logical indices.
  %idx = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [128, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<128x128xf32> to !tt.tensordesc<128x128xf32>
  // phys_src=[1, 0, 1] phys_op=[1, 0, 2] phys_arg=[64, 0, 64]
  // Physical dim 0: logical dim 1, floordiv 64
  // Physical dim 1: logical dim 0, identity
  // Physical dim 2: logical dim 1, mod 64
  tt.spyre_tensor_layout %2 {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <128x128xf32>
  // Both index operands are the same SSA value (%idx).
  %3 = ktdp.construct_access_tile %1[%idx, %idx] {access_tile_order = #map, access_tile_set = #set} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
  %4 = ktdp.load %3 : <128x128xindex> -> tensor<128x128xf32>
  tt.return
}
}

// -----

// Case 2 -- the subscript is an enclosing loop's induction variable.
//
// The divsi/remsi land inside the loop body, over the IV, and the loop itself is
// left exactly as it is.
//
// This replaces RewriteDescriptorLayout/loop-rescale.mlir, whose whole subject
// was rescaling such a loop from block units to stick units. The old pass needed
// that because it consumed the IV directly as a physical stick index, having
// synthesized its own stick loops. Here the split is built at the subscript, so
// the loop's own iteration space never changes meaning and there is nothing to
// rescale.

// CHECK: #[[$LOOP_ORDER:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$LOOP_SET:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// The loop this pass did NOT add: no second scf.for anywhere, and no arith.muli
// rescaling the bounds of the one that was already there.
// CHECK-LABEL:   tt.func @loop_left_alone(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_6:.*]] = ktdp.construct_memory_view %[[VAL_5]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$LOOP_SET]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_7:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_8:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$LOOP_SET]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:             %[[VAL_10:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_11:.*]] = arith.divsi %[[VAL_9]], %[[VAL_10]] : index
// CHECK:             %[[VAL_12:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_13:.*]] = arith.remsi %[[VAL_9]], %[[VAL_12]] : index
// CHECK:             %[[VAL_14:.*]] = ktdp.construct_access_tile %[[VAL_6]]{{\[}}%[[VAL_11]], %[[VAL_2]], %[[VAL_13]]] {access_tile_order = #[[$LOOP_ORDER]], access_tile_set = #[[$LOOP_SET]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:             %[[VAL_15:.*]] = ktdp.load %[[VAL_14]] : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:             %[[VAL_16:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_17:.*]] = arith.divsi %[[VAL_9]], %[[VAL_16]] : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_19:.*]] = arith.remsi %[[VAL_9]], %[[VAL_18]] : index
// CHECK:             %[[VAL_20:.*]] = ktdp.construct_access_tile %[[VAL_8]]{{\[}}%[[VAL_17]], %[[VAL_2]], %[[VAL_19]]] {access_tile_order = #[[$LOOP_ORDER]], access_tile_set = #[[$LOOP_SET]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:             %[[VAL_21:.*]] = tensor.empty() : tensor<2x64x64xf32>
// CHECK:             %[[VAL_22:.*]] = linalg.generic {indexing_maps = [#[[$LOOP_ORDER]], #[[$LOOP_ORDER]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_15]] : tensor<2x64x64xf32>) outs(%[[VAL_21]] : tensor<2x64x64xf32>) {
// CHECK:             ^bb0(%[[VAL_23:.*]]: f32, %[[VAL_24:.*]]: f32):
// CHECK:               %[[VAL_25:.*]] = arith.negf %[[VAL_23]] : f32
// CHECK:               linalg.yield %[[VAL_25]] : f32
// CHECK:             } -> tensor<2x64x64xf32>
// CHECK:             ktdp.store %[[VAL_22]], %[[VAL_20]] : tensor<2x64x64xf32>, <2x64x64xindex>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
tt.func @loop_left_alone(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %odd = builtin.unrealized_conversion_cast %ov : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %odd {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <64x128xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    %at = ktdp.construct_access_tile %av[%c0, %i] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
    %ot = ktdp.construct_access_tile %ov[%c0, %i] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %e = tensor.empty() : tensor<64x128xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
    ^bb0(%x: f32, %y: f32):
      %n = arith.negf %x : f32
      linalg.yield %n : f32
    } -> tensor<64x128xf32>
    ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  }
  tt.return
}
}
