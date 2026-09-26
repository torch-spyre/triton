// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// An operand that holds whole a dim some other operand splits carries the
// composite.
//
// The rebuilt loop domain splits every logical dim that ANY operand splits, so a
// split dim becomes a (stick, elem) pair of loop dims. An operand whose own
// layout splits that dim names the pair directly, as a plain projected
// permutation. An operand that holds the dim whole cannot: its own physical shape
// has one dim there, and the only expression naming that element from the pair is
// `stick * W + elem`. So that operand's map carries the composite, and this is
// the one place the rebuild puts arithmetic into a map at all -- see
// rebuild-passthrough.mlir for the cases where it has nothing to add.
//
// Cases 1 to 3 are the three ways that can land, and nothing in any of them is
// transpose-specific or store-specific: both operands splitting the same loop dim
// (so no composite is needed and the maps come out identical), the operands
// splitting different loop dims (so each carries the other's composite), and the
// same at a ktdp.store, which is why this pass needs no widening stage WHEREVER A
// GENERIC MEDIATES THE STORE.
//
// Cases 4 and 5 are the two ends of that "wherever". Case 4 is a pure
// load-to-store copy with NO generic at all and both ends annotated, which needs
// none: phase 1 physicalizes the two views and the two tiles, and the load's
// retyped result already matches the store's retyped tile. With only the
// destination annotated the same copy has no vehicle for the shape change and the
// pass declines -- invalid-layout.mlir case 4 is that one, and case 4 here is its
// positive twin. Case 5 is an operand on no layout at all, which is the extreme of
// holding dims whole: every dim the annotated operand splits is a composite in the
// unannotated one's map.
//
// Case 6 raises the permutation from a rank-2 swap to a rank-3 3-CYCLE, which is
// not its own inverse: a map restated wrongly there is a shape error rather than a
// silent value one.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py, which numbers captures globally across a file and
// would renumber every section on any edit.

// Case 1 -- a transpose folded into a consumer's map, both operands splitting the
// SAME loop dim.
//
// Worth pinning because the result looks wrong at a glance: the two emitted maps
// are IDENTICAL and still express a transpose. Each physical position already
// names the logical dim it came from, and the physical order comes from each
// operand's own layout, so the permutation is absorbed and no composite is needed.
//
// The one line that makes this case what it is is the input's phys_src = [0,1,0]:
// the input's indexing map is (d0, d1) -> (d1, d0), so splitting the input's
// logical dim 0 splits loop dim d1 -- the same loop dim the output's phys_src =
// [1,0,1] splits.

// CHECK: #[[$ID3_SAME:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$SET_SAME:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#in  = affine_map<(d0, d1) -> (d1, d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
#sin  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sout = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   tt.func @transpose_same_split_dim(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET_SAME]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_8:.*]] = arith.remsi %[[VAL_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_8]]] {access_tile_order = #[[$ID3_SAME]], access_tile_set = #[[$SET_SAME]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_10:.*]] = ktdp.load %[[VAL_9]] : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_11]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET_SAME]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_14:.*]] = arith.divsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_2]], %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_2]], %[[VAL_16]]] {access_tile_order = #[[$ID3_SAME]], access_tile_set = #[[$SET_SAME]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<2x64x64xf32>
// Both maps the same, and neither carries arithmetic.
// CHECK:           %[[VAL_19:.*]] = linalg.generic {indexing_maps = [#[[$ID3_SAME]], #[[$ID3_SAME]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_10]] : tensor<2x64x64xf32>) outs(%[[VAL_18]] : tensor<2x64x64xf32>) {
// CHECK:           ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_20]] : f32
// CHECK:           } -> tensor<2x64x64xf32>
// CHECK:           ktdp.store %[[VAL_19]], %[[VAL_17]] : tensor<2x64x64xf32>, <2x64x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @transpose_same_split_dim(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  // Split the input's logical dim 0, which the consumer's map reads as loop d1.
  %av = ktdp.construct_memory_view %ai, sizes: [128, 64], strides: [64, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sin} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
  %al = ktdp.load %at : <128x64xindex> -> tensor<128x64xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  // Split the output's logical dim 1, which is also loop d1.
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sout} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<128x64xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    linalg.yield %x : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 2 -- the same transpose, the operands splitting DIFFERENT loop dims.
//
// One line differs from case 1: the input's phys_src is [1,0,1] rather than
// [0,1,0], so the input now splits its own logical dim 1, which its map reads as
// loop d0, while the output still splits loop d1. Both loop dims are therefore
// split, each operand holds the other's split dim whole, and each map carries the
// other's composite. That is this file's rule applied twice, with no
// transpose-specific logic anywhere.

// CHECK: #[[$ID3_DIFF:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$IN_COMPOSITE:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d0 * 64 + d3, d2)>
// CHECK: #[[$OUT_COMPOSITE:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 64 + d2, d3)>
// CHECK: #[[$SET_IN_DIFF:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$SET_OUT_DIFF:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#in  = affine_map<(d0, d1) -> (d1, d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
#sin  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sout = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   tt.func @transpose_different_split_dims(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [1, 128, 64], strides: [8192, 64, 1] {coordinate_set = #[[$SET_IN_DIFF]], memory_space = #ktdp.memory_space<global>} : memref<1x128x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_8:.*]] = arith.remsi %[[VAL_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_8]]] {access_tile_order = #[[$ID3_DIFF]], access_tile_set = #[[$SET_IN_DIFF]]} : memref<1x128x64xf32> -> !ktdp.access_tile<1x128x64xindex>
// CHECK:           %[[VAL_10:.*]] = ktdp.load %[[VAL_9]] : <1x128x64xindex> -> tensor<1x128x64xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_11]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET_OUT_DIFF]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_14:.*]] = arith.divsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_2]], %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_2]], %[[VAL_16]]] {access_tile_order = #[[$ID3_DIFF]], access_tile_set = #[[$SET_OUT_DIFF]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<2x64x64xf32>
// Four loop dims now, and each map carries the composite for the dim the other
// operand splits.
// CHECK:           %[[VAL_19:.*]] = linalg.generic {indexing_maps = [#[[$IN_COMPOSITE]], #[[$OUT_COMPOSITE]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_10]] : tensor<1x128x64xf32>) outs(%[[VAL_18]] : tensor<2x64x64xf32>) {
// CHECK:           ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_20]] : f32
// CHECK:           } -> tensor<2x64x64xf32>
// CHECK:           ktdp.store %[[VAL_19]], %[[VAL_17]] : tensor<2x64x64xf32>, <2x64x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @transpose_different_split_dims(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  // The one line that differs from case 1: split the input's logical dim 1, which
  // the consumer's map reads as loop d0.
  %av = ktdp.construct_memory_view %ai, sizes: [128, 64], strides: [64, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sin} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
  %al = ktdp.load %at : <128x64xindex> -> tensor<128x64xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  // Unchanged from case 1: split the output's logical dim 1, which is loop d1.
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sout} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<128x64xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    linalg.yield %x : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 3 -- the split dims differ across a ktdp.store, and the tiles still agree.
//
// This is the case the no-widen-stage claim rests on. The generic's input splits
// logical dim 1 and the store destination splits logical dim 0, so -- as in case
// 2 -- both dims are split in the loop domain and each operand carries the
// composite for the dim the other splits. The store's data tile then has exactly
// the shape its access tile names, by construction: with a generic on the chain
// there is nothing left for a widening stage to do.
//
// The condition is the generic, not the store: it is the generic's outs operand
// that the rebuild gives the store's physical shape. Take it away and the two ends
// have nothing to agree through -- invalid-layout.mlir case 4.

// CHECK: #[[$ID3_STORE:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$STORE_IN_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0 * 64 + d1, d3)>
// CHECK: #[[$STORE_OUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 64 + d3)>
// CHECK: #[[$SET_STORE_IN:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$SET_STORE_OUT:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL:   tt.func @store_agrees_without_widening(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$SET_STORE_IN]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_8:.*]] = arith.remsi %[[VAL_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_8]]] {access_tile_order = #[[$ID3_STORE]], access_tile_set = #[[$SET_STORE_IN]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_10:.*]] = ktdp.load %[[VAL_9]] : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_11]], sizes: [1, 64, 128], strides: [8192, 128, 1] {coordinate_set = #[[$SET_STORE_OUT]], memory_space = #ktdp.memory_space<global>} : memref<1x64x128xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_14:.*]] = arith.divsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_2]], %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_16]], %[[VAL_2]]] {access_tile_order = #[[$ID3_STORE]], access_tile_set = #[[$SET_STORE_OUT]]} : memref<1x64x128xf32> -> !ktdp.access_tile<1x64x128xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<1x64x128xf32>
// CHECK:           %[[VAL_19:.*]] = linalg.generic {indexing_maps = [#[[$STORE_IN_MAP]], #[[$STORE_OUT_MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_10]] : tensor<2x64x64xf32>) outs(%[[VAL_18]] : tensor<1x64x128xf32>) {
// CHECK:           ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             %[[VAL_22:.*]] = arith.negf %[[VAL_20]] : f32
// CHECK:             linalg.yield %[[VAL_22]] : f32
// CHECK:           } -> tensor<1x64x128xf32>
// The store's two shapes: the data is <1x64x128> and so is the access tile.
// CHECK:           ktdp.store %[[VAL_19]], %[[VAL_17]] : tensor<1x64x128xf32>, <1x64x128xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @store_agrees_without_widening(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  // The store destination splits logical dim 0, not dim 1.
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 0, 1>, phys_op = array<i64: 1, 2, 0>, phys_arg = array<i64: 64, 64, 0>}} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 4 -- a pure load-to-store copy with BOTH ends annotated.
//
// No linalg.generic anywhere, so no rebuild runs. Phase 1 alone is the whole
// answer: each view is restated at [2, 128, 64], each tile's one subscript per
// logical dim becomes a divsi/remsi pair, and the load's result is retyped along
// with the tile it reads -- which is exactly the type the store's retyped tile
// wants.

// CHECK: #[[$CPY_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$CPY_SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#idc = affine_map<(d0, d1) -> (d0, d1)>
#sc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL:   tt.func @copy_both_annotated(
// CHECK-NOT:       linalg.generic
// CHECK:           %[[AV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 128, 64], strides: [8192, 64, 1] {coordinate_set = #[[$CPY_SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x128x64xf32>
// CHECK:           %[[AT:.*]] = ktdp.construct_access_tile %[[AV]]{{.*}} {access_tile_order = #[[$CPY_ID3]], access_tile_set = #[[$CPY_SET3]]} : memref<2x128x64xf32> -> !ktdp.access_tile<2x128x64xindex>
// CHECK:           %[[AL:.*]] = ktdp.load %[[AT]] : <2x128x64xindex> -> tensor<2x128x64xf32>
// CHECK:           %[[OV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 128, 64], strides: [8192, 64, 1] {coordinate_set = #[[$CPY_SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x128x64xf32>
// CHECK:           %[[OT:.*]] = ktdp.construct_access_tile %[[OV]]{{.*}} : memref<2x128x64xf32> -> !ktdp.access_tile<2x128x64xindex>
// The loaded value goes straight to the store, both at physical rank 3.
// CHECK:           ktdp.store %[[AL]], %[[OT]] : tensor<2x128x64xf32>, <2x128x64xindex>
// CHECK-NOT:       linalg.generic
// CHECK:           tt.return
tt.func @copy_both_annotated(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [128, 128], strides: [128, 1] {coordinate_set = #sc, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #idc, access_tile_set = #sc} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
  %al = ktdp.load %at : <128x128xindex> -> tensor<128x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [128, 128], strides: [128, 1] {coordinate_set = #sc, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #idc, access_tile_set = #sc} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
  ktdp.store %al, %ot : tensor<128x128xf32>, <128x128xindex>
  tt.return
}
}

// -----

// Case 5 -- an annotated load through an elementwise generic to an UNANNOTATED
// store.
//
// The store's destination carries no layout, so the generic's outs stays logical at
// [64, 128] and holds the split dim whole: its map carries the composite
// d1 * 64 + d2. The input names the pair directly. So the shape change stops at the
// generic's result rather than reaching the store, and no bridging loop, slice or
// insert is emitted to get it there -- the maps do all of it.

// CHECK: #[[$UAS_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$UAS_ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$UAS_IN:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: #[[$UAS_OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d1 * 64 + d2)>
// CHECK: #[[$UAS_SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$UAS_SET2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#ide = affine_map<(d0, d1) -> (d0, d1)>
#se = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL:   tt.func @elementwise_unannotated_store(
// CHECK:           %[[AV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$UAS_SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// The destination stays logical, and its tile takes no divsi/remsi.
// CHECK:           %[[OV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [64, 128], strides: [128, 1] {coordinate_set = #[[$UAS_SET2]], memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
// CHECK-NOT:       arith.divsi
// CHECK:           %[[OT:.*]] = ktdp.construct_access_tile %[[OV]]{{.*}} {access_tile_order = #[[$UAS_ID2]], access_tile_set = #[[$UAS_SET2]]} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
// CHECK:           %[[E:.*]] = tensor.empty() : tensor<64x128xf32>
// Three parallel loops, the outs at logical shape and carrying the composite.
// CHECK:           %[[R:.*]] = linalg.generic {indexing_maps = [#[[$UAS_IN]], #[[$UAS_OUT]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[AL]] : tensor<2x64x64xf32>) outs(%[[E]] : tensor<64x128xf32>) {
// CHECK:           } -> tensor<64x128xf32>
// No bridging loop and no slicing between the generic and the store.
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.extract_slice
// CHECK:           ktdp.store %[[R]], %[[OT]] : tensor<64x128xf32>, <64x128xindex>
tt.func @elementwise_unannotated_store(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #se, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #ide, access_tile_set = #se} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #se, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #ide, access_tile_set = #se} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#ide, #ide], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 6 -- a rank-3 3-cycle permutation into a split store.
//
// A batch contraction on no layout feeds a permuting generic whose store target
// IS annotated, so the permutation and the split land in the SAME map. The
// permutation is [2, 0, 1], a 3-cycle: it is not its own inverse, so restating it
// as its inverse -- or dropping it -- changes the shape rather than just the
// values, and the emitted map has to compose it with the split's linearization in
// one expression.
//
// The contraction is left alone: none of its operands is annotated, so it is not
// adjacent to any physicalized view. The permuting generic's `outs` is a
// tensor.empty that follows the store to [96/32, 4, 64, 96%32] = [3, 4, 64, 32],
// and its input map becomes (d1, d2, d0 * 32 + d3) -- the 3-cycle's positions
// carrying dim 0's composite. One generic, one map: no second permutation and no
// scatter, which is what the named pass emitted for this kernel.
//
// bmm A[4, 64, 128] @ B[4, 128, 96] -> [4, 64, 96], permuted to [96, 4, 64], then
// stored to a D that is stick-on-dim-0 at width 32.

// CHECK: #[[$S6_BA:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$S6_BB:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$S6_BC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$S6_PERM:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0 * 32 + d3)>
// CHECK: #[[$S6_ID4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#bA = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#bB = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#bC = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#perm = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 95 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
// CHECK-LABEL:   tt.func @sink_trans_3cycle_bmm(
// The contraction is untouched: rank 3 in, rank 3 out, original maps.
// CHECK:           %[[BMM:.*]] = linalg.generic {indexing_maps = [#[[$S6_BA]], #[[$S6_BB]], #[[$S6_BC]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%{{.*}}, %{{.*}} : tensor<4x64x128xf32>, tensor<4x128x96xf32>) outs(%{{.*}} : tensor<4x64x96xf32>) {
// CHECK:           } -> tensor<4x64x96xf32>
// The permuting generic's outs follows the store's physical shape.
// CHECK:           %[[E:.*]] = tensor.empty() : tensor<3x4x64x32xf32>
// CHECK:           %[[P:.*]] = linalg.generic {indexing_maps = [#[[$S6_PERM]], #[[$S6_ID4]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[BMM]] : tensor<4x64x96xf32>) outs(%[[E]] : tensor<3x4x64x32xf32>) {
// CHECK:           } -> tensor<3x4x64x32xf32>
// The 3-cycle is expressed once, inside that one map. Between that generic and
// the store the named pass emitted a second linalg.transpose for the layout and a
// scatter loop over the three sticks; nothing stands here.
// CHECK-NOT:       linalg.transpose
// CHECK-NOT:       linalg.generic
// CHECK-NOT:       scf.for
// CHECK-NOT:       tensor.insert_slice
// CHECK:           ktdp.store %[[P]], %{{.*}} : tensor<3x4x64x32xf32>, <3x4x64x32xindex>
// CHECK:           tt.return
tt.func @sink_trans_3cycle_bmm(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<4x64x96xf32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [4, 64, 128], strides: [8192, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x64x128xf32>
  %2 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x64x128xf32> -> !ktdp.access_tile<4x64x128xindex>
  %3 = ktdp.load %2 : <4x64x128xindex> -> tensor<4x64x128xf32>
  %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
  %5 = ktdp.construct_memory_view %4, sizes: [4, 128, 96], strides: [12288, 96, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<4x128x96xf32>
  %6 = ktdp.construct_access_tile %5[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<4x128x96xf32> -> !ktdp.access_tile<4x128x96xindex>
  %7 = ktdp.load %6 : <4x128x96xindex> -> tensor<4x128x96xf32>
  %8 = linalg.generic {indexing_maps = [#bA, #bB, #bC], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<4x64x128xf32>, tensor<4x128x96xf32>) outs(%cst : tensor<4x64x96xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32):
    %p = arith.mulf %x, %y : f32
    %s = arith.addf %z, %p : f32
    linalg.yield %s : f32
  } -> tensor<4x64x96xf32>
  %9 = tensor.empty() : tensor<96x4x64xf32>
  %10 = linalg.generic {indexing_maps = [#perm, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<4x64x96xf32>) outs(%9 : tensor<96x4x64xf32>) {
  ^bb0(%x: f32, %o: f32):
    linalg.yield %x : f32
  } -> tensor<96x4x64xf32>
  %11 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
  %12 = ktdp.construct_memory_view %11, sizes: [96, 4, 64], strides: [256, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 2, 0>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 32, 0, 0, 32>}} : memref<96x4x64xf32>
  %13 = ktdp.construct_access_tile %12[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<96x4x64xf32> -> !ktdp.access_tile<96x4x64xindex>
  ktdp.store %10, %13 : tensor<96x4x64xf32>, <96x4x64xindex>
  tt.return
}
}
