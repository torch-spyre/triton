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
// The three cases are the three ways that can land, and nothing in any of them is
// transpose-specific or store-specific: both operands splitting the same loop dim
// (so no composite is needed and the maps come out identical), the operands
// splitting different loop dims (so each carries the other's composite), and the
// same at a ktdp.store, which is why this pass has no widening stage.
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
// the shape its access tile names, by construction: there is nothing left for a
// widening stage to do, which is why this pass has none.

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
