// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// What the rebuild leaves alone.
//
// Restating a generic over the physical loop domain replaces each map result that
// names a stick-split logical dim. Everything else is copied across: a map result
// that is not an AffineDimExpr at all is passed through untouched, and a split dim
// that every operand splits the same way needs no arithmetic anywhere. The two
// cases below are those two ways of having nothing to add -- there is no
// broadcast-specific logic and no elementwise-specific logic, which is the claim.
//
// Contrast rebuild-composite.mlir, where one operand holds whole a dim another
// operand splits and the maps do have to carry a composite.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py, which numbers captures globally across a file and
// would renumber every section on any edit.

// Case 1 -- every operand split identically.
//
// All three layouts split logical dim 1 at the same width, so no operand holds a
// split dim whole and no map carries arithmetic. The physical dim order is the
// layout's, (K/W, M, K%W), so the emitted identity is over (ks, m, kl) rather
// than the logical order grown on the right.
//
// The width here is 32, not the 64 every other positive case in this directory
// uses. Nothing in the pass hardcodes a stick width -- it comes from phys_arg --
// and this case is what keeps that true: at 32 the view sizes, the view strides,
// the divsi/remsi constants, the tile type and the tensor types all have to move
// together.

// CHECK: #[[$EW_ORDER:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$EW_SET:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#id = affine_map<(d0, d1) -> (d0, d1)>
#s2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL:   tt.func @elementwise_split_alike(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [4, 64, 32], strides: [2048, 32, 1] {coordinate_set = #[[$EW_SET]], memory_space = #ktdp.memory_space<global>} : memref<4x64x32xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_3]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_9:.*]] = arith.remsi %[[VAL_3]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = ktdp.construct_access_tile %[[VAL_5]]{{\[}}%[[VAL_7]], %[[VAL_3]], %[[VAL_9]]] {access_tile_order = #[[$EW_ORDER]], access_tile_set = #[[$EW_SET]]} : memref<4x64x32xf32> -> !ktdp.access_tile<4x64x32xindex>
// CHECK:           %[[VAL_11:.*]] = ktdp.load %[[VAL_10]] : <4x64x32xindex> -> tensor<4x64x32xf32>
// CHECK:           %[[VAL_12:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_13:.*]] = ktdp.construct_memory_view %[[VAL_12]], sizes: [4, 64, 32], strides: [2048, 32, 1] {coordinate_set = #[[$EW_SET]], memory_space = #ktdp.memory_space<global>} : memref<4x64x32xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_15:.*]] = arith.divsi %[[VAL_3]], %[[VAL_14]] : index
// CHECK:           %[[VAL_16:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_17:.*]] = arith.remsi %[[VAL_3]], %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]] = ktdp.construct_access_tile %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_3]], %[[VAL_17]]] {access_tile_order = #[[$EW_ORDER]], access_tile_set = #[[$EW_SET]]} : memref<4x64x32xf32> -> !ktdp.access_tile<4x64x32xindex>
// CHECK:           %[[VAL_19:.*]] = ktdp.load %[[VAL_18]] : <4x64x32xindex> -> tensor<4x64x32xf32>
// CHECK:           %[[VAL_20:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_21:.*]] = ktdp.construct_memory_view %[[VAL_20]], sizes: [4, 64, 32], strides: [2048, 32, 1] {coordinate_set = #[[$EW_SET]], memory_space = #ktdp.memory_space<global>} : memref<4x64x32xf32>
// CHECK:           %[[VAL_22:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_23:.*]] = arith.divsi %[[VAL_3]], %[[VAL_22]] : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_25:.*]] = arith.remsi %[[VAL_3]], %[[VAL_24]] : index
// CHECK:           %[[VAL_26:.*]] = ktdp.construct_access_tile %[[VAL_21]]{{\[}}%[[VAL_23]], %[[VAL_3]], %[[VAL_25]]] {access_tile_order = #[[$EW_ORDER]], access_tile_set = #[[$EW_SET]]} : memref<4x64x32xf32> -> !ktdp.access_tile<4x64x32xindex>
// CHECK:           %[[VAL_27:.*]] = tensor.empty() : tensor<4x64x32xf32>
// CHECK:           %[[VAL_28:.*]] = linalg.generic {indexing_maps = [#[[$EW_ORDER]], #[[$EW_ORDER]], #[[$EW_ORDER]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_11]], %[[VAL_19]] : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%[[VAL_27]] : tensor<4x64x32xf32>) {
// CHECK:           ^bb0(%[[VAL_29:.*]]: f32, %[[VAL_30:.*]]: f32, %[[VAL_31:.*]]: f32):
// CHECK:             %[[VAL_32:.*]] = arith.addf %[[VAL_29]], %[[VAL_30]] : f32
// CHECK:             linalg.yield %[[VAL_32]] : f32
// CHECK:           } -> tensor<4x64x32xf32>
// CHECK:           ktdp.store %[[VAL_28]], %[[VAL_26]] : tensor<4x64x32xf32>, <4x64x32xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @elementwise_split_alike(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f32> to index
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}} : memref<64x128xf32>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %bl = ktdp.load %bt : <64x128xindex> -> tensor<64x128xf32>
  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}} : memref<64x128xf32>
  %ct = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%al, %bl : tensor<64x128xf32>, tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ct : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 2 -- a broadcast folded into a consumer's map, that operand not split.
//
// The constant 0 in the broadcast operand's map is the broadcast. It is not an
// AffineDimExpr, so it passes through untouched, and the split dims do not appear
// in that operand's map at all -- correct, because the operand does not vary
// along them. That is the whole of it: a map result that is not a dim is passed
// through, with nothing anywhere that knows the word "broadcast".

// CHECK: #[[$ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$LANE0:.+]] = affine_map<(d0, d1, d2) -> (d1, 0)>
// CHECK: #[[$SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$SET_STAT:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 23 >= 0, d1 >= 0, -d1 >= 0)>
#x    = affine_map<(d0, d1) -> (d0, d1)>
#stat = affine_map<(d0, d1) -> (d0, 0)>
#sx  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 23 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#ss  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 23 >= 0, d1 >= 0, -d1 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   tt.func @broadcast_operand_not_split(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_5:.*]] = ktdp.construct_memory_view %[[VAL_4]], sizes: [2, 24, 64], strides: [1536, 64, 1] {coordinate_set = #[[$SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x24x64xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_7:.*]] = arith.divsi %[[VAL_3]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_9:.*]] = arith.remsi %[[VAL_3]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = ktdp.construct_access_tile %[[VAL_5]]{{\[}}%[[VAL_7]], %[[VAL_3]], %[[VAL_9]]] {access_tile_order = #[[$ID3]], access_tile_set = #[[$SET3]]} : memref<2x24x64xf32> -> !ktdp.access_tile<2x24x64xindex>
// CHECK:           %[[VAL_11:.*]] = ktdp.load %[[VAL_10]] : <2x24x64xindex> -> tensor<2x24x64xf32>
// CHECK:           %[[VAL_12:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_13:.*]] = ktdp.construct_memory_view %[[VAL_12]], sizes: [24, 1], strides: [1, 1] {coordinate_set = #[[$SET_STAT]], memory_space = #ktdp.memory_space<global>} : memref<24x1xf32>
// CHECK:           %[[VAL_14:.*]] = ktdp.construct_access_tile %[[VAL_13]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {access_tile_order = #[[$ID2]], access_tile_set = #[[$SET_STAT]]} : memref<24x1xf32> -> !ktdp.access_tile<24x1xindex>
// CHECK:           %[[VAL_15:.*]] = ktdp.load %[[VAL_14]] : <24x1xindex> -> tensor<24x1xf32>
// CHECK:           %[[VAL_16:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_17:.*]] = ktdp.construct_memory_view %[[VAL_16]], sizes: [2, 24, 64], strides: [1536, 64, 1] {coordinate_set = #[[$SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x24x64xf32>
// CHECK:           %[[VAL_18:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_19:.*]] = arith.divsi %[[VAL_3]], %[[VAL_18]] : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_21:.*]] = arith.remsi %[[VAL_3]], %[[VAL_20]] : index
// CHECK:           %[[VAL_22:.*]] = ktdp.construct_access_tile %[[VAL_17]]{{\[}}%[[VAL_19]], %[[VAL_3]], %[[VAL_21]]] {access_tile_order = #[[$ID3]], access_tile_set = #[[$SET3]]} : memref<2x24x64xf32> -> !ktdp.access_tile<2x24x64xindex>
// CHECK:           %[[VAL_23:.*]] = tensor.empty() : tensor<2x24x64xf32>
// The statistic operand's map: the split dims are absent, and the 0 rides through.
// CHECK:           %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#[[$ID3]], #[[$LANE0]], #[[$ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_11]], %[[VAL_15]] : tensor<2x24x64xf32>, tensor<24x1xf32>) outs(%[[VAL_23]] : tensor<2x24x64xf32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: f32, %[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[VAL_28:.*]] = arith.subf %[[VAL_25]], %[[VAL_26]] : f32
// CHECK:             linalg.yield %[[VAL_28]] : f32
// CHECK:           } -> tensor<2x24x64xf32>
// CHECK:           ktdp.store %[[VAL_24]], %[[VAL_22]] : tensor<2x24x64xf32>, <2x24x64xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @broadcast_operand_not_split(%a: !tt.ptr<f32>, %s: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [24, 128], strides: [128, 1] {coordinate_set = #sx, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<24x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sx} : memref<24x128xf32> -> !ktdp.access_tile<24x128xindex>
  %al = ktdp.load %at : <24x128xindex> -> tensor<24x128xf32>
  %si = builtin.unrealized_conversion_cast %s : !tt.ptr<f32> to index
  %sv = ktdp.construct_memory_view %si, sizes: [24, 1], strides: [1, 1] {coordinate_set = #ss, memory_space = #ktdp.memory_space<global>} : memref<24x1xf32>
  %st = ktdp.construct_access_tile %sv[%c0, %c0] {access_tile_order = #id2, access_tile_set = #ss} : memref<24x1xf32> -> !ktdp.access_tile<24x1xindex>
  %sl = ktdp.load %st : <24x1xindex> -> tensor<24x1xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [24, 128], strides: [128, 1] {coordinate_set = #sx, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<24x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sx} : memref<24x128xf32> -> !ktdp.access_tile<24x128xindex>
  %e = tensor.empty() : tensor<24x128xf32>
  %r = linalg.generic {indexing_maps = [#x, #stat, #x], iterator_types = ["parallel", "parallel"]} ins(%al, %sl : tensor<24x128xf32>, tensor<24x1xf32>) outs(%e : tensor<24x128xf32>) {
  ^bb0(%xx: f32, %st2: f32, %oo: f32):
    %d = arith.subf %xx, %st2 : f32
    linalg.yield %d : f32
  } -> tensor<24x128xf32>
  ktdp.store %r, %ot : tensor<24x128xf32>, <24x128xindex>
  tt.return
}
}
