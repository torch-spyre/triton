// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// Operands that split the same logical loop dim from different physical
// positions.
//
// The rebuilt loop domain splits every logical dim that ANY operand splits, so a
// split dim becomes a (stick, lane) pair of loop dims, and every operand that
// reads that dim splits it too -- an operand holding it whole is refused, see
// invalid-layout.mlir cases 7 to 12. So each operand names the pair directly and
// every map is a projected permutation, however the operands order their
// physical dims. Where an operand's physical order differs from the result's,
// the difference is a permutation in its map and nothing more -- see
// rebuild-passthrough.mlir for the cases where there is nothing to permute.
//
// Case 1 is a transpose whose two operands split the SAME loop dim, so the maps
// come out identical. Case 2 is the same at a ktdp.store with a generic between,
// which is why this pass needs no widening stage WHEREVER A GENERIC MEDIATES THE
// STORE.
//
// Case 3 is the other end of that "wherever": a pure load-to-store copy with NO
// generic at all and both ends annotated, which needs none: phase 1 physicalizes
// the two views and the two tiles, and the load's retyped result already matches
// the store's retyped tile. With only the destination annotated the same copy has
// no vehicle for the shape change and the pass declines -- invalid-layout.mlir
// case 4 is that one, and case 3 here is its positive twin.
//
// Case 4 raises the permutation from a rank-2 swap to a rank-3 3-CYCLE, which is
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
// operand's own layout, so the permutation is absorbed and the maps carry none.
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

// Case 2 -- both ends split the same dim across a ktdp.store, and the tiles agree.
//
// This is the case the no-widen-stage claim rests on. The generic's input and the
// store destination both split logical dim 0, so the domain splits it once and
// both operands name its pair. The store's data tile then has exactly the shape
// its access tile names, by construction: with a generic on the chain there is
// nothing left for a widening stage to do.
//
// The condition is the generic, not the store: it is the generic's outs operand
// that the rebuild gives the store's physical shape. Take it away and the two ends
// have nothing to agree through -- invalid-layout.mlir case 4.

// CHECK: #[[$ID3_STORE:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$SET_STORE:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL:   tt.func @store_agrees_without_widening(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [2, 128, 32], strides: [4096, 32, 1] {coordinate_set = #[[$SET_STORE]], memory_space = #ktdp.memory_space<global>} : memref<2x128x32xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_8:.*]] = arith.remsi %[[VAL_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_8]]] {access_tile_order = #[[$ID3_STORE]], access_tile_set = #[[$SET_STORE]]} : memref<2x128x32xf32> -> !ktdp.access_tile<2x128x32xindex>
// CHECK:           %[[VAL_10:.*]] = ktdp.load %[[VAL_9]] : <2x128x32xindex> -> tensor<2x128x32xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_11]], sizes: [2, 128, 32], strides: [4096, 32, 1] {coordinate_set = #[[$SET_STORE]], memory_space = #ktdp.memory_space<global>} : memref<2x128x32xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_14:.*]] = arith.divsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_2]], %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_2]], %[[VAL_16]]] {access_tile_order = #[[$ID3_STORE]], access_tile_set = #[[$SET_STORE]]} : memref<2x128x32xf32> -> !ktdp.access_tile<2x128x32xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<2x128x32xf32>
// CHECK:           %[[VAL_19:.*]] = linalg.generic {indexing_maps = [#[[$ID3_STORE]], #[[$ID3_STORE]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_10]] : tensor<2x128x32xf32>) outs(%[[VAL_18]] : tensor<2x128x32xf32>) {
// CHECK:           ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             %[[VAL_22:.*]] = arith.negf %[[VAL_20]] : f32
// CHECK:             linalg.yield %[[VAL_22]] : f32
// CHECK:           } -> tensor<2x128x32xf32>
// The store's two shapes: the data is <2x128x32> and so is the access tile.
// CHECK:           ktdp.store %[[VAL_19]], %[[VAL_17]] : tensor<2x128x32xf32>, <2x128x32xindex>
// CHECK:           tt.return
// CHECK:         }
tt.func @store_agrees_without_widening(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  // Both ends split logical dim 0.
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}} : memref<64x128xf32>
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

// Case 3 -- a pure load-to-store copy with BOTH ends annotated.
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

// Case 4 -- a rank-3 3-cycle permutation into a split store.
//
// A permuting generic reads a stick-on-dim-2 input and writes a store target that
// is stick-on-dim-0, and both splits name the same logical loop dim -- the
// input's dim 2 and the output's dim 0 are both loop d0 of the generic. The
// permutation is [2, 0, 1], a 3-cycle: it is not its own inverse, so restating it
// as its inverse -- or dropping it -- changes the shape rather than just the
// values. The domain is ordered off the result, so the result's map is the
// identity and the input's carries the 3-cycle over the split pieces:
// (d1, d2, d0, d3). One generic, one map: no second permutation and no scatter,
// which is what the named pass emitted for this kernel.
//
// A[4, 64, 96] stick-on-dim-2(32) -> [4, 64, 3, 32], permuted to [96, 4, 64] and
// stored to a D that is stick-on-dim-0 at width 32 -> [3, 4, 64, 32].

// CHECK: #[[$S6_ID4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$S6_PERM:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#perm = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 95 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
// CHECK-LABEL:   tt.func @sink_trans_3cycle(
// CHECK:           %[[AL:.*]] = ktdp.load %{{.*}} : <4x64x3x32xindex> -> tensor<4x64x3x32xf32>
// The permuting generic's outs follows the store's physical shape.
// CHECK:           %[[E:.*]] = tensor.empty() : tensor<3x4x64x32xf32>
// CHECK:           %[[P:.*]] = linalg.generic {indexing_maps = [#[[$S6_PERM]], #[[$S6_ID4]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[AL]] : tensor<4x64x3x32xf32>) outs(%[[E]] : tensor<3x4x64x32xf32>) {
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
tt.func @sink_trans_3cycle(%arg0: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [4, 64, 96], strides: [6144, 96, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 2, 2>, phys_op = array<i64: 0, 0, 1, 2>, phys_arg = array<i64: 0, 0, 32, 32>}} : memref<4x64x96xf32>
  %2 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x64x96xf32> -> !ktdp.access_tile<4x64x96xindex>
  %3 = ktdp.load %2 : <4x64x96xindex> -> tensor<4x64x96xf32>
  %9 = tensor.empty() : tensor<96x4x64xf32>
  %10 = linalg.generic {indexing_maps = [#perm, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<4x64x96xf32>) outs(%9 : tensor<96x4x64xf32>) {
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
