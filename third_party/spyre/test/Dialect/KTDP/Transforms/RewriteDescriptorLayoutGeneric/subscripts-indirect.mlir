// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics
// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics | FileCheck %s

// An indirect access tile's subscripts are split by affine substitution.
//
// ktdp.construct_indirect_access_tile carries one affine subscript MAP per base
// dim, over a space of intermediate variables, rather than one SSA index per dim.
// So the split of a logical dim is not the arith.divsi/remsi pair a direct tile
// gets (see subscripts-direct.mlir) -- it is a substitution on those maps.
// Refining the variable space splits the logical dim into a (stick, elem) pair,
// and the subscript that named the whole dim recovers it as `stick * W + elem`:
// the same composite the generic rewrite emits when one operand holds a dim some
// other operand splits.
//
// Neither the tile's rank nor the indirect subscript's position is a parameter of
// the rewrite. It walks physical dims; each one names the logical dim it came from
// and the coord op that made it, and the indirect subscript rides along at
// whatever physical position its logical dim lands at. Cases 1 to 3 are the
// variations that shows up in; case 4 is the one shape that cannot be expressed
// at all. Cases 5 to 7 are an indirect tile with both a load and a store: under
// a splat layout with a broadcast read, without that read, and without a splat
// dim.
//
// Cases 2 to 4 are unreachable from Triton-level input: tt.descriptor_gather only
// ever builds a rank-2 tile with the indirect subscript at dim 0, so only a module
// hand-written at this level -- post-LowerDescriptorMemory KTIR, fed straight to
// the pass -- can present a higher rank, a different indirect position, or a
// layout that splits the indirect dim.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py, which emits only positive CHECKs and can write
// neither the CHECK-NOT in case 1 nor case 4's expected-error.

// Case 1 -- rank 2, the indirect subscript at dim 0: what tt.descriptor_gather
// emits.
//
// The gathered dim 0 is the indirect subscript and stays whole; the contiguous
// dim 1 is what splits, and its subscript is recovered as the composite.
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
// The data view is physical, and so is the tile over it.
// CHECK:           %[[DATAV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [2, 512, 64], strides: [32768, 64, 1]
// CHECK-SAME:        : memref<2x512x64xf32>
// CHECK:           %[[TILE:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[V0:.*]], %[[V1:.*]], %[[V2:.*]]) %[[DATAV]]{{\[}}((%[[C0]] + %[[V0]] * 64 + %[[V2]]) floordiv 64), ind(%[[IDXV]]{{\[}}%[[C0]] + %[[V1]]]), ((%[[C0]] + %[[V0]] * 64 + %[[V2]]) mod 64)]
// CHECK-SAME:        -> !ktdp.access_tile<2x32x64xindex>
// CHECK:           %[[LOAD:.*]] = ktdp.load %[[TILE]] : <2x32x64xindex> -> tensor<2x32x64xf32>
//
// The output descriptor carries the same layout, so its DIRECT tile is split by
// arith.divsi/remsi on the subscript rather than by a substitution -- the two
// carriers, side by side in one kernel.
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
// No layout attribute survives: the physical views are clones with
// tts.tensor_layout removed, and the logical views they superseded are gone.
// CHECK-NOT:       tts.tensor_layout
tt.func @gather_with_layout(%data: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index

  // Index tensor: 32 row indices, unannotated, so it stays logical.
  %ii = builtin.unrealized_conversion_cast %idx : !tt.ptr<i32> to index
  %iv = ktdp.construct_memory_view %ii, sizes: [32], strides: [1] {coordinate_set = #sidx, memory_space = #ktdp.memory_space<global>} : memref<32xi32>

  // Data: [512, 128] stick-on-N(64).
  %di = builtin.unrealized_conversion_cast %data : !tt.ptr<f32> to index
  %dv = ktdp.construct_memory_view %di, sizes: [512, 128], strides: [128, 1] {coordinate_set = #sdata, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<512x128xf32>
  %dt = ktdp.construct_indirect_access_tile intermediate_variables(%v0, %v1) %dv[ind(%iv[%c0 + %v0]), (%c0 + %v1)] {variables_space_order = #varorder, variables_space_set = #stile} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
  %dl = ktdp.load %dt : <32x128xindex> -> tensor<32x128xf32>

  // Output: [32, 128] under the same layout, reached by a direct tile.
  %oi = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [32, 128], strides: [128, 1] {coordinate_set = #stile, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<32x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #varorder, access_tile_set = #stile} : memref<32x128xf32> -> !ktdp.access_tile<32x128xindex>
  ktdp.store %dl, %ot : tensor<32x128xf32>, <32x128xindex>
  tt.return
}
}

// -----

// Case 2 -- rank 3, with the indirect subscript in the MIDDLE.
//
// Logical dim 2 is stick-split, so it contributes two physical dims and its
// subscript is recovered as `stick * 64 + elem`; logical dims 0 and 1 are carried
// whole. The indirect subscript lands at physical position 2, since the split of
// dim 2 inserts a stick dim ahead of it.
#order3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#s_idx = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#s_base = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#s_tile = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 127 >= 0)>
module {
// CHECK: #[[$ORDER:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$SET_IDX:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK: #[[$SET_BASE:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 511 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$SET_TILE:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK-LABEL: tt.func @rank3_indirect_middle(
// CHECK-SAME:      %[[BASE:.*]]: index, %[[IDX:.*]]: index, %[[C0:.*]]: index) {
tt.func @rank3_indirect_middle(%base: index, %idx: index, %c0: index) {
  %iv = ktdp.construct_memory_view %idx, sizes: [32], strides: [1] {coordinate_set = #s_idx, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  // Stick-on-dim-2 (stick=64), dims 0 and 1 whole:
  //   phys_src=[0, 2, 1, 2] phys_op=[id, floordiv, id, mod] phys_arg=[0, 64, 0, 64]
  //   => physical base [4, 128/64, 512, 64] = [4, 2, 512, 64]
  %mv = ktdp.construct_memory_view %base, sizes: [4, 512, 128], strides: [65536, 128, 1] {coordinate_set = #s_base, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 2, 1, 2>, phys_op = array<i64: 0, 1, 0, 2>, phys_arg = array<i64: 0, 64, 0, 64>}} : memref<4x512x128xf32>
  // CHECK:      %[[IV:.*]] = ktdp.construct_memory_view %[[IDX]], sizes: [32]
  // CHECK:      %[[MV:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [4, 2, 512, 64], strides: [65536, 32768, 64, 1] {coordinate_set = #[[$SET_BASE]], memory_space = #ktdp.memory_space<global>} : memref<4x2x512x64xf32>
  // The split of logical dim 2 gives variables %[[S]] (stick) and %[[E]] (elem);
  // %[[B]] and %[[M]] carry logical dims 0 and 1 whole.
  // CHECK:      %[[T:.*]] = ktdp.construct_indirect_access_tile intermediate_variables(%[[B:.*]], %[[S:.*]], %[[M:.*]], %[[E:.*]]) %[[MV]][(%[[C0]] + %[[B]]), ((%[[C0]] + %[[S]] * 64 + %[[E]]) floordiv 64), ind(%[[IV]]{{\[}}%[[C0]] + %[[M]]]), ((%[[C0]] + %[[S]] * 64 + %[[E]]) mod 64)] {variables_space_order = #[[$ORDER]], variables_space_set = #[[$SET_TILE]]} : memref<4x2x512x64xf32>, memref<32xi32> -> !ktdp.access_tile<4x2x32x64xindex>
  // CHECK:      ktdp.load %[[T]] : <4x2x32x64xindex> -> tensor<4x2x32x64xf32>
  %t = ktdp.construct_indirect_access_tile intermediate_variables(%b, %m, %k) %mv[(%c0 + %b), ind(%iv[%c0 + %m]), (%c0 + %k)] {variables_space_order = #order3, variables_space_set = #s_tile} : memref<4x512x128xf32>, memref<32xi32> -> !ktdp.access_tile<4x32x128xindex>
  %l = ktdp.load %t : <4x32x128xindex> -> tensor<4x32x128xf32>
  tt.return
}
}

// -----

// Case 3 -- a BROADCAST physical dim on a direct subscript.
//
// The replication axis addresses no element of the logical dim it names, so its
// subscript is the axis's own origin: the constant 0. That is the same answer
// physicalizeAccessTile gives a direct tile by pushing an arith.constant 0 --
// here it is a constant affine expression instead, because the carrier is a map.
// The tile's only user is a load, so it is 1 wide on that dim: a read takes lane
// 0 (#225).
//
// A broadcast on an INDIRECT dim is not reachable: it is a non-identity coord op,
// so case 4's gate takes it first.
#order2b = affine_map<(d0, d1) -> (d0, d1)>
#s_idx3 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#s_base3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 0 >= 0)>
#s_tile3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 0 >= 0)>
module {
// CHECK-LABEL: tt.func @broadcast_direct_dim(
// CHECK-SAME:      %[[BASE:.*]]: index, %[[IDX:.*]]: index, %[[C0:.*]]: index) {
tt.func @broadcast_direct_dim(%base: index, %idx: index, %c0: index) {
  %iv = ktdp.construct_memory_view %idx, sizes: [32], strides: [1] {coordinate_set = #s_idx3, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  // CHECK:      %[[IV:.*]] = ktdp.construct_memory_view %[[IDX]], sizes: [32]
  // Logical dim 1 carried whole AND replicated across 64 lanes:
  //   phys_src=[0, 1, 1] phys_op=[id, id, broadcast] phys_arg=[0, 0, 64]
  %mv = ktdp.construct_memory_view %base, sizes: [512, 1], strides: [1, 1] {coordinate_set = #s_base3, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 0, 3>, phys_arg = array<i64: 0, 0, 64>}} : memref<512x1xf32>
  // CHECK:      %[[MV:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [512, 1, 64]
  // CHECK:      ktdp.construct_indirect_access_tile intermediate_variables(%[[M:.*]], %[[K:.*]], %{{.*}}) %[[MV]][ind(%[[IV]]{{\[}}%[[C0]] + %[[M]]]), (%[[C0]] + %[[K]]), (0)] {{.*}} -> !ktdp.access_tile<32x1x1xindex>
  %t = ktdp.construct_indirect_access_tile intermediate_variables(%m, %k) %mv[ind(%iv[%c0 + %m]), (%c0 + %k)] {variables_space_order = #order2b, variables_space_set = #s_tile3} : memref<512x1xf32>, memref<32xi32> -> !ktdp.access_tile<32x1xindex>
  %l = ktdp.load %t : <32x1xindex> -> tensor<32x1xf32>
  tt.return
}
}

// -----

// Case 4 -- the layout stick-splits the INDIRECT dim, which cannot be expressed.
//
// A representational limit of the op, not missing work. `ind(IDX[expr])` does not
// compute the base coordinate: it computes an index into IDX, and the coordinate
// is the value LOADED from there. The split's floordiv/mod would have to apply to
// that loaded value, and an affine subscript expression cannot reference the
// result of a load -- so there is nowhere to write it. Splitting `expr` instead
// would split the position in the index array rather than the coordinate it
// yields, which addresses different elements.
#order2 = affine_map<(d0, d1) -> (d0, d1)>
#s_idx2 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#s_base2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#s_tile2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @split_indirect_dim(%base: index, %idx: index, %c0: index) {
  %iv = ktdp.construct_memory_view %idx, sizes: [32], strides: [1] {coordinate_set = #s_idx2, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  // Stick-on-dim-0 (stick=32) -- and dim 0 is the indirect one.
  %mv = ktdp.construct_memory_view %base, sizes: [512, 128], strides: [128, 1] {coordinate_set = #s_base2, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}} : memref<512x128xf32>
  // expected-error @below {{tts.tensor_layout: logical dim 0 is an indirect (gather) subscript, so it cannot be stick-split}}
  %t = ktdp.construct_indirect_access_tile intermediate_variables(%m, %k) %mv[ind(%iv[%c0 + %m]), (%c0 + %k)] {variables_space_order = #order2, variables_space_set = #s_tile2} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
  %l = ktdp.load %t : <32x128xindex> -> tensor<32x128xf32>
  tt.return
}
}

// -----

// Case 5 -- one indirect tile, loaded and stored, under a splat layout.
//
// The load's only use is a generic's `ins`, so it is a Lane0 load: the tile is
// rebuilt twice over the same subscripts, 64 lanes wide for the store and one lane
// wide for the load, and the generic reads the splat dim at lane 0.
// CHECK-DAG: #[[$C5_LANE0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// CHECK-DAG: #[[$C5_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$C5_SET_S:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-DAG: #[[$C5_SET_1:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#order5 = affine_map<(d0, d1) -> (d0, d1)>
#id5 = affine_map<(d0, d1) -> (d0, d1)>
#s_idx5 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#s_base5 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 0 >= 0)>
#s_tile5 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 0 >= 0)>
module {
// CHECK-LABEL: tt.func @splat_load_and_store(
// CHECK:         %[[MV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [512, 1, 64]
// CHECK:         %[[WT:.*]] = ktdp.construct_indirect_access_tile {{.*}} %[[MV]][{{.*}}, (0)] {variables_space_order = #[[$C5_ID3]], variables_space_set = #[[$C5_SET_S]]} {{.*}} -> !ktdp.access_tile<32x1x64xindex>
// CHECK:         %[[RT:.*]] = ktdp.construct_indirect_access_tile {{.*}} %[[MV]][{{.*}}, (0)] {variables_space_order = #[[$C5_ID3]], variables_space_set = #[[$C5_SET_1]]} {{.*}} -> !ktdp.access_tile<32x1x1xindex>
// CHECK:         %[[L:.*]] = ktdp.load %[[RT]] : <32x1x1xindex> -> tensor<32x1x1xf32>
// CHECK:         %[[R:.*]] = linalg.generic {indexing_maps = [#[[$C5_LANE0]], #[[$C5_ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[L]] : tensor<32x1x1xf32>) outs(%{{.*}} : tensor<32x1x64xf32>)
// CHECK:         ktdp.store %[[R]], %[[WT]] : tensor<32x1x64xf32>, <32x1x64xindex>
tt.func @splat_load_and_store(%base: index, %idx: index, %c0: index) {
  %iv = ktdp.construct_memory_view %idx, sizes: [32], strides: [1] {coordinate_set = #s_idx5, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  %mv = ktdp.construct_memory_view %base, sizes: [512, 1], strides: [1, 1] {coordinate_set = #s_base5, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 0, 3>, phys_arg = array<i64: 0, 0, 64>}} : memref<512x1xf32>
  %t = ktdp.construct_indirect_access_tile intermediate_variables(%m, %k) %mv[ind(%iv[%c0 + %m]), (%c0 + %k)] {variables_space_order = #order5, variables_space_set = #s_tile5} : memref<512x1xf32>, memref<32xi32> -> !ktdp.access_tile<32x1xindex>
  %l = ktdp.load %t : <32x1xindex> -> tensor<32x1xf32>
  %e = tensor.empty() : tensor<32x1xf32>
  %r = linalg.generic {indexing_maps = [#id5, #id5], iterator_types = ["parallel", "parallel"]} ins(%l : tensor<32x1xf32>) outs(%e : tensor<32x1xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %d = arith.addf %in, %in : f32
    linalg.yield %d : f32
  } -> tensor<32x1xf32>
  ktdp.store %r, %t : tensor<32x1xf32>, <32x1xindex>
  tt.return
}
}

// -----

// Case 6 -- the same indirect tile under the splat layout, its load stored back
// directly. A store is not a broadcasting use, so the load keeps the full stick
// and shares the store's one tile. The load and the store sharing %[[T]] is
// the check.
#order6 = affine_map<(d0, d1) -> (d0, d1)>
#s_idx6 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#s_base6 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 0 >= 0)>
#s_tile6 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 0 >= 0)>
module {
// CHECK-LABEL: tt.func @splat_load_stored_back(
// CHECK:         %[[T:.*]] = ktdp.construct_indirect_access_tile {{.*}} -> !ktdp.access_tile<32x1x64xindex>
// CHECK:         %[[L:.*]] = ktdp.load %[[T]] : <32x1x64xindex> -> tensor<32x1x64xf32>
// CHECK:         ktdp.store %[[L]], %[[T]] : tensor<32x1x64xf32>, <32x1x64xindex>
tt.func @splat_load_stored_back(%base: index, %idx: index, %c0: index) {
  %iv = ktdp.construct_memory_view %idx, sizes: [32], strides: [1] {coordinate_set = #s_idx6, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  %mv = ktdp.construct_memory_view %base, sizes: [512, 1], strides: [1, 1] {coordinate_set = #s_base6, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 1>, phys_op = array<i64: 0, 0, 3>, phys_arg = array<i64: 0, 0, 64>}} : memref<512x1xf32>
  %t = ktdp.construct_indirect_access_tile intermediate_variables(%m, %k) %mv[ind(%iv[%c0 + %m]), (%c0 + %k)] {variables_space_order = #order6, variables_space_set = #s_tile6} : memref<512x1xf32>, memref<32xi32> -> !ktdp.access_tile<32x1xindex>
  %l = ktdp.load %t : <32x1xindex> -> tensor<32x1xf32>
  ktdp.store %l, %t : tensor<32x1xf32>, <32x1xindex>
  tt.return
}
}

// -----

// Case 7 -- one indirect tile, loaded and stored, under stick-on-N with no splat
// dim. Every user shares the one physical tile. The load and the store
// sharing %[[T]] is the check.
#order7 = affine_map<(d0, d1) -> (d0, d1)>
#id7 = affine_map<(d0, d1) -> (d0, d1)>
#s_idx7 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#s_base7 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#s_tile7 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: tt.func @stick_load_and_store(
// CHECK:         %[[T:.*]] = ktdp.construct_indirect_access_tile {{.*}} -> !ktdp.access_tile<2x32x64xindex>
// CHECK:         %[[L:.*]] = ktdp.load %[[T]] : <2x32x64xindex> -> tensor<2x32x64xf32>
// CHECK:         ktdp.store %{{.*}}, %[[T]] : tensor<2x32x64xf32>, <2x32x64xindex>
tt.func @stick_load_and_store(%base: index, %idx: index, %c0: index) {
  %iv = ktdp.construct_memory_view %idx, sizes: [32], strides: [1] {coordinate_set = #s_idx7, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  %mv = ktdp.construct_memory_view %base, sizes: [512, 128], strides: [128, 1] {coordinate_set = #s_base7, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<512x128xf32>
  %t = ktdp.construct_indirect_access_tile intermediate_variables(%m, %k) %mv[ind(%iv[%c0 + %m]), (%c0 + %k)] {variables_space_order = #order7, variables_space_set = #s_tile7} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
  %l = ktdp.load %t : <32x128xindex> -> tensor<32x128xf32>
  %e = tensor.empty() : tensor<32x128xf32>
  %r = linalg.generic {indexing_maps = [#id7, #id7], iterator_types = ["parallel", "parallel"]} ins(%l : tensor<32x128xf32>) outs(%e : tensor<32x128xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %d = arith.addf %in, %in : f32
    linalg.yield %d : f32
  } -> tensor<32x128xf32>
  ktdp.store %r, %t : tensor<32x128xf32>, <32x128xindex>
  tt.return
}
}
