// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file | FileCheck %s

// A splat physical dim is direction-aware (#225).
//
// A splat layout replicates logical [M] across `phys_arg` lanes, [M] -> [M, S].
// A write fills every lane: its access tile is S wide on the splat dim and the
// generic supplying it names that dim with a loop of its own. A read takes lane
// 0: its access tile is 1 wide on the splat dim, the load result follows, and
// the generic reading it names that dim with the constant 0 and allocates no
// loop for it. The view itself is [M, S] whichever way it is accessed.
//
// Case 1 is a store and a load through one view on separate access tiles, case 2
// the same through ONE access tile (the shape CSE leaves), which the pass splits
// into one physical tile per direction, and case 3 a view that is only read.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py.

// Case 1 -- a statistic stored and read back through one splat view.
//
// G1 reduces x over its stick-split N and writes the row sums S wide; G2 reads
// them back one lane wide and subtracts them from every element of x.

// CHECK-DAG: #[[$RIN:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[$ROUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$LANE0:.+]] = affine_map<(d0, d1, d2) -> (d1, 0)>
// CHECK-DAG: #[[$SET_S:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$SET_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @stat_chain_one_view(
// CHECK-NOT:     tts.tensor_layout
// The view is the full [M, S] for both directions.
// CHECK:         %[[SV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [64, 64], strides: [64, 1] {coordinate_set = #[[$SET_S]], memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
// G1: the write keeps its S-wide tile and its splat loop d3.
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f16) outs(%{{.*}} : tensor<64x64xf16>) -> tensor<64x64xf16>
// CHECK:         %[[SUM:.*]] = linalg.generic {indexing_maps = [#[[$RIN]], #[[$ROUT]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%{{.*}} : tensor<2x64x64xf16>) outs(%[[FILL]] : tensor<64x64xf16>)
// CHECK:         %[[WT:.*]] = ktdp.construct_access_tile %[[SV]][%{{.*}}, %{{.*}}] {access_tile_order = #{{.*}}, access_tile_set = #[[$SET_S]]} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
// CHECK:         ktdp.store %[[SUM]], %[[WT]] : tensor<64x64xf16>, <64x64xindex>
// G2: the read is one lane wide, and its map names the splat dim as 0.
// CHECK:         %[[RT:.*]] = ktdp.construct_access_tile %[[SV]][%{{.*}}, %{{.*}}] {access_tile_order = #{{.*}}, access_tile_set = #[[$SET_1]]} : memref<64x64xf16> -> !ktdp.access_tile<64x1xindex>
// CHECK:         %[[RL:.*]] = ktdp.load %[[RT]] : <64x1xindex> -> tensor<64x1xf16>
// CHECK:         %[[DIFF:.*]] = linalg.generic {indexing_maps = [#[[$ID3]], #[[$LANE0]], #[[$ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}}, %[[RL]] : tensor<2x64x64xf16>, tensor<64x1xf16>) outs(%{{.*}} : tensor<2x64x64xf16>)
// CHECK:         ktdp.store %[[DIFF]], %{{.*}} : tensor<2x64x64xf16>, <2x64x64xindex>
// CHECK:         tt.return
tt.func @stat_chain_one_view(%x_ptr: index, %stat_ptr: index, %out_ptr: index) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f16
  %x_desc = ktdp.construct_memory_view %x_ptr, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %stat = ktdp.construct_memory_view %stat_ptr, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>}} : memref<64xf16>
  %out_desc = ktdp.construct_memory_view %out_ptr, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %x1 = ktdp.construct_access_tile %x_desc[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %x1_l = ktdp.load %x1 : <64x128xindex> -> tensor<64x128xf16>
  %e0 = tensor.empty() : tensor<64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%e0 : tensor<64xf16>) -> tensor<64xf16>
  %sum = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%x1_l : tensor<64x128xf16>) outs(%init : tensor<64xf16>) {
  ^bb0(%in: f16, %acc: f16):
    %a = arith.addf %in, %acc : f16
    linalg.yield %a : f16
  } -> tensor<64xf16>
  %wt = ktdp.construct_access_tile %stat[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  ktdp.store %sum, %wt : tensor<64xf16>, <64xindex>
  %x2 = ktdp.construct_access_tile %x_desc[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %x2_l = ktdp.load %x2 : <64x128xindex> -> tensor<64x128xf16>
  %rt = ktdp.construct_access_tile %stat[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %s = ktdp.load %rt : <64xindex> -> tensor<64xf16>
  %e1 = tensor.empty() : tensor<64x128xf16>
  %diff = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%x2_l, %s : tensor<64x128xf16>, tensor<64xf16>) outs(%e1 : tensor<64x128xf16>) {
  ^bb0(%in: f16, %in_s: f16, %acc: f16):
    %d = arith.subf %in, %in_s : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %out_desc[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %diff, %ot : tensor<64x128xf16>, <64x128xindex>
  tt.return
}
}

// -----

// Case 2 -- case 1 with the statistic's store and load sharing ONE access tile.
//
// The logical tile has a store user and a load user. It becomes two physical
// tiles over the same view and subscripts: an S-wide one the store is redirected
// to and a 1-wide one the load is, each with the dense range of its own block as
// its access_tile_set.

// CHECK-DAG: #[[$C2_ROUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$C2_LANE0:.+]] = affine_map<(d0, d1, d2) -> (d1, 0)>
// CHECK-DAG: #[[$C2_SET_S:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[$C2_SET_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @stat_chain_shared_tile(
// CHECK:         %[[SV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [64, 64], strides: [64, 1] {{.*}} : memref<64x64xf16>
// The two physical tiles, both over the one view.
// CHECK-DAG:     %[[WT:.*]] = ktdp.construct_access_tile %[[SV]][%{{.*}}, %{{.*}}] {access_tile_order = #{{.*}}, access_tile_set = #[[$C2_SET_S]]} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
// CHECK-DAG:     %[[RT:.*]] = ktdp.construct_access_tile %[[SV]][%{{.*}}, %{{.*}}] {access_tile_order = #{{.*}}, access_tile_set = #[[$C2_SET_1]]} : memref<64x64xf16> -> !ktdp.access_tile<64x1xindex>
// CHECK:         %[[SUM:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #[[$C2_ROUT]]], {{.*}}} ins(%{{.*}} : tensor<2x64x64xf16>) outs(%{{.*}} : tensor<64x64xf16>)
// CHECK:         ktdp.store %[[SUM]], %[[WT]] : tensor<64x64xf16>, <64x64xindex>
// CHECK:         %[[RL:.*]] = ktdp.load %[[RT]] : <64x1xindex> -> tensor<64x1xf16>
// CHECK:         linalg.generic {indexing_maps = [#{{.*}}, #[[$C2_LANE0]], #{{.*}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}}, %[[RL]] : tensor<2x64x64xf16>, tensor<64x1xf16>) outs(%{{.*}} : tensor<2x64x64xf16>)
// CHECK:         tt.return
tt.func @stat_chain_shared_tile(%x_ptr: index, %stat_ptr: index, %out_ptr: index) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f16
  %x_desc = ktdp.construct_memory_view %x_ptr, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %stat = ktdp.construct_memory_view %stat_ptr, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>}} : memref<64xf16>
  %out_desc = ktdp.construct_memory_view %out_ptr, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %xt = ktdp.construct_access_tile %x_desc[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %st = ktdp.construct_access_tile %stat[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %x1_l = ktdp.load %xt : <64x128xindex> -> tensor<64x128xf16>
  %e0 = tensor.empty() : tensor<64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%e0 : tensor<64xf16>) -> tensor<64xf16>
  %sum = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%x1_l : tensor<64x128xf16>) outs(%init : tensor<64xf16>) {
  ^bb0(%in: f16, %acc: f16):
    %a = arith.addf %in, %acc : f16
    linalg.yield %a : f16
  } -> tensor<64xf16>
  ktdp.store %sum, %st : tensor<64xf16>, <64xindex>
  %x2_l = ktdp.load %xt : <64x128xindex> -> tensor<64x128xf16>
  %s = ktdp.load %st : <64xindex> -> tensor<64xf16>
  %e1 = tensor.empty() : tensor<64x128xf16>
  %diff = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%x2_l, %s : tensor<64x128xf16>, tensor<64xf16>) outs(%e1 : tensor<64x128xf16>) {
  ^bb0(%in: f16, %in_s: f16, %acc: f16):
    %d = arith.subf %in, %in_s : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %out_desc[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %diff, %ot : tensor<64x128xf16>, <64x128xindex>
  tt.return
}
}

// -----

// Case 3 -- a splat view that is only read.
//
// No store reaches the statistic, so nothing in the function is S wide on the
// splat dim but the view. The rebuilt generic has the three loops the split
// output gives it and none for the splat.

// CHECK-DAG: #[[$C3_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$C3_LANE0:.+]] = affine_map<(d0, d1, d2) -> (d1, 0)>
// CHECK-DAG: #[[$C3_SET_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @splat_read_only(
// CHECK:         %[[SV:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [64, 64], strides: [64, 1] {{.*}} : memref<64x64xf16>
// CHECK:         %[[RT:.*]] = ktdp.construct_access_tile %[[SV]][%{{.*}}, %{{.*}}] {access_tile_order = #{{.*}}, access_tile_set = #[[$C3_SET_1]]} : memref<64x64xf16> -> !ktdp.access_tile<64x1xindex>
// CHECK:         %[[RL:.*]] = ktdp.load %[[RT]] : <64x1xindex> -> tensor<64x1xf16>
// CHECK:         %[[DIFF:.*]] = linalg.generic {indexing_maps = [#[[$C3_ID3]], #[[$C3_LANE0]], #[[$C3_ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}}, %[[RL]] : tensor<2x64x64xf16>, tensor<64x1xf16>) outs(%{{.*}} : tensor<2x64x64xf16>)
// CHECK:         ktdp.store %[[DIFF]], %{{.*}} : tensor<2x64x64xf16>, <2x64x64xindex>
// CHECK:         tt.return
tt.func @splat_read_only(%x_ptr: index, %stat_ptr: index, %out_ptr: index) {
  %c0 = arith.constant 0 : index
  %x_desc = ktdp.construct_memory_view %x_ptr, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %stat = ktdp.construct_memory_view %stat_ptr, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>}} : memref<64xf16>
  %out_desc = ktdp.construct_memory_view %out_ptr, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %xt = ktdp.construct_access_tile %x_desc[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %x_l = ktdp.load %xt : <64x128xindex> -> tensor<64x128xf16>
  %rt = ktdp.construct_access_tile %stat[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %s = ktdp.load %rt : <64xindex> -> tensor<64xf16>
  %e1 = tensor.empty() : tensor<64x128xf16>
  %diff = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%x_l, %s : tensor<64x128xf16>, tensor<64xf16>) outs(%e1 : tensor<64x128xf16>) {
  ^bb0(%in: f16, %in_s: f16, %acc: f16):
    %d = arith.subf %in, %in_s : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %out_desc[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %diff, %ot : tensor<64x128xf16>, <64x128xindex>
  tt.return
}
}
