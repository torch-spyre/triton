// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic=data-layout=device | FileCheck %s

// Every generic here must be rebuilt EXACTLY ONCE.
//
// The pass carries its layouts in a map keyed on `Value`, and a `Value` is a
// handle onto storage MLIR reuses once the op holding it is erased. The rewrite
// erases a generic every time it fires and clones a new one immediately after, so
// an entry left behind for a dead result can be handed to a value allocated onto
// the same address later -- and that value is then judged against a layout that
// was never its own, found inconsistent, and rebuilt a SECOND time over operands
// that are already physical.
//
// The shape below is the smallest one that shows it, and each part is load-bearing:
//
//   the reduce stores its statistic through a BROADCAST layout, physical rank 2,
//   which is a different rank from everything else here -- a stale entry of the
//   same rank would be invisible;
//   the chain is THREE generics deep, so the rewrite fires often enough for a
//   clone to land on a freed address (a two-generic chain does not, which is why
//   `stat_chain_on_stick` never caught this);
//   the last generic is elementwise at physical rank 3, so a rank-2 layout
//   arriving from nowhere makes it inconsistent rather than merely wrong.
//
// The symptom was a domain one loop too wide with the extra loop named by no map,
// reported by the linalg verifier as
//
//   'linalg.generic' op invalid indexing maps are non-invertible:
//   ((d0, d1, d2, d3) -> (d0, d1, d2, d0, d1, d2))
//
// several passes away from anything an author wrote. So this test is a WITNESS,
// not a proof: it depends on the allocator actually reusing that address. What
// guarantees the property is the invalidation itself -- see `forgetLayout` in
// RewriteDescriptorLayoutGeneric.cpp.
//
// Input is the already-lowered form the pass sees, so it does not depend on what
// the frontend happens to emit. It is `max_shift_exp_on_stick` in
// fixtures/reduce/kernel.py: out = exp(x - max(x, axis=1)), fp32, 32-lane sticks.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>

// The five maps the output states, checked where they are printed -- at the top,
// ahead of the function.
// CHECK-DAG:   #[[$ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$RIN:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$ROUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:   #[[$LANE0:.+]] = affine_map<(d0, d1, d2) -> (d1, 0)>

// CHECK-LABEL: tt.func public @max_shift_exp_on_stick
// CHECK-NOT:   tt.spyre_tensor_layout

// G1: the max reduce. Its input is stick-on-N, so the reduced dim is split across
// two loops and the surviving dim rides as the third; the broadcast output adds a
// fourth loop that only its own map names. Four loops here is CORRECT -- it is the
// same shape one_tile_on_stick_bcast_max emits.
//
// Its init is the neutral-element linalg.fill LowerComputeOps puts on every
// reduction, and it follows the reduce to physical rank 2 along with the
// tensor.empty underneath it.
// CHECK:       %[[MAXINIT:.*]] = tensor.empty() : tensor<64x32xf32>
// CHECK:       %[[MAXFILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[MAXINIT]] : tensor<64x32xf32>) -> tensor<64x32xf32>
// CHECK:       linalg.generic {indexing_maps = [#[[$RIN:.+]], #[[$ROUT:.+]]], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%{{.*}} : tensor<2x64x32xf32>) outs(%[[MAXFILL]] : tensor<64x32xf32>)
// CHECK:         arith.maximumf
// CHECK:       ktdp.store %{{.*}} : tensor<64x32xf32>, <64x32xindex>

// G2: the shift. Three loops, the statistic read at a constant lane.
// CHECK:       linalg.generic {indexing_maps = [#[[$ID3:.+]], #[[$LANE0:.+]], #[[$ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}}, %{{.*}} : tensor<2x64x32xf32>, tensor<64x1xf32>) outs(%{{.*}} : tensor<2x64x32xf32>)
// CHECK:         arith.subf
// CHECK:       ktdp.store %{{.*}} : tensor<2x64x32xf32>, <2x64x32xindex>

// G3: the exp. THREE parallel loops and two identity maps -- this is the one that
// came out with four loops and an unnamed d3 before the layouts were invalidated.
// CHECK:       linalg.generic {indexing_maps = [#[[$ID3]], #[[$ID3]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<2x64x32xf32>) outs(%{{.*}} : tensor<2x64x32xf32>)
// CHECK:         math.exp
// CHECK:       ktdp.store %{{.*}} : tensor<2x64x32xf32>, <2x64x32xindex>

module {
  tt.func public @max_shift_exp_on_stick(%x: !tt.ptr<f32>, %max: !tt.ptr<f32>,
                                         %diff: !tt.ptr<f32>, %out: !tt.ptr<f32>)
      attributes {noinline = false} {
    %c0 = arith.constant 0 : index

    %x_i = builtin.unrealized_conversion_cast %x : !tt.ptr<f32> to index
    %x_view = ktdp.construct_memory_view %x_i, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %x_desc = builtin.unrealized_conversion_cast %x_view : memref<64x64xf32> to !tt.tensordesc<64x64xf32>

    %diff_i = builtin.unrealized_conversion_cast %diff : !tt.ptr<f32> to index
    %diff_view = ktdp.construct_memory_view %diff_i, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %diff_desc = builtin.unrealized_conversion_cast %diff_view : memref<64x64xf32> to !tt.tensordesc<64x64xf32>

    %out_i = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
    %out_view = ktdp.construct_memory_view %out_i, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %out_desc = builtin.unrealized_conversion_cast %out_view : memref<64x64xf32> to !tt.tensordesc<64x64xf32>

    // The statistic, written rank-1 through a broadcast layout...
    %max_i = builtin.unrealized_conversion_cast %max : !tt.ptr<f32> to index
    %max_view = ktdp.construct_memory_view %max_i, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf32>
    %max_w = builtin.unrealized_conversion_cast %max_view : memref<64xf32> to !tt.tensordesc<64xf32>
    // ...and read back as the [64, 32] that broadcast made, one lane wide. No
    // layout: its logical shape already is its physical one.
    %max_r_i = builtin.unrealized_conversion_cast %max : !tt.ptr<f32> to index
    %max_r_view = ktdp.construct_memory_view %max_r_i, sizes: [64, 32], strides: [32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>

    tt.spyre_tensor_layout %x_desc {phys_arg = array<i64: 32, 0, 32>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    tt.spyre_tensor_layout %diff_desc {phys_arg = array<i64: 32, 0, 32>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    tt.spyre_tensor_layout %out_desc {phys_arg = array<i64: 32, 0, 32>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    tt.spyre_tensor_layout %max_w {phys_arg = array<i64: 0, 32>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>} : <64xf32>

    // G1: max(x, axis=1) stored stick-wide.
    %t0 = ktdp.construct_access_tile %x_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %x0 = ktdp.load %t0 : <64x64xindex> -> tensor<64x64xf32>
    %ninf = arith.constant 0xFF800000 : f32
    %e0empty = tensor.empty() : tensor<64xf32>
    %e0 = linalg.fill ins(%ninf : f32) outs(%e0empty : tensor<64xf32>) -> tensor<64xf32>
    %maxes = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%x0 : tensor<64x64xf32>) outs(%e0 : tensor<64xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %m = arith.maximumf %in, %acc : f32
      linalg.yield %m : f32
    } -> tensor<64xf32>
    %t1 = ktdp.construct_access_tile %max_view[%c0] {access_tile_order = #map2, access_tile_set = #set1} : memref<64xf32> -> !ktdp.access_tile<64xindex>
    ktdp.store %maxes, %t1 : tensor<64xf32>, <64xindex>

    // G2: x - max, the statistic read at lane 0.
    %t2 = ktdp.construct_access_tile %x_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %x1 = ktdp.load %t2 : <64x64xindex> -> tensor<64x64xf32>
    %t3 = ktdp.construct_access_tile %max_r_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set3} : memref<64x32xf32> -> !ktdp.access_tile<64x1xindex>
    %m0 = ktdp.load %t3 : <64x1xindex> -> tensor<64x1xf32>
    %e1 = tensor.empty() : tensor<64x64xf32>
    %shifted = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%x1, %m0 : tensor<64x64xf32>, tensor<64x1xf32>) outs(%e1 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_m: f32, %acc: f32):
      %d = arith.subf %in, %in_m : f32
      linalg.yield %d : f32
    } -> tensor<64x64xf32>
    %t4 = ktdp.construct_access_tile %diff_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %shifted, %t4 : tensor<64x64xf32>, <64x64xindex>

    // G3: exp of that.
    %t5 = ktdp.construct_access_tile %diff_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %d0 = ktdp.load %t5 : <64x64xindex> -> tensor<64x64xf32>
    %e2 = tensor.empty() : tensor<64x64xf32>
    %exps = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%d0 : tensor<64x64xf32>) outs(%e2 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %ex = math.exp %in : f32
      linalg.yield %ex : f32
    } -> tensor<64x64xf32>
    %t6 = ktdp.construct_access_tile %out_view[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %exps, %t6 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}
