// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic | FileCheck %s

// Case 3 -- a batch contraction where K is split into two real sticks on A and
// sits whole at 128 on B. The non-degenerate case, and the one that proves the
// loop elimination.
//
// The named path emits an scf.for over the K sticks plus an arith.muli and a pair
// of extract_slices for this same kernel. None of that appears here.
//
// The truncf between the contraction and the store is spelled as a generic, which
// is what convert_elementwise_to_linalg produces; as a bare arith.truncf on a
// tensor this pass declines it by name, since it rewrites only generics. It is
// also what carries the result to the store's shape: the contraction's outs is an
// unmarked splat constant, so it stays logical, and this generic re-sticks it.
//
// Input produced from rewrite-descriptor-layout-batch-matmul.mlir by lowering to
// pre-pass IR and running --linalg-morph-ops=named-to-generic. CHECK lines are
// hand-written: generate-test-checks.py cannot segment a kernel with a nested
// scf.for.

// CHECK-DAG: #[[ID4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// A splits K, so it names the two K loops (d4, d5) as a plain projected
// permutation.
// CHECK-DAG: #[[A:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d0, d1, d5)>
// B holds K whole, so it is the operand that carries the composite.
// CHECK-DAG: #[[B:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d4 * 64 + d5, d3)>
// The accumulator is an unmarked splat constant, so it stays logical and composes
// the N split it does not carry.
// CHECK-DAG: #[[ACC:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 64 + d3)>
// CHECK-DAG: #[[RESIN:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 64 + d3)>
// CHECK-DAG: #[[RESOUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1, d3)>

// CHECK-LABEL: tt.func public @bmm_matmul_kernel
// K=128 at stick 64 gives two sticks on A; N=64 gives one on B and on the output.
// CHECK:         ktdp.construct_memory_view %{{.*}}, sizes: [2, 4, 64, 64], strides: [16384, 4096, 64, 1]
// CHECK-SAME:      memref<2x4x64x64xf16>
// CHECK:         ktdp.construct_memory_view %{{.*}}, sizes: [1, 4, 128, 64], strides: [32768, 8192, 64, 1]
// CHECK-SAME:      memref<1x4x128x64xf16>
// CHECK:         ktdp.construct_memory_view %{{.*}}, sizes: [1, 4, 64, 64], strides: [16384, 4096, 64, 1]
// CHECK-SAME:      memref<1x4x64x64xf16>
// The kernel's own program-id loop is left exactly as it is: this pass adds no
// loop and rescales none.
// CHECK:         scf.for
// CHECK:           ktdp.load %{{.*}} : <2x4x64x64xindex> -> tensor<2x4x64x64xf16>
// CHECK:           ktdp.load %{{.*}} : <1x4x128x64xindex> -> tensor<1x4x128x64xf16>
// Two reduction loops for the split K; batch, M and the N split are parallel.
// CHECK:           linalg.generic {indexing_maps = [#[[A]], #[[B]], #[[ACC]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%{{.*}}, %{{.*}} : tensor<2x4x64x64xf16>, tensor<1x4x128x64xf16>) outs(%{{.*}} : tensor<4x64x64xf32>)
// CHECK:           tensor.empty() : tensor<1x4x64x64xf16>
// CHECK:           linalg.generic {indexing_maps = [#[[RESIN]], #[[RESOUT]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%{{.*}} : tensor<4x64x64xf32>) outs(%{{.*}} : tensor<1x4x64x64xf16>)
// The store's data tile agrees with its access tile with no widening stage.
// CHECK:           ktdp.store %{{.*}}, %{{.*}} : tensor<1x4x64x64xf16>, <1x4x64x64xindex>
// No second loop, no slicing, and no marker survives.
// CHECK-NOT:     tensor.extract_slice
// CHECK-NOT:     tensor.insert_slice
// CHECK-NOT:     tt.spyre_tensor_layout

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 127 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  tt.func public @bmm_matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64x64xf32>
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.divsi %1, %1 : i32
    %3 = arith.muli %0, %2 : i32
    %4 = arith.addi %3, %2 : i32
    %5 = arith.minsi %4, %c1_i32 : i32
    %6 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f16> to index
    %7 = ktdp.construct_memory_view %6, sizes: [4, 64, 128], strides: [8192, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x64x128xf16>
    %8 = builtin.unrealized_conversion_cast %7 : memref<4x64x128xf16> to !tt.tensordesc<4x64x128xf16>
    %9 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f16> to index
    %10 = ktdp.construct_memory_view %9, sizes: [4, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<4x128x64xf16>
    %11 = builtin.unrealized_conversion_cast %10 : memref<4x128x64xf16> to !tt.tensordesc<4x128x64xf16>
    %12 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to index
    %13 = ktdp.construct_memory_view %12, sizes: [4, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<4x64x64xf16>
    %14 = builtin.unrealized_conversion_cast %13 : memref<4x64x64xf16> to !tt.tensordesc<4x64x64xf16>
    tt.spyre_tensor_layout %8 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x64x128xf16>
    tt.spyre_tensor_layout %11 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x128x64xf16>
    tt.spyre_tensor_layout %14 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <4x64x64xf16>
    scf.for %arg3 = %3 to %5 step %c1_i32  : i32 {
      %15 = arith.muli %arg3, %c4_i32 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = ktdp.construct_access_tile %7[%16, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x64x128xf16> -> !ktdp.access_tile<4x64x128xindex>
      %18 = ktdp.load %17 : <4x64x128xindex> -> tensor<4x64x128xf16>
      %19 = arith.index_cast %15 : i32 to index
      %20 = ktdp.construct_access_tile %10[%19, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<4x128x64xf16> -> !ktdp.access_tile<4x128x64xindex>
      %21 = ktdp.load %20 : <4x128x64xindex> -> tensor<4x128x64xf16>
      %22 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%18, %21 : tensor<4x64x128xf16>, tensor<4x128x64xf16>) outs(%cst : tensor<4x64x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %out: f32):
        %26 = arith.extf %in : f16 to f32
        %27 = arith.extf %in_0 : f16 to f32
        %28 = arith.mulf %26, %27 : f32
        %29 = arith.addf %out, %28 : f32
        linalg.yield %29 : f32
      } -> tensor<4x64x64xf32>
      %e = tensor.empty() : tensor<4x64x64xf16>
      %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<4x64x64xf32>) outs(%e : tensor<4x64x64xf16>) {
      ^bb0(%in: f32, %out: f16):
        %t = arith.truncf %in : f32 to f16
        linalg.yield %t : f16
      } -> tensor<4x64x64xf16>
      %24 = arith.index_cast %15 : i32 to index
      %25 = ktdp.construct_access_tile %13[%24, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x64x64xf16> -> !ktdp.access_tile<4x64x64xindex>
      ktdp.store %23, %25 : tensor<4x64x64xf16>, <4x64x64xindex>
    }
    tt.return
  }
}
