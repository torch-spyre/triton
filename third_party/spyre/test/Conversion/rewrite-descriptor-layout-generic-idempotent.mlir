// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic > %t.once
// RUN: spyre-triton-opt %t.once --rewrite-descriptor-layout-generic > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

// Running the pass on its own output must change nothing, and the diff above is
// the assertion.
//
// This holds for a structural reason rather than a guard: Phase 3 erases the
// markers, so a second run finds no marker, hence no root, hence nothing to
// physicalize. It is independent of how Phase 2's driver behaves.
//
// Within a single run the same property is separately guaranteed, by the
// consistency guard being exactly what the rewrite establishes -- an op the
// rewrite has finished satisfies the guard and is not re-fired.

// CHECK-LABEL: tt.func @idempotent
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load %{{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// The reduced dim is the unsplit one (logical dim 0), and the surviving dim is
// split on both operands -- so its two loops appear in both maps and neither map
// carries arithmetic.
// CHECK:       linalg.generic {indexing_maps = [#[[IN:.*]], #[[OUT:.*]]], iterator_types = ["reduction", "parallel", "parallel"]}
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.store %{{.*}} : tensor<2x64xf32>, <2x64xindex>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return

#in  = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>
#sin  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#sout = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>
module {
tt.func @idempotent(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sin} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [128], strides: [1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>} : memref<128xf32>
  %od = builtin.unrealized_conversion_cast %ov : memref<128xf32> to !tt.tensordesc<128xf32>
  tt.spyre_tensor_layout %od {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : <128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0] {access_tile_order = #id1, access_tile_set = #sout} : memref<128xf32> -> !ktdp.access_tile<128xindex>
  %e = tensor.empty() : tensor<128xf32>
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["reduction", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<128xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %s = arith.addf %x, %acc : f32
    linalg.yield %s : f32
  } -> tensor<128xf32>
  ktdp.store %r, %ot : tensor<128xf32>, <128xindex>
  tt.return
}
}
