// RUN: spyre-triton-opt %s --split-input-file --rewrite-descriptor-layout-generic > %t.once
// RUN: spyre-triton-opt %t.once --split-input-file --rewrite-descriptor-layout-generic > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

// The two ways this pass does nothing, and they are the same way.
//
// The markers are the entire scope of the rewrite: Phase 1 physicalizes one
// descriptor per marker and records the view it produced, and Phase 2 looks only
// at generics adjacent to a recorded view. No marker therefore means no root,
// nothing seeded as physical, and every op found consistent -- so an unannotated
// kernel is a no-op rather than something the pass has an opinion about, and a
// SECOND run is the same no-op, because Phase 3 erased the markers the first run
// consumed.
//
// Idempotence here is structural, not a guard: it does not depend on how Phase
// 2's driver behaves. (Within a single run the property is separately
// guaranteed, by the consistency guard being exactly what the rewrite
// establishes -- an op the rewrite has finished satisfies the guard and is not
// re-fired.)
//
// The diff on the third RUN line is the idempotence assertion, and it covers
// BOTH modules below. FileCheck then reads the first run's output, so the
// positive checks say what that no-op and that one rewrite produced.
//
// Captures are hand-named and this file is hand-maintained: do not regenerate it
// with generate-test-checks.py, which would rewrite the RUN block into a single
// pipe and lose the diff.

// Case 1 -- no marker anywhere, so nothing to physicalize.
//
// Every shape below is the logical one it arrived as.

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: tt.func @no_marker
// CHECK:         ktdp.construct_memory_view %{{.*}}, sizes: [64, 128], strides: [128, 1]
// CHECK-SAME:      memref<64x128xf32>
// CHECK:         ktdp.construct_access_tile %{{.*}} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
// CHECK:         ktdp.load %{{.*}} : <64x128xindex> -> tensor<64x128xf32>
// CHECK:         linalg.generic {indexing_maps = [#[[MAP:.*]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : tensor<64x128xf32>) outs(%{{.*}} : tensor<64x128xf32>)
// CHECK:         ktdp.store %{{.*}} : tensor<64x128xf32>, <64x128xindex>
tt.func @no_marker(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %i = builtin.unrealized_conversion_cast %in : !tt.ptr<f32> to index
  %iv = ktdp.construct_memory_view %i, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %it = ktdp.construct_access_tile %iv[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %a = ktdp.load %it : <64x128xindex> -> tensor<64x128xf32>
  %o = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %o, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 2 -- markers, so a real rewrite, and running it again changes nothing.
//
// The reduced dim is the unsplit one (logical dim 0), and the surviving dim is
// split on both operands -- so its two loops appear in both maps and neither map
// carries arithmetic. That is the shape the diff above is taken over: a module
// the pass really did rewrite, not a module it declined.

// CHECK: #[[$IDEM_ID3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$IDEM_ID2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$IDEM_OUT:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$IDEM_SET3:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$IDEM_SET2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

#in  = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>
#sin  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#sout = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   tt.func @idempotent(
// CHECK-SAME:  %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_4:.*]] = ktdp.construct_memory_view %[[VAL_3]], sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$IDEM_SET3]], memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.divsi %[[VAL_2]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_8:.*]] = arith.remsi %[[VAL_2]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = ktdp.construct_access_tile %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_2]], %[[VAL_8]]] {access_tile_order = #[[$IDEM_ID3]], access_tile_set = #[[$IDEM_SET3]]} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
// CHECK:           %[[VAL_10:.*]] = ktdp.load %[[VAL_9]] : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<f32> to index
// CHECK:           %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_11]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$IDEM_SET2]], memory_space = #ktdp.memory_space<global>} : memref<2x64xf32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_14:.*]] = arith.divsi %[[VAL_2]], %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_2]], %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]] = ktdp.construct_access_tile %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_16]]] {access_tile_order = #[[$IDEM_ID2]], access_tile_set = #[[$IDEM_SET2]]} : memref<2x64xf32> -> !ktdp.access_tile<2x64xindex>
// CHECK:           %[[VAL_18:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_19:.*]] = tensor.empty() : tensor<2x64xf32>
// CHECK:           %[[VAL_20:.*]] = linalg.fill ins(%[[VAL_18]] : f32) outs(%[[VAL_19]] : tensor<2x64xf32>) -> tensor<2x64xf32>
// CHECK:           %[[VAL_21:.*]] = linalg.generic {indexing_maps = [#[[$IDEM_ID3]], #[[$IDEM_OUT]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[VAL_10]] : tensor<2x64x64xf32>) outs(%[[VAL_20]] : tensor<2x64xf32>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: f32):
// CHECK:             %[[VAL_24:.*]] = arith.addf %[[VAL_22]], %[[VAL_23]] : f32
// CHECK:             linalg.yield %[[VAL_24]] : f32
// CHECK:           } -> tensor<2x64xf32>
// CHECK:           ktdp.store %[[VAL_21]], %[[VAL_17]] : tensor<2x64xf32>, <2x64xindex>
// CHECK:           tt.return
// CHECK:         }
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
  %zero = arith.constant 0.000000e+00 : f32
  %e0 = tensor.empty() : tensor<128xf32>
  %e = linalg.fill ins(%zero : f32) outs(%e0 : tensor<128xf32>) -> tensor<128xf32>
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["reduction", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<128xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %s = arith.addf %x, %acc : f32
    linalg.yield %s : f32
  } -> tensor<128xf32>
  ktdp.store %r, %ot : tensor<128xf32>, <128xindex>
  tt.return
}
}
