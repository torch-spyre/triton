// RUN: spyre-triton-opt %s --rewrite-descriptor-layout > %t.once
// RUN: spyre-triton-opt %t.once --rewrite-descriptor-layout > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

// Phase 2 runs under a greedy driver that re-enqueues an op whenever a
// neighbour it feeds or consumes is rewritten, so ops are visited repeatedly
// until a fixpoint. Every pattern's match condition must be falsified by its own
// rewrite. Running the pass on its own output must therefore be a no-op, which
// the diff above asserts.
//
// The reduce here retypes in place rather than changing rank, so it is the case
// that does not falsify its precondition for free: its guard is what this pins.

// CHECK-LABEL: tt.func @idempotent_reduce
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-SAME:  dimensions = [0, 2]
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
  tt.func @idempotent_reduce(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    // A[64, 128] stick-on-N(64): phys [N/64, M, N%64] = [2, 64, 64]
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %7 = tensor.empty() : tensor<64xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce ins(%4 : tensor<64x128xf32>) outs(%8 : tensor<64xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %9 = arith.addf %in, %init : f32
        linalg.yield %9 : f32
      }
    %10 = ktdp.construct_access_tile %6[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf32> -> !ktdp.access_tile<64xindex>
    ktdp.store %reduced, %10 : tensor<64xf32>, <64xindex>
    tt.return
  }
}
