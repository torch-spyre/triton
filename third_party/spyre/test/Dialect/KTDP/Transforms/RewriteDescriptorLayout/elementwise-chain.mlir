// RUN: spyre-triton-opt %s --split-input-file --rewrite-descriptor-layout | FileCheck %s

// Three cases exercising Phase 2's forward propagation through an
// elementwise chain, seeded only from values Phase 1 physicalized.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.

// Two annotated inputs feeding an annotated store through arith.addf: the
// multi-tensor-operand case RewriteElementwisePattern's local shape rule
// covers. Both operands physicalize to the same stick shape, addf's result
// is retyped to match, and the store's destination marker lets it
// physicalize too -- so the whole chain stays physical with no bridging
// loop, unlike the unannotated-store case below.
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: tt.func @elementwise_annotated_addf
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       arith.addf {{.*}} : tensor<2x64x64xf32>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK-NOT:   scf.for
// CHECK:       ktdp.store {{.*}} : tensor<2x64x64xf32>, <2x64x64xindex>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @elementwise_annotated_addf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %7 = builtin.unrealized_conversion_cast %6 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %7 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %9 = ktdp.load %8 : <64x128xindex> -> tensor<64x128xf32>
    %10 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %11 = ktdp.construct_memory_view %10, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %12 = builtin.unrealized_conversion_cast %11 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %12 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %13 = arith.addf %4, %9 : tensor<64x128xf32>
    %14 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    ktdp.store %13, %14 : tensor<64x128xf32>, <64x128xindex>
    tt.return
  }
}

// -----

// Annotated input feeding an unannotated store through arith.negf. Phase 2
// decides the store itself, sees there is no destination marker to
// physicalize into, and emits a bridging loop (emitBridgeToLogical) that
// reassembles the logical shape from the physical data tile's stick slices
// before the store.
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: tt.func @elementwise_unannotated_store
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       arith.negf {{.*}} : tensor<2x64x64xf32>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tensor.empty() : tensor<64x128xf32>
// CHECK:       scf.for
// CHECK:         tensor.extract_slice
// CHECK:         tensor.insert_slice
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.store {{.*}} : tensor<64x128xf32>, <64x128xindex>
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @elementwise_unannotated_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %7 = arith.negf %4 : tensor<64x128xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    ktdp.store %7, %8 : tensor<64x128xf32>, <64x128xindex>
    tt.return
  }
}

// -----

// Regression test: RewriteElementwisePattern's local shape rule (single
// RankedTensorType result + tensor operands sharing one shape + a differently
// shaped result) must NOT retype an op that merely satisfies that shape
// pattern by coincidence when it is not reachable from any physicalized
// ktdp.load. tt.expand_dims is exactly such an op: rank-changing, one tensor
// operand, shape mismatch by construction -- but here it sits downstream of a
// tt.reduce over an ANNOTATED load, on a path with no marker of its own, so it
// must be left as a plain logical reshape.
//
// A is annotated (stick-on-K(64), phys [K/64, M, K%64] = [2, 64, 64]) and
// feeds arith.negf + tt.reduce; the physicalized reduce loop only covers that
// part of the chain. Downstream of the reduce -- expand_dims, then a
// collapse_shape/broadcast idiom -- runs on logical tensor<64xf32> shapes the
// whole way to an unannotated store, exercising exactly the reachability gap
// ctx.physicalValues closes: membership in that set stops propagating once
// the reduce's plain (non-elementwise) result is produced, so expand_dims
// downstream of it is simply never a candidate.
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: tt.func @elementwise_expand_dims_unannotated
// CHECK:       ktdp.load {{.*}} : <2x64x64xindex> -> tensor<2x64x64xf32>
// CHECK:       arith.negf {{.*}} : tensor<2x64x64xf32>
// CHECK:       linalg.reduce{{.*}}dimensions = [0, 2]
// CHECK:       tensor.expand_shape %{{.*}} {{.*}} output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
// CHECK:       tensor.collapse_shape %{{.*}} : tensor<64x1xf32> into tensor<64xf32>
// CHECK:       linalg.broadcast ins(%{{.*}} : tensor<64xf32>) outs(%{{.*}} : tensor<64x128xf32>)
// CHECK:       ktdp.store %{{.*}} : tensor<64x128xf32>, <64x128xindex>
// CHECK:       tt.return
  tt.func @elementwise_expand_dims_unannotated(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
    %5 = arith.negf %4 : tensor<64x128xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %6 = tensor.empty() : tensor<64xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce ins(%5 : tensor<64x128xf32>) outs(%7 : tensor<64xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %collapsed = tensor.collapse_shape %expanded [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %9 = tensor.empty() : tensor<64x128xf32>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%9 : tensor<64x128xf32>) dimensions = [1] 
    %10 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %11 = ktdp.construct_memory_view %10, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %12 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    ktdp.store %broadcasted, %12 : tensor<64x128xf32>, <64x128xindex>
    tt.return
  }
}
