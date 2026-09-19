// RUN: spyre-triton-opt %s --rewrite-descriptor-layout -split-input-file | FileCheck %s

// What RewriteDescriptorLayout must leave absent. The first four cases are
// case 1 of the shape check in docs/spyre-tensor-layouts.md: layouts that
// physicalize with no synthesized loop. The last two only assert marker
// erasure, which every case checks.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.
//
// Each case is post-lowering KTIR: the descriptor is a bridge cast off the
// memory view, the marker rides that cast, and the access is a
// construct_access_tile / load / store triple.

// A stick-on-K(64) with K=64, B stick-on-N(64) with N=64: one stick each
// side, nothing to accumulate across.
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @matmul_single_stick
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.matmul
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @matmul_single_stick(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index

    // A[64, 64] stick-on-K(64): phys [1, 64, 64] = [K/64, M, K%64]
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %4 = ktdp.load %3 : <64x64xindex> -> tensor<64x64xf32>

    // B[64, 64] stick-on-N(64): phys [1, 64, 64] = [N/64, K, N%64]
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %7 = builtin.unrealized_conversion_cast %6 : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
    tt.spyre_tensor_layout %7 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %9 = ktdp.load %8 : <64x64xindex> -> tensor<64x64xf32>

    // C[64, 64] -- unannotated accumulator, so no bridge cast and no marker
    %10 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %11 = ktdp.construct_memory_view %10, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %12 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %13 = ktdp.load %12 : <64x64xindex> -> tensor<64x64xf32>

    %14 = linalg.matmul ins(%4, %9 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>

    %15 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %14, %15 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Both operands stick-split on a parallel axis (M for A, N for B), each with
// trip count 1.
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @matmul_both_parallel_trip1
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.matmul
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @matmul_both_parallel_trip1(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index

    // A[M=64, K=128] stick-on-M(64): phys [M/64, K, M%64] = [1, 128, 64]
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 0, 1, 0>} : <64x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>

    // B[K=128, N=64] stick-on-N(64): phys [N/64, K, N%64] = [1, 128, 64]
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [128, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<128x64xf32>
    %7 = builtin.unrealized_conversion_cast %6 : memref<128x64xf32> to !tt.tensordesc<128x64xf32>
    tt.spyre_tensor_layout %7 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x64xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %9 = ktdp.load %8 : <128x64xindex> -> tensor<128x64xf32>

    // C[64, 64] -- unannotated accumulator
    %10 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %11 = ktdp.construct_memory_view %10, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %12 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %13 = ktdp.load %12 : <64x64xindex> -> tensor<64x64xf32>

    %14 = linalg.matmul ins(%4, %9 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>

    %15 = ktdp.construct_access_tile %11[%c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %14, %15 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Reduce with an identity layout (phys_op = [0, 0]): no dim is stick-split.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @reduce_identity_layout
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @reduce_identity_layout(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index

    // A[64, 64] identity layout: phys_src=[0,1] phys_op=[0,0] phys_arg=[0,0]
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 0, 0>, phys_op = array<i64: 0, 0>, phys_src = array<i64: 0, 1>} : <64x64xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %4 = ktdp.load %3 : <64x64xindex> -> tensor<64x64xf32>

    // C[64] -- unannotated output
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf32>

    %cst = arith.constant 0.000000e+00 : f32
    %7 = tensor.empty() : tensor<64xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce ins(%4 : tensor<64x64xf32>) outs(%8 : tensor<64xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %10 = arith.addf %in, %init : f32
        linalg.yield %10 : f32
      }

    %9 = ktdp.construct_access_tile %6[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf32> -> !ktdp.access_tile<64xindex>
    ktdp.store %reduced, %9 : tensor<64xf32>, <64xindex>
    tt.return
  }
}

// -----

// Reduce whose stick-split floor dim has extent 1: A[2,64,64] stick-on-dim2
// with dim2 == 64, so the floor dim exists but its trip count is 1.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @reduce_floor_dim_extent_one
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @reduce_floor_dim_extent_one(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index

    // A[2, 64, 64] stick-on-dim2(64): phys [dim2/64, dim0, dim1, dim2%64] = [1, 2, 64, 64]
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [2, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x64x64xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<2x64x64xf32> to !tt.tensordesc<2x64x64xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 0, 64>, phys_op = array<i64: 1, 0, 0, 2>, phys_src = array<i64: 2, 0, 1, 2>} : <2x64x64xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x64x64xf32> -> !ktdp.access_tile<2x64x64xindex>
    %4 = ktdp.load %3 : <2x64x64xindex> -> tensor<2x64x64xf32>

    // C[2, 64] -- unannotated output
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x64xf32>

    %cst = arith.constant 0.000000e+00 : f32
    %7 = tensor.empty() : tensor<2x64xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<2x64xf32>) -> tensor<2x64xf32>
    %reduced = linalg.reduce ins(%4 : tensor<2x64x64xf32>) outs(%8 : tensor<2x64xf32>) dimensions = [2]
      (%in: f32, %init: f32) {
        %10 = arith.addf %in, %init : f32
        linalg.yield %10 : f32
      }

    %9 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<2x64xf32> -> !ktdp.access_tile<2x64xindex>
    ktdp.store %reduced, %9 : tensor<2x64xf32>, <2x64xindex>
    tt.return
  }
}

// -----

// RewriteDescriptorLayout erases every tt.spyre_tensor_layout marker in its
// Phase 3 -- see docs/spyre-tensor-layouts.md. Each case below carries a
// marker on input, so the CHECK-NOT proves erasure rather than passing on an
// input that never had one.
//
// Positive anchors are kept minimal on purpose: CHECK-NOT only searches the
// gap up to the next positive match, so it is repeated in every gap.

// Elementwise chain with a marker on both input and output descriptors: the
// whole chain is absorbed by the forward retype, so nothing is synthesized.
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: tt.func @pointwise
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       ktdp.load
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @pointwise(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    // [M=128, N=128] stick-on-N with stick_size=64 -> physical [N/64, M, 64] = [2, 128, 64]
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [128, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<128x128xf32> to !tt.tensordesc<128x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
    %4 = ktdp.load %3 : <128x128xindex> -> tensor<128x128xf32>
    // Output with same layout
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [128, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<128x128xf32>
    %7 = builtin.unrealized_conversion_cast %6 : memref<128x128xf32> to !tt.tensordesc<128x128xf32>
    tt.spyre_tensor_layout %7 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <128x128xf32>
    %8 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<128x128xf32> -> !ktdp.access_tile<128x128xindex>
    ktdp.store %4, %8 : tensor<128x128xf32>, <128x128xindex>
    tt.return
  }
}

// -----

// Reduce with markers on both the input and the output descriptor, so erasure
// is proven on the sink side too. This case does emit a loop -- its input is
// stick-split on the reduced axis -- so it asserts marker absence only.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
// CHECK-LABEL: tt.func @reduce_annotated_output
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
  tt.func @reduce_annotated_output(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index

    // A[64, 128] stick-on-N(64): phys [N/64, M, N%64] = [2, 64, 64]
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>

    // C[64] stick-on-M(64): phys [M/64, M%64] = [1, 64]
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf32>
    %7 = builtin.unrealized_conversion_cast %6 : memref<64xf32> to !tt.tensordesc<64xf32>
    tt.spyre_tensor_layout %7 {phys_arg = array<i64: 64, 64>, phys_op = array<i64: 1, 2>, phys_src = array<i64: 0, 0>} : <64xf32>

    %cst = arith.constant 0.000000e+00 : f32
    %8 = tensor.empty() : tensor<64xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce ins(%4 : tensor<64x128xf32>) outs(%9 : tensor<64xf32>) dimensions = [1]
      (%in: f32, %init: f32) {
        %11 = arith.addf %in, %init : f32
        linalg.yield %11 : f32
      }

    %10 = ktdp.construct_access_tile %6[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf32> -> !ktdp.access_tile<64xindex>
    ktdp.store %reduced, %10 : tensor<64xf32>, <64xindex>
    tt.return
  }
}
