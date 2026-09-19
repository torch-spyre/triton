// RUN: spyre-triton-opt %s --rewrite-descriptor-layout -split-input-file -verify-diagnostics

// The three layouts RewriteDescriptorLayout declines, each on post-lowering
// KTIR: the descriptor is a bridge cast off the memory view, the marker rides
// that cast, and the access is the tile the pass rebuilds. The diagnostic is
// therefore anchored on the tile or on the compute op, not on a tt.descriptor_*.

// Test 1: block extent of stick dim is smaller than the stick size.
// The descriptor is 64x32 but phys_arg (stick_size) is 64 on dim 1,
// meaning the load's block extent on dim 1 (32) < stick_size (64).
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
  tt.func @block_smaller_than_stick(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x32xf32> to !tt.tensordesc<64x32xf32>
    // Stick-on-N with stick_size=64, but N=32 < 64
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x32xf32>
    // expected-error @below {{spyre_tensor_layout: block extent of stick dim (32) is smaller than the stick size (64); a stick dim cannot be sub-stick}}
    %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x32xf32> -> !ktdp.access_tile<64x32xindex>
    %4 = ktdp.load %3 : <64x32xindex> -> tensor<64x32xf32>
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %6 = ktdp.construct_memory_view %5, sizes: [64, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
    %7 = ktdp.construct_access_tile %6[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x32xf32> -> !ktdp.access_tile<64x32xindex>
    ktdp.store %4, %7 : tensor<64x32xf32>, <64x32xindex>
    tt.return
  }
}

// -----

// Test 2: stick-splitting the indirect (gather) row dim is not supported.
// The layout annotation applies floordiv to the indirect access tile's gathered
// dimension (dim 0).
#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
  tt.func @gather_stick_split_indirect_dim(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index

    // Index tensor: 32 row indices.
    %0 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<i32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [32], strides: [1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<32xi32>

    // Data: [512, 128] with stick-on-M layout (stick=64 on dim 0).
    //   phys_src=[0, 1, 0] phys_op=[1, 0, 2] phys_arg=[64, 0, 64]
    //   This applies floordiv to dim 0, which is the indirect (gather) dim.
    %2 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %3 = ktdp.construct_memory_view %2, sizes: [512, 128], strides: [128, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<512x128xf32>
    %4 = builtin.unrealized_conversion_cast %3 : memref<512x128xf32> to !tt.tensordesc<512x128xf32>
    tt.spyre_tensor_layout %4 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 0, 1, 0>} : <512x128xf32>

    // Gather 32 non-contiguous rows — error: stick-splitting the indirect row dim.
    // expected-error @below {{spyre_tensor_layout: stick-splitting the indirect (gather) row dim is not supported}}
    %5 = ktdp.construct_indirect_access_tile intermediate_variables(%arg3, %arg4) %3[ind(%1[%c0 + %arg3]), (%c0 + %arg4)] {variables_space_order = #map, variables_space_set = #set2} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
    %6 = ktdp.load %5 : <32x128xindex> -> tensor<32x128xf32>

    %7 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %8 = ktdp.construct_memory_view %7, sizes: [32, 128], strides: [128, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<32x128xf32>
    %9 = ktdp.construct_access_tile %8[%c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<32x128xf32> -> !ktdp.access_tile<32x128xindex>
    ktdp.store %6, %9 : tensor<32x128xf32>, <32x128xindex>
    tt.return
  }
}

// -----

// Test 3: two annotated operands whose parallel floor dims land on *different*
// output axes. A[B,M,K] is split on M and K, B[B,K,N] on K and N, so A wants a
// parallel scatter loop on the accumulator's M axis and B wants one on N. That
// needs two independent nested scatter loops (a genuine 2-D output tiling);
// emitSourceStage carries a single (factor, axis) pair, so it rejects instead of
// silently scattering only one axis.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 127 >= 0)>
module {
  tt.func @parallel_scatter_axis_disagreement(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f16> to index
    %1 = ktdp.construct_memory_view %0, sizes: [2, 128, 128], strides: [16384, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x128x128xf16>
    %2 = builtin.unrealized_conversion_cast %1 : memref<2x128x128xf16> to !tt.tensordesc<2x128x128xf16>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 64, 0, 64, 64>, phys_op = array<i64: 1, 1, 0, 2, 2>, phys_src = array<i64: 1, 2, 0, 1, 2>} : <2x128x128xf16>
    %3 = ktdp.construct_access_tile %1[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x128x128xf16> -> !ktdp.access_tile<2x128x128xindex>
    %4 = ktdp.load %3 : <2x128x128xindex> -> tensor<2x128x128xf16>
    %5 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f16> to index
    %6 = ktdp.construct_memory_view %5, sizes: [2, 128, 128], strides: [16384, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x128x128xf16>
    %7 = builtin.unrealized_conversion_cast %6 : memref<2x128x128xf16> to !tt.tensordesc<2x128x128xf16>
    tt.spyre_tensor_layout %7 {phys_arg = array<i64: 64, 64, 0, 64, 64>, phys_op = array<i64: 1, 1, 0, 2, 2>, phys_src = array<i64: 1, 2, 0, 1, 2>} : <2x128x128xf16>
    %8 = ktdp.construct_access_tile %6[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x128x128xf16> -> !ktdp.access_tile<2x128x128xindex>
    %9 = ktdp.load %8 : <2x128x128xindex> -> tensor<2x128x128xf16>
    %10 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to index
    %11 = ktdp.construct_memory_view %10, sizes: [2, 128, 128], strides: [16384, 128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x128x128xf16>
    %12 = ktdp.construct_access_tile %11[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x128x128xf16> -> !ktdp.access_tile<2x128x128xindex>
    %13 = ktdp.load %12 : <2x128x128xindex> -> tensor<2x128x128xf16>
    // expected-error @below {{spyre_tensor_layout: operands disagree on the parallel multi-stick scatter}}
    %14 = linalg.batch_matmul ins(%4, %9 : tensor<2x128x128xf16>, tensor<2x128x128xf16>) outs(%13 : tensor<2x128x128xf16>) -> tensor<2x128x128xf16>
    %15 = ktdp.construct_access_tile %11[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x128x128xf16> -> !ktdp.access_tile<2x128x128xindex>
    ktdp.store %14, %15 : tensor<2x128x128xf16>, <2x128x128xindex>
    tt.return
  }
}
