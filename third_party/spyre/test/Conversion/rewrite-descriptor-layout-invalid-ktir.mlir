// RUN: spyre-triton-opt %s --rewrite-descriptor-layout -split-input-file -verify-diagnostics

// These tests feed hand-crafted post-LowerDescriptorMemory / post-LowerComputeOps
// KTIR directly to --rewrite-descriptor-layout, bypassing earlier passes.
// This lets us trigger errors that are unreachable when the full pipeline
// normalizes inputs before they reach RewriteDescriptorLayout.

// -----

// Test 1: desc operand is not a lowered descriptor.
// The tt.spyre_tensor_layout's desc operand is a block argument (not a
// builtin.unrealized_conversion_cast from a memref), so the pass rejects it.
module {
tt.func @desc_not_lowered(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{spyre_tensor_layout: desc operand is not a lowered descriptor}}
  tt.spyre_tensor_layout %desc {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
  tt.return
}
}

// -----

// Test 2: unexpected user of access tile.
// The access tile's only user is a func.call (not ktdp.load or ktdp.store),
// which the pass does not know how to physicalize.
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func private @sink(%t: !ktdp.access_tile<64x64xindex>)
tt.func @unexpected_access_tile_user(%arg0: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
  %3 = arith.index_cast %c0_i32 : i32 to index
  %4 = arith.index_cast %c0_i32 : i32 to index
  %5 = ktdp.construct_access_tile %1[%3, %4] {access_tile_order = #map2, access_tile_set = #set2} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
  // expected-error @below {{spyre_tensor_layout: unexpected user of access tile}}
  tt.call @sink(%5) : (!ktdp.access_tile<64x64xindex>) -> ()
  tt.return
}
}

// -----

// Test 3: cannot locate construct_memory_view behind cast.
// The desc operand IS a bridge cast from a memref, but the memref is a block
// argument (not from ktdp.construct_memory_view).
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func @no_construct_memory_view(%memview: memref<64x64xf32>) {
  %cast = builtin.unrealized_conversion_cast %memview : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
  // expected-error @below {{spyre_tensor_layout: cannot locate construct_memory_view behind cast}}
  tt.spyre_tensor_layout %cast {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
  tt.return
}
}

// -----

// Test 4: phys_src out of range (memview path).
// The marker's phys_src[0] references logical dim 2, which is in range for the
// rank-3 descriptor the op verifier sees but out of range for the rank-2
// memview the pass reads sizes/strides from.
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#set4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func @phys_src_out_of_range(%arg0: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0b_i32 = arith.constant 0 : i32
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x64xf32> to !tt.tensordesc<1x64x64xf32>
  // expected-error @below {{spyre_tensor_layout: phys_src out of range}}
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 2, 0, 1>} : <1x64x64xf32>
  %3 = arith.index_cast %c0_i32 : i32 to index
  %4 = arith.index_cast %c0b_i32 : i32 to index
  %5 = ktdp.construct_access_tile %1[%3, %4] {access_tile_order = #map4, access_tile_set = #set4} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
  %6 = ktdp.load %5 : <64x64xindex> -> tensor<64x64xf32>
  tt.return
}
}

// -----

// Test 5: cannot derive static block_shape.
// The access tile has a dynamic dimension (?), so applyCoordMap cannot compute
// a static physical block shape.
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#set5 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func @cannot_derive_static_block_shape(%arg0: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0b_i32 = arith.constant 0 : i32
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set5, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
  %3 = arith.index_cast %c0_i32 : i32 to index
  %4 = arith.index_cast %c0b_i32 : i32 to index
  // expected-error @below {{spyre_tensor_layout: cannot derive static block_shape}}
  %5 = ktdp.construct_access_tile %1[%3, %4] {access_tile_order = #map5, access_tile_set = #set5} : memref<64x64xf32> -> !ktdp.access_tile<?x64xindex>
  %6 = ktdp.load %5 : !ktdp.access_tile<?x64xindex> -> tensor<?x64xf32>
  tt.return
}
}

// -----

// Test 6: block extent of stick dim is smaller than the stick size.
// The access tile's logical block for the stick source dim (32) is less than
// the stick size (64), which is invalid.
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#set6 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
tt.func @stick_dim_too_small(%arg0: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0b_i32 = arith.constant 0 : i32
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 32], strides: [32, 1] {coordinate_set = #set6, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x32xf32> to !tt.tensordesc<64x32xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x32xf32>
  %3 = arith.index_cast %c0_i32 : i32 to index
  %4 = arith.index_cast %c0b_i32 : i32 to index
  // expected-error @below {{spyre_tensor_layout: block extent of stick dim (32) is smaller than the stick size (64); a stick dim cannot be sub-stick}}
  %5 = ktdp.construct_access_tile %1[%3, %4] {access_tile_order = #map6, access_tile_set = #set6} : memref<64x32xf32> -> !ktdp.access_tile<64x32xindex>
  %6 = ktdp.load %5 : <64x32xindex> -> tensor<64x32xf32>
  tt.return
}
}

// -----

// Test 7: physicalizing an indirect access tile with rank != 2.
// The indirect tile has logical rank 3 (3 intermediate variables), which is
// not supported — only rank-2 (gather) indirect tiles can be physicalized.
#set7a = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 15 >= 0)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set7b = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
module {
tt.func @indirect_rank3(%arg0: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [32, 64, 16], strides: [1024, 16, 1] {coordinate_set = #set7a, memory_space = #ktdp.memory_space<global>} : memref<32x64x16xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<32x64x16xf32> to !tt.tensordesc<32x64x16xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 0, 0, 0>, phys_op = array<i64: 0, 0, 0>, phys_src = array<i64: 0, 1, 2>} : <32x64x16xf32>
  %3 = builtin.unrealized_conversion_cast %idx_ptr : !tt.ptr<i32> to index
  %4 = ktdp.construct_memory_view %3, sizes: [32], strides: [1] {coordinate_set = #set7b, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  // expected-error @below {{physicalizing an indirect access tile is only supported for a rank-2 gather (got logical rank 3)}}
  %5 = ktdp.construct_indirect_access_tile intermediate_variables(%m, %n, %p) %1[ind(%4[%m]), (%c0 + %n), (%c0 + %p)] {variables_space_order = #map7, variables_space_set = #set7a} : memref<32x64x16xf32>, memref<32xi32> -> !ktdp.access_tile<32x64x16xindex>
  %6 = ktdp.load %5 : !ktdp.access_tile<32x64x16xindex> -> tensor<32x64x16xf32>
  tt.return
}
}

// -----

// Test 8: physicalizing an indirect access tile assumes dim 0 is indirect.
// The indirect tile has dim 0 direct and dim 1 indirect, which violates the
// assumption that dim 0 must be the gather (indirect) dimension.
#set8a = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
#set8b = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
module {
tt.func @indirect_dim1_is_indirect(%arg0: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 32], strides: [32, 1] {coordinate_set = #set8a, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x32xf32> to !tt.tensordesc<64x32xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 0, 0>, phys_op = array<i64: 0, 0>, phys_src = array<i64: 0, 1>} : <64x32xf32>
  %3 = builtin.unrealized_conversion_cast %idx_ptr : !tt.ptr<i32> to index
  %4 = ktdp.construct_memory_view %3, sizes: [32], strides: [1] {coordinate_set = #set8b, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  // expected-error @below {{physicalizing an indirect access tile assumes logical dim 0 is indirect (gather) and logical dim 1 is direct}}
  %5 = ktdp.construct_indirect_access_tile intermediate_variables(%m, %n) %1[(%c0 + %m), ind(%4[%n])] {variables_space_order = #map8, variables_space_set = #set8a} : memref<64x32xf32>, memref<32xi32> -> !ktdp.access_tile<64x32xindex>
  %6 = ktdp.load %5 : !ktdp.access_tile<64x32xindex> -> tensor<64x32xf32>
  tt.return
}
}

// -----

// Test 9: stick-splitting the indirect (gather) row dim is not supported.
// The marker applies a FloorDiv op to the gather source dim (dim 0), which
// would require splitting the indirect row — not supported.
#set9a = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#set9b = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
module {
tt.func @indirect_stick_split_row(%arg0: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [32, 128], strides: [128, 1] {coordinate_set = #set9a, memory_space = #ktdp.memory_space<global>} : memref<32x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<32x128xf32> to !tt.tensordesc<32x128xf32>
  // phys_src=[0, 1, 0] with phys_op=[1, 0, 2]: stick-splits dim 0 (the indirect row dim)
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 0, 1, 0>} : <32x128xf32>
  %3 = builtin.unrealized_conversion_cast %idx_ptr : !tt.ptr<i32> to index
  %4 = ktdp.construct_memory_view %3, sizes: [32], strides: [1] {coordinate_set = #set9b, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  // expected-error @below {{stick-splitting the indirect (gather) row dim is not supported}}
  %5 = ktdp.construct_indirect_access_tile intermediate_variables(%m, %n) %1[ind(%4[%m]), (%c0 + %n)] {variables_space_order = #map9, variables_space_set = #set9a} : memref<32x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
  %6 = ktdp.load %5 : !ktdp.access_tile<32x128xindex> -> tensor<32x128xf32>
  tt.return
}
}

// -----

// Test 10: cannot derive static block_shape for indirect access tile.
// The indirect tile type has a dynamic dimension (?), so applyCoordMap fails.
#set10a = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#map10 = affine_map<(d0, d1) -> (d0, d1)>
#set10b = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
module {
tt.func @indirect_dynamic_block_shape(%arg0: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [512, 128], strides: [128, 1] {coordinate_set = #set10a, memory_space = #ktdp.memory_space<global>} : memref<512x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<512x128xf32> to !tt.tensordesc<512x128xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <512x128xf32>
  %3 = builtin.unrealized_conversion_cast %idx_ptr : !tt.ptr<i32> to index
  %4 = ktdp.construct_memory_view %3, sizes: [32], strides: [1] {coordinate_set = #set10b, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  // expected-error @below {{cannot derive static block_shape for indirect access tile}}
  %5 = ktdp.construct_indirect_access_tile intermediate_variables(%m, %n) %1[ind(%4[%m]), (%c0 + %n)] {variables_space_order = #map10, variables_space_set = #set10a} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<?x128xindex>
  %6 = ktdp.load %5 : !ktdp.access_tile<?x128xindex> -> tensor<?x128xf32>
  tt.return
}
}

// -----

// Test 11: unexpected user of indirect access tile.
// The indirect tile's user is a func.call (not ktdp.load), which the pass
// cannot physicalize.
#set11a = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#map11 = affine_map<(d0, d1) -> (d0, d1)>
#set11b = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
module {
tt.func private @sink(%t: !ktdp.access_tile<32x128xindex>)
tt.func @indirect_unexpected_user(%arg0: !tt.ptr<f32>, %idx_ptr: !tt.ptr<i32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [512, 128], strides: [128, 1] {coordinate_set = #set11a, memory_space = #ktdp.memory_space<global>} : memref<512x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<512x128xf32> to !tt.tensordesc<512x128xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <512x128xf32>
  %3 = builtin.unrealized_conversion_cast %idx_ptr : !tt.ptr<i32> to index
  %4 = ktdp.construct_memory_view %3, sizes: [32], strides: [1] {coordinate_set = #set11b, memory_space = #ktdp.memory_space<global>} : memref<32xi32>
  %5 = ktdp.construct_indirect_access_tile intermediate_variables(%m, %n) %1[ind(%4[%m]), (%c0 + %n)] {variables_space_order = #map11, variables_space_set = #set11a} : memref<512x128xf32>, memref<32xi32> -> !ktdp.access_tile<32x128xindex>
  // expected-error @below {{spyre_tensor_layout: unexpected user of indirect access tile}}
  tt.call @sink(%5) : (!ktdp.access_tile<32x128xindex>) -> ()
  tt.return
}
}

// -----

// Test 12: operands share a stickified contraction axis without annotation.
// A has a multi-stick layout on K (K=128, stick=64 → 2 sticks) while B has no
// layout marker, violating the requirement that shared stickified contraction
// axes must be annotated on all operands.
#map12 = affine_map<(d0, d1) -> (d0, d1)>
#set12a = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set12b = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set12c = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  tt.func @matmul_r6_shared_stickified(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set12a, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
    %3 = arith.index_cast %c0_i32 : i32 to index
    %4 = arith.index_cast %c0_i32 : i32 to index
    %5 = ktdp.construct_access_tile %1[%3, %4] {access_tile_order = #map12, access_tile_set = #set12a} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %6 = ktdp.load %5 : <64x128xindex> -> tensor<64x128xf32>
    %7 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %8 = ktdp.construct_memory_view %7, sizes: [128, 64], strides: [64, 1] {coordinate_set = #set12b, memory_space = #ktdp.memory_space<global>} : memref<128x64xf32>
    %9 = arith.index_cast %c0_i32 : i32 to index
    %10 = arith.index_cast %c0_i32 : i32 to index
    %11 = ktdp.construct_access_tile %8[%9, %10] {access_tile_order = #map12, access_tile_set = #set12b} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %12 = ktdp.load %11 : <128x64xindex> -> tensor<128x64xf32>
    %13 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %14 = ktdp.construct_memory_view %13, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set12c, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %15 = arith.index_cast %c0_i32 : i32 to index
    %16 = arith.index_cast %c0_i32 : i32 to index
    %17 = ktdp.construct_access_tile %14[%15, %16] {access_tile_order = #map12, access_tile_set = #set12c} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %18 = ktdp.load %17 : <64x64xindex> -> tensor<64x64xf32>
    // expected-error @below {{operands share a stickified contraction axis but not all are annotated}}
    %19 = linalg.matmul ins(%6, %12 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%18 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %20 = arith.index_cast %c0_i32 : i32 to index
    %21 = arith.index_cast %c0_i32 : i32 to index
    %22 = ktdp.construct_access_tile %14[%20, %21] {access_tile_order = #map12, access_tile_set = #set12c} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %19, %22 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Test 13: a source op operand is produced by a reshape on a physicalized chain.
// A is annotated (stick-on-K(64), single stick), so its ktdp.load is a Phase-1
// root, but an expand_shape/collapse_shape pair sits between that load and the
// matmul. Such an operand is neither a physical load nor a plain logical
// scratchpad: a reshape's element-to-index mapping matches neither, so there is
// no defined conversion and treating it as either would silently compute the
// wrong slice. This is a legality question, distinct from asking whether the
// operand is physical -- hence a hard error rather than a classification.
//
// The reshape must sit ON a physicalized chain to reach this check. The benign
// case, a reshape downstream of a logical reduce result, is covered by
// @elementwise_expand_dims_unannotated in
// rewrite-descriptor-layout-elementwise-chain.mlir: it is never reachable from
// a Phase-1 root, so the operand decision is never asked about it.
#map13 = affine_map<(d0, d1) -> (d0, d1)>
#set13 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  tt.func @matmul_operand_from_reshape(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set13, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %2 = builtin.unrealized_conversion_cast %1 : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
    tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    %3 = arith.index_cast %c0_i32 : i32 to index
    %4 = arith.index_cast %c0_i32 : i32 to index
    %5 = ktdp.construct_access_tile %1[%3, %4] {access_tile_order = #map13, access_tile_set = #set13} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %6 = ktdp.load %5 : <64x64xindex> -> tensor<64x64xf32>
    %7 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %8 = ktdp.construct_memory_view %7, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set13, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %9 = builtin.unrealized_conversion_cast %8 : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
    tt.spyre_tensor_layout %9 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
    %10 = arith.index_cast %c0_i32 : i32 to index
    %11 = arith.index_cast %c0_i32 : i32 to index
    %12 = ktdp.construct_access_tile %8[%10, %11] {access_tile_order = #map13, access_tile_set = #set13} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %13 = ktdp.load %12 : <64x64xindex> -> tensor<64x64xf32>
    %14 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %15 = ktdp.construct_memory_view %14, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set13, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %16 = arith.index_cast %c0_i32 : i32 to index
    %17 = arith.index_cast %c0_i32 : i32 to index
    %18 = ktdp.construct_access_tile %15[%16, %17] {access_tile_order = #map13, access_tile_set = #set13} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %19 = ktdp.load %18 : <64x64xindex> -> tensor<64x64xf32>
    %20 = tensor.expand_shape %6 [[0], [1, 2]] output_shape [64, 64, 1] : tensor<64x64xf32> into tensor<64x64x1xf32>
    %21 = tensor.collapse_shape %20 [[0], [1, 2]] : tensor<64x64x1xf32> into tensor<64x64xf32>
    // expected-error @below {{spyre_tensor_layout: source op operand is produced by a reshape/broadcast, which cannot be treated as a physical load or a plain logical scratchpad}}
    %22 = linalg.matmul ins(%21, %13 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%19 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %23 = arith.index_cast %c0_i32 : i32 to index
    %24 = arith.index_cast %c0_i32 : i32 to index
    %25 = ktdp.construct_access_tile %15[%23, %24] {access_tile_order = #map13, access_tile_set = #set13} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %22, %25 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}
