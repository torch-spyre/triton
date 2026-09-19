// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics

// Diagnostics a marker reaches through the shapes the backend's own lowering
// produces: a layout that does not fit the block it annotates, and two layouts
// that disagree about one loop dim. Diagnostics about a malformed marker or a
// malformed KTIR chain live in invalid-ktir.mlir.

// Case 1 -- a stick dim narrower than one stick.
//
// The mod dim's extent IS the stick width, so a block whose logical extent on
// the split dim is below that width would give a physical dim wider than the
// data it indexes.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
tt.func @block_smaller_than_stick(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 32], strides: [32, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<64x32xf32> to !tt.tensordesc<64x32xf32>
  // Stick-on-N at width 64, but N is only 32.
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <64x32xf32>
  // expected-error @below {{spyre_tensor_layout: block extent of stick dim (32) is smaller than the stick size (64); a stick dim cannot be sub-stick}}
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x32xf32> -> !ktdp.access_tile<64x32xindex>
  %al = ktdp.load %at : <64x32xindex> -> tensor<64x32xf32>

  %oi = builtin.unrealized_conversion_cast %out : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 32], strides: [32, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x32xf32> -> !ktdp.access_tile<64x32xindex>
  ktdp.store %al, %ot : tensor<64x32xf32>, <64x32xindex>
  tt.return
}
}

// -----

// Case 2 -- two operands split the same loop dim at different widths.
//
// The rebuilt domain gives a split dim a (stick, elem) pair with ONE width, and
// an operand holding that dim whole addresses it as `stick * width + elem`. Two
// widths on one loop dim leave no such composite, and either choice would
// address the wrong elements of the third operand.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @two_stick_widths_on_one_dim(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : <64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>

  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f32> to index
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %bd = builtin.unrealized_conversion_cast %bv : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  // Same logical dim, same role, width 32 rather than 64.
  tt.spyre_tensor_layout %bd {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>} : <64x128xf32>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %bl = ktdp.load %bt : <64x128xindex> -> tensor<64x128xf32>

  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ct = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>

  %e = tensor.empty() : tensor<64x128xf32>
  // expected-error @below {{loop dim 1 is split at two different stick widths (64 and 32), so no single composite addresses it}}
  %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%al, %bl : tensor<64x128xf32>, tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ct : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}
