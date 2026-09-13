// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics

// Hand-crafted post-LowerDescriptorMemory KTIR, fed straight to the pass. These
// diagnostics are unreachable from Triton-level input: the earlier passes
// normalize the shapes, orders and sets involved, so only a module written at
// this level can present them.
//
// Several of the marker's own fields are checked twice over -- once by
// tt.spyre_tensor_layout's verifier and once by the pass -- and the verifier
// runs first, so those pass-side checks are not reachable at all. Where a
// diagnostic below looks like a verifier check, it is reachable precisely
// because the pass measures against something the verifier cannot see (the
// memory view's rank rather than the descriptor's) or checks something the
// verifier does not (a lone half of a split).

// Case 1 -- the marker's desc is not a lowered descriptor.
module {
tt.func @desc_not_lowered(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{spyre_tensor_layout: desc operand is not a lowered descriptor}}
  tt.spyre_tensor_layout %desc {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
  tt.return
}
}

// -----

// Case 2 -- the bridge cast is there but has no memory view behind it.
//
// Everything the pass rewrites is read off construct_memory_view, so the cast
// alone is not enough to work from.
module {
tt.func @no_construct_memory_view(%memview: memref<64x64xf32>) {
  %cast = builtin.unrealized_conversion_cast %memview : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
  // expected-error @below {{spyre_tensor_layout: cannot locate construct_memory_view behind the bridge cast}}
  tt.spyre_tensor_layout %cast {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
  tt.return
}
}

// -----

// Case 3 -- phys_src out of range.
//
// The pass indexes the memory view's size and stride arrays with phys_src, so
// it validates against the VIEW's rank. The op verifier validates against the
// descriptor's block rank. A rank-3 descriptor over a rank-2 view satisfies the
// verifier and still leaves phys_src[0] = 2 unusable here.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @phys_src_out_of_range(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<1x64x128xf32>
  // expected-error @below {{spyre_tensor_layout: phys_src out of range for logical rank 2}}
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 2, 0, 2>} : <1x64x128xf32>
  tt.return
}
}

// -----

// Case 4 -- a floordiv half with no matching mod half.
//
// A split names its logical dim twice. The op verifier only rejects a REPEATED
// logical dim that is not a well-formed split, so a lone half passes it; the
// pass rejects it because the map it would build cannot say where the dim's
// elements sit within a stick.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @lone_floordiv_half(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  // expected-error @below {{spyre_tensor_layout: logical dim 1 has a floordiv physical dim without the matching mod half}}
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0>, phys_op = array<i64: 1, 0>, phys_src = array<i64: 1, 0>} : <64x128xf32>
  tt.return
}
}

// -----

// Case 5 -- a mod half with no matching floordiv half.
//
// The mirror of case 4, and a separate case because the message names which
// half is missing.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @lone_mod_half(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  // expected-error @below {{spyre_tensor_layout: logical dim 1 has a mod physical dim without the matching floordiv half}}
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0>, phys_op = array<i64: 2, 0>, phys_src = array<i64: 1, 0>} : <64x128xf32>
  tt.return
}
}

// -----

// Case 6 -- a partitioned coordinate_set on the memory view.
//
// The physical view's set is recomputed as the dense range of its own physical
// sizes, so a set saying anything more than the dense range of the logical ones
// -- which is what the attribute exists to express -- would be silently
// discarded. Note this fires on the VIEW, not the marker.
module {
tt.func @partitioned_coordinate_set(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  // expected-error @below {{spyre_tensor_layout: coordinate_set must be the dense range of the view's sizes to physicalize it; a partitioned set would be overwritten}}
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
  tt.return
}
}

// -----

// Case 7 -- an access tile whose block shape has no static physical form.
//
// The block extent is a runtime symbol, so no physical extent can be stated in
// the rebuilt tile's type.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#dynset = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @dynamic_block_shape(%arg0: !tt.ptr<f32>, %n: index) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
  // expected-error @below {{spyre_tensor_layout: cannot derive a static physical block shape for this access tile}}
  %3 = ktdp.construct_access_tile %1[%c0, %c0] symbols(%n) {access_tile_order = #id, access_tile_set = #dynset} : memref<64x128xf32> -> !ktdp.access_tile<?x128xindex>
  %4 = ktdp.load %3 : !ktdp.access_tile<?x128xindex> -> tensor<?x128xf32>
  tt.return
}
}

// -----

// Case 8 -- a sub-stick block on the split dim.
//
// The same rejection as case 1 of rewrite-descriptor-layout-generic-invalid.mlir,
// reached here on a tile whose block is narrower than the view it reads, which
// the earlier passes never produce.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
tt.func @sub_stick_block(%arg0: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x32xf32> to !tt.tensordesc<64x32xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x32xf32>
  // expected-error @below {{spyre_tensor_layout: block extent of stick dim (32) is smaller than the stick size (64); a stick dim cannot be sub-stick}}
  %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<64x32xf32> -> !ktdp.access_tile<64x32xindex>
  %4 = ktdp.load %3 : <64x32xindex> -> tensor<64x32xf32>
  tt.return
}
}

// -----

// Case 9 -- a permuted access_tile_order.
//
// The rebuilt tile states its order over the PHYSICAL dims and so recomputes it
// as the identity. A permutation on the input says something about the logical
// dims that the recompute would drop.
#perm = affine_map<(d0, d1) -> (d1, d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @permuted_access_tile_order(%arg0: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
  // expected-error @below {{spyre_tensor_layout: access_tile_order must be the identity to physicalize this tile; a permuted order would be overwritten}}
  %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #perm, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
  tt.return
}
}

// -----

// Case 10 -- a non-dense access_tile_set.
//
// Same reason as case 9 for the set rather than the order: it is recomputed as
// the dense range of the physical block, so a strided set on the input would be
// lost. This one is a stride-2 subset of the same range.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#strided = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0, d1 mod 2 == 0)>
module {
tt.func @non_dense_access_tile_set(%arg0: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
  // expected-error @below {{spyre_tensor_layout: access_tile_set must be the dense range of the block shape to physicalize this tile; a non-dense set would be overwritten}}
  %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #id, access_tile_set = #strided} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
  tt.return
}
}

// -----

// Case 11 -- an access tile user that is neither a load nor a store.
//
// Phase 1 re-points loads and stores at the physical tile; it has nothing to
// re-point for anything else, and leaving the user on the erased logical tile
// would be invalid IR.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func private @sink_tile(%t: !ktdp.access_tile<64x128xindex>)
tt.func @unexpected_access_tile_user(%arg0: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %2 = builtin.unrealized_conversion_cast %1 : memref<64x128xf32> to !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %2 {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x128xf32>
  %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  // expected-error @below {{spyre_tensor_layout: unexpected user of an access tile}}
  tt.call @sink_tile(%3) : (!ktdp.access_tile<64x128xindex>) -> ()
  tt.return
}
}

// -----

// Case 12 -- a non-splat constant on a physicalized chain.
//
// The accumulator inherits the store destination's layout, so the constant
// behind it has to be restated at physical shape. A splat is a relabelling and
// is accepted; anything else would need its elements moved into stick order,
// which is a data rewrite this pass does not perform.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>
module {
tt.func @non_splat_accumulator(%a: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x4xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<2x4xf32> to !tt.tensordesc<2x4xf32>
  tt.spyre_tensor_layout %ad {phys_arg = array<i64: 2, 0, 2>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <2x4xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<2x4xf32> -> !ktdp.access_tile<2x4xindex>
  %al = ktdp.load %at : <2x4xindex> -> tensor<2x4xf32>
  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x4xf32>
  %cd = builtin.unrealized_conversion_cast %cv : memref<2x4xf32> to !tt.tensordesc<2x4xf32>
  tt.spyre_tensor_layout %cd {phys_arg = array<i64: 2, 0, 2>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <2x4xf32>
  %ct = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<2x4xf32> -> !ktdp.access_tile<2x4xindex>
  // expected-error @below {{rewrite-descriptor-layout-generic: cannot physicalize a non-splat constant; its elements would have to be reordered into stick layout}}
  %acc = arith.constant dense<[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]> : tensor<2x4xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<2x4xf32>) outs(%acc : tensor<2x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    %z = arith.addf %x, %o : f32
    linalg.yield %z : f32
  } -> tensor<2x4xf32>
  ktdp.store %r, %ct : tensor<2x4xf32>, <2x4xindex>
  tt.return
}
}

// -----

// Case 13 -- a producer the rewrite cannot restate at physical shape.
//
// tensor.empty, a splat constant and linalg.generic are the producers a shape
// change propagates cleanly through. tensor.insert is not one: its result shape
// is not a function this pass can re-derive, so the chain stops at it and says
// so instead of emitting IR that fails later.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>
module {
tt.func @unrestatable_producer(%a: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x4xf32>
  %ad = builtin.unrealized_conversion_cast %av : memref<2x4xf32> to !tt.tensordesc<2x4xf32>
  tt.spyre_tensor_layout %ad {phys_arg = array<i64: 2, 0, 2>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <2x4xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<2x4xf32> -> !ktdp.access_tile<2x4xindex>
  %al = ktdp.load %at : <2x4xindex> -> tensor<2x4xf32>
  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x4xf32>
  %cd = builtin.unrealized_conversion_cast %cv : memref<2x4xf32> to !tt.tensordesc<2x4xf32>
  tt.spyre_tensor_layout %cd {phys_arg = array<i64: 2, 0, 2>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <2x4xf32>
  %ct = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<2x4xf32> -> !ktdp.access_tile<2x4xindex>
  %zero = arith.constant 0.0 : f32
  %sp = arith.constant dense<0.0> : tensor<2x4xf32>
  // expected-error @below {{rewrite-descriptor-layout-generic: this op produces a value on a physicalized chain but the rewrite cannot restate it at physical shape}}
  %acc = tensor.insert %zero into %sp[%c0, %c0] : tensor<2x4xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<2x4xf32>) outs(%acc : tensor<2x4xf32>) {
  ^bb0(%x: f32, %o: f32):
    %z = arith.addf %x, %o : f32
    linalg.yield %z : f32
  } -> tensor<2x4xf32>
  ktdp.store %r, %ct : tensor<2x4xf32>, <2x4xindex>
  tt.return
}
}
