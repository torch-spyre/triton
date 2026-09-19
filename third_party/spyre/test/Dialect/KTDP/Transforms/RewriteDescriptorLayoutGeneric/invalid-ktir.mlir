// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics

// A layout the pass cannot consume, or a malformed KTIR chain beneath one it can.
//
// Hand-crafted post-LowerDescriptorMemory KTIR, fed straight to the pass. These
// diagnostics are unreachable from Triton-level input: the earlier passes
// normalize the shapes, orders and sets involved, so only a module written at
// this level can present them. A layout that is well formed and merely
// ill-fitting -- the kind the backend's own lowering does produce -- is in
// invalid-layout.mlir.
//
// Every diagnostic here is the PASS's, and that is now a real division rather
// than a coincidence. The tts dialect's verifier owns the layout's STRUCTURAL
// rules -- parallel array lengths, the phys_src and phys_op ranges, phys_arg
// positivity, the repeated-dim pairings -- and it runs before any pass, so none
// of those is reachable from here at all; they are covered in
// test/Dialect/TTS/tensor-layout-verifier.mlir. What is left below is what the
// pass checks and the verifier deliberately does not: a layout that is a valid
// coordinate map yet leaves this rewrite nothing to build an addressable map
// from (cases 1-3), and the KTIR beneath it (cases 4 onward).

// Case 1 -- a floordiv half with no matching mod half.
//
// A split names its logical dim twice. The verifier only constrains a REPEATED
// logical dim, so a lone half passes it; the pass rejects it because the map it
// would build cannot say where the dim's elements sit within a stick.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @lone_floordiv_half(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  // expected-error @below {{tts.tensor_layout: logical dim 1 has a floordiv physical dim without the matching mod half}}
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0>, phys_op = array<i64: 1, 0>, phys_arg = array<i64: 64, 0>}} : memref<64x128xf32>
  tt.return
}
}

// -----

// Case 2 -- a mod half with no matching floordiv half.
//
// The mirror of case 1, and a separate case because the message names which
// half is missing.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @lone_mod_half(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  // expected-error @below {{tts.tensor_layout: logical dim 1 has a mod physical dim without the matching floordiv half}}
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0>, phys_op = array<i64: 2, 0>, phys_arg = array<i64: 64, 0>}} : memref<64x128xf32>
  tt.return
}
}

// -----

// Case 3 -- a logical dim named by no physical dim.
//
// The layout here splits logical dim 1 and says nothing about logical dim 0, so
// the physical type it prescribes is [2, 64] -- dim 0's extent of 64 is simply
// gone. The verifier permits it: it constrains a logical dim named TWICE and
// says nothing about one named zero times. The pass rejects it because the
// physical layout would no longer address the dim's elements, and because a dim
// no operand's map can name is how a loop dim enters the rebuilt domain unnamed.
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @logical_dim_named_by_nothing(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  // expected-error @below {{tts.tensor_layout: logical dim 0 is named by no phys_src entry, so its extent would be dropped from the physical layout}}
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 1>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>}} : memref<64x128xf32>
  tt.return
}
}

// -----

// Case 4 -- a partitioned coordinate_set on the annotated view.
//
// The physical view's set is recomputed as the dense range of its own physical
// sizes, so a set saying anything more than the dense range of the logical ones
// -- which is what the layout exists to express -- would be silently discarded.
module {
tt.func @partitioned_coordinate_set(%arg0: !tt.ptr<f32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  // expected-error @below {{tts.tensor_layout: coordinate_set must be the dense range of the view's sizes to physicalize it; a partitioned set would be overwritten}}
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  tt.return
}
}

// -----

// Case 5 -- an access tile whose block shape has no static physical form.
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
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  // expected-error @below {{tts.tensor_layout: cannot derive a static physical block shape for this access tile}}
  %3 = ktdp.construct_access_tile %1[%c0, %c0] symbols(%n) {access_tile_order = #id, access_tile_set = #dynset} : memref<64x128xf32> -> !ktdp.access_tile<?x128xindex>
  %4 = ktdp.load %3 : !ktdp.access_tile<?x128xindex> -> tensor<?x128xf32>
  tt.return
}
}

// -----

// Case 6 -- a permuted access_tile_order.
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
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  // expected-error @below {{tts.tensor_layout: access_tile_order must be the identity to physicalize this tile; a permuted order would be overwritten}}
  %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #perm, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
  tt.return
}
}

// -----

// Case 7 -- a non-dense access_tile_set.
//
// Same reason as case 6 for the set rather than the order: it is recomputed as
// the dense range of the physical block, so a strided set on the input would be
// lost. This one is a stride-2 subset of the same range.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#strided = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0, d1 mod 2 == 0)>
module {
tt.func @non_dense_access_tile_set(%arg0: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  // expected-error @below {{tts.tensor_layout: access_tile_set must be the dense range of the block shape to physicalize this tile; a non-dense set would be overwritten}}
  %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #id, access_tile_set = #strided} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %4 = ktdp.load %3 : <64x128xindex> -> tensor<64x128xf32>
  tt.return
}
}

// -----

// Case 8 -- an access tile user that is neither a load nor a store.
//
// physicalizeAccessTile re-points loads and stores at the physical tile; it has
// nothing to re-point for anything else, and leaving the user on the erased
// logical tile would be invalid IR.
#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func private @sink_tile(%t: !ktdp.access_tile<64x128xindex>)
tt.func @unexpected_access_tile_user(%arg0: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
  %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %3 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  // expected-error @below {{tts.tensor_layout: unexpected user of an access tile}}
  tt.call @sink_tile(%3) : (!ktdp.access_tile<64x128xindex>) -> ()
  tt.return
}
}

// -----

// Case 9 -- a non-splat constant on a physicalized chain.
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
  %av = ktdp.construct_memory_view %ai, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 2, 0, 2>}} : memref<2x4xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<2x4xf32> -> !ktdp.access_tile<2x4xindex>
  %al = ktdp.load %at : <2x4xindex> -> tensor<2x4xf32>
  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 2, 0, 2>}} : memref<2x4xf32>
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

// Case 10 -- a producer the rewrite cannot restate at physical shape.
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
  %av = ktdp.construct_memory_view %ai, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 2, 0, 2>}} : memref<2x4xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #set} : memref<2x4xf32> -> !ktdp.access_tile<2x4xindex>
  %al = ktdp.load %at : <2x4xindex> -> tensor<2x4xf32>
  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f32> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [2, 4], strides: [4, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 2, 0, 2>}} : memref<2x4xf32>
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

// -----

// Case 11 -- a value the rewrite retyped, read by something that is not a
// linalg.generic.
//
// tensor.extract_slice is the reachable shape of this: LowerComputeOps lowers
// tt.split to a pair of them, so the op arrives from the same pipeline that
// produces the rest of this IR. It names its offsets and sizes per logical dim,
// which is exactly what physicalizing invalidates -- the operand becomes
// <2x64x64> while the op still says <64x128>. Without this check the only
// complaint would come from a verifier naming neither this pass nor the op.
//
// The consumer here reads the GENERIC's result, not the load's. That is what
// makes it land in rewriteGeneric rather than the up-front consumer scan, which
// only looks at what a ktdp.load feeds.
//
// The check is on the consumer, not on the value: by then the value IS at
// physical rank, so asking about the value would answer yes and let it through.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#s2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func @sliced_consumer(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  // The store is what makes %r physicalize in the first place; the slice below
  // is the second reader, and the one that cannot follow.
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  // expected-error @below {{rewrite-descriptor-layout-generic: this op reads a value the rewrite retyped, but the rewrite restates only linalg.generic, so this op still names the logical type}}
  %half = tensor.extract_slice %r[0, 0] [64, 64] [1, 1] : tensor<64x128xf32> to tensor<64x64xf32>
  %ht = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s2} : memref<64x128xf32> -> !ktdp.access_tile<64x64xindex>
  ktdp.store %half, %ht : tensor<64x64xf32>, <64x64xindex>
  tt.return
}
}
