// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics

// A layout that is well formed and still wrong for the context it annotates.
//
// Every tts.tensor_layout here passes the tts dialect's own verifier: the fields
// are consistent, the splits are paired, the ranks line up. What is wrong is the
// fit between the layout and something the verifier cannot see -- the block the
// tile reads, another operand's layout, or the op that consumes the load. So each
// diagnostic has to come from the pass, and each has to name the pass or the op
// rather than surfacing later as a verifier failure about an indexing map.
//
// Every case here is reachable through the shapes the backend's own lowering
// produces. Diagnostics about a layout the pass cannot consume at all, or a
// malformed KTIR chain beneath one it can, live in invalid-ktir.mlir; the
// structural rules the verifier owns are in
// test/Dialect/TTS/tensor-layout-verifier.mlir.

// Case 1 -- the layout does not fit the block.
//
// The mod dim's extent IS the stick width, so a block whose logical extent on the
// split dim is below that width would give a physical dim wider than the data it
// indexes.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
tt.func @block_smaller_than_stick(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f32> to index
  // Stick-on-N at width 64, but N is only 32.
  %av = ktdp.construct_memory_view %ai, sizes: [64, 32], strides: [32, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x32xf32>
  // expected-error @below {{tts.tensor_layout: block extent of stick dim (32) is smaller than the stick size (64); a stick dim cannot be sub-stick}}
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

// Case 2 -- two layouts disagree about one loop dim.
//
// The rebuilt domain gives a split dim a (stick, elem) pair with ONE width, and an
// operand holding that dim whole addresses it as `stick * width + elem` -- see
// rebuild-composite.mlir. The pair can have only one pair of extents, so two
// widths on one loop dim leave no such composite -- and naming the two candidates
// shows why neither choice is safe. The third operand, %c, carries no layout and
// so holds logical dim 1 whole at 128; it is the one that would have to read a
// composite, and its two candidates are
//
//   s * 64 + e   with (s, e) running (0..1, 0..63)
//   s * 32 + e   with (s, e) running (0..3, 0..31)
//
// At s = 0 the first names elements 0..63 of %c and the second names 0..31; at
// s = 1 the first names 64..127 and the second 32..63. They agree on nothing past
// the zeroth stick, so whichever the domain declares, the operand that split at
// the other width reads %c at the wrong elements -- silently.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @two_stick_widths_on_one_dim(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>

  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f32> to index
  // Same logical dim, same role, width 32 rather than 64.
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}} : memref<64x128xf32>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %bl = ktdp.load %bt : <64x128xindex> -> tensor<64x128xf32>

  // The third operand carries no layout, so it holds logical dim 1 whole and is
  // the one that would have to read a composite.
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

// -----

// Case 3 -- the layout is fine and the consumer is the wrong shape of op.
//
// LowerComputeOps lowers tt.dot to a NAMED linalg.matmul, which is fixed at
// logical rank and so cannot carry a stick dim. This pass rewrites only generics,
// so it rejects the op up front, before physicalizeDescriptors retypes the load
// underneath it.
// The ordering is the point: retyping first would leave a rank mismatch that
// MLIR's own verifier reports against an indexing map, naming neither this pass
// nor what it could not restate.
//
// A[M=128, K=64] stick-on-M(64) -> physical [M/64, K, M%64] = [2, 64, 64].

#id = affine_map<(d0, d1) -> (d0, d1)>
#sa = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
tt.func @named_matmul_declines(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  %av = ktdp.construct_memory_view %ai, sizes: [128, 64], strides: [64, 1] {coordinate_set = #sa, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf16>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #sa} : memref<128x64xf16> -> !ktdp.access_tile<128x64xindex>
  %al = ktdp.load %at : <128x64xindex> -> tensor<128x64xf16>

  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f16> to index
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 64], strides: [64, 1] {coordinate_set = #sb, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id, access_tile_set = #sb} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
  %bl = ktdp.load %bt : <64x64xindex> -> tensor<64x64xf16>

  %ci = builtin.unrealized_conversion_cast %c : !tt.ptr<f16> to index
  %cv = ktdp.construct_memory_view %ci, sizes: [128, 64], strides: [64, 1] {coordinate_set = #sa, memory_space = #ktdp.memory_space<global>} : memref<128x64xf16>
  %ct = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #sa} : memref<128x64xf16> -> !ktdp.access_tile<128x64xindex>
  %cl = ktdp.load %ct : <128x64xindex> -> tensor<128x64xf16>

  // expected-error @below {{rewrite-descriptor-layout-generic: this op reads a value on a physicalized chain, but the rewrite restates only linalg.generic; spell this op as one}}
  %d = linalg.matmul ins(%al, %bl : tensor<128x64xf16>, tensor<64x64xf16>) outs(%cl : tensor<128x64xf16>) -> tensor<128x64xf16>

  %st = ktdp.construct_access_tile %cv[%c0, %c0] {access_tile_order = #id, access_tile_set = #sa} : memref<128x64xf16> -> !ktdp.access_tile<128x64xindex>
  ktdp.store %d, %st : tensor<128x64xf16>, <128x64xindex>
  tt.return
}
}

// -----

// Case 4 -- only the destination is annotated, and nothing mediates the data.
//
// A pure load-to-store copy, so there is no linalg.generic anywhere -- and
// rewriteGeneric is the only thing in this pass that changes a data value's type.
// Phase 1 redirects the store's access tile to the physical tile regardless, so the
// store would end up with a rank-3 access tile and its rank-2 loaded data, and
// ktdp.store's own verifier would report `data tile shape must match access tile
// shape` about an op nobody named.
//
// The named pass absorbs this in a widening stage; this one declines instead, and
// the remedy is one more marker. rebuild-composite.mlir case 3 is the same store
// with a generic in between, which needs no widening because the rebuild gives both
// ends the same domain -- that contrast is why the decline is narrow.

#idc = affine_map<(d0, d1) -> (d0, d1)>
#sc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @store_only_annotated_copy(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  // The source carries no layout, so its load stays rank 2.
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sc, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #idc, access_tile_set = #sc} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>

  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sc, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #idc, access_tile_set = #sc} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  // expected-error @below {{rewrite-descriptor-layout-generic: this store's access tile is physicalized but its data is not on a physicalized chain, and this pass restates only linalg.generic; a one-sided annotation has no vehicle for the shape change, so annotate the source descriptor too, at a layout compatible with this one}}
  ktdp.store %al, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 5 -- an annotated view this pass never reaches.
//
// The post-condition, and the thing that makes SpyreBackend's device-footprint
// claim safe. Every DECLINE above fails the compile; this is the other way an
// annotation goes unhonoured -- not declined, just not walked. The pass redirects
// access-tile users and nothing else, and it erases the logical view only once
// nothing uses it, so a view held by any other user survives with its layout
// intact.
//
// A bridge cast back to !tt.tensordesc is that other user here: no access
// tile, so physicalizeDescriptor finds nothing to redirect and the view stays live.
// On the real pipeline LowerTTSMarkers erases that cast along with the marker it
// existed for, so reaching this state means a descriptor nothing reads or writes --
// which is exactly the case a footprint claim must not be made for. Without the
// check this lowers cleanly to a LOGICAL artifact while
// metadata["device_layouts"] reports the physical footprint the kernel asked for --
// a claim about memory the kernel never addresses, which a launcher would then
// bounds-check against.

#sd = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>
module {
tt.func @annotation_never_reached(%o: !tt.ptr<f16>) {
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f16> to index
  // expected-error @below {{rewrite-descriptor-layout-generic: this memory view still carries a tts.tensor_layout after physicalization}}
  %ov = ktdp.construct_memory_view %oi, sizes: [128], strides: [1] {coordinate_set = #sd, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>}} : memref<128xf16>
  %od = builtin.unrealized_conversion_cast %ov : memref<128xf16> to !tt.tensordesc<128xf16>
  tt.return
}
}
