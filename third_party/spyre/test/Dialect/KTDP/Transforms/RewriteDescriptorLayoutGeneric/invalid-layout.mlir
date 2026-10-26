// RUN: spyre-triton-opt %s --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics
// RUN: spyre-triton-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics

// A layout that is well formed and still wrong for the context it annotates.
//
// Every tts.tensor_layout in cases 1 to 5 passes the tts dialect's own verifier:
// the fields are consistent, the splits are paired, the ranks line up. What is
// wrong is the fit between the layout and something the verifier cannot see --
// the block the tile reads, another operand's layout, or the op that consumes the
// load. So each diagnostic has to come from the pass, and each has to name the
// pass or the op rather than surfacing later as a verifier failure about an
// indexing map.
//
// Every one of those cases is reachable through the shapes the backend's own lowering
// produces. Diagnostics about a layout the pass cannot consume at all, or a
// malformed KTIR chain beneath one it can, live in invalid-ktir.mlir; the
// structural rules the verifier owns are in
// test/Dialect/TTS/tensor-layout-verifier.mlir.
//
// Case 6 is the exception: a layout that breaks one of those structural rules,
// read by the pass itself, once per stick-dim rule. The second RUN line skips
// the verifier on parsing, so the diagnostic there comes from readCoordMap,
// which calls the same shared checker; under the first RUN line the parser's
// verifier emits it.
//
// Cases 7 to 12 are one refusal in six shapes: an operand that holds whole a dim
// another operand of the same generic splits. Restating it would take the
// linearizing map `stick * W + lane`, which the scheduler cannot take, so the pass
// refuses the generic, once per such operand, with a note at the operand's
// producer.

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
// operand holding that dim whole would address it as `stick * width + elem`. The
// pair can have only one pair of extents, so two widths on one loop dim leave no
// such composite -- and naming the two candidates shows why neither choice is
// safe. The third operand, %c, carries no layout and so holds logical dim 1 whole
// at 128; it is the one that would have to read a composite, and its two
// candidates are
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
// the remedy is one more marker. rebuild-composite.mlir case 2 is the same store
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

// -----

// Case 6 -- a layout breaking a stick-dim rule, read by the pass, one rule each.
//
// Two stick dims: [M, K] -> [M/64, K/64, M%64, K%64]. Each logical dim is a
// well-formed split, and the layout as a whole has two stick dims. readCoordMap
// rejects it with the shared checker's message.

#s6 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#id6 = affine_map<(d0, d1) -> (d0, d1)>
module {
tt.func @two_mods(%a: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  // expected-error @below {{tts.tensor_layout: a layout has at most one stick dim (a mod or splat entry), got 2}}
  %av = ktdp.construct_memory_view %ai, sizes: [128, 128], strides: [128, 1] {coordinate_set = #s6, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 1, 0, 1>, phys_op = array<i64: 1, 1, 2, 2>, phys_arg = array<i64: 64, 64, 64, 64>}} : memref<128x128xf16>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id6, access_tile_set = #s6} : memref<128x128xf16> -> !ktdp.access_tile<128x128xindex>
  %al = ktdp.load %at : <128x128xindex> -> tensor<128x128xf16>
  tt.return
}
}

// -----

// Case 6, the stick dim not last: [M, K] -> [K/64, K%64, M].

#s7 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#id7 = affine_map<(d0, d1) -> (d0, d1)>
module {
tt.func @stick_not_last(%a: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  // expected-error @below {{tts.tensor_layout: the stick dim must be the last physical dim, got it at phys dim 1 of 3}}
  %av = ktdp.construct_memory_view %ai, sizes: [128, 128], strides: [128, 1] {coordinate_set = #s7, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 1, 0>, phys_op = array<i64: 1, 2, 0>, phys_arg = array<i64: 64, 64, 0>}} : memref<128x128xf16>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id7, access_tile_set = #s7} : memref<128x128xf16> -> !ktdp.access_tile<128x128xindex>
  %al = ktdp.load %at : <128x128xindex> -> tensor<128x128xf16>
  tt.return
}
}

// -----

// Case 6, a split of two widths: [M, K] -> [K/32, M, K%64]. The split's two
// halves name different stick widths.

#s8 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#id8 = affine_map<(d0, d1) -> (d0, d1)>
module {
tt.func @split_width_mismatch(%a: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  // expected-error @below {{tts.tensor_layout: logical dim 1 is split with floordiv 32 but mod 64; the two must be the same stick width}}
  %av = ktdp.construct_memory_view %ai, sizes: [128, 128], strides: [128, 1] {coordinate_set = #s8, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 64>}} : memref<128x128xf16>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id8, access_tile_set = #s8} : memref<128x128xf16> -> !ktdp.access_tile<128x128xindex>
  %al = ktdp.load %at : <128x128xindex> -> tensor<128x128xf16>
  tt.return
}
}

// -----

// Case 7 -- an operand on no layout reads a dim another operand splits.
//
// The focused form of the refusal. The domain splits logical loop dim 1 because
// %a's layout does, so it has a stick loop and a lane loop for it. %b carries no
// layout, so it keeps its one physical dim there, and the only map result naming
// its element from the two loop dims is `stick * 64 + lane`. The pass refuses the
// generic rather than emitting that map, names the operand, and points at the op
// that produces it. Nothing is retyped first, so the generic is left as it was.

#id = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @unannotated_operand_reads_split_dim(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %o: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f16> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf16>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf16>
  %bi = builtin.unrealized_conversion_cast %b : !tt.ptr<f16> to index
  %bv = ktdp.construct_memory_view %bi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>} : memref<64x128xf16>
  %bt = ktdp.construct_access_tile %bv[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  // expected-note @below {{operand 1 is produced here}}
  %bl = ktdp.load %bt : <64x128xindex> -> tensor<64x128xf16>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f16> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf16>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id, access_tile_set = #s} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %e = tensor.empty() : tensor<64x128xf16>
  // expected-error @below {{rewrite-descriptor-layout-generic: operand 1 has no layout and reads logical loop dim 1 whole, which operand 0 splits into stick and lane at width 64; restating operand 1 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 1 a layout that splits loop dim 1 at width 64 too}}
  %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%al, %bl : tensor<64x128xf16>, tensor<64x128xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%x: f16, %y: f16, %z: f16):
    %s = arith.addf %x, %y : f16
    linalg.yield %s : f16
  } -> tensor<64x128xf16>
  ktdp.store %r, %ot : tensor<64x128xf16>, <64x128xindex>
  tt.return
}
}

// -----

// Case 8 -- an operand on no layout at the result.
//
// The same refusal on the `outs` side: the input splits logical dim 1 and the
// unannotated store destination leaves the generic's result holding it whole.

#ide = affine_map<(d0, d1) -> (d0, d1)>
#se = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
tt.func @elementwise_unannotated_store(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  %av = ktdp.construct_memory_view %ai, sizes: [64, 128], strides: [128, 1] {coordinate_set = #se, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #ide, access_tile_set = #se} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  %al = ktdp.load %at : <64x128xindex> -> tensor<64x128xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #se, memory_space = #ktdp.memory_space<global>} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #ide, access_tile_set = #se} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  // expected-note @below {{operand 1 is produced here}}
  %e = tensor.empty() : tensor<64x128xf32>
  // expected-error @below {{rewrite-descriptor-layout-generic: operand 1 has no layout and reads logical loop dim 1 whole, which operand 0 splits into stick and lane at width 64; restating operand 1 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 1 a layout that splits loop dim 1 at width 64 too}}
  %r = linalg.generic {indexing_maps = [#ide, #ide], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    %n = arith.negf %x : f32
    linalg.yield %n : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 9 -- a transpose whose two operands split DIFFERENT loop dims.
//
// Each operand's layout carries whole the dim the other splits, so each would
// need the other's linearization, and both are named.

#in  = affine_map<(d0, d1) -> (d1, d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
#sin  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#sout = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
tt.func @transpose_different_split_dims(%a: !tt.ptr<f32>, %o: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : index
  %ai = builtin.unrealized_conversion_cast %a : !tt.ptr<f32> to index
  // Split the input's logical dim 1, which the consumer's map reads as loop d0.
  %av = ktdp.construct_memory_view %ai, sizes: [128, 64], strides: [64, 1] {coordinate_set = #sin, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
  %at = ktdp.construct_access_tile %av[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sin} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
  // expected-note @below {{operand 0 is produced here}}
  %al = ktdp.load %at : <128x64xindex> -> tensor<128x64xf32>
  %oi = builtin.unrealized_conversion_cast %o : !tt.ptr<f32> to index
  // Split the output's logical dim 1, which is loop d1.
  %ov = ktdp.construct_memory_view %oi, sizes: [64, 128], strides: [128, 1] {coordinate_set = #sout, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  %ot = ktdp.construct_access_tile %ov[%c0, %c0] {access_tile_order = #id2, access_tile_set = #sout} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
  // expected-note @below {{operand 1 is produced here}}
  %e = tensor.empty() : tensor<64x128xf32>
  // expected-error @below {{rewrite-descriptor-layout-generic: operand 0's layout carries logical loop dim 1 whole, which operand 1 splits into stick and lane at width 64; restating operand 0 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 0 a layout that splits loop dim 1 at width 64 too}}
  // expected-error @below {{rewrite-descriptor-layout-generic: operand 1's layout carries logical loop dim 0 whole, which operand 0 splits into stick and lane at width 64; restating operand 1 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 1 a layout that splits loop dim 0 at width 64 too}}
  %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel"]} ins(%al : tensor<128x64xf32>) outs(%e : tensor<64x128xf32>) {
  ^bb0(%x: f32, %y: f32):
    linalg.yield %x : f32
  } -> tensor<64x128xf32>
  ktdp.store %r, %ot : tensor<64x128xf32>, <64x128xindex>
  tt.return
}
}

// -----

// Case 10 -- a contraction whose physicalized result splits a dim an input holds
// whole.
//
// A and B are stick-on-K and the output D is stick-on-N, so B, whose layout
// splits K, carries N whole.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  tt.func @matmul_k_split_annotated_output(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    // expected-note @below {{operand 1 is produced here}}
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    // expected-error @below {{rewrite-descriptor-layout-generic: operand 1's layout carries logical loop dim 1 whole, which operand 2 splits into stick and lane at width 64; restating operand 1 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 1 a layout that splits loop dim 1 at width 64 too}}
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 11 -- both inputs split a PARALLEL dim, and the result holds both whole.
//
// A is stick-on-M and B stick-on-N; the unannotated result reads M and N whole,
// so one operand would need two linearizations. One error per operand, naming
// the first split dim it holds whole.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  tt.func @matmul_both_parallel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    // expected-note @below {{operand 2 is produced here}}
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    // expected-error @below {{rewrite-descriptor-layout-generic: operand 2 has no layout and reads logical loop dim 0 whole, which operand 0 splits into stick and lane at width 64; restating operand 2 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 2 a layout that splits loop dim 0 at width 64 too}}
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}

// -----

// Case 12 -- one input splits a parallel dim and the other the reduced dim.
//
// A is stick-on-M and B stick-on-K, so A carries K whole and the unannotated
// result carries M whole.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#seta = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#setb = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#setc = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  tt.func @matmul_a_parallel_b_ksplit(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to index
    %1 = ktdp.construct_memory_view %0, sizes: [64, 128], strides: [128, 1] {coordinate_set = #seta, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
    %2 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map, access_tile_set = #seta} : memref<64x128xf32> -> !ktdp.access_tile<64x128xindex>
    // expected-note @below {{operand 0 is produced here}}
    %3 = ktdp.load %2 : <64x128xindex> -> tensor<64x128xf32>
    %4 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to index
    %5 = ktdp.construct_memory_view %4, sizes: [128, 64], strides: [64, 1] {coordinate_set = #setb, memory_space = #ktdp.memory_space<global>,
        tts.tensor_layout = {phys_src = array<i64: 0, 1, 0>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<128x64xf32>
    %6 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map, access_tile_set = #setb} : memref<128x64xf32> -> !ktdp.access_tile<128x64xindex>
    %7 = ktdp.load %6 : <128x64xindex> -> tensor<128x64xf32>
    %8 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to index
    %9 = ktdp.construct_memory_view %8, sizes: [64, 64], strides: [64, 1] {coordinate_set = #setc, memory_space = #ktdp.memory_space<global>} : memref<64x64xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    // expected-note @below {{operand 2 is produced here}}
    %11 = ktdp.load %10 : <64x64xindex> -> tensor<64x64xf32>
    // expected-error @below {{rewrite-descriptor-layout-generic: operand 0's layout carries logical loop dim 2 whole, which operand 1 splits into stick and lane at width 64; restating operand 0 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 0 a layout that splits loop dim 2 at width 64 too}}
    // expected-error @below {{rewrite-descriptor-layout-generic: operand 2 has no layout and reads logical loop dim 0 whole, which operand 0 splits into stick and lane at width 64; restating operand 2 over that split would need the linearizing map `stick * 64 + lane`, which the scheduler cannot take. Give operand 2 a layout that splits loop dim 0 at width 64 too}}
    %12 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %7 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%11 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<64x64xf32>
    %13 = ktdp.construct_access_tile %9[%c0, %c0] {access_tile_order = #map, access_tile_set = #setc} : memref<64x64xf32> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %12, %13 : tensor<64x64xf32>, <64x64xindex>
    tt.return
  }
}
