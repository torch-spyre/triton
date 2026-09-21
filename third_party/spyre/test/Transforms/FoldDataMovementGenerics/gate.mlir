// RUN: spyre-triton-opt %s --fold-data-movement-generics -split-input-file -verify-diagnostics

// THE GATE: the condition is "absorption failed" AND "on a path
// RewriteDescriptorLayoutGeneric will physicalize". absorb.mlir has the same
// unabsorbable shape ops with no annotated memory view anywhere and leaves every
// one of them silently; here each sits on such a path and is REFUSED. Its own
// file because the subject is the path rather than the op -- and because proving
// a diagnostic needs -verify-diagnostics rather than FileCheck.
//
// Why gated at all is forward-looking rather than protective: as annotation
// coverage grows, the gate fires more often and converges to ungated behaviour by
// itself, so it never needs removing. See FoldDataMovementGenerics.cpp.
//
// THE SHAPE OF CASES 1-6 is `stat_chain_on_stick`'s, because that is the shape
// nothing else catches: the data view is annotated, but the STATISTIC view that
// feeds the re-indexing op is deliberately NOT, its logical shape already being
// its physical one. RewriteDescriptorLayoutGeneric calls retypeToPhysical only for
// an operand that HAS a layout, so an operand that has none is bridged by
// rebuildMap with no check at all, and a forward-only walk from the annotated
// view's loads would miss all of it.

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#lane0 = affine_map<(d0, d1) -> (d0, 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// Case 1: a `tensor.reshape` on a physicalized path. No result-to-source map
// exists for it at all -- its shape is a tensor operand, so there is no static
// structure to read one off.
module {
func.func @reshape_on_physicalized_path(%base: index, %stat: index) {
  %c0 = arith.constant 0 : index
  %shape = arith.constant dense<[64, 1]> : tensor<2xindex>
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %l = ktdp.load %t : <64x128xindex> -> tensor<64x128xf16>
  // The statistic view, UNANNOTATED on purpose.
  %sv = ktdp.construct_memory_view %stat, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
  %stile = ktdp.construct_access_tile %sv[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %sl = ktdp.load %stile : <64xindex> -> tensor<64xf16>
  // expected-error @below {{fold-data-movement-generics: this op re-indexes a value on a path the layout pass will physicalize and it cannot be restated as an indexing map on its consumer (no affine map from its result coordinates to its source coordinates is derivable -- its shape is an operand rather than a reassociation), so the layout pass would bridge it with a linearizing map, which the scheduler cannot project loop IVs through}}
  %r = tensor.reshape %sl(%shape) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
  %e = tensor.empty() : tensor<64x128xf16>
  %g = linalg.generic {indexing_maps = [#map, #lane0, #map], iterator_types = ["parallel", "parallel"]} ins(%l, %r : tensor<64x128xf16>, tensor<64x1xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %d = arith.subf %a, %b : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %g, %ot : tensor<64x128xf16>, <64x128xindex>
  return
}
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#lane0 = affine_map<(d0, d1) -> (d0, 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>

// Case 2: a `tensor.concat` on a physicalized path. Not a coordinate change at
// all -- it selects between operands per coordinate, and one operand map names
// one operand.
module {
func.func @concat_on_physicalized_path(%base: index, %stat: index) {
  %c0 = arith.constant 0 : index
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %l = ktdp.load %t : <64x128xindex> -> tensor<64x128xf16>
  %sv = ktdp.construct_memory_view %stat, sizes: [64, 1], strides: [1, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>} : memref<64x1xf16>
  %stile = ktdp.construct_access_tile %sv[%c0, %c0] {access_tile_order = #map2, access_tile_set = #set3} : memref<64x1xf16> -> !ktdp.access_tile<64x1xindex>
  %sl = ktdp.load %stile : <64x1xindex> -> tensor<64x1xf16>
  // expected-error @below {{fold-data-movement-generics: this op re-indexes a value on a path the layout pass will physicalize and it cannot be restated as an indexing map on its consumer (it selects between operands per coordinate, which one operand map cannot state), so the layout pass would bridge it with a linearizing map, which the scheduler cannot project loop IVs through}}
  %c = tensor.concat dim(1) %sl, %sl : (tensor<64x1xf16>, tensor<64x1xf16>) -> tensor<64x2xf16>
  %e = tensor.empty() : tensor<64x128xf16>
  %g = linalg.generic {indexing_maps = [#map, #lane0, #map], iterator_types = ["parallel", "parallel"]} ins(%l, %c : tensor<64x128xf16>, tensor<64x2xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %d = arith.subf %a, %b : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %g, %ot : tensor<64x128xf16>, <64x128xindex>
  return
}
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0)>

// Case 3: a LINEARIZING `tensor.collapse_shape` on a physicalized path. This one
// HAS a result-to-source map -- `(d0) -> (d0 floordiv 2, d0 mod 2)`, which this
// pass emits rather than refusing to state, correcting the claim that it is
// inexpressible. What refuses it is the projection check on the map the consumer
// would be left holding: composed with `(d0, d1) -> (d1)` that is
// `(d0, d1) -> (d1 floordiv 2, d1 mod 2)`, and a loop IV does not go through a
// floordiv.
module {
func.func @linearizing_collapse_on_physicalized_path(%base: index, %stat: index) {
  %c0 = arith.constant 0 : index
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %l = ktdp.load %t : <64x128xindex> -> tensor<64x128xf16>
  %sv = ktdp.construct_memory_view %stat, sizes: [64, 2], strides: [2, 1] {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>} : memref<64x2xf16>
  %stile = ktdp.construct_access_tile %sv[%c0, %c0] {access_tile_order = #map2, access_tile_set = #set4} : memref<64x2xf16> -> !ktdp.access_tile<64x2xindex>
  %sl = ktdp.load %stile : <64x2xindex> -> tensor<64x2xf16>
  // expected-error @below {{fold-data-movement-generics: this op re-indexes a value on a path the layout pass will physicalize and it cannot be restated as an indexing map on its consumer (reassociation group 0 fuses 2 non-unit dims, which only a linearizing map can state), so the layout pass would bridge it with a linearizing map, which the scheduler cannot project loop IVs through}}
  %c = tensor.collapse_shape %sl [[0, 1]] : tensor<64x2xf16> into tensor<128xf16>
  %e = tensor.empty() : tensor<64x128xf16>
  %g = linalg.generic {indexing_maps = [#map, #col, #map], iterator_types = ["parallel", "parallel"]} ins(%l, %c : tensor<64x128xf16>, tensor<128xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %d = arith.subf %a, %b : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %g, %ot : tensor<64x128xf16>, <64x128xindex>
  return
}
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set5 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 255 >= 0)>

// Case 4: `tensor.extract_slice` on a physicalized path is NOT rejected, which is
// what pins the decision to classify it as "not a coordinate restatement" rather
// than as one this pass cannot absorb. The reason is not that the scheduler looks
// through it (it does) -- it is that a slice CROPS, and a linalg operand map
// cannot state an extent change: linalg infers its loop bounds from the operand
// shapes through the maps, so no map on the unsliced source reproduces the
// cropped one. "Not a restatement" means neither absorbed nor refused, here and
// in absorb.mlir's test 9, which is the status quo and deliberate.
module {
func.func @extract_slice_on_physicalized_path(%base: index, %stat: index) {
  %c0 = arith.constant 0 : index
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %l = ktdp.load %t : <64x128xindex> -> tensor<64x128xf16>
  %sv = ktdp.construct_memory_view %stat, sizes: [64, 256], strides: [256, 1] {coordinate_set = #set5, memory_space = #ktdp.memory_space<global>} : memref<64x256xf16>
  %stile = ktdp.construct_access_tile %sv[%c0, %c0] {access_tile_order = #map2, access_tile_set = #set5} : memref<64x256xf16> -> !ktdp.access_tile<64x256xindex>
  %sl = ktdp.load %stile : <64x256xindex> -> tensor<64x256xf16>
  %s = tensor.extract_slice %sl[0, 0] [64, 128] [1, 1] : tensor<64x256xf16> to tensor<64x128xf16>
  %e = tensor.empty() : tensor<64x128xf16>
  %g = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%l, %s : tensor<64x128xf16>, tensor<64x128xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %d = arith.subf %a, %b : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  return
}
}

// -----

// THE ANTI-VACUITY PAIR, which matters more than any of the four above. Cases 5
// and 6 are BYTE-IDENTICAL except that case 5's data view carries
// `tts.tensor_layout` and case 6's does not. Case 5 must error and case 6 must
// come out clean.
//
// That pairing is the property worth testing, because neither failure mode is
// visible on its own: a gate stuck always-false leaves case 5 with no diagnostic,
// and a gate stuck always-true makes case 6 fail. One case alone catches neither.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#lane0 = affine_map<(d0, d1) -> (d0, 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// Case 5: ANNOTATED. The reshape is refused.
module {
func.func @pair_annotated(%base: index, %stat: index) {
  %c0 = arith.constant 0 : index
  %shape = arith.constant dense<[64, 1]> : tensor<2xindex>
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %l = ktdp.load %t : <64x128xindex> -> tensor<64x128xf16>
  %sv = ktdp.construct_memory_view %stat, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
  %stile = ktdp.construct_access_tile %sv[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %sl = ktdp.load %stile : <64xindex> -> tensor<64xf16>
  // expected-error @below {{fold-data-movement-generics: this op re-indexes a value on a path the layout pass will physicalize}}
  %r = tensor.reshape %sl(%shape) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
  %e = tensor.empty() : tensor<64x128xf16>
  %g = linalg.generic {indexing_maps = [#map, #lane0, #map], iterator_types = ["parallel", "parallel"]} ins(%l, %r : tensor<64x128xf16>, tensor<64x1xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %d = arith.subf %a, %b : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %g, %ot : tensor<64x128xf16>, <64x128xindex>
  return
}
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#lane0 = affine_map<(d0, d1) -> (d0, 0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// Case 6: UNANNOTATED, and otherwise identical to case 5. No `expected-error`, so
// -verify-diagnostics fails this chunk if the pass says anything at all.
module {
func.func @pair_unannotated(%base: index, %stat: index) {
  %c0 = arith.constant 0 : index
  %shape = arith.constant dense<[64, 1]> : tensor<2xindex>
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x128xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  %l = ktdp.load %t : <64x128xindex> -> tensor<64x128xf16>
  %sv = ktdp.construct_memory_view %stat, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
  %stile = ktdp.construct_access_tile %sv[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  %sl = ktdp.load %stile : <64xindex> -> tensor<64xf16>
  %r = tensor.reshape %sl(%shape) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
  %e = tensor.empty() : tensor<64x128xf16>
  %g = linalg.generic {indexing_maps = [#map, #lane0, #map], iterator_types = ["parallel", "parallel"]} ins(%l, %r : tensor<64x128xf16>, tensor<64x1xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %d = arith.subf %a, %b : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %ot = ktdp.construct_access_tile %v[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
  ktdp.store %g, %ot : tensor<64x128xf16>, <64x128xindex>
  return
}
}

// -----

// THE STORE SIDE, and its own anti-vacuity pair. Cases 1-6 above are all the
// `ins` side; this is the exact mirror, and it needs its own coverage because a
// different function of the layout pass is what goes wrong.
//
// findLayoutForResult walks a generic's result users looking for a `ktdp.store`
// DIRECTLY -- `dyn_cast<mlir::ktdp::StoreOp>`, `continue` otherwise. Put a shape
// op between the generic and the store and that dyn_cast fails, the function
// returns null, the `outs` operand gets CoordOp::Identity, and rebuildMap folds
// the LINEARIZATION into the outs map rather than an ins map. Same hazard, other
// end of the op.
//
// Rejection only, never absorption: see THE STORE SIDE in the header of
// FoldDataMovementGenerics.cpp for why, and for what the eventual fix chooses
// between. `gather__1d` shows the shape is real -- an `8x1 -> 8` `tt.reshape`
// feeding `ktdp.store` -- on a kernel this pass never sees.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// Case 7: ANNOTATED store view, with a `tensor.reshape` between the generic that
// produces the data and the store that consumes it. Refused.
module {
func.func @reshape_between_generic_and_store(%src: index, %out: index) {
  %c0 = arith.constant 0 : index
  %shape = arith.constant dense<[64]> : tensor<1xindex>
  %sv = ktdp.construct_memory_view %src, sizes: [64, 1], strides: [1, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x1xf16>
  %stile = ktdp.construct_access_tile %sv[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x1xf16> -> !ktdp.access_tile<64x1xindex>
  %sl = ktdp.load %stile : <64x1xindex> -> tensor<64x1xf16>
  %e = tensor.empty() : tensor<64x1xf16>
  %g = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%sl : tensor<64x1xf16>) outs(%e : tensor<64x1xf16>) {
  ^bb0(%in: f16, %o: f16):
    %d = arith.mulf %in, %in : f16
    linalg.yield %d : f16
  } -> tensor<64x1xf16>
  // expected-error @below {{fold-data-movement-generics: this op re-indexes a value on a path the layout pass will physicalize and it cannot be restated as an indexing map on its consumer (no affine map from its result coordinates to its source coordinates is derivable -- its shape is an operand rather than a reassociation), so the layout pass would bridge it with a linearizing map, which the scheduler cannot project loop IVs through}}
  %r = tensor.reshape %g(%shape) : (tensor<64x1xf16>, tensor<1xindex>) -> tensor<64xf16>
  %ov = ktdp.construct_memory_view %out, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>}} : memref<64xf16>
  %ot = ktdp.construct_access_tile %ov[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  ktdp.store %r, %ot : tensor<64xf16>, <64xindex>
  return
}
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// Case 8: UNANNOTATED, and otherwise identical to case 7. No `expected-error`, so
// -verify-diagnostics fails this chunk if the pass says anything at all.
module {
func.func @reshape_between_generic_and_store_unannotated(%src: index, %out: index) {
  %c0 = arith.constant 0 : index
  %shape = arith.constant dense<[64]> : tensor<1xindex>
  %sv = ktdp.construct_memory_view %src, sizes: [64, 1], strides: [1, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x1xf16>
  %stile = ktdp.construct_access_tile %sv[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<64x1xf16> -> !ktdp.access_tile<64x1xindex>
  %sl = ktdp.load %stile : <64x1xindex> -> tensor<64x1xf16>
  %e = tensor.empty() : tensor<64x1xf16>
  %g = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%sl : tensor<64x1xf16>) outs(%e : tensor<64x1xf16>) {
  ^bb0(%in: f16, %o: f16):
    %d = arith.mulf %in, %in : f16
    linalg.yield %d : f16
  } -> tensor<64x1xf16>
  %r = tensor.reshape %g(%shape) : (tensor<64x1xf16>, tensor<1xindex>) -> tensor<64xf16>
  %ov = ktdp.construct_memory_view %out, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
  %ot = ktdp.construct_access_tile %ov[%c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<64xf16> -> !ktdp.access_tile<64xindex>
  ktdp.store %r, %ot : tensor<64xf16>, <64xindex>
  return
}
}
