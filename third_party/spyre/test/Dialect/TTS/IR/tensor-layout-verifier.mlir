// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.tensor_layout verifier, on its own -- no pass runs here.
//
// That is the point of the file. `tts.tensor_layout` is a DISCARDABLE attribute,
// and the usual expectation is that MLIR does not check one at all. It does when
// the name carries a dialect prefix: the verifier looks up the attribute name's
// dialect and calls its verifyOperationAttribute, on any op, at every
// verification point. The prefix is ours, so these checks are ours -- and they
// fire on a bare parse, ahead of any pass, which is what the empty RUN line
// below asserts.
//
// The structural rules live in one function, tts::verifyTensorLayoutArrays, and
// RewriteDescriptorLayoutGeneric's readCoordMap calls the same one rather than
// restating it. So the cases below cover both callers. The rules that pass alone
// adds -- a layout that is a valid coordinate map yet leaves the rewrite nothing
// addressable to build -- are in
// test/Dialect/KTDP/Transforms/RewriteDescriptorLayoutGeneric/invalid-ktir.mlir.

// Case 1 -- the accepted form, which nothing else here would distinguish a
// silently-unchecked attribute from.
//
// Stick-on-N at width 64 over a [64, 128] view: phys [ceildiv(128,64), 64, 64].
// It survives the round trip unchanged, and the entries come back in the
// printer's sorted order rather than the order written.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
// CHECK-LABEL: func.func @accepted
// CHECK: ktdp.construct_memory_view
// CHECK-SAME: tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}
func.func @accepted(%base: index) {
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 2 -- the attribute on an op it means nothing on.
//
// The layout describes the tensor a memory view addresses. Anywhere else it is
// inert rather than wrong, which is exactly the failure a verifier has to catch:
// a writer that put it on the load instead of the view would see no complaint
// and no effect.
func.func @wrong_op(%a: i32) -> i32 {
  // expected-error @below {{'tts.tensor_layout' is only meaningful on a ktdp.construct_memory_view, which this op is not}}
  %c = arith.addi %a, %a {tts.tensor_layout = {phys_src = array<i64: 0>, phys_op = array<i64: 0>, phys_arg = array<i64: 0>}} : i32
  return %c : i32
}

// -----

// Case 3 -- a name in our namespace that we do not define.
//
// Owning the prefix means owning every name under it, so a typo is rejected
// rather than carried. Without this the dialect would silently accept
// `tts.tensor_layouts` and the layout would simply never be read.
func.func @unknown_name(%a: i32) -> i32 {
  // expected-error @below {{attribute 'tts.tensor_layouts' is not one the tts dialect defines}}
  %c = arith.addi %a, %a {tts.tensor_layouts = 1 : i64} : i32
  return %c : i32
}

// -----

// Case 4 -- the value is not a dictionary at all.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @not_a_dictionary(%base: index) {
  // expected-error @below {{tts.tensor_layout: must be a dictionary of phys_src, phys_op and phys_arg}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = 7 : i64} : memref<64x128xf32>
  return
}
}

// -----

// Case 5 -- an entry missing.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @missing_entry(%base: index) {
  // expected-error @below {{tts.tensor_layout: missing 'phys_arg' entry}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 6 -- an entry of the wrong type.
//
// A plain ArrayAttr of integers prints almost identically to a dense i64 array
// and is not one, so this is the likeliest way to get the spelling wrong.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @wrong_entry_type(%base: index) {
  // expected-error @below {{tts.tensor_layout: 'phys_op' must be a dense i64 array}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = [1, 0, 2], phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 7 -- an entry we do not recognise.
//
// Rejected rather than ignored: a misspelled key alongside the three real ones
// would otherwise leave whatever the writer meant by it silently unread.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @unknown_entry(%base: index) {
  // expected-error @below {{tts.tensor_layout: expected exactly the entries phys_src, phys_op and phys_arg, got 4 entries}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>, phys_width = array<i64: 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 8 -- the three arrays are not parallel.
//
// Every consumer indexes all three with the same physical-dim index, so a short
// one is an out-of-bounds read waiting to happen rather than a partial layout.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @unequal_lengths(%base: index) {
  // expected-error @below {{tts.tensor_layout: phys_src, phys_op and phys_arg must have the same number of entries, got 3, 2 and 3}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 9 -- no physical dim at all.
//
// Parallel and consistent, and still not a layout: there is nothing in it to
// read, and every consumer seeds itself from some physical dim.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @no_physical_dims(%base: index) {
  // expected-error @below {{tts.tensor_layout: must describe at least one physical dim}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64>, phys_op = array<i64>, phys_arg = array<i64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 10 -- phys_src outside the logical rank.
//
// The rank is the VIEW's, read off the result memref. That is the same rank the
// marker op's verifier measured against, which read the descriptor's block type
// -- the extents differ between the two, the rank does not.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @phys_src_out_of_range(%base: index) {
  // expected-error @below {{tts.tensor_layout: phys_src[0] must be in [0, 2), got 2}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 2, 0, 2>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 11 -- a coord op code that is not one of the four.
//
// Consumers cast phys_op to an enum and switch on it without a default, so an
// unknown code leaves the derived coordinate expression unset.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @bad_coord_op(%base: index) {
  // expected-error @below {{tts.tensor_layout: phys_op[0] must be 0 (identity), 1 (floordiv), 2 (mod) or 3 (splat), got 4}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 4, 0, 2>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 12 -- a non-identity dim with a non-positive argument.
//
// phys_arg is the floordiv divisor / mod modulus / splat lane count. Zero
// divides by zero when deriving physical extents, and yields a zero-width stick
// or a zero-lane splat.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @zero_arg(%base: index) {
  // expected-error @below {{tts.tensor_layout: phys_arg[0] must be > 0 for a floordiv/mod/splat dim, got 0}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 0, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 13 -- a repeated logical dim as floordiv + splat.
//
// Two admissible pairings exist -- a stick split (floordiv + mod) and a splat
// re-stick (identity + splat) -- and they are not two halves of one relaxation.
// A split partitions the dim between its two entries; a splat replicates a dim
// that is also carried whole. Mixing them names the dim twice with neither
// meaning: there is no lane to say where in a stick an element sits, and nothing
// carrying the dim whole for the splat to replicate.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @floordiv_plus_splat(%base: index) {
  // expected-error @below {{tts.tensor_layout: logical dim 1 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry) or a splat re-stick (exactly one identity entry and one splat entry), got 0 identity, 1 floordiv, 0 mod, 1 splat}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 3>, phys_arg = array<i64: 64, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 14 -- the other mixed pairing, identity + mod.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @identity_plus_mod(%base: index) {
  // expected-error @below {{tts.tensor_layout: logical dim 1 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry) or a splat re-stick (exactly one identity entry and one splat entry), got 1 identity, 0 floordiv, 1 mod, 0 splat}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 0, 0, 2>, phys_arg = array<i64: 0, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 15 -- a logical dim named three times.
//
// Even a well-formed split plus a third entry is rejected: the pairings are
// exact counts, not lower bounds, because a consumer assigning one position per
// role has no position left for the extra dim.
#s = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
func.func @named_three_times(%base: index) {
  // expected-error @below {{tts.tensor_layout: logical dim 1 appears in 3 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry) or a splat re-stick (exactly one identity entry and one splat entry), got 1 identity, 1 floordiv, 1 mod, 0 splat}}
  %v = ktdp.construct_memory_view %base, sizes: [64, 128], strides: [128, 1] {coordinate_set = #s, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 1, 1, 0, 1>, phys_op = array<i64: 1, 0, 0, 2>, phys_arg = array<i64: 64, 0, 0, 64>}} : memref<64x128xf32>
  return
}
}

// -----

// Case 16 -- the splat re-stick, accepted.
//
// The pairing case 13 and case 14 are measured against: a rank-1 logical result
// physicalizing to (dim, lanes), which is the reduce-on-stick output layout.
#s1 = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
module {
// CHECK-LABEL: func.func @splat_restick_accepted
// CHECK: tts.tensor_layout = {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>}
func.func @splat_restick_accepted(%base: index) {
  %v = ktdp.construct_memory_view %base, sizes: [256], strides: [1] {coordinate_set = #s1, memory_space = #ktdp.memory_space<global>,
      tts.tensor_layout = {phys_src = array<i64: 0, 0>, phys_op = array<i64: 0, 3>, phys_arg = array<i64: 0, 64>}} : memref<256xf32>
  return
}
}
