// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.tensor_layout OP's verifier -- no pass runs here either.
//
// This file mirrors tensor-layout-verifier.mlir, which covers the same rules for
// the ATTRIBUTE form, and the mirroring is the point: both callers reach
// tts::verifyTensorLayoutArrays, so every message below is character-for-character
// the message the attribute produces for the same malformed layout. A rule that
// drifted between the two spellings would show up as a diff between these two
// files.
//
// That is also why the op's verifier uses emitError rather than emitOpError: the
// shared checker already names `tts.tensor_layout` in every message, so the
// op-error prefix would both say it twice and make the two files disagree.
//
// One rule is measured differently, and only one: the logical rank. Here it is
// the DESCRIPTOR's block type; for the attribute it is the memory view's memref.
// The extents differ between those two -- a descriptor's block is a tile of the
// view's tensor -- but the rank does not, so `phys_src[k] in [0, rank)` means the
// same thing on both sides. Cases @src_out_of_range and @negative_src are where
// that shows.
//
// Cases the attribute form has and this one cannot: the attribute's spelling
// checks (not a dictionary, a missing or misnamed or mistyped entry, the
// attribute on an op it means nothing on). ODS makes all of those unreachable
// here -- the operand type is constrained to !tt.tensordesc and the three arrays
// are inherent DenseI64ArrayAttrs -- and @missing_array below pins the one that
// still produces a diagnostic, from the generated verifier rather than ours.

// Case 1 -- the accepted form, which nothing else here would distinguish a
// silently-unverified op from.
//
// Stick-on-dim-1 at width 64 over a rank-2 block: phys [stick, dim0, lane].
// CHECK-LABEL: tt.func @accepted
// CHECK: tts.tensor_layout %{{.*}} {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <64x64xf32>
tt.func @accepted(%desc: !tt.tensordesc<64x64xf32>) {
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 2 -- one of the three arrays missing.
//
// Caught by the generated verifier, not ours: the arrays are inherent, so ODS
// requires them before our rules ever run. The attribute form has to check the
// same thing by hand, because a dictionary has no required keys.
tt.func @missing_array(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{'tts.tensor_layout' op requires attribute 'phys_arg'}}
  "tts.tensor_layout"(%desc) {phys_src = array<i64: 0>, phys_op = array<i64: 0>}
      : (!tt.tensordesc<64x64xf32>) -> ()
  tt.return
}

// -----

// Case 3 -- the three arrays are not parallel.
//
// Every consumer indexes all three with the same physical-dim index, so a short
// one is an out-of-bounds read waiting to happen rather than a partial layout.
tt.func @unequal_lengths(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: phys_src, phys_op and phys_arg must have the same number of entries, got 3, 2 and 3}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 4 -- no physical dim at all.
//
// Parallel and consistent, and still not a layout: there is nothing in it to
// read, and every consumer seeds itself from some physical dim.
tt.func @no_physical_dims(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: must describe at least one physical dim}}
  tts.tensor_layout %desc
    {phys_src = array<i64>,
     phys_op = array<i64>,
     phys_arg = array<i64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 5 -- phys_src past the logical rank.
//
// The rank is the DESCRIPTOR's, read off its block type. Same bound the
// attribute form states against the view's memref rank.
tt.func @src_out_of_range(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: phys_src[0] must be in [0, 2), got 2}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 2, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 6 -- phys_src below it. The bound is two-sided; a negative index reads
// off the front of the shape/stride arrays rather than off the back.
tt.func @negative_src(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: phys_src[1] must be in [0, 2), got -1}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, -1, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 7 -- a coord op code that is not one of the four.
//
// Consumers cast phys_op to an enum and switch on it without a default, so an
// unknown code leaves the derived coordinate expression unset.
tt.func @bad_coord_op(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: phys_op[0] must be 0 (identity), 1 (floordiv), 2 (mod) or 3 (splat), got 4}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 4, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 8 -- a non-identity dim with a non-positive argument.
//
// phys_arg is the floordiv divisor / mod modulus / splat lane count. Zero
// divides by zero when deriving physical extents, and yields a zero-width stick
// or a zero-lane splat.
tt.func @zero_arg(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: phys_arg[0] must be > 0 for a floordiv/mod/splat dim, got 0}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 0, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 9 -- a repeated logical dim as floordiv + splat.
//
// Two admissible pairings exist -- a stick split (floordiv + mod) and a splat
// re-stick (identity + splat) -- and they are not two halves of one relaxation.
// A split partitions the dim between its two entries; a splat replicates a dim
// that is also carried whole. Mixing them names the dim twice with neither
// meaning.
tt.func @floordiv_plus_splat(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: logical dim 1 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry) or a splat re-stick (exactly one identity entry and one splat entry), got 0 identity, 1 floordiv, 0 mod, 1 splat}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 3>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 10 -- the other mixed pairing, identity + mod.
tt.func @identity_plus_mod(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: logical dim 1 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry) or a splat re-stick (exactly one identity entry and one splat entry), got 1 identity, 0 floordiv, 1 mod, 0 splat}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 0, 0, 2>,
     phys_arg = array<i64: 0, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 11 -- a logical dim named twice with the same role.
tt.func @duplicate_identity(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: logical dim 0 appears in 2 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry) or a splat re-stick (exactly one identity entry and one splat entry), got 2 identity, 0 floordiv, 0 mod, 0 splat}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 0, 0, 1>,
     phys_op = array<i64: 0, 0, 0>,
     phys_arg = array<i64: 0, 0, 0>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 12 -- a logical dim named three times.
//
// Even a well-formed split plus a third entry is rejected: the pairings are
// exact counts, not lower bounds, because a consumer assigning one position per
// role has no position left for the extra dim.
tt.func @named_three_times(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @below {{tts.tensor_layout: logical dim 1 appears in 3 physical dims; a repeated logical dim is only valid as a stick split (exactly one floordiv entry and one mod entry) or a splat re-stick (exactly one identity entry and one splat entry), got 1 identity, 1 floordiv, 1 mod, 0 splat}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 1, 0, 1>,
     phys_op = array<i64: 1, 0, 0, 2>,
     phys_arg = array<i64: 64, 0, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Case 13 -- the splat re-stick, accepted.
//
// The pairing cases 9 and 10 are measured against: a rank-1 logical result
// physicalizing to (dim, lanes), which is the reduce-on-stick output layout.
// CHECK-LABEL: tt.func @splat_restick_accepted
// CHECK: tts.tensor_layout %{{.*}} {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>} : <256xf32>
tt.func @splat_restick_accepted(%desc: !tt.tensordesc<256xf32>) {
  tts.tensor_layout %desc
    {phys_src = array<i64: 0, 0>,
     phys_op = array<i64: 0, 3>,
     phys_arg = array<i64: 0, 64>} : !tt.tensordesc<256xf32>
  tt.return
}
