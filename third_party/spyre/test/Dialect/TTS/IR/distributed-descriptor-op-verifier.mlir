// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// The tts.make_distributed_descriptor OP's verifier. No pass runs here.
//
// Every rule below is answerable from the op alone: the table's form, and whether
// `axes` and `block_shape` agree with it and with the share. What the op cannot
// see is the launch grid -- so whether the table has one entry per TILE, which is
// how a holder is resolved, is the lowering's rule and is tested with it in
// Conversion/TritonToKTIR/LowerInterTile/.
//
// The table rules are `tts::readWorkSliceTable`'s, shared with that lowering, so
// each message below is the one a hand-written module and a generated one both
// get.
//
// `block_shape` is bounded by the COMPOSED extent and not by the share's, which is
// derived from the table and so is answerable here. Larger than the share is legal
// -- a gather spans partitions, an all-reduce takes the whole axis -- and the
// round-trip file's case (d) is the form these rules must not reject.
//
// Note there are no rules of dashes anywhere in this file, and no comment quotes
// the split marker: -split-input-file matches it as a substring.

// An empty table composes nothing, so there is no view to build.
tt.func @empty_table(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{work_slices must not be empty}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [], axes = ["", ""], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// Ragged keys. Every entry is one partition's grid coordinate, so a differing key
// set describes no grid at all.
tt.func @ragged_keys(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{work_slices[1] has keys [m], expected [n]: every entry describes the same grid, so the keys must be identical}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {m = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// A slice index is a position in a division, so it cannot be negative -- and the
// slice count is derived as one more than the largest index, which a negative one
// would make meaningless rather than merely wrong.
tt.func @negative_index(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{work_slices[0]['n'] is negative: -1}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = -1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// An entry must be a dictionary. Rejected before anything reads it as one, so the
// message names the table rather than reporting a cast failure.
tt.func @entry_not_a_dict(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{work_slices[0] must be a dictionary of dimension key to slice index}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [0 : i64], axes = ["", "n"],
       block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// `axes` is per tensor DIMENSION, not per key, so its length is the share's rank.
// A shorter list would be matched positionally against the dimensions and would
// silently shift which dimension each key divides.
tt.func @axes_too_short(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{axes has 1 entries for a rank-2 share: one per tensor dimension, with "" for a dimension the work was not divided on}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["n"],
       block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// A key `axes` names but the table does not carry divides nothing, because no
// entry says which slice of it this partition owns.
tt.func @axes_names_unknown_key(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{axes[1] names 'q', which no work_slices entry carries}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["", "q"],
       block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// And the converse, which is worse than a wasted key: a key the table carries and
// `axes` does not name divides no dimension, so every partition's region is the
// whole tensor and they COLLIDE rather than tile it.
tt.func @table_key_unnamed(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{work_slices carries key 'n', which axes does not name, so it divides no dimension}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["", ""],
       block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// One dimension per key. Two dimensions divided along one key would ask a single
// slice index to pick a region in both, which is a projection the table cannot
// express.
tt.func @key_names_two_dims(%share: tensor<32x32xf16>) {
  // expected-error @+1 {{axes names 'n' twice: a partition key divides one dimension}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["n", "n"],
       block_shape = array<i64: 32, 32>}
      : tensor<32x32xf16> -> !tt.tensordesc<32x32xf16>
  tt.return
}

// -----
// block_shape IS the result descriptor's block shape. Stated twice in the IR --
// once as an attribute the lowering reads, once in the type a reader sees -- so
// the two must agree or one of them is a lie.
tt.func @block_shape_mismatch(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{block_shape does not match the result descriptor's block shape}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["", "n"],
       block_shape = array<i64: 32, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// Rank, separately from extents: a rank mismatch is reported as one rather than
// as a shape mismatch, because the fix differs.
tt.func @block_shape_rank(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{block_shape has 3 entries but the result descriptor's block type is rank 2}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["", "n"],
       block_shape = array<i64: 64, 32, 1>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// -----
// The element type comes from the SHARE, since that is the value being composed;
// a descriptor over a different one describes memory nobody wrote.
tt.func @element_type_mismatch(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{the result descriptor's element type 'f32' does not match the share's 'f16'}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["", "n"],
       block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf32>
  tt.return
}

// -----
// A block past the COMPOSED extent, which is the only bound there is. Larger than
// the SHARE is legal and common -- a gather spans several partitions, and an
// all-reduce takes the whole composed axis -- so the bound cannot be the share's;
// see case (d) of the round-trip file for the form this rule must not reject.
//
// One partition on `n` here, so the composed extent is the share's 32 and a block
// of 64 asks for memory no partition holds. Nothing else would refuse it: Triton
// relates block_shape only to the descriptor's own block type, and
// ktdp.construct_access_tile's verifier checks ranks and maps but not extents
// against the view the tile is taken on.
tt.func @block_past_the_composed_extent(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{block_shape[1] is 64, larger than the composed extent 32 on that dimension (the share's 32 times 1 slices)}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["", "n"],
       block_shape = array<i64: 64, 64>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x64xf16>
  tt.return
}

// -----
// And on an UNDIVIDED dimension, where the composed extent IS the share's -- so the
// rule applies to every dimension and not only the ones a key names.
tt.func @block_past_an_undivided_dim(%share: tensor<64x32xf16>) {
  // expected-error @+1 {{block_shape[0] is 128, larger than the composed extent 64 on that dimension}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}], axes = ["", "n"],
       block_shape = array<i64: 128, 64>}
      : tensor<64x32xf16> -> !tt.tensordesc<128x64xf16>
  tt.return
}

// -----
// A dynamic extent cannot be composed: the slice width is the share's extent, so
// there is no region to give any partition. Refused by the operand's type
// constraint rather than by a hand-written rule, as `tts.pin` is.
tt.func @dynamic_share(%share: tensor<?x32xf16>) {
  // expected-error @+1 {{operand #0 must be statically shaped tensor of any type values}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}], axes = ["", "n"],
       block_shape = array<i64: 64, 32>}
      : tensor<?x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}
