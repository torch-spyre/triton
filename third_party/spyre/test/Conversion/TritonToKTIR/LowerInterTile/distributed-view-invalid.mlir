// RUN: spyre-triton-opt %s --lower-inter-tile=grid=2 -split-input-file -verify-diagnostics

// What the compose mode refuses. Negative tests for the FIRST mode
// (tt.inter_tile_reduce) are in invalid.mlir beside this file.
//
// Each of these is a rule the OP cannot check, and the reason is the same in every
// case: the op sees its own attributes and nothing else, while these need the
// launch grid, or the IR the share came out of, or the ops that read the result.
// The op's own rules -- the table's form, and axes and block_shape agreeing with
// it -- are in Dialect/TTS/IR/distributed-descriptor-op-verifier.mlir.
//
// Note there are no rules of dashes anywhere in this file, since -split-input-file
// matches its marker as a substring.

#id = affine_map<(d0, d1) -> (d0, d1)>
#share = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>

// A share with no `tts.pin` marker among its users, which is exactly what an
// UNPINNED share looks like: nothing says where it lives, and a partition offset
// cannot be invented.
//
// An absent marker can only mean that. An earlier revision recovered the offset by
// walking `ktdp.load -> construct_access_tile -> construct_memory_view`, and a failed
// walk conflated "never pinned" with "pinned, but the IR shape was not one the walk
// recognized" -- which is why the message can now name the cause.
tt.func @unpinned_share(%x: tensor<64x32xf16>) -> tensor<64x32xf16> {
  %share = math.exp %x : tensor<64x32xf16>
  // expected-error @+1 {{cannot tell where the share lives: nothing pinned it. Pin it with tl.spyre_pin, which is what supplies a partition's offset}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  %mine = tt.descriptor_load %w[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  tt.return %mine : tensor<64x32xf16>
}

// -----
// More partitions than the grid has tiles. The holder is the partition's index in
// the table, which is only the right answer when the table has one entry per tile;
// resolving a shorter or longer one needs the tile coordinate table, which this op
// does not carry. Refused rather than guessed, because a guess would put data in a
// core that does not hold it and nothing downstream could tell.
#id3 = affine_map<(d0, d1) -> (d0, d1)>
#share3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
tt.func @table_longer_than_grid(%x: tensor<64x32xf16>) -> tensor<64x32xf16> {
  %share = math.exp %x : tensor<64x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 4096 : i32} : tensor<64x32xf16>
  // expected-error @+1 {{work_slices has 4 partitions for a grid of 2: only one partition per tile is supported}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  %mine = tt.descriptor_load %w[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  tt.return %mine : tensor<64x32xf16>
}

// -----
// Fewer partitions than tiles, which is the BROADCAST case -- a source held by one
// core and read by all of them. Refused by the same rule and worth its own case,
// because it is the shape a reader is most likely to reach for next: it is
// expressible in the design and needs the second table, not a different op.
#id4 = affine_map<(d0, d1) -> (d0, d1)>
#share4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
tt.func @broadcast_source(%x: tensor<64x32xf16>) -> tensor<64x32xf16> {
  %share = math.exp %x : tensor<64x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 4096 : i32} : tensor<64x32xf16>
  // expected-error @+1 {{work_slices has 1 partitions for a grid of 2}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  %mine = tt.descriptor_load %w[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  tt.return %mine : tensor<64x32xf16>
}

// -----
// A STORE through a distributed descriptor. Not forbidden by KTDP, and this design
// composes no destination view: under the pull model each destination holder writes
// into its own scratchpad, so there is nothing on that side to compose. Refused
// here so the restriction is the design's and visible, rather than a store that
// silently lowered to something nobody specified.
#id5 = affine_map<(d0, d1) -> (d0, d1)>
#share5 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
tt.func @store_through_the_view(%x: tensor<64x32xf16>, %val: tensor<64x32xf16>) {
  %share = math.exp %x : tensor<64x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 4096 : i32} : tensor<64x32xf16>
  // expected-error @+1 {{a distributed descriptor may only be read: 'tt.descriptor_store' is not a tt.descriptor_load}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  tt.descriptor_store %w[%c0_i32, %c0_i32], %val : !tt.tensordesc<64x32xf16>, tensor<64x32xf16>
  tt.return
}
