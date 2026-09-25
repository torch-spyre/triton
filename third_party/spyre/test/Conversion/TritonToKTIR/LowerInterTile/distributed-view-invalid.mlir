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

// A share that is not a load from a memory view. This is what an UNPINNED share
// looks like by the time this pass runs: PlacePinnedValues rewrites a pinned
// value's uses into a load from the buffer it built, so a share still arriving as
// a bare compute result was never pinned -- and nothing else says where it lives.
tt.func @unpinned_share(%x: tensor<64x32xf16>) -> tensor<64x32xf16> {
  %share = math.exp %x : tensor<64x32xf16>
  // expected-error @+1 {{cannot tell where the share lives: its value does not come from a ktdp.load of a memory view. Pin it with tl.spyre_pin, which is what supplies a partition's address}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  %mine = tt.descriptor_load %w[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  tt.return %mine : tensor<64x32xf16>
}

// -----
// A share in GLOBAL memory. It traces, so the previous rule passes, and it is
// still not composable: a redistribution is scratchpad on both sides by
// definition, and the partitions this would build would name device memory with a
// ct_id, which describes nothing. Reachable only from hand-written IR, since the
// pin that would place it admits ct_local alone.
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#share2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
tt.func @global_share(%ptr: !tt.ptr<f16>) -> tensor<64x32xf16> {
  %c0 = arith.constant 0 : index
  %base = builtin.unrealized_conversion_cast %ptr : !tt.ptr<f16> to index
  %v = ktdp.construct_memory_view %base, sizes: [64, 32], strides: [32, 1]
      {coordinate_set = #share2, memory_space = #ktdp.memory_space<global>}
      : memref<64x32xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id2, access_tile_set = #share2}
      : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>
  %share = ktdp.load %t : <64x32xindex> -> tensor<64x32xf16>
  // expected-error @+1 {{the share lives in 'global', and a redistribution composes scratchpad shares: pin it in 'ct_local'}}
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
tt.func @table_longer_than_grid() -> tensor<64x32xf16> {
  %c0 = arith.constant 0 : index
  %off = arith.constant 4096 : index
  %v = ktdp.construct_memory_view %off, sizes: [64, 32], strides: [32, 1]
      {coordinate_set = #share3, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x32xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id3, access_tile_set = #share3}
      : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>
  %share = ktdp.load %t : <64x32xindex> -> tensor<64x32xf16>
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
tt.func @broadcast_source() -> tensor<64x32xf16> {
  %c0 = arith.constant 0 : index
  %off = arith.constant 4096 : index
  %v = ktdp.construct_memory_view %off, sizes: [64, 32], strides: [32, 1]
      {coordinate_set = #share4, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x32xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id4, access_tile_set = #share4}
      : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>
  %share = ktdp.load %t : <64x32xindex> -> tensor<64x32xf16>
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
tt.func @store_through_the_view(%val: tensor<64x32xf16>) {
  %c0 = arith.constant 0 : index
  %off = arith.constant 4096 : index
  %v = ktdp.construct_memory_view %off, sizes: [64, 32], strides: [32, 1]
      {coordinate_set = #share5, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x32xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id5, access_tile_set = #share5}
      : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>
  %share = ktdp.load %t : <64x32xindex> -> tensor<64x32xf16>
  // expected-error @+1 {{a distributed descriptor may only be read: 'tt.descriptor_store' is not a tt.descriptor_load}}
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  tt.descriptor_store %w[%c0_i32, %c0_i32], %val : !tt.tensordesc<64x32xf16>, tensor<64x32xf16>
  tt.return
}
