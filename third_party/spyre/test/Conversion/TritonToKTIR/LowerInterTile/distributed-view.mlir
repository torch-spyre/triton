// RUN: spyre-triton-opt %s --lower-inter-tile=grid=2 -split-input-file | FileCheck %s
// RUN: spyre-triton-opt %s --lower-inter-tile=grid=2 -split-input-file --mlir-print-local-scope | FileCheck %s --check-prefix=INLINE

// The pass's SECOND mode: tts.make_distributed_descriptor becoming a composed
// view, and the reads through it becoming transfers. The first mode
// (tt.inter_tile_reduce) is in basic.mlir beside this file.
//
// This is design section 3's three phases, and each case below is about one thing
// the lowering has to get right:
//
//   phase 1  one ktdp.construct_memory_view per partition, differing ONLY in
//            coordinate_set and ct_id -- offsets, sizes and strides identical --
//            composed by ktdp.construct_distributed_memory_view whose result
//            extent is the union;
//   phase 2  a ktdp.construct_access_tile at the offsets this instance passed to
//            .load(), on the COMPOSED view;
//   phase 3  one ktdp.load, which IS the transfer.
//
// The identical-offsets property is the one worth asserting hardest, and every
// case does it by binding %OFF once and matching it in both views: a partition
// lives at the same offset in its own core's scratchpad as every other partition
// does in its, because every core runs the same program text. What differs is
// which core, and that is the ct_id.
//
// The landing store a received tile needs is NOT here. It is a tl.spyre_pin on the
// loaded value, and MaterializePinnedBuffers builds it a stage later; these inputs
// are hand-written and stop at the load.
//
// Input is what the pipeline hands this pass: the share is a plain value with the
// `tts.pin` MARKER still standing among its users, because LowerTTSMarkers runs at
// the END of the `ktir` stage -- after this pass. So the offset and the space are
// read off the marker, not off a buffer and not off the attribute.
//
// That order is also what keeps the share alive. This pass erases the compose, the
// share's only other use, so the marker is the only thing holding the value until
// MaterializePinnedBuffers honours it; converting it before this stage's closing DCE
// would let that DCE delete the share. Note there are no rules of dashes anywhere in
// this file, since -split-input-file matches its marker as a substring.


// Two partitions, divided on the last dimension. The two coordinate sets are the
// halves of the composed extent -- d1 in [0, 32) and d1 in [32, 64) -- and the
// composed memref is 64x64 where each share is 64x32.
// CHECK-LABEL: func @two_partitions
// The offset is built by the pass from the pin's offset, not borrowed from the
// input, which is what an attribute offset buys: nothing has to be in scope.
// CHECK: %[[OFF:.*]] = arith.constant 4096 : index
// CHECK: %[[P0:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [64, 32], strides: [32, 1] {coordinate_set = #[[SET0:.*]], memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<64x32xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
// CHECK: %[[P1:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [64, 32], strides: [32, 1] {coordinate_set = #[[SET1:.*]], memory_space = #ktdp.memory_space<ct_local, ct_id = 1>} : memref<64x32xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
// CHECK: %[[W:.*]] = ktdp.construct_distributed_memory_view(%[[P0]], %[[P1]] : {{.*}}) : memref<64x64xf16>
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile %[[W]]
// CHECK: ktdp.load %[[TILE]] : <64x32xindex> -> tensor<64x32xf16>
// CHECK-NOT: tts.make_distributed_descriptor
// CHECK-NOT: tt.descriptor_load
tt.func @two_partitions(%x: tensor<64x32xf16>) -> tensor<64x32xf16> {
  %share = math.exp %x : tensor<64x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 4096 : i32} : tensor<64x32xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  %mine = tt.descriptor_load %w[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  tt.return %mine : tensor<64x32xf16>
}

// -----
// The table's ORDER is the holder, not its contents. Here partition 0 owns the
// SECOND slice, so ct_id 0 carries the set starting at 32 -- which a lowering
// deriving the holder from the slice index instead would get exactly backwards,
// and would get backwards invisibly, since both views verify either way. This is
// the case that pins "holder = the partition's index in the table".
// CHECK-LABEL: func @holder_is_the_index
// CHECK: ktdp.construct_memory_view %{{.*}} memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
// CHECK: ktdp.construct_memory_view %{{.*}} memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
//
// Checked on the INLINE run, because this is the one claim in the file that ties a
// ct_id to a particular coordinate SET, and through an alias it cannot be: the alias
// definitions print before the function, so a CHECK-DAG after the label can never
// reach them. Printed inline, the pairing is one line and the assertion is direct.
// INLINE-LABEL: func @holder_is_the_index
// INLINE: coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 - 32 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
// INLINE: coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
tt.func @holder_is_the_index(%x: tensor<64x32xf16>) -> tensor<64x32xf16> {
  %share = math.exp %x : tensor<64x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 4096 : i32} : tensor<64x32xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 1 : i64}, {n = 0 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  %mine = tt.descriptor_load %w[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  tt.return %mine : tensor<64x32xf16>
}

// -----
// Two reads through ONE composed view: a tile and a load each, and the view built
// once. Asserted because a lowering that composed per read would emit twice the
// partitions and, on a real device, twice the addressing for the same data.
// CHECK-LABEL: func @two_reads
// CHECK: ktdp.construct_memory_view %{{.*}} memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
// CHECK: ktdp.construct_memory_view %{{.*}} memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
// CHECK: %[[W:.*]] = ktdp.construct_distributed_memory_view
// CHECK-NOT: ktdp.construct_distributed_memory_view
// CHECK: %[[T1:.*]] = ktdp.construct_access_tile %[[W]]
// CHECK: ktdp.load %[[T1]]
// CHECK: %[[T2:.*]] = ktdp.construct_access_tile %[[W]]
// CHECK: ktdp.load %[[T2]]
tt.func @two_reads(%x: tensor<64x32xf16>, %o1: i32, %o2: i32) -> tensor<64x32xf16> {
  %share = math.exp %x : tensor<64x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 4096 : i32} : tensor<64x32xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %c0_i32 = arith.constant 0 : i32
  %a = tt.descriptor_load %w[%c0_i32, %o1] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  %b = tt.descriptor_load %w[%c0_i32, %o2] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  %sum = arith.addf %a, %b : tensor<64x32xf16>
  tt.return %sum : tensor<64x32xf16>
}
