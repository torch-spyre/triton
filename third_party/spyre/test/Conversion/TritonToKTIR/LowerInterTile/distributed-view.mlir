// RUN: spyre-triton-opt %s --lower-inter-tile=grid=2 -split-input-file | FileCheck %s

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
// loaded value, placed by PlacePinnedValues two passes earlier; these inputs are
// hand-written and stop at the load.
//
// Input is post-PlacePinnedValues IR: the share arrives as a ktdp.load from the
// ct_local view that pass built, which is where the lowering recovers the address
// from. Note there are no rules of dashes anywhere in this file, since
// -split-input-file matches its marker as a substring.

#id = affine_map<(d0, d1) -> (d0, d1)>
#share = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>

// Two partitions, divided on the last dimension. The two coordinate sets are the
// halves of the composed extent -- d1 in [0, 32) and d1 in [32, 64) -- and the
// composed memref is 64x64 where each share is 64x32.
// CHECK-LABEL: func @two_partitions
// CHECK: %[[OFF:.*]] = arith.constant 4096 : index
// CHECK: %[[SHARE:.*]] = ktdp.load
// CHECK: %[[P0:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [64, 32], strides: [32, 1] {coordinate_set = #[[SET0:.*]], memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<64x32xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
// CHECK: %[[P1:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [64, 32], strides: [32, 1] {coordinate_set = #[[SET1:.*]], memory_space = #ktdp.memory_space<ct_local, ct_id = 1>} : memref<64x32xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
// CHECK: %[[W:.*]] = ktdp.construct_distributed_memory_view(%[[P0]], %[[P1]] : {{.*}}) : memref<64x64xf16>
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile %[[W]]
// CHECK: ktdp.load %[[TILE]] : <64x32xindex> -> tensor<64x32xf16>
// CHECK-NOT: tts.make_distributed_descriptor
// CHECK-NOT: tt.descriptor_load
tt.func @two_partitions() -> tensor<64x32xf16> {
  %c0 = arith.constant 0 : index
  %off = arith.constant 4096 : index
  %v = ktdp.construct_memory_view %off, sizes: [64, 32], strides: [32, 1]
      {coordinate_set = #share, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x32xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id, access_tile_set = #share}
      : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>
  %share = ktdp.load %t : <64x32xindex> -> tensor<64x32xf16>
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
// CHECK: ktdp.construct_memory_view %{{.*}} {coordinate_set = #[[HI:.*]], memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
// CHECK: ktdp.construct_memory_view %{{.*}} {coordinate_set = #[[LO:.*]], memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
// CHECK-DAG: #[[HI]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 - 32 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[LO]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#share2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
tt.func @holder_is_the_index() -> tensor<64x32xf16> {
  %c0 = arith.constant 0 : index
  %off = arith.constant 4096 : index
  %v = ktdp.construct_memory_view %off, sizes: [64, 32], strides: [32, 1]
      {coordinate_set = #share2, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x32xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id2, access_tile_set = #share2}
      : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>
  %share = ktdp.load %t : <64x32xindex> -> tensor<64x32xf16>
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
#id3 = affine_map<(d0, d1) -> (d0, d1)>
#share3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
tt.func @two_reads(%o1: i32, %o2: i32) -> tensor<64x32xf16> {
  %c0 = arith.constant 0 : index
  %off = arith.constant 4096 : index
  %v = ktdp.construct_memory_view %off, sizes: [64, 32], strides: [32, 1]
      {coordinate_set = #share3, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x32xf16>
  %t = ktdp.construct_access_tile %v[%c0, %c0]
      {access_tile_order = #id3, access_tile_set = #share3}
      : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>
  %share = ktdp.load %t : <64x32xindex> -> tensor<64x32xf16>
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
