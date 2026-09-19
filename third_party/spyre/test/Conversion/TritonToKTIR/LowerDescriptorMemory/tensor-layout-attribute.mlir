// RUN: spyre-triton-opt %s -split-input-file --lower-descriptor-memory | FileCheck %s

// Tests for --lower-descriptor-memory as the WRITER of the tts.tensor_layout
// attribute.
//
// A descriptor carrying a `tts.tensor_layout` op has three things happen to it,
// and all three are the point:
//
//   1. the three coordinate arrays land on the ktdp.construct_memory_view built
//      for that descriptor, as the `tts.tensor_layout` attribute;
//   2. the marker op is erased;
//   3. so is the builtin.unrealized_conversion_cast bridging the view's memref
//      back to !tt.tensordesc.
//
// (3) is the one worth asserting negatively, and every case below does. That
// cast exists only to keep a !tt.tensordesc operand alive, and while it is the
// pass's own intra-pass handoff -- the access-op patterns reach the memref
// *through* it -- nothing wants it once the marker that needed the type is gone.
// A writer that annotated the view but left the bridge would look correct in
// every positive CHECK here.
//
// The absence guards sit between the memory view and the tt.return, inside the
// function body, so they are not in the module's empty tail where they could
// never fire. The cast and the marker are emitted in exactly that gap when they
// survive -- see tensor-layout-both-paths.mlir, where one descriptor does keep
// both.
//
// The surviving `!tt.ptr -> index` cast is a different cast with a different
// lifetime: getBasePtrAsIndex builds it, and ConvertFunctions consumes it later.
// So the guards name the tensordesc direction specifically rather than banning
// unrealized_conversion_cast outright.
//
// The attribute's entries come back in the printer's sorted order (phys_arg,
// phys_op, phys_src) rather than the order the op spells them.

// -----
// Static shapes. The view's sizes are the FULL tensor extent (128x256) while the
// descriptor's block type is the tile (128x64) -- the layout describes the
// former, and the attribute verifier measures phys_src against the view's rank.
tt.func @static_2d(%ptr: !tt.ptr<f16>) -> tensor<128x64xf16> {
  %m = arith.constant 128 : i32
  %n = arith.constant 256 : i32
  %sr = arith.constant 256 : i64
  %sc = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %ptr, [%m, %n], [%sr, %sc]
      : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf16>
  %c0 = arith.constant 0 : i32
  %d = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
  tt.return %d : tensor<128x64xf16>
}

// CHECK-LABEL: tt.func @static_2d
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f16> to index
// CHECK: %[[VIEW:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [128, 256], strides: [256, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<128x256xf16>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc
// CHECK: ktdp.construct_access_tile %[[VIEW]]
// CHECK: ktdp.load
// CHECK: tt.return

// -----
// Dynamic shapes. Same three outcomes; the layout is independent of whether the
// extents are constants, since it names logical DIMS and not their sizes.
tt.func @dynamic_2d(%ptr: !tt.ptr<f16>, %M: i32, %N: i32, %sr: i64, %sc: i64)
    -> tensor<128x64xf16> {
  %desc = tt.make_tensor_descriptor %ptr, [%M, %N], [%sr, %sc]
      : !tt.ptr<f16>, !tt.tensordesc<128x64xf16>
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<128x64xf16>
  %c0 = arith.constant 0 : i32
  %d = tt.descriptor_load %desc[%c0, %c0] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
  tt.return %d : tensor<128x64xf16>
}

// CHECK-LABEL: tt.func @dynamic_2d
// CHECK: %[[BASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f16> to index
// CHECK: %[[VIEW:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: {{\[}}%{{.*}}, %{{.*}}], strides: {{\[}}%{{.*}}, %{{.*}}] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<?x?xf16>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc
// CHECK: ktdp.construct_access_tile %[[VIEW]]
// CHECK: ktdp.load
// CHECK: tt.return

// -----
// Two annotated descriptors in one kernel, with DIFFERENT layouts: the input is
// stick-tiled on dim 1, the output is identity. Each attribute has to land on
// its own view, which a writer keying off "the kernel's layout" rather than off
// the descriptor would get wrong -- and would get wrong invisibly, since both
// views would still verify.
tt.func @two_descriptors(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %m = arith.constant 64 : i32
  %n = arith.constant 128 : i32
  %sr = arith.constant 128 : i64
  %sc = arith.constant 1 : i64
  %c0 = arith.constant 0 : i32
  %src = tt.make_tensor_descriptor %in, [%m, %n], [%sr, %sc]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tts.tensor_layout %src
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %dst = tt.make_tensor_descriptor %out, [%m, %n], [%sr, %sc]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tts.tensor_layout %dst
    {phys_src = array<i64: 0, 1>,
     phys_op = array<i64: 0, 0>,
     phys_arg = array<i64: 0, 0>} : !tt.tensordesc<64x128xf32>
  %d = tt.descriptor_load %src[%c0, %c0] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  tt.descriptor_store %dst[%c0, %c0], %d : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}

// CHECK-LABEL: tt.func @two_descriptors
// CHECK: %[[INBASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// CHECK: %[[INVIEW:.*]] = ktdp.construct_memory_view %[[INBASE]], sizes: [64, 128], strides: [128, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>}} : memref<64x128xf32>
// CHECK: %[[OUTBASE:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !tt.ptr<f32> to index
// CHECK: %[[OUTVIEW:.*]] = ktdp.construct_memory_view %[[OUTBASE]], sizes: [64, 128], strides: [128, 1] {coordinate_set = {{.*}}, memory_space = #ktdp.memory_space<global>, tts.tensor_layout = {phys_arg = array<i64: 0, 0>, phys_op = array<i64: 0, 0>, phys_src = array<i64: 0, 1>}} : memref<64x128xf32>
// CHECK-NOT: tts.tensor_layout %
// CHECK-NOT: to !tt.tensordesc
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile %[[INVIEW]]
// CHECK: %[[VAL:.*]] = ktdp.load %[[TILE]]
// CHECK: %[[OUTTILE:.*]] = ktdp.construct_access_tile %[[OUTVIEW]]
// CHECK: ktdp.store %[[VAL]], %[[OUTTILE]]
// CHECK: tt.return
