// RUN: spyre-triton-opt %s --lower-descriptor-memory -split-input-file | FileCheck %s

// A read through a DISTRIBUTED descriptor passes this pass untouched.
//
// This is the one thing tts.make_distributed_descriptor asks of a pass other than
// its own lowering, and it would otherwise be invisible: descriptor_load is
// illegal here, so without an exemption a kernel that composes would fail two
// passes before the compose was ever reached.
//
// Why the exemption rather than lowering it here: the view such a read goes
// through is composed by LowerInterTile, five passes later, because a partition's
// address comes from the buffer PlacePinnedValues builds and that pass has not run
// when this one does. So the read travels with its descriptor and is lowered
// alongside the compose it belongs to.
//
// Two places have to agree about which reads are exempt -- the precondition walk
// and the conversion target -- and they share one predicate. The pair below is what
// keeps that true: an ordinary descriptor_load in the same function is still
// lowered, so the exemption is narrow rather than a blanket legality.
//
// Note there are no rules of dashes anywhere in this file, since -split-input-file
// matches its marker as a substring.

// CHECK-LABEL: tt.func @distributed_read_survives
// The ordinary descriptor becomes a memory view and its read becomes a ktdp.load.
// CHECK: ktdp.construct_memory_view
// CHECK: ktdp.load
// The composed one, and its read, are left exactly as they were.
// CHECK: %[[W:.*]] = tts.make_distributed_descriptor
// CHECK: tt.descriptor_load %[[W]]
tt.func @distributed_read_survives(%ptr: !tt.ptr<f16>) -> tensor<64x32xf16> {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  %c32_i64 = arith.constant 32 : i64
  %c1_i64 = arith.constant 1 : i64
  %d = tt.make_tensor_descriptor %ptr, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64]
      : <f16>, <64x32xf16>
  %share = tt.descriptor_load %d[%c0_i32, %c0_i32]
      : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  %mine = tt.descriptor_load %w[%c0_i32, %c0_i32]
      : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
  tt.return %mine : tensor<64x32xf16>
}
