// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout > %t.once
// RUN: spyre-triton-opt %t.once --rewrite-descriptor-layout > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

// Phase 2 runs under a greedy driver that re-enqueues an op whenever a
// neighbour it feeds or consumes is rewritten, so ops are visited repeatedly
// until a fixpoint. Every pattern's match condition must be falsified by its own
// rewrite. Running the pass on its own output must therefore be a no-op, which
// the diff above asserts.
//
// The reduce here retypes in place rather than changing rank, so it is the case
// that does not falsify its precondition for free: its guard is what this pins.

// CHECK-LABEL: tt.func @idempotent_reduce
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       linalg.reduce
// CHECK-SAME:  dimensions = [0, 2]
// CHECK-NOT:   scf.for
// CHECK-NOT:   tt.spyre_tensor_layout
// CHECK:       tt.return
module {
tt.func @idempotent_reduce(%a_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c128_i64 = arith.constant 128 : i64
  %c1_i64 = arith.constant 1 : i64

  // A[64, 128] stick-on-N(64): phys [N/64, M, N%64] = [2, 64, 64]
  %a_desc = tt.make_tensor_descriptor %a_ptr, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %a_desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %a = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>

  %c_desc = tt.make_tensor_descriptor %c_ptr, [%c64_i32], [%c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64xf32>
  %c = tt.descriptor_load %c_desc[%c0_i32] : !tt.tensordesc<64xf32> -> tensor<64xf32>

  %r = "tt.reduce"(%a) ({
  ^bb0(%x: f32, %y: f32):
    %s = arith.addf %x, %y : f32
    tt.reduce.return %s : f32
  }) {axis = 1 : i32} : (tensor<64x128xf32>) -> tensor<64xf32>

  tt.descriptor_store %c_desc[%c0_i32], %r : !tt.tensordesc<64xf32>, tensor<64xf32>
  tt.return
}
}
