// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --rewrite-descriptor-layout -split-input-file | FileCheck %s

// Triton computes offsets in i32, so a pid-derived subscript arrives as
// index_cast(muli(index_cast(pid), c) : i32).  The coordinate split is emitted
// in `index`, and its input is lifted there too: the scheduler's symbolic
// start-address analysis treats a cast as opaque and rejects an address
// computed through one.  Checks are hand-written and minimal on purpose --
// the claim is about which domain the subscript arithmetic lands in, not about
// the whole module.
//
// This pass leaves the original i32 chain in place (dead, once nothing else
// reads it) and emits one index multiply per subscript; the canonicalizer and
// CSE that the frontend runs after DistributeWork collapse both.

// CHECK-LABEL:   tt.func @pid_offset_lifted_to_index(
// CHECK:           %[[PID:.*]] = tt.get_program_id x : i32
// The floordiv subscript: pid cast once, then multiplied in `index`.
// CHECK:           %[[PIDX:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:           %[[C64:.*]] = arith.constant 64 : index
// CHECK:           %[[OFF:.*]] = arith.muli %[[PIDX]], %[[C64]] : index
// Nothing casts back into the subscript between the multiply and the split:
// this is what fails on an unpatched build, where the i32 chain reached it.
// CHECK-NOT:       arith.index_cast
// CHECK:           arith.divsi %[[OFF]], %{{.*}} : index
// The mod subscript, same shape.
// CHECK:           %[[PIDX2:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:           %[[OFF2:.*]] = arith.muli %[[PIDX2]], %{{.*}} : index
// CHECK:           arith.remsi %[[OFF2]], %{{.*}} : index
tt.func @pid_offset_lifted_to_index(%ptr: !tt.ptr<f16>) {
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %pid = tt.get_program_id x : i32
  %off = arith.muli %pid, %c64_i32 : i32
  // [n=128] stick-on-n with stick_size=64 -> physical [n/64, n%64] = [2, 64]
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64xf16>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf16>
  %d = tt.descriptor_load %desc[%off] : !tt.tensordesc<64xf16> -> tensor<64xf16>
  tt.descriptor_store %desc[%off], %d : !tt.tensordesc<64xf16>, tensor<64xf16>
  tt.return
}

// -----

// A run-time i32 scalar is not a grid coordinate, so its arithmetic keeps the
// width Triton gave it -- rebuilding in 64-bit `index` would change what the
// expression means on overflow.  This is the path the dynamic-shape kernels
// take, and it must reach the split exactly as it did before.

// CHECK-LABEL:   tt.func @runtime_scalar_offset_unchanged(
// CHECK-SAME:      %{{.*}}: !tt.ptr<f16>, %[[N:.*]]: i32)
// CHECK:           %[[OFF:.*]] = arith.muli %[[N]], %{{.*}} : i32
// CHECK:           %[[OFFX:.*]] = arith.index_cast %[[OFF]] : i32 to index
// CHECK:           arith.divsi %[[OFFX]], %{{.*}} : index
// CHECK:           arith.remsi %[[OFFX]], %{{.*}} : index
tt.func @runtime_scalar_offset_unchanged(%ptr: !tt.ptr<f16>, %n: i32) {
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %off = arith.muli %n, %c64_i32 : i32
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64xf16>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf16>
  %d = tt.descriptor_load %desc[%off] : !tt.tensordesc<64xf16> -> tensor<64xf16>
  tt.descriptor_store %desc[%off], %d : !tt.tensordesc<64xf16>, tensor<64xf16>
  tt.return
}

// -----

// A truncation is not value-preserving: it discards the high bits, and that is
// the whole point of writing one.  Lifting the expression above the trunc would
// feed the *untruncated* 64-bit product to the subscript, addressing a
// different tile than the i32 expression named -- exactly the wraparound the
// rewrite must not silently undo.  So the subscript must keep reading the
// truncated i32 value through a single cast, with the wide multiply left where
// it was.

// CHECK-LABEL:   tt.func @trunc_not_lifted(
// CHECK:           %[[WIDE:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
// CHECK:           %[[TR:.*]] = arith.trunci %[[WIDE]] : i64 to i32
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[TR]] : i32 to index
// The multiply stays in i64 above the trunc; nothing re-multiplies in `index`.
// CHECK-NOT:       arith.muli %{{.*}} : index
// CHECK:           arith.divsi %[[IDX]], %{{.*}} : index
tt.func @trunc_not_lifted(%ptr: !tt.ptr<f16>) {
  %c64_i64 = arith.constant 64 : i64
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %pid = tt.get_program_id x : i32
  %pid64 = arith.extsi %pid : i32 to i64
  %big = arith.muli %pid64, %c64_i64 : i64
  %off = arith.trunci %big : i64 to i32
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64xf16>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf16>
  %d = tt.descriptor_load %desc[%off] : !tt.tensordesc<64xf16> -> tensor<64xf16>
  tt.descriptor_store %desc[%off], %d : !tt.tensordesc<64xf16>, tensor<64xf16>
  tt.return
}

// -----

// An unsigned widening feeding signed division cannot be lifted either: a
// zero-extended negative i32 is a large positive i64, so `divsi` on the wide
// value and `divsi` on the rebuilt narrow value disagree.  The whole chain
// must reach the subscript in the width it was written in.

// CHECK-LABEL:   tt.func @extui_into_signed_div_not_lifted(
// CHECK:           %[[NEG:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:           %[[W:.*]] = arith.extui %[[NEG]] : i32 to i64
// CHECK:           %[[D:.*]] = arith.divsi %[[W]], %{{.*}} : i64
// CHECK:           %[[TR:.*]] = arith.trunci %[[D]] : i64 to i32
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[TR]] : i32 to index
// No part of that chain is re-emitted in `index`.
// CHECK-NOT:       arith.muli %{{.*}} : index
// CHECK:           arith.divsi %[[IDX]], %{{.*}} : index
tt.func @extui_into_signed_div_not_lifted(%ptr: !tt.ptr<f16>) {
  %cneg = arith.constant -3 : i32
  %c64_i64 = arith.constant 64 : i64
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %pid = tt.get_program_id x : i32
  %neg = arith.muli %pid, %cneg : i32
  %wide = arith.extui %neg : i32 to i64
  %off64 = arith.divsi %wide, %c64_i64 : i64
  %off = arith.trunci %off64 : i64 to i32
  %desc = tt.make_tensor_descriptor %ptr, [%c128_i32], [%c1_i64]
      : !tt.ptr<f16>, !tt.tensordesc<64xf16>
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 0, 0>, phys_op = array<i64: 1, 2>, phys_arg = array<i64: 64, 64>} : !tt.tensordesc<64xf16>
  %d = tt.descriptor_load %desc[%off] : !tt.tensordesc<64xf16> -> tensor<64xf16>
  tt.descriptor_store %desc[%off], %d : !tt.tensordesc<64xf16>, tensor<64xf16>
  tt.return
}
