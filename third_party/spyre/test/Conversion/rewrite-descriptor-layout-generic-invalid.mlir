// RUN: spyre-triton-opt %s --lower-descriptor-memory --lower-scalar-load --lower-compute-ops --convert-elementwise-to-linalg --unalias-linalg-outs --rewrite-descriptor-layout-generic -split-input-file -verify-diagnostics

// Diagnostics reachable from Triton-level input, i.e. through the same pass
// pipeline the backend runs. Diagnostics that only a hand-crafted post-lowering
// module can reach live in rewrite-descriptor-layout-generic-invalid-ktir.mlir.

// Case 1 -- a stick dim narrower than one stick.
//
// The mod dim's extent IS the stick width, so a block whose logical extent on
// the split dim is below that width would give a physical dim wider than the
// data it indexes.
module {
tt.func @block_smaller_than_stick(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c32_i32 = arith.constant 32 : i32
  %c32_i64 = arith.constant 32 : i64
  %c1_i64 = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %ptr, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x32xf32>
  // Stick-on-N at width 64, but N is only 32.
  tt.spyre_tensor_layout %desc {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x32xf32>
  // expected-error @below {{spyre_tensor_layout: block extent of stick dim (32) is smaller than the stick size (64); a stick dim cannot be sub-stick}}
  %d = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf32> -> tensor<64x32xf32>
  %out_desc = tt.make_tensor_descriptor %out, [%c64_i32, %c32_i32], [%c32_i64, %c1_i64]
      : !tt.ptr<f32>, !tt.tensordesc<64x32xf32>
  tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %d : !tt.tensordesc<64x32xf32>, tensor<64x32xf32>
  tt.return
}
}

// -----

// Case 2 -- two operands split the same loop dim at different widths.
//
// The rebuilt domain gives a split dim a (stick, elem) pair with ONE width, and
// an operand holding that dim whole addresses it as `stick * width + elem`. Two
// widths on one loop dim leave no such composite, and either choice would
// address the wrong elements of the third operand.
module {
tt.func @two_stick_widths_on_one_dim(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c128_i64 = arith.constant 128 : i64
  %ad = tt.make_tensor_descriptor %a, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.spyre_tensor_layout %ad {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x128xf32>
  %av = tt.descriptor_load %ad[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  %bd = tt.make_tensor_descriptor %b, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  // Same logical dim, same role, width 32 rather than 64.
  tt.spyre_tensor_layout %bd {phys_src = array<i64: 1, 0, 1>, phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>} : !tt.tensordesc<64x128xf32>
  %bv = tt.descriptor_load %bd[%c0_i32, %c0_i32] : !tt.tensordesc<64x128xf32> -> tensor<64x128xf32>
  // expected-error @below {{loop dim 1 is split at two different stick widths (64 and 32), so no single composite addresses it}}
  %s = arith.addf %av, %bv : tensor<64x128xf32>
  %cd = tt.make_tensor_descriptor %c, [%c64_i32, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<64x128xf32>
  tt.descriptor_store %cd[%c0_i32, %c0_i32], %s : !tt.tensordesc<64x128xf32>, tensor<64x128xf32>
  tt.return
}
}
