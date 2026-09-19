// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for the tts.tensor_layout OP -- the authoring
// form, on a !tt.tensordesc.
//
// The assembly format is deliberately the same shape as tt.spyre_tensor_layout's
// (see spyre-triton-opt/spyre-tensor-layout-roundtrip.mlir) so the two are
// legible side by side while both exist. Two things about the printed form are
// worth pinning rather than assuming:
//
//   * the three coordinate arrays are inherent attributes printed through
//     `attr-dict`, so they come back in the printer's sorted order (phys_arg,
//     phys_op, phys_src) rather than the order written;
//   * the trailing type prints in the elided form `<512x1024xf32>`, since the
//     dialect prefix is implied by the operand's type constraint.

// ---------------------------------------------------------------------------
// (a) 2D tensor [M, N] stick-tiled on N with stick size 64:
//     physical layout [ceil(N/64), M, N%64]
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @stick_tiled_2d(
// CHECK-SAME:  %[[DESC:.*]]: !tt.tensordesc<512x1024xf32>) {
// CHECK: tts.tensor_layout %[[DESC]] {phys_arg = array<i64: 64, 0, 64>, phys_op = array<i64: 1, 0, 2>, phys_src = array<i64: 1, 0, 1>} : <512x1024xf32>
// CHECK: tt.return
tt.func @stick_tiled_2d(%desc: !tt.tensordesc<512x1024xf32>) {
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<512x1024xf32>
  tt.return
}

// ---------------------------------------------------------------------------
// (b) 1D tensor [N], identity layout (no tiling)
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @identity_1d(
// CHECK-SAME:  %[[DESC:.*]]: !tt.tensordesc<1024xf16>) {
// CHECK: tts.tensor_layout %[[DESC]] {phys_arg = array<i64: 0>, phys_op = array<i64: 0>, phys_src = array<i64: 0>} : <1024xf16>
// CHECK: tt.return
tt.func @identity_1d(%desc: !tt.tensordesc<1024xf16>) {
  tts.tensor_layout %desc
    {phys_src = array<i64: 0>,
     phys_op = array<i64: 0>,
     phys_arg = array<i64: 0>} : !tt.tensordesc<1024xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (c) The splat re-stick: a rank-1 logical result physicalized to (dim, lanes).
//     The pairing that distinguishes a splat from a stick split, kept here so
//     the round trip covers all four coord-op codes across the file.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @splat_restick_1d(
// CHECK-SAME:  %[[DESC:.*]]: !tt.tensordesc<256xf32>) {
// CHECK: tts.tensor_layout %[[DESC]] {phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, phys_src = array<i64: 0, 0>} : <256xf32>
// CHECK: tt.return
tt.func @splat_restick_1d(%desc: !tt.tensordesc<256xf32>) {
  tts.tensor_layout %desc
    {phys_src = array<i64: 0, 0>,
     phys_op = array<i64: 0, 3>,
     phys_arg = array<i64: 0, 64>} : !tt.tensordesc<256xf32>
  tt.return
}
