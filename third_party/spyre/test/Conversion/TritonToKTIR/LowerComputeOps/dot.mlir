// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops | FileCheck %s

// Tests for --lower-compute-ops on tt.dot.
//
// The lowering is a rank dispatch and nothing else: rank 2 becomes linalg.matmul,
// rank 3 becomes linalg.batch_matmul. Both take the tt.dot accumulator operand
// straight through as their outs operand -- no tensor.empty, no zero fill, because
// the accumulator already holds the values to add into. A lowering that
// materialized a fresh destination would drop the accumulator and silently
// compute a * b instead of a * b + c, so each case pins that outs names the
// third argument.
//
// The f16-input case is not repeated here: it is already pinned by
// Conversion/matmul.mlir, which runs the same pass on the same IR.
//
// Rank 4 and above is rejected, by the upstream Triton verifier rather than by
// this pass -- see lower-compute-ops-invalid.mlir.

// -----
// Rank-2 f32. All three operands and the result share the element type, and the
// contraction is 16x32 by 32x8 into 16x8.
//
// Triton source pattern:
//
//   acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
//   for k in range(k_tiles):
//       a   = a_desc.load([m * BM, k * BK])   # tensor<BM x BK x f32>
//       b   = b_desc.load([k * BK, n * BN])   # tensor<BK x BN x f32>
//       acc = tl.dot(a, b, acc)                # tensor<BM x BN x f32>

// CHECK-LABEL:   tt.func @dot_f32(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<16x32xf32>, %[[VAL_1:.*]]: tensor<32x8xf32>, %[[VAL_2:.*]]: tensor<16x8xf32>) -> tensor<16x8xf32> {
// The tensor.empty guard sits ABOVE the matmul, not below it. A lowering that
// materialized a fresh destination instead of threading the accumulator would
// emit its tensor.empty *before* the matmul that consumes it, and a guard placed
// after the matmul line never sees it -- the preceding CHECK simply scans past
// any intervening lines. Verified: with the guard below, output of the form
// "tensor.empty / linalg.matmul outs(%0)" passes; above, it fires.
// CHECK-NOT:       tensor.empty
// CHECK:           %[[VAL_3:.*]] = linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : tensor<16x32xf32>, tensor<32x8xf32>) outs(%[[VAL_2]] : tensor<16x8xf32>) -> tensor<16x8xf32>
// CHECK-NOT:       tt.dot
// CHECK:           tt.return %[[VAL_3]] : tensor<16x8xf32>
// CHECK:         }
tt.func @dot_f32(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>,
                 %c: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = tt.dot %a, %b, %c : tensor<16x32xf32> * tensor<32x8xf32> -> tensor<16x8xf32>
  tt.return %0 : tensor<16x8xf32>
}

// -----
// Rank-2 at realistic tile sizes, 128x64 by 64x128. Tile extents do not enter the
// lowering -- they are copied out of the operand types -- so this case exists to
// pin that nothing in the pattern is sensitive to magnitude, in particular that no
// size threshold quietly selects a different op.

// CHECK-LABEL:   tt.func @dot_large(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<128x64xf32>, %[[VAL_1:.*]]: tensor<64x128xf32>, %[[VAL_2:.*]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK:           %[[VAL_3:.*]] = linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : tensor<128x64xf32>, tensor<64x128xf32>) outs(%[[VAL_2]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-NOT:       tt.dot
// CHECK-NOT:       linalg.batch_matmul
// CHECK:           tt.return %[[VAL_3]] : tensor<128x128xf32>
// CHECK:         }
tt.func @dot_large(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>,
                   %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tt.dot %a, %b, %c : tensor<128x64xf32> * tensor<64x128xf32> -> tensor<128x128xf32>
  tt.return %0 : tensor<128x128xf32>
}

// -----
// Rank 3 takes the other branch: linalg.batch_matmul, with the leading dim 4 read
// as the batch. The guard against linalg.matmul is the real content here -- the
// rank dispatch is the whole pattern, and a fallthrough to the rank-2 branch would
// build a linalg.matmul on rank-3 operands.

// CHECK-LABEL:   tt.func @dot_batch_matmul(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x16x32xf32>, %[[VAL_1:.*]]: tensor<4x32x8xf32>, %[[VAL_2:.*]]: tensor<4x16x8xf32>) -> tensor<4x16x8xf32> {
// On the guard the prose above calls "the real content": a fallthrough to the
// rank-2 branch would emit linalg.matmul *instead of* linalg.batch_matmul, and the
// positive CHECK below already catches that -- verified, it fires on output where
// the batch_matmul is replaced by a matmul. So the rank dispatch is pinned; the
// CHECK-NOT below adds the narrower claim that no second, rank-2 matmul appears
// alongside the batch_matmul. A leading CHECK-NOT would sit in a zero-length
// window (nothing precedes the batch_matmul in real output) and add nothing.
// CHECK:           %[[VAL_3:.*]] = linalg.batch_matmul ins(%[[VAL_0]], %[[VAL_1]] : tensor<4x16x32xf32>, tensor<4x32x8xf32>) outs(%[[VAL_2]] : tensor<4x16x8xf32>) -> tensor<4x16x8xf32>
// CHECK-NOT:       tt.dot
// CHECK-NOT:       linalg.matmul
// CHECK:           tt.return %[[VAL_3]] : tensor<4x16x8xf32>
// CHECK:         }
tt.func @dot_batch_matmul(%a: tensor<4x16x32xf32>, %b: tensor<4x32x8xf32>,
                          %c: tensor<4x16x8xf32>) -> tensor<4x16x8xf32> {
  %0 = tt.dot %a, %b, %c : tensor<4x16x32xf32> * tensor<4x32x8xf32> -> tensor<4x16x8xf32>
  tt.return %0 : tensor<4x16x8xf32>
}
