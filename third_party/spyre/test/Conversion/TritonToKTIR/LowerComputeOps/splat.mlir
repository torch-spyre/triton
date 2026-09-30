// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops | FileCheck %s

// Tests for --lower-compute-ops on tt.splat.
//
// A tt.splat carries a scalar and a result tensor type; there is no operand to
// reshape, so the lowering materializes the destination itself: tensor.empty for
// the shape, then linalg.fill to write the scalar into every element. The empty
// is what makes the result type explicit in the output -- the fill inherits it
// from its outs operand -- so both ops are pinned per case.
//
// Rank is the only axis of variation the pattern has (the element type is copied
// through untouched), so the three cases below are rank 1, 2 and 3. Each pins
// the shape on the tensor.empty as well as the fill, because a pattern that read
// the operand type instead of the result type would still emit a well-formed
// fill -- just at the wrong shape.
//
// Note the pass runs cleanupDeadOps, so a tt.splat whose result is unused is
// simply erased and the output is empty. Every kernel here returns its value.

// -----
// Rank 1, f32. The scalar is broadcast to fill all 1024 elements.
//
// Triton source pattern:
//
//   scalar = 1.0
//   tensor = tl.broadcast(scalar, shape=[BLOCK_SIZE])   # tl.splat
//
// The absence guard sits between the fill and the return so it is inside the
// function body, not in the module's empty tail where it would match vacuously.

// CHECK-LABEL:   tt.func @splat_f32_1d(
// CHECK-SAME:  %[[VAL_0:.*]]: f32) -> tensor<1024xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.fill ins(%[[VAL_0]] : f32) outs(%[[VAL_1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-NOT:       tt.splat
// CHECK:           tt.return %[[VAL_2]] : tensor<1024xf32>
// CHECK:         }
tt.func @splat_f32_1d(%s: f32) -> tensor<1024xf32> {
  %0 = tt.splat %s : f32 -> tensor<1024xf32>
  tt.return %0 : tensor<1024xf32>
}

// -----
// Rank 2, i32. Integer element types take the same path as float -- linalg.fill
// is type-agnostic -- and the 2-D shape appears on both the empty and the fill.

// CHECK-LABEL:   tt.func @splat_i32_2d(
// CHECK-SAME:  %[[VAL_0:.*]]: i32) -> tensor<4x8xi32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x8xi32>
// CHECK:           %[[VAL_2:.*]] = linalg.fill ins(%[[VAL_0]] : i32) outs(%[[VAL_1]] : tensor<4x8xi32>) -> tensor<4x8xi32>
// CHECK-NOT:       tt.splat
// CHECK:           tt.return %[[VAL_2]] : tensor<4x8xi32>
// CHECK:         }
tt.func @splat_i32_2d(%s: i32) -> tensor<4x8xi32> {
  %0 = tt.splat %s : i32 -> tensor<4x8xi32>
  tt.return %0 : tensor<4x8xi32>
}

// -----
// Rank 3, f16. Rank beyond 2 needs no extra machinery: the destination shape is
// read wholesale off the result type, so a 3-D splat is one tensor.empty and one
// fill exactly as the 1-D case is.

// CHECK-LABEL:   tt.func @splat_f16_3d(
// CHECK-SAME:  %[[VAL_0:.*]]: f16) -> tensor<2x4x8xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<2x4x8xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.fill ins(%[[VAL_0]] : f16) outs(%[[VAL_1]] : tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
// CHECK-NOT:       tt.splat
// CHECK:           tt.return %[[VAL_2]] : tensor<2x4x8xf16>
// CHECK:         }
tt.func @splat_f16_3d(%s: f16) -> tensor<2x4x8xf16> {
  %0 = tt.splat %s : f16 -> tensor<2x4x8xf16>
  tt.return %0 : tensor<2x4x8xf16>
}
