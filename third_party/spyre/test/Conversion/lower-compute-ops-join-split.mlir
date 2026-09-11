// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops | FileCheck %s

// Tests for --lower-compute-ops on tt.join and tt.split, the two composition ops.
//
// They are inverses and are tested together because their lowerings are mirror
// images, and reading them side by side is what makes each one's shape bookkeeping
// legible:
//
//   tt.join   takes two rank-N tensors and produces one rank-N+1 whose new
//             trailing dim has size 2. Lowering: tensor.expand_shape each operand
//             to a trailing size-1 dim, then one tensor.concat along that dim.
//   tt.split  takes one rank-N tensor whose trailing dim is size 2 and produces
//             two rank-N-1 tensors. Lowering: two tensor.extract_slice at trailing
//             offsets 0 and 1, then tensor.collapse_shape each to drop the now
//             size-1 dim.
//
// The interesting quantity in both is the dim index the operation acts on -- the
// `dim(N)` on the concat, the offset vectors on the slices -- since it is derived
// from operand rank rather than given by an attribute. Each op gets a rank-1 and
// a rank-2 case precisely so that index has to change between them.

// -----
// tt.join of two 1-D tensors into a 2-D result. Each tensor<8xf32> operand is
// expanded to tensor<8x1xf32>, then concatenated along dim 1.
//
// Triton source pattern:
//
//   real = tl.load(real_ptr + offsets)   # tensor<BLOCK x f32>
//   imag = tl.load(imag_ptr + offsets)   # tensor<BLOCK x f32>
//   pair = tl.join(real, imag)           # tensor<BLOCK x 2 x f32>

// CHECK-LABEL:   tt.func @join_1d(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8xf32>, %[[VAL_1:.*]]: tensor<8xf32>) -> tensor<8x2xf32> {
// CHECK:           %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] output_shape [8, 1] : tensor<8xf32> into tensor<8x1xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0, 1]] output_shape [8, 1] : tensor<8xf32> into tensor<8x1xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.concat dim(1) %[[VAL_2]], %[[VAL_3]] : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
// CHECK-NOT:       tt.join
// CHECK:           tt.return %[[VAL_4]] : tensor<8x2xf32>
// CHECK:         }
tt.func @join_1d(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8x2xf32> {
  %0 = tt.join %a, %b : tensor<8xf32> -> tensor<8x2xf32>
  tt.return %0 : tensor<8x2xf32>
}

// -----
// tt.join of two 2-D tensors into a 3-D result. The concat dim moves to 2 and the
// reassociation list gains a leading singleton group [[0], [1, 2]] -- the pair of
// changes that track operand rank. Fixing either at a constant would break this
// case while leaving join_1d green.

// CHECK-LABEL:   tt.func @join_2d(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8xf16>, %[[VAL_1:.*]]: tensor<4x8xf16>) -> tensor<4x8x2xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1, 2]] output_shape [4, 8, 1] : tensor<4x8xf16> into tensor<4x8x1xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0], [1, 2]] output_shape [4, 8, 1] : tensor<4x8xf16> into tensor<4x8x1xf16>
// CHECK:           %[[VAL_4:.*]] = tensor.concat dim(2) %[[VAL_2]], %[[VAL_3]] : (tensor<4x8x1xf16>, tensor<4x8x1xf16>) -> tensor<4x8x2xf16>
// CHECK-NOT:       tt.join
// CHECK:           tt.return %[[VAL_4]] : tensor<4x8x2xf16>
// CHECK:         }
tt.func @join_2d(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>) -> tensor<4x8x2xf16> {
  %0 = tt.join %a, %b : tensor<4x8xf16> -> tensor<4x8x2xf16>
  tt.return %0 : tensor<4x8x2xf16>
}

// -----
// tt.split of a 2-D tensor into two 1-D tensors. Two slices at trailing offsets
// [0, 0] and [0, 1], each of size [8, 1], then a collapse of the size-1 dim.
//
// The offsets are the whole claim: both slices have identical sizes and strides
// and differ only in the trailing offset, so a lowering that emitted offset 0
// twice would produce two copies of the first lane and still typecheck. Hence the
// two offset vectors are matched literally.
//
// Triton source pattern:
//
//   pair = tl.load(pair_desc, [pid * BLOCK])   # tensor<BLOCK x 2 x f32>
//   real, imag = tl.split(pair)                # two tensor<BLOCK x f32>

// CHECK-LABEL:   tt.func @split_2d(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8x2xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [8, 1] [1, 1] : tensor<8x2xf32> to tensor<8x1xf32>
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 1] [8, 1] [1, 1] : tensor<8x2xf32> to tensor<8x1xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0, 1]] : tensor<8x1xf32> into tensor<8xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1]] : tensor<8x1xf32> into tensor<8xf32>
// CHECK-NOT:       tt.split
// CHECK:           tt.return %[[VAL_3]], %[[VAL_4]] : tensor<8xf32>, tensor<8xf32>
// CHECK:         }
tt.func @split_2d(%t: tensor<8x2xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %0, %1 = tt.split %t : tensor<8x2xf32> -> tensor<8xf32>
  tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----
// tt.split of a 3-D tensor into two 2-D tensors. The offset, size and stride
// vectors all gain an entry and the differing offset moves to position 2 --
// the rank-derived index again, mirroring join_2d above.

// CHECK-LABEL:   tt.func @split_3d(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x8x2xf16>) -> (tensor<4x8xf16>, tensor<4x8xf16>) {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0] [4, 8, 1] [1, 1, 1] : tensor<4x8x2xf16> to tensor<4x8x1xf16>
// CHECK:           %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 1] [4, 8, 1] [1, 1, 1] : tensor<4x8x2xf16> to tensor<4x8x1xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0], [1, 2]] : tensor<4x8x1xf16> into tensor<4x8xf16>
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] : tensor<4x8x1xf16> into tensor<4x8xf16>
// CHECK-NOT:       tt.split
// CHECK:           tt.return %[[VAL_3]], %[[VAL_4]] : tensor<4x8xf16>, tensor<4x8xf16>
// CHECK:         }
tt.func @split_3d(%t: tensor<4x8x2xf16>) -> (tensor<4x8xf16>, tensor<4x8xf16>) {
  %0, %1 = tt.split %t : tensor<4x8x2xf16> -> tensor<4x8xf16>
  tt.return %0, %1 : tensor<4x8xf16>, tensor<4x8xf16>
}
