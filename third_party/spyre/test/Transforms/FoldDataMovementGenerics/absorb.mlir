// RUN: spyre-triton-opt %s --fold-data-movement-generics -split-input-file | FileCheck %s

// ABSORPTION: a tensor shape op in front of a generic, rewritten into that
// generic's operand map. Collapse and expand, the positives and the declines, all
// in one file -- the two directions are one mechanism read forwards and
// backwards, and keeping them apart hid that.
//
// Three separable pieces:
//
//   resultToSourceMap    one shape op's RESULT coords -> its SOURCE coords,
//                        knowing nothing about consumers
//   AbsorbCoordinateOp   compose that with the consumer's operand map and repoint
//                        the operand at the source
//   the projection check on the COMPOSED map: every result a bare loop dim or a
//                        constant
//
// Every decline here is SILENT. No annotated memory view appears in this file, so
// nothing is on a path the layout pass physicalizes; gate.mlir is the same ops
// with a path.

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, 0)>

// Test 1: the reproducer's shape -- a unit-dim tensor.collapse_shape in front of
// the broadcast, which is what tt.broadcast's lowering emits because
// linalg.broadcast takes its input rank-reduced. Both fold: the consumer ends up
// reading the rank-2 statistic at a CONSTANT lane, (d0, d1) -> (d0, 0), which is
// the form hand-written reference KTIR states. Leaving the reshape behind is not
// enough -- the consumer would read a rank-1 value that no longer matches the
// rank-2 access tile its load came from, and the scheduler refuses that.
module {
// CHECK-LABEL:   func.func @unit_dim_collapse_and_broadcast(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>, %[[VAL_1:.*]]: tensor<64x1xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64x1xf16>) outs(%[[VAL_2]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.subf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           return %[[VAL_3]] : tensor<64x128xf16>
// CHECK:         }
func.func @unit_dim_collapse_and_broadcast(%x: tensor<64x128xf16>, %s: tensor<64x1xf16>)
    -> tensor<64x128xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<64x1xf16> into tensor<64xf16>
  %e = tensor.empty() : tensor<64x128xf16>
  %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%c : tensor<64xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64x128xf16>
  %o = tensor.empty() : tensor<64x128xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x, %b : tensor<64x128xf16>, tensor<64x128xf16>) outs(%o : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b2: f16, %out: f16):
    %d = arith.subf %a, %b2 : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  return %r : tensor<64x128xf16>
}
}

// -----

// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1) -> (d0)>

// Test 2: a unit-dim `tensor.expand_shape` -- the collapse's mirror image, and
// the real softmax_2pass shape: a rank-1 reduce result expanded to rank 2 to meet
// a loop-carried rank-2 accumulator. Its reassociation reads SOURCE dim -> RESULT
// dims, the opposite of the collapse's, so each source coordinate is a
// LINEARIZATION of its group rather than a delinearization of one -- and with a
// single non-unit result dim that linearization is just that dim. The consumer
// ends up reading the rank-1 value at (d0, d1) -> (d0).
module {
// CHECK-LABEL:   func.func @unit_dim_expand_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x1xf32>, %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4x1xf32> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<4x1xf32>, tensor<4xf32>) outs(%[[VAL_2]] : tensor<4x1xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.maxnumf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           } -> tensor<4x1xf32>
// CHECK:           return %[[VAL_3]] : tensor<4x1xf32>
// CHECK:         }
func.func @unit_dim_expand_absorbed(%acc: tensor<4x1xf32>, %s: tensor<4xf32>)
    -> tensor<4x1xf32> {
  %x = tensor.expand_shape %s [[0, 1]] output_shape [4, 1]
      : tensor<4xf32> into tensor<4x1xf32>
  %e = tensor.empty() : tensor<4x1xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%acc, %x : tensor<4x1xf32>, tensor<4x1xf32>) outs(%e : tensor<4x1xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %v = arith.maxnumf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<4x1xf32>
  return %r : tensor<4x1xf32>
}
}

// -----

// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1) -> (d1, 0)>
// CHECK: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 3: a CHAIN of two shape ops on one operand, one of each kind, absorbed one
// after the other into the same map. `[64,1] -> [64]` collapses and `[64] ->
// [1,64]` expands, and the consumer reading the rank-2 result at the identity
// ends up reading the original `tensor<64x1xf16>` at (d0, d1) -> (d1, 0). This is
// what makes the three pieces separable rather than one hardcoded case: neither
// op knows the other is there, and the greedy driver may take them in either
// order.
module {
// CHECK-LABEL:   func.func @chained_collapse_and_expand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x1xf16>, %[[VAL_1:.*]]: tensor<1x64xf16>) -> tensor<1x64xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP4]], #[[$MAP5]], #[[$MAP5]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x1xf16>, tensor<1x64xf16>) outs(%[[VAL_2]] : tensor<1x64xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<1x64xf16>
// CHECK:           return %[[VAL_3]] : tensor<1x64xf16>
// CHECK:         }
func.func @chained_collapse_and_expand(%s: tensor<64x1xf16>, %y: tensor<1x64xf16>)
    -> tensor<1x64xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<64x1xf16> into tensor<64xf16>
  %x = tensor.expand_shape %c [[0, 1]] output_shape [1, 64]
      : tensor<64xf16> into tensor<1x64xf16>
  %e = tensor.empty() : tensor<1x64xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x, %y : tensor<1x64xf16>, tensor<1x64xf16>) outs(%e : tensor<1x64xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.mulf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<1x64xf16>
  return %r : tensor<1x64xf16>
}
}

// -----

// CHECK: #[[$MAP6:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP7:.+]] = affine_map<(d0, d1) -> (0, 0)>

// Test 4: the case that proves the projection check belongs on the COMPOSED map
// and not on the reassociation. `[64,2] -> [128]` fuses two non-unit dims, so its
// result-to-source map is `(d0) -> (d0 floordiv 2, d0 mod 2)` -- a genuine
// linearization, which this pass emits rather than refusing to state. But the
// consumer reads the collapsed value at a CONSTANT coordinate, so the composite
// is `(d0, d1) -> (0, 0)`: the floordiv and the mod fold away against the
// constant and nothing is left for a loop IV to be projected through. Absorbed.
// A check on the reassociation would have declined it.
module {
// CHECK-LABEL:   func.func @linearizing_collapse_at_a_constant(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x2xf16>, %[[VAL_1:.*]]: tensor<8x16xf16>) -> tensor<8x16xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8x16xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP6]], #[[$MAP7]], #[[$MAP6]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]], %[[VAL_0]] : tensor<8x16xf16>, tensor<64x2xf16>) outs(%[[VAL_2]] : tensor<8x16xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.subf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<8x16xf16>
// CHECK:           return %[[VAL_3]] : tensor<8x16xf16>
// CHECK:         }
func.func @linearizing_collapse_at_a_constant(%s: tensor<64x2xf16>,
                                              %y: tensor<8x16xf16>)
    -> tensor<8x16xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<64x2xf16> into tensor<128xf16>
  %e = tensor.empty() : tensor<8x16xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%y, %c : tensor<8x16xf16>, tensor<128xf16>) outs(%e : tensor<8x16xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.subf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<8x16xf16>
  return %r : tensor<8x16xf16>
}
}

// -----

// CHECK: #[[$MAP8:.+]] = affine_map<(d0) -> (d0)>

// Test 5: the same `[64,2] -> [128]` collapse as test 4, read at a LOOP DIM this
// time rather than at a constant. Nothing folds away, the consumer would be left
// holding the floordiv and the mod, and the projection check declines it. The pair
// is the point: identical reassociation, opposite answers.
module {
// CHECK-LABEL:   func.func @linearizing_collapse_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x2xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<64x2xf16> into tensor<128xf16>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP8]], #[[$MAP8]]], iterator_types = ["parallel"]} ins(%[[VAL_1]] : tensor<128xf16>) outs(%[[VAL_2]] : tensor<128xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<128xf16>
// CHECK:           return %[[VAL_3]] : tensor<128xf16>
// CHECK:         }
func.func @linearizing_collapse_not_absorbed(%s: tensor<64x2xf16>) -> tensor<128xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<64x2xf16> into tensor<128xf16>
  %e = tensor.empty() : tensor<128xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%c : tensor<128xf16>) outs(%e : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %d = arith.mulf %in, %in : f16
    linalg.yield %d : f16
  } -> tensor<128xf16>
  return %r : tensor<128xf16>
}
}

// -----

// CHECK: #[[$MAP9:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 6 (NEGATIVE): a linearizing `tensor.expand_shape`, the mirror of test 5.
// `[128] -> [64,2]` splits one dim into two non-unit ones, so the source
// coordinate is `d0 * 2 + d1` -- a mul-add, exactly as expressible and exactly as
// unprojectable as a floordiv.
module {
// CHECK-LABEL:   func.func @linearizing_expand_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<128xf16>) -> tensor<64x2xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] output_shape [64, 2] : tensor<128xf16> into tensor<64x2xf16>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x2xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP9]], #[[$MAP9]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]] : tensor<64x2xf16>) outs(%[[VAL_2]] : tensor<64x2xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<64x2xf16>
// CHECK:           return %[[VAL_3]] : tensor<64x2xf16>
// CHECK:         }
func.func @linearizing_expand_not_absorbed(%s: tensor<128xf16>) -> tensor<64x2xf16> {
  %x = tensor.expand_shape %s [[0, 1]] output_shape [64, 2]
      : tensor<128xf16> into tensor<64x2xf16>
  %e = tensor.empty() : tensor<64x2xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<64x2xf16>) outs(%e : tensor<64x2xf16>) {
  ^bb0(%in: f16, %out: f16):
    %v = arith.mulf %in, %in : f16
    linalg.yield %v : f16
  } -> tensor<64x2xf16>
  return %r : tensor<64x2xf16>
}
}

// -----

// CHECK: #[[$MAP10:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 7 (NEGATIVE): `tensor.reshape`. Its shape is a tensor OPERAND, so there is
// no reassociation to read and no static structure to derive a map from at all --
// not a map that linearizes, no map. One of the two ops the gate exists for; see
// gate.mlir for the same op with a path.
module {
// CHECK-LABEL:   func.func @reshape_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64xf16>, %[[VAL_1:.*]]: tensor<2xindex>) -> tensor<64x1xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_1]]) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x1xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$MAP10]], #[[$MAP10]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64x1xf16>) outs(%[[VAL_3]] : tensor<64x1xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_5]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x1xf16>
// CHECK:           return %[[VAL_4]] : tensor<64x1xf16>
// CHECK:         }
func.func @reshape_not_absorbed(%s: tensor<64xf16>, %shape: tensor<2xindex>)
    -> tensor<64x1xf16> {
  %x = tensor.reshape %s(%shape) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
  %e = tensor.empty() : tensor<64x1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<64x1xf16>) outs(%e : tensor<64x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %v = arith.mulf %in, %in : f16
    linalg.yield %v : f16
  } -> tensor<64x1xf16>
  return %r : tensor<64x1xf16>
}
}

// -----

// CHECK: #[[$MAP11:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 8 (NEGATIVE): `tensor.concat`. Not a coordinate change at all: it SELECTS
// between operands per coordinate, and one operand map names one operand.
module {
// CHECK-LABEL:   func.func @concat_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x1xf16>, %[[VAL_1:.*]]: tensor<64x1xf16>) -> tensor<64x2xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.concat dim(1) %[[VAL_0]], %[[VAL_1]] : (tensor<64x1xf16>, tensor<64x1xf16>) -> tensor<64x2xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x2xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$MAP11]], #[[$MAP11]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64x2xf16>) outs(%[[VAL_3]] : tensor<64x2xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_5]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x2xf16>
// CHECK:           return %[[VAL_4]] : tensor<64x2xf16>
// CHECK:         }
func.func @concat_not_absorbed(%a: tensor<64x1xf16>, %b: tensor<64x1xf16>)
    -> tensor<64x2xf16> {
  %c = tensor.concat dim(1) %a, %b
      : (tensor<64x1xf16>, tensor<64x1xf16>) -> tensor<64x2xf16>
  %e = tensor.empty() : tensor<64x2xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%c : tensor<64x2xf16>) outs(%e : tensor<64x2xf16>) {
  ^bb0(%in: f16, %out: f16):
    %v = arith.mulf %in, %in : f16
    linalg.yield %v : f16
  } -> tensor<64x2xf16>
  return %r : tensor<64x2xf16>
}
}

// -----

// CHECK: #[[$MAP12:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 9 (NEGATIVE, silent): `tensor.extract_slice`, left alone DELIBERATELY and
// not for the reason one would guess. Its coordinate map is perfectly derivable --
// static offsets and unit strides give `(d0, d1) -> (d0 + 4, d1 + 8)`. What is not
// derivable is its EXTENT change, and a linalg operand map cannot state one:
// linalg infers loop bounds from the operand shapes THROUGH the indexing maps, so
// repointing this operand at the 64x128 source under any map makes the inferred
// extents inconsistent and fails linalg's own verifier. A slice crops; an operand
// map re-indexes. It is also the one shape op the scheduler already looks through,
// so it is classified as NOT a coordinate restatement -- which means it is neither
// absorbed nor rejected, on a physicalized path or off one.
module {
// CHECK-LABEL:   func.func @extract_slice_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>) -> tensor<8x16xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][4, 8] [8, 16] [1, 1] : tensor<64x128xf16> to tensor<8x16xf16>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8x16xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$MAP12]], #[[$MAP12]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]] : tensor<8x16xf16>) outs(%[[VAL_2]] : tensor<8x16xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<8x16xf16>
// CHECK:           return %[[VAL_3]] : tensor<8x16xf16>
// CHECK:         }
func.func @extract_slice_not_absorbed(%x: tensor<64x128xf16>) -> tensor<8x16xf16> {
  %s = tensor.extract_slice %x[4, 8] [8, 16] [1, 1]
      : tensor<64x128xf16> to tensor<8x16xf16>
  %e = tensor.empty() : tensor<8x16xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%s : tensor<8x16xf16>) outs(%e : tensor<8x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %v = arith.mulf %in, %in : f16
    linalg.yield %v : f16
  } -> tensor<8x16xf16>
  return %r : tensor<8x16xf16>
}
}
