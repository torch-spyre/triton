// RUN: spyre-triton-opt %s --canonicalize-generic-operands -split-input-file | FileCheck %s

// ReshapeToGeneric: a static collapse_shape / expand_shape whose every user is
// a linalg.generic ins, and whose result -> source map is projected, becomes a
// forwarding generic with that map as its input map. A projected map has every
// result a bare dim or a constant: unit-dim inserts and drops, and rank 0. The
// consumer is not touched.

// A unit-dim collapse in front of a broadcast generic, the tt.broadcast shape:
// `[64,1] -> [64]` reads `(d0) -> (d0, 0)`.
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0, 0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @unit_dim_collapse(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x1xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<64xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<64x1xf16>) outs(%[[VAL_1]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_3]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64xf16>) outs(%[[VAL_5]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           return %[[VAL_6]] : tensor<64x128xf16>
// CHECK:         }
func.func @unit_dim_collapse(%s: tensor<64x1xf16>) -> tensor<64x128xf16> {
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
  return %b : tensor<64x128xf16>
}
}

// -----

// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// A unit-dim expand, the tt.expand_dims shape: `[4] -> [4,1]` reads
// `(d0, d1) -> (d0)`.
module {
// CHECK-LABEL:   func.func @unit_dim_expand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x1xf32>, %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4x1xf32> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]] : tensor<4xf32>) outs(%[[VAL_2]] : tensor<4x1xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_4]] : f32
// CHECK:           } -> tensor<4x1xf32>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:           %[[VAL_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_3]] : tensor<4x1xf32>, tensor<4x1xf32>) outs(%[[VAL_6]] : tensor<4x1xf32>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f32, %[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: f32):
// CHECK:             %[[VAL_11:.*]] = arith.maxnumf %[[VAL_8]], %[[VAL_9]] : f32
// CHECK:             linalg.yield %[[VAL_11]] : f32
// CHECK:           } -> tensor<4x1xf32>
// CHECK:           return %[[VAL_7]] : tensor<4x1xf32>
// CHECK:         }
func.func @unit_dim_expand(%acc: tensor<4x1xf32>, %s: tensor<4xf32>) -> tensor<4x1xf32> {
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

// CHECK: #[[$ATTR_6:.+]] = affine_map<() -> (0, 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<() -> ()>
// CHECK: #[[$ATTR_8:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0) -> ()>

// A collapse to rank 0 has an empty reassociation, and every source dim reads
// 0: `() -> (0, 0)`.
module {
// CHECK-LABEL:   func.func @rank0_collapse(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x1xf16>, %[[VAL_1:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<f16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_7]]], iterator_types = []} ins(%[[VAL_0]] : tensor<1x1xf16>) outs(%[[VAL_2]] : tensor<f16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_4]] : f16
// CHECK:           } -> tensor<f16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_8]], #[[$ATTR_9]], #[[$ATTR_8]]], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_3]] : tensor<8xf16>, tensor<f16>) outs(%[[VAL_6]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK:             %[[VAL_11:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f16
// CHECK:             linalg.yield %[[VAL_11]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_7]] : tensor<8xf16>
// CHECK:         }
func.func @rank0_collapse(%s: tensor<1x1xf16>, %x: tensor<8xf16>) -> tensor<8xf16> {
  %c = tensor.collapse_shape %s [] : tensor<1x1xf16> into tensor<f16>
  %e = tensor.empty() : tensor<8xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %c : tensor<8xf16>, tensor<f16>) outs(%e : tensor<8xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0) -> ()>
// CHECK: #[[$ATTR_11:.+]] = affine_map<(d0) -> (d0)>

// An expand from rank 0: the result has no source dims to read, `(d0) -> ()`.
module {
// CHECK-LABEL:   func.func @rank0_expand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<f16>, %[[VAL_1:.*]]: tensor<1xf16>) -> tensor<1xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_10]], #[[$ATTR_11]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<f16>) outs(%[[VAL_2]] : tensor<1xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_4]] : f16
// CHECK:           } -> tensor<1xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<1xf16>
// CHECK:           %[[VAL_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_11]], #[[$ATTR_11]], #[[$ATTR_11]]], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_3]] : tensor<1xf16>, tensor<1xf16>) outs(%[[VAL_6]] : tensor<1xf16>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK:             %[[VAL_11:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f16
// CHECK:             linalg.yield %[[VAL_11]] : f16
// CHECK:           } -> tensor<1xf16>
// CHECK:           return %[[VAL_7]] : tensor<1xf16>
// CHECK:         }
func.func @rank0_expand(%s: tensor<f16>, %x: tensor<1xf16>) -> tensor<1xf16> {
  %c = tensor.expand_shape %s [] output_shape [1] : tensor<f16> into tensor<1xf16>
  %e = tensor.empty() : tensor<1xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %c : tensor<1xf16>, tensor<1xf16>) outs(%e : tensor<1xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<1xf16>
  return %r : tensor<1xf16>
}
}

// -----

// CHECK: #[[$ATTR_12:.+]] = affine_map<(d0) -> (d0, 0)>
// CHECK: #[[$ATTR_13:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_14:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$ATTR_15:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// A chain: the collapse's only user is the expand, which is not a generic, so
// the collapse is left until the expand has become a generic; then both are
// generics.
module {
// CHECK-LABEL:   func.func @chained_collapse_and_expand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x1xf16>, %[[VAL_1:.*]]: tensor<1x64xf16>) -> tensor<1x64xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_12]], #[[$ATTR_13]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<64x1xf16>) outs(%[[VAL_2]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_4]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK:           %[[VAL_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_14]], #[[$ATTR_15]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_3]] : tensor<64xf16>) outs(%[[VAL_6]] : tensor<1x64xf16>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<1x64xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK:           %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_15]], #[[$ATTR_15]], #[[$ATTR_15]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_7]], %[[VAL_1]] : tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[VAL_10]] : tensor<1x64xf16>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: f16, %[[VAL_13:.*]]: f16, %[[VAL_14:.*]]: f16):
// CHECK:             %[[VAL_15:.*]] = arith.mulf %[[VAL_12]], %[[VAL_13]] : f16
// CHECK:             linalg.yield %[[VAL_15]] : f16
// CHECK:           } -> tensor<1x64xf16>
// CHECK:           return %[[VAL_11]] : tensor<1x64xf16>
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

// CHECK: #[[$ATTR_16:.+]] = affine_map<(d0) -> (d0)>

// Left alone: a linearizing collapse. `[64,2] -> [128]` merges two non-unit
// dims, so its result -> source map, `(d0) -> (d0 floordiv 2, d0 mod 2)`, has
// a result that is neither a bare dim nor a constant.
module {
// CHECK-LABEL:   func.func @linearizing_collapse(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x2xf16>, %[[VAL_1:.*]]: tensor<128xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<64x2xf16> into tensor<128xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<128xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_16]], #[[$ATTR_16]], #[[$ATTR_16]]], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_2]] : tensor<128xf16>, tensor<128xf16>) outs(%[[VAL_3]] : tensor<128xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.subf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<128xf16>
// CHECK:           return %[[VAL_4]] : tensor<128xf16>
// CHECK:         }
func.func @linearizing_collapse(%s: tensor<64x2xf16>, %y: tensor<128xf16>) -> tensor<128xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<64x2xf16> into tensor<128xf16>
  %e = tensor.empty() : tensor<128xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%y, %c : tensor<128xf16>, tensor<128xf16>) outs(%e : tensor<128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.subf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<128xf16>
  return %r : tensor<128xf16>
}
}

// -----

// CHECK: #[[$ATTR_17:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Left alone: a linearizing expand. `[128] -> [64,2]` splits one dim into two
// non-unit ones, so its result -> source map, `(d0, d1) -> (d0 * 2 + d1)`, has
// a result that is neither a bare dim nor a constant.
module {
// CHECK-LABEL:   func.func @linearizing_expand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<128xf16>, %[[VAL_1:.*]]: tensor<64x2xf16>) -> tensor<64x2xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] output_shape [64, 2] : tensor<128xf16> into tensor<64x2xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x2xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_17]], #[[$ATTR_17]], #[[$ATTR_17]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]], %[[VAL_2]] : tensor<64x2xf16>, tensor<64x2xf16>) outs(%[[VAL_3]] : tensor<64x2xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.subf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<64x2xf16>
// CHECK:           return %[[VAL_4]] : tensor<64x2xf16>
// CHECK:         }
func.func @linearizing_expand(%s: tensor<128xf16>, %y: tensor<64x2xf16>) -> tensor<64x2xf16> {
  %c = tensor.expand_shape %s [[0, 1]] output_shape [64, 2]
      : tensor<128xf16> into tensor<64x2xf16>
  %e = tensor.empty() : tensor<64x2xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%y, %c : tensor<64x2xf16>, tensor<64x2xf16>) outs(%e : tensor<64x2xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.subf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<64x2xf16>
  return %r : tensor<64x2xf16>
}
}

// -----

// CHECK: #[[$ATTR_20:.+]] = affine_map<(d0) -> (d0)>

// Left alone: a reshape with a user that is not a generic ins, here the
// function's return, the stand-in for a ktdp.store.
module {
// CHECK-LABEL:   func.func @non_generic_user(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8x1xf16>, %[[VAL_1:.*]]: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<8x1xf16> into tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_20]], #[[$ATTR_20]], #[[$ATTR_20]]], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_2]] : tensor<8xf16>, tensor<8xf16>) outs(%[[VAL_3]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_4]], %[[VAL_2]] : tensor<8xf16>, tensor<8xf16>
// CHECK:         }
func.func @non_generic_user(%s: tensor<8x1xf16>, %y: tensor<8xf16>) -> (tensor<8xf16>, tensor<8xf16>) {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<8x1xf16> into tensor<8xf16>
  %e = tensor.empty() : tensor<8xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%y, %c : tensor<8xf16>, tensor<8xf16>) outs(%e : tensor<8xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %r, %c : tensor<8xf16>, tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_21:.+]] = affine_map<(d0) -> (d0)>

// Left alone: a reshape read as a generic's outs rather than its ins.
module {
// CHECK-LABEL:   func.func @outs_user(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<8x1xf16>, %[[VAL_1:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<8x1xf16> into tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_21]], #[[$ATTR_21]]], iterator_types = ["parallel"]} ins(%[[VAL_1]] : tensor<8xf16>) outs(%[[VAL_2]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_3]] : tensor<8xf16>
// CHECK:         }
func.func @outs_user(%s: tensor<8x1xf16>, %y: tensor<8xf16>) -> tensor<8xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<8x1xf16> into tensor<8xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%y : tensor<8xf16>) outs(%c : tensor<8xf16>) {
  ^bb0(%a: f16, %out: f16):
    %v = arith.addf %a, %out : f16
    linalg.yield %v : f16
  } -> tensor<8xf16>
  return %r : tensor<8xf16>
}
}

// -----

// CHECK: #[[$ATTR_22:.+]] = affine_map<(d0) -> (d0)>

// Left alone: a dynamic extent.
module {
// CHECK-LABEL:   func.func @dynamic(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<?x1xf16>, %[[VAL_1:.*]]: tensor<?xf16>) -> tensor<?xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<?x1xf16> into tensor<?xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_22]], #[[$ATTR_22]]], iterator_types = ["parallel"]} ins(%[[VAL_2]] : tensor<?xf16>) outs(%[[VAL_1]] : tensor<?xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = arith.negf %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<?xf16>
// CHECK:           return %[[VAL_3]] : tensor<?xf16>
// CHECK:         }
func.func @dynamic(%s: tensor<?x1xf16>, %o: tensor<?xf16>) -> tensor<?xf16> {
  %c = tensor.collapse_shape %s [[0, 1]] : tensor<?x1xf16> into tensor<?xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%c : tensor<?xf16>) outs(%o : tensor<?xf16>) {
  ^bb0(%a: f16, %out: f16):
    %v = arith.negf %a : f16
    linalg.yield %v : f16
  } -> tensor<?xf16>
  return %r : tensor<?xf16>
}
}

// -----

// CHECK: #[[$ATTR_23:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Not this pattern's producers: a tensor.reshape, a generic, and a splat
// constant (the other pattern's) feeding generics stay what they are.
module {
// CHECK-LABEL:   func.func @other_producers(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64xf16>, %[[VAL_1:.*]]: tensor<2xindex>, %[[VAL_2:.*]]: tensor<64x1xf16>) -> tensor<64x1xf16> {
// CHECK:           %[[VAL_3:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_1]]) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<64x1xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_23]], #[[$ATTR_23]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64x1xf16>) outs(%[[VAL_4]] : tensor<64x1xf16>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<64x1xf16>
// CHECK:           %[[VAL_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_23]], #[[$ATTR_23]], #[[$ATTR_23]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_3]], %[[VAL_5]] : tensor<64x1xf16>, tensor<64x1xf16>) outs(%[[VAL_4]] : tensor<64x1xf16>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16, %[[VAL_11:.*]]: f16):
// CHECK:             %[[VAL_12:.*]] = arith.addf %[[VAL_9]], %[[VAL_10]] : f16
// CHECK:             linalg.yield %[[VAL_12]] : f16
// CHECK:           } -> tensor<64x1xf16>
// CHECK:           return %[[VAL_8]] : tensor<64x1xf16>
// CHECK:         }
func.func @other_producers(%s: tensor<64xf16>, %shape: tensor<2xindex>,
                           %y: tensor<64x1xf16>) -> tensor<64x1xf16> {
  %r = tensor.reshape %s(%shape) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
  %e = tensor.empty() : tensor<64x1xf16>
  %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%y : tensor<64x1xf16>) outs(%e : tensor<64x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64x1xf16>
  %g = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%r, %b : tensor<64x1xf16>, tensor<64x1xf16>) outs(%e : tensor<64x1xf16>) {
  ^bb0(%a: f16, %c: f16, %out: f16):
    %v = arith.addf %a, %c : f16
    linalg.yield %v : f16
  } -> tensor<64x1xf16>
  return %g : tensor<64x1xf16>
}
}
