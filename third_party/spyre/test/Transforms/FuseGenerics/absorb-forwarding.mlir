// RUN: spyre-triton-opt %s --fuse-generics -split-input-file | FileCheck %s

// AbsorbForwardingGeneric: a consumer ins whose producer generic yields, for
// that result, one of its own ins block arguments is repointed at that input,
// under A o R^-1 o C. The reshape cases arrive as CanonicalizeGenericOperands
// leaves them: a forwarding generic in place of each collapse_shape and
// expand_shape. The declines are D1-D4, a non-permutation result map, and the
// other pattern's producers (a generic yielding an op result, a tensor op).

// -----


// A broadcast generic absorbed into an elementwise consumer. The producer's
// operand map (d0, d1) -> (d0) is composed into the consumer's, the producer and
// its tensor.empty go, and the consumer's body still holds exactly one compute.
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0)>
module {
// CHECK-LABEL:   func.func @broadcast_into_elementwise(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>, %[[VAL_1:.*]]: tensor<64xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64xf16>) outs(%[[VAL_2]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.subf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           return %[[VAL_3]] : tensor<64x128xf16>
// CHECK:         }
func.func @broadcast_into_elementwise(%x: tensor<64x128xf16>, %s: tensor<64xf16>)
    -> tensor<64x128xf16> {
  %e = tensor.empty() : tensor<64x128xf16>
  %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%s : tensor<64xf16>) outs(%e : tensor<64x128xf16>) {
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

// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0)>


// A generic that is BOTH a compute and a shape change -- a subtract
// already carrying a broadcast in its operand map. It is not absorbed as a PRODUCER
// (the multiply below it stays a separate op), and it is still a CONSUMER (the
// broadcast above it is absorbed into its map).
module {
// CHECK-LABEL:   func.func @shape_carrying_compute(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>, %[[VAL_1:.*]]: tensor<64xf16>, %[[VAL_2:.*]]: tensor<64x128xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64xf16>) outs(%[[VAL_3]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.subf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_4]], %[[VAL_2]] : tensor<64x128xf16>, tensor<64x128xf16>) outs(%[[VAL_9]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: f16, %[[VAL_12:.*]]: f16, %[[VAL_13:.*]]: f16):
// CHECK:             %[[VAL_14:.*]] = arith.mulf %[[VAL_11]], %[[VAL_12]] : f16
// CHECK:             linalg.yield %[[VAL_14]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           return %[[VAL_10]] : tensor<64x128xf16>
// CHECK:         }
func.func @shape_carrying_compute(%x: tensor<64x128xf16>, %s: tensor<64xf16>,
                                  %y: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %e = tensor.empty() : tensor<64x128xf16>
  %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%s : tensor<64xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64x128xf16>
  %e2 = tensor.empty() : tensor<64x128xf16>
  %sub = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x, %b : tensor<64x128xf16>, tensor<64x128xf16>) outs(%e2 : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b2: f16, %out: f16):
    %d = arith.subf %a, %b2 : f16
    linalg.yield %d : f16
  } -> tensor<64x128xf16>
  %o = tensor.empty() : tensor<64x128xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%sub, %y : tensor<64x128xf16>, tensor<64x128xf16>) outs(%o : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b2: f16, %out: f16):
    %v = arith.mulf %a, %b2 : f16
    linalg.yield %v : f16
  } -> tensor<64x128xf16>
  return %r : tensor<64x128xf16>
}
}

// -----

// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>


// A REDUCTION consumer. The permutation is absorbed into the reduction's operand map
// while its iterator_types and its single-compute body are untouched -- which is
// also what keeps DropReductionInitFill's precondition (a reduce body of exactly
// two ops) true, a pure data-movement producer contributing no body op.
module {
// CHECK-LABEL:   func.func @permute_into_reduction(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x24xf16>) -> tensor<24x64xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<24x64xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_5]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[VAL_0]] : tensor<64x24xf16>) outs(%[[VAL_1]] : tensor<24x64xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_3]] : f16
// CHECK:             linalg.yield %[[VAL_5]] : f16
// CHECK:           } -> tensor<24x64xf16>
// CHECK:           return %[[VAL_2]] : tensor<24x64xf16>
// CHECK:         }
func.func @permute_into_reduction(%a: tensor<64x24xf16>) -> tensor<24x64xf16> {
  %e = tensor.empty() : tensor<24x64xf16>
  %t = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a : tensor<64x24xf16>) outs(%e : tensor<24x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<24x64xf16>
  %o = tensor.empty() : tensor<24x64xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%t : tensor<24x64xf16>) outs(%o : tensor<24x64xf16>) {
  ^bb0(%in: f16, %acc: f16):
    %s = arith.addf %acc, %in : f16
    linalg.yield %s : f16
  } -> tensor<24x64xf16>
  return %r : tensor<24x64xf16>
}
}

// -----

// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (d0, d1)>


// A chain of two shape changes folds to a single composed map -- the
// transpose and the broadcast both disappear into the consumer's operand map.
module {
// CHECK-LABEL:   func.func @chained_shape_changes(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<24xf16>, %[[VAL_1:.*]]: tensor<64x24xf16>) -> tensor<64x24xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x24xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_7]], #[[$ATTR_7]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<24xf16>, tensor<64x24xf16>) outs(%[[VAL_2]] : tensor<64x24xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x24xf16>
// CHECK:           return %[[VAL_3]] : tensor<64x24xf16>
// CHECK:         }
func.func @chained_shape_changes(%a: tensor<24xf16>, %y: tensor<64x24xf16>)
    -> tensor<64x24xf16> {
  %e = tensor.empty() : tensor<24x64xf16>
  %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a : tensor<24xf16>) outs(%e : tensor<24x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<24x64xf16>
  %e2 = tensor.empty() : tensor<64x24xf16>
  %t = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%b : tensor<24x64xf16>) outs(%e2 : tensor<64x24xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64x24xf16>
  %o = tensor.empty() : tensor<64x24xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%t, %y : tensor<64x24xf16>, tensor<64x24xf16>) outs(%o : tensor<64x24xf16>) {
  ^bb0(%p: f16, %q: f16, %out: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<64x24xf16>
  return %r : tensor<64x24xf16>
}
}

// -----

// CHECK: #[[$ATTR_8:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0, d1) -> (d0, 0)>

// The reproducer's shape -- a unit-dim tensor.collapse_shape in front of
// the broadcast, which is what tt.broadcast's lowering emits because
// linalg.broadcast takes its input rank-reduced. Both fold: the consumer ends up
// reading the rank-2 statistic at a CONSTANT lane, (d0, d1) -> (d0, 0), which is
// the form hand-written reference KTIR states. Leaving the reshape behind is not
// enough -- the consumer would read a rank-1 value that no longer matches the
// rank-2 access tile its load came from, and the scheduler refuses that.
#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @unit_dim_collapse_and_broadcast(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>, %[[VAL_1:.*]]: tensor<64x1xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_8]], #[[$ATTR_9]], #[[$ATTR_8]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64x1xf16>) outs(%[[VAL_2]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.subf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           return %[[VAL_3]] : tensor<64x128xf16>
// CHECK:         }
  func.func @unit_dim_collapse_and_broadcast(%arg0: tensor<64x128xf16>, %arg1: tensor<64x1xf16>) -> tensor<64x128xf16> {
    %0 = tensor.empty() : tensor<64xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg1 : tensor<64x1xf16>) outs(%0 : tensor<64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<64xf16>
    %2 = tensor.empty() : tensor<64x128xf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<64xf16>) outs(%2 : tensor<64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<64x128xf16>
    %4 = tensor.empty() : tensor<64x128xf16>
    %5 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %3 : tensor<64x128xf16>, tensor<64x128xf16>) outs(%4 : tensor<64x128xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %6 = arith.subf %in, %in_0 : f16
      linalg.yield %6 : f16
    } -> tensor<64x128xf16>
    return %5 : tensor<64x128xf16>
  }
}

// -----

// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_11:.+]] = affine_map<(d0, d1) -> (d0)>

// A unit-dim `tensor.expand_shape` -- the collapse's mirror image, and
// the real softmax_2pass shape: a rank-1 reduce result expanded to rank 2 to meet
// a loop-carried rank-2 accumulator. Its reassociation reads SOURCE dim -> RESULT
// dims, the opposite of the collapse's, so each source coordinate is a
// LINEARIZATION of its group rather than a delinearization of one -- and with a
// single non-unit result dim that linearization is just that dim. The consumer
// ends up reading the rank-1 value at (d0, d1) -> (d0).
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @unit_dim_expand_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4x1xf32>, %[[VAL_1:.*]]: tensor<4xf32>) -> tensor<4x1xf32> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_10]], #[[$ATTR_11]], #[[$ATTR_10]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<4x1xf32>, tensor<4xf32>) outs(%[[VAL_2]] : tensor<4x1xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.maxnumf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           } -> tensor<4x1xf32>
// CHECK:           return %[[VAL_3]] : tensor<4x1xf32>
// CHECK:         }
  func.func @unit_dim_expand_absorbed(%arg0: tensor<4x1xf32>, %arg1: tensor<4xf32>) -> tensor<4x1xf32> {
    %0 = tensor.empty() : tensor<4x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<4xf32>) outs(%0 : tensor<4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x1xf32>
    %2 = tensor.empty() : tensor<4x1xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%2 : tensor<4x1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.maxnumf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4x1xf32>
    return %3 : tensor<4x1xf32>
  }
}

// -----

// CHECK: #[[$ATTR_12:.+]] = affine_map<(d0, d1) -> (d1, 0)>
// CHECK: #[[$ATTR_13:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// A CHAIN of two shape ops on one operand, one of each kind, as forwarding
// generics, absorbed one after the other into the same map. `[64,1] -> [64]` collapses and `[64] ->
// [1,64]` expands, and the consumer reading the rank-2 result at the identity
// ends up reading the original `tensor<64x1xf16>` at (d0, d1) -> (d1, 0). Neither
// generic knows the other is there, and the greedy driver may take them in
// either order.
#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @chained_collapse_and_expand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x1xf16>, %[[VAL_1:.*]]: tensor<1x64xf16>) -> tensor<1x64xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_12]], #[[$ATTR_13]], #[[$ATTR_13]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x1xf16>, tensor<1x64xf16>) outs(%[[VAL_2]] : tensor<1x64xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<1x64xf16>
// CHECK:           return %[[VAL_3]] : tensor<1x64xf16>
// CHECK:         }
  func.func @chained_collapse_and_expand(%arg0: tensor<64x1xf16>, %arg1: tensor<1x64xf16>) -> tensor<1x64xf16> {
    %0 = tensor.empty() : tensor<64xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<64x1xf16>) outs(%0 : tensor<64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<64xf16>
    %2 = tensor.empty() : tensor<1x64xf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<64xf16>) outs(%2 : tensor<1x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x64xf16>
    %4 = tensor.empty() : tensor<1x64xf16>
    %5 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3, %arg1 : tensor<1x64xf16>, tensor<1x64xf16>) outs(%4 : tensor<1x64xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %6 = arith.mulf %in, %in_0 : f16
      linalg.yield %6 : f16
    } -> tensor<1x64xf16>
    return %5 : tensor<1x64xf16>
  }
}

// -----

// CHECK: #[[$ATTR_14:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_15:.+]] = affine_map<(d0, d1) -> (0, 0)>

// D2 reads the COMPOSED map, not the reassociation. `[64,2] -> [128]` fuses two non-unit dims, so its
// result-to-source map is `(d0) -> (d0 floordiv 2, d0 mod 2)` -- a genuine
// linearization, which this pass emits rather than refusing to state. But the
// consumer reads the collapsed value at a CONSTANT coordinate, so the composite
// is `(d0, d1) -> (0, 0)`: the floordiv and the mod fold away against the
// constant and nothing is left for a loop IV to be projected through. Restated.
#map = affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (0)>
module {
// CHECK-LABEL:   func.func @linearizing_collapse_at_a_constant(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x2xf16>, %[[VAL_1:.*]]: tensor<8x16xf16>) -> tensor<8x16xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8x16xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_14]], #[[$ATTR_15]], #[[$ATTR_14]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]], %[[VAL_0]] : tensor<8x16xf16>, tensor<64x2xf16>) outs(%[[VAL_2]] : tensor<8x16xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.subf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<8x16xf16>
// CHECK:           return %[[VAL_3]] : tensor<8x16xf16>
// CHECK:         }
  func.func @linearizing_collapse_at_a_constant(%arg0: tensor<64x2xf16>, %arg1: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %0 = tensor.empty() : tensor<128xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<64x2xf16>) outs(%0 : tensor<128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<128xf16>
    %2 = tensor.empty() : tensor<8x16xf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1, %1 : tensor<8x16xf16>, tensor<128xf16>) outs(%2 : tensor<8x16xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %4 = arith.subf %in, %in_0 : f16
      linalg.yield %4 : f16
    } -> tensor<8x16xf16>
    return %3 : tensor<8x16xf16>
  }
}

// -----

// CHECK: #[[$ATTR_16:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_17:.+]] = affine_map<(d0, d1) -> (d0)>


// A data-movement generic with two consumers is absorbed into each, and then
// erased as trivially dead.
module {
// CHECK-LABEL:   func.func @two_consumers(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64xf16>, %[[VAL_1:.*]]: tensor<64x8xf16>) -> (tensor<64x8xf16>, tensor<64x8xf16>) {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_16]], #[[$ATTR_17]], #[[$ATTR_16]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]], %[[VAL_0]] : tensor<64x8xf16>, tensor<64xf16>) outs(%[[VAL_2]] : tensor<64x8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<64x8xf16>
// CHECK:           %[[VAL_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_16]], #[[$ATTR_17]], #[[$ATTR_16]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]], %[[VAL_0]] : tensor<64x8xf16>, tensor<64xf16>) outs(%[[VAL_2]] : tensor<64x8xf16>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16, %[[VAL_11:.*]]: f16):
// CHECK:             %[[VAL_12:.*]] = arith.mulf %[[VAL_9]], %[[VAL_10]] : f16
// CHECK:             linalg.yield %[[VAL_12]] : f16
// CHECK:           } -> tensor<64x8xf16>
// CHECK:           return %[[VAL_3]], %[[VAL_8]] : tensor<64x8xf16>, tensor<64x8xf16>
// CHECK:         }
func.func @two_consumers(%s: tensor<64xf16>, %x: tensor<64x8xf16>) -> (tensor<64x8xf16>, tensor<64x8xf16>) {
  %e = tensor.empty() : tensor<64x8xf16>
  %b = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%s : tensor<64xf16>) outs(%e : tensor<64x8xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64x8xf16>
  %r1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x, %b : tensor<64x8xf16>, tensor<64x8xf16>) outs(%e : tensor<64x8xf16>) {
  ^bb0(%a: f16, %c: f16, %out: f16):
    %v = arith.addf %a, %c : f16
    linalg.yield %v : f16
  } -> tensor<64x8xf16>
  %r2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%x, %b : tensor<64x8xf16>, tensor<64x8xf16>) outs(%e : tensor<64x8xf16>) {
  ^bb0(%a: f16, %c: f16, %out: f16):
    %v = arith.mulf %a, %c : f16
    linalg.yield %v : f16
  } -> tensor<64x8xf16>
  return %r1, %r2 : tensor<64x8xf16>, tensor<64x8xf16>
}
}

// -----

// CHECK: #[[$ATTR_18:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_19:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_20:.+]] = affine_map<(d0, d1) -> (d0)>

// NEGATIVE: a data-movement generic feeding a reduction's `outs` survives. Only
// an `ins` is absorbed.
module {
// CHECK-LABEL:   func.func @dm_feeds_outs(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64xf16>, %[[VAL_1:.*]]: tensor<64x8xf16>) -> tensor<64xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_18]], #[[$ATTR_18]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<64xf16>) outs(%[[VAL_2]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_4]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_19]], #[[$ATTR_20]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_1]] : tensor<64x8xf16>) outs(%[[VAL_3]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK:             linalg.yield %[[VAL_9]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           return %[[VAL_6]] : tensor<64xf16>
// CHECK:         }
func.func @dm_feeds_outs(%s: tensor<64xf16>, %x: tensor<64x8xf16>) -> tensor<64xf16> {
  %e = tensor.empty() : tensor<64xf16>
  %b = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%s : tensor<64xf16>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<64x8xf16>) outs(%b : tensor<64xf16>) {
  ^bb0(%a: f16, %acc: f16):
    %v = arith.addf %a, %acc : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}
}

// -----

// CHECK: #[[$ATTR_21:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_22:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_23:.+]] = affine_map<(d0) -> (d0)>

// NEGATIVE: a lone-yield generic with a reduction loop is not absorbed
// (D1).
module {
// CHECK-LABEL:   func.func @lone_yield_reduction(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x8xf16>, %[[VAL_1:.*]]: tensor<64xf16>) -> tensor<64xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_21]], #[[$ATTR_22]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_0]] : tensor<64x8xf16>) outs(%[[VAL_2]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_4]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_23]], #[[$ATTR_23]], #[[$ATTR_23]]], iterator_types = ["parallel"]} ins(%[[VAL_3]], %[[VAL_1]] : tensor<64xf16>, tensor<64xf16>) outs(%[[VAL_2]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK:             %[[VAL_10:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK:             linalg.yield %[[VAL_10]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           return %[[VAL_6]] : tensor<64xf16>
// CHECK:         }
func.func @lone_yield_reduction(%x: tensor<64x8xf16>, %y: tensor<64xf16>) -> tensor<64xf16> {
  %e = tensor.empty() : tensor<64xf16>
  %b = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<64x8xf16>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%b, %y : tensor<64xf16>, tensor<64xf16>) outs(%e : tensor<64xf16>) {
  ^bb0(%a: f16, %c: f16, %out: f16):
    %v = arith.addf %a, %c : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}
}

// -----

// CHECK: #[[$ATTR_24:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_25:.+]] = affine_map<(d0) -> (0, 0)>

// A rank-0 collapse: both source dims are in no group, and read 0.
#map = affine_map<() -> (0, 0)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
module {
// CHECK-LABEL:   func.func @rank0_collapse(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x1xf16>, %[[VAL_1:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_24]], #[[$ATTR_25]], #[[$ATTR_24]]], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_0]] : tensor<8xf16>, tensor<1x1xf16>) outs(%[[VAL_2]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_3]] : tensor<8xf16>
// CHECK:         }
  func.func @rank0_collapse(%arg0: tensor<1x1xf16>, %arg1: tensor<8xf16>) -> tensor<8xf16> {
    %0 = tensor.empty() : tensor<f16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = []} ins(%arg0 : tensor<1x1xf16>) outs(%0 : tensor<f16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<f16>
    %2 = tensor.empty() : tensor<8xf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%arg1, %1 : tensor<8xf16>, tensor<f16>) outs(%2 : tensor<8xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %4 = arith.addf %in, %in_0 : f16
      linalg.yield %4 : f16
    } -> tensor<8xf16>
    return %3 : tensor<8xf16>
  }
}

// -----

// CHECK: #[[$ATTR_26:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_27:.+]] = affine_map<(d0) -> (0, 0)>

// An all-unit group: every source dim of it reads 0.
#map = affine_map<(d0) -> (0, 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (0)>
module {
// CHECK-LABEL:   func.func @all_unit_collapse(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x1xf16>, %[[VAL_1:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_26]], #[[$ATTR_27]], #[[$ATTR_26]]], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_0]] : tensor<8xf16>, tensor<1x1xf16>) outs(%[[VAL_2]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_3]] : tensor<8xf16>
// CHECK:         }
  func.func @all_unit_collapse(%arg0: tensor<1x1xf16>, %arg1: tensor<8xf16>) -> tensor<8xf16> {
    %0 = tensor.empty() : tensor<1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<1x1xf16>) outs(%0 : tensor<1xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1xf16>
    %2 = tensor.empty() : tensor<8xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel"]} ins(%arg1, %1 : tensor<8xf16>, tensor<1xf16>) outs(%2 : tensor<8xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %4 = arith.addf %in, %in_0 : f16
      linalg.yield %4 : f16
    } -> tensor<8xf16>
    return %3 : tensor<8xf16>
  }
}

// -----

// CHECK: #[[$ATTR_28:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_29:.+]] = affine_map<(d0) -> ()>

// A rank-0 expand: the source has no dims, so the consumer reads it under
// `(d0) -> ()`.
#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (0, 0)>
module {
// CHECK-LABEL:   func.func @rank0_expand(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<f16>, %[[VAL_1:.*]]: tensor<8xf16>) -> tensor<8xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_28]], #[[$ATTR_29]], #[[$ATTR_28]]], iterator_types = ["parallel"]} ins(%[[VAL_1]], %[[VAL_0]] : tensor<8xf16>, tensor<f16>) outs(%[[VAL_2]] : tensor<8xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK:             linalg.yield %[[VAL_7]] : f16
// CHECK:           } -> tensor<8xf16>
// CHECK:           return %[[VAL_3]] : tensor<8xf16>
// CHECK:         }
  func.func @rank0_expand(%arg0: tensor<f16>, %arg1: tensor<8xf16>) -> tensor<8xf16> {
    %0 = tensor.empty() : tensor<1x1xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<f16>) outs(%0 : tensor<1x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x1xf16>
    %2 = tensor.empty() : tensor<8xf16>
    %3 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%arg1, %1 : tensor<8xf16>, tensor<1x1xf16>) outs(%2 : tensor<8xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %4 = arith.addf %in, %in_0 : f16
      linalg.yield %4 : f16
    } -> tensor<8xf16>
    return %3 : tensor<8xf16>
  }
}

// -----

// CHECK: #[[$ATTR_30:.+]] = affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>
// CHECK: #[[$ATTR_31:.+]] = affine_map<(d0) -> (d0)>

// D2, repoint: a `[64,2] -> [128]` collapse read at a LOOP DIM. Nothing folds
// away, the consumer would hold the floordiv and the mod, and D2 declines it.
// The same collapse read at a constant, above, is accepted.
#map = affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>
#map1 = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @linearizing_collapse_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x2xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<128xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_30]], #[[$ATTR_31]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<64x2xf16>) outs(%[[VAL_1]] : tensor<128xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_3]] : f16
// CHECK:           } -> tensor<128xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<128xf16>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_31]], #[[$ATTR_31]]], iterator_types = ["parallel"]} ins(%[[VAL_2]] : tensor<128xf16>) outs(%[[VAL_5]] : tensor<128xf16>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK:             %[[VAL_9:.*]] = arith.mulf %[[VAL_7]], %[[VAL_7]] : f16
// CHECK:             linalg.yield %[[VAL_9]] : f16
// CHECK:           } -> tensor<128xf16>
// CHECK:           return %[[VAL_6]] : tensor<128xf16>
// CHECK:         }
  func.func @linearizing_collapse_not_absorbed(%arg0: tensor<64x2xf16>) -> tensor<128xf16> {
    %0 = tensor.empty() : tensor<128xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<64x2xf16>) outs(%0 : tensor<128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<128xf16>
    %2 = tensor.empty() : tensor<128xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%1 : tensor<128xf16>) outs(%2 : tensor<128xf16>) {
    ^bb0(%in: f16, %out: f16):
      %4 = arith.mulf %in, %in : f16
      linalg.yield %4 : f16
    } -> tensor<128xf16>
    return %3 : tensor<128xf16>
  }
}

// -----

// CHECK: #[[$ATTR_32:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: #[[$ATTR_33:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// D2, repoint: a linearizing `tensor.expand_shape`. `[128] -> [64,2]` splits one
// dim into two non-unit ones, so the source coordinate is `d0 * 2 + d1`.
#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @linearizing_expand_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<128xf16>) -> tensor<64x2xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<64x2xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_32]], #[[$ATTR_33]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<128xf16>) outs(%[[VAL_1]] : tensor<64x2xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_3]] : f16
// CHECK:           } -> tensor<64x2xf16>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<64x2xf16>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_33]], #[[$ATTR_33]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64x2xf16>) outs(%[[VAL_5]] : tensor<64x2xf16>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK:             %[[VAL_9:.*]] = arith.mulf %[[VAL_7]], %[[VAL_7]] : f16
// CHECK:             linalg.yield %[[VAL_9]] : f16
// CHECK:           } -> tensor<64x2xf16>
// CHECK:           return %[[VAL_6]] : tensor<64x2xf16>
// CHECK:         }
  func.func @linearizing_expand_not_absorbed(%arg0: tensor<128xf16>) -> tensor<64x2xf16> {
    %0 = tensor.empty() : tensor<64x2xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128xf16>) outs(%0 : tensor<64x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<64x2xf16>
    %2 = tensor.empty() : tensor<64x2xf16>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<64x2xf16>) outs(%2 : tensor<64x2xf16>) {
    ^bb0(%in: f16, %out: f16):
      %4 = arith.mulf %in, %in : f16
      linalg.yield %4 : f16
    } -> tensor<64x2xf16>
    return %3 : tensor<64x2xf16>
  }
}

// -----

// CHECK: #[[$ATTR_34:.+]] = affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>
// CHECK: #[[$ATTR_35:.+]] = affine_map<(d0) -> (d0)>

// D2, repoint: a lone-yield generic whose ins map linearizes. Its composed
// map is `(d0) -> (d0 floordiv 2, d0 mod 2)`, which D2 declines.
module {
// CHECK-LABEL:   func.func @d2_lone_yield_linearizing(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x2xf16>, %[[VAL_1:.*]]: tensor<128xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_34]], #[[$ATTR_35]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<64x2xf16>) outs(%[[VAL_2]] : tensor<128xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_4]] : f16
// CHECK:           } -> tensor<128xf16>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_35]], #[[$ATTR_35]], #[[$ATTR_35]]], iterator_types = ["parallel"]} ins(%[[VAL_3]], %[[VAL_1]] : tensor<128xf16>, tensor<128xf16>) outs(%[[VAL_2]] : tensor<128xf16>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK:             %[[VAL_10:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK:             linalg.yield %[[VAL_10]] : f16
// CHECK:           } -> tensor<128xf16>
// CHECK:           return %[[VAL_6]] : tensor<128xf16>
// CHECK:         }
func.func @d2_lone_yield_linearizing(%s: tensor<64x2xf16>, %y: tensor<128xf16>) -> tensor<128xf16> {
  %e = tensor.empty() : tensor<128xf16>
  %b = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%s : tensor<64x2xf16>) outs(%e : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<128xf16>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%b, %y : tensor<128xf16>, tensor<128xf16>) outs(%e : tensor<128xf16>) {
  ^bb0(%a: f16, %c: f16, %out: f16):
    %v = arith.addf %a, %c : f16
    linalg.yield %v : f16
  } -> tensor<128xf16>
  return %r : tensor<128xf16>
}
}

// -----

// CHECK: #[[$ATTR_36:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_37:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// D3, repoint: `expand_dims` then a reduce over the inserted unit dim. The
// absorbed operand would name d1 nowhere, so the expand generic stays.
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @d3_expand_into_reduce(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_36]], #[[$ATTR_37]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<4xf32>) outs(%[[VAL_1]] : tensor<4x1xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<4x1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_37]], #[[$ATTR_36]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_2]] : tensor<4x1xf32>) outs(%[[VAL_5]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[VAL_6]] : tensor<4xf32>
// CHECK:         }
  func.func @d3_expand_into_reduce(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tensor.empty() : tensor<4x1xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x1xf32>
    %2 = tensor.empty() : tensor<4xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "reduction"]} ins(%1 : tensor<4x1xf32>) outs(%2 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<4xf32>
    return %3 : tensor<4xf32>
  }
}

// -----

// CHECK: #[[$ATTR_38:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_39:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// D3, repoint: the same shape as the previous case, spelled as a broadcast
// generic. Declined the same way.
module {
// CHECK-LABEL:   func.func @d3_broadcast_into_reduce(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<4x1xf32>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_38]], #[[$ATTR_39]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<4xf32>) outs(%[[VAL_1]] : tensor<4x1xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<4x1xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_39]], #[[$ATTR_38]]], iterator_types = ["parallel", "reduction"]} ins(%[[VAL_2]] : tensor<4x1xf32>) outs(%[[VAL_5]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[VAL_6]] : tensor<4xf32>
// CHECK:         }
func.func @d3_broadcast_into_reduce(%s: tensor<4xf32>) -> tensor<4xf32> {
  %e1 = tensor.empty() : tensor<4x1xf32>
  %x = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%s : tensor<4xf32>) outs(%e1 : tensor<4x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x1xf32>
  %e = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<4x1xf32>) outs(%e : tensor<4xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %v = arith.addf %in, %acc : f32
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
}

// -----

// CHECK: #[[$ATTR_40:.+]] = affine_map<(d0) -> (d0, 0)>
// CHECK: #[[$ATTR_41:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_42:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$ATTR_43:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// D4, repoint: the collapse's source is the consumer's `outs`, so restating it
// would make one value both an ins and an outs.
#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @d4_collapse_of_outs(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x1xf16>) -> tensor<64x1xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<64xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_40]], #[[$ATTR_41]]], iterator_types = ["parallel"]} ins(%[[VAL_0]] : tensor<64x1xf16>) outs(%[[VAL_1]] : tensor<64xf16>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f16, %[[VAL_4:.*]]: f16):
// CHECK:             linalg.yield %[[VAL_3]] : f16
// CHECK:           } -> tensor<64xf16>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_42]], #[[$ATTR_43]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64xf16>) outs(%[[VAL_0]] : tensor<64x1xf16>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.mulf %[[VAL_6]], %[[VAL_6]] : f16
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<64x1xf16>
// CHECK:           return %[[VAL_5]] : tensor<64x1xf16>
// CHECK:         }
  func.func @d4_collapse_of_outs(%arg0: tensor<64x1xf16>) -> tensor<64x1xf16> {
    %0 = tensor.empty() : tensor<64xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<64x1xf16>) outs(%0 : tensor<64xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<64xf16>
    %2 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<64xf16>) outs(%arg0 : tensor<64x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      %3 = arith.mulf %in, %in : f16
      linalg.yield %3 : f16
    } -> tensor<64x1xf16>
    return %2 : tensor<64x1xf16>
  }
}

// -----

// CHECK: #[[$ATTR_44:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_45:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_46:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[$ATTR_47:.+]] = affine_map<(d0, d1) -> (d0, d1)>


// NEGATIVE: a contraction feeding an elementwise op. Its body is a
// multiply and an add, not a forwarded input, so AbsorbForwardingGeneric does
// not match it, and its add forms no group, so the two stay separate.
module {
// CHECK-LABEL:   func.func @no_fuse_contraction(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x32xf16>, %[[VAL_1:.*]]: tensor<32x24xf16>, %[[VAL_2:.*]]: tensor<64x24xf16>) -> tensor<64x24xf16> {
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x24xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_44]], #[[$ATTR_45]], #[[$ATTR_46]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x32xf16>, tensor<32x24xf16>) outs(%[[VAL_3]] : tensor<64x24xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.mulf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK:             linalg.yield %[[VAL_9]] : f16
// CHECK:           } -> tensor<64x24xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<64x24xf16>
// CHECK:           %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_47]], #[[$ATTR_47]], #[[$ATTR_47]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_4]], %[[VAL_2]] : tensor<64x24xf16>, tensor<64x24xf16>) outs(%[[VAL_10]] : tensor<64x24xf16>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: f16, %[[VAL_13:.*]]: f16, %[[VAL_14:.*]]: f16):
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f16
// CHECK:             linalg.yield %[[VAL_15]] : f16
// CHECK:           } -> tensor<64x24xf16>
// CHECK:           return %[[VAL_11]] : tensor<64x24xf16>
// CHECK:         }
func.func @no_fuse_contraction(%a: tensor<64x32xf16>, %b: tensor<32x24xf16>,
                               %y: tensor<64x24xf16>) -> tensor<64x24xf16> {
  %e = tensor.empty() : tensor<64x24xf16>
  %m = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<64x32xf16>, tensor<32x24xf16>) outs(%e : tensor<64x24xf16>) {
  ^bb0(%p: f16, %q: f16, %acc: f16):
    %v = arith.mulf %p, %q : f16
    %s = arith.addf %acc, %v : f16
    linalg.yield %s : f16
  } -> tensor<64x24xf16>
  %o = tensor.empty() : tensor<64x24xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%m, %y : tensor<64x24xf16>, tensor<64x24xf16>) outs(%o : tensor<64x24xf16>) {
  ^bb0(%p: f16, %q: f16, %out: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<64x24xf16>
  return %r : tensor<64x24xf16>
}
}

// -----

// CHECK: #[[$ATTR_48:.+]] = affine_map<(d0, d1) -> (d0, d1)>


// NEGATIVE: `tensor.reshape`. Its shape is a tensor OPERAND, so there is
// no reassociation to read and no static structure to derive a map from at all --
// not a map that linearizes, no map. CanonicalizeGenericOperands leaves it a
// tensor op, and no pattern here matches a tensor op.
module {
// CHECK-LABEL:   func.func @reshape_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64xf16>, %[[VAL_1:.*]]: tensor<2xindex>) -> tensor<64x1xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.reshape %[[VAL_0]](%[[VAL_1]]) : (tensor<64xf16>, tensor<2xindex>) -> tensor<64x1xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x1xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_48]], #[[$ATTR_48]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64x1xf16>) outs(%[[VAL_3]] : tensor<64x1xf16>) {
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

// CHECK: #[[$ATTR_49:.+]] = affine_map<(d0, d1) -> (d0, d1)>


// NEGATIVE: `tensor.concat`. Not a coordinate change at all: it SELECTS
// between operands per coordinate, and one operand map names one operand.
module {
// CHECK-LABEL:   func.func @concat_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x1xf16>, %[[VAL_1:.*]]: tensor<64x1xf16>) -> tensor<64x2xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.concat dim(1) %[[VAL_0]], %[[VAL_1]] : (tensor<64x1xf16>, tensor<64x1xf16>) -> tensor<64x2xf16>
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x2xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_49]], #[[$ATTR_49]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_2]] : tensor<64x2xf16>) outs(%[[VAL_3]] : tensor<64x2xf16>) {
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

// CHECK: #[[$ATTR_50:.+]] = affine_map<(d0, d1) -> (d0, d1)>


// NEGATIVE: `tensor.extract_slice`, left alone DELIBERATELY and
// not for the reason one would guess. Its coordinate map is perfectly derivable --
// static offsets and unit strides give `(d0, d1) -> (d0 + 4, d1 + 8)`. What is not
// derivable is its EXTENT change, and a linalg operand map cannot state one:
// linalg infers loop bounds from the operand shapes THROUGH the indexing maps, so
// repointing this operand at the 64x128 source under any map makes the inferred
// extents inconsistent and fails linalg's own verifier. A slice crops; an operand
// map re-indexes. It is also the one shape op the scheduler already looks through,
// so it is left a tensor op, and no pattern here matches one.
module {
// CHECK-LABEL:   func.func @extract_slice_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>) -> tensor<8x16xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][4, 8] [8, 16] [1, 1] : tensor<64x128xf16> to tensor<8x16xf16>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<8x16xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_50]], #[[$ATTR_50]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]] : tensor<8x16xf16>) outs(%[[VAL_2]] : tensor<8x16xf16>) {
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

// -----

// CHECK: #[[$ATTR_51:.+]] = affine_map<(d0, d1) -> (d0, d1)>



// NEGATIVE: two plain computes, not a compare and its reader, stay apart. No
// rule merges two computes except into a pair LowerSpyreOps selects, and no i1
// crosses between them.
module {
// CHECK-LABEL:   func.func @no_fuse_two_computes(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>, %[[VAL_1:.*]]: tensor<64x128xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_51]], #[[$ATTR_51]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<64x128xf16>) outs(%[[VAL_2]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = math.exp %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_51]], #[[$ATTR_51]], #[[$ATTR_51]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_3]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64x128xf16>) outs(%[[VAL_7]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16, %[[VAL_11:.*]]: f16):
// CHECK:             %[[VAL_12:.*]] = arith.mulf %[[VAL_9]], %[[VAL_10]] : f16
// CHECK:             linalg.yield %[[VAL_12]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           return %[[VAL_8]] : tensor<64x128xf16>
// CHECK:         }
func.func @no_fuse_two_computes(%x: tensor<64x128xf16>, %y: tensor<64x128xf16>)
    -> tensor<64x128xf16> {
  %e = tensor.empty() : tensor<64x128xf16>
  %p = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<64x128xf16>) outs(%e : tensor<64x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %v = math.exp %in : f16
    linalg.yield %v : f16
  } -> tensor<64x128xf16>
  %o = tensor.empty() : tensor<64x128xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%p, %y : tensor<64x128xf16>, tensor<64x128xf16>) outs(%o : tensor<64x128xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %v = arith.mulf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<64x128xf16>
  return %r : tensor<64x128xf16>
}
}
