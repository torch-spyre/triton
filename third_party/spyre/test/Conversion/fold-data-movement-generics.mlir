// RUN: spyre-triton-opt %s --fold-data-movement-generics -split-input-file | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0)>

// FoldDataMovementGenerics folds a coordinate change into the indexing map of the
// op that consumes it, so no op whose only effect is to re-index survives.
//
// The mechanism is upstream's elementwise fusion; the policy is the control
// function, which permits a fusion only when the PRODUCER is pure data movement
// -- a linalg.generic whose body is exactly one operation, a linalg.yield of an
// input block argument. The negative tests below are the important ones: they are
// what would have caught the wrong choice of mechanism, since upstream's own pass
// (--linalg-fuse-elementwise-ops) fuses every one of them.

// Test 1: a broadcast generic folded into an elementwise consumer. The producer's
// operand map (d0, d1) -> (d0) is composed into the consumer's, the producer and
// its tensor.empty go, and the consumer's body still holds exactly one compute.
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
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, 0)>

// Test 2: the reproducer's shape -- a unit-dim tensor.collapse_shape in front of
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
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64x1xf16>) outs(%[[VAL_2]] : tensor<64x128xf16>) {
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

// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>

// Test 3: a collapse that merges two NON-unit dims is a genuine linearization.
// It cannot be written as an affine operand map at all, so it is left alone --
// this is the case upstream answers by changing the consumer's iteration space,
// which is what puts a reshape on the physicalized data path.
module {
// CHECK-LABEL:   func.func @linearizing_collapse_not_absorbed(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x2xf16>) -> tensor<128xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0, 1]] : tensor<64x2xf16> into tensor<128xf16>
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[VAL_1]] : tensor<128xf16>) outs(%[[VAL_2]] : tensor<128xf16>) {
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

// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 4 (NEGATIVE): two plain computes. Identity maps throughout, no shape
// change anywhere. Upstream's pass merges them into one generic holding both
// math.exp and arith.mulf; the backend takes one compute per group, so this must
// stay two ops.
module {
// CHECK-LABEL:   func.func @no_fuse_two_computes(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>, %[[VAL_1:.*]]: tensor<64x128xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]] : tensor<64x128xf16>) outs(%[[VAL_2]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK:             %[[VAL_6:.*]] = math.exp %[[VAL_4]] : f16
// CHECK:             linalg.yield %[[VAL_6]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_3]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64x128xf16>) outs(%[[VAL_7]] : tensor<64x128xf16>) {
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

// -----

// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (d0)>

// Test 5: a generic that is BOTH a compute and a shape change -- a subtract
// already carrying a broadcast in its operand map. It is declined as a PRODUCER
// (the multiply below it stays a separate op) and remains eligible as a CONSUMER
// (the broadcast above it folds into its map). Both roles at once, and neither
// needs a case of its own: the predicate only ever looks at the producer.
module {
// CHECK-LABEL:   func.func @shape_carrying_compute(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x128xf16>, %[[VAL_1:.*]]: tensor<64xf16>, %[[VAL_2:.*]]: tensor<64x128xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_7]], #[[$ATTR_6]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x128xf16>, tensor<64xf16>) outs(%[[VAL_3]] : tensor<64x128xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.subf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             linalg.yield %[[VAL_8]] : f16
// CHECK:           } -> tensor<64x128xf16>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<64x128xf16>
// CHECK:           %[[VAL_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_6]], #[[$ATTR_6]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_4]], %[[VAL_2]] : tensor<64x128xf16>, tensor<64x128xf16>) outs(%[[VAL_9]] : tensor<64x128xf16>) {
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

// CHECK: #[[$ATTR_8:.+]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// Test 6: a REDUCTION consumer. Because the predicate keys on the producer, this
// needs nothing special: the permutation folds into the reduction's operand map
// while its iterator_types and its single-compute body are untouched -- which is
// also what keeps DropReductionInitFill's precondition (a reduce body of exactly
// two ops) true, a pure data-movement producer contributing no body op.
module {
// CHECK-LABEL:   func.func @permute_into_reduction(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x24xf16>) -> tensor<24x64xf16> {
// CHECK:           %[[VAL_1:.*]] = tensor.empty() : tensor<24x64xf16>
// CHECK:           %[[VAL_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_8]], #[[$ATTR_9]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[VAL_0]] : tensor<64x24xf16>) outs(%[[VAL_1]] : tensor<24x64xf16>) {
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

// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$ATTR_11:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 7: a chain of two shape changes folds to a single composed map -- the
// transpose and the broadcast both disappear into the consumer's operand map.
module {
// CHECK-LABEL:   func.func @chained_shape_changes(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<24xf16>, %[[VAL_1:.*]]: tensor<64x24xf16>) -> tensor<64x24xf16> {
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<64x24xf16>
// CHECK:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_10]], #[[$ATTR_11]], #[[$ATTR_11]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<24xf16>, tensor<64x24xf16>) outs(%[[VAL_2]] : tensor<64x24xf16>) {
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

// CHECK: #[[$ATTR_12:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_13:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_14:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[$ATTR_15:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// Test 8 (NEGATIVE): a contraction feeding an elementwise op. Its body is a
// multiply and an add, so it is a compute however permuting its maps look, and
// the two stay separate.
module {
// CHECK-LABEL:   func.func @no_fuse_contraction(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x32xf16>, %[[VAL_1:.*]]: tensor<32x24xf16>, %[[VAL_2:.*]]: tensor<64x24xf16>) -> tensor<64x24xf16> {
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<64x24xf16>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_12]], #[[$ATTR_13]], #[[$ATTR_14]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<64x32xf16>, tensor<32x24xf16>) outs(%[[VAL_3]] : tensor<64x24xf16>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK:             %[[VAL_8:.*]] = arith.mulf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK:             linalg.yield %[[VAL_9]] : f16
// CHECK:           } -> tensor<64x24xf16>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<64x24xf16>
// CHECK:           %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_15]], #[[$ATTR_15]], #[[$ATTR_15]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_4]], %[[VAL_2]] : tensor<64x24xf16>, tensor<64x24xf16>) outs(%[[VAL_10]] : tensor<64x24xf16>) {
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
