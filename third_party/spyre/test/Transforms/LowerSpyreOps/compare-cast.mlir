// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file | FileCheck %s

// The compare rule: a comparison whose answer is wanted as a NUMBER rather than as
// a flag. `arith.cmpf` + `arith.uitofp` is the mask shape -- `(m != 0)` used
// multiplicatively -- and `spyreop.compare` is one op that does it, its answer
// coming "in the width compared rather than as a boolean".
//
// Why the pair is one choice and not two: `arith.cmpf` alone gives an `i1`, which
// no spyreop op produces and the device cannot represent in a compute body, so the
// compare is not selectable by itself. What consumes the `i1` decides what the pair
// becomes.
//
// EVERY CASE HERE IS HAND-BUILT INTO ONE BODY, so what is under test is a rule's
// own decision rather than whether fusion happened. compare-from-tensor.mlir drives
// the shape the pipeline actually produces. Every way of failing to select the
// pair leaves an `i1` behind and is refused: invalid.mlir holds the ones that can
// never lower, and not-yet-lowered.mlir the ones a future rule could lower.

//===----------------------------------------------------------------------===//
// Single-body inputs: one decision per case, fusion taken as read
//===----------------------------------------------------------------------===//

// `m != 0` as a float mask, ordered.
//
// Hand-built into one body and driven WITHOUT the fusion pass, so the splat zero is
// still an `ins` operand and the body reads it as a block argument. The rule does
// not care: it reads the compare's operands as the body holds them.
// CHECK-LABEL:   func.func @mask_notequal_f16(
// CHECK-SAME:  %[[M:.*]]: tensor<4xf16>) -> tensor<4xf16> {
// CHECK-NOT:       arith.cmpf
// CHECK-NOT:       arith.uitofp
// CHECK:           %[[Z:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf16>
// CHECK:           linalg.generic {{.*}} ins(%[[M]], %[[Z]] :
// CHECK:           ^bb0(%[[A:.*]]: f16, %[[ZS:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK:             %[[R:.*]] = spyreop.compare <notequal> %[[A]], %[[ZS]] : f16
// CHECK:             linalg.yield %[[R]] : f16
func.func @mask_notequal_f16(%m: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant dense<0.0> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%m, %zero : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %z: f16, %out: f16):
    %c = arith.cmpf one, %a, %z : f16
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// `a == 0`, the other half of the same mask idiom, at f32.
// CHECK-LABEL:   func.func @mask_equal_f32(
// CHECK-NOT:       arith.cmpf
// CHECK:             spyreop.compare <equal> {{.*}} : f32
func.func @mask_equal_f32(%m: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %init = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%m, %zero : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%a: f32, %z: f32, %out: f32):
    %c = arith.cmpf oeq, %a, %z : f32
    %f = arith.uitofp %c : i1 to f32
    linalg.yield %f : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// All four ordered inequalities in one body, so the predicate table is covered
// rather than sampled. The two ordered equalities are the cases above, which makes
// all six accounted for.
// CHECK-LABEL:   func.func @all_ordered_inequalities(
// CHECK-NOT:       arith.cmpf
// CHECK:             spyreop.compare <greaterthan>
// CHECK:             spyreop.compare <greaterequal>
// CHECK:             spyreop.compare <lesserthan>
// CHECK:             spyreop.compare <lesserequal>
func.func @all_ordered_inequalities(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %c0 = arith.cmpf ogt, %a, %b : f16
    %f0 = arith.uitofp %c0 : i1 to f16
    %c1 = arith.cmpf oge, %a, %b : f16
    %f1 = arith.uitofp %c1 : i1 to f16
    %c2 = arith.cmpf olt, %a, %b : f16
    %f2 = arith.uitofp %c2 : i1 to f16
    %c3 = arith.cmpf ole, %a, %b : f16
    %f3 = arith.uitofp %c3 : i1 to f16
    %s0 = arith.addf %f0, %f1 : f16
    %s1 = arith.addf %f2, %f3 : f16
    %s = arith.addf %s0, %s1 : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// All six UNORDERED predicates, each selected as its ordered counterpart. The two
// spellings differ only when an operand is NaN, and the pass assumes no NaN (see
// NO NaN REACHES A COMPUTE BODY in LowerSpyreOps.cpp). `une` is the case that
// matters in practice: it is what Triton emits for `!=`, so without this mapping
// `(a != b).to(f16)` had no device form and was refused.
//
// Operand order is checked as well as the predicate: `ugt %a, %b` must become
// `greaterthan %a, %b`, and a mapping that swapped the operands would compute
// `lesserthan` while printing the right name.
// CHECK-LABEL:   func.func @all_unordered_predicates(
// CHECK-NOT:       arith.cmpf
// CHECK:           ^bb0(%[[A:.*]]: f16, %[[B:.*]]: f16, %{{.*}}: f16):
// CHECK:             spyreop.compare <equal> %[[A]], %[[B]] : f16
// CHECK:             spyreop.compare <notequal> %[[A]], %[[B]] : f16
// CHECK:             spyreop.compare <greaterthan> %[[A]], %[[B]] : f16
// CHECK:             spyreop.compare <greaterequal> %[[A]], %[[B]] : f16
// CHECK:             spyreop.compare <lesserthan> %[[A]], %[[B]] : f16
// CHECK:             spyreop.compare <lesserequal> %[[A]], %[[B]] : f16
// CHECK-NOT:       arith.cmpf
func.func @all_unordered_predicates(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %c0 = arith.cmpf ueq, %a, %b : f16
    %f0 = arith.uitofp %c0 : i1 to f16
    %c1 = arith.cmpf une, %a, %b : f16
    %f1 = arith.uitofp %c1 : i1 to f16
    %c2 = arith.cmpf ugt, %a, %b : f16
    %f2 = arith.uitofp %c2 : i1 to f16
    %c3 = arith.cmpf uge, %a, %b : f16
    %f3 = arith.uitofp %c3 : i1 to f16
    %c4 = arith.cmpf ult, %a, %b : f16
    %f4 = arith.uitofp %c4 : i1 to f16
    %c5 = arith.cmpf ule, %a, %b : f16
    %f5 = arith.uitofp %c5 : i1 to f16
    %s0 = arith.addf %f0, %f1 : f16
    %s1 = arith.addf %f2, %f3 : f16
    %s2 = arith.addf %f4, %f5 : f16
    %s3 = arith.addf %s0, %s1 : f16
    %s = arith.addf %s3, %s2 : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// A reader with no compare in front of it is not lowered by itself.
// `arith.uitofp` is the reader of the compare-cast rule and the anchor of no
// rule, so a cast of an `i1` the body is handed stays as it is, and so does the
// cast of any other value. Nothing is reported: an `i1` block argument is the
// tensor form crossing a body boundary, which FuseGenerics reports.
//
// The same holds for `arith.select`: compare-select.mlir's
// @condition_from_outside_is_left_alone. A rule without readers anchored on
// either op would be tried against the group rules in the order the driver
// visits ops, which verifySpyreOpLoweringRules refuses.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @cast_of_outside_condition_is_left_alone(
// CHECK:           ^bb0(%[[C:.*]]: i1, %{{.*}}: f16):
// CHECK-NEXT:        %[[F:.*]] = arith.uitofp %[[C]] : i1 to f16
// CHECK-NEXT:        linalg.yield %[[F]] : f16
func.func @cast_of_outside_condition_is_left_alone(%c: tensor<4xi1>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel"]}
      ins(%c : tensor<4xi1>) outs(%init : tensor<4xf16>) {
  ^bb0(%cond: i1, %o: f16):
    %f = arith.uitofp %cond : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// THE SCOPE, and it is the scope of the DIAGNOSTIC too. The same pair outside any
// generic body is not matched -- the body is where compute is -- and it is not
// refused either, because the `i1` check walks generic bodies and nothing else. An
// `i1` in ordinary scalar code is not this pass's business, and a mask on the
// address path is exactly that.
// CHECK-LABEL:   func.func @outside_a_body(
// CHECK-NOT:       spyreop.compare
// CHECK:           arith.cmpf
// CHECK:           arith.uitofp
func.func @outside_a_body(%a: f16, %b: f16) -> f16 {
  %c = arith.cmpf oeq, %a, %b : f16
  %f = arith.uitofp %c : i1 to f16
  return %f : f16
}
