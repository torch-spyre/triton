// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics | FileCheck %s

// Tests for the rule that turns `arith.cmpf` + `arith.select` into spyreop ops.
//
// Background, in the order the tests need it.
//
// `arith.select %c, %p, %q` takes %p where %c is true. Its condition %c is an
// `i1`, and the Spyre device has no `i1`: `spyreop.select` instead takes a
// condition of the SAME FLOAT TYPE as the values it is choosing between, and
// takes the first one wherever that float is not zero. So a `1.0` condition
// chooses %p and a `0.0` condition chooses %q.
//
// `spyreop.compare` produces exactly those floats: 1.0 where its predicate holds
// and 0.0 where it does not, in the width it compared. So the usual lowering is a
// pair -- compare producing the float, select consuming it:
//
//     %c = spyreop.compare <greaterthan> %x, %y : f32     // 1.0 or 0.0
//     %s = spyreop.select %c, %p, %q : f32                // picks on non-zero
//
// One case does better. When the kernel's own comparison is already "is this
// value non-zero", `spyreop.select` performs that same test on its condition, so
// the comparison is doing work the select repeats. The rule then drops the
// comparison and hands the select the value that was being compared -- leaving a
// select with NO compare beside it. `tl.where(mask != 0, p, q)` is that case, and
// it is the shape a kernel reads a stored mask back in.
//
// Each test below states which of those two outputs it expects and why.

#map = affine_map<(d0) -> (d0)>

// PURPOSE: the ordinary lowering, where the kernel compares two data values.
//
// `x > y` is not a test against zero, so nothing can be dropped: the comparison
// has to run, and its float answer feeds the select. Asserts BOTH ops appear, and
// that the select reads the compare's result rather than one of the inputs -- a
// rule that wired the wrong operand in would still emit two ops and pass a weaker
// check.
// CHECK-LABEL:   func.func @compare_two_values_keeps_both_ops(
// CHECK:           ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[C:.*]] = spyreop.compare <greaterthan> %[[X]], %[[Y]] : f32
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @compare_two_values_keeps_both_ops(%a: tensor<8xf32>, %b: tensor<8xf32>,
                                             %p: tensor<8xf32>, %q: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf ogt, %x, %y : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: `m != 0` lowers to a select with no compare, because
// `spyreop.select` already tests its condition against zero.
//
// The CHECK-NEXT chain from `^bb0` to `linalg.yield` catches:
//   - a compare left in the body, e.g. `spyreop.compare <notequal> %m, %zero`;
//   - the wrong condition, e.g. `spyreop.select %p, ...`.
//
// `one` is the ordered spelling; Triton's `une` is the next case.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @mask_not_equal_zero_needs_no_compare(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[M]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @mask_not_equal_zero_needs_no_compare(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                                %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf one, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: pin that `m != 0` written in Triton reaches the same one-op form.
//
// Triton's `!=` emits `une`, arith's UNORDERED not-equal, which differs from the
// ordered `one` above only for a NaN input -- and the pass assumes no NaN (NO NaN
// REACHES A COMPUTE BODY, in LowerSpyreOps.cpp). Without this case the previous
// test would pass while every kernel actually written `m != 0` took the two-op
// path.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @triton_not_equal_reaches_the_same_form(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[M]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @triton_not_equal_reaches_the_same_form(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                                  %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf une, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: `m == 0` folds with the two values exchanged, because the select
// picks the FIRST value where `m` is not zero, the opposite of `m == 0`.
//
// `select(m == 0, P, Q)` must become `spyreop.select %m, Q, P`. This catches a
// rule that forgets the exchange, which emits one clean op and returns the
// wrong value on every lane.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @mask_equal_zero_swaps_the_values(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[M]], %[[Q]], %[[P]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @mask_equal_zero_swaps_the_values(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                            %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf oeq, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: pin that `ueq`, the unordered spelling of `m == 0`, takes the same
// exchanged one-op form as `oeq`.
//
// Under the pass's no-NaN assumption `ueq` and `oeq` compute the same answer, so
// the rule must treat them alike. A rule that listed only `oeq` would send `ueq`
// down the compare path -- still correct, one op larger -- and the CHECK-NEXT
// chain below, which leaves no room for a compare, is what catches that.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @unordered_equal_zero_swaps_the_values(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[M]], %[[Q]], %[[P]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @unordered_equal_zero_swaps_the_values(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                                 %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf ueq, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: pin that `tl.where(a != b, p, q)` -- Triton's `!=` between two DATA
// values, so `une` with no zero to fold -- lowers to a compare feeding a select.
//
// This is the case that was refused before the unordered predicates were mapped:
// `une` is not a zero test here, so the rule needs a `spyreop.compare`, and
// `une` had no counterpart. It now maps to `<notequal>` under the pass's no-NaN
// assumption. Asserts the wiring as well as the ops, as
// compare_two_values_keeps_both_ops does.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @triton_not_equal_of_two_values(
// CHECK:           ^bb0(%[[X:.*]]: f32, %[[Y:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[C:.*]] = spyreop.compare <notequal> %[[X]], %[[Y]] : f32
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @triton_not_equal_of_two_values(%a: tensor<8xf32>, %b: tensor<8xf32>,
                                          %p: tensor<8xf32>, %q: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf une, %x, %y : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: an ordering against zero keeps its compare, because it disagrees
// with "not zero" on values a mask can hold:
//   - `m > 0` is false for `m = -1`, where "not zero" is true;
//   - `m >= 0` is also true for `m = 0`.
//
// The checks bind the compare to `%m` and the select to the compare, catching a
// rule that emits both ops but wires the select to `%m`. `oge`, `olt` and `ole`
// are the `*_zero_retains_compare` cases.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @greater_than_zero_keeps_its_compare(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[C:.*]] = spyreop.compare <greaterthan> %[[M]], %{{.*}} : f32
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @greater_than_zero_keeps_its_compare(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                               %q: tensor<8xf32>) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf ogt, %mask, %zero : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: only a comparison against ZERO folds, not one against any constant.
//
// `m != 2.0` keeps its compare. This catches a rule keyed on "compares against
// a constant". The checks bind the compare to `%m` and the select to the
// compare, as in the previous case.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @nonzero_constant_keeps_its_compare(
// CHECK:           ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[C:.*]] = spyreop.compare <notequal> %[[M]], %{{.*}} : f32
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @nonzero_constant_keeps_its_compare(%m: tensor<8xf32>, %p: tensor<8xf32>,
                                              %q: tensor<8xf32>) -> tensor<8xf32> {
  %two = arith.constant 2.0 : f32
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%mask: f32, %t: f32, %f: f32, %o: f32):
    %c = arith.cmpf one, %mask, %two : f32
    %s = arith.select %c, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: one comparison read by both a cast and a select is fully lowered.
//
// Each consumer's rule builds its own `spyreop.compare`, so the body has two:
//   - `%f`, from the cast rule, used as the select's first value;
//   - `%c`, from the select rule, used as its condition.
//
// The CHECK-NEXT chain pins that body exactly, so a surviving `arith.cmpf` or
// `arith.select` fails it. This test runs no common-subexpression elimination;
// one run afterwards could merge the two compares.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @one_compare_two_consumers(
// CHECK:           ^bb0(%[[A:.*]]: f16, %[[B:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:        %[[F:.*]] = spyreop.compare <equal> %[[A]], %[[B]] : f16
// CHECK-NEXT:        %[[C:.*]] = spyreop.compare <equal> %[[A]], %[[B]] : f16
// CHECK-NEXT:        %[[S:.*]] = spyreop.select %[[C]], %[[F]], %[[B]] : f16
// CHECK-NEXT:        linalg.yield %[[S]] : f16
func.func @one_compare_two_consumers(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %c = arith.cmpf oeq, %a, %b : f16
    %f = arith.uitofp %c : i1 to f16
    %s = arith.select %c, %f, %b : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// PURPOSE: a condition passed in as an `i1` block argument is left alone, and
// QUIETLY: no rewrite and no diagnostic.
//
//   - No rewrite: there is no float for `spyreop.select` to take, and inventing
//     one would mean guessing how true is encoded.
//   - No diagnostic: an `i1` block argument is the tensor form crossing a body
//     boundary, which FuseGenerics reports, not this pass.
//
// The next case is the one that IS reported.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @condition_from_outside_is_left_alone(
// CHECK:           ^bb0(%[[C:.*]]: i1, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:        %[[S:.*]] = arith.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT:        linalg.yield %[[S]] : f32
func.func @condition_from_outside_is_left_alone(%c: tensor<8xi1>, %p: tensor<8xf32>,
                                              %q: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%c, %p, %q : tensor<8xi1>, tensor<8xf32>, tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%cond: i1, %t: f32, %f: f32, %o: f32):
    %s = arith.select %cond, %t, %f : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// PURPOSE: an integer ternary is refused rather than mislowered.
//
// `spyreop.select` exists only for floats, so the rule declines an i32 select.
// This differs from the previous case:
//   - previous: the `i1` is a block argument -> quiet decline;
//   - here: the `i1` is made by an `arith.cmpf` in this body and read in it
//     -> reported by rejectSurvivingBooleans.
#map = affine_map<(d0) -> (d0)>
func.func @integer_values_are_refused(%a: tensor<8xf32>, %b: tensor<8xf32>,
                                      %p: tensor<8xi32>, %q: tensor<8xi32>) -> tensor<8xi32> {
  %init = tensor.empty() : tensor<8xi32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<8xf32>, tensor<8xf32>, tensor<8xi32>, tensor<8xi32>)
      outs(%init : tensor<8xi32>) {
  ^bb0(%x: f32, %y: f32, %t: i32, %f: i32, %o: i32):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'ogt' does have a spyreop.compare counterpart}}
    %c = arith.cmpf ogt, %x, %y : f32
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %c, %t, %f : i32
    linalg.yield %s : i32
  } -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----

// Uniform zero is recognized on either side; the selected condition remains
// the input element and equality exchanges the values. Negative zero is zero.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @une_left_zero_f16_scalar(
// CHECK: ^bb0(%[[M:.*]]: f16, %[[P:.*]]: f16, %[[Q:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[M]], %[[P]], %[[Q]] : f16
// CHECK-NEXT: linalg.yield %[[S]] : f16
func.func @une_left_zero_f16_scalar(%m: tensor<4xf16>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
    %zero = arith.constant 0.0 : f16
    %init = tensor.empty() : tensor<4xf16>
    %r = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]}
        ins(%m, %p, %q : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
      ^bb0(%mask: f16, %t: f16, %f: f16, %out: f16):
        %c = arith.cmpf une, %zero, %mask : f16
        %s = arith.select %c, %t, %f : f16
        linalg.yield %s : f16
    } -> tensor<4xf16>
    return %r : tensor<4xf16>
}

// -----

// Uniform zero is recognized on either side; the selected condition remains
// the input element and equality exchanges the values. Negative zero is zero.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @oeq_left_zero_f16_scalar(
// CHECK: ^bb0(%[[M:.*]]: f16, %[[P:.*]]: f16, %[[Q:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[M]], %[[Q]], %[[P]] : f16
// CHECK-NEXT: linalg.yield %[[S]] : f16
func.func @oeq_left_zero_f16_scalar(%m: tensor<4xf16>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
    %zero = arith.constant -0.0 : f16
    %init = tensor.empty() : tensor<4xf16>
    %r = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]}
        ins(%m, %p, %q : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
      ^bb0(%mask: f16, %t: f16, %f: f16, %out: f16):
        %c = arith.cmpf oeq, %zero, %mask : f16
        %s = arith.select %c, %t, %f : f16
        linalg.yield %s : f16
    } -> tensor<4xf16>
    return %r : tensor<4xf16>
}

// -----

// Uniform zero is recognized on either side; the selected condition remains
// the input element and equality exchanges the values. Negative zero is zero.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @one_right_zero_f16_input(
// CHECK: ^bb0(%[[M:.*]]: f16, %[[P:.*]]: f16, %[[Q:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[M]], %[[P]], %[[Q]] : f16
// CHECK-NEXT: linalg.yield %[[S]] : f16
func.func @one_right_zero_f16_input(%m: tensor<4xf16>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
    %zero = arith.constant dense<0.0> : tensor<4xf16>
    %init = tensor.empty() : tensor<4xf16>
    %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
        ins(%m, %p, %q, %zero : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
      ^bb0(%mask: f16, %t: f16, %f: f16, %z: f16, %out: f16):
        %c = arith.cmpf one, %mask, %z : f16
        %s = arith.select %c, %t, %f : f16
        linalg.yield %s : f16
    } -> tensor<4xf16>
    return %r : tensor<4xf16>
}

// -----

// Uniform zero is recognized on either side; the selected condition remains
// the input element and equality exchanges the values. Negative zero is zero.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ueq_left_zero_f32_input(
// CHECK: ^bb0(%[[M:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[M]], %[[Q]], %[[P]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @ueq_left_zero_f32_input(%m: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
    %zero = arith.constant dense<-0.0> : tensor<4xf32>
    %init = tensor.empty() : tensor<4xf32>
    %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
        ins(%m, %p, %q, %zero : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
      ^bb0(%mask: f32, %t: f32, %f: f32, %z: f32, %out: f32):
        %c = arith.cmpf ueq, %z, %mask : f32
        %s = arith.select %c, %t, %f : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
}

// -----

// Ordering against zero must retain the comparison; a nonzero test is not
// equivalent for negative inputs or the equality boundary.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @oge_zero_retains_compare(
// CHECK: ^bb0(%[[M:.*]]: f16, %[[P:.*]]: f16, %[[Q:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <greaterequal> %[[M]], %{{.*}} : f16
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f16
// CHECK-NEXT: linalg.yield %[[S]] : f16
func.func @oge_zero_retains_compare(%m: tensor<4xf16>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<4xf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
    ^bb0(%mask: f16, %t: f16, %f: f16, %out: f16):
      %c = arith.cmpf oge, %mask, %zero : f16
      %s = arith.select %c, %t, %f : f16
      linalg.yield %s : f16
  } -> tensor<4xf16>
  return %r : tensor<4xf16>
}

// -----

// Ordering against zero must retain the comparison; a nonzero test is not
// equivalent for negative inputs or the equality boundary.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @olt_zero_retains_compare(
// CHECK: ^bb0(%[[M:.*]]: f16, %[[P:.*]]: f16, %[[Q:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <lesserthan> %[[M]], %{{.*}} : f16
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f16
// CHECK-NEXT: linalg.yield %[[S]] : f16
func.func @olt_zero_retains_compare(%m: tensor<4xf16>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<4xf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
    ^bb0(%mask: f16, %t: f16, %f: f16, %out: f16):
      %c = arith.cmpf olt, %mask, %zero : f16
      %s = arith.select %c, %t, %f : f16
      linalg.yield %s : f16
  } -> tensor<4xf16>
  return %r : tensor<4xf16>
}

// -----

// Ordering against zero must retain the comparison; a nonzero test is not
// equivalent for negative inputs or the equality boundary.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ole_zero_retains_compare(
// CHECK: ^bb0(%[[M:.*]]: f16, %[[P:.*]]: f16, %[[Q:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <lesserequal> %[[M]], %{{.*}} : f16
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f16
// CHECK-NEXT: linalg.yield %[[S]] : f16
func.func @ole_zero_retains_compare(%m: tensor<4xf16>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<4xf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%m, %p, %q : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
    ^bb0(%mask: f16, %t: f16, %f: f16, %out: f16):
      %c = arith.cmpf ole, %mask, %zero : f16
      %s = arith.select %c, %t, %f : f16
      linalg.yield %s : f16
  } -> tensor<4xf16>
  return %r : tensor<4xf16>
}

// -----

// A tensor containing both zero and nonzero elements is not a uniform zero.
// Retain the input element comparison rather than folding all lanes together.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @nonuniform_zero_input_retains_compare(
// CHECK: ^bb0(%[[M:.*]]: f16, %[[P:.*]]: f16, %[[Q:.*]]: f16, %[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <notequal> %[[X]], %[[M]] : f16
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f16
// CHECK-NEXT: linalg.yield %[[S]] : f16
func.func @nonuniform_zero_input_retains_compare(%m: tensor<4xf16>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
  %mixed = arith.constant dense<[0.0, 1.0, -0.0, -1.0]> : tensor<4xf16>
  %init = tensor.empty() : tensor<4xf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%m, %p, %q, %mixed : tensor<4xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
    ^bb0(%mask: f16, %t: f16, %f: f16, %mixed_value: f16, %out: f16):
      %c = arith.cmpf une, %mixed_value, %mask : f16
      %s = arith.select %c, %t, %f : f16
      linalg.yield %s : f16
  } -> tensor<4xf16>
  return %r : tensor<4xf16>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @oeq_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <equal> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @oeq_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf oeq, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @oge_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <greaterequal> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @oge_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf oge, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @olt_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <lesserthan> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @olt_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf olt, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ole_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <lesserequal> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @ole_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf ole, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ueq_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <equal> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @ueq_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf ueq, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ugt_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <greaterthan> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @ugt_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf ugt, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @uge_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <greaterequal> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @uge_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf uge, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ult_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <lesserthan> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @ult_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf ult, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nonzero-input selection uses the predicate's numeric comparison result.
#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ule_two_inputs_select(
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[C:.*]] = spyreop.compare <lesserequal> %[[A]], %[[B]] : f32
// CHECK-NEXT: %[[S:.*]] = spyreop.select %[[C]], %[[P]], %[[Q]] : f32
// CHECK-NEXT: linalg.yield %[[S]] : f32
func.func @ule_two_inputs_select(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      %c = arith.cmpf ule, %x, %y : f32
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// A comparison cannot supply a numeric mask at a different selected width.
#map = affine_map<(d0) -> (d0)>
func.func @select_width_mismatch(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf16>, %q: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
    ^bb0(%x: f32, %y: f32, %t: f16, %f: f16, %out: f16):
      // expected-error @below {{an i1 value survives inside a compute body}}
      // expected-note @below {{the predicate 'oeq' does have a spyreop.compare counterpart}}
      %c = arith.cmpf oeq, %x, %y : f32
      // expected-note @below {{read here, by 'arith.select'}}
      %v = arith.select %c, %t, %f : f16
      linalg.yield %v : f16
  } -> tensor<4xf16>
  return %r : tensor<4xf16>
}

// -----

// NaN classification has no numeric device comparison counterpart.
#map = affine_map<(d0) -> (d0)>
func.func @ord_select_is_refused(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      // expected-error @below {{an i1 value survives inside a compute body}}
      // expected-note @below {{the predicate 'ord' has no spyreop.compare counterpart}}
      %c = arith.cmpf ord, %x, %y : f32
      // expected-note @below {{read here, by 'arith.select'}}
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// NaN classification has no numeric device comparison counterpart.
#map = affine_map<(d0) -> (d0)>
func.func @uno_select_is_refused(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      // expected-error @below {{an i1 value survives inside a compute body}}
      // expected-note @below {{the predicate 'uno' has no spyreop.compare counterpart}}
      %c = arith.cmpf uno, %x, %y : f32
      // expected-note @below {{read here, by 'arith.select'}}
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Constant predicates left in a body have no device comparison counterpart.
#map = affine_map<(d0) -> (d0)>
func.func @false_select_is_refused(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      // expected-error @below {{an i1 value survives inside a compute body}}
      // expected-note @below {{the predicate 'false' has no spyreop.compare counterpart}}
      %c = arith.cmpf false, %x, %y : f32
      // expected-note @below {{read here, by 'arith.select'}}
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Constant predicates left in a body have no device comparison counterpart.
#map = affine_map<(d0) -> (d0)>
func.func @true_select_is_refused(%a: tensor<4xf32>, %b: tensor<4xf32>, %p: tensor<4xf32>, %q: tensor<4xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%a, %b, %p, %q : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %t: f32, %f: f32, %out: f32):
      // expected-error @below {{an i1 value survives inside a compute body}}
      // expected-note @below {{the predicate 'true' has no spyreop.compare counterpart}}
      %c = arith.cmpf true, %x, %y : f32
      // expected-note @below {{read here, by 'arith.select'}}
      %v = arith.select %c, %t, %f : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  return %r : tensor<4xf32>
}
