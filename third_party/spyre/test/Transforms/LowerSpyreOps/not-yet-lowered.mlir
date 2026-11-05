// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics

// REFUSED TODAY, WHERE A FUTURE RULE COULD LOWER IT.
//
// Each case leaves an `i1` in a compute body, so rejectSurvivingBooleans reports
// it, as in invalid.mlir. Unlike invalid.mlir's cases, each compare has a
// spyreop.compare counterpart and each reader a group shape: what declines is
// that the compare has more than one reader. A case moves out of this file, into
// compare-cast.mlir or compare-select.mlir, when a rule lowers it.

//===----------------------------------------------------------------------===//
// More than one reader
//===----------------------------------------------------------------------===//

// matchLowerableGroup takes a candidate with exactly one reader, so a compare
// read by two ops is not lowered, whatever the readers are.

// One comparison read by two casts. A group takes exactly one reader, so the
// compare and both casts stay arith and the i1 is reported. Pins the decline:
// one spyreop.compare could replace both casts, a possible extension noted at
// SelectCompareGroup.
// Tracked by #254.
func.func @one_compare_two_casts(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> (d0)>],
                       iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{but the compare is read by 2 ops, and a compare is lowered only together with exactly one reader}}
    %c = arith.cmpf olt, %a, %b : f16
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f0 = arith.uitofp %c : i1 to f16
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f1 = arith.uitofp %c : i1 to f16
    %m = arith.mulf %f0, %f1 : f16
    linalg.yield %m : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// PURPOSE: one comparison read by two selects is not lowered.
//
// A group takes exactly one reader, so the compare and both selects stay arith
// and the `i1` is reported. Pins the decline: one `spyreop.compare` could feed
// both `spyreop.select`s, a possible extension noted at SelectCompareGroup.
// Tracked by #254.
#map = affine_map<(d0) -> (d0)>
func.func @one_compare_two_selects(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{but the compare is read by 2 ops, and a compare is lowered only together with exactly one reader}}
    %c = arith.cmpf ogt, %a, %b : f16
    // expected-note @below {{read here, by 'arith.select'}}
    %s0 = arith.select %c, %a, %b : f16
    // expected-note @below {{read here, by 'arith.select'}}
    %s1 = arith.select %c, %b, %s0 : f16
    linalg.yield %s1 : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// PURPOSE: one comparison read by both a cast and a select is not lowered.
//
// A group takes exactly one reader, so the compare and both readers stay arith
// and the `i1` is reported. The in-body form of FuseGenerics/invalid.mlir's
// @both_readers. Pins the decline: one `spyreop.compare` mask could serve both
// readers, a possible extension noted at SelectCompareGroup.
// Tracked by #254.
#map = affine_map<(d0) -> (d0)>
func.func @one_compare_two_consumers(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'oeq' does have a spyreop.compare counterpart}}
    %c = arith.cmpf oeq, %a, %b : f16
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %c : i1 to f16
    // expected-note @below {{read here, by 'arith.select'}}
    %s = arith.select %c, %f, %b : f16
    linalg.yield %s : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}
