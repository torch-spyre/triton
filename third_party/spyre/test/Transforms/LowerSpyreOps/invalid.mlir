// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics

// AN i1 LEFT IN A COMPUTE BODY IS REFUSED, and it is the one thing this pass
// reports.
//
// THIS FILE HOLDS WHAT CAN NEVER LOWER, each case for a reason in the spyreop
// dialect itself: a predicate with no spyreop.compare counterpart, a reader that
// computes something else, a compare and reader at different widths, or a select
// of a type spyreop.select does not take. The last section is the one exception:
// a reader outside the group's generic body, which is the scope and not the
// dialect. not-yet-lowered.mlir holds the refusals a future rule could remove: a
// compare with more than one reader.
//
// Everything else unmatched flows through, because it is a CAPABILITY gap -- an
// f64 `math.sqrt` is valid IR that a future device or a future rule may do, so the
// backend is the right judge, and pass-through.mlir is that claim. An `i1` is
// a different kind of thing: no spyreop intrinsic produces or consumes that type,
// so no rule anyone could add would ever remove it. That is a property of the
// dialect rather than of a device generation, and it makes the program
// unlowerable, not merely unsupported.
//
// Reported HERE rather than downstream because this is the only place that still
// knows the PREDICATE. By the time the backend sees it, the failure names an op
// several lowerings below the one the author wrote.
//
// Three kinds of way in, and the diagnostic distinguishes them: a predicate with
// no counterpart at all, a predicate that HAS one where the READER was the
// problem (both in this file), and a compare read by more than one op (in
// not-yet-lowered.mlir). The note says which, because that is what an author can
// act on.
//
// THE SCOPE IS NARROWER THAN "any i1 in a body", and deliberately. The check reads
// op RESULTS only, and skips one that is yielded -- so what it refuses is an i1
// this pass produced and KEPT, made inside one body and read inside the same body.
// That is exactly a group it had the chance to select and did not.
//
// An i1 crossing a generic boundary is the other pass's finding, not this one's: it
// is the TENSOR form, FuseGenerics reports it,
// and before that pass has run every compare sits in a generic of its own yielding
// precisely that. Refusing it here would report IR a predecessor was meant to
// reshape while naming the wrong cause. compare-from-tensor.mlir's NOFUSE run line
// is that claim.

//===----------------------------------------------------------------------===//
// No counterpart for the predicate, read by arith.uitofp (compare-cast)
//===----------------------------------------------------------------------===//

// `ord` asks whether neither operand is NaN, which a comparison intrinsic does
// not answer directly. The unordered orderings (`une`, `ugt`, ...) are NOT here: under
// the pass's no-NaN assumption each computes what its ordered counterpart does,
// and compare-cast.mlir's `all_unordered_predicates` pins that they are selected.
func.func @ord_has_no_counterpart(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'ord' has no spyreop.compare counterpart}}
    %c = arith.cmpf ord, %a, %b : f16
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// Unorderedness cannot be converted to a device numeric comparison.
func.func @uno_has_no_counterpart(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'uno' has no spyreop.compare counterpart}}
    %c = arith.cmpf uno, %a, %b : f16
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// Constant predicates retained in a body have no device comparison counterpart.
func.func @false_has_no_counterpart(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'false' has no spyreop.compare counterpart}}
    %c = arith.cmpf false, %a, %b : f16
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// Constant predicates retained in a body have no device comparison counterpart.
func.func @true_has_no_counterpart(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'true' has no spyreop.compare counterpart}}
    %c = arith.cmpf true, %a, %b : f16
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

//===----------------------------------------------------------------------===//
// No counterpart for the predicate, read by arith.select (compare-select)
//===----------------------------------------------------------------------===//

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

// -----

//===----------------------------------------------------------------------===//
// The predicate has a counterpart; the reader computes something else
//===----------------------------------------------------------------------===//

// Never lowerable: an i1 read as signed is 0 or -1, so `arith.sitofp` gives -1.0
// where the predicate holds, which no spyreop.compare result is. The pair is not
// one intrinsic, and the note points at the reader, since the predicate is not
// what needs changing.
func.func @sitofp_reader(%x: tensor<4xf16>, %y: tensor<4xf16>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'oeq' does have a spyreop.compare counterpart}}
    %c = arith.cmpf oeq, %a, %b : f16
    // expected-note @below {{read here, by 'arith.sitofp'}}
    %f = arith.sitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

//===----------------------------------------------------------------------===//
// The compare and its reader are at different widths
//===----------------------------------------------------------------------===//

// Never lowerable: a compare at one width cast to another. spyreop.compare has
// SameOperandsAndResultType, so it cannot compare f32 and answer in f16, and
// spyreop has no float-width conversion. The note's "at that same width" is the
// actionable part.
func.func @width_change(%x: tensor<4xf32>, %y: tensor<4xf32>) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf32>, tensor<4xf32>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f32, %b: f32, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'oeq' does have a spyreop.compare counterpart}}
    %c = arith.cmpf oeq, %a, %b : f32
    // expected-note @below {{read here, by 'arith.uitofp'}}
    %f = arith.uitofp %c : i1 to f16
    linalg.yield %f : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// Never lowerable: spyreop.select takes one type for its condition and values,
// and spyreop has no float-width conversion, so a comparison cannot supply a
// numeric mask at a different selected width.
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

//===----------------------------------------------------------------------===//
// A select of a type spyreop.select does not take
//===----------------------------------------------------------------------===//

// Never lowerable: spyreop.select exists only for floats and spyreop has no
// integer select, so the rule declines an i32 select and the pass refuses it
// rather than mislowering it.
// This differs from compare-select.mlir's @condition_from_outside_is_left_alone:
//   - there: the `i1` is a block argument -> quiet decline;
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

//===----------------------------------------------------------------------===//
// Outside the scope
//===----------------------------------------------------------------------===//

// A reader nested in a region of the body is not part of the group, however
// well it fits: a group's reader must be directly in the group's generic. The
// compare and the select stay arith and the `i1` is reported. This is the scope,
// not a missing rule.
#map = affine_map<(d0) -> (d0)>
func.func @reader_nested_in_scf_if(%x: tensor<4xf16>, %y: tensor<4xf16>, %flag: i1) -> tensor<4xf16> {
  %init = tensor.empty() : tensor<4xf16>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
                       iterator_types = ["parallel"]}
      ins(%x, %y : tensor<4xf16>, tensor<4xf16>) outs(%init : tensor<4xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    // expected-error @below {{an i1 value survives inside a compute body}}
    // expected-note @below {{the predicate 'ogt' does have a spyreop.compare counterpart}}
    %c = arith.cmpf ogt, %a, %b : f16
    %r = scf.if %flag -> f16 {
      // expected-note @below {{read here, by 'arith.select'}}
      %s = arith.select %c, %a, %b : f16
      scf.yield %s : f16
    } else {
      scf.yield %a : f16
    }
    linalg.yield %r : f16
  } -> tensor<4xf16>
  return %0 : tensor<4xf16>
}
