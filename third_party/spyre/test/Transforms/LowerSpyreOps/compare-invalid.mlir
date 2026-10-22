// RUN: spyre-triton-opt %s --lower-spyre-ops -split-input-file -verify-diagnostics

// AN i1 LEFT IN A COMPUTE BODY IS REFUSED, and it is one of the two things this
// pass reports; the other is an intrinsic request left unselected, which
// request-invalid.mlir covers.
//
// Everything else unmatched flows through, because it is a CAPABILITY gap -- an
// f64 `math.sqrt` is valid IR that a future device or a future rule may do, so the
// backend is the right judge, and unsupported-types.mlir is that claim. An `i1` is
// a different kind of thing: no spyreop intrinsic produces or consumes that type,
// so no rule anyone could add would ever remove it. That is a property of the
// dialect rather than of a device generation, and it makes the program
// unlowerable, not merely unsupported.
//
// Reported HERE rather than downstream because this is the only place that still
// knows the PREDICATE. By the time the backend sees it, the failure names an op
// several lowerings below the one the author wrote.
//
// Two kinds of way in, and the diagnostic distinguishes them: a predicate with no
// counterpart at all, and a predicate that HAS one where the READER was the
// problem. The note says which, because that is what an author can act on.
//
// THE SCOPE IS NARROWER THAN "any i1 in a body", and deliberately. The check reads
// op RESULTS only, and skips one that is yielded -- so what it refuses is an i1
// this pass produced and KEPT, made inside one body and read inside the same body.
// That is exactly a group it had the chance to select and did not.
//
// An i1 crossing a generic boundary is the other pass's finding, not this one's: it
// is the TENSOR form, FuseComputeAndDataMovement has the clause that removes it,
// and before that pass has run every compare sits in a generic of its own yielding
// precisely that. Refusing it here would report IR a predecessor was meant to
// reshape while naming the wrong cause. compare-from-tensor.mlir's NOFUSE run line
// is that claim.

//===----------------------------------------------------------------------===//
// No counterpart for the predicate
//===----------------------------------------------------------------------===//

// `ord` asks whether neither operand is NaN, which a comparison intrinsic does
// not answer directly. The unordered orderings (`une`, `ugt`, ...) are NOT here: under
// the pass's no-NaN assumption each computes what its ordered counterpart does,
// and compare.mlir's `all_unordered_predicates` pins that they are selected.
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

//===----------------------------------------------------------------------===//
// The predicate was fine; the reader was not
//===----------------------------------------------------------------------===//

// `sitofp` rather than `uitofp`: an `i1` read as SIGNED is 0 or -1, so this gives
// -1.0 where the predicate holds. A different computation, so the pair is not one
// intrinsic -- and the note points at the reader, since the predicate is not what
// needs changing.
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

// A compare at one width cast to another. `spyreop.compare` has
// SameOperandsAndResultType, so it cannot compare f32 and answer in f16 -- the
// note's "at that same width" is the actionable part.
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
