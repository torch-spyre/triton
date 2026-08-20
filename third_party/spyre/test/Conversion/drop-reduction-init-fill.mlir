// RUN: spyre-triton-opt %s --drop-reduction-init-fill -split-input-file | FileCheck %s

// DropReductionInitFill removes the zero linalg.fill that LowerComputeOps gives a
// tt.reduce for its accumulator, leaving the bare tensor.empty that hand-written
// reference KTIR states directly.
//
// The rewrite is sound ONLY because MapReductionPartials overwrites the accumulator
// with its own hardcoded 0.0 reset before it is read — a reduction payload does read
// its init, and tensor.empty is explicitly unspecified. So the gate is narrow: the
// op must be a shape that pass actually rewrites (one ins, one init, simple body),
// the combiner must be addf/subf, and the fill must be zero. Anything else is left
// alone or reported.
//
// Inputs here are written in the already-lowered form the pass actually sees, so
// they do not depend on what the upstream producer happens to emit.
//
// Rejection cases (our op, but its init cannot be honoured) are in
// drop-reduction-init-fill-invalid.mlir.

// Test 1: the shape LowerComputeOps produces for tl.sum — linalg.reduce over the
// middle axis, outs initialised by fill(0.0). The fill goes, the reduce takes the
// empty, and the now-dead zero constant is left for canonicalization.
module {
// CHECK-LABEL:   func.func @sum_reduce(
// CHECK-NOT:       linalg.fill
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xf16>) outs(%[[EMPTY]] : tensor<2x64xf16>) dimensions = [1]
func.func @sum_reduce(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 2: the same init on a linalg.generic carrying a reduction iterator, which
// is the form the scheduler's own reference KTIR uses. Matching the LinalgOp
// interface rather than the op name is what covers both.
module {
// CHECK-LABEL:   func.func @sum_generic(
// CHECK-NOT:       linalg.fill
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           linalg.generic {{.*}} outs(%[[EMPTY]] : tensor<2x64xf16>)
func.func @sum_generic(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) {
  ^bb0(%in: f16, %acc: f16):
    %s = arith.addf %in, %acc : f16
    linalg.yield %s : f16
  } -> tensor<2x64xf16>
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 3: -0.0 is a zero too. Rejecting it would be a needless recompile for an
// input that is numerically identical to the reset the scheduler writes.
module {
// CHECK-LABEL:   func.func @negative_zero(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @negative_zero(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant -0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 4: subf is the other zero-neutral combiner the scheduler lowers correctly.
module {
// CHECK-LABEL:   func.func @sub_reduce(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @sub_reduce(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.subf %acc, %in : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 5: MATMUL — the reason the gate cannot be the fill value alone. A
// contraction is an addf-accumulate reduction whose neutral genuinely IS zero, so
// a zero-only gate would drop this init. But MapReductionPartials never rewrites a
// matmul — its assert is a single `ins` — so nothing resets that accumulator and
// the fill is load-bearing. Skipped silently:
// diagnosing a matmul is not this pass's job, and failing here would stop any
// pipeline that merely contains one.
module {
// CHECK-LABEL:   func.func @matmul_fill_survives(
// CHECK:           linalg.fill
// CHECK:           linalg.matmul
func.func @matmul_fill_survives(%a: tensor<64x128xf16>,
                                %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<64x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %r = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %r : tensor<64x64xf16>
}
}

// -----

// Test 6: a single-input reduction whose body is more than payload + yield —
// sum-of-squares. LinalgLowering maps one payload op to one vectorchain binary op,
// so this is not a shape the scheduler handles; the fill stays.
module {
// CHECK-LABEL:   func.func @compound_body_fill_survives(
// CHECK:           linalg.fill
// CHECK:           linalg.reduce
func.func @compound_body_fill_survives(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %sq = arith.mulf %in, %in : f16
      %s = arith.addf %sq, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 7: a multi-init reduction. MapReductionPartials asserts exactly one output,
// so this is out of scope however zero the fills are.
module {
// CHECK-LABEL:   func.func @multi_init_fill_survives(
// CHECK:           linalg.fill
// CHECK:           linalg.generic
func.func @multi_init_fill_survives(%a: tensor<2x256x64xf16>)
    -> (tensor<2x64xf16>, tensor<2x64xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %e0 = tensor.empty() : tensor<2x64xf16>
  %e1 = tensor.empty() : tensor<2x64xf16>
  %i0 = linalg.fill ins(%zero : f16) outs(%e0 : tensor<2x64xf16>) -> tensor<2x64xf16>
  %i1 = linalg.fill ins(%zero : f16) outs(%e1 : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a : tensor<2x256x64xf16>)
      outs(%i0, %i1 : tensor<2x64xf16>, tensor<2x64xf16>) {
  ^bb0(%in: f16, %acc0: f16, %acc1: f16):
    %s0 = arith.addf %in, %acc0 : f16
    %s1 = arith.addf %in, %acc1 : f16
    linalg.yield %s0, %s1 : f16, f16
  } -> (tensor<2x64xf16>, tensor<2x64xf16>)
  return %r#0, %r#1 : tensor<2x64xf16>, tensor<2x64xf16>
}
}

// -----

// Test 8: an ELEMENTWISE op's fill is out of scope, zero or not. tt.splat lowers
// to linalg.fill, and it is a real initialiser there — no reduction reset covers
// it. The pass must leave it alone rather than treat every fill as redundant.
module {
// CHECK-LABEL:   func.func @elementwise_fill_survives(
// CHECK:           linalg.fill
func.func @elementwise_fill_survives(%a: tensor<2x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %splat = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %out = tensor.empty() : tensor<2x64xf16>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %splat : tensor<2x64xf16>, tensor<2x64xf16>)
      outs(%out : tensor<2x64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %s = arith.addf %x, %y : f16
    linalg.yield %s : f16
  } -> tensor<2x64xf16>
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 9: a fill writing over live data rather than a tensor.empty is left alone.
// Repointing outs at that value would substitute its contents for the stated
// init, which is a different rewrite from dropping a redundant one.
module {
// CHECK-LABEL:   func.func @fill_over_live_data(
// CHECK:           linalg.fill
// CHECK:           linalg.reduce
func.func @fill_over_live_data(%a: tensor<2x256x64xf16>,
                               %live: tensor<2x64xf16>) -> tensor<2x64xf16> {
  %zero = arith.constant 0.000000e+00 : f16
  %init = linalg.fill ins(%zero : f16) outs(%live : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 10: a fill with another user is dropped from the reduction's outs but not
// erased, since something else still needs the filled tensor.
module {
// CHECK-LABEL:   func.func @fill_with_another_user(
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f16) outs(%[[EMPTY]] : tensor<2x64xf16>)
// CHECK:           linalg.reduce ins(%{{.*}} : tensor<2x256x64xf16>) outs(%[[EMPTY]] : tensor<2x64xf16>) dimensions = [1]
// CHECK:           return %[[REDUCED:.*]], %[[FILL]]
func.func @fill_with_another_user(%a: tensor<2x256x64xf16>)
    -> (tensor<2x64xf16>, tensor<2x64xf16>) {
  %zero = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r, %init : tensor<2x64xf16>, tensor<2x64xf16>
}
}
