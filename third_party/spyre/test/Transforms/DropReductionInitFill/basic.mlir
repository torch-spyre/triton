// RUN: spyre-triton-opt %s --drop-reduction-init-fill -split-input-file | FileCheck %s

// DropReductionInitFill removes the linalg.fill that LowerComputeOps gives a
// tt.reduce for its accumulator, leaving the bare tensor.empty that hand-written
// reference KTIR states directly.
//
// The rewrite is sound ONLY because MapReductionPartials re-establishes the
// accumulator before it is read — a reduction payload does read its init, and
// tensor.empty is explicitly unspecified. What it writes is the neutral element of
// THAT reduction's combiner (getNeutralAttr), so the combiner is not a reason to
// refuse a reduction and the fill value is not read here at all. The gate is shape
// only: one ins, one init, a body of payload + yield that reads the init. Anything
// else — matmul above all — is left alone, silently.
//
// The pass therefore never fails, and there is no invalid.mlir. Its cases are
// rewrites now, including the one rewrite that is WRONG (Test 14).
//
// Inputs here are written in the already-lowered form the pass actually sees, so
// they do not depend on what the upstream producer happens to emit.

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

// Test 4: subf, whose neutral is also 0.0.
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

// -----

// Test 11: a mulf reduction, neutral 1.0. Rewritten, not refused: getNeutralAttr
// answers 1.0 for mulf, so the accumulator the scheduler establishes is the one
// this fill states. The combiner is not this pass's business.
module {
// CHECK-LABEL:   func.func @mul_reduction_neutral_is_one(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @mul_reduction_neutral_is_one(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %one = arith.constant 1.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%one : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.mulf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 12: a max reduction, neutral -inf — the softmax/layernorm path.
// getNeutralAttr answers -inf for arith.maximumf, so this is a rewrite too.
//
// NOTE what this does NOT claim: Triton spells its max as arith.maxnumf, which
// getNeutralAttr does not match, so the softmax family still fails -- inside
// dbo-opt, with its own `unsupported floating-point reduction combiner`, until a
// pass normalizes the spelling. That is the right place for it: a combiner table
// mirrored here would be one more copy to rot.
module {
// CHECK-LABEL:   func.func @max_reduction_neutral_is_neg_inf(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @max_reduction_neutral_is_neg_inf(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %neg_inf = arith.constant 0xFC00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%neg_inf : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.maximumf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// Test 13: an INTEGER add. getNeutralAttr covers addi/subi/muli, so an integer
// reduction is not special here either — the element type is not something this
// pass reads.
module {
// CHECK-LABEL:   func.func @integer_add_reduction(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @integer_add_reduction(%a: tensor<2x256x64xi32>) -> tensor<2x64xi32> {
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<2x64xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x64xi32>) -> tensor<2x64xi32>
  %r = linalg.reduce ins(%a : tensor<2x256x64xi32>) outs(%init : tensor<2x64xi32>) dimensions = [1]
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }
  return %r : tensor<2x64xi32>
}
}

// -----

// Test 14: the rewrite this pass gets WRONG, pinned deliberately. A SEEDED
// accumulator — addf over a fill of 2.5 — states `2.5 + sum`, and after the
// rewrite it computes `sum`. No neutral element can recover the 2.5, because a
// bias is not a property of the combiner, so widening the gate cannot fix this and
// narrowing it back to zero-fills only would refuse every reduction above for one
// nothing can produce.
//
// Reachable from hand-written KTIR only: tl.reduce / tl.sum / tl.max take no init
// parameter, and tt.reduce's ODS is `ins Variadic<TT_Tensor>:$srcs, I32Attr:$axis`
// with no init operand, so nothing above this pass can express a seeded reduction.
// The fix is downstream honouring the fill, which is also what deletes this pass.
module {
// CHECK-LABEL:   func.func @biased_accumulator(
// CHECK-NOT:       linalg.fill
// CHECK:           linalg.reduce
func.func @biased_accumulator(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %bias = arith.constant 2.500000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%bias : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}
