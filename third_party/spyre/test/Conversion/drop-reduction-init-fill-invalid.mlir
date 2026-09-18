// RUN: spyre-triton-opt %s --drop-reduction-init-fill -split-input-file -verify-diagnostics

// Reductions this pass IS responsible for — single ins, single init, simple body —
// but whose stated init cannot be discarded soundly. Both alternatives would be
// wrong: dropping it discards an init the IR states, and passing it through hits
// ConstructThreeStagePipeline's one-compute-op assertion in the scheduler. Such a
// reduction cannot currently be lowered correctly at all, so the diagnostic is the
// only honest outcome, and it is worth more here than a wrong answer several
// passes later.
//
// Two ways to land here, and they get different messages:
//
//   the COMBINER is one MapReductionPartials derives no neutral for, so nothing
//   downstream re-establishes the accumulator at all;
//   the combiner is fine but the FILL VALUE is not that combiner's neutral, so
//   dropping it loses an init the IR meant.
//
// NOTE the difference from the cases in drop-reduction-init-fill.mlir that are
// merely SKIPPED (matmul, compound body, multi-init, elementwise). Those are not
// this pass's ops at all, so it stays quiet and lets whatever cannot lower them
// speak. Here the op is ours and the answer is no.
//
// The successful rewrites are in drop-reduction-init-fill.mlir.

// An INTEGER add. Its neutral is zero and the fill states zero, so a
// neutral-value-only gate would drop this happily. The scheduler does have an
// integer branch in its combiner table, but no integer reduction in this tree has
// been carried to a binary, so nothing has shown the rewrite is sound there — and
// an early diagnostic beats finding out in the scheduler. Restricting the table
// this pass mirrors to its FLOAT entries is what produces this message.
module {
func.func @integer_add_reduction(%a: tensor<2x256x64xi32>) -> tensor<2x64xi32> {
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<2x64xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x64xi32>) -> tensor<2x64xi32>
  // expected-error @below {{reduction 'outs' operand #1 is combined with 'arith.addi', which the dataflow-scheduler derives no neutral element for}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xi32>) outs(%init : tensor<2x64xi32>) dimensions = [1]
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }
  return %r : tensor<2x64xi32>
}
}

// -----

// arith.maxnumf, the spelling Triton's tl.max emits. Its NORMALIZED form
// (arith.maximumf) is accepted — see drop-reduction-init-fill.mlir — and this one
// is not, deliberately: the scheduler lowers maxnumf to vectorchain `abs_max`, a
// magnitude comparison, so a reduction carrying it computes the wrong answer
// whatever this pass does with its init. Accepting it here would hide that behind
// a successful compile. NormalizeFloatMinMax runs ahead of this pass in the
// pipeline precisely so this case does not arise; reaching it means that pass was
// skipped.
module {
func.func @max_reduction_unnormalized(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %neg_inf = arith.constant 0xFC00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%neg_inf : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is combined with 'arith.maxnumf', which the dataflow-scheduler derives no neutral element for}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.maxnumf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// An accumulation onto a bias: addf IS an admitted combiner, so we get past that
// check and the fill VALUE is what is rejected. This is the case that keeps the
// two diagnostics distinct.
module {
func.func @biased_accumulator(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %bias = arith.constant 2.500000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%bias : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is initialised by a linalg.fill of a value that is not 'arith.addf's neutral element}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// A max reduce whose init is a CLAMP rather than a neutral — a floor of 0.0, the
// shape of a fused relu. maximumf is an admitted combiner, so this too gets past
// the first check and is refused on the value: 0.0 is not -inf, and the scheduler
// would overwrite the floor with its own neutral and return the unclamped max.
// This is the same wrong the biased accumulator above is, on the other combiner,
// and it is what makes the value check earn its keep now that it is no longer
// simply `is it zero`.
module {
func.func @clamped_max(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %floor = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%floor : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is initialised by a linalg.fill of a value that is not 'arith.maximumf's neutral element}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %m = arith.maximumf %in, %acc : f16
      linalg.yield %m : f16
    }
  return %r : tensor<2x64xf16>
}
}
