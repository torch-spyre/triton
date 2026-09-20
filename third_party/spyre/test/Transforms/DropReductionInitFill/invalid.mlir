// RUN: spyre-triton-opt %s --drop-reduction-init-fill -split-input-file -verify-diagnostics

// Reductions this pass IS responsible for — single ins, single init, simple body —
// but whose stated init cannot be discarded soundly. Both alternatives would be
// wrong: dropping it discards an init the IR states, and passing it through hits
// ConstructThreeStagePipeline's one-compute-op assertion in the scheduler. So the
// diagnostic is the only honest outcome, and it is worth more here than a wrong
// answer several passes later.
//
// The REASON is per-combiner and the messages say so. It is not that the
// scheduler's accumulator reset is a hardcoded zero whatever this pass does --
// MapReductionPartials' lowerIterArgInitializer asks getNeutralAttr and fills with
// the per-combiner answer. Each arm below is refused for its own reason: a neutral
// the scheduler will not write for mulf, a mis-lowering for maxnumf, and a
// float-attribute reset that rejects the type outright for the integer ops.
//
// NOTE the difference from the cases in drop-reduction-init-fill.mlir that are
// merely SKIPPED (matmul, compound body, multi-init, elementwise). Those are not
// this pass's ops at all, so it stays quiet and lets whatever cannot lower them
// speak. Here the op is ours and the answer is no.
//
// The successful rewrites are in drop-reduction-init-fill.mlir.

// A mulf reduction. Its neutral IS 1.0 — correctly stated by the producer — and
// dropping it is still wrong, because nothing downstream restores a 1.0 to an
// `outs` this pass has emptied. The combiner is what decides that, so that is what
// the message names.
module {
func.func @mul_reduction_neutral_is_one(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %one = arith.constant 1.000000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%one : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is combined with 'arith.mulf', which the dataflow scheduler does not lower correctly against a dropped neutral element}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.mulf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// A max reduction: neutral -inf. This is the softmax/layernorm path, and the
// reason the combiner check exists rather than only a fill-value check.
module {
func.func @max_reduction_neutral_is_neg_inf(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %neg_inf = arith.constant 0xFC00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%neg_inf : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is combined with 'arith.maximumf', which the dataflow scheduler does not lower correctly against a dropped neutral element}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.maximumf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}

// -----

// An INTEGER add. Its neutral IS zero and the fill states zero, so the old
// zero-only gate dropped this happily — and then the scheduler aborted, because it
// builds the reset as a float attribute, which rejects a non-float type outright.
// Restricting the combiner allowlist to the float ops turns that abort into this
// diagnostic. The most visible reason the message cannot talk about neutrals: this
// one's neutral is exactly the value the old wording said it was not.
module {
func.func @integer_add_reduction(%a: tensor<2x256x64xi32>) -> tensor<2x64xi32> {
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<2x64xi32>
  %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<2x64xi32>) -> tensor<2x64xi32>
  // expected-error @below {{reduction 'outs' operand #1 is combined with 'arith.addi', which the dataflow scheduler does not lower correctly against a dropped neutral element}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xi32>) outs(%init : tensor<2x64xi32>) dimensions = [1]
    (%in: i32, %acc: i32) {
      %s = arith.addi %in, %acc : i32
      linalg.yield %s : i32
    }
  return %r : tensor<2x64xi32>
}
}

// -----

// An accumulation onto a bias: addf IS an allowed combiner, so we get past that
// check and the fill value is what is rejected. This is the case that keeps the
// non-zero-fill diagnostic distinct from the combiner one.
module {
func.func @biased_accumulator(%a: tensor<2x256x64xf16>) -> tensor<2x64xf16> {
  %bias = arith.constant 2.500000e+00 : f16
  %empty = tensor.empty() : tensor<2x64xf16>
  %init = linalg.fill ins(%bias : f16) outs(%empty : tensor<2x64xf16>) -> tensor<2x64xf16>
  // expected-error @below {{reduction 'outs' operand #1 is initialised by a linalg.fill of a non-zero value}}
  %r = linalg.reduce ins(%a : tensor<2x256x64xf16>) outs(%init : tensor<2x64xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  return %r : tensor<2x64xf16>
}
}
