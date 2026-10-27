// RUN: spyre-triton-opt %s --convert-elementwise-to-linalg --canonicalize-generic-operands --fuse-generics --lower-spyre-ops | FileCheck %s --check-prefix=FROMTENSOR
// The rule fires ONLY because the fusion pass ran. Driven without it, the group is
// still two generics and nothing is selected -- which is the claim, so it is a run
// line rather than a sentence.
// RUN: spyre-triton-opt %s --convert-elementwise-to-linalg --lower-spyre-ops | FileCheck %s --check-prefix=NOFUSE

// The compare rule: a comparison whose answer is wanted as a NUMBER rather than as
// a flag. `arith.cmpf` + `arith.uitofp` is the mask shape -- `(m != 0)` used
// multiplicatively -- and `spyreop.compare` is one op that does it, its answer
// coming "in the width compared rather than as a boolean".
//
// Why the pair is one choice and not two: `arith.cmpf` alone gives an `i1`, which
// no spyreop op produces and the device cannot represent in a compute body, so the
// compare is not selectable by itself. What consumes the `i1` decides what the
// pair becomes.
//
// THE SHAPE THE PIPELINE REALLY PRODUCES, which is why this case is in a file of
// its own: a rule matches ops in ONE body, and ConvertElementwiseToLinalg gives
// every tensor-level op a body of its own, so this group starts out spread over TWO
// generics with a `tensor<i1>` between them. No rule can see that.
// FuseGenerics is what brings it together.
//
// One module per file here, because the interesting claim is "ONE generic comes
// out" and a -NOT directive asserting that cannot be scoped to one module in a
// -split-input-file run: it would scan on into the next.
//
// compare.mlir hand-builds the fused body instead, isolating each rule decision
// from whether fusion happened; compare-invalid.mlir holds the refusals.

// Two tensor ops, and therefore two generics with a `tensor<4xi1>` between them
// until the fusion pass fuses across the `i1`. Out comes ONE generic holding one
// intrinsic, and no `i1` of any kind -- neither as a tensor nor in a body.
//
// FROMTENSOR-LABEL: func.func @from_tensor_mask(
// FROMTENSOR-NOT:     tensor<4xi1>
// FROMTENSOR-NOT:     arith.cmpf
// FROMTENSOR-NOT:     arith.uitofp
// FROMTENSOR:         %[[C:.*]] = arith.constant 0.000000e+00 : f16
// FROMTENSOR:         linalg.generic
// FROMTENSOR:           spyreop.compare <notequal> %{{.*}}, %[[C]] : f16
// FROMTENSOR-NOT:     linalg.generic
func.func @from_tensor_mask(%m: tensor<4xf16>) -> tensor<4xf16> {
  %zero = arith.constant dense<0.0> : tensor<4xf16>
  %c = arith.cmpf one, %m, %zero : tensor<4xf16>
  %f = arith.uitofp %c : tensor<4xi1> to tensor<4xf16>
  return %f : tensor<4xf16>
}

// NOFUSE, the control. Without the fusion pass the group is still two generics with
// a `tensor<4xi1>` between them, nothing is selected, and -- importantly -- nothing
// is REFUSED either: the `i1` check walks generic BODIES, and here the i1 is a
// tensor flowing between two of them rather than a value inside one. So the pass is
// silent, which is the right answer for IR that a pass ahead of it was meant to
// reshape.
//
// NOFUSE-NOT: spyreop.compare
// NOFUSE: linalg.generic
// NOFUSE: arith.cmpf one
// NOFUSE: linalg.generic
// NOFUSE: arith.uitofp
