// RUN: spyre-triton-opt %s --unalias-linalg-outs -split-input-file -verify-diagnostics

// Rejection cases for --unalias-linalg-outs. An aliased `outs` operand that
// cannot be replaced with tensor.empty is reported as an error rather than
// skipped: leaving the aliasing in place crashes the Spyre dataflow-scheduler
// much later, with nothing pointing back to this pass.
//
// The successful rewrites are in unalias-linalg-outs.mlir.

#map = affine_map<(d0, d1) -> (d0, d1)>

// A generic whose body reads its %out block argument. The initial value of the
// outs operand is live, so tensor.empty (which supplies no initial value) would
// change the result. Not rewritable at any cost — the aliasing has to be fixed
// by whichever pass produced this op.

func.func @body_reads_out(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
// expected-error @below {{linalg 'outs' operand #1 is the same value as an 'ins' operand and the op body reads its initial value}}
  %0 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a : tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// linalg.reduce accumulates into its outs operand: the combiner region's second
// block argument IS the running accumulator. Aliasing it with an ins operand is
// rejected for the same reason as above. This is the case that makes matching on
// the LinalgOp interface (rather than linalg.generic alone) require a guard —
// LowerComputeOps emits linalg.reduce for tt.reduce.

func.func @aliased_reduction(%a: tensor<4xf32>) -> tensor<4xf32> {
// expected-error @below {{is the same value as an 'ins' operand and the op body reads its initial value}}
  %r = linalg.reduce ins(%a : tensor<4xf32>) outs(%a : tensor<4xf32>)
       dimensions = []
    (%in: f32, %acc: f32) {
      %s = arith.addf %in, %acc : f32
      linalg.yield %s : f32
    }
  return %r : tensor<4xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// linalg on memrefs (post-bufferization form) has no results and an aliased
// memref outs would need an allocation, not a tensor.empty. The Spyre pipeline
// runs entirely on tensors, so this is out of scope and rejected rather than
// silently skipped.

func.func @memref_outs(%a: memref<4xf32>) {
// expected-error @below {{aliases an 'ins' operand but has non-tensor type 'memref<4xf32>'; expected a ranked tensor}}
  linalg.generic {indexing_maps = [#map, #map],
                  iterator_types = ["parallel"]}
       ins(%a : memref<4xf32>) outs(%a : memref<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.mulf %in, %in : f32
    linalg.yield %1 : f32
  }
  return
}
