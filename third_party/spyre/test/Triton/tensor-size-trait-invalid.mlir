// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// Control for tensor-size-trait-roundtrip.mlir. That file shows an over-cap,
// non-power-of-two tensor is printed back, which on its own cannot distinguish
// "verifyTensorSize let it through" from "no verifier ran at all". This shows
// verification is live on the same op at the same size: a shape error on an 8Mi
// tt.broadcast is still diagnosed.

tt.func @broadcast_shape_still_checked(%arg0: tensor<2048x4096xf16>) -> tensor<4096x2048xf16> {
  // expected-error @below {{Different dimensions at index 0 between source and result}}
  %0 = tt.broadcast %arg0 : tensor<2048x4096xf16> -> tensor<4096x2048xf16>
  tt.return %0 : tensor<4096x2048xf16>
}
