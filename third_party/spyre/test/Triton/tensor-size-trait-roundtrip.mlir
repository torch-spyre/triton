// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// verifyTensorSize (lib/Dialect/Triton/IR/Traits.cpp) is the C++ half of two
// Spyre relaxations: the element-count cap (TRITON_MAX_TENSOR_NUMEL, 2**20) and
// the requirement that the total be a power of two. Both are GPU register-budget
// artifacts, and the whole trait is compiled out under TRITON_BUILD_TTIR_ONLY.
//
// No pass is needed to reach it. TensorSizeTrait sits on the TT_Op base
// (TritonOps.td:29), so every tt op carries it and parsing this file runs the
// verifier -- "printed back" means "the trait let it through". Which is also why
// the ops here have to be tt.*: func.func and func.return do not carry the trait,
// so writing this with them would pass without reaching what it checks.

// 192 elements: not a power of two. tt.splat carries the trait on its result.
// CHECK-LABEL: tt.func @non_pow2_splat
// CHECK-SAME: tensor<192xf32>
tt.func @non_pow2_splat(%arg0: f32) -> tensor<192xf32> {
  %0 = tt.splat %arg0 : f32 -> tensor<192xf32>
  tt.return %0 : tensor<192xf32>
}

// ---------------------------------------------------------------------------

// 4 * 192 * 64 = 49152: not a power of two, on an operand rather than a result.
// CHECK-LABEL: tt.func @non_pow2_reduce
// CHECK-SAME: tensor<4x192x64xf32>
tt.func @non_pow2_reduce(%arg0: tensor<4x192x64xf32>) -> tensor<4x192xf32> {
  %0 = "tt.reduce"(%arg0) <{axis = 2 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %1 = arith.addf %a, %b : f32
    "tt.reduce.return"(%1) : (f32) -> ()
  }) : (tensor<4x192x64xf32>) -> tensor<4x192xf32>
  tt.return %0 : tensor<4x192xf32>
}

// ---------------------------------------------------------------------------

// 2048 * 4096 = 8Mi, eight times the cap. issue #3's shape, as a block the
// descriptor path takes directly.
// CHECK-LABEL: tt.func @over_cap
// CHECK-SAME: tensor<2048x4096xf16>
tt.func @over_cap(%arg0: tensor<2048x4096xf16>) -> tensor<2048x4096xf16> {
  %0 = tt.broadcast %arg0 : tensor<2048x4096xf16> -> tensor<2048x4096xf16>
  tt.return %0 : tensor<2048x4096xf16>
}

// ---------------------------------------------------------------------------

// 3000 * 3000 = 9,000,000: over the cap *and* not a power of two, so it trips
// both of the trait's limits at once.
// CHECK-LABEL: tt.func @over_cap_and_non_pow2
// CHECK-SAME: tensor<3000x3000xf16>
tt.func @over_cap_and_non_pow2(%arg0: tensor<3000x3000xf16>) -> tensor<3000x3000xf16> {
  %0 = tt.broadcast %arg0 : tensor<3000x3000xf16> -> tensor<3000x3000xf16>
  tt.return %0 : tensor<3000x3000xf16>
}
