// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for tts.spyre_op and its terminator.
//
// The op is consumed in the `ktir` stage, so its printed form is no contract
// with anything outside this tree. What this pins is that the custom assembly
// prints and re-parses: the name, the operand list, the functional type and
// the region with its block arguments.

// The shape tracing produces: the fallback reached through a tt.call, which
// the verifier admits without asking what the callee does.
// CHECK-LABEL: tt.func @traced(
// CHECK:         %[[R:.*]] = tts.spyre_op "sigmoid"(%{{.*}}) : (tensor<128xf32>) -> tensor<128xf32> {
// CHECK:         ^bb0(%[[A:.*]]: tensor<128xf32>):
// CHECK:           %[[C:.*]] = tt.call @fallback(%[[A]])
// CHECK:           tts.spyreop_yield %[[C]] : tensor<128xf32>
// CHECK:         }
// CHECK:         tt.return %[[R]]
tt.func private @fallback(%x: tensor<128xf32>) -> tensor<128xf32> {
  tt.return %x : tensor<128xf32>
}
tt.func @traced(%x: tensor<128xf32>) -> tensor<128xf32> {
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<128xf32>) -> tensor<128xf32> {
  ^bb0(%a: tensor<128xf32>):
    %c = tt.call @fallback(%a) : (tensor<128xf32>) -> tensor<128xf32>
    tts.spyreop_yield %c : tensor<128xf32>
  }
  tt.return %r : tensor<128xf32>
}

// The shape after the inliner: the fallback's ops, constants inside the body.
// CHECK-LABEL: tt.func @inlined(
// CHECK:         tts.spyre_op "silu"(%{{.*}}) : (tensor<64xf16>) -> tensor<64xf16> {
// CHECK:           arith.constant dense<1.000000e+00> : tensor<64xf32>
// CHECK:           math.exp
// CHECK:           tts.spyreop_yield %{{.*}} : tensor<64xf16>
tt.func @inlined(%x: tensor<64xf16>) -> tensor<64xf16> {
  %r = tts.spyre_op "silu" (%x) : (tensor<64xf16>) -> tensor<64xf16> {
  ^bb0(%a: tensor<64xf16>):
    %one = arith.constant dense<1.0> : tensor<64xf32>
    %xf = arith.extf %a : tensor<64xf16> to tensor<64xf32>
    %n = arith.negf %xf : tensor<64xf32>
    %e = math.exp %n : tensor<64xf32>
    %d = arith.addf %one, %e : tensor<64xf32>
    %q = arith.divf %xf, %d : tensor<64xf32>
    %y = arith.truncf %q : tensor<64xf32> to tensor<64xf16>
    tts.spyreop_yield %y : tensor<64xf16>
  }
  tt.return %r : tensor<64xf16>
}
