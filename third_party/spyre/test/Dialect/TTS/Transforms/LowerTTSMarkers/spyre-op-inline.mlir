// RUN: spyre-triton-opt %s -split-input-file --lower-tts-markers | FileCheck %s

// tts.spyre_op dissolving into its body, and the body's ops taking the tag.
//
// Four steps per request, and each case below pins one of them:
//
//   1. the body's ops move in front of the request, block arguments replaced
//      by the operands;
//   2. every moved op but a constant carries
//      `tts.spyreop_hint = {name = <name>, id = <id>}`, ops nested in a moved
//      op's region included;
//   3. the request's results are replaced by the yielded values;
//   4. the request and its terminator are gone.

// One request, the f16 sigmoid fallback as LowerComputeOps leaves it: every op
// hinted but the constant, and the return reading the last of them.
// CHECK-LABEL: func.func @one_request(
// CHECK-SAME:    %[[X:.*]]: tensor<64xf16>
// CHECK:         %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<64xf32>
// CHECK:         %[[XF:.*]] = arith.extf %[[X]] {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// CHECK:         %[[N:.*]] = arith.negf %[[XF]] {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// CHECK:         %[[E:.*]] = math.exp %[[N]] {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// CHECK:         %[[D:.*]] = arith.addf %[[ONE]], %[[E]] {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// CHECK:         %[[Q:.*]] = arith.divf %[[ONE]], %[[D]] {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// CHECK:         %[[Y:.*]] = arith.truncf %[[Q]] {tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
// CHECK:         return %[[Y]] : tensor<64xf16>
// CHECK-NOT:     tts.
func.func @one_request(%x: tensor<64xf16>) -> tensor<64xf16> {
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<64xf16>) -> tensor<64xf16> {
  ^bb0(%a: tensor<64xf16>):
    %one = arith.constant dense<1.0> : tensor<64xf32>
    %xf = arith.extf %a : tensor<64xf16> to tensor<64xf32>
    %n = arith.negf %xf : tensor<64xf32>
    %e = math.exp %n : tensor<64xf32>
    %d = arith.addf %one, %e : tensor<64xf32>
    %q = arith.divf %one, %d : tensor<64xf32>
    %y = arith.truncf %q : tensor<64xf32> to tensor<64xf16>
    tts.spyreop_yield %y : tensor<64xf16>
  }
  return %r : tensor<64xf16>
}

// -----

// Two requests for the same intrinsic -- one helper inlined twice -- keep two
// ids, and an untagged op between them stays untagged. The second id
// is the next one, not a hash of anything: unique per request is the contract.
// CHECK-LABEL: func.func @two_requests(
// CHECK:         %[[A:.*]] = math.exp %{{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "silu"}}
// CHECK:         %[[M:.*]] = arith.mulf %[[A]], %[[A]] : tensor<64xf32>
// CHECK:         math.exp %[[M]] {tts.spyreop_hint = {id = 1 : i64, name = "silu"}}
func.func @two_requests(%x: tensor<64xf32>) -> tensor<64xf32> {
  %r0 = tts.spyre_op "silu" (%x) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyreop_yield %e : tensor<64xf32>
  }
  %m = arith.mulf %r0, %r0 : tensor<64xf32>
  %r1 = tts.spyre_op "silu" (%m) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyreop_yield %e : tensor<64xf32>
  }
  return %r1 : tensor<64xf32>
}

// -----

// Ops nested in a moved op's region are tagged too: a reduction's combiner is
// the body a later pass reads once the reduction is a generic. Terminators are
// not, since every rewrite keeps them and the tag would outlive its request.
// CHECK-LABEL: func.func @nested(
// CHECK:         linalg.reduce {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}}
// CHECK:           arith.addf {{.*}} {tts.spyreop_hint = {id = 0 : i64, name = "gelu"}}
// CHECK:           linalg.yield %{{.*}} : f16
func.func @nested(%x: tensor<4x64xf16>, %init: tensor<4xf16>) -> tensor<4xf16> {
  %r = tts.spyre_op "gelu" (%x, %init) : (tensor<4x64xf16>, tensor<4xf16>) -> tensor<4xf16> {
  ^bb0(%a: tensor<4x64xf16>, %i: tensor<4xf16>):
    %s = linalg.reduce ins(%a : tensor<4x64xf16>) outs(%i : tensor<4xf16>) dimensions = [1]
      (%in: f16, %acc: f16) {
        %t = arith.addf %in, %acc : f16
        linalg.yield %t : f16
      }
    tts.spyreop_yield %s : tensor<4xf16>
  }
  return %r : tensor<4xf16>
}

// -----

// Group ids start above any already in the module, so a tag written earlier --
// by hand here -- is not reused for a different request.
// CHECK-LABEL: func.func @existing_tags(
// CHECK:         math.exp %{{.*}} {tts.spyreop_hint = {id = 7 : i64, name = "sigmoid"}}
// CHECK:         math.exp %{{.*}} {tts.spyreop_hint = {id = 8 : i64, name = "sigmoid"}}
func.func @existing_tags(%x: tensor<64xf32>) -> tensor<64xf32> {
  %p = math.exp %x {tts.spyreop_hint = {id = 7 : i64, name = "sigmoid"}} : tensor<64xf32>
  %r = tts.spyre_op "sigmoid" (%p) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyreop_yield %e : tensor<64xf32>
  }
  return %r : tensor<64xf32>
}

// -----

// A pin naming a request's result lands on the inlined op producing it. The
// requests are inlined before the markers are lowered; the other order would
// write the pin onto the request and erase it with the request.
// CHECK-LABEL: func.func @pinned_result(
// CHECK:         math.exp %{{.*}} {tts.pin = {memory_space = "ct_local", offset = 0 : i32}, tts.spyreop_hint = {id = 0 : i64, name = "sigmoid"}}
func.func @pinned_result(%x: tensor<64xf32>) -> tensor<64xf32> {
  %r = tts.spyre_op "sigmoid" (%x) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyreop_yield %e : tensor<64xf32>
  }
  tts.pin %r {memory_space = "ct_local", offset = 0 : i32} : tensor<64xf32>
  return %r : tensor<64xf32>
}
