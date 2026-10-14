// RUN: spyre-triton-opt %s -split-input-file --convert-elementwise-to-linalg --fuse-compute-and-data-movement | FileCheck %s

// The THIRD fusion clause: two generics whose bodies carry `tts.spyre_intrinsic`
// tags of one `id` are one intrinsic request, and are fused into one body for
// LowerSpyreIntrinsicCalls to replace whole.
//
// The input is tensor-level, as LowerTTSMarkers leaves a request, and the RUN
// line converts it first. That conversion is what puts each tagged tensor op's
// tag onto the scalar op in its generic's body, which is where this clause
// reads it -- so this file also pins that the tag survives that conversion and
// the fusion itself.

// The f16 sigmoid fallback, seven tensor ops: one generic, every compute op
// hinted, the splat constant folded in as an unhinted scalar.
// CHECK-LABEL: func.func @one_request(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic
// CHECK:           arith.extf {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}}
// CHECK:           arith.negf {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:           math.exp {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:           arith.addf {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:           arith.divf {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:           arith.truncf {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:           linalg.yield
// CHECK-NOT:     linalg.generic
func.func @one_request(%x: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %one = arith.constant dense<1.0> : tensor<64x128xf32>
  %xf = arith.extf %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}} : tensor<64x128xf16> to tensor<64x128xf32>
  %n = arith.negf %xf {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<64x128xf32>
  %e = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<64x128xf32>
  %d = arith.addf %one, %e {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<64x128xf32>
  %q = arith.divf %one, %d {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<64x128xf32>
  %y = arith.truncf %q {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<64x128xf32> to tensor<64x128xf16>
  return %y : tensor<64x128xf16>
}

// -----

// A producer read twice by one request -- silu's `x * sigmoid(x)` reads the
// widened input in two places -- is fused into the one body, and the widening
// may be duplicated there; nothing of the request is left outside.
// CHECK-LABEL: func.func @multi_use_within_request(
// CHECK:         linalg.generic
// CHECK-NOT:     linalg.generic
// CHECK:           arith.divf {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 3 : i64, target = "spyreop.silu"}}
// CHECK:           linalg.yield
// CHECK-NOT:     linalg.generic
func.func @multi_use_within_request(%x: tensor<64xf16>) -> tensor<64xf16> {
  %one = arith.constant dense<1.0> : tensor<64xf32>
  %xf = arith.extf %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 3 : i64, target = "spyreop.silu"}} : tensor<64xf16> to tensor<64xf32>
  %n = arith.negf %xf {tts.spyre_intrinsic = {arity = 1 : i64, id = 3 : i64, target = "spyreop.silu"}} : tensor<64xf32>
  %e = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 3 : i64, target = "spyreop.silu"}} : tensor<64xf32>
  %d = arith.addf %one, %e {tts.spyre_intrinsic = {arity = 1 : i64, id = 3 : i64, target = "spyreop.silu"}} : tensor<64xf32>
  %q = arith.divf %xf, %d {tts.spyre_intrinsic = {arity = 1 : i64, id = 3 : i64, target = "spyreop.silu"}} : tensor<64xf32>
  %y = arith.truncf %q {tts.spyre_intrinsic = {arity = 1 : i64, id = 3 : i64, target = "spyreop.silu"}} : tensor<64xf32> to tensor<64xf16>
  return %y : tensor<64xf16>
}

// -----

// The negatives. Two requests one after the other stay two bodies, even for the
// same intrinsic: the clause compares the WHOLE hint, id included. And an
// untagged compute between or after them fuses with neither -- two computes
// that are not one request never merge.
// CHECK-LABEL: func.func @distinct_requests(
// CHECK:         linalg.generic
// CHECK:           math.exp {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:           arith.negf {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:           linalg.yield
// CHECK:         linalg.generic
// CHECK:           math.exp {{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 1 : i64, target = "spyreop.sigmoid"}}
// CHECK:           linalg.yield
// CHECK:         linalg.generic
// CHECK:           arith.mulf
// CHECK-NOT:       tts.spyre_intrinsic
// CHECK:           linalg.yield
func.func @distinct_requests(%x: tensor<64xf32>) -> tensor<64xf32> {
  %a = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<64xf32>
  %b = arith.negf %a {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : tensor<64xf32>
  %c = math.exp %b {tts.spyre_intrinsic = {arity = 1 : i64, id = 1 : i64, target = "spyreop.sigmoid"}} : tensor<64xf32>
  %d = arith.mulf %c, %c : tensor<64xf32>
  return %d : tensor<64xf32>
}

// -----

// Two operands read in the opposite order to the call's, `exp(b) / a` for
// `realdiv(a, b)`: the fused body's `ins` follow the ops, `b` first, and what
// still says which block argument is which call operand is each op's
// `operands`, kept through the conversion and the fusion on the op and in its
// operand order.
// CHECK-LABEL: func.func @two_operands(
// CHECK-SAME:    %[[A:[^:]*]]: tensor<64xf32>, %[[B:[^:]*]]: tensor<64xf32>
// CHECK:         linalg.generic {{.*}} ins(%[[B]], %[[A]] : tensor<64xf32>, tensor<64xf32>)
// CHECK-NEXT:    ^bb0(%[[INB:[^:]*]]: f32, %[[INA:[^:]*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[E:.*]] = math.exp %[[INB]] {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1>, target = "spyreop.realdiv"}}
// CHECK-NEXT:      arith.divf %[[E]], %[[INA]] {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.realdiv"}}
// CHECK-NOT:     linalg.generic
func.func @two_operands(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %e = math.exp %b {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1>, target = "spyreop.realdiv"}} : tensor<64xf32>
  %q = arith.divf %e, %a {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.realdiv"}} : tensor<64xf32>
  return %q : tensor<64xf32>
}
