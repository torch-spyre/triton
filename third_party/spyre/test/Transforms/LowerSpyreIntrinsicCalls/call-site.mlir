// RUN: spyre-triton-opt %s --lower-spyre-intrinsic-calls -split-input-file | FileCheck %s
// RUN: spyre-triton-opt %s --lower-spyre-intrinsic-calls -split-input-file | spyre-triton-opt -split-input-file | FileCheck %s --check-prefix=NOHINT

// NOHINT-NOT: tts.spyre_intrinsic

// THE REQUEST RULE: a compute body whose ops, constants aside, all carry
// `tts.spyre_intrinsic`s of one `id` is one `tl.spyre_intrinsic` call site's whole
// fallback, and becomes the intrinsic the hint names applied to the body's
// inputs, in the call's order: the `operands` of the members reading them say
// which block argument is which call operand.
//
// Each body below is the fallback the backend registers for that intrinsic, as
// FuseComputeAndDataMovement leaves it: one generic, its scalar constants
// captured from above and unhinted. The fallback is replaced WHOLE -- the casts
// around an f16 request included -- so what survives is the intrinsic at the
// type the body reads and yields.
//
// This pass runs no implicit rule: each body holds a `math.exp` or an
// `arith.divf`, which LowerSpyreOps would select as spyreop.exp or
// spyreop.realdiv, and none is left here, so none reaches it. The inputs a
// selected body stops reading are removed by this pass's own cleanup.

#map = affine_map<(d0) -> (d0)>

// gelu, f16: 0.5 * x * (1 + erf(x / sqrt(2))), computed in f32.
// CHECK-LABEL: func.func @gelu_f16(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:      %[[R:.*]] = spyreop.gelu %[[IN]] : f16
// CHECK-NEXT:      linalg.yield %[[R]] : f16
// CHECK-NOT:     tts.spyre_intrinsic
func.func @gelu_f16(%x: tensor<64xf16>) -> tensor<64xf16> {
  %half = arith.constant 5.000000e-01 : f32
  %one = arith.constant 1.000000e+00 : f32
  %rsqrt2 = arith.constant 0.707106769 : f32
  %e = tensor.empty() : tensor<64xf16>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf16>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %xf = arith.extf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.gelu"}} : f16 to f32
    %h = arith.mulf %xf, %half {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.gelu"}} : f32
    %s = arith.mulf %xf, %rsqrt2 {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.gelu"}} : f32
    %er = math.erf %s {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.gelu"}} : f32
    %p = arith.addf %er, %one {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.gelu"}} : f32
    %m = arith.mulf %h, %p {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.gelu"}} : f32
    %t = arith.truncf %m {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.gelu"}} : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

#map = affine_map<(d0) -> (d0)>

// silu, f32: x / (1 + exp(-x)). No casts at f32; the input is read twice.
// CHECK-LABEL: func.func @silu_f32(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.silu %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @silu_f32(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 4 : i64, operands = array<i64: 0>, target = "spyreop.silu"}} : f32
    %ex = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 4 : i64, target = "spyreop.silu"}} : f32
    %d = arith.addf %ex, %one {tts.spyre_intrinsic = {arity = 1 : i64, id = 4 : i64, target = "spyreop.silu"}} : f32
    %q = arith.divf %in, %d {tts.spyre_intrinsic = {arity = 1 : i64, id = 4 : i64, operands = array<i64: 0, -1>, target = "spyreop.silu"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// silu, f16: the same fallback widened, with the widening duplicated by fusion.
// CHECK-LABEL: func.func @silu_f16(
// CHECK:         spyreop.silu %{{.*}} : f16
// CHECK-NOT:     arith.extf
func.func @silu_f16(%x: tensor<64xf16>) -> tensor<64xf16> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf16>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf16>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %xf = arith.extf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.silu"}} : f16 to f32
    %n = arith.negf %xf {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.silu"}} : f32
    %ex = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.silu"}} : f32
    %d = arith.addf %ex, %one {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.silu"}} : f32
    %xf2 = arith.extf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.silu"}} : f16 to f32
    %q = arith.divf %xf2, %d {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.silu"}} : f32
    %t = arith.truncf %q {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.silu"}} : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

#map = affine_map<(d0) -> (d0)>

// sigmoid, f32: 1 / (1 + exp(-x)). The divide's numerator is a one, which the
// reciprocal rule would otherwise take.
// CHECK-LABEL: func.func @sigmoid_f32(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.sigmoid %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @sigmoid_f32(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}} : f32
    %ex = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %d = arith.addf %ex, %one {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %q = arith.divf %one, %d {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// sigmoid, f16, with its constant still a splat `ins` operand -- the form
// before the fusion folds it. The block argument reading the constant is bound
// to no call operand, so the target does not read it, and the unused operand is
// then erased.
// CHECK-LABEL: func.func @sigmoid_f16_splat_operand(
// CHECK:         arith.constant dense<1.000000e+00> : tensor<64xf32>
// CHECK:         linalg.generic {{.*}} ins(%{{.*}} : tensor<64xf16>)
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:      %[[R:.*]] = spyreop.sigmoid %[[IN]] : f16
// CHECK-NEXT:      linalg.yield %[[R]] : f16
// CHECK-NOT:     tts.spyre_intrinsic
func.func @sigmoid_f16_splat_operand(%x: tensor<64xf16>) -> (tensor<64xf16>, tensor<64xf32>) {
  %one = arith.constant dense<1.0> : tensor<64xf32>
  %e = tensor.empty() : tensor<64xf16>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%x, %one : tensor<64xf16>, tensor<64xf32>) outs(%e : tensor<64xf16>) {
  ^bb0(%in: f16, %c: f32, %out: f16):
    %xf = arith.extf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}} : f16 to f32
    %n = arith.negf %xf {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %ex = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %d = arith.addf %ex, %c {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %q = arith.divf %c, %d {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %t = arith.truncf %q {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r, %one : tensor<64xf16>, tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A pin on the call site's result is on the op that produced it, which the
// selection erases; the pin moves to the target op, the hint does not.
// CHECK-LABEL: func.func @pinned_result(
// CHECK:         %[[R:.*]] = spyreop.sigmoid %{{.*}} {tts.pin = {memory_space = "ct_local", offset = 0 : i32}} : f32
// CHECK-NEXT:    linalg.yield %[[R]] : f32
func.func @pinned_result(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}} : f32
    %x1 = math.exp %n {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %d = arith.addf %x1, %one {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    %q = arith.divf %one, %d {tts.pin = {memory_space = "ct_local", offset = 0 : i32}, tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// TWO OPERANDS, `ins` in the opposite order to the call. A fallback for
// `realdiv(a, b)` reading `b` first -- `exp(b) / a`, say -- converts to a
// generic with `ins(%b, %a)`, so block argument 0 is call operand 1. The
// binding, not the order, gives the target its operands: realdiv takes the
// block argument holding `a` first. And call operand 0 read from two block
// arguments holding one value through one map -- two `ins` the fusion did not
// merge -- is one input.
// CHECK-LABEL: func.func @two_operands_reversed(
// CHECK-SAME:    %[[A:[^:]*]]: tensor<64xf32>, %[[B:[^:]*]]: tensor<64xf32>
// CHECK:         linalg.generic {{.*}} ins(%[[B]], %[[A]] : tensor<64xf32>, tensor<64xf32>)
// CHECK-NEXT:    ^bb0(%[[INB:[^:]*]]: f32, %[[INA:[^:]*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.realdiv %[[INA]], %[[INB]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
// CHECK-NOT:     tts.spyre_intrinsic
func.func @two_operands_reversed(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]}
      ins(%b, %a, %a : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%inb: f32, %ina: f32, %ina2: f32, %out: f32):
    %x = math.exp %inb {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1>, target = "spyreop.realdiv"}} : f32
    %q = arith.divf %x, %ina {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.realdiv"}} : f32
    %s = arith.addf %q, %ina2 {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.realdiv"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A COMMUTATIVE MEMBER WHOSE OPERANDS MOVED. The addf was tagged
// `operands = array<i64: -1, 0>` with the call operand second; the commutative
// fold has since put the constant last, so position 1 holds no input. A
// commutative member's binding is a set, so its one input is call operand 0.
// CHECK-LABEL: func.func @commutative_operands_moved(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.sigmoid %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @commutative_operands_moved(%x: tensor<64xf32>) -> tensor<64xf32> {
  %one = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %d = arith.addf %in, %one {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.sigmoid"}} : f32
    %q = arith.divf %one, %d {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// A COMMUTATIVE MEMBER READING TWO CALL OPERANDS, `b + a`. Every position it
// lists holds an input, so it binds by position -- call operand 1 to the first,
// 0 to the second -- and that agrees with the divf, which binds call operand 0
// to the same block argument. The target takes `a` first.
// CHECK-LABEL: func.func @commutative_two_call_operands(
// CHECK-SAME:    %[[A:[^:]*]]: tensor<64xf32>, %[[B:[^:]*]]: tensor<64xf32>
// CHECK:         linalg.generic {{.*}} ins(%[[B]], %[[A]] : tensor<64xf32>, tensor<64xf32>)
// CHECK-NEXT:    ^bb0(%[[INB:[^:]*]]: f32, %[[INA:[^:]*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.realdiv %[[INA]], %[[INB]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @commutative_two_call_operands(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
      ins(%b, %a : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%inb: f32, %ina: f32, %out: f32):
    %s = arith.addf %inb, %ina {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1, 0>, target = "spyreop.realdiv"}} : f32
    %q = arith.divf %s, %ina {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.realdiv"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// ONE VALUE PASSED AS TWO CALL OPERANDS, `realdiv(x, x)`: LowerTTSMarkers bound
// two block arguments as call operands 0 and 1, and fusion merged their equal
// `ins` into one, so both call operands are bound to it. Not a conflict: a
// conflict is ONE call operand bound to two different inputs. The target reads
// that argument twice.
// CHECK-LABEL: func.func @one_value_two_call_operands(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:[^:]*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.realdiv %[[IN]], %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @one_value_two_call_operands(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %q = arith.divf %in, %in {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 0, 1>, target = "spyreop.realdiv"}} : f32
    linalg.yield %q : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

// The commutative variant, `addf %x, %x` binding call operands 0 and 1. Both
// positions it lists hold an input, so it binds by position, both to the one
// block argument.
// CHECK-LABEL: func.func @one_value_two_call_operands_commutative(
// CHECK:         linalg.generic
// CHECK-NEXT:    ^bb0(%[[IN:[^:]*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[R:.*]] = spyreop.realdiv %[[IN]], %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[R]] : f32
func.func @one_value_two_call_operands_commutative(%x: tensor<64xf32>) -> tensor<64xf32> {
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%x : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %in {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 0, 1>, target = "spyreop.realdiv"}} : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}
