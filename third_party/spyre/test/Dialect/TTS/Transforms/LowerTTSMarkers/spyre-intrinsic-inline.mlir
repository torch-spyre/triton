// RUN: spyre-triton-opt %s -split-input-file --lower-tts-markers | FileCheck %s

// tts.spyre_intrinsic dissolving into its body, and the body's ops taking the tag.
//
// Four steps per request, and each case below pins one of them:
//
//   1. the body's ops move in front of the request, block arguments replaced
//      by the operands;
//   2. every moved op but a constant carries
//      `tts.spyre_intrinsic = {arity = <n>, id = <id>, target = <target>}`, and
//      one reading a block argument `operands` too: per operand of it, the
//      call operand it is, or -1;
//   3. the request's results are replaced by the yielded values;
//   4. the request and its terminator are gone.

// One request, the f16 sigmoid fallback as LowerComputeOps leaves it: every op
// hinted but the constant, and the return reading the last of them.
// CHECK-LABEL: func.func @one_request(
// CHECK-SAME:    %[[X:.*]]: tensor<64xf16>
// CHECK:         %[[ONE:.*]] = arith.constant dense<1.000000e+00> : tensor<64xf32>
// CHECK:         %[[XF:.*]] = arith.extf %[[X]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}}
// CHECK:         %[[N:.*]] = arith.negf %[[XF]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:         %[[E:.*]] = math.exp %[[N]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:         %[[D:.*]] = arith.addf %[[ONE]], %[[E]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:         %[[Q:.*]] = arith.divf %[[ONE]], %[[D]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:         %[[Y:.*]] = arith.truncf %[[Q]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:         return %[[Y]] : tensor<64xf16>
// CHECK-NOT:     tts.
func.func @one_request(%x: tensor<64xf16>) -> tensor<64xf16> {
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<64xf16>) -> tensor<64xf16> {
  ^bb0(%a: tensor<64xf16>):
    %one = arith.constant dense<1.0> : tensor<64xf32>
    %xf = arith.extf %a : tensor<64xf16> to tensor<64xf32>
    %n = arith.negf %xf : tensor<64xf32>
    %e = math.exp %n : tensor<64xf32>
    %d = arith.addf %one, %e : tensor<64xf32>
    %q = arith.divf %one, %d : tensor<64xf32>
    %y = arith.truncf %q : tensor<64xf32> to tensor<64xf16>
    tts.spyre_intrinsic_yield %y : tensor<64xf16>
  }
  return %r : tensor<64xf16>
}

// -----

// Two requests for the same intrinsic -- one helper inlined twice -- keep two
// ids, and an untagged op between them stays untagged. The second id
// is the next one, not a hash of anything: unique per request is the contract.
// CHECK-LABEL: func.func @two_requests(
// CHECK:         %[[A:.*]] = math.exp %{{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.silu"}}
// CHECK:         %[[M:.*]] = arith.mulf %[[A]], %[[A]] : tensor<64xf32>
// CHECK:         math.exp %[[M]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 1 : i64, operands = array<i64: 0>, target = "spyreop.silu"}}
func.func @two_requests(%x: tensor<64xf32>) -> tensor<64xf32> {
  %r0 = tts.spyre_intrinsic "spyreop.silu" (%x) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyre_intrinsic_yield %e : tensor<64xf32>
  }
  %m = arith.mulf %r0, %r0 : tensor<64xf32>
  %r1 = tts.spyre_intrinsic "spyreop.silu" (%m) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyre_intrinsic_yield %e : tensor<64xf32>
  }
  return %r1 : tensor<64xf32>
}

// -----

// Group ids start above any already in the module, so a tag written earlier --
// by hand here -- is not reused for a different request.
// CHECK-LABEL: func.func @existing_tags(
// CHECK:         math.exp %{{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 7 : i64, target = "spyreop.sigmoid"}}
// CHECK:         math.exp %{{.*}} {tts.spyre_intrinsic = {arity = 1 : i64, id = 8 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}}
func.func @existing_tags(%x: tensor<64xf32>) -> tensor<64xf32> {
  %p = math.exp %x {tts.spyre_intrinsic = {arity = 1 : i64, id = 7 : i64, target = "spyreop.sigmoid"}} : tensor<64xf32>
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%p) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyre_intrinsic_yield %e : tensor<64xf32>
  }
  return %r : tensor<64xf32>
}

// -----

// A pin naming a request's result lands on the inlined op producing it. The
// requests are inlined before the markers are lowered; the other order would
// write the pin onto the request and erase it with the request.
// CHECK-LABEL: func.func @pinned_result(
// CHECK:         math.exp %{{.*}} {tts.pin = {memory_space = "ct_local", offset = 0 : i32}, tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0>, target = "spyreop.sigmoid"}}
func.func @pinned_result(%x: tensor<64xf32>) -> tensor<64xf32> {
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %e = math.exp %a : tensor<64xf32>
    tts.spyre_intrinsic_yield %e : tensor<64xf32>
  }
  tts.pin %r {memory_space = "ct_local", offset = 0 : i32} : tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

// Two operands, read by different ops and in the opposite order to the call's
// -- `realdiv(a, b)` falling back to `exp(b) / a` -- so the binding is what
// says which is which: the exp reads call operand 1, the divf reads the exp's
// result and call operand 0. Written now, because after
// ConvertElementwiseToLinalg the divf's generic would have `ins(%e, %a)` and
// nothing would say which of its block arguments is the call's first.
// CHECK-LABEL: func.func @two_operands(
// CHECK-SAME:    %[[A:[^:]*]]: tensor<64xf32>, %[[B:[^:]*]]: tensor<64xf32>
// CHECK:         %[[E:.*]] = math.exp %[[B]] {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: 1>, target = "spyreop.realdiv"}}
// CHECK:         %[[Q:.*]] = arith.divf %[[E]], %[[A]] {tts.spyre_intrinsic = {arity = 2 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.realdiv"}}
// CHECK:         return %[[Q]]
func.func @two_operands(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %r = tts.spyre_intrinsic "spyreop.realdiv" (%a, %b) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%x: tensor<64xf32>, %y: tensor<64xf32>):
    %e = math.exp %y : tensor<64xf32>
    %q = arith.divf %e, %x : tensor<64xf32>
    tts.spyre_intrinsic_yield %q : tensor<64xf32>
  }
  return %r : tensor<64xf32>
}

// -----

// One operand read twice by one op, and once more by another: an entry per
// operand of the op, so `x * x` binds both of its operands to call operand 0,
// and an op reading no block argument has no `operands`.
// CHECK-LABEL: func.func @one_operand_read_twice(
// CHECK-SAME:    %[[X:[^:]*]]: tensor<64xf32>
// CHECK:         %[[S:.*]] = arith.mulf %[[X]], %[[X]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: 0, 0>, target = "spyreop.sigmoid"}}
// CHECK:         %[[E:.*]] = math.exp %[[S]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, target = "spyreop.sigmoid"}}
// CHECK:         arith.addf %[[E]], %[[X]] {tts.spyre_intrinsic = {arity = 1 : i64, id = 0 : i64, operands = array<i64: -1, 0>, target = "spyreop.sigmoid"}}
func.func @one_operand_read_twice(%x: tensor<64xf32>) -> tensor<64xf32> {
  %r = tts.spyre_intrinsic "spyreop.sigmoid" (%x) : (tensor<64xf32>) -> tensor<64xf32> {
  ^bb0(%a: tensor<64xf32>):
    %s = arith.mulf %a, %a : tensor<64xf32>
    %e = math.exp %s : tensor<64xf32>
    %t = arith.addf %e, %a : tensor<64xf32>
    tts.spyre_intrinsic_yield %t : tensor<64xf32>
  }
  return %r : tensor<64xf32>
}
