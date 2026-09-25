// RUN: spyre-triton-opt %s -split-input-file --lower-tts-markers | FileCheck %s

// The `tts.pin` marker becoming the `tts.pin` attribute.
//
// Two things happen, and the first is the one with a choice in it:
//
//   1. the marker's memory_space and address land on the op DEFINING the pinned
//      value, as a `tts.pin` dictionary attribute;
//   2. the marker op is erased.
//
// (1) is where this marker differs from `tts.tensor_layout`. A layout names a
// descriptor, which has resolved to one particular op -- a memory view -- and the
// pass checks that it did. A pin names a VALUE, and a value's only op is the one
// defining it, whatever that op happens to be. So there is no admissibility test
// on the carrier: the cases below pin a `math` result and a named `linalg` one --
// the two forms a pinned value actually takes at this point in the pipeline -- and
// the pass treats them the same way, because a consumer reads the attribute and
// never the op's identity.
//
// No bridge-cast assertions here, unlike tensor-layout-attribute.mlir. A pin's
// operand is a `tensor` throughout -- no pass retypes it -- so there is never a
// cast standing between the marker and the value, and nothing for the marker's
// erasure to leave dead.
//
// The address is carried THROUGH rather than interpreted. Whether it is a single
// i32 or an array, and whatever numbers are in it, the attribute holds what the
// op held: capacity, alignment and disjointness are arithmetic for a consumer,
// and no consumer exists yet.

// An elementwise producer, the common case. `math.exp` survives this stage as
// itself -- ConvertElementwiseToLinalg runs in `spyrecode` -- so the attribute
// lands on the math op.
// CHECK-LABEL: tt.func @elementwise_producer
// CHECK: math.exp {{.*}} {tts.pin = {address = 4096 : i32, memory_space = "ct_local"}}
// CHECK-NOT: tts.pin %
tt.func @elementwise_producer(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", address = 4096 : i32} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// A named linalg producer. This is what a pinned `tl.sum` is by the time the pass
// runs, and the reason the pass sits after LowerComputeOps: before it the value
// is produced by a tt.reduce, which that pass REPLACES, and an attribute written
// on the tt.reduce would be dropped with it.
// CHECK-LABEL: tt.func @linalg_producer
// CHECK: linalg.reduce
// CHECK-SAME: {tts.pin = {address = 8192 : i32, memory_space = "ct_local"}}
// CHECK-NOT: tts.pin %
tt.func @linalg_producer(%x: tensor<4x64xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.reduce { arith.addf } ins(%x : tensor<4x64xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
  tts.pin %r {memory_space = "ct_local", address = 8192 : i32} : tensor<4xf32>
  tt.return %r : tensor<4xf32>
}

// -----
// A per-core address. The array is reused as the attribute's `address` entry
// unchanged -- not summarized into coefficients, and not collapsed when its
// entries happen to be equal.
// CHECK-LABEL: tt.func @per_core_address
// CHECK: math.exp {{.*}} {tts.pin = {address = array<i32: 4096, 4352, 4608>, memory_space = "ct_local"}}
// CHECK-NOT: tts.pin %
tt.func @per_core_address(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", address = array<i32: 4096, 4352, 4608>} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// No address. The entry is ABSENT from the dictionary rather than zero, which is
// what keeps "stated no address" distinguishable from "stated 0" -- 0 being a
// legitimate element index.
// CHECK-LABEL: tt.func @no_address
// CHECK: math.exp {{.*}} {tts.pin = {memory_space = "ct_local"}}
// CHECK-NOT: address
tt.func @no_address(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// Two pins on two values produced by the same KIND of op, to show the attribute
// is per-op and not per-function: each lands on its own producer with its own
// address.
// CHECK-LABEL: tt.func @two_pins
// CHECK: math.exp {{.*}} {tts.pin = {address = 0 : i32, memory_space = "ct_local"}}
// CHECK: math.sqrt {{.*}} {tts.pin = {address = 512 : i32, memory_space = "ct_local"}}
// CHECK-NOT: tts.pin %
tt.func @two_pins(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %s = math.sqrt %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", address = 0 : i32} : tensor<4x64xf16>
  tts.pin %s {memory_space = "ct_local", address = 512 : i32} : tensor<4x64xf16>
  %y = arith.addf %e, %s : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// A pin whose position in the block is BELOW a use of the value. The marker's
// location does not matter, because the annotation is about the value and the
// carrier is its producer -- so this lowers exactly like the same pin written
// above the use. That is the difference between an annotation on a value and a
// marker with a program point.
// CHECK-LABEL: tt.func @pin_below_a_use
// CHECK: math.exp {{.*}} {tts.pin = {address = 4096 : i32, memory_space = "ct_local"}}
// CHECK-NOT: tts.pin %
tt.func @pin_below_a_use(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", address = 4096 : i32} : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}
