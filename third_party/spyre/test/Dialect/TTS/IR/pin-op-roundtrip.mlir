// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for the tts.pin OP.
//
// Deliberately SMALL, and what it leaves out is the point. `tts.pin` is consumed
// in the `ktir` stage and never reaches an artifact, so its printed form is no
// contract with anything outside this tree; and its assembly format is
// `$value attr-dict : type($value)`, which hand-specifies nothing -- every field
// rides in `attr-dict`. Asserting that attr-dict sorts its entries, or that a
// dense array prints as one, would be asserting MLIR's behaviour rather than
// ours. The address spellings are covered where they mean something: by the
// verifier in pin-op-verifier.mlir, and by the lowering in
// Transforms/pin-attribute.mlir, both of which parse this op to do it.
//
// What is left is the one property the OP itself decides, and a check that it
// prints at all -- which nothing else covers, since pin-op-verifier.mlir only
// parses and Transforms/pin-attribute.mlir checks the ATTRIBUTE's printed form
// after the op is gone.

// Rank 0. A reduction can produce one and a rank-0 buffer is a legal memory view
// (buildRangeSetND has a rank-0 case), so `AnyStaticShapeTensor` must admit it --
// which is a property of the constraint this op chose, not of the printer.
// CHECK-LABEL: tt.func @rank0(
// CHECK: tts.pin %{{.*}} {address = 0 : i32, memory_space = #ktdp.memory_space<ct_local>} : tensor<f16>
tt.func @rank0(%x: tensor<f16>) {
  %e = math.exp %x : tensor<f16>
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = 0 : i32} : tensor<f16>
  tt.return
}
