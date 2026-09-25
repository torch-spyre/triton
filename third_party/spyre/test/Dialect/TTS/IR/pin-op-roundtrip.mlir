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
// What is left is the two things the OP itself decides:

// A block argument. Admitted -- a `tensor` is a value whichever way it was
// defined -- and this is the positive half of a deliberate split: the op says
// well-formed, and LowerTTSMarkers refuses it, because the annotation's carrier
// is the op DEFINING the value and a block argument has none. See
// @pinned_block_argument in Transforms/invalid.mlir for the other half. Neither
// file states the rule alone.
// CHECK-LABEL: tt.func @block_argument(
// CHECK: tts.pin %arg0 {address = 4096 : i32, memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
tt.func @block_argument(%x: tensor<4x64xf16>) {
  tts.pin %x {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

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
