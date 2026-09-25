// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for the tts.pin OP. No pass runs here.
//
// Three things about the printed form are worth pinning rather than assuming:
//
//   * the address is an OPTIONAL ATTRIBUTE, so an unaddressed pin prints no
//     `address` entry at all rather than a sentinel -- which is what keeps
//     "stated no address" and "stated 0" distinguishable, 0 being a legitimate
//     element index;
//   * both address spellings survive the round trip as themselves. A single
//     `i32` does not become a one-element array and an array does not collapse
//     when its entries are equal, because which one the author wrote is what
//     says whether the address varies per core;
//   * everything prints through `attr-dict`, so the entries come out in
//     ALPHABETICAL order -- `address` before `memory_space` -- and not in the
//     order the ODS argument list declares them.
//
// The trailing type is the pinned VALUE's, and prints in full rather than elided:
// there is no dialect prefix to imply, unlike tts.tensor_layout's
// `<512x1024xf32>`.

// ---------------------------------------------------------------------------
// (a) One i32 -- one offset, the same on every core.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @uniform_address(
// CHECK: %[[E:.*]] = math.exp
// CHECK: tts.pin %[[E]] {address = 4096 : i32, memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
tt.func @uniform_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (b) An array -- one offset per program id, positionally. This is the form an
//     address varying with the core takes: the arms are stated, not the
//     coefficients of an expression that computes them.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @per_core_address(
// CHECK: tts.pin %{{.*}} {address = array<i32: 4096, 4352, 4608, 4864>, memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
tt.func @per_core_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = array<i32: 4096, 4352, 4608, 4864>} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (c) An array whose entries are all equal. Prints as an array, because
//     collapsing it to an i32 would erase the author's statement that this
//     address is indexed per core -- it happens to be uniform, which is not the
//     same claim.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @uniform_array_stays_an_array(
// CHECK: tts.pin %{{.*}} {address = array<i32: 4096, 4096>, memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
tt.func @uniform_array_stays_an_array(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = array<i32: 4096, 4096>} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (d) No address. Well formed, and the form in which the design's baseline --
//     the compiler places every intermediate -- would be written.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @no_address(
// CHECK: tts.pin %{{.*}} {memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
// CHECK-NOT: address
tt.func @no_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (e) A block argument. The OP admits one -- a `tensor` is a value whichever way
//     it was defined -- so it round trips like any other pin. What refuses it is
//     LowerTTSMarkers, because the annotation's carrier is the op DEFINING the
//     value and a block argument has none; see invalid.mlir beside that pass.
//     The two halves are deliberately split: well-formedness is the op's
//     question, and whether anything can carry the annotation is the lowering's.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @block_argument(
// CHECK: tts.pin %arg0 {address = 4096 : i32, memory_space = #ktdp.memory_space<ct_local>} : tensor<4x64xf16>
tt.func @block_argument(%x: tensor<4x64xf16>) {
  tts.pin %x {memory_space = #ktdp.memory_space<ct_local>, address = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (f) Rank 0. A reduction can produce one, and a rank-0 buffer is a legal
//     memory view (buildRangeSetND has a rank-0 case), so the op must not
//     exclude it by accident.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @rank0(
// CHECK: tts.pin %{{.*}} {address = 0 : i32, memory_space = #ktdp.memory_space<ct_local>} : tensor<f16>
tt.func @rank0(%x: tensor<f16>) {
  %e = math.exp %x : tensor<f16>
  tts.pin %e {memory_space = #ktdp.memory_space<ct_local>, address = 0 : i32} : tensor<f16>
  tt.return
}
