// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for the tts.pin OP. No pass runs here.
//
// Three things about the printed form are worth pinning rather than assuming:
//
//   * the address is an OPTIONAL operand printed inside a keyword group, so an
//     addressed pin prints `, address %v` and an unaddressed one prints nothing
//     at all -- not a sentinel, which is what keeps "stated no address" and
//     "stated 0" distinguishable;
//   * `memory_space` is a plain string attribute and prints through `attr-dict`
//     as one. The dialect defines no attribute type on purpose (see the note in
//     TTSDialect.td), so the ktdp vocabulary is checked by the verifier rather
//     than carried in the spelling;
//   * the trailing type is the pinned VALUE's, and prints in full rather than
//     elided -- there is no dialect prefix to imply, unlike tts.tensor_layout's
//     `<512x1024xf32>`.
//
// Each case pins a value with a defining op, because the verifier admits nothing
// else: a block argument is an entry input, which lives where its base says.

// ---------------------------------------------------------------------------
// (a) A constant address -- one offset, the same on every core.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @constant_address(
// CHECK: %[[E:.*]] = math.exp
// CHECK: %[[A:.*]] = arith.constant 4096 : i32
// CHECK: tts.pin %[[E]], address %[[A]] {memory_space = "ct_local"} : tensor<4x64xf16>
tt.func @constant_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %a = arith.constant 4096 : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (b) An address affine in the program id. The expression is ordinary arith,
//     so the round trip has nothing special to say about it -- which is the
//     point: what makes it admissible is its SHAPE, and that is the verifier's
//     business, not the assembly format's.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @affine_address(
// CHECK: %[[PID:.*]] = tt.get_program_id x : i32
// CHECK: %[[M:.*]] = arith.muli %[[PID]]
// CHECK: %[[A:.*]] = arith.addi
// CHECK: tts.pin %{{.*}}, address %[[A]] {memory_space = "ct_local"} : tensor<4x64xf16>
tt.func @affine_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  %pid = tt.get_program_id x : i32
  %stride = arith.constant 256 : i32
  %base = arith.constant 4096 : i32
  %m = arith.muli %pid, %stride : i32
  %a = arith.addi %base, %m : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (c) No address. Well formed, and the form in which the design's baseline --
//     the compiler places every intermediate -- would be written. Nothing
//     lowers it today; see PlacePinnedValues.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @no_address(
// CHECK: tts.pin %{{.*}} {memory_space = "ct_local"} : tensor<4x64xf16>
// CHECK-NOT: address
tt.func @no_address(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (d) A block argument. Pinnable like any other value: a `tensor` is a value
//     whichever way it was defined, and receiving one is the same state a
//     ktdp.load result is in. The "only a value the author named" rule belongs
//     to the surface, where an anonymous subexpression has no name to pass.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @block_argument(
// CHECK: tts.pin %arg0, address %{{.*}} {memory_space = "ct_local"} : tensor<4x64xf16>
tt.func @block_argument(%x: tensor<4x64xf16>) {
  %a = arith.constant 4096 : i32
  tts.pin %x, address %a {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (e) Rank 0. A reduction can produce one, and a rank-0 buffer is a legal
//     memory view (buildRangeSetND has a rank-0 case), so the op must not
//     exclude it by accident.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @rank0(
// CHECK: tts.pin %{{.*}}, address %{{.*}} {memory_space = "ct_local"} : tensor<f16>
tt.func @rank0(%x: tensor<f16>) {
  %e = math.exp %x : tensor<f16>
  %a = arith.constant 0 : i32
  tts.pin %e, address %a {memory_space = "ct_local"} : tensor<f16>
  tt.return
}
