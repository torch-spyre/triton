// RUN: spyre-triton-opt %s | spyre-triton-opt | FileCheck %s

// Parse-print-parse round trip for the tts.make_distributed_descriptor OP. No
// pass runs here.
//
// Three things about the printed form are worth pinning rather than assuming:
//
//   * this is the only op in the dialect with a RESULT, so the trailing types are
//     a pair -- the share's tensor type and the descriptor it composes to -- and
//     the descriptor prints in the elided form `<64x32xf16>`, since `!tt.` is
//     implied by the operand's type constraint;
//   * the three attributes are inherent and print through `attr-dict`, so they
//     come back in the printer's sorted order (axes, block_shape, work_slices)
//     rather than the order written;
//   * an undivided dimension is `""` in `axes` and survives as one. The Python
//     surface spells it `None`, which reads better in a kernel and which an
//     attribute array cannot hold -- so the two spellings differ on purpose and
//     the IR's is the one checked here.
//
// Every case states its COMPOSED shape, because that shape appears nowhere in the
// IR: the slice count is one more than the largest index in the table, so the
// domain a read's offsets live in is derived rather than written. It is also what
// `block_shape` is bounded by -- not the share's extents -- which is the one thing
// about this op that reads backwards until the three shapes are written down
// together.
//
//   share      what this instance contributes, the operand's type
//   composed   what the partitions compose to, derived from the table
//   block      what one `.load()` takes, anywhere from less than a share to the
//              whole composed extent

// ---------------------------------------------------------------------------
// (a) Two partitions, divided on the last dimension.
//
//     share [64, 32]  composed [64, 64]  block [64, 32]
//
//     The common case: one region per load. The composed extent is not in the
//     type, because a descriptor's type carries its BLOCK shape and nothing else
//     -- which is `tt.make_tensor_descriptor`'s own convention, where the tensor's
//     shape is an operand and the offsets are global coordinates into it.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @two_partitions(
// CHECK: tts.make_distributed_descriptor %arg0 {axes = ["", "n"], block_shape = array<i64: 64, 32>, work_slices = [{n = 0 : i64}, {n = 1 : i64}]} : tensor<64x32xf16> -> <64x32xf16>
tt.func @two_partitions(%share: tensor<64x32xf16>) {
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (b) Two dimensions divided at once, four partitions.
//
//     share [32, 32]  composed [64, 64]  block [32, 32]
//
//     The table's keys are a grid coordinate per partition, so a 2x2 division is
//     four entries and not two lists -- which is the shape that lets ownership be
//     strided rather than a function of any one axis count.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @two_axes(
// CHECK: tts.make_distributed_descriptor %arg0 {axes = ["m", "n"], block_shape = array<i64: 32, 32>, work_slices = [{m = 0 : i64, n = 0 : i64}, {m = 0 : i64, n = 1 : i64}, {m = 1 : i64, n = 0 : i64}, {m = 1 : i64, n = 1 : i64}]} : tensor<32x32xf16> -> <32x32xf16>
tt.func @two_axes(%share: tensor<32x32xf16>) {
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{m = 0 : i64, n = 0 : i64}, {m = 0 : i64, n = 1 : i64},
                      {m = 1 : i64, n = 0 : i64}, {m = 1 : i64, n = 1 : i64}],
       axes = ["m", "n"], block_shape = array<i64: 32, 32>}
      : tensor<32x32xf16> -> !tt.tensordesc<32x32xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (c) A block SMALLER than the share.
//
//     share [64, 32]  composed [64, 64]  block [64, 16]
//
//     Legal, and what a relayout whose destination division is finer than the
//     source's looks like: the instance takes less than its own share per read.
//     The descriptor's block type follows block_shape rather than the share's
//     extents, and the two being different is what makes that visible.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @partial_block(
// CHECK: tts.make_distributed_descriptor %arg0 {axes = ["", "n"], block_shape = array<i64: 64, 16>, work_slices = [{n = 0 : i64}, {n = 1 : i64}]} : tensor<64x32xf16> -> <64x16xf16>
tt.func @partial_block(%share: tensor<64x32xf16>) {
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 16>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x16xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (d) A block LARGER than the share, at the limit: the whole composed extent.
//
//     share [64, 32]  composed [64, 64]  block [64, 64]
//
//     This is the all-reduce form -- every instance takes the entire composed
//     axis, so a fold over it reduces across every core -- and the gather family's
//     limit, where one access spans several partitions. Bounded by the COMPOSED
//     extent and not by the share's, which is the rule this case exists to pin:
//     `block_shape` has no relation to the share's type at all, and Triton relates
//     it only to the descriptor's own block type.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @block_is_the_whole(
// CHECK: tts.make_distributed_descriptor %arg0 {axes = ["", "n"], block_shape = array<i64: 64, 64>, work_slices = [{n = 0 : i64}, {n = 1 : i64}]} : tensor<64x32xf16> -> <64x64xf16>
tt.func @block_is_the_whole(%share: tensor<64x32xf16>) {
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 64>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x64xf16>
  tt.return
}

// ---------------------------------------------------------------------------
// (e) One partition.
//
//     share [64, 32]  composed [64, 32]  block [64, 32]
//
//     Degenerate and well formed: one region, and composing it changes nothing,
//     so all three shapes coincide. Kept because the op must not require more
//     than one operand to the compose.
// ---------------------------------------------------------------------------
// CHECK-LABEL: tt.func @one_partition(
// CHECK: tts.make_distributed_descriptor %arg0 {axes = ["", "n"], block_shape = array<i64: 64, 32>, work_slices = [{n = 0 : i64}]} : tensor<64x32xf16> -> <64x32xf16>
tt.func @one_partition(%share: tensor<64x32xf16>) {
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}],
       axes = ["", "n"], block_shape = array<i64: 64, 32>}
      : tensor<64x32xf16> -> !tt.tensordesc<64x32xf16>
  tt.return
}
