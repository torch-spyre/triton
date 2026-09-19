// RUN: spyre-triton-opt %s -split-input-file --lower-descriptor-memory -verify-diagnostics

// Inputs that --lower-descriptor-memory does not accept.
//
// The six cases here fail at three different stages, and which stage a case
// hits is itself worth knowing, because it decides what a kernel author sees:
//
//   * Parse time. The Triton op verifiers run while the module is being read,
//     so a descriptor whose block shape breaks the gather contract is rejected
//     before this pass is ever invoked. Three cases below are of this kind.
//   * Conversion time, as a legalization failure. The gather pattern returns
//     failure() and applyPartialConversion leaves the illegal op in place,
//     producing two diagnostics -- one naming the op it could not legalize, one
//     naming the pass.
//   * Conversion time, as an explicit diagnostic from the pattern itself.
//
// Each expected-error annotation matches by substring against the diagnostic
// attached to the operation below it, and -split-input-file keeps one rejection
// from masking the next.
//
// Three of these are gaps rather than invariants -- inputs a kernel author can
// reasonably write and that a future change should make work. Those say so, and
// name what the fix would involve, so that closing the gap turns into an
// expected test failure here rather than silent divergence.

// -----
// A rank-reduced descriptor load. GAP, not an invariant.
//
// The idiom is standard for 3-D batched matmul: declare a descriptor with a
// leading singleton block dim, load, then reshape away the 1 because tl.dot is
// 2-D only.
//
//   a_desc = tl.make_tensor_descriptor(a_ptr, shape=[B, M, K],
//                                      strides=[M*K, K, 1],
//                                      block_shape=[1, BLOCK_M, BLOCK_K])
//   a3 = a_desc.load([b_idx, m, k])            # tensor<1xBLOCK_MxBLOCK_K>
//   a2 = tl.reshape(a3, [BLOCK_M, BLOCK_K])    # required for tl.dot
//
// Upstream's RankedReduceDescriptorLoads pattern, in the triton-combine pass
// (run as passes.ttir.add_combine in the spyre _make_ttir pipeline), folds that
// reshape into the load. Afterwards the descriptor stays 3-D
// (!tt.tensordesc<1x16x16xf32>) while the load's result type becomes 2-D
// (tensor<16x16xf32>). DescriptorLoadOp::verify accepts the mismatch because it
// compares element counts, not ranks, and 1*16*16 == 16*16.
//
// The post-combine IR is written out by hand here rather than produced from a
// reshape, because this RUN line names only LowerDescriptorMemory and
// triton-combine is not in it.
//
// What then goes wrong: the pass builds the access tile from the descriptor's
// 3-D block shape [1, 16, 16] and emits a 3-D ktdp.load, but ktdp.load requires
// its access tile shape to equal its result tensor shape, which is 2-D. So the
// KTDP verifier rejects the op this pass just built. Note the diagnostic is
// attached to ktdp.load -- the op the pass created -- not to any op in the input,
// which is why the annotation sits on the tt.descriptor_load that triggers it.
//
// Fix would be for LowerDescriptorMemory to notice that the block rank exceeds
// the load result's rank with all dropped leading dims equal to 1, and then
// either emit the tile and load at the reduced rank, or emit at full rank and
// insert a tensor.collapse_shape. Once that lands, this case should become a
// positive test.

tt.func @rank_reduced_load_rejected(%ptr: !tt.ptr<f32>, %b_idx: i32, %m: i32, %k: i32) {
  %B = arith.constant 4 : i32
  %M = arith.constant 128 : i32
  %K = arith.constant 32 : i32
  %stride_b = arith.constant 4096 : i64
  %stride_m = arith.constant 32 : i64
  %stride_k = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %ptr, [%B, %M, %K],
              [%stride_b, %stride_m, %stride_k] : <f32>, <1x16x16xf32>
  // expected-error @below {{'ktdp.load' op access tile shape must match result tensor shape}}
  %data = tt.descriptor_load %desc[%b_idx, %m, %k]
      : !tt.tensordesc<1x16x16xf32> -> tensor<16x16xf32>
  tt.return
}

// -----
// x_offsets as a tensor-typed function argument. Rejected by design.
//
// buildIndirectAccessTile resolves the index buffer by walking
// load -> construct_access_tile -> construct_memory_view and reusing that
// memref. A tensor-typed block argument has no such chain, so the trace misses,
// the pattern returns failure(), and applyPartialConversion reports the op it
// could not legalize followed by the pass's own failure.
//
// This used to lower. The old path built a fresh memref via
// unrealized_conversion_cast and emitted a bare-d0 index map, which was silently
// wrong the moment the load offset was non-zero -- see the offset-capture case
// in lower-descriptor-memory-gather.mlir for the bug that produced. Spyre
// kernels always stage indices as !tt.ptr<i32> plus tt.descriptor_load, so the
// fallback was dead code masking a misuse, and removing it turned a wrong answer
// into a compile error.
//
// Both diagnostics are pinned. The second is attached to the module rather than
// to any op inside it, so its annotation sits above `module`; an @below inside
// the function body would look for it on the wrong operation.
//
//   # NOT supported: x_offsets as a tensor-typed kernel argument.
//   @triton.jit
//   def k(ptr, x_offsets, y_offset):     # x_offsets: tensor<32xi32> -- REJECTED
//       desc = tl.make_tensor_descriptor(ptr, shape=[M, K], strides=[K, 1],
//                                        block_shape=[1, 64])
//       data = tl.descriptor_gather(desc, x_offsets, y_offset)

// expected-error @below {{LowerDescriptorMemory: failed to convert descriptor ops}}
module {
  tt.func @gather_x_offsets_arg_rejected(%ptr: !tt.ptr<f16>,
                                         %x_offsets: tensor<32xi32>,
                                         %y_offset: i32) -> tensor<32x64xf16> {
    %M = arith.constant 1024 : i32
    %K = arith.constant 128 : i32
    %stride_row = arith.constant 128 : i64
    %stride_col = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
        : <f16>, <1x64xf16>
    // expected-error @below {{failed to legalize operation 'tt.descriptor_gather' that was explicitly marked illegal}}
    %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
        : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
    tt.return %data : tensor<32x64xf16>
  }
}

// -----
// Rank-2 block whose leading dim is not 1. Invariant, enforced at parse time.
//
// <2x64xf16> declares a block of 2 rows. The gather contract is that dim 0 is
// fanned out by the index buffer one page at a time, so the block can carry
// exactly one. This is a static property of the descriptor type, so
// DescriptorGatherOp::verify fires while the module is parsed, before the pass
// runs at all.
//
//   # REJECTED: block_shape[0] != 1
//   desc = tl.make_tensor_descriptor(ptr, shape=[M, K], strides=[K, 1],
//                                    block_shape=[2, 64])
//   data = tl.descriptor_gather(desc, x_offsets, y_offset)

tt.func @gather_block_dim0_not_one_rejected(%ptr: !tt.ptr<f16>,
                                            %x_offsets: tensor<32xi32>,
                                            %y_offset: i32) {
  %M = arith.constant 1024 : i32
  %K = arith.constant 64 : i32
  %stride_row = arith.constant 64 : i64
  %stride_col = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
      : <f16>, <2x64xf16>
  // expected-error @below {{descriptor block must have exactly 1 row}}
  %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
      : (!tt.tensordesc<2x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
  tt.return
}

// -----
// The same leading-1 rule at rank 3. Invariant, parse time.
//
// Stated separately from the rank-2 case because the N-D relaxation widened
// which ranks are legal and an author's natural inference is that it also
// widened which dimension has to be 1 -- "any dim that is 1 is fine". It did
// not. Dim 0 specifically, at every rank, and this is where the per-dim loop in
// the relaxed verifier actually runs.
//
// Workarounds if a kernel genuinely wants to fetch two pages at once: keep
// block_shape[0] = 1 and pair the pages into one row (block_shape=[1, 2*16, 128]
// with a paired index buffer), or issue two gathers and concatenate.
//
//   # REJECTED: leading dim 2 even though the rank is now legal
//   desc = tl.make_tensor_descriptor(ptr, shape=[P, B, D], strides=[B*D, D, 1],
//                                    block_shape=[2, 16, 128])

tt.func @gather_3d_block_dim0_not_one_rejected(%ptr: !tt.ptr<f16>,
                                               %x_offsets: tensor<32xi32>,
                                               %y_offset: i32) {
  %P = arith.constant 1024 : i32
  %B = arith.constant 16 : i32
  %D = arith.constant 128 : i32
  %s0 = arith.constant 2048 : i64
  %s1 = arith.constant 128 : i64
  %s2 = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
      : <f16>, <2x16x128xf16>
  // expected-error @below {{descriptor block must have exactly 1 row}}
  %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
      : (!tt.tensordesc<2x16x128xf16>, tensor<32xi32>, i32) -> tensor<32x16x128xf16>
  tt.return
}

// -----
// A rank-1 descriptor block. Invariant, parse time.
//
// verifyGatherScatterOp requires rank >= 2: dim 0 is the indirect axis and must
// be 1 in the block, and at least one trailing direct dim has to exist. A
// rank-1 block has no separate indirect axis, so the op is structurally
// undefined. Spyre relaxed upstream's "must be exactly 2-D" to "rank >= 2", but
// rank 1 is still illegal.
//
// The user-facing consequence is that a 1-D source vector whose elements are
// gathered as scalars -- out[i] = in[idx[i]] -- cannot be expressed with a
// rank-1 descriptor. The supported idiom is to model the K elements as a [K, 1]
// column matrix with block_shape [1, 1]; the positive counterpart is
// @gather_1d_source_as_column in lower-descriptor-memory-gather.mlir. If the
// verifier is ever relaxed to accept rank 1, this rejection and that workaround
// should be revisited together.
//
//   # NOT supported: rank-1 descriptor block for a 1-D-source gather
//   in_desc = tl.make_tensor_descriptor(in_ptr, shape=[K], strides=[1],
//                                       block_shape=[BLOCK_COLS])  # REJECTED
//   out = tl.descriptor_gather(in_desc, idx, 0)

tt.func @gather_rank1_block_rejected(%ptr: !tt.ptr<f16>,
                                     %x_offsets: tensor<32xi32>,
                                     %y_offset: i32) {
  %K = arith.constant 1024 : i32
  %stride = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %ptr, [%K], [%stride] : <f16>, <32xf16>
  // expected-error @below {{descriptor block must be at least 2D}}
  %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
      : (!tt.tensordesc<32xf16>, tensor<32xi32>, i32) -> tensor<32xf16>
  tt.return
}

// -----
// A descriptor arriving as a runtime value. Invariant, reported by the pattern.
//
// Worth separating from the dynamic-shape cases, which are fine: a descriptor
// whose tensor *extent* is a runtime argument lowers happily to memref<?>, as
// lower-descriptor-memory-load.mlir shows. What cannot work is a descriptor that
// is *itself* an opaque runtime value -- a block argument, a call result -- since
// the lowering needs the shape and stride operands of tt.make_tensor_descriptor
// to build the view and the tile, and there is no way to recover them from a
// !tt.tensordesc value.
//
// Unlike the three verifier cases this is reported by the conversion pattern, so
// the diagnostic is the pattern's own text rather than a verifier message.

tt.func @descriptor_from_block_arg_rejected(%desc: !tt.tensordesc<1x64xf16>,
                                            %x_offsets: tensor<32xi32>,
                                            %y_offset: i32) {
  // expected-error @below {{cannot lower descriptor op}}
  %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
      : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
  tt.return
}
