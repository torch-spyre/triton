// RUN: spyre-triton-opt %s --lower-descriptor-memory --convert-ttir-functions -verify-diagnostics

// tt.addptr feeding tt.make_tensor_descriptor. GAP, not an invariant.
//
// This file needs two passes in its RUN line, which is why it is separate from
// lower-descriptor-memory-invalid.mlir. LowerDescriptorMemory alone accepts this
// input: it lowers the descriptor ops via getBasePtrAsIndex, which casts the
// base !tt.ptr operand to index, and leaves the tt.addptr that computed that
// base still reading the !tt.ptr function argument. Nothing is wrong yet. The
// failure only surfaces in ConvertFunctions, which retypes !tt.ptr arguments to
// index and can only fix up unrealized_conversion_cast users of such an
// argument. It finds tt.addptr using the argument directly, so it rejects the
// module up front -- before any retyping -- and names the real cause: the memory
// passes were supposed to consume every !tt.ptr use first.
//
// So the diagnostic lands on the tt.addptr, not on the descriptor op, and the
// annotation sits accordingly.
//
// This is what keeps batched matmul's per-batch-offset variant disabled. Batched
// matmul itself compiles fine today through 3-D descriptors whose base is the
// raw buffer pointer (fixtures/matmul/kernel.py::bmm_matmul_kernel). The gap is
// specifically the idiom where each batch step offsets the base pointer before
// building the descriptor:
//
//   # NOT supported: tt.addptr result as a descriptor base
//   base = a_ptr + b_idx * stride_batch          # tt.addptr
//   desc = tl.make_tensor_descriptor(base, shape=[M, K], strides=[K, 1],
//                                    block_shape=[BLOCK_M, BLOCK_K])
//
// fixtures/matmul/meta.py disables matmul__bmm_addptr and bmm_addptr_dynamic and
// points its tracking_test at this file, so if the gap closes and this test is
// deleted, that pointer goes stale and should be cleaned up along with the
// disabled block.
//
// Fix would be for LowerDescriptorMemory to fold the tt.addptr into the base and
// offset it hands to construct_memory_view, or to canonicalize the pointer
// arithmetic into index arithmetic before ConvertFunctions runs. Once that
// lands, this becomes a positive descriptor-with-offset test.
//
// One reason to pin ConvertFunctions' precondition text specifically: without
// that check the pass succeeds and tt.addptr's own verifier complains instead,
// with a "must be ptr ... got 'index'" message about an op that did nothing
// wrong. That text must not come back here.

tt.func @addptr_into_descriptor_rejected(%a_ptr: !tt.ptr<f32>, %offset: i32) {
  %c0_i32 = arith.constant 0 : i32
  %M = arith.constant 1024 : i32
  %K = arith.constant 64 : i32
  %stride_row = arith.constant 64 : i64
  %stride_col = arith.constant 1 : i64
  // expected-error @below {{cannot convert function signature: !tt.ptr argument #0 of 'addptr_into_descriptor_rejected' is used by an op that is not an unrealized_conversion_cast}}
  %base = tt.addptr %a_ptr, %offset : !tt.ptr<f32>, i32
  %desc = tt.make_tensor_descriptor %base, [%M, %K], [%stride_row, %stride_col]
      : <f32>, <16x16xf32>
  %data = tt.descriptor_load %desc[%c0_i32, %c0_i32]
      : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
  tt.return
}
