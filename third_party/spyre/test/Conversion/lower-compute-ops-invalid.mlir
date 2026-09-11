// RUN: spyre-triton-opt %s -split-input-file --lower-compute-ops -verify-diagnostics

// Inputs that --lower-compute-ops does not accept.
//
// The two cases here fail at different stages, which is the reason they are worth
// reading together:
//
//   * The subf reduce parses fine and is rejected by the pass, because the
//     conversion target marks tt.reduce illegal and no pattern can legalize it.
//   * The rank-4 dot never reaches the pass at all -- the upstream Triton verifier
//     rejects it while the module is being parsed.
//
// Each annotation matches by substring on the diagnostic attached to the line
// below it. -split-input-file keeps one rejection from masking the next.

// -----
// arith.subf has no neutral element, so tt.reduce with a subtraction combiner
// cannot be lowered.
//
// linalg.reduce needs an identity value to initialize the accumulator (0.0 for
// addf, -inf for maxnumf -- see lower-compute-ops-reduce.mlir). The pass obtains
// it from arith::getNeutralElement, which returns nullopt for subf, so the
// conversion pattern returns failure() and tt.reduce stays illegal.
//
// Two diagnostics come out and both are pinned: the conversion framework reports
// the op it could not legalize, and the pass then reports its own failure against
// the module. The annotation for the second sits above `module` because that is
// the operation it is attached to; a `@below` inside the function body would look
// for it on the wrong op. Note also that the tt.reduce annotation must sit
// directly above the `%0 = "tt.reduce"` line rather than above the enclosing
// tt.func, since @below binds to the next operation.
//
// Triton source pattern:
//
//   # arith.subf has no neutral element -- tl.reduce with subtraction is rejected
//   result = tl.reduce(x, axis=1, combine_fn=lambda a, b: a - b)

// expected-error @below {{LowerComputeOps: failed to convert compute ops}}
module {
  tt.func @reduce_subf_combiner_rejected(%t: tensor<4x8xf32>) -> tensor<4xf32> {
    // expected-error @below {{failed to legalize operation 'tt.reduce' that was explicitly marked illegal}}
    %0 = "tt.reduce"(%t) ({
    ^bb0(%a: f32, %b: f32):
      %sub = arith.subf %a, %b : f32
      tt.reduce.return %sub : f32
    }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// -----
// Rank-4 tt.dot is rejected by the upstream Triton verifier, before this pass runs.
//
// DotOpInterface in OpInterfaces.cpp enforces that the operand rank is 2 or 3, so
// rank 4 fails at parse time. Two reasons to pin it here rather than leave it
// implicit:
//
//   * lower-compute-ops-dot.mlir holds only positive 2-D and 3-D cases, so the
//     rank cap is otherwise invisible until a kernel author trips over it.
//   * It explains why an N-D gather feeding a matmul must reshape first: tl.dot
//     does not flatten its inputs, so a rank-4 [NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE,
//     INNER_DIM] tile has to be collapsed to rank-2 [OUT_LEN, INNER_DIM] before the
//     dot. That reshape step is pinned by
//     lower-compute-ops-shape.mlir's reshape_4d_to_2d_collapse_three_leading_dims.
//
// A gap this leaves open, worth recording: ConvertTTDot dispatches on
// aType.getRank() with branches for rank 2 and rank 3, and rank >= 4 falls through
// to the rank-2 branch, where it would build an invalid linalg.matmul. That path is
// dead today only because the verifier rejects rank 4 first. If the upstream
// verifier ever relaxed, the fallthrough would activate silently. Tightening
// ConvertTTDot to emit its own failure() with a diagnostic -- and adding a positive
// test for that rejection -- would close it.
//
// Triton source pattern:
//
//   # REJECTED: rank-4 tl.dot -- the verifier accepts only rank 2 or 3.
//   a4  = tl.descriptor_gather(a_desc, idx, y)   # rank-4 tile
//   b4  = tl.descriptor_gather(b_desc, idx, y)   # same rank-4 shape
//   out = tl.dot(a4, b4)                          # 'expected operands to be 2d or 3d'
//   # Workaround: collapse the leading dims first.
//   a2  = tl.reshape(a4, [OUT_LEN, INNER])       # rank-2
//   b2  = tl.reshape(b4, [INNER, OUT_LEN])
//   out = tl.dot(a2, b2)                          # accepted

tt.func @dot_rank_4_rejected(%a: tensor<2x4x16x32xf32>, %b: tensor<2x4x32x8xf32>,
                             %c: tensor<2x4x16x8xf32>) -> tensor<2x4x16x8xf32> {
  // expected-error @below {{'tt.dot' op expected operands to be 2d or 3d}}
  %0 = tt.dot %a, %b, %c
      : tensor<2x4x16x32xf32> * tensor<2x4x32x8xf32> -> tensor<2x4x16x8xf32>
  tt.return %0 : tensor<2x4x16x8xf32>
}
