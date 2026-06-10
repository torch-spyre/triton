#!/usr/bin/env python3
"""
Unit tests for individual LowerComputeOps conversion patterns.

Organization: one test class per Triton op (TestSplat, TestReshape,
TestReduce, …).  Each class contains positive tests for shape/type
variants plus negative tests for expected failure modes.

Test modules must *use* the op result (e.g. ``tt.return %0``) so
cleanupDeadOps doesn't erase the newly created ops.

Negative tests use ``pytest.raises`` + ``assert_stderr(capfd, ...)``
to verify both the RuntimeError and the MLIR diagnostic on stderr.
"""

import pytest
from conftest import SinglePassTester
from utils_pattern import pattern


class LowerComputeOpsTester(SinglePassTester):
    """Shared base for all LowerComputeOps pattern tests."""
    PASS = "add_lower_compute_ops"


# =========================================================================
# A1. tt.splat → linalg.fill
# =========================================================================

class TestSplat(LowerComputeOpsTester):
    # tt.splat → linalg.fill into tensor.empty
    # Scalar is broadcast to fill every element of the output tensor.
    #
    # test_f32_1d    — 1-D tensor, f32
    # test_i32_2d    — 2-D tensor, i32 (also checks tensor.empty emitted)
    # test_f16_3d    — 3-D tensor, f16

    @pattern("splat", category="compute", example=[
        "scalar = 1.0",
        "tensor = tl.broadcast(scalar, shape=[BLOCK_SIZE])  # tl.splat",
    ])
    def test_f32_1d(self):
        """Broadcast a scalar to fill every element of a 1-D tensor.

        ``tt.splat`` (``tl.broadcast``) lowers to ``linalg.fill`` into a
        ``tensor.empty``.  The scalar value is broadcast to every element;
        the output shape comes from the ``tt.splat`` result type.
        """
        self.run("""
        module {
          tt.func @k(%s: f32) -> tensor<1024xf32> {
            %0 = tt.splat %s : f32 -> tensor<1024xf32>
            tt.return %0 : tensor<1024xf32>
          }
        }
        """)
        self.assert_present("linalg.fill")
        self.assert_absent("tt.splat")
        self.assert_result_type("linalg.fill", "tensor<1024xf32>")

    def test_i32_2d(self):
        self.run("""
        module {
          tt.func @k(%s: i32) -> tensor<4x8xi32> {
            %0 = tt.splat %s : i32 -> tensor<4x8xi32>
            tt.return %0 : tensor<4x8xi32>
          }
        }
        """)
        self.assert_present("linalg.fill", "tensor.empty")
        self.assert_absent("tt.splat")
        self.assert_result_type("tensor.empty", "tensor<4x8xi32>")
        self.assert_result_type("linalg.fill", "tensor<4x8xi32>")

    def test_f16_3d(self):
        self.run("""
        module {
          tt.func @k(%s: f16) -> tensor<2x4x8xf16> {
            %0 = tt.splat %s : f16 -> tensor<2x4x8xf16>
            tt.return %0 : tensor<2x4x8xf16>
          }
        }
        """)
        self.assert_present("linalg.fill", "tensor.empty")
        self.assert_absent("tt.splat")
        self.assert_result_type("tensor.empty", "tensor<2x4x8xf16>")


# =========================================================================
# A2. tt.reshape → tensor.reshape
# =========================================================================

class TestReshape(LowerComputeOpsTester):
    # tt.reshape → tensor.reshape (same elements, new shape).
    # Shape tensor is built via tensor.from_elements with constant dims.
    #
    # test_1d_to_2d    — flatten → matrix
    # test_2d_to_1d    — matrix → flatten
    # test_2d_to_3d    — increase rank (also checks tensor.from_elements)
    # test_allow_reorder — allow_reorder attribute is accepted

    @pattern("reshape", category="compute", example=[
        "x = tl.reshape(x, [BLOCK_M, BLOCK_N])  # reinterpret flat tile as 2D",
    ])
    def test_1d_to_2d(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<512xf32>) -> tensor<16x32xf32> {
            %0 = tt.reshape %t : tensor<512xf32> -> tensor<16x32xf32>
            tt.return %0 : tensor<16x32xf32>
          }
        }
        """)
        self.assert_present("tensor.reshape")
        self.assert_absent("tt.reshape")
        self.assert_result_type("tensor.reshape", "tensor<16x32xf32>")

    def test_2d_to_1d(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<8x16xf16>) -> tensor<128xf16> {
            %0 = tt.reshape %t : tensor<8x16xf16> -> tensor<128xf16>
            tt.return %0 : tensor<128xf16>
          }
        }
        """)
        self.assert_present("tensor.reshape")
        self.assert_absent("tt.reshape")
        self.assert_result_type("tensor.reshape", "tensor<128xf16>")

    def test_2d_to_3d(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<4x16xi32>) -> tensor<2x2x16xi32> {
            %0 = tt.reshape %t : tensor<4x16xi32> -> tensor<2x2x16xi32>
            tt.return %0 : tensor<2x2x16xi32>
          }
        }
        """)
        self.assert_present("tensor.reshape", "tensor.from_elements")
        self.assert_absent("tt.reshape")
        self.assert_result_type("tensor.reshape", "tensor<2x2x16xi32>")

    def test_allow_reorder(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<512xf32>) -> tensor<16x32xf32> {
            %0 = tt.reshape %t allow_reorder : tensor<512xf32> -> tensor<16x32xf32>
            tt.return %0 : tensor<16x32xf32>
          }
        }
        """)
        self.assert_present("tensor.reshape")
        self.assert_absent("tt.reshape")

    @pattern("reshape", category="compute", example=[
        "# Collapse the rank-4 output of an N-D gather into a rank-2 tile",
        "# so tl.dot can consume it. OUT_LEN = NUM_BLOCKS * NUM_GROUPS * BLOCK_SIZE.",
        "gathered_4d = src_desc.gather(indices, group_idx)  # [NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM]",
        "gathered_2d = tl.reshape(gathered_4d, [OUT_LEN, INNER_DIM])",
        "out         = tl.dot(lhs, tl.trans(gathered_2d))   # 2-D dot",
    ])
    def test_4d_to_2d_collapse_three_leading_dims(self):
        """Rank-4 → rank-2 contiguous flatten that collapses three leading dims into one.

        ``tt.dot`` accepts only rank-2 (``linalg.matmul``) and rank-3
        (``linalg.batch_matmul``) operands, so the rank-4 output of an
        N-D ``tt.descriptor_gather`` must be reshaped before it can
        feed a matmul.  The reshape collapses the three leading dims
        ``[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE]`` into a single
        ``OUT_LEN`` axis while preserving the trailing contiguous dim
        ``INNER_DIM`` unchanged.

        The existing ``test_1d_to_2d`` / ``test_2d_to_1d`` /
        ``test_2d_to_3d`` cases do not exercise a rank-4 input, so a
        regression in ``LowerComputeOps``'s rank-decreasing
        ``tt.reshape`` lowering could otherwise pass unnoticed.

        Shape values: ``NUM_BLOCKS = 8``, ``NUM_GROUPS = 1``,
        ``BLOCK_SIZE = 16``, ``INNER_DIM = 64`` →
        ``OUT_LEN = NUM_BLOCKS * NUM_GROUPS * BLOCK_SIZE = 128``.
        """
        self.run("""
        module {
          tt.func @k(%t: tensor<8x1x16x64xf16>) -> tensor<128x64xf16> {
            %0 = tt.reshape %t : tensor<8x1x16x64xf16> -> tensor<128x64xf16>
            tt.return %0 : tensor<128x64xf16>
          }
        }
        """)
        self.assert_present("tensor.reshape")
        self.assert_absent("tt.reshape")
        self.assert_result_type("tensor.reshape", "tensor<128x64xf16>")


# =========================================================================
# A3. tt.expand_dims → tensor.expand_shape
# =========================================================================

class TestExpandDims(LowerComputeOpsTester):
    # tt.expand_dims → tensor.expand_shape (insert a size-1 dimension).
    # Axis attribute controls where the new dim is inserted.
    #
    # test_axis_0         — insert size-1 dim at front (1-D → 2-D)
    # test_axis_1         — insert size-1 dim at end   (1-D → 2-D)
    # test_2d_middle_axis — insert size-1 dim in middle (2-D → 3-D)

    @pattern("expand-dims", category="compute", example=[
        "x = tl.load(x_ptr + offsets)          # tensor<BLOCK x f32>",
        "x = tl.expand_dims(x, axis=0)         # tensor<1 x BLOCK x f32>",
    ])
    def test_axis_0(self):
        """tensor<8xf32> → tensor<1x8xf32>  (insert dim at front)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<8xf32>) -> tensor<1x8xf32> {
            %0 = tt.expand_dims %t {axis = 0 : i32} : tensor<8xf32> -> tensor<1x8xf32>
            tt.return %0 : tensor<1x8xf32>
          }
        }
        """)
        self.assert_present("tensor.expand_shape")
        self.assert_absent("tt.expand_dims")
        self.assert_result_type("tensor.expand_shape", "tensor<1x8xf32>")

    def test_axis_1(self):
        """tensor<8xf32> → tensor<8x1xf32>  (insert dim at end)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<8xf32>) -> tensor<8x1xf32> {
            %0 = tt.expand_dims %t {axis = 1 : i32} : tensor<8xf32> -> tensor<8x1xf32>
            tt.return %0 : tensor<8x1xf32>
          }
        }
        """)
        self.assert_present("tensor.expand_shape")
        self.assert_absent("tt.expand_dims")
        self.assert_result_type("tensor.expand_shape", "tensor<8x1xf32>")

    def test_2d_middle_axis(self):
        """tensor<4x8xf16> → tensor<4x1x8xf16>  (insert dim in middle)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8xf16>) -> tensor<4x1x8xf16> {
            %0 = tt.expand_dims %t {axis = 1 : i32} : tensor<4x8xf16> -> tensor<4x1x8xf16>
            tt.return %0 : tensor<4x1x8xf16>
          }
        }
        """)
        self.assert_present("tensor.expand_shape")
        self.assert_absent("tt.expand_dims")
        self.assert_result_type("tensor.expand_shape", "tensor<4x1x8xf16>")


# =========================================================================
# A4. tt.broadcast → tensor.collapse_shape + linalg.broadcast
# =========================================================================

class TestBroadcast(LowerComputeOpsTester):
    # tt.broadcast → tensor.collapse_shape + linalg.broadcast.
    # Size-1 dims are collapsed out, then linalg.broadcast expands them.
    #
    # test_expand_first_dim    — broadcast dim 0 (1→4)
    # test_expand_last_dim     — broadcast dim 1 (1→8)
    # test_expand_multiple_dims — broadcast two dims simultaneously
    # test_noop_same_shape     — no size-1 dims → replaced with input directly

    @pattern("broadcast", category="compute", example=[
        "row = tl.load(row_ptr + tl.arange(0, N))  # tensor<N x f32>",
        "row = tl.expand_dims(row, axis=0)          # tensor<1 x N x f32>",
        "mat = tl.broadcast_to(row, [M, N])         # tensor<M x N x f32>",
    ])
    def test_expand_first_dim(self):
        """tensor<1x8xf32> → tensor<4x8xf32>  (broadcast dim 0)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<1x8xf32>) -> tensor<4x8xf32> {
            %0 = tt.broadcast %t : tensor<1x8xf32> -> tensor<4x8xf32>
            tt.return %0 : tensor<4x8xf32>
          }
        }
        """)
        self.assert_present("linalg.broadcast")
        self.assert_absent("tt.broadcast")
        self.assert_result_type("linalg.broadcast", "tensor<4x8xf32>")

    def test_expand_last_dim(self):
        """tensor<4x1xf32> → tensor<4x8xf32>  (broadcast dim 1)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<4x1xf32>) -> tensor<4x8xf32> {
            %0 = tt.broadcast %t : tensor<4x1xf32> -> tensor<4x8xf32>
            tt.return %0 : tensor<4x8xf32>
          }
        }
        """)
        self.assert_present("linalg.broadcast")
        self.assert_absent("tt.broadcast")
        self.assert_result_type("linalg.broadcast", "tensor<4x8xf32>")

    def test_expand_multiple_dims(self):
        """tensor<1x8x1xf16> → tensor<4x8x16xf16>  (broadcast dims 0 and 2)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<1x8x1xf16>) -> tensor<4x8x16xf16> {
            %0 = tt.broadcast %t : tensor<1x8x1xf16> -> tensor<4x8x16xf16>
            tt.return %0 : tensor<4x8x16xf16>
          }
        }
        """)
        self.assert_present("linalg.broadcast")
        self.assert_absent("tt.broadcast")
        self.assert_result_type("linalg.broadcast", "tensor<4x8x16xf16>")

    def test_all_dims_broadcast(self):
        """tensor<1x1xf32> → tensor<4x8xf32>  (all dims are broadcast dims)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<1x1xf32>) -> tensor<4x8xf32> {
            %0 = tt.broadcast %t : tensor<1x1xf32> -> tensor<4x8xf32>
            tt.return %0 : tensor<4x8xf32>
          }
        }
        """)
        self.assert_present("linalg.broadcast")
        self.assert_absent("tt.broadcast")
        self.assert_result_type("linalg.broadcast", "tensor<4x8xf32>")

    def test_noop_same_shape(self):
        """tensor<4x8xf32> → tensor<4x8xf32>  (no size-1 dims → no-op)"""
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8xf32>) -> tensor<4x8xf32> {
            %0 = tt.broadcast %t : tensor<4x8xf32> -> tensor<4x8xf32>
            tt.return %0 : tensor<4x8xf32>
          }
        }
        """)
        self.assert_absent("tt.broadcast", "linalg.broadcast")


# =========================================================================
# A5. tt.trans → linalg.transpose
# =========================================================================

class TestTrans(LowerComputeOpsTester):
    # tt.trans → linalg.transpose (permute dimensions via order attr).
    # The order attribute specifies the output-to-input dimension mapping.
    #
    # test_2d_transpose — swap rows and columns
    # test_3d_permute   — 3-D permutation [2, 0, 1]

    @pattern("transpose", category="compute", example=[
        "a = tl.load(a_desc, [m * BM, k * BK])     # tensor<BM x BK x f32>",
        "a_t = tl.trans(a)                          # tensor<BK x BM x f32>",
    ])
    def test_2d_transpose(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8xf32>) -> tensor<8x4xf32> {
            %0 = tt.trans %t {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
            tt.return %0 : tensor<8x4xf32>
          }
        }
        """)
        self.assert_present("linalg.transpose")
        self.assert_absent("tt.trans")
        self.assert_result_type("linalg.transpose", "tensor<8x4xf32>")

    def test_3d_permute(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<2x4x8xf16>) -> tensor<8x2x4xf16> {
            %0 = tt.trans %t {order = array<i32: 2, 0, 1>} : tensor<2x4x8xf16> -> tensor<8x2x4xf16>
            tt.return %0 : tensor<8x2x4xf16>
          }
        }
        """)
        self.assert_present("linalg.transpose")
        self.assert_absent("tt.trans")
        self.assert_result_type("linalg.transpose", "tensor<8x2x4xf16>")


# =========================================================================
# A6. tt.join → tensor.expand_shape + tensor.concat
# =========================================================================

class TestJoin(LowerComputeOpsTester):
    # tt.join → tensor.expand_shape + tensor.concat.
    # Two same-shape tensors are joined along a new last dim of size 2.
    #
    # test_1d — two 1-D tensors → 2-D result
    # test_2d — two 2-D tensors → 3-D result

    @pattern("join", category="compute", example=[
        "real = tl.load(real_ptr + offsets)   # tensor<BLOCK x f32>",
        "imag = tl.load(imag_ptr + offsets)   # tensor<BLOCK x f32>",
        "pair = tl.join(real, imag)           # tensor<BLOCK x 2 x f32>",
    ])
    def test_1d(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8x2xf32> {
            %0 = tt.join %a, %b : tensor<8xf32> -> tensor<8x2xf32>
            tt.return %0 : tensor<8x2xf32>
          }
        }
        """)
        self.assert_present("tensor.concat")
        self.assert_absent("tt.join")
        self.assert_result_type("tensor.concat", "tensor<8x2xf32>")

    def test_2d(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<4x8xf16>, %b: tensor<4x8xf16>) -> tensor<4x8x2xf16> {
            %0 = tt.join %a, %b : tensor<4x8xf16> -> tensor<4x8x2xf16>
            tt.return %0 : tensor<4x8x2xf16>
          }
        }
        """)
        self.assert_present("tensor.concat")
        self.assert_absent("tt.join")
        self.assert_result_type("tensor.concat", "tensor<4x8x2xf16>")


# =========================================================================
# A7. tt.split → tensor.extract_slice + tensor.collapse_shape
# =========================================================================

class TestSplit(LowerComputeOpsTester):
    # tt.split → tensor.extract_slice × 2 + tensor.collapse_shape × 2.
    # Last dim (size 2) is split into two tensors with that dim removed.
    #
    # test_2d — tensor<8x2> → two tensor<8>
    # test_3d — tensor<4x8x2> → two tensor<4x8>

    @pattern("split", category="compute", example=[
        "pair = tl.load(pair_desc, [pid * BLOCK])  # tensor<BLOCK x 2 x f32>",
        "real, imag = tl.split(pair)               # two tensor<BLOCK x f32>",
    ])
    def test_2d(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<8x2xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
            %0, %1 = tt.split %t : tensor<8x2xf32> -> tensor<8xf32>
            tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
          }
        }
        """)
        self.assert_present("tensor.extract_slice", "tensor.collapse_shape")
        self.assert_absent("tt.split")
        self.assert_result_type("tensor.collapse_shape", "tensor<8xf32>")

    def test_3d(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8x2xf16>) -> (tensor<4x8xf16>, tensor<4x8xf16>) {
            %0, %1 = tt.split %t : tensor<4x8x2xf16> -> tensor<4x8xf16>
            tt.return %0, %1 : tensor<4x8xf16>, tensor<4x8xf16>
          }
        }
        """)
        self.assert_present("tensor.extract_slice", "tensor.collapse_shape")
        self.assert_absent("tt.split")
        self.assert_result_type("tensor.collapse_shape", "tensor<4x8xf16>")


# =========================================================================
# B1. tt.reduce → linalg.reduce
# =========================================================================

class TestReduce(LowerComputeOpsTester):
    # tt.reduce → linalg.reduce (reduce along axis with combiner region).
    # Combiner body is cloned; identity element derived from the combiner op.
    #
    # test_sum_f32              — addf combiner (identity 0.0)
    # test_max_f32              — maxnumf combiner (identity -inf)
    # test_unsupported_combiner — subf has no identity → expected failure

    def test_axis_0(self):
        """tensor<4x8xf32> reduce along axis=0 → tensor<8xf32>"""
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8xf32>) -> tensor<8xf32> {
            %0 = "tt.reduce"(%t) ({
            ^bb0(%a: f32, %b: f32):
              %add = arith.addf %a, %b : f32
              tt.reduce.return %add : f32
            }) {axis = 0 : i32} : (tensor<4x8xf32>) -> tensor<8xf32>
            tt.return %0 : tensor<8xf32>
          }
        }
        """)
        self.assert_present("linalg.reduce")
        self.assert_absent("tt.reduce")
        self.assert_result_type("linalg.reduce", "tensor<8xf32>")

    @pattern("reduce", category="compute", example=[
        "x = tl.load(x_ptr + offsets)       # tensor<BLOCK_M x BLOCK_N x f32>",
        "row_sum = tl.sum(x, axis=1)        # tensor<BLOCK_M x f32>",
    ])
    def test_sum_f32(self):
        """Sum-reduce a 2-D tensor along an axis using ``arith.addf``.

        ``tt.reduce`` with an ``addf`` combiner lowers to ``linalg.reduce``
        with identity ``0.0``.  The output rank is one less than the input.
        """
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8xf32>) -> tensor<4xf32> {
            %0 = "tt.reduce"(%t) ({
            ^bb0(%a: f32, %b: f32):
              %add = arith.addf %a, %b : f32
              tt.reduce.return %add : f32
            }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
            tt.return %0 : tensor<4xf32>
          }
        }
        """)
        self.assert_present("linalg.reduce")
        self.assert_absent("tt.reduce")
        self.assert_result_type("linalg.reduce", "tensor<4xf32>")

    @pattern("reduce", category="compute", example=[
        "x = tl.load(x_ptr + offsets)       # tensor<BLOCK_M x BLOCK_N x f32>",
        "row_max = tl.max(x, axis=1)        # tensor<BLOCK_M x f32>",
    ])
    def test_max_f32(self):
        """Max-reduce a 2-D tensor along an axis using ``arith.maxnumf``.

        ``tt.reduce`` with a ``maxnumf`` combiner lowers to ``linalg.reduce``
        with identity ``-inf``.  NaN-handling follows ``arith.maxnumf``
        semantics (NaN propagates from the right operand only).
        """
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8xf32>) -> tensor<4xf32> {
            %0 = "tt.reduce"(%t) ({
            ^bb0(%a: f32, %b: f32):
              %max = arith.maxnumf %a, %b : f32
              tt.reduce.return %max : f32
            }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
            tt.return %0 : tensor<4xf32>
          }
        }
        """)
        self.assert_present("linalg.reduce")
        self.assert_absent("tt.reduce")
        self.assert_result_type("linalg.reduce", "tensor<4xf32>")

    @pattern("reduce", category="compute", negative=True, example=[
        "# arith.subf has no neutral element — tl.reduce with subtraction is not supported",
        "result = tl.reduce(x, axis=1, combine_fn=lambda a, b: a - b)",
    ])
    def test_subf_combiner_fails(self, capfd):
        """arith.subf has no neutral element → pattern returns failure().

        linalg.reduce needs an identity/init value for the accumulator
        (e.g. 0.0 for addf, -inf for maxnumf).  arith.subf has no
        neutral element — arith::getNeutralElement returns nullopt —
        so ConvertTTReduce returns failure() and the op stays illegal.
        """
        import pytest
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%t: tensor<4x8xf32>) -> tensor<4xf32> {
                %0 = "tt.reduce"(%t) ({
                ^bb0(%a: f32, %b: f32):
                  %sub = arith.subf %a, %b : f32
                  tt.reduce.return %sub : f32
                }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
                tt.return %0 : tensor<4xf32>
              }
            }
            """)
        self.assert_stderr(capfd,
            "failed to legalize operation 'tt.reduce'",
            "LowerComputeOps: failed to convert compute ops",
        )

    def test_math_combiner(self):
        # math.absf appears inside the combiner body alongside arith.addf.
        self.run("""
        module {
          tt.func @k(%t: tensor<4x8xf32>) -> tensor<4xf32> {
            %0 = "tt.reduce"(%t) ({
            ^bb0(%a: f32, %b: f32):
              %abs = math.absf %b : f32
              %add = arith.addf %a, %abs : f32
              tt.reduce.return %add : f32
            }) {axis = 1 : i32} : (tensor<4x8xf32>) -> tensor<4xf32>
            tt.return %0 : tensor<4xf32>
          }
        }
        """)
        self.assert_present("linalg.reduce")
        self.assert_absent("tt.reduce")

    def test_reduce_to_scalar(self):
        # Reduce 1-D tensor along axis=0 → scalar result.
        self.run("""
        module {
          tt.func @k(%t: tensor<8xf32>) -> f32 {
            %0 = "tt.reduce"(%t) ({
            ^bb0(%a: f32, %b: f32):
              %add = arith.addf %a, %b : f32
              tt.reduce.return %add : f32
            }) {axis = 0 : i32} : (tensor<8xf32>) -> f32
            tt.return %0 : f32
          }
        }
        """)
        self.assert_present("linalg.reduce")
        self.assert_absent("tt.reduce")

    @pattern("reduce", category="compute", example=[
        "values, indices = tl.argmax(x, axis=1, return_indices=True)",
        "# lowers to linalg.reduce with two inputs (f32 values + i32 indices)",
        "# index lane initialised with -1 as an invalid-index sentinel",
    ])
    def test_multi_operand_reduce(self):
        """Multi-operand ``tt.reduce`` (e.g. argmax) lowers to a single ``linalg.reduce``.

        Two input tensors are reduced together: the value lane uses ``arith.maxnumf``
        (identity -inf) and the index lane uses ``arith.select`` (identity -1).
        Both are passed as inputs to one ``linalg.reduce`` op whose combiner
        region is cloned directly from the ``tt.reduce`` combiner.
        """
        self.run("""
        module {
          tt.func @k(%vals: tensor<4x8xf32>, %idxs: tensor<4x8xi32>)
              -> (tensor<4xf32>, tensor<4xi32>) {
            %0:2 = "tt.reduce"(%vals, %idxs) ({
            ^bb0(%a: f32, %ai: i32, %b: f32, %bi: i32):
              %cmp = arith.cmpf ogt, %a, %b : f32
              %max = arith.maxnumf %a, %b : f32
              %idx = arith.select %cmp, %ai, %bi : i32
              tt.reduce.return %max, %idx : f32, i32
            }) {axis = 1 : i32} : (tensor<4x8xf32>, tensor<4x8xi32>) -> (tensor<4xf32>, tensor<4xi32>)
            tt.return %0#0, %0#1 : tensor<4xf32>, tensor<4xi32>
          }
        }
        """)
        self.assert_present("linalg.reduce")
        self.assert_absent("tt.reduce")


# =========================================================================
# C1. tt.dot → linalg.matmul
# =========================================================================

class TestDot(LowerComputeOpsTester):
    # tt.dot → linalg.matmul / linalg.batch_matmul.
    # The accumulator c is passed as the outs operand.
    #
    # test_f32         — basic f32 matmul (2D)
    # test_f16         — f16 inputs with f32 accumulator (2D)
    # test_large       — larger tile sizes (2D)
    # test_batch_matmul — 3D tt.dot → linalg.batch_matmul

    @pattern("dot", category="compute", example=[
        "acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)",
        "for k in range(k_tiles):",
        "    a = a_desc.load([m * BM, k * BK])  # tensor<BM x BK x f32>",
        "    b = b_desc.load([k * BK, n * BN])  # tensor<BK x BN x f32>",
        "    acc = tl.dot(a, b, acc)             # tensor<BM x BN x f32>",
    ])
    def test_f32(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<16x32xf32>, %b: tensor<32x8xf32>,
                     %c: tensor<16x8xf32>) -> tensor<16x8xf32> {
            %0 = tt.dot %a, %b, %c : tensor<16x32xf32> * tensor<32x8xf32> -> tensor<16x8xf32>
            tt.return %0 : tensor<16x8xf32>
          }
        }
        """)
        self.assert_present("linalg.matmul")
        self.assert_absent("tt.dot")
        self.assert_result_type("linalg.matmul", "tensor<16x8xf32>")

    def test_f16(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<16x32xf16>, %b: tensor<32x8xf16>,
                     %c: tensor<16x8xf32>) -> tensor<16x8xf32> {
            %0 = tt.dot %a, %b, %c : tensor<16x32xf16> * tensor<32x8xf16> -> tensor<16x8xf32>
            tt.return %0 : tensor<16x8xf32>
          }
        }
        """)
        self.assert_present("linalg.matmul")
        self.assert_absent("tt.dot")
        self.assert_result_type("linalg.matmul", "tensor<16x8xf32>")

    def test_large(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>,
                     %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
            %0 = tt.dot %a, %b, %c : tensor<128x64xf32> * tensor<64x128xf32> -> tensor<128x128xf32>
            tt.return %0 : tensor<128x128xf32>
          }
        }
        """)
        self.assert_present("linalg.matmul")
        self.assert_absent("tt.dot")
        self.assert_result_type("linalg.matmul", "tensor<128x128xf32>")

    def test_batch_matmul(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<4x16x32xf32>, %b: tensor<4x32x8xf32>,
                     %c: tensor<4x16x8xf32>) -> tensor<4x16x8xf32> {
            %0 = tt.dot %a, %b, %c : tensor<4x16x32xf32> * tensor<4x32x8xf32> -> tensor<4x16x8xf32>
            tt.return %0 : tensor<4x16x8xf32>
          }
        }
        """)
        self.assert_present("linalg.batch_matmul")
        self.assert_absent("tt.dot")
        self.assert_result_type("linalg.batch_matmul", "tensor<4x16x8xf32>")

    @pattern("dot", category="compute", negative=True, example=[
        "# REJECTED: rank-4 tt.dot.",
        "#   tl.dot(a4, b4)  where a4, b4 are rank-4 tensors",
        "#       →  'tt.dot op expected operands to be 2d or 3d'",
        "#",
        "# Why this matters for kernel authors: an N-D indirect-access",
        "# gather produces rank-4 (or rank-5 stickified) tiles. Those",
        "# tiles must be reshaped down to rank-2 BEFORE feeding tl.dot —",
        "# the dot op itself does not flatten its inputs.",
    ])
    def test_dot_4d_rejected(self, capfd):
        """Rank ≥ 4 ``tt.dot`` is rejected by the upstream Triton verifier.

        The verifier in ``OpInterfaces.cpp::DotOpInterface`` enforces
        ``aShape.size() in {2, 3}``; rank 4+ is rejected at parse time
        with ``'tt.dot' op expected operands to be 2d or 3d``. The
        diagnostic fires during ``ir.parse_mlir_module``, before the
        ``LowerComputeOps`` pass runs, so the raised exception is
        ``"Parse MLIR file failed"``.

        Why pin this:

        * Documents the rank cap in the test suite — readers of
          ``TestDot`` see only positive 2-D/3-D tests, so the cap is
          otherwise invisible until a kernel author tries it.
        * Explains why an N-D gather feeding a matmul must first
          reshape its rank-4 output down to rank-2: ``tl.dot`` cannot
          consume rank-4 directly, so the kernel must collapse the
          rank-4 ``[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM]``
          tile to rank-2 ``[OUT_LEN, INNER_DIM]`` first. The companion
          test ``TestReshape.test_4d_to_2d_collapse_three_leading_dims``
          pins that reshape step.

        Note on the silent-fallthrough in ``ConvertTTDot``: the Spyre
        lowering pattern dispatches on ``aType.getRank()`` with
        branches for rank 2 (``linalg.matmul``) and rank 3
        (``linalg.batch_matmul``); rank ≥ 4 falls through to the rank-2
        branch and would build an invalid ``linalg.matmul``. That code
        path is dead today because the verifier rejects rank-4 first,
        but if the upstream verifier ever relaxed, the fallthrough
        would activate. A separate test pinning that the lowering
        rejects rank ≥ 4 with a Spyre-specific diagnostic — rather
        than silently building broken ``linalg.matmul`` IR — would
        close that gap.

        TODO: tighten ``ConvertTTDot`` to either handle rank ≥ 4 or
        explicitly emit ``failure()`` with a diagnostic, then add a
        positive test for the rejected-with-diagnostic path.
        """
        with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
            self.run("""
            module {
              tt.func @k(%a: tensor<2x4x16x32xf32>, %b: tensor<2x4x32x8xf32>,
                         %c: tensor<2x4x16x8xf32>) -> tensor<2x4x16x8xf32> {
                %0 = tt.dot %a, %b, %c
                    : tensor<2x4x16x32xf32> * tensor<2x4x32x8xf32> -> tensor<2x4x16x8xf32>
                tt.return %0 : tensor<2x4x16x8xf32>
              }
            }
            """)
        self.assert_stderr(capfd, "expected operands to be 2d or 3d")
