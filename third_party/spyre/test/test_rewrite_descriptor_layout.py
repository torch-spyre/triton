#!/usr/bin/env python3
"""Unit tests for the RewriteDescriptorLayout pass.

RewriteDescriptorLayout runs *last* in the TTIR→KTDP pipeline (after
LowerDescriptorMemory + LowerComputeOps, before ConvertFunctions). It consumes
``tt.spyre_tensor_layout`` markers and rewrites the KTDP memory ops they
annotate from logical shape to physical (stick-tiled) device layout, then
erases the markers.

Because the pass operates on already-lowered KTDP IR, tests here run the three
preceding passes (LowerDescriptorMemory + LowerComputeOps + ConvertFunctions are
installed by ``FullPipelineTester._build_passes``) before invoking the pass
under test. Assertions check KTDP op shapes and presence/absence rather than
Triton descriptor types.

The marker carries the OpSpec ``device_coordinates`` form as three i64 arrays,
one entry per *physical* dim:
  phys_src[k] : which logical dim physical dim k derives from
  phys_op[k]  : 0 = identity, 1 = floordiv, 2 = mod
  phys_arg[k] : divisor (floordiv) / modulus (mod); ignored for identity

Worked example used throughout — an ``[M, N]`` tensor stick-tiled on ``N``
(the vector_add / pointwise shape), ``device_size = [N//64, M, 64]``:
  phys_src = [1, 0, 1]   # N//64 <- dim1 ; M <- dim0 ; N%64 <- dim1
  phys_op  = [1, 0, 2]   # floordiv, identity, mod
  phys_arg = [64, 0, 64]
"""

import pytest
from conftest import SinglePassTester
from utils_pattern import pattern


# Marker attr for the [M,N] stick-on-N layout, reused across tests.
_STICK_ON_N = (
    "{phys_src = array<i64: 1, 0, 1>, "
    "phys_op = array<i64: 1, 0, 2>, "
    "phys_arg = array<i64: 64, 0, 64>}"
)


class RewriteLayoutTester(SinglePassTester):
    """Shared base: run the full TTIR→KTDP pipeline up to and including
    RewriteDescriptorLayout.

    LowerDescriptorMemory + LowerComputeOps run first so the pass sees
    lowered KTDP IR (ktdp.construct_memory_view / construct_access_tile /
    load / store) as its input, which is where it now operates.
    """
    PASS = "add_rewrite_descriptor_layout"

    def _build_passes(self, pm):
        from triton._C.libtriton import spyre
        spyre.passes.ttir_to_ktdp.add_lower_descriptor_memory(pm)
        spyre.passes.ttir_to_ktdp.add_lower_compute_ops(pm)
        spyre.passes.ttir_to_ktdp.add_rewrite_descriptor_layout(pm)


# =========================================================================
# Static shapes: [M, N] stick-on-N -> physical [N//64, M, 64]
# =========================================================================

class TestStaticLayout(RewriteLayoutTester):

    def _kernel(self, M, N, BM, BN):
        # Logical 2D descriptor + stick-on-N marker + a load and a store of an
        # elementwise result, mirroring the add kernel's body shape.
        return f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %m: i32, %n: i32) {{
            %M = arith.constant {M} : i32
            %N = arith.constant {N} : i32
            %sr = arith.constant {N} : i64
            %sc = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%M, %N], [%sr, %sc]
                : <f16>, <{BM}x{BN}xf16>
            tt.spyre_tensor_layout %desc {_STICK_ON_N} : <{BM}x{BN}xf16>
            %d = tt.descriptor_load %desc[%m, %n]
                : !tt.tensordesc<{BM}x{BN}xf16> -> tensor<{BM}x{BN}xf16>
            tt.descriptor_store %desc[%m, %n], %d
                : !tt.tensordesc<{BM}x{BN}xf16>, tensor<{BM}x{BN}xf16>
            tt.return
          }}
        }}
        """

    def test_marker_erased(self):
        # The pass consumes and removes every tt.spyre_tensor_layout marker.
        self.run(self._kernel(512, 256, 16, 64))
        self.assert_absent("tt.spyre_tensor_layout")

    @pattern("physical-layout-rewrite", category="memory", example=[
        "desc = tl.make_tensor_descriptor(ptr, shape=[M, N], strides=[N, 1],",
        "                                 block_shape=[BLOCK_M, BLOCK_N])",
        "# annotate the physical (stick-tiled) device layout; the pass rewrites",
        "# the lowered KTDP memory view + access tile to the physical layout.",
        "tl.spyre_tensor_layout(desc, [(1, 'floordiv', 64),  # N // 64 (stick index)",
        "                              0,                     # M       (identity)",
        "                              (1, 'mod', 64)])       # N % 64  (stick lane)",
    ])
    def test_memory_view_is_physical_rank3(self):
        # M=512, N=256, stick-on-N: full tensor shape 256//64=4 x 512 x 64.
        # The memory view carries the full tensor extents; the access tile
        # carries the block shape (BN//64 x BM x 64 = 1x16x64).
        self.run(self._kernel(512, 256, 16, 64))
        self.assert_result_type("ktdp.construct_memory_view", "4x512x64xf16")

    def test_load_is_physical_rank3(self):
        # ktdp.load result is the physical block: BN//64 x BM x lane = 1x16x64.
        self.run(self._kernel(512, 256, 16, 64))
        self.assert_result_type("ktdp.load", "tensor<1x16x64xf16>")

    @pytest.mark.parametrize("BN", [64, 128])
    def test_stick_count_in_block(self, BN):
        # Physical block dim 0 = BN // 64 (number of sticks per tile).
        self.run(self._kernel(512, 256, 16, BN))
        self.assert_result_type("ktdp.load",
                                f"tensor<{BN // 64}x16x64xf16>")


# =========================================================================
# Dynamic shapes: M, N runtime -> physical layout reuses the live SSA values
# =========================================================================

class TestDynamicLayout(RewriteLayoutTester):

    def _kernel(self, BM, BN):
        # M, N are runtime block args (no arith.constant), so the physical
        # shape's stick-count dim stays symbolic (divsi over %N).
        # Include a store that uses the load result so DCE doesn't drop it.
        return f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %M: i32, %N: i32, %m: i32, %n: i32) {{
            %sc = arith.constant 1 : i64
            %sr = arith.extsi %N : i32 to i64
            %desc = tt.make_tensor_descriptor %ptr, [%M, %N], [%sr, %sc]
                : <f16>, <{BM}x{BN}xf16>
            tt.spyre_tensor_layout %desc {_STICK_ON_N} : <{BM}x{BN}xf16>
            %d = tt.descriptor_load %desc[%m, %n]
                : !tt.tensordesc<{BM}x{BN}xf16> -> tensor<{BM}x{BN}xf16>
            tt.descriptor_store %desc[%m, %n], %d
                : !tt.tensordesc<{BM}x{BN}xf16>, tensor<{BM}x{BN}xf16>
            tt.return
          }}
        }}
        """

    def test_marker_erased(self):
        self.run(self._kernel(16, 64))
        self.assert_absent("tt.spyre_tensor_layout")

    def test_load_is_physical_rank3(self):
        # Block shape is static even when tensor dims are runtime.
        # BN=64 -> 1 stick per block -> block 1x16x64.
        self.run(self._kernel(16, 64))
        self.assert_result_type("ktdp.load", "tensor<1x16x64xf16>")

    def test_divsi_emitted_for_stick_count(self):
        # The stick-count dim (N // 64) is built from the live %N via arith.divsi
        # — this is the dynamic path that the abandoned live-SSA-value design
        # could not preserve through constant folding.
        self.run(self._kernel(16, 64))
        self.assert_present("arith.divsi")


# =========================================================================
# Negative / pass-through behaviour
# =========================================================================

class TestPassThrough(RewriteLayoutTester):

    def test_unannotated_descriptor_untouched(self):
        # A descriptor with no tt.spyre_tensor_layout marker is left exactly as
        # is — still logical 2D. The pass only fires on annotated descriptors.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %m: i32, %n: i32) {
            %M = arith.constant 512 : i32
            %N = arith.constant 256 : i32
            %sr = arith.constant 256 : i64
            %sc = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%M, %N], [%sr, %sc]
                : <f16>, <16x64xf16>
            %d = tt.descriptor_load %desc[%m, %n]
                : !tt.tensordesc<16x64xf16> -> tensor<16x64xf16>
            tt.descriptor_store %desc[%m, %n], %d
                : !tt.tensordesc<16x64xf16>, tensor<16x64xf16>
            tt.return
          }
        }
        """)
        # Unchanged: no marker → no rewrite. Memory view is logical 512x256;
        # load result is the logical 2D block 16x64.
        self.assert_result_type("ktdp.construct_memory_view", "512x256xf16")
        self.assert_result_type("ktdp.load", "tensor<16x64xf16>")


# =========================================================================
# Known gaps (NOT design limits) — annotated descriptors the pass cannot
# rewrite yet. Each asserts the current failure and documents the fix plan;
# delete in favor of a positive test once supported.
# =========================================================================

class TestGather(RewriteLayoutTester):
    """Annotated descriptor used by ``descriptor_gather``.

    After LowerDescriptorMemory, the gather lowers to
    ktdp.construct_indirect_access_tile which our pass does not physicalize (it
    only handles direct construct_access_tile). The indirect tile is therefore
    left pointing at the old logical memView. The physical memView is built
    correctly but the indirect tile's result is unused (gather result dropped by
    DCE), so the pipeline succeeds — the physical layout annotation is honoured
    for the descriptor's memory view even if the indirect tile itself was not
    re-anchored.

    In practice, gather descriptors do not carry a spyre_tensor_layout marker
    today (gather is the SDSC path). This test documents the current behaviour
    and serves as a regression guard if a future path annotates gather operands.
    """

    @pattern("physical-layout-gather", category="memory", example=[
        "# Annotated gather: the memory view is physicalized; the indirect",
        "# access tile is not re-anchored (its result was dropped by DCE).",
        "desc = tl.make_tensor_descriptor(ptr, shape=[M, K], strides=[K, 1],",
        "                                 block_shape=[1, 64])",
        "tl.spyre_tensor_layout(desc, [(1, 'floordiv', 64), 0, (1, 'mod', 64)])",
        "tile = tl.descriptor_gather(desc, x_offsets, y)   # memory view physicalized",
    ])
    def test_gather_marker_erased(self):
        # The physical memory view is built; the marker is erased; the pipeline
        # succeeds because the indirect tile's result is dead (DCE removes it).
        gather_layout = (
            "{phys_src = array<i64: 1, 0, 1>, "
            "phys_op = array<i64: 1, 0, 2>, "
            "phys_arg = array<i64: 64, 0, 64>}"
        )
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %idx_ptr: !tt.ptr<i32>, %y: i32) {{
            %M = arith.constant 512 : i32
            %K = arith.constant 256 : i32
            %sr = arith.constant 256 : i64
            %sc = arith.constant 1 : i64
            %ic = arith.constant 32 : i32
            %is = arith.constant 1 : i64
            %c0 = arith.constant 0 : i32
            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%ic], [%is]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>
            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%sr, %sc]
                : <f16>, <1x64xf16>
            tt.spyre_tensor_layout %desc {gather_layout} : <1x64xf16>
            %g = tt.descriptor_gather %desc[%x_offsets, %y]
                : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
            tt.return
          }}
        }}
        """)
        self.assert_absent("tt.spyre_tensor_layout")
        # Physical memory view produced for the annotated descriptor.
        self.assert_result_type("ktdp.construct_memory_view", "4x512x64xf16")


class TestMatmulSingleStick(RewriteLayoutTester):
    """Annotated matmul: singleton-stick case (block fits in one stick).

    A[M,K] stick-on-M with block_shape=[64,K]: physical A = [1, K, 64] — one
    stick on the M dim. B[K,N] stick-on-N with block_shape=[K,64]: physical
    B = [1, K, 64] — one stick on the N dim. Both have a leading-1 stick dim
    that can be collapsed before feeding linalg.matmul, then the result stays
    2-D for C.

    Shapes used:
      M=256, K=128, N=256
      block A = [64, 128]  -> phys A = [64//64, 128, 64] = [1, 128, 64]
      block B = [128, 64]  -> phys B = [64//64, 128, 64] = [1, 128, 64]
      C = [64, 64] (logical output tile, unchanged)
    """

    # A[M,K] stick-on-M: phys_src=[0,1,0] => dim0=M//64, dim1=K, dim2=M%64
    _A_LAYOUT = ("{phys_src = array<i64: 0, 1, 0>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # B[K,N] stick-on-N: phys_src=[1,0,1] => dim0=N//64, dim1=K, dim2=N%64
    _B_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")

    _KERNEL = """
        module {{
          tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                      %m: i32, %k: i32, %n: i32) {{
            %M = arith.constant 256 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 256 : i32
            %sK = arith.constant 128 : i64
            %sN = arith.constant 256 : i64
            %sM = arith.constant 256 : i64
            %one = arith.constant 1 : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K], [%sK, %one]
                : <f16>, <64x128xf16>
            %bdesc = tt.make_tensor_descriptor %b, [%K, %N], [%sN, %one]
                : <f16>, <128x64xf16>
            %cdesc = tt.make_tensor_descriptor %c, [%M, %N], [%sM, %one]
                : <f16>, <64x64xf16>
            tt.spyre_tensor_layout %adesc {a_layout} : <64x128xf16>
            tt.spyre_tensor_layout %bdesc {b_layout} : <128x64xf16>
            %at = tt.descriptor_load %adesc[%m, %k]
                : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
            %bt = tt.descriptor_load %bdesc[%k, %n]
                : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
            %acc = arith.constant dense<0.0> : tensor<64x64xf32>
            %d = tt.dot %at, %bt, %acc
                : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
            %dh = arith.truncf %d : tensor<64x64xf32> to tensor<64x64xf16>
            tt.descriptor_store %cdesc[%m, %n], %dh
                : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
            tt.return
          }}
        }}
        """

    @pattern("physical-layout-matmul-single-stick", category="memory", example=[
        "# Both operands have a singleton leading stick dim [1,m,k] x [1,k,n].",
        "# The pass collapses to [m,k] x [k,n], runs linalg.matmul, result is 2-D.",
        "a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M,K], strides=[K,1],",
        "                                   block_shape=[64, K])",
        "tl.spyre_tensor_layout(a_desc, [(0,'floordiv',64), 1, (0,'mod',64)])",
    ])
    def test_matmul_single_stick_lowers(self):
        self.run(self._KERNEL.format(
            a_layout=self._A_LAYOUT, b_layout=self._B_LAYOUT))
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_absent("linalg.generic")
        self.assert_present("linalg.matmul")
        # Physical loads produce rank-3 tensors [1x128x64xf16] — the leading 1
        # is the singleton stick dim.  The loop synthesis extracts + transposes
        # before feeding linalg.matmul so the inner matmul sees [64x128] x [128x64].
        self.assert_result_type("ktdp.load", "1x128x64xf16")


class TestMatmulKSplit(RewriteLayoutTester):
    """Annotated matmul: K-split case (scf.for over K-sticks).

    A[M,K] stick-on-K: physical A = [K//64, M, 64] — the contraction dim K is
    split across sticks. B[K,N] stick-on-N: physical B = [N//64, K, 64] — N
    split, K flat. An scf.for loops over A's K-sticks, offsetting into B's
    K-flat dim by ks * KA each iteration and accumulating into [M, N].

    Shapes used:
      M=256, K=128, N=256
      block A = [64, 64]  -> phys A = [64//64, 64, 64] = [1, 64, 64]
      block B = [64, 64]  -> phys B = [64//64, 128, 64] = [1, 128, 64]
      C = [64, 64] (logical output tile, unchanged)
    """

    # A[M,K] stick-on-K: phys_src=[1,0,1] => dim0=K//64, dim1=M, dim2=K%64
    _A_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # B[K,N] stick-on-N: phys_src=[1,0,1] => dim0=N//64, dim1=K, dim2=N%64
    _B_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")

    _KERNEL = """
        module {{
          tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                      %m: i32, %n: i32) {{
            %M = arith.constant 256 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 256 : i32
            %num_k = arith.constant 2 : i32
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %sK = arith.constant 128 : i64
            %sN = arith.constant 256 : i64
            %sM = arith.constant 256 : i64
            %one = arith.constant 1 : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K], [%sK, %one]
                : <f16>, <64x64xf16>
            %bdesc = tt.make_tensor_descriptor %b, [%K, %N], [%sN, %one]
                : <f16>, <64x64xf16>
            %cdesc = tt.make_tensor_descriptor %c, [%M, %N], [%sM, %one]
                : <f16>, <64x64xf16>
            tt.spyre_tensor_layout %adesc {a_layout} : <64x64xf16>
            tt.spyre_tensor_layout %bdesc {b_layout} : <64x64xf16>
            %acc_init = arith.constant dense<0.0> : tensor<64x64xf32>
            %result = scf.for %k = %c0 to %num_k step %c1
                iter_args(%acc = %acc_init) -> (tensor<64x64xf32>) : i32 {{
              %k64 = arith.muli %k, %c1 : i32
              %at = tt.descriptor_load %adesc[%m, %k64]
                  : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
              %bt = tt.descriptor_load %bdesc[%k64, %n]
                  : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
              %d = tt.dot %at, %bt, %acc
                  : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
              scf.yield %d : tensor<64x64xf32>
            }}
            %dh = arith.truncf %result : tensor<64x64xf32> to tensor<64x64xf16>
            tt.descriptor_store %cdesc[%m, %n], %dh
                : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
            tt.return
          }}
        }}
        """

    @pattern("physical-layout-matmul-k-split", category="memory", example=[
        "# A stick-on-K: scf.for over K-sticks, accumulating [M,N] result.",
        "a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M,K], strides=[K,1],",
        "                                   block_shape=[M, 64])",
        "tl.spyre_tensor_layout(a_desc, [(1,'floordiv',64), 0, (1,'mod',64)])",
        "acc = tl.dot(a_tile, b_tile, acc)   # K-stick loop synthesized by pass",
    ])
    def test_matmul_k_split_lowers(self):
        self.run(self._KERNEL.format(
            a_layout=self._A_LAYOUT, b_layout=self._B_LAYOUT))
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_absent("linalg.generic")
        self.assert_present("linalg.matmul")
        self.assert_present("scf.for")


# =========================================================================
# Sink stage: annotated output (C/store) descriptor
# =========================================================================

class TestMatmulAnnotatedOutput(RewriteLayoutTester):
    """Annotated output (C) descriptor: the store sink stage.

    A[M,K] stick-on-M, B[K,N] stick-on-N (same as TestMatmulSingleStick),
    and now C[M,N] is ALSO annotated stick-on-N:
      phys C = [N//64, M, 64] = [1, 64, 64]  (N=64, M=64, one N-stick).

    The source stage (matmul) leaves C as a LOGICAL tensor<64x64xf32>.
    The sink stage scatters that into the physical [1,64,64] buffer via
    tensor.insert_slice, then redirects the ktdp.store's data_tile to the
    physical result.

    Shapes used (same as TestMatmulSingleStick):
      M=256, K=128, N=256
      block A = [64,128] -> phys A = [1,128,64]
      block B = [128,64] -> phys B = [1,128,64]
      block C = [64,64]  -> phys C = [1,64,64]
    """

    # A[M,K] stick-on-M: phys_src=[0,1,0] => [M//64, K, M%64]
    _A_LAYOUT = ("{phys_src = array<i64: 0, 1, 0>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # B[K,N] stick-on-N: phys_src=[1,0,1] => [N//64, K, N%64]
    _B_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # C[M,N] stick-on-N: phys_src=[1,0,1] => [N//64, M, N%64]
    _C_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")

    _KERNEL = """
        module {{
          tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                      %m: i32, %k: i32, %n: i32) {{
            %M = arith.constant 256 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 256 : i32
            %sK = arith.constant 128 : i64
            %sN = arith.constant 256 : i64
            %sM = arith.constant 256 : i64
            %one = arith.constant 1 : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K], [%sK, %one]
                : <f16>, <64x128xf16>
            %bdesc = tt.make_tensor_descriptor %b, [%K, %N], [%sN, %one]
                : <f16>, <128x64xf16>
            %cdesc = tt.make_tensor_descriptor %c, [%M, %N], [%sM, %one]
                : <f16>, <64x64xf16>
            tt.spyre_tensor_layout %adesc {a_layout} : <64x128xf16>
            tt.spyre_tensor_layout %bdesc {b_layout} : <128x64xf16>
            tt.spyre_tensor_layout %cdesc {c_layout} : <64x64xf16>
            %at = tt.descriptor_load %adesc[%m, %k]
                : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
            %bt = tt.descriptor_load %bdesc[%k, %n]
                : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
            %acc = arith.constant dense<0.0> : tensor<64x64xf32>
            %d = tt.dot %at, %bt, %acc
                : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
            %dh = arith.truncf %d : tensor<64x64xf32> to tensor<64x64xf16>
            tt.descriptor_store %cdesc[%m, %n], %dh
                : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
            tt.return
          }}
        }}
        """

    @pattern("physical-layout-store-annotated-output", category="memory", example=[
        "# Annotated C/output: the sink stage scatters logical C into a",
        "# physical [N//64, M, lane] buffer via tensor.insert_slice.",
        "c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M,N], strides=[N,1],",
        "                                   block_shape=[BLOCK_M, BLOCK_N])",
        "tl.spyre_tensor_layout(c_desc, [(1,'floordiv',64), 0, (1,'mod',64)])",
        "# The store data_tile is replaced by a physical [N//64,M,64] tensor.",
    ])
    def test_annotated_output_lowers(self):
        self.run(self._KERNEL.format(
            a_layout=self._A_LAYOUT, b_layout=self._B_LAYOUT,
            c_layout=self._C_LAYOUT))
        # Markers are erased.
        self.assert_absent("tt.spyre_tensor_layout")
        # The sink stage synthesizes a tensor.insert_slice to scatter logical C
        # into the physical output buffer.
        self.assert_present("tensor.insert_slice")
        # Physical output store access tile is rank-3: [1,64,64].
        self.assert_result_type("ktdp.construct_access_tile", "access_tile<1x64x64xindex>")

    def test_annotated_output_no_insert_slice_when_unannotated(self):
        """Unannotated C: the fallback path is untouched, no insert_slice."""
        # Reuse TestMatmulSingleStick's kernel (A and B annotated, C not).
        kernel = """
        module {{
          tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                      %m: i32, %k: i32, %n: i32) {{
            %M = arith.constant 256 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 256 : i32
            %sK = arith.constant 128 : i64
            %sN = arith.constant 256 : i64
            %sM = arith.constant 256 : i64
            %one = arith.constant 1 : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K], [%sK, %one]
                : <f16>, <64x128xf16>
            %bdesc = tt.make_tensor_descriptor %b, [%K, %N], [%sN, %one]
                : <f16>, <128x64xf16>
            %cdesc = tt.make_tensor_descriptor %c, [%M, %N], [%sM, %one]
                : <f16>, <64x64xf16>
            tt.spyre_tensor_layout %adesc {a_layout} : <64x128xf16>
            tt.spyre_tensor_layout %bdesc {b_layout} : <128x64xf16>
            %at = tt.descriptor_load %adesc[%m, %k]
                : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
            %bt = tt.descriptor_load %bdesc[%k, %n]
                : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
            %acc = arith.constant dense<0.0> : tensor<64x64xf32>
            %d = tt.dot %at, %bt, %acc
                : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
            %dh = arith.truncf %d : tensor<64x64xf32> to tensor<64x64xf16>
            tt.descriptor_store %cdesc[%m, %n], %dh
                : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
            tt.return
          }}
        }}
        """.format(a_layout=self._A_LAYOUT, b_layout=self._B_LAYOUT)
        self.run(kernel)
        self.assert_absent("tt.spyre_tensor_layout")
        # Unannotated C → no sink stage → no insert_slice.
        self.assert_absent("tensor.insert_slice")
        # Logical C store: access tile shape matches the logical block [64x64].
        self.assert_result_type("ktdp.construct_access_tile", "access_tile<64x64xindex>")


# =========================================================================
# Negative (frontend): the layout list must be passed inline, not via a local
# =========================================================================

class TestInlineOnly:
    """The `layout` arg must be an inline literal at the call site.

    Binding it to a local first makes the @triton.jit code generator try to
    tensor-convert the keyword strings (a known rough edge). This is a frontend
    (TTIR codegen) failure, not a pass failure, so it uses compile_to_ttir
    directly rather than SinglePassTester.
    """

    SIGNATURE = {"x_ptr": "*fp16", "M": "i32", "N": "i32",
                 "BLOCK_M": "i32", "BLOCK_N": "i32"}

    @pattern("physical-layout-rewrite", category="memory", negative=True, example=[
        "# NOT supported: binding the layout list to a local first.",
        "# The keyword strings get tensor-converted by the jit code generator.",
        "lay = [(1, 'floordiv', 64), 0, (1, 'mod', 64)]",
        "tl.spyre_tensor_layout(desc, lay)        # ❌ raises CompilationError",
        "# Pass it inline instead:",
        "tl.spyre_tensor_layout(desc, [(1, 'floordiv', 64), 0, (1, 'mod', 64)])  # ✅",
    ])
    def test_layout_via_local_fails(self):
        import triton
        import triton.language as tl
        from utils import compile_to_ttir

        @triton.jit
        def kernel(x_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
            d = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1],
                                          block_shape=[BLOCK_M, BLOCK_N])
            lay = [(1, "floordiv", 64), 0, (1, "mod", 64)]  # bound to a local
            tl.spyre_tensor_layout(d, lay)
            v = d.load([0, 0])
            d.store([0, 0], v)

        with pytest.raises(Exception, match="(?i)tensor"):
            compile_to_ttir(kernel, self.SIGNATURE,
                            {"M": 512, "N": 256, "BLOCK_M": 16, "BLOCK_N": 64})

    def test_layout_inline_ok(self):
        # The same kernel with the list inline compiles fine (control for the
        # negative above).
        import triton
        import triton.language as tl
        from utils import compile_to_ttir

        @triton.jit
        def kernel(x_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
            d = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1],
                                          block_shape=[BLOCK_M, BLOCK_N])
            tl.spyre_tensor_layout(d, [(1, "floordiv", 64), 0, (1, "mod", 64)])
            v = d.load([0, 0])
            d.store([0, 0], v)

        txt = compile_to_ttir(kernel, self.SIGNATURE,
                              {"M": 512, "N": 256, "BLOCK_M": 16, "BLOCK_N": 64})
        assert "tt.spyre_tensor_layout" in txt
