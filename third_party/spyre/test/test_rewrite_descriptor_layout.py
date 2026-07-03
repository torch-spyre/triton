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

# Marker attr for the [M,N] stick-on-N layout used by the reduce test.
# Same encoding as _STICK_ON_N (S=64): phys = [N//64, M, N%64].
_STICK_ON_N_REDUCE = (
    "{phys_src = array<i64: 1, 0, 1>, "
    "phys_op = array<i64: 1, 0, 2>, "
    "phys_arg = array<i64: 64, 0, 64>}"
)

# Marker attr for a 1-D [M] stick layout used by the reduce output descriptor.
# phys = [M//64, M%64] (stick-on-M, S=64).
_STICK_ON_M_1D = (
    "{phys_src = array<i64: 0, 0>, "
    "phys_op = array<i64: 1, 2>, "
    "phys_arg = array<i64: 64, 64>}"
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

    def test_non_aligned_column_offset(self):
        # T014/T4/Q5/C3: gather where the column base (y_offset) is a
        # non-multiple of S=64 (constant 50).  ColMap substitution inserts
        # floordiv/mod arithmetic; the pass must still complete successfully
        # and erase the marker.
        gather_layout = (
            "{phys_src = array<i64: 1, 0, 1>, "
            "phys_op = array<i64: 1, 0, 2>, "
            "phys_arg = array<i64: 64, 0, 64>}"
        )
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %idx_ptr: !tt.ptr<i32>) {{
            %M = arith.constant 512 : i32
            %K = arith.constant 256 : i32
            %sr = arith.constant 256 : i64
            %sc = arith.constant 1 : i64
            %ic = arith.constant 32 : i32
            %is = arith.constant 1 : i64
            %c0 = arith.constant 0 : i32
            %c_y = arith.constant 50 : i32
            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%ic], [%is]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>
            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%sr, %sc]
                : <f16>, <1x64xf16>
            tt.spyre_tensor_layout %desc {gather_layout} : <1x64xf16>
            %g = tt.descriptor_gather %desc[%x_offsets, %c_y]
                : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
            tt.return
          }}
        }}
        """)
        # Q1: marker is erased.
        self.assert_absent("tt.spyre_tensor_layout")
        # The pass completes and rebuilds the physical memory view.
        self.assert_result_type("ktdp.construct_memory_view", "4x512x64xf16")
        # Note: DCE drops the indirect access tile (and its ColMap arith) because
        # the gather result is unused after lowering.  The meaningful assertion is
        # structural correctness — the pass runs without error and the physical
        # memview is present.  (arith.divsi is NOT asserted here because it is
        # part of the dead indirect-tile subgraph that DCE removes.)


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
            %c64 = arith.constant 64 : i32
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
              %k64 = arith.muli %k, %c64 : i32
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

    def test_matmul_f32(self):
        # T015/T9b: K-split matmul with f32 element type — the pass must be
        # element-type-agnostic; physical memory view and load use f32.
        # We build a fresh f32 variant of _KERNEL by substituting the dtype.
        kernel_f32 = """
        module {{
          tt.func @mm(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>,
                      %m: i32, %n: i32) {{
            %M = arith.constant 256 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 256 : i32
            %num_k = arith.constant 2 : i32
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %c64 = arith.constant 64 : i32
            %sK = arith.constant 128 : i64
            %sN = arith.constant 256 : i64
            %sM = arith.constant 256 : i64
            %one = arith.constant 1 : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K], [%sK, %one]
                : <f32>, <64x64xf32>
            %bdesc = tt.make_tensor_descriptor %b, [%K, %N], [%sN, %one]
                : <f32>, <64x64xf32>
            %cdesc = tt.make_tensor_descriptor %c, [%M, %N], [%sM, %one]
                : <f32>, <64x64xf32>
            tt.spyre_tensor_layout %adesc {a_layout} : <64x64xf32>
            tt.spyre_tensor_layout %bdesc {b_layout} : <64x64xf32>
            %acc_init = arith.constant dense<0.0> : tensor<64x64xf32>
            %result = scf.for %k = %c0 to %num_k step %c1
                iter_args(%acc = %acc_init) -> (tensor<64x64xf32>) : i32 {{
              %k64 = arith.muli %k, %c64 : i32
              %at = tt.descriptor_load %adesc[%m, %k64]
                  : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>
              %bt = tt.descriptor_load %bdesc[%k64, %n]
                  : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32>
              %d = tt.dot %at, %bt, %acc
                  : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
              scf.yield %d : tensor<64x64xf32>
            }}
            tt.descriptor_store %cdesc[%m, %n], %result
                : !tt.tensordesc<64x64xf32>, tensor<64x64xf32>
            tt.return
          }}
        }}
        """.format(a_layout=self._A_LAYOUT, b_layout=self._B_LAYOUT)
        self.run(kernel_f32)
        # T9b/P6: marker erased.
        self.assert_absent("tt.spyre_tensor_layout")
        # Physical memory view result type uses f32.
        self.assert_result_type("ktdp.construct_memory_view", "f32")
        # ktdp.load result type uses f32.
        self.assert_result_type("ktdp.load", "f32")


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
# Chained matmul: D = A @ (B @ C) — scratchpad intermediate (Step 8b)
# =========================================================================

class TestMatmulChainedScratchpad(RewriteLayoutTester):
    """D = A @ (B @ C): all three input descriptors are annotated (physical).

    The inner tt.dot(B_tile, C_tile) produces a logical [BLOCK_K1, BLOCK_N]
    scratchpad tile bc.  The outer tt.dot(A_tile, bc, acc) has one physical
    operand (A) and one scratchpad operand (bc) that has no descriptor and
    no marker.  dispatchSource must recognise bc as a scratchpad (walkToLoad
    returns null → not an error) and pass it through whole, while still
    slicing + transposing A_tile normally.

    Shapes:
      A[M,K1] stick-on-M:  block [64,64]  -> phys [1,64,64]
      B[K1,K2] stick-on-K2: block [64,32] -> phys [1,64,32]  (K2//32=1 stick)
      C[K2,N]  stick-on-N:  block [32,64] -> phys [1,32,64]  (N//64=1 stick)
      bc scratchpad: logical [64,64] (f16, result of inner dot)
      D[M,N]   stick-on-N:  block [64,64] -> phys [1,64,64]
    """

    # A[M,K1] stick-on-M: phys_src=[0,1,0] => [M//64, K1, M%64]
    _A_LAYOUT = ("{phys_src = array<i64: 0, 1, 0>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # B[K1,K2] stick-on-K2: phys_src=[1,0,1] => [K2//32, K1, K2%32]
    _B_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 32, 0, 32>}")
    # C[K2,N] stick-on-N: phys_src=[1,0,1] => [N//64, K2, N%64]
    _C_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # D[M,N] stick-on-N: phys_src=[1,0,1] => [N//64, M, N%64]
    _D_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")

    _KERNEL = """
        module {{
          tt.func @chained(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>,
                           %c: !tt.ptr<f16>, %d: !tt.ptr<f16>,
                           %m: i32, %n: i32, %k1: i32) {{
            %M  = arith.constant 256 : i32
            %K1 = arith.constant 64  : i32
            %K2 = arith.constant 32  : i32
            %N  = arith.constant 256 : i32
            %sK1 = arith.constant 64  : i64
            %sK2 = arith.constant 32  : i64
            %sN  = arith.constant 256 : i64
            %sM  = arith.constant 256 : i64
            %one = arith.constant 1   : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K1], [%sK1, %one]
                : <f16>, <64x64xf16>
            %bdesc = tt.make_tensor_descriptor %b, [%K1, %K2], [%sK2, %one]
                : <f16>, <64x32xf16>
            %cdesc = tt.make_tensor_descriptor %c, [%K2, %N], [%sN, %one]
                : <f16>, <32x64xf16>
            %ddesc = tt.make_tensor_descriptor %d, [%M, %N], [%sN, %one]
                : <f16>, <64x64xf16>
            tt.spyre_tensor_layout %adesc {a_layout} : <64x64xf16>
            tt.spyre_tensor_layout %bdesc {b_layout} : <64x32xf16>
            tt.spyre_tensor_layout %cdesc {c_layout} : <32x64xf16>
            tt.spyre_tensor_layout %ddesc {d_layout} : <64x64xf16>
            %bt = tt.descriptor_load %bdesc[%k1, %n]
                : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
            %ct = tt.descriptor_load %cdesc[%n, %n]
                : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
            %bc_init = arith.constant dense<0.0> : tensor<64x64xf16>
            %bc = tt.dot %bt, %ct, %bc_init
                : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf16>
            %at = tt.descriptor_load %adesc[%m, %k1]
                : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
            %acc_init = arith.constant dense<0.0> : tensor<64x64xf16>
            %result = tt.dot %at, %bc, %acc_init
                : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf16>
            tt.descriptor_store %ddesc[%m, %n], %result
                : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
            tt.return
          }}
        }}
        """

    def test_chained_scratchpad_lowers(self):
        self.run(self._KERNEL.format(
            a_layout=self._A_LAYOUT, b_layout=self._B_LAYOUT,
            c_layout=self._C_LAYOUT, d_layout=self._D_LAYOUT))
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_absent("tt.dot")
        self.assert_present("linalg.matmul")
        # The store sink stage scatters the logical [64,64] output into
        # the physical [1,64,64] D buffer.
        self.assert_present("tensor.insert_slice")

    def test_chained_scratchpad_a_sliced(self):
        # A is physical → ktdp.load produces rank-3 [1,64,64]; the source
        # stage extracts a 2D slice before feeding linalg.matmul.
        self.run(self._KERNEL.format(
            a_layout=self._A_LAYOUT, b_layout=self._B_LAYOUT,
            c_layout=self._C_LAYOUT, d_layout=self._D_LAYOUT))
        # Physical loads for A and B/C all produce rank-3 results.
        self.assert_result_type("ktdp.load", "1x64x64xf16")


# =========================================================================
# Reduce: 2-D stick-on-N input + 1-D output stick layout (spec §6b T5)
# =========================================================================

class TestReduceStick(RewriteLayoutTester):
    """Row-sum reduce with annotated 2-D input and 1-D output descriptor.

    Input  in_ptr:  [M=64, N=256] stick-on-N -> phys [N//64, M, 64] = [4, 64, 64]
    Output out_ptr: [M=64]        stick       -> phys [M//64, 64]   = [1, 64]

    The input has N=256 N-sticks, so f=4 sticks per block (N/S = 256/64 = 4).
    The pass synthesizes an scf.for over the 4 input sticks and accumulates
    the partial sums via linalg.reduce.  The output is scattered into the
    physical [1, 64] buffer by the rank-1 sink path (C5/Q7).
    """

    def _kernel(self, dtype="f16"):
        # 2-D input [M=64, N=256] stick-on-N; 1-D output [M=64] stick-on-M.
        # The descriptor block_shape for the input spans the whole N axis so
        # the reduce fires a single load per row; the sink stage writes the
        # [M] result back via the 1-D output descriptor.
        return f"""
        module {{
          tt.func @k(%in_ptr: !tt.ptr<{dtype}>, %out_ptr: !tt.ptr<{dtype}>) {{
            %M  = arith.constant 64  : i32
            %N  = arith.constant 256 : i32
            %sN = arith.constant 256 : i64
            %s1 = arith.constant 1   : i64
            %c0 = arith.constant 0   : i32
            %in_desc = tt.make_tensor_descriptor %in_ptr, [%M, %N], [%sN, %s1]
                : <{dtype}>, <64x256x{dtype}>
            tt.spyre_tensor_layout %in_desc {_STICK_ON_N_REDUCE} : <64x256x{dtype}>
            %out_desc = tt.make_tensor_descriptor %out_ptr, [%M], [%s1]
                : <{dtype}>, <64x{dtype}>
            tt.spyre_tensor_layout %out_desc {_STICK_ON_M_1D} : <64x{dtype}>
            %tile = tt.descriptor_load %in_desc[%c0, %c0]
                : !tt.tensordesc<64x256x{dtype}> -> tensor<64x256x{dtype}>
            %sum = "tt.reduce"(%tile) ({{
            ^bb0(%a: {dtype}, %b: {dtype}):
              %add = arith.addf %a, %b : {dtype}
              tt.reduce.return %add : {dtype}
            }}) {{axis = 1 : i32}} : (tensor<64x256x{dtype}>) -> tensor<64x{dtype}>
            tt.descriptor_store %out_desc[%c0], %sum
                : !tt.tensordesc<64x{dtype}>, tensor<64x{dtype}>
            tt.return
          }}
        }}
        """

    def test_marker_erased(self):
        # Q1: the pass consumes and erases every tt.spyre_tensor_layout marker.
        self.run(self._kernel())
        self.assert_absent("tt.spyre_tensor_layout")

    def test_reduce_stick_loop_emitted(self):
        # E2/Q6: N is stickified (f=4 sticks), so the pass synthesizes an
        # scf.for over the input sticks to accumulate the partial sums.
        self.run(self._kernel())
        self.assert_present("scf.for")

    @pattern("physical-layout-reduce-stick", category="memory", example=[
        "in_desc  = tl.make_tensor_descriptor(in_ptr,  shape=[M, N], strides=[N, 1],",
        "                                     block_shape=[BLOCK_M, N])",
        "out_desc = tl.make_tensor_descriptor(out_ptr, shape=[M],    strides=[1],",
        "                                     block_shape=[BLOCK_M])",
        "tl.spyre_tensor_layout(in_desc,  [(1,'floordiv',64), 0, (1,'mod',64)])",
        "tl.spyre_tensor_layout(out_desc, [(0,'floordiv',64), (0,'mod',64)])",
        "# sink stage: linalg.reduce result scattered into [M//S, S] via insert_slice",
    ])
    def test_rank1_sink_insert_slice(self):
        # C5/Q7: the sink stage for a 1-D output descriptor produces a
        # rank-2 physical container [M/S, S] = [1, 64] and scatters the
        # logical [M] result via tensor.insert_slice (not a hardcoded rank-3
        # scatter as the matmul path uses).
        self.run(self._kernel())
        self.assert_present("tensor.insert_slice")
        self.assert_present("ktdp.store")
        # Physical access tile for the output store is rank-2: [1, 64].
        self.assert_result_type("ktdp.construct_access_tile",
                                "access_tile<1x64xindex>")

    def test_reduce_f32(self):
        # T9b partial: the reduce path must be element-type-agnostic; verify
        # f32 produces the same structural output (scf.for + insert_slice).
        self.run(self._kernel("f32"))
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_present("scf.for")
        self.assert_present("tensor.insert_slice")


# =========================================================================
# Loop rescale: f>1 stick factor causes loop bound/step scaling (spec §6b T6)
# =========================================================================

class TestLoopRescale(RewriteLayoutTester):
    """Explicit f>1 loop rescaling: block spans multiple sticks on K.

    A[M,K] stick-on-K with S=64.  When the block spans BLOCK_K=128 elements
    (f = BLOCK_K/S = 2 sticks), the outer scf.for driving the K dimension must
    be rescaled: its upper bound is multiplied by f and its step is set to f,
    so the per-iteration IV increment equals one stick.  The muli(iv, C)
    multiplier inside the loop is correspondingly reduced from C=BLOCK_K to S.

    The complementary f=1 test (BLOCK_K=64) verifies the no-op path: the loop
    bound and step are left unchanged, and the muli multiplier stays at 64.
    """

    # A[M,K] stick-on-K: phys_src=[1,0,1] => dim0=K//64, dim1=M, dim2=K%64
    _A_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # B[K,N] stick-on-N: phys_src=[1,0,1] => dim0=N//64, dim1=K, dim2=N%64
    _B_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")

    def _kernel_f2(self):
        # A block spans 2 sticks (BLOCK_K=128, S=64, f=2).
        # Loop: 0 to num_k=2 step 1; muli(iv, 128) is the K offset.
        # After rescaling: bound = 2*2 = 4, step = 2; muli multiplier = 64.
        return f"""
        module {{
          tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                      %m: i32, %n: i32) {{
            %M = arith.constant 64 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 64 : i32
            %num_k = arith.constant 2 : i32
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %c128 = arith.constant 128 : i32
            %sK = arith.constant 128 : i64
            %sN = arith.constant 64 : i64
            %sM = arith.constant 64 : i64
            %one = arith.constant 1 : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K], [%sK, %one]
                : <f16>, <64x128xf16>
            %bdesc = tt.make_tensor_descriptor %b, [%K, %N], [%sN, %one]
                : <f16>, <128x64xf16>
            %cdesc = tt.make_tensor_descriptor %c, [%M, %N], [%sM, %one]
                : <f16>, <64x64xf16>
            tt.spyre_tensor_layout %adesc {self._A_LAYOUT} : <64x128xf16>
            tt.spyre_tensor_layout %bdesc {self._B_LAYOUT} : <128x64xf16>
            %acc_init = arith.constant dense<0.0> : tensor<64x64xf32>
            %result = scf.for %k = %c0 to %num_k step %c1
                iter_args(%acc = %acc_init) -> (tensor<64x64xf32>) : i32 {{
              %k128 = arith.muli %k, %c128 : i32
              %at = tt.descriptor_load %adesc[%m, %k128]
                  : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
              %bt = tt.descriptor_load %bdesc[%k128, %n]
                  : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
              %d = tt.dot %at, %bt, %acc
                  : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
              scf.yield %d : tensor<64x64xf32>
            }}
            %dh = arith.truncf %result : tensor<64x64xf32> to tensor<64x64xf16>
            tt.descriptor_store %cdesc[%m, %n], %dh
                : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
            tt.return
          }}
        }}
        """

    def _kernel_f1(self):
        # A block spans 1 stick (BLOCK_K=64, S=64, f=1) — same as TestMatmulKSplit.
        # Loop: 0 to num_k=2 step 1; muli(iv, 64) is the K offset.
        # After the pass: bound=2, step=1 unchanged; muli multiplier stays 64.
        return f"""
        module {{
          tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                      %m: i32, %n: i32) {{
            %M = arith.constant 256 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 256 : i32
            %num_k = arith.constant 2 : i32
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %c64 = arith.constant 64 : i32
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
            tt.spyre_tensor_layout %adesc {self._A_LAYOUT} : <64x64xf16>
            tt.spyre_tensor_layout %bdesc {self._B_LAYOUT} : <64x64xf16>
            %acc_init = arith.constant dense<0.0> : tensor<64x64xf32>
            %result = scf.for %k = %c0 to %num_k step %c1
                iter_args(%acc = %acc_init) -> (tensor<64x64xf32>) : i32 {{
              %k64 = arith.muli %k, %c64 : i32
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

    @pattern("physical-layout-loop-rescale", category="memory", example=[
        "# f>1: BLOCK_K=128, S=64 -> f=2. The outer loop iterates over K-blocks.",
        "# The pass rescales: bound *= f, step = f, muli multiplier reduced to S.",
        "a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M,K], strides=[K,1],",
        "                                   block_shape=[64, 128])  # BLOCK_K=128",
        "tl.spyre_tensor_layout(a_desc, [(1,'floordiv',64), 0, (1,'mod',64)])",
        "# scf.for %iv = 0 to (num_k * 2) step 2: muli(%iv, 64) -> K offset",
    ])
    def test_block_wider_than_one_stick(self):
        # f=2: BLOCK_K=128, S=64. Original loop: 0 to 2 step 1, muli(iv, 128).
        # After rescaling: bound = 4, step = 2; muli multiplier = 64 (= S).
        # (T6, Q4, C2)
        self.run(self._kernel_f2())
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_present("scf.for")
        # muli multiplier reduced to S=64 (not 128)
        self.assert_operand("arith.muli", 1, value=64)
        # loop step = f = 2 (not 1)
        self.assert_operand("scf.for", 2, value=2)

    def test_f1_loop_unchanged(self):
        # f=1: BLOCK_K=64, S=64. Loop: 0 to 2 step 1. No rescaling.
        # Bound stays 2, step stays 1, muli multiplier stays 64.
        # (T6 no-op path, §6a)
        self.run(self._kernel_f1())
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_present("scf.for")
        # muli multiplier stays 64 (= S = BLOCK_K); the f1 no-op path leaves it.
        self.assert_operand("arith.muli", 1, value=64)
        # loop step stays 1 (no rescaling)
        self.assert_operand("scf.for", 2, value=1)


# =========================================================================
# Double rescale guard: two descriptors sharing one scf.for (spec §6b T7)
# =========================================================================

class TestDoubleRescaleGuard(RewriteLayoutTester):
    """Two annotated descriptors sharing one enclosing scf.for.

    Both A[M,K] stick-on-K and B[K,N] stick-on-K have f=2, and both feed
    from the same `muli(iv, 128)` K offset inside a single scf.for.  The
    pass must rescale the shared loop exactly once — not twice (which would
    produce bound = f^2 * original = 8, step = f^2 = 4).

    Expected result: loop bound = 4 (= 2 * f), step = 2 (= f).
    """

    # A[M,K] stick-on-K: phys_src=[1,0,1] => dim0=K//64, dim1=M, dim2=K%64
    _A_LAYOUT = ("{phys_src = array<i64: 1, 0, 1>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
    # B[K,N] stick-on-K: phys_src=[0,1,0] => dim0=K//64, dim1=N, dim2=K%64
    _B_LAYOUT = ("{phys_src = array<i64: 0, 1, 0>, "
                 "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")

    def _kernel(self):
        # Two descriptors, both stick-on-K with f=2 (BLOCK_K=128, S=64).
        # A and B each use the same muli(iv, 128) K offset from one scf.for.
        return f"""
        module {{
          tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                      %m: i32, %n: i32) {{
            %M = arith.constant 64 : i32
            %K = arith.constant 128 : i32
            %N = arith.constant 64 : i32
            %num_k = arith.constant 2 : i32
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %c128 = arith.constant 128 : i32
            %sK = arith.constant 128 : i64
            %sM = arith.constant 64 : i64
            %one = arith.constant 1 : i64
            %adesc = tt.make_tensor_descriptor %a, [%M, %K], [%sK, %one]
                : <f16>, <64x128xf16>
            %bdesc = tt.make_tensor_descriptor %b, [%K, %N], [%sK, %one]
                : <f16>, <128x64xf16>
            %cdesc = tt.make_tensor_descriptor %c, [%M, %N], [%sM, %one]
                : <f16>, <64x64xf16>
            tt.spyre_tensor_layout %adesc {self._A_LAYOUT} : <64x128xf16>
            tt.spyre_tensor_layout %bdesc {self._B_LAYOUT} : <128x64xf16>
            %acc_init = arith.constant dense<0.0> : tensor<64x64xf32>
            %result = scf.for %k = %c0 to %num_k step %c1
                iter_args(%acc = %acc_init) -> (tensor<64x64xf32>) : i32 {{
              %k128 = arith.muli %k, %c128 : i32
              %at = tt.descriptor_load %adesc[%m, %k128]
                  : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
              %bt = tt.descriptor_load %bdesc[%k128, %n]
                  : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
              %d = tt.dot %at, %bt, %acc
                  : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
              scf.yield %d : tensor<64x64xf32>
            }}
            %dh = arith.truncf %result : tensor<64x64xf32> to tensor<64x64xf16>
            tt.descriptor_store %cdesc[%m, %n], %dh
                : !tt.tensordesc<64x64xf16>, tensor<64x64xf16>
            tt.return
          }}
        }}
        """

    @pattern("physical-layout-double-rescale-guard", category="memory", example=[
        "# Two annotated descriptors (A, B), both stick-on-K with f=2,",
        "# both reading from the same muli(iv, 128) K offset inside one scf.for.",
        "# The pass rescales the shared loop exactly once: bound = 4, step = 2.",
        "# (Not twice: bound != 8, step != 4.)",
        "tl.spyre_tensor_layout(a_desc, [(1,'floordiv',64), 0, (1,'mod',64)])",
        "tl.spyre_tensor_layout(b_desc, [(0,'floordiv',64), 1, (0,'mod',64)])",
    ])
    def test_two_descriptors_one_loop(self):
        # Two descriptors, both stick-on-K with f=2.  The rescaledLoops guard
        # in RewriteDescriptorLayout.cpp must prevent double-rescaling.
        # Correct: bound = f * original = 2 * 2 = 4, step = f = 2.
        # Wrong (double): bound = f^2 * original = 8, step = f^2 = 4. (T7, Q4, C2)
        self.run(self._kernel())
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_present("scf.for")
        # Step should be f=2, not f^2=4.
        self.assert_operand("scf.for", 2, value=2)
        # Verify the loop was NOT double-scaled: constant 8 must be absent.
        # (f^2 * original_ub = 4 * 2 = 8 would appear if the loop was rescaled twice.)
        from utils import walk_module
        ops = walk_module(self.mod)
        const_values = []
        for o in ops:
            if o.name == "arith.constant":
                v = o._op.get_constant_value()
                if v is not None:
                    const_values.append(v)
        assert 8 not in const_values, (
            f"Constant 8 found — loop was rescaled twice (f^2 bound). "
            f"All integer constants: {const_values}"
        )


# =========================================================================
# T016: Rejected inputs — negative tests for precondition violations
# =========================================================================

class TestRejectedInputs(RewriteLayoutTester):
    """Negative tests (T10/R1–R3): verify the pass rejects invalid inputs.

    Each test constructs IR that violates a RewriteDescriptorLayout precondition
    and asserts the pass signals failure via an exception.
    """

    def test_r2_unknown_layout_op(self):
        # R2: phys_op contains value 3 (not a valid op: 0=identity, 1=floordiv,
        # 2=mod). The pass should fail/raise when it encounters the unknown op
        # code during physicalization.
        bad_layout = (
            "{phys_src = array<i64: 1, 0, 1>, "
            "phys_op = array<i64: 3, 0, 2>, "   # 3 is invalid
            "phys_arg = array<i64: 64, 0, 64>}"
        )
        with pytest.raises(Exception):
            self.run(f"""
            module {{
              tt.func @k(%ptr: !tt.ptr<f16>, %m: i32, %n: i32) {{
                %M = arith.constant 512 : i32
                %N = arith.constant 256 : i32
                %sr = arith.constant 256 : i64
                %sc = arith.constant 1 : i64
                %desc = tt.make_tensor_descriptor %ptr, [%M, %N], [%sr, %sc]
                    : <f16>, <16x64xf16>
                tt.spyre_tensor_layout %desc {bad_layout} : <16x64xf16>
                %d = tt.descriptor_load %desc[%m, %n]
                    : !tt.tensordesc<16x64xf16> -> tensor<16x64xf16>
                tt.descriptor_store %desc[%m, %n], %d
                    : !tt.tensordesc<16x64xf16>, tensor<16x64xf16>
                tt.return
              }}
            }}
            """)

    def test_r3_non_muli_floor_coord(self):
        # R3: A[M,K] stick-on-K inside an scf.for, but the K-offset is computed
        # as addi(iv, C) instead of muli(iv, C). The pass expects muli to rescale
        # the loop induction variable; it should fail when it can't match the
        # pattern.
        # A[M,K] stick-on-K: phys_src=[1,0,1] => dim0=K//64, dim1=M, dim2=K%64
        a_layout = ("{phys_src = array<i64: 1, 0, 1>, "
                    "phys_op = array<i64: 1, 0, 2>, phys_arg = array<i64: 64, 0, 64>}")
        with pytest.raises(Exception):
            self.run(f"""
            module {{
              tt.func @mm(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %c: !tt.ptr<f16>,
                          %m: i32, %n: i32) {{
                %M = arith.constant 256 : i32
                %K = arith.constant 128 : i32
                %N = arith.constant 256 : i32
                %num_k = arith.constant 2 : i32
                %c0 = arith.constant 0 : i32
                %c1 = arith.constant 1 : i32
                %c64 = arith.constant 64 : i32
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
                %acc_init = arith.constant dense<0.0> : tensor<64x64xf32>
                %result = scf.for %k = %c0 to %num_k step %c1
                    iter_args(%acc = %acc_init) -> (tensor<64x64xf32>) : i32 {{
                  %k_off = arith.addi %k, %c64 : i32
                  %at = tt.descriptor_load %adesc[%m, %k_off]
                      : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
                  %bt = tt.descriptor_load %bdesc[%k_off, %n]
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
            """)

    # TODO R4 (test_r4_physical_operand_at_fixpoint): constructing a matmul
    # where an operand is physically typed but can't be reduced to canonical
    # form is difficult to express in inline MLIR without full pipeline context.
    # Skip for now; add once the pass exposes a cleaner rejection hook.


# =========================================================================
# T017: Higher-rank inputs — rank-3 logical descriptor
# =========================================================================

class TestHigherRank(RewriteLayoutTester):
    """T11: Kernel with a rank-3 [B=2, M=64, N=128] input descriptor.

    The descriptor is annotated stick-on-N with S=64.  Physical shape:
      phys = [B, N//64, M, N%64] = [2, 2, 64, 64]  (rank 4)

    phys_src = [0, 2, 1, 2]   # B <- dim0 ; N//64 <- dim2 ; M <- dim1 ; N%64 <- dim2
    phys_op  = [0, 1, 0, 2]   # identity, floordiv, identity, mod
    phys_arg = [0, 64, 0, 64]
    """

    # Rank-3 [B, M, N] stick-on-N (last dim N split by S=64):
    #   phys_src = [0, 2, 1, 2] : B, N//64, M, N%64
    #   phys_op  = [0, 1, 0, 2] : identity, floordiv, identity, mod
    #   phys_arg = [0, 64, 0, 64]
    _LAYOUT_3D = (
        "{phys_src = array<i64: 0, 2, 1, 2>, "
        "phys_op = array<i64: 0, 1, 0, 2>, "
        "phys_arg = array<i64: 0, 64, 0, 64>}"
    )

    def _kernel(self):
        # [B=2, M=64, N=128] stick-on-N: phys = [2, 2, 64, 64]
        # block_shape = [2, 16, 64] (load the whole tile in one shot for B=2)
        return f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %b: i32, %m: i32, %n: i32) {{
            %B  = arith.constant 2   : i32
            %M  = arith.constant 64  : i32
            %N  = arith.constant 128 : i32
            %sB = arith.constant 8192 : i64
            %sM = arith.constant 128  : i64
            %sN = arith.constant 1    : i64
            %c0 = arith.constant 0    : i32
            %desc = tt.make_tensor_descriptor %ptr, [%B, %M, %N], [%sB, %sM, %sN]
                : <f16>, <2x16x64xf16>
            tt.spyre_tensor_layout %desc {self._LAYOUT_3D} : <2x16x64xf16>
            %d = tt.descriptor_load %desc[%c0, %m, %n]
                : !tt.tensordesc<2x16x64xf16> -> tensor<2x16x64xf16>
            tt.descriptor_store %desc[%c0, %m, %n], %d
                : !tt.tensordesc<2x16x64xf16>, tensor<2x16x64xf16>
            tt.return
          }}
        }}
        """

    def test_rank3_input_physical_shape(self):
        # T11/C1/Q2: the physical memory view covers the full [B, N//64, M, 64]
        # tensor = [2, 2, 64, 64] for B=2, M=64, N=128, S=64.
        self.run(self._kernel())
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_result_type("ktdp.construct_memory_view", "2x2x64x64xf16")

    def test_rank3_load_retyped(self):
        # T11/Q2: the ktdp.load result is the physical block type
        # [B=2, N//64_per_block=1, BM=16, lane=64] = [2, 1, 16, 64].
        # The block_shape is <2x16x64xf16> which covers 64 N-columns = 1 stick,
        # so N//64 per block = 1.  The full tensor has 2 N-sticks (128/64=2),
        # but the block accesses only 1 stick at a time.
        self.run(self._kernel())
        self.assert_absent("tt.spyre_tensor_layout")
        self.assert_result_type("ktdp.load", "2x1x16x64xf16")


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
