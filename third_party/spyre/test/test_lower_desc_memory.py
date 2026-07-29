#!/usr/bin/env python3
"""
Unit tests for individual LowerDescriptorMemory conversion patterns.

Organization: one test class per Triton descriptor op (TestDescriptorLoad,
TestDescriptorStore, TestDescriptorGather, TestDescriptorScatter).  Each
class contains positive tests for static/dynamic shape variants plus
negative tests for expected failure modes.

Note on TTIR types: tt.make_tensor_descriptor requires shape operands as
i32 and stride operands as i64 (see TritonOps.td: Variadic<I32>:$shape,
Variadic<I64>:$strides).  The MLIR in test strings must match these types.

Shape vs block shape:
  tt.make_tensor_descriptor shape args = full tensor dimensions (e.g. 1024)
  tt.tensordesc<BxC> type              = block (tile) shape (e.g. 64)
  The memory view must use the full tensor shape; the access tile carries
  the block shape and is positioned by the load/store indices.
"""

import re

import pytest
from conftest import SinglePassTester
from utils_pattern import pattern


def _parse_indirect_subscripts(text: str):
    """Return the top-level per-dim subscript strings of the first
    ``ktdp.construct_indirect_access_tile`` op in ``text``.

    The custom printer writes the per-dim subscripts inline rather than as a
    ``subscript_kinds = [...]`` attribute list, e.g. (single-line, simplified)::

        ktdp.construct_indirect_access_tile
            intermediate_variables(%d0, %d1, %d2)
            %view[ind(%idx[%xoff + %d0]), (%yoff + %d1), (%d2)]

    The kind is encoded by the leading token of each subscript:
      - ``ind(...)`` → indirect (kindTrue)
      - bare ``(...)`` → direct (kindFalse)

    Returns the list of subscript strings (one per base dimension), split on
    top-level commas only (the indirect dim's inner ``[...]`` is balanced over).
    Asserts the op is present and its bracket list is well-formed.
    """
    m = re.search(r"ktdp\.construct_indirect_access_tile([\s\S]*?)\{", text)
    assert m, f"expected a `ktdp.construct_indirect_access_tile` op in:\n{text}"
    head = m.group(1)

    # Extract the bracketed subscript list: everything between the outermost
    # `[` and its matching `]`. A simple regex won't do — each subscript may
    # itself contain `[...]`, so balance brackets manually.
    lo = head.find("[")
    assert lo >= 0, f"no `[...]` subscript list found:\n{head}"
    depth = 0
    hi = -1
    for i, ch in enumerate(head[lo:], start=lo):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                hi = i
                break
    assert hi > lo, f"unbalanced brackets in subscript list:\n{head}"
    blob = head[lo + 1:hi]

    # Split on commas NOT inside a nested `[...]` (the indirect dim's inner pair).
    subscripts = []
    depth = 0
    last = 0
    for i, ch in enumerate(blob):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            subscripts.append(blob[last:i].strip())
            last = i + 1
    subscripts.append(blob[last:].strip())
    return subscripts


class LowerDescMemoryTester(SinglePassTester):
    """Shared base for all LowerDescriptorMemory pattern tests."""
    PASS = "add_lower_descriptor_memory"


# =========================================================================
# tt.descriptor_load → construct_memory_view + construct_access_tile + load
# =========================================================================

class TestDescriptorLoad(LowerDescMemoryTester):
    # tt.descriptor_load → ktdp.load via memory view + access tile.
    # Memory view uses the full tensor shape (not the block shape).
    #
    # tt.make_tensor_descriptor syntax:
    #   tt.make_tensor_descriptor %ptr, [shape...], [strides...] : <elem>, <block>
    #
    #   shape operands (i32) — full tensor dimensions, e.g. [1024, 64]
    #   strides operands (i64) — element strides for each dimension
    #   result type <BM x BK x elem> — block (tile) shape, e.g. <32x64xf16>
    #
    # Example: a 1024×64 tensor tiled with 32×64 blocks:
    #   %desc = tt.make_tensor_descriptor %ptr, [%M=1024, %K=64], [64, 1]
    #               : <f16>, <32x64xf16>
    #   → memory view covers the full 1024×64 tensor
    #   → each load/store positions a 32×64 access tile at the given indices
    #
    # test_static_shape_1d[N]          — 1-D, parametrized over N values
    # test_static_shape_2d[M,K]        — 2-D, parametrized over (M,K) pairs
    # test_memory_view_full_shape      — full tensor shape, not block shape
    # test_elem_type[f16/f32/bf16/f64] — pass is not f16-specific
    # test_dynamic_shape_1d            — 1-D, runtime shape → memref<?xf16>
    # test_dynamic_shape_2d            — 2-D, both dims runtime → memref<?x?xf16>

    @pattern("descriptor-load-static", category="memory", example=[
        "desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1], block_shape=[BLOCK])",
        "tile = tl.descriptor_load(desc, [pid * BLOCK])  # loads tensor<BLOCKxf16>",
    ])
    @pytest.mark.parametrize("N", [512, 1024, 4096])
    def test_static_shape_1d(self, N):
        """Load a 1-D tile from a statically-shaped tensor descriptor.

        ``tt.make_tensor_descriptor`` with a compile-time constant shape lowers
        to ``ktdp.construct_memory_view`` (full tensor extent baked in) +
        ``ktdp.construct_access_tile`` (block-sized tile positioned by the load
        index) + ``ktdp.load``.  The block shape (e.g. 64) lives only on the
        access tile; the memory view always carries the full tensor size.
        """
        # 1-D load.  %N is an arith.constant — known at compile time.
        # The memory view gets shape [N]; the access tile is positioned by %off.
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %off: i32) {{
            %N = arith.constant {N} : i32      // full tensor size — compile-time constant
            %stride = arith.constant 1 : i64   // element stride (contiguous)
            %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                : <f16>, <64xf16>              // block shape = 64 (tile size)
            %data = tt.descriptor_load %desc[%off]
                : !tt.tensordesc<64xf16> -> tensor<64xf16>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_load")
        self.assert_result("ktdp.construct_memory_view", shape=[N], elem_type="f16")
        self.assert_result_type("ktdp.construct_access_tile", "xindex")
        # coordinate_set: 1 dim, 0 symbols (static), 2 constraints (lo ≥ 0, hi ≤ N−1)
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=1, num_symbols=0, num_constraints=2)
        # access_tile_set: 1 dim, 0 symbols, 2 constraints (lo ≥ 0, hi ≤ 63 = block−1)
        self.assert_integer_set("ktdp.construct_access_tile", "access_tile_set",
                                num_dims=1, num_symbols=0, num_constraints=2)

    @pytest.mark.parametrize("M,K", [(512, 64), (1024, 128), (2048, 256)])
    def test_static_shape_2d(self, M, K):
        # 2-D load.  %M and %K are arith.constant — known at compile time.
        # %m, %k are the tile's top-left corner in the full tensor (not the tile size).
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %m: i32, %k: i32) {{
            %M = arith.constant {M} : i32          // full tensor rows — compile-time
            %K = arith.constant {K} : i32          // full tensor cols — compile-time
            %stride_row = arith.constant {K} : i64 // row stride = K (row-major)
            %stride_col = arith.constant 1 : i64   // col stride = 1 (contiguous)
            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <32x64xf16>               // block shape = 32×64 (tile size)
            %data = tt.descriptor_load %desc[%m, %k]  // %m, %k: tile position in full tensor
                : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_load")
        self.assert_result("ktdp.construct_memory_view", shape=[M, K], elem_type="f16")
        # coordinate_set: 2 dims, 0 symbols, 4 constraints (lo/hi per dim)
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=2, num_symbols=0, num_constraints=4)
        # access_tile_set: 2 dims, range [0,31]×[0,63] = block shape − 1
        self.assert_integer_set("ktdp.construct_access_tile", "access_tile_set",
                                num_dims=2, num_symbols=0, num_constraints=4)

    def test_memory_view_full_shape(self):
        # Memory view should describe the full tensor, not the block.
        # tt.make_tensor_descriptor shape [%N=1024] is the full tensor size.
        # The block shape <64xf16> is the tile — only the access tile uses it.
        # The memory view must cover [1024], not [64].
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %off: i32) {
            %N = arith.constant 1024 : i32     // full tensor size
            %stride = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                : <f16>, <64xf16>              // block shape = 64
            %data = tt.descriptor_load %desc[%off]
                : !tt.tensordesc<64xf16> -> tensor<64xf16>
            tt.return
          }
        }
        """)
        # Memory view must use full tensor shape [1024], not block shape [64]
        self.assert_result("ktdp.construct_memory_view", shape=[1024],
                           elem_type="f16")
        self.assert_result("ktdp.construct_memory_view", shape_not=[64])

    @pytest.mark.parametrize("elem", ["f16", "f32", "bf16", "f64"])
    def test_elem_type(self, elem):
        # elem in make_tensor_descriptor comes from the pointer type.
        # The pass is not f16-specific; the element type flows through to the
        # memref type unchanged.
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<{elem}>, %off: i32) {{
            %N = arith.constant 1024 : i32
            %stride = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                : <{elem}>, <64x{elem}>
            %data = tt.descriptor_load %desc[%off]
                : !tt.tensordesc<64x{elem}> -> tensor<64x{elem}>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_load")
        self.assert_result("ktdp.construct_memory_view", shape=[1024], elem_type=elem)
        self.assert_result("ktdp.load", shape=[64], elem_type=elem)

    @pattern("descriptor-load-dynamic", category="memory", example=[
        "# N is a runtime kernel argument — descriptor emits memref<?xf16>",
        "desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1], block_shape=[BLOCK])",
        "tile = tl.descriptor_load(desc, [pid * BLOCK])",
    ])
    def test_dynamic_shape_1d(self):
        """Load from a 1-D descriptor whose shape is a runtime argument.

        When the tensor size ``%N`` is a ``tt.func`` argument rather than an
        ``arith.constant``, the compiler cannot see it at compile time and emits
        ``kDynamic`` for that dimension — producing ``memref<?xf16>``.  The
        ``coordinate_set`` gains an ``IntegerSet`` symbol bound to ``%N`` at
        runtime so the range constraint remains correct.
        """
        # %N is a tt.func argument (not an arith.constant), so buildBaseMemoryView
        # cannot extract a compile-time size — it emits kDynamic, producing
        # memref<?xf16>.  The coordinate_set uses an IntegerSet symbol bound to
        # the runtime %N value, so the range constraint is correct at runtime.
        #
        # NOTE: Full-pipeline execution requires the KTIR CPU backend to support
        # dynamic memrefs — verify end-to-end before using in production.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %off: i32, %N: i32) {
            // %N is a runtime argument — not a compile-time constant
            %stride = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                : <f16>, <64xf16>
            %data = tt.descriptor_load %desc[%off]
                : !tt.tensordesc<64xf16> -> tensor<64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.load")
        self.assert_absent("tt.descriptor_load")
        # Runtime shape → memref<?xf16>; coordinate_set has 1 dim, 1 symbol, 2 constraints
        self.assert_result_type("ktdp.construct_memory_view", "memref<?")
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=1, num_symbols=1, num_constraints=2)
        # Dynamic size becomes operand 1 (cast from i32 to index by arith.index_cast)
        self.assert_operand("ktdp.construct_memory_view", 1,
                            defined_by="arith.index_cast", type_substr="index")

    @pattern("descriptor-load-dynamic", category="memory", example=[
        "# M and K are runtime kernel arguments — descriptor emits memref<?x?xf16>",
        "desc = tl.make_tensor_descriptor(ptr, shape=[M, K], strides=[K, 1],",
        "                                 block_shape=[BLOCK_M, BLOCK_K])",
        "tile = tl.descriptor_load(desc, [pid_m * BLOCK_M, pid_k * BLOCK_K])",
    ])
    def test_dynamic_shape_2d(self):
        """Load from a 2-D descriptor where both dimensions are runtime arguments.

        Both ``%M`` and ``%K`` arrive as ``tt.func`` arguments, so the compiler
        emits ``kDynamic`` for each — producing ``memref<?x?xf16>``.  Each
        dynamic dimension gets its own ``IntegerSet`` symbol in the
        ``coordinate_set``, bound positionally to the corresponding ``dynSizes``
        operand.  The block (tile) shape in the descriptor type is always fixed
        at compile time.
        """
        # %M and %K are tt.func arguments (not arith.constant), so the compiler
        # cannot see the tensor size at compile time — both dims become kDynamic,
        # producing memref<?x?xf16>.  buildRangeSetND emits one IntegerSet symbol
        # per dynamic dim, each bound positionally to the corresponding dynSizes
        # operand, so the range constraint is correct at runtime.
        #
        # Note: <32x64xf16> is the block (tile) shape encoded in the descriptor
        # type — it is always fixed at compile time.  Only the full tensor
        # dimensions [%M, %K] are dynamic here.
        #
        # NOTE: Full-pipeline execution requires the KTIR CPU backend to support
        # dynamic memrefs — verify end-to-end before using in production.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %m: i32, %k: i32,
                     %M: i32, %K: i32) {
            // %M, %K are runtime arguments — full tensor dims are not compile-time constants
            // block shape <32x64xf16> in the descriptor type is fixed at compile time
            %stride_row = arith.constant 64 : i64  // row stride — still static
            %stride_col = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%M, %K],
                        [%stride_row, %stride_col] : <f16>, <32x64xf16>
            %data = tt.descriptor_load %desc[%m, %k]  // %m, %k: tile position
                : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.load")
        self.assert_absent("tt.descriptor_load")
        # Both runtime dims → memref<?x?xf16>; coordinate_set has 2 dims, 2 symbols, 4 constraints
        self.assert_result_type("ktdp.construct_memory_view", "memref<?x?")
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=2, num_symbols=2, num_constraints=4)

    @pattern("descriptor-load-dynamic-from-scalar-load", category="memory", example=[
        "# seqlen is read from memory (e.g. a per-batch sequence length),",
        "# not a tt.func argument — descriptor still emits memref<?xf16>",
        "seqlen = tl.load(seqlen_ptr)",
        "desc = tl.make_tensor_descriptor(ptr, shape=[seqlen], strides=[1],",
        "                                 block_shape=[BLOCK])",
        "tile = tl.descriptor_load(desc, [pid * BLOCK])",
    ])
    def test_dynamic_shape_from_scalar_load(self):
        """A shape operand fed by ``tt.load`` still yields kDynamic.

        Runs `LowerDescriptorMemory` alone (`LowerScalarLoad` never runs
        here) on IR where the shape operand's defining op is an unconverted
        `tt.load`. `getConstantInt` only recognizes a materialized
        `arith.constant`, so it treats the `tt.load` result as opaque and
        falls through to `kDynamic` — pinning that this pass doesn't depend
        on `LowerScalarLoad` running first.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %off: i32, %seqlen_ptr: !tt.ptr<i32>) {
            // %seqlen comes from tt.load, not arith.constant or a tt.func
            // argument — SinglePassTester runs only LowerDescriptorMemory,
            // so this tt.load is never touched by LowerScalarLoad here.
            %seqlen = tt.load %seqlen_ptr : !tt.ptr<i32>
            %stride = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%seqlen], [%stride]
                : <f16>, <64xf16>
            %data = tt.descriptor_load %desc[%off]
                : !tt.tensordesc<64xf16> -> tensor<64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.load")
        self.assert_absent("tt.descriptor_load")
        # The tt.load survives untouched — LowerDescriptorMemory's
        # ConversionTarget marks only the four descriptor ops illegal, so
        # tt.load is neither converted nor required to be legal elsewhere.
        self.assert_present("tt.load")
        self.assert_result_type("ktdp.construct_memory_view", "memref<?")
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=1, num_symbols=1, num_constraints=2)
        self.assert_operand("ktdp.construct_memory_view", 1,
                            defined_by="arith.index_cast", type_substr="index")


# =========================================================================
# tt.descriptor_store → construct_memory_view + construct_access_tile + store
# =========================================================================

class TestDescriptorStore(LowerDescMemoryTester):
    # tt.descriptor_store → ktdp.store via memory view + access tile.
    # Same structure as load — memory view over full tensor, access tile
    # at block position.  Only the final op differs (store vs load).
    #
    # tt.make_tensor_descriptor syntax (same as load):
    #   shape operands (i32) — full tensor dimensions
    #   strides operands (i64) — element strides
    #   result type <BM x BK x elem> — block (tile) shape
    #
    # test_static_shape_1d[N]   — 1-D, parametrized over N values
    # test_static_shape_2d[M,K] — 2-D, parametrized over (M,K) pairs
    # test_dynamic_shape_1d     — non-constant shape produces memref<?>

    @pattern("descriptor-store-static", category="memory", example=[
        "desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1], block_shape=[BLOCK])",
        "tl.descriptor_store(desc, tile, [pid * BLOCK])  # writes tensor<BLOCKxf16>",
    ])
    @pytest.mark.parametrize("N", [512, 1024, 4096])
    def test_static_shape_1d(self, N):
        # 1-D store.  %N is arith.constant — the memory view gets shape [N].
        # %data is the tensor<64xf16> tile to write; %off is the tile position.
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %off: i32, %data: tensor<64xf16>) {{
            %N = arith.constant {N} : i32      // full tensor size — compile-time
            %stride = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                : <f16>, <64xf16>              // block shape = 64
            tt.descriptor_store %desc[%off], %data
                : !tt.tensordesc<64xf16>, tensor<64xf16>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_access_tile", "ktdp.store")
        self.assert_absent("tt.descriptor_store")
        self.assert_result("ktdp.construct_memory_view", shape=[N], elem_type="f16")
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=1, num_symbols=0, num_constraints=2)
        self.assert_integer_set("ktdp.construct_access_tile", "access_tile_set",
                                num_dims=1, num_symbols=0, num_constraints=2)

    @pytest.mark.parametrize("M,K", [(512, 64), (1024, 128), (2048, 256)])
    def test_static_shape_2d(self, M, K):
        # 2-D store.  %M and %K are arith.constant — memory view gets shape [M, K].
        # %data is the tensor<32x64xf16> tile to write; %m, %k are the tile position.
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>, %m: i32, %k: i32,
                     %data: tensor<32x64xf16>) {{
            %M = arith.constant {M} : i32          // full tensor rows — compile-time
            %K = arith.constant {K} : i32          // full tensor cols — compile-time
            %stride_row = arith.constant {K} : i64 // row stride = K (row-major)
            %stride_col = arith.constant 1 : i64   // col stride = 1 (contiguous)
            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <32x64xf16>               // block shape = 32×64 (tile size)
            tt.descriptor_store %desc[%m, %k], %data  // %m, %k: tile position
                : !tt.tensordesc<32x64xf16>, tensor<32x64xf16>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_access_tile", "ktdp.store")
        self.assert_absent("tt.descriptor_store")
        self.assert_result("ktdp.construct_memory_view", shape=[M, K], elem_type="f16")
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=2, num_symbols=0, num_constraints=4)
        self.assert_integer_set("ktdp.construct_access_tile", "access_tile_set",
                                num_dims=2, num_symbols=0, num_constraints=4)

    @pattern("descriptor-store-dynamic", category="memory", example=[
        "# N is a runtime kernel argument — descriptor emits memref<?xf16>",
        "desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1], block_shape=[BLOCK])",
        "tl.descriptor_store(desc, tile, [pid * BLOCK])",
    ])
    def test_dynamic_shape_1d(self):
        # %N is a tt.func argument (not an arith.constant), producing memref<?xf16>.
        # The coordinate_set uses an IntegerSet symbol bound to the runtime %N value.
        #
        # Full-pipeline execution requires the KTIR CPU backend to support
        # dynamic memrefs — verify end-to-end before using in production.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %off: i32, %N: i32,
                     %data: tensor<64xf16>) {
            // %N is a runtime argument — not a compile-time constant
            %stride = arith.constant 1 : i64
            %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                : <f16>, <64xf16>
            tt.descriptor_store %desc[%off], %data
                : !tt.tensordesc<64xf16>, tensor<64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.store")
        self.assert_absent("tt.descriptor_store")
        # Runtime shape → memref<?xf16>; coordinate_set has 1 dim, 1 symbol, 2 constraints
        self.assert_result_type("ktdp.construct_memory_view", "memref<?")
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=1, num_symbols=1, num_constraints=2)

# =========================================================================
# tt.descriptor_load with rank-reduced result — NOT YET LOWERED
# =========================================================================

class TestRankReducedDescriptorLoad(LowerDescMemoryTester):
    """Descriptor with leading singleton dim + rank-reduced load result: not yet supported.

    A common Triton idiom for 3D batched matmul — see

        a_desc = tl.make_tensor_descriptor(a_ptr,
            shape=[B, M, K], strides=[M*K, K, 1],
            block_shape=[1, BLOCK_M, BLOCK_K])
        a3 = a_desc.load([b_idx, m, k])           # frontend: tensor<1xBLOCK_MxBLOCK_K>
        a2 = tl.reshape(a3, [BLOCK_M, BLOCK_K])   # required for tl.dot (2D-only)

    Upstream Triton's ``RankedReduceDescriptorLoads`` pattern in the
    ``triton-combine`` pass (``lib/Dialect/Triton/Transforms/Combine.cpp``;
    invoked as ``passes.ttir.add_combine`` in the spyre ``_make_ttir``
    pipeline) folds the ``tt.reshape(tt.descriptor_load)`` pattern into
    a rank-reduced load when the dropped dims are size 1. After
    ``add_combine`` runs, the descriptor stays 3D
    (``!tt.tensordesc<1x16x16xf32>``) but the load's result type
    becomes 2D (``tensor<16x16xf32>``). ``DescriptorLoadOp::verify``
    accepts this because it checks element count, not rank — and
    1*16*16 == 16*16.

    Current state: ``LowerDescriptorMemory`` builds the
    ``ktdp.construct_access_tile`` from the descriptor's 3D
    ``block_shape`` (``[1, 16, 16]``) and emits a 3D ``ktdp.load``,
    but ``ktdp.load`` requires its access tile shape to match the
    result tensor shape — which is 2D. The op verifier raises
    ``access tile shape must match result tensor shape`` and the
    pipeline fails.

    Fix plan: ``LowerDescriptorMemory`` should detect when the
    descriptor's ``block_shape`` rank exceeds the load result's
    rank and the dropped leading dims are all size 1, then either
    (a) emit the access tile + ``ktdp.load`` at the reduced rank
    directly, or (b) emit at full rank and insert a
    ``tensor.collapse_shape`` to bring the ``ktdp.load`` result down
    to 2D, to match the 2D uses of ``tt.descriptor_load`` in the
    rest of the IR.
    Once fixed, delete this test in favor of a positive
    rank-reduced-load test.
    """

    @pattern("descriptor-rank-reduce", category="memory", negative=True, example=[
        "# NOT supported: 3D descriptor with rank-reduced (2D) load result",
        "# Produced by triton-combine when it folds tt.reshape(tt.descriptor_load)",
        "# where the reshaped-away leading dims are all size 1:",
        "a_desc = tl.make_tensor_descriptor(a_ptr,",
        "    shape=[B, M, K], strides=[M*K, K, 1],",
        "    block_shape=[1, BLOCK_M, BLOCK_K])  # 3D descriptor",
        "a = tl.reshape(a_desc.load([b, m, k]), [BLOCK_M, BLOCK_K])  # rank-reduced",
    ])
    def test_rank_reduced_load_fails(self, capfd):
        # Minimal reproducer: a 3D descriptor whose load result has
        # been rank-reduced to 2D. We hand-write the post-combine IR
        # directly because SinglePassTester runs only the named pass —
        # the triton-combine pass (RankedReduceDescriptorLoads) that
        # would produce this shape mismatch from a tt.reshape pattern
        # is not in the pipeline.
        #
        # Element count agrees (1*16*16 == 16*16 == 256) so
        # DescriptorLoadOp::verify accepts the IR. The mismatch
        # only fires in LowerDescriptorMemory when ktdp.load is
        # built with the descriptor's 3D access tile and the
        # original 2D result type.
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%ptr: !tt.ptr<f32>, %b_idx: i32, %m: i32, %k: i32) {
                %B = arith.constant 4 : i32          // full tensor batch
                %M = arith.constant 128 : i32        // full tensor rows
                %K = arith.constant 32 : i32         // full tensor cols
                %stride_b = arith.constant 4096 : i64  // M*K = 128*32
                %stride_m = arith.constant 32 : i64    // K
                %stride_k = arith.constant 1 : i64
                %desc = tt.make_tensor_descriptor %ptr, [%B, %M, %K],
                            [%stride_b, %stride_m, %stride_k]
                          : <f32>, <1x16x16xf32>     // 3D block: leading 1
                // Result is 2D — the rank-reduced shape produced by the
                // triton-combine pass when an explicit tt.reshape collapses
                // the leading singleton. Element count matches (256 == 256),
                // so DescriptorLoadOp::verify accepts this IR.
                %data = tt.descriptor_load %desc[%b_idx, %m, %k]
                          : !tt.tensordesc<1x16x16xf32> -> tensor<16x16xf32>
                tt.return
              }
            }
            """)
        # Pin the exact verifier diagnostic from ktdp.load. A drift in
        # the error text (e.g. if the verifier message is rephrased)
        # or a fix that closes the gap will flag here.
        self.assert_stderr(capfd,
                           "ktdp.load",
                           "access tile shape must match result tensor shape")

# =========================================================================
# tt.descriptor_gather → construct_memory_view + construct_indirect_access_tile + load
# =========================================================================

class TestDescriptorGather(LowerDescMemoryTester):
    # tt.descriptor_gather → ktdp.load via indirect access tile.
    #
    # Gather reads non-contiguous rows from a 2-D tensor.  The block type
    # must have exactly 1 row; the number of rows actually gathered comes
    # from x_offsets, not the block shape.
    #
    # Example: a 1024×128 tensor, gathering 32 non-contiguous rows,
    # reading a 64-wide column slice starting at y_offset:
    #   %desc = tt.make_tensor_descriptor %ptr, [%M=1024, %K=128], [128, 1]
    #               : <f16>, <1x64xf16>          // block: 1 row × 64 cols
    #   %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
    #               : (..., tensor<32xi32>, i32) -> tensor<32x64xf16>
    #   → memory view covers the full 1024×128 tensor (memref<1024x128xf16>)
    #   → x_offsets (tensor<32xi32>) names the 32 row indices to read
    #   → y_offset is the starting column for the 64-wide tile within each row
    #      (e.g. y_offset=0 → cols 0–63, y_offset=64 → cols 64–127)
    #   → result shape [32, 64] = [len(x_offsets), block_cols]
    #
    # tt.make_tensor_descriptor: shape operands are i32, strides are i64.
    #
    # test_gather_2d[M,K]            — row-gather, parametrized over (M,K) pairs;
    #                                   x_offsets is staged via descriptor_load
    # test_gather_base_memview_shape — base memory view has full tensor shape
    # test_gather_from_descriptor_load_emits_no_cast
    #                                 — descriptor-load provenance: the trace
    #                                   path reuses the upstream memory view
    #                                   instead of inserting a cast.
    # test_gather_from_descriptor_load_captures_x_offset
    #                                 — same path, but with a non-zero
    #                                   descriptor_load offset.  Pins that
    #                                   the offset is threaded into the
    #                                   indirect subscript as
    #                                   `idx[<offset> + d0]` rather than
    #                                   silently dropped.
    # test_gather_with_x_offsets_arg_fails_to_legalize
    #                                 — x_offsets as a tensor-typed function
    #                                   arg has no traceable provenance, so
    #                                   the gather pattern returns failure()
    #                                   and applyPartialConversion emits
    #                                   "failed to legalize 'tt.descriptor_gather'".
    # test_descriptor_from_arg_fails — descriptor from block arg fails lowering

    @pattern("descriptor-gather", category="memory", example=[
        "# Gather 32 non-contiguous rows from a 2D tensor",
        "result = tl.descriptor_gather(desc, x_offsets, y_offset)",
        "# lowers to ktdp.construct_indirect_access_tile + ktdp.load",
    ])
    @pytest.mark.parametrize("M,K", [(512, 128), (1024, 128), (2048, 256)])
    def test_gather_2d(self, M, K):
        # Row-gather: x_offsets (tensor<32xi32>) names 32 non-contiguous rows.
        # The block type <1x64xf16> means each row tile is 1×64; gather fans
        # this out to 32 separate rows → result shape tensor<32x64xf16>.
        # y_offset is the starting column for the 64-wide tile within each row
        # (e.g. 0 → cols 0–63, 64 → cols 64–127 in a 128-col tensor).
        # The indirect access tile (not the direct one used by load/store) holds
        # the x_offsets buffer and a region with the per-row subscript maps.
        #
        # x_offsets is staged via a 1-D descriptor_load — the only provenance
        # the gather pattern accepts after the fallback removal (see
        # docs/gather_fallback_deep_dive.md).  Spyre kernels always pass index
        # buffers as !tt.ptr<i32> + tt.descriptor_load.
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,      // raw index buffer (HBM ptr)
                     %y_offset: i32) {{           // starting column for the 64-wide tile
            %M = arith.constant {M} : i32         // full tensor rows — compile-time
            %K = arith.constant {K} : i32         // full tensor cols — compile-time
            %stride_row = arith.constant {K} : i64  // row stride = K (row-major)
            %stride_col = arith.constant 1 : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1 : i64
            %c0_i32     = arith.constant 0 : i32

            // Stage the row indices via a 1-D descriptor — what real kernels do.
            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>               // block: 1 row × 64 cols (gather constraint)
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                // result: 32 gathered rows × 64 cols
                : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        # No cast: the trace path reuses the index descriptor's memory view.
        self.assert_absent("unrealized_conversion_cast")
        self.assert_has_region("ktdp.construct_indirect_access_tile")
        # Two memory views: one for the index buffer, one for the base table.
        # A regression that re-introduced the fallback would build a third.
        self.assert_count("ktdp.construct_memory_view", 2, cmp="eq")
        # Base-table memory view: 2 dims, 0 symbols, 4 constraints (lo/hi per dim).
        # assert_result matches by shape= so it picks the base view (not the
        # 1-D index view, whose shape is [32]).
        self.assert_result("ktdp.construct_memory_view", shape=[M, K], elem_type="f16")
        # Base-table coordinate_set: 2 dims, 0 symbols, 4 constraints. The
        # 1-D index-buffer view has only 2 constraints, so this assertion
        # picks the 2-D view (assert_integer_set is "any match").
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=2, num_symbols=0, num_constraints=4)
        # variables_space_set: [0,31]×[0,63] = [len(x_offsets)−1, block_cols−1]
        self.assert_integer_set("ktdp.construct_indirect_access_tile", "variables_space_set",
                                num_dims=2, num_symbols=0, num_constraints=4)
        # gather load result shape = [len(x_offsets), block_cols].  The
        # descriptor_load for the index buffer also produces a ktdp.load with
        # shape [32]; assert_result matches by shape= so it picks the gather
        # result (not the index load).
        self.assert_result("ktdp.load", shape=[32, 64], elem_type="f16")

    def test_gather_base_memview_shape(self):
        # The block type <1x64xf16> is the tile size — only 64 cols are read per
        # gather.  The memory view must still cover the full 1024×128 tensor so
        # the access tile can address any column offset within each row.
        # Assert shape=[1024, 128] (full tensor), not [1, 64] (block shape).
        #
        # x_offsets is staged via descriptor_load (the only legal provenance
        # post-fallback-removal); the assertions here pin the *base* table
        # memory view, not the index buffer's 1-D view, so they match by
        # shape= regardless of how many memory views the lowering builds.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %M = arith.constant 1024 : i32        // full tensor rows
            %K = arith.constant 128 : i32         // full tensor cols
            %stride_row = arith.constant 128 : i64  // row stride = K
            %stride_col = arith.constant 1 : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1 : i64
            %c0_i32     = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>               // block: 1 row × 64 cols
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
            tt.return
          }
        }
        """)
        # Memory view must use full tensor shape [1024, 128], not block shape [1, 64]
        self.assert_result("ktdp.construct_memory_view", shape=[1024, 128],
                           elem_type="f16")
        self.assert_result("ktdp.construct_memory_view", shape_not=[1, 64])

    def test_gather_from_descriptor_load_emits_no_cast(self):
        # Compiled-kernel path: x_offsets is defined in the same function
        # via a descriptor_load, not passed in as a function argument.
        #
        # buildIndirectAccessTile in LowerDescriptorMemory.cpp resolves
        # x_offsets by walking
        #   load → construct_access_tile → construct_memory_view
        # and reusing the load's source memref directly — no cast inserted.
        # If the trace fails (e.g. x_offsets is a tensor-typed block arg),
        # the gather pattern returns failure() and applyPartialConversion
        # surfaces the standard "failed to legalize" diagnostic; there is
        # no longer a fallback that wraps the SSA value in a fresh memref
        # via unrealized_conversion_cast.
        #
        # This test pins the trace-success path. A regression that makes
        # the trace return nullopt for valid input would now fail to
        # legalize instead of silently introducing a cast.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %M = arith.constant 1024 : i32
            %K = arith.constant 128 : i32
            %stride_row = arith.constant 128 : i64
            %stride_col = arith.constant 1 : i64
            %idx_count = arith.constant 32 : i32
            %idx_stride = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // Load x_offsets via a descriptor — same idiom as the
            // fixtures/gather/kernel.py compiled output.
            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            // Gather that consumes the descriptor-loaded indices.
            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
            tt.return
          }
        }
        """)
        # Core assertion: the trace path must eliminate the cast.
        self.assert_absent("unrealized_conversion_cast")
        # Both source ops are gone.
        self.assert_absent("tt.descriptor_gather", "tt.descriptor_load")
        # The downstream ktdp ops show up.
        self.assert_present("ktdp.construct_indirect_access_tile",
                            "ktdp.load")
        # Exactly two memory views: one for the index tensor (reused
        # from the descriptor_load lowering) and one for the base
        # table.  A regression that constructs an extra view for the
        # index buffer (e.g. by re-introducing a cast-based fallback)
        # would bump this count to 3.
        self.assert_count("ktdp.construct_memory_view", 2, cmp="eq")

    @pattern("descriptor-gather", category="memory", example=[
        "# Non-zero load offset is captured in the indirect subscript map",
        "idx = idx_desc.load([offset_m])             # offset propagated",
        "result = tl.descriptor_gather(desc, idx, y_offset)",
        "# → ind(%idx[offset_m + d0]) in construct_indirect_access_tile",
    ])
    def test_gather_from_descriptor_load_captures_x_offset(self):
        # Regression test for the embedding-fixture row-tile bug.
        #
        # When a kernel calls ``idx_desc.load([offset_m])`` inside a loop
        # and then ``in_desc.gather(idx, ...)``, the gather must read
        # ``indices[offset_m + d0]`` for d0 in [0, BLOCK_M) — not
        # ``indices[d0]``.  Before the fix, traceToSourceMemoryView
        # discarded the descriptor_load's offset operand, so the
        # generated indirect access tile always indexed the index
        # buffer's prefix, which silently corrupted every iteration of
        # the loop except the one with offset = 0 (see
        # fixtures/embedding/kernel.py and the original failure pattern:
        # rows 0–7 correct, rows 8+ identical to rows 0–7).
        #
        # The fix captures the descriptor_load offset in the
        # construct_indirect_access_tile's ``captured_variables`` and
        # changes the indirect subscript map to ``c_x + d0``.  In the
        # printed IR this manifests as ``ind(%idx[<ssa> + <iv>])``
        # rather than ``ind(%idx[<iv>])``.
        #
        # We use a non-constant ``%offset`` (a function arg) so that the
        # captured offset is visibly named in the IR, not folded away.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %offset: i32,        // descriptor_load offset — opaque to the pass
                     %y_offset: i32) {
            %M = arith.constant 1024 : i32
            %K = arith.constant 128 : i32
            %stride_row = arith.constant 128 : i64
            %stride_col = arith.constant 1 : i64
            %idx_count = arith.constant 64 : i32
            %idx_stride = arith.constant 1 : i64

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            // Non-zero offset — the value the fix must propagate.
            %x_offsets = tt.descriptor_load %idx_desc[%offset]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
            tt.return
          }
        }
        """)

        # Lowering succeeded and produced an indirect access tile.
        self.assert_present("ktdp.construct_indirect_access_tile")
        self.assert_absent("tt.descriptor_gather", "tt.descriptor_load",
                           "unrealized_conversion_cast")

        # The printed indirect subscript must be `ind(%X[<lhs> + <rhs>])`,
        # i.e. the row index is a sum — not a bare iv.  Matching by shape
        # (sum vs no sum) keeps the assertion robust to SSA renaming.
        text = str(self.mod)
        m = re.search(r"ktdp\.construct_indirect_access_tile.*?ind\(([^\)]*)\)",
                      text)
        assert m, (
            "expected a `ktdp.construct_indirect_access_tile … ind(…)` "
            f"line in the rendered module:\n{text}"
        )
        indirect_subscript = m.group(1)  # e.g. "%idx[%off_idx + %arg4]"
        assert " + " in indirect_subscript, (
            "indirect subscript must include the captured x-offset "
            f"(form `idx[<offset> + <iv>]`); got `ind({indirect_subscript})` "
            f"in:\n{text}"
        )

    def test_gather_from_descriptor_load_signed_block_type(self):
        """Regression: descriptor block ``<NxSI32>`` (signed) must trace
        cleanly even though ``tt.descriptor_load`` canonicalises its
        result tensor to ``tensor<NxI32>`` (signless).

        Triton's frontend produces this asymmetry whenever the index
        buffer is declared via ``!tt.ptr<i32>`` — the descriptor block
        type comes out as ``si32`` while the load result stays as
        signless ``i32``.  ``resolveIndexView`` therefore compares
        element types by integer bit width, not by identity.

        Without the bit-width compare, the assert in ``resolveIndexView``
        (``"ConvertDescriptorLoad invariant violated"``) would fire on
        every compiled gather kernel.  This test pins that case in a
        ``SinglePassTester`` setup so the bit-width branch is exercised
        without depending on the full Triton frontend.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %M = arith.constant 1024 : i32
            %K = arith.constant 128 : i32
            %stride_row = arith.constant 128 : i64
            %stride_col = arith.constant 1 : i64
            %idx_count = arith.constant 32 : i32
            %idx_stride = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // Signed descriptor block type — matches the IR Triton emits
            // for an !tt.ptr<i32> index buffer.
            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xsi32>
            // Load result is signless — the mismatch the bit-width
            // compare in resolveIndexView is there to absorb.
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xsi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
            tt.return
          }
        }
        """)
        # Lowering must succeed end-to-end — no unconverted descriptor
        # ops, no fallback cast, indirect access tile present.
        self.assert_absent("tt.descriptor_gather", "tt.descriptor_load",
                           "unrealized_conversion_cast")
        self.assert_present("ktdp.construct_indirect_access_tile",
                            "ktdp.load")

    @pattern("descriptor-gather", category="memory", negative=True, example=[
        "# NOT supported: x_offsets passed in as a tensor-typed kernel arg.",
        "# Spyre kernels must stage indices via tl.make_tensor_descriptor +",
        "# .load() from a !tt.ptr<i32> arg so the gather pattern can trace",
        "# the index buffer's provenance.",
        "@triton.jit",
        "def k(ptr, x_offsets, y_offset):  # x_offsets: tensor<32xi32> arg — REJECTED",
        "    desc = tl.make_tensor_descriptor(ptr, shape=[M, K], strides=[K, 1],",
        "                                     block_shape=[1, 64])",
        "    data = tl.descriptor_gather(desc, x_offsets, y_offset)",
    ])
    def test_gather_with_x_offsets_arg_fails_to_legalize(self, capfd):
        """An ``x_offsets`` that is a tensor-typed function argument is no
        longer lowered. The gather pattern returns ``failure()`` on the
        trace miss; ``applyPartialConversion`` then leaves the illegal
        ``tt.descriptor_gather`` op in place and fails with its standard
        "failed to legalize" diagnostic.

        Reason this exists: prior to the fallback removal this same input
        lowered through a path that built a fresh memref via
        ``unrealized_conversion_cast`` and emitted a bare-d0 indirect
        map — silently wrong as soon as ``offset_m`` was non-zero.  Spyre
        kernels always pass index buffers as ``!tt.ptr<i32>`` +
        ``tt.descriptor_load``, so the fallback was dead code masking a
        misuse.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%ptr: !tt.ptr<f16>,
                         %x_offsets: tensor<32xi32>,
                         %y_offset: i32) {
                %M = arith.constant 1024 : i32
                %K = arith.constant 128 : i32
                %stride_row = arith.constant 128 : i64
                %stride_col = arith.constant 1 : i64
                %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                    : <f16>, <1x64xf16>
                %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                    : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
                tt.return
              }
            }
            """)
        # MLIR's standard diagnostic for an unconverted illegal op.  If a
        # future change adds a custom emitOpError ahead of the failure()
        # return, tighten this substring to match the new wording.
        self.assert_stderr(capfd,
                           "failed to legalize operation 'tt.descriptor_gather'")

    def test_gather_block_dim0_not_one_fails(self, capfd):
        """Block dim-0 != 1 is rejected by DescriptorGatherOp::verify.

        ``<2x64xf16>`` means the block has 2 rows; the verifier requires
        exactly 1. This is a static property of the descriptor type, so
        ``DescriptorGatherOp::verify`` fires during ``ir.parse_mlir_module``
        (MLIR runs op verifiers at parse time), before the pass manager runs.
        Hence the exception message is "Parse MLIR file failed", not
        "PassManager::run failed".
        """
        with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
            self.run("""
            module {
              tt.func @k(%ptr: !tt.ptr<f16>,
                         %x_offsets: tensor<32xi32>, %y_offset: i32) {
                %M = arith.constant 1024 : i32
                %K = arith.constant 64 : i32
                %stride_row = arith.constant 64 : i64
                %stride_col = arith.constant 1 : i64
                %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                    : <f16>, <2x64xf16>
                %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                    : (!tt.tensordesc<2x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
                tt.return
              }
            }
            """)
        self.assert_stderr(capfd, "descriptor block must have exactly 1 row")

    def test_descriptor_from_arg_fails(self, capfd):
        """Descriptor arriving as a runtime value (block argument, call
        result, etc.) cannot be lowered.

        Dynamic tensor *shapes* inside a descriptor are fine — they
        produce `memref<?>` at compile time. What is not supported is a
        descriptor whose shape/stride info is not visible at compile
        time because the descriptor itself is a runtime value. The
        lowering needs the shape and stride constants from
        `tt.make_tensor_descriptor` to build the memory view and access
        tile; there is no way to recover them from an opaque descriptor.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%desc: !tt.tensordesc<1x64xf16>,  // descriptor as arg — no shape/stride info
                         %x_offsets: tensor<32xi32>, %y_offset: i32) {
                %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                    : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
                tt.return
              }
            }
            """)
        self.assert_stderr(capfd, "cannot lower descriptor op")

    # ----------------------------------------------------------------
    # Rank-1 source rejection + reshape-to-rank-2 workaround.
    #
    # ``tt.descriptor_gather`` indexes the descriptor's outermost
    # dimension. The verifier (``verifyGatherScatterOp`` in
    # ``Ops.cpp``) requires the block to be at least rank-2 even on
    # Spyre — the leading dim is the indirect axis (must be size 1
    # in the block), and at least one trailing direct dim must
    # exist. A truly rank-1 descriptor block has no separate indirect
    # axis to pin to size 1, so the op is structurally undefined.
    #
    # User-facing consequence: a 1D source vector ``in[K]`` whose
    # elements are gathered as scalars (``out[i] = in[idx[i]]``)
    # cannot be expressed by passing a rank-1 descriptor. The
    # workaround is to model the same K elements as a ``[K, 1]``
    # column matrix with ``block_shape=[1, 1]``; the gather then
    # indexes dim 0 (size K) and produces a ``[K_INDICES, 1]``
    # result, which the kernel can collapse back to 1D if needed.
    # ----------------------------------------------------------------

    @pattern("descriptor-gather", category="memory", negative=True, example=[
        "# NOT supported: rank-1 descriptor block for 1D-source gather.",
        "# The frontend rejects this with 'descriptor must be at least 2D';",
        "# at the IR level the op verifier emits the same diagnostic.",
        "in_desc = tl.make_tensor_descriptor(in_ptr,",
        "                                    shape=[K], strides=[1],",
        "                                    block_shape=[BLOCK_COLS])  # rank-1 — REJECTED",
        "out = tl.descriptor_gather(in_desc, idx, 0)",
    ])
    def test_gather_rank1_block_rejected(self, capfd):
        """Rank-1 descriptor block is rejected by the gather verifier.

        ``<32xf16>`` is a rank-1 block — there is no leading
        indirect dim to fan ``x_offsets`` over, so the verifier
        in ``Ops.cpp::verifyGatherScatterOp`` rejects it at parse
        time with "descriptor block must be at least 2D" (Spyre
        relaxed the upstream "must be 2D exactly" check to "rank
        >= 2", but rank-1 is still illegal).

        Hence the exception message is "Parse MLIR file failed",
        not "PassManager::run failed" — the verifier fires before
        the pass manager runs. The companion positive test
        :meth:`test_gather_1d_source_via_rank2_reshape_lowers`
        shows the working idiom: model the 1D source as a
        ``[K, 1]`` column matrix with ``block_shape=[1, 1]``.
        """
        with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
            self.run("""
            module {
              tt.func @k(%ptr: !tt.ptr<f16>,
                         %x_offsets: tensor<32xi32>, %y_offset: i32) {
                %K = arith.constant 1024 : i32
                %stride = arith.constant 1 : i64
                %desc = tt.make_tensor_descriptor %ptr, [%K], [%stride]
                    : <f16>, <32xf16>
                %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                    : (!tt.tensordesc<32xf16>, tensor<32xi32>, i32) -> tensor<32xf16>
                tt.return
              }
            }
            """)
        self.assert_stderr(capfd, "descriptor block must be at least 2D")

    @pattern("descriptor-gather", category="memory", example=[
        "# Workaround for rank-1 source: model the 1D vector as a [K, 1]",
        "# column matrix and gather with a [1, 1] block. The gather still",
        "# produces a rank-2 result tensor<K_INDICES x 1 x f16>; collapse",
        "# it to 1D in the kernel if needed.",
        "in_desc = tl.make_tensor_descriptor(in_ptr,",
        "                                    shape=[K, 1], strides=[1, 1],",
        "                                    block_shape=[1, 1])",
        "out = tl.descriptor_gather(in_desc, idx, 0)  # tensor<K_INDICES x 1 x f16>",
    ])
    def test_gather_1d_source_via_rank2_reshape_lowers(self):
        """A 1D source vector gathered as a rank-2 ``[K, 1]`` matrix lowers.

        Source described as ``[1024, 1]`` with block ``<1x1xf16>`` —
        the leading 1 in the block satisfies the "block dim 0 == 1"
        gather constraint, and the trailing dim of size 1 is the
        per-row payload (one scalar per gathered row). The result
        is rank-2 ``tensor<32x1xf16>``.

        This is the Spyre-supported way to express ``out[i] =
        in_1d[idx[i]]`` on a 1D source: there is no rank-1 gather
        op in the dialect, so the kernel must treat the vector as
        a column matrix. Pinned alongside
        :meth:`test_gather_rank1_block_rejected` so that if the
        verifier is ever relaxed to accept rank-1 blocks, both the
        rejection and the workaround status of this idiom show up
        as XPASS together.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %K = arith.constant 1024 : i32
            %one = arith.constant 1 : i32
            %stride_row = arith.constant 1 : i64
            %stride_col = arith.constant 1 : i64
            %idx_count = arith.constant 32 : i32
            %idx_stride = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // Index buffer staged via descriptor_load — same idiom as
            // the rank-2 gather test above. The gather pattern's
            // index-trace requires this provenance.
            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            // 1D source modelled as [K, 1]; block <1x1xf16> picks one
            // scalar per gathered index. Result is rank-2 [32, 1].
            %desc = tt.make_tensor_descriptor %ptr, [%K, %one], [%stride_row, %stride_col]
                : <f16>, <1x1xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x1xf16>, tensor<32xi32>, i32) -> tensor<32x1xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")


# =========================================================================
# tt.descriptor_gather — N-D block (Spyre extension)
# =========================================================================

class TestDescriptorGatherND(LowerDescMemoryTester):
    # N-D ``tt.descriptor_gather`` accepts rank ≥ 2 source blocks.
    # ``buildGatherSubscriptMaps`` / ``buildIndirectAccessTile`` produce
    # one indirect map on dim 0 (``c_x + d_0``), one direct map with
    # ``y_offset`` on dim 1 (``c_y + d_1``), and a plain ``d_i`` direct
    # map for every dim ``i ∈ [2, rank)``.  The ``IntegerSet`` covers
    # the full result shape; the order is identity over rank.
    #
    # Tests:
    #   test_gather_3d_lowered    — rank-3 block, one indirect dim + two direct
    #   test_gather_4d_lowered    — rank-4 block, indirect dim + group dim + two direct
    #   test_gather_5d_lowered    — rank-5 block, stickified inner dim
    #   test_scatter_3d_lowered   — scatter mirror of the rank-3 case

    @pattern("descriptor-gather-nd", category="memory", example=[
        "# rank-3 N-D indirect-access gather (Spyre extension):",
        "src_desc = tl.make_tensor_descriptor(src_ptr, [NUM_BLOCKS, BLOCK_SIZE, INNER_DIM], ...,",
        "                                     block_shape=[1, BLOCK_SIZE, INNER_DIM])",
        "result = tl.descriptor_gather(src_desc, indices, 0)",
        "# → tensor<32 x BLOCK_SIZE x INNER_DIM x f16>  (3D result; one indirect axis,",
        "#                                                two direct dims with no offset)",
    ])
    def test_gather_3d_lowered(self):
        """Rank-3 ``tt.descriptor_gather`` lowers cleanly.

        Block shape ``<1 x BLOCK_SIZE x INNER_DIM>`` (the source tensor is
        ``NUM_BLOCKS x BLOCK_SIZE x INNER_DIM``). Indices drive dim 0 (block id);
        ``y_offset`` lands on dim 1 (in-block row); dim 2 (the contiguous inner
        dim) is direct with no offset.

        Subscript-map shape after lowering:
          dim 0 (indirect): c_x + d_0
          dim 1 (direct):   c_y + d_1
          dim 2 (direct):           d_2

        The full iteration space is ``0 ≤ d_0 < 32, 0 ≤ d_1 < 16, 0 ≤ d_2 < 128``.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %P = arith.constant 1024 : i32     // NUM_BLOCKS
            %B = arith.constant 16   : i32     // BLOCK_SIZE
            %D = arith.constant 128  : i32     // INNER_DIM
            %s0 = arith.constant 2048 : i64    // block stride = BLOCK_SIZE * INNER_DIM
            %s1 = arith.constant 128  : i64    // row stride   = INNER_DIM
            %s2 = arith.constant 1    : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1  : i64
            %c0_i32     = arith.constant 0  : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
                : <f16>, <1x16x128xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x16x128xf16>, tensor<32xi32>, i32) -> tensor<32x16x128xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        self.assert_absent("unrealized_conversion_cast")
        self.assert_count("ktdp.construct_memory_view", 2, cmp="eq")
        self.assert_result("ktdp.construct_memory_view",
                           shape=[1024, 16, 128], elem_type="f16")
        # variables_space_set: 3 dims × 2 inequalities each = 6 constraints.
        self.assert_integer_set("ktdp.construct_indirect_access_tile",
                                "variables_space_set",
                                num_dims=3, num_symbols=0, num_constraints=6)
        # Result tile is the rank-3 N-D-gather shape.
        self.assert_result("ktdp.load", shape=[32, 16, 128], elem_type="f16")

    @pattern("descriptor-gather-4d", category="memory", example=[
        "# rank-4 N-D indirect-access gather with a group dim:",
        "src_desc = tl.make_tensor_descriptor(",
        "    src_ptr,",
        "    shape=[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM],   # groups at dim 1",
        "    strides=[..., INNER_DIM, NUM_GROUPS*INNER_DIM, 1],",
        "    block_shape=[1, 1, BLOCK_SIZE, INNER_DIM])",
        "result = tl.descriptor_gather(src_desc, indices, group_idx)",
    ])
    def test_gather_4d_lowered(self):
        """Rank-4 gather: indirect dim plus a group dim that takes the y_offset.

        The descriptor's group axis is permuted to dim 1 so the single
        ``y_offset = group_idx`` lands there; dims 2 and 3 (the inner
        ``BLOCK_SIZE`` and ``INNER_DIM`` axes) are direct with no offset.
        Pins that the lowering handles a rank > 3 result block where the
        indirect axis is on dim 0 and exactly one direct axis carries an
        offset.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %group_idx: i32) {
            %P = arith.constant 64  : i32     // NUM_BLOCKS
            %H = arith.constant 8   : i32     // NUM_GROUPS (permuted to dim 1)
            %B = arith.constant 16  : i32     // BLOCK_SIZE
            %D = arith.constant 128 : i32     // INNER_DIM
            // Strides over the permuted shape; physical layout is row-major
            // [NUM_BLOCKS, BLOCK_SIZE, NUM_GROUPS, INNER_DIM]:
            %s0 = arith.constant 16384 : i64  // block stride = B*H*D
            %s1 = arith.constant 128   : i64  // group stride = D    (non-monotone)
            %s2 = arith.constant 1024  : i64  // row   stride = H*D
            %s3 = arith.constant 1     : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1  : i64
            %c0_i32     = arith.constant 0  : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %indices = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr,
                        [%P, %H, %B, %D], [%s0, %s1, %s2, %s3]
                : <f16>, <1x1x16x128xf16>
            %data = tt.descriptor_gather %desc[%indices, %group_idx]
                : (!tt.tensordesc<1x1x16x128xf16>, tensor<32xi32>, i32)
                  -> tensor<32x1x16x128xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        self.assert_absent("unrealized_conversion_cast")
        # variables_space_set: 4 dims × 2 = 8 constraints.
        self.assert_integer_set("ktdp.construct_indirect_access_tile",
                                "variables_space_set",
                                num_dims=4, num_symbols=0, num_constraints=8)
        self.assert_result("ktdp.load", shape=[32, 1, 16, 128], elem_type="f16")

    @pattern("descriptor-gather-5d", category="memory", example=[
        "# rank-5 N-D indirect-access gather with a stickified inner dim:",
        "src_desc = tl.make_tensor_descriptor(",
        "    src_ptr,",
        "    shape=[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, NUM_STICKS, STICK_SIZE],",
        "    block_shape=[1, 1, BLOCK_SIZE, NUM_STICKS, STICK_SIZE])",
        "result = tl.descriptor_gather(src_desc, indices, group_idx)",
    ])
    def test_gather_5d_lowered(self):
        """Rank-5 gather: indirect dim, group dim, and a split (stickified) inner dim.

        Group axis permuted to dim 1 (takes the ``y_offset = group_idx``);
        ``BLOCK_SIZE`` at dim 2; the trailing contiguous dim is split
        across dim 3 (``NUM_STICKS``, the stick group) and dim 4
        (``STICK_SIZE``, the within-stick offset). Endpoint coverage for
        the relaxed rank range — a regression that silently dropped
        trailing dims would surface here.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %group_idx: i32) {
            %P  = arith.constant 64  : i32    // NUM_BLOCKS
            %H  = arith.constant 8   : i32    // NUM_GROUPS
            %B  = arith.constant 16  : i32    // BLOCK_SIZE
            %S  = arith.constant 2   : i32    // NUM_STICKS
            %SE = arith.constant 64  : i32    // STICK_SIZE  (INNER_DIM = NUM_STICKS*STICK_SIZE = 128)
            %s0 = arith.constant 16384 : i64
            %s1 = arith.constant 128   : i64
            %s2 = arith.constant 1024  : i64
            %s3 = arith.constant 64    : i64
            %s4 = arith.constant 1     : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1  : i64
            %c0_i32     = arith.constant 0  : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %indices = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr,
                        [%P, %H, %B, %S, %SE], [%s0, %s1, %s2, %s3, %s4]
                : <f16>, <1x1x16x2x64xf16>
            %data = tt.descriptor_gather %desc[%indices, %group_idx]
                : (!tt.tensordesc<1x1x16x2x64xf16>, tensor<32xi32>, i32)
                  -> tensor<32x1x16x2x64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        self.assert_integer_set("ktdp.construct_indirect_access_tile",
                                "variables_space_set",
                                num_dims=5, num_symbols=0, num_constraints=10)
        self.assert_result("ktdp.load", shape=[32, 1, 16, 2, 64], elem_type="f16")

    @pattern("descriptor-scatter-nd", category="memory", example=[
        "# rank-3 scatter mirror of test_gather_3d_lowered.",
        "tl.descriptor_scatter(dst_desc, indices, y_offset, value)",
    ])
    def test_scatter_3d_lowered(self):
        """Rank-3 scatter mirror — same machinery via ``ConvertDescriptorScatter``.

        Confirms the lowering's N-D generalisation applies symmetrically
        to writes.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32,
                     %src: tensor<32x16x128xf16>) {
            %P = arith.constant 1024 : i32
            %B = arith.constant 16   : i32
            %D = arith.constant 128  : i32
            %s0 = arith.constant 2048 : i64
            %s1 = arith.constant 128  : i64
            %s2 = arith.constant 1    : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1  : i64
            %c0_i32     = arith.constant 0  : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
                : <f16>, <1x16x128xf16>
            tt.descriptor_scatter %desc[%x_offsets, %y_offset], %src
                : !tt.tensordesc<1x16x128xf16>, tensor<32xi32>, i32, tensor<32x16x128xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_indirect_access_tile", "ktdp.store")
        self.assert_absent("tt.descriptor_scatter")
        self.assert_integer_set("ktdp.construct_indirect_access_tile",
                                "variables_space_set",
                                num_dims=3, num_symbols=0, num_constraints=6)

    @pattern("descriptor-gather-nd-subscripts", category="memory", example=[
        "# Rank-3 subscript role split: dim 0 indirect, dim 1 y_offset, dim 2 full",
        "# desc block shape <1 x TOKEN_DIM x HEAD_DIM>",
        "result = tl.descriptor_gather(desc, x_offsets, y_offset)",
        "# lowers to: ind(idx[c_x + d0]), (c_y + d1), (d2)",
        "# Trailing dim 2 (HEAD_DIM): no offset — full block extent always read.",
        "# To slice dim 2, reshape the result after the gather.",
    ])
    def test_gather_3d_subscript_kinds_pin_offset_axis(self):
        """Pin the dim 0 / dim 1 / dim ≥ 2 role split — the headline N-D limitation.

        At every rank ≥ 2 the lowering encodes exactly two "movable"
        axes: dim 0 (indirect, fed by the index buffer) and dim 1
        (direct, offset by ``y_offset``).  Every dim ``i ≥ 2`` is a
        bare ``d_i`` — no offset, no capture, the kernel reads the
        block's full extent on that axis with no ability to slice.

        That asymmetry is the most kernel-author-visible consequence
        of the relaxed verifier: at rank 2 it didn't matter (there
        was no dim ≥ 2), at rank 3+ it constrains every kernel that
        wants more than one offset axis or wants to gather along
        anything other than the outermost axis.

        Without pinning the textual form of ``subscript_kinds`` and
        ``subscript_maps`` here, a regression that emitted, say, a
        second indirect axis or a captured ``c_y`` on a trailing dim
        would slip past the existing rank-3 structural counts (which
        only check ``num_dims`` / ``num_constraints`` on
        ``variables_space_set``).
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %P = arith.constant 1024 : i32
            %B = arith.constant 16   : i32
            %D = arith.constant 128  : i32
            %s0 = arith.constant 2048 : i64
            %s1 = arith.constant 128  : i64
            %s2 = arith.constant 1    : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1  : i64
            %c0_i32     = arith.constant 0  : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
                : <f16>, <1x16x128xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x16x128xf16>, tensor<32xi32>, i32) -> tensor<32x16x128xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_indirect_access_tile")

        # The custom printer for ``construct_indirect_access_tile`` writes
        # the per-dim subscripts inline; ``_parse_indirect_subscripts`` peels
        # them apart (see that helper for the printed form / kind encoding).
        subscripts = _parse_indirect_subscripts(str(self.mod))

        assert len(subscripts) == 3, (
            f"rank-3 indirect tile must have 3 subscripts; got {len(subscripts)}: "
            f"{subscripts}"
        )

        # Dim 0: the *only* indirect axis. Must start with `ind(`.
        assert subscripts[0].startswith("ind("), (
            "rank-3: dim 0 must be the indirect axis (`ind(...)`); "
            f"got `{subscripts[0]}`"
        )
        # And must capture x_offset — the printed form is
        # `ind(<view>[<x_offset> + <iv0>])`, so the inner expression
        # contains a ` + `.
        assert " + " in subscripts[0], (
            "rank-3: dim 0 indirect subscript must capture x_offset "
            f"(form `ind(idx[<offset> + <iv>])`); got `{subscripts[0]}`"
        )

        # Dim 1: direct, captures y_offset → printed as `(<y_off> + <iv1>)`.
        assert not subscripts[1].startswith("ind("), (
            f"rank-3: dim 1 must NOT be indirect; got `{subscripts[1]}`"
        )
        assert " + " in subscripts[1], (
            "rank-3: dim 1 direct subscript must capture y_offset "
            f"(form `(<y_offset> + <iv>)`); got `{subscripts[1]}`"
        )

        # Dim 2: direct, NO capture → printed as bare `(<iv2>)`.
        # This is the headline N-D limitation: trailing dims read full
        # extent only.
        assert not subscripts[2].startswith("ind("), (
            f"rank-3: dim 2 must NOT be indirect; got `{subscripts[2]}`"
        )
        assert " + " not in subscripts[2], (
            "rank-3: dim 2 must NOT capture an offset (the N-D lowering "
            "supports at most one offset axis, on dim 1); got "
            f"`{subscripts[2]}`"
        )

    @pattern("descriptor-gather-nd-permuted-strides", category="memory", example=[
        "# Physical layout: [BLOCK_SIZE, NUM_BLOCKS, INNER_DIM] — block id is NOT dim 0.",
        "# Declare descriptor shape with the gathered axis at dim 0, fix up via strides:",
        "desc = tl.make_tensor_descriptor(",
        "    ptr,",
        "    shape=[NUM_BLOCKS, BLOCK_SIZE, INNER_DIM],",
        "    strides=[INNER_DIM, NUM_BLOCKS * INNER_DIM, 1],  # inverted stride order",
        "    block_shape=[1, BLOCK_SIZE, INNER_DIM])",
        "result = tl.descriptor_gather(desc, block_ids, y_offset)",
    ])
    def test_gather_3d_inner_axis_via_stride_permutation(self):
        """Gathering a 'logically inner' physical axis via stride permutation.

        The lowering wires the index buffer onto dim 0 of the descriptor
        — there is no way to point it at any other dim.  Kernel authors
        whose physical tensor stores the gathered axis in the middle or
        at the end have one option: declare the descriptor's *shape*
        with the gathered axis at dim 0, then express the original
        physical layout through the *strides*.

        The example here is a tensor laid out as
        ``[BLOCK_SIZE, NUM_BLOCKS, INNER_DIM]`` (block-row-major rather
        than block-id-major) but which the kernel still wants to
        *gather* by block id.  The descriptor's shape is declared
        ``[NUM_BLOCKS, BLOCK_SIZE, INNER_DIM]`` and the stride list is
        ``[INNER_DIM, NUM_BLOCKS*INNER_DIM, 1]`` — the block-id-stride
        is smaller than the block-row-stride, the inverse of the
        outermost-block-id case.

        From the lowering's perspective, nothing changes: it still
        fans index ``i`` onto dim 0.  The trick is purely on the
        kernel-author side, and the resulting access pattern is
        identical to the outermost-block-id rank-3 gather.

        Pins: lowering succeeds, view shape matches the *declared*
        descriptor shape (not the physical layout), result tile
        matches the gathered shape.  A regression that re-derived the
        memory view from anything other than the descriptor's shape
        operands would surface as a wrong view shape here.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            // Declared shape exposes block ids as dim 0, even though the
            // physical layout stores BLOCK_SIZE outermost.
            %P  = arith.constant 1024 : i32     // NUM_BLOCKS (declared dim 0)
            %B  = arith.constant 16   : i32     // BLOCK_SIZE (declared dim 1)
            %D  = arith.constant 128  : i32     // INNER_DIM  (declared dim 2)
            // Strides reflect the physical [BLOCK_SIZE, NUM_BLOCKS, INNER_DIM]:
            //   block-id stride  = INNER_DIM              = 128
            //   block-row stride = NUM_BLOCKS*INNER_DIM   = 1024 * 128 = 131072
            //   inner stride     = 1
            // Note: block-id stride < block-row stride, the *inverse* of
            // the outermost-block-id layout used elsewhere.
            %s0 = arith.constant 128    : i64
            %s1 = arith.constant 131072 : i64
            %s2 = arith.constant 1      : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1  : i64
            %c0_i32     = arith.constant 0  : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
                : <f16>, <1x16x128xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x16x128xf16>, tensor<32xi32>, i32) -> tensor<32x16x128xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        # Memory view shape comes from the *declared* shape operands —
        # the physical-layout strides are recorded but do not influence
        # the view's logical shape.
        self.assert_result("ktdp.construct_memory_view",
                           shape=[1024, 16, 128], elem_type="f16")
        self.assert_result("ktdp.load", shape=[32, 16, 128], elem_type="f16")

    @pattern("descriptor-gather-nd-trailing-one", category="memory", example=[
        "# Block <1x1x128>: trailing dim 1 = 1 — newly legal.",
        "# Pre-relaxation rejected two ways: rank != 2 and",
        "# block.shape[1] < min_cols (TMA rule). Both gone.",
        "src_desc = tl.make_tensor_descriptor(src_ptr, ..., block_shape=[1, 1, 128])",
        "result = tl.descriptor_gather(src_desc, indices, y_offset)",
    ])
    def test_gather_3d_inner_dim_one_lowered(self):
        """Block ``<1x1x128>``: a singleton at dim 1 is now legal.

        A natural shape for an N-D gather where each indirectly-addressed
        block contains exactly one row (so dim 1 collapses to size 1 and
        only the trailing contiguous dim has interior extent).  Before
        the rank-N relaxation, this was rejected for two unrelated
        reasons:

        * The descriptor was rank 3, and the verifier required rank 2.
        * Even after relaxing rank, the TMA min-cols rule required
          ``block.shape[1] >= 32 / bitwidth * 8`` — for ``f16`` that's
          ``>= 16``, so ``block.shape[1] = 1`` was rejected.

        Both rules are now gone for Spyre.  This test pins the
        smallest-possible interior dim so a regression that
        re-introduced the min-cols check (or that silently
        rank-reduced the trailing 1 inside the lowering) would surface
        immediately.

        Coverage handover: the rank-3/4/5 ``test_gather_*_lowered``
        tests already cover the structural shape of N-D gather; this
        test specifically probes the *boundary* where dim 1 = 1, which
        the old verifier would have rejected.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %P  = arith.constant 1024 : i32
            %B  = arith.constant 16   : i32
            %D  = arith.constant 128  : i32
            %s0 = arith.constant 2048 : i64
            %s1 = arith.constant 128  : i64
            %s2 = arith.constant 1    : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1  : i64
            %c0_i32     = arith.constant 0  : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            // Block: 1 page, 1 token row, full 128-wide head dim.
            %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
                : <f16>, <1x1x128xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x1x128xf16>, tensor<32xi32>, i32) -> tensor<32x1x128xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        # The trailing 1 must survive — must NOT be silently rank-reduced.
        self.assert_result("ktdp.load", shape=[32, 1, 128], elem_type="f16")


# =========================================================================
# tt.descriptor_gather / scatter — 2-D (rank-N) x_offsets
# =========================================================================
#
# STATUS: implemented (Steps 1–4 of docs/impl-strategy-2d-x-offsets.md).
#
# The N-D relaxation (#8) widened the *descriptor block* and *result* to
# rank ≥ 2 but kept ``x_offsets`` strictly 1-D.  The rank-K relaxation lets
# ``x_offsets`` itself be rank ≥ 1, so a kernel can gather/scatter a
# multi-dimensional *grid* of pages (e.g. one page per (sequence, head) pair)
# without flattening by hand.  (The old ``test_gather_2d_indices_rejected``,
# which pinned the verifier rejection, was removed with that change.)
#
# Semantics (K = rank(x_offsets), R = block rank, result rank = K + R - 1):
#
#   x_offsets : tensor<N x M x i32>          (rank-2 index grid, K=2)
#   desc block: <1 x C x dtype>              (leading 1 still required)
#   result    : tensor<N x M x C x dtype>    (index grid dims, then block[1:])
#
#   result[d_0, d_1, d_2] = base[ x_offsets[c_x0 + d_0, c_x1 + d_1],  // ONE
#                                 y_offset + d_2 ]                     // base dim 0
#
# Crucially this is **one** indirect base dim (the page/row axis) whose
# subscript map produces K results — the K-D address into the index view —
# NOT K separate indirect base dims.  That is exactly the form
# ``ktdp.construct_indirect_access_tile`` expects: the KTDP verifier requires
# an indirect subscript map's result count to equal the index memref rank
# (KtdpOps.cpp).  So the printed access tile has R subscripts (one per base
# dim), of which the first is ``ind(idx[c_x0 + d_0, ..., c_x{K-1} + d_{K-1}])``.
# See ``docs/impl-strategy-2d-x-offsets.md``.

class TestDescriptorGatherScatter2DIndices(LowerDescMemoryTester):
    """Lowering of ``tt.descriptor_gather`` / ``tt.descriptor_scatter`` with a
    rank-2 ``x_offsets`` index grid.

    Each test stages ``x_offsets`` via a 2-D ``tt.descriptor_load`` — the same
    "index buffer comes from a descriptor_load" provenance the rank-1 tests use
    (the only provenance the gather pattern accepts post-fallback-removal),
    just at rank 2.  The descriptor block keeps its leading 1 (the gather
    contract: dim 0 of the block is fanned out one page at a time); the *two*
    index-grid dims become the two leading result dims, and the trailing block
    columns follow.

    Tests:
      test_gather_2d_indices_lowered          — 2-D grid gather → rank-3 result
      test_gather_2d_indices_two_indirect_axes — pin dim0/dim1 BOTH indirect
      test_gather_2d_indices_result_shape     — result = [*idx_grid, block_cols]
      test_scatter_2d_indices_lowered         — scatter mirror
      test_gather_2d_indices_3d_block         — 2-D grid × rank-3 block (rank-4 result)
    """

    @pattern("descriptor-gather-2d-indices", category="memory", example=[
        "# Gather a 2-D grid of pages — one page per (seq, head) index pair:",
        "idx_desc = tl.make_tensor_descriptor(idx_ptr, [SEQ, HEADS], [HEADS, 1],",
        "                                     block_shape=[8, 4])",
        "x_offsets = tl.descriptor_load(idx_desc, [0, 0])   # tensor<8x4xi32>",
        "data = tl.descriptor_gather(desc, x_offsets, y_offset)",
        "# → tensor<8 x 4 x BLOCK_COLS x f16>  (two indirect axes, then block cols)",
    ])
    def test_gather_2d_indices_lowered(self):
        """A rank-2 ``x_offsets`` (``tensor<8x4xi32>``) gathers an 8×4 grid of
        rows from a ``<1x64>`` block, producing ``tensor<8x4x64xf16>``.

        Both leading result dims (8, 4) are *indirect* — driven by the index
        grid — and the trailing 64 is the block-column extent.  Lowers to one
        ``ktdp.construct_indirect_access_tile`` over the 2-D index view plus a
        ``ktdp.load`` of the rank-3 result tile.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %M = arith.constant 1024 : i32        // full tensor rows
            %K = arith.constant 128 : i32         // full tensor cols
            %stride_row = arith.constant 128 : i64
            %stride_col = arith.constant 1 : i64
            // 2-D index buffer: 8x4 grid of row indices.
            %idx_rows = arith.constant 8 : i32
            %idx_cols = arith.constant 4 : i32
            %idx_srow = arith.constant 4 : i64    // index row stride
            %idx_scol = arith.constant 1 : i64
            %c0_i32   = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_rows, %idx_cols],
                            [%idx_srow, %idx_scol]
                : <i32>, <8x4xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32, %c0_i32]
                : !tt.tensordesc<8x4xi32> -> tensor<8x4xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                // result: 8×4 grid of gathered rows × 64 cols
                : (!tt.tensordesc<1x64xf16>, tensor<8x4xi32>, i32) -> tensor<8x4x64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        self.assert_absent("unrealized_conversion_cast")
        self.assert_has_region("ktdp.construct_indirect_access_tile")
        # Two memory views: the 2-D index grid + the base table.
        self.assert_count("ktdp.construct_memory_view", 2, cmp="eq")
        self.assert_result("ktdp.construct_memory_view", shape=[1024, 128],
                           elem_type="f16")
        # variables_space_set spans the full rank-3 result:
        # [0,7]×[0,3]×[0,63] = 3 dims, 6 constraints.
        self.assert_integer_set("ktdp.construct_indirect_access_tile",
                                "variables_space_set",
                                num_dims=3, num_symbols=0, num_constraints=6)
        # Gather load result = [*index_grid, block_cols] = [8, 4, 64].
        self.assert_result("ktdp.load", shape=[8, 4, 64], elem_type="f16")

    @pattern("descriptor-gather-2d-indices-subscripts", category="memory", example=[
        "# 2-D x_offsets → ONE indirect base dim with a 2-D index address:",
        "x_offsets = tl.descriptor_load(idx_desc, [0, 0])   # tensor<8x4xi32>",
        "data = tl.descriptor_gather(desc, x_offsets, y_offset)",
        "# lowers to: ind(idx[c_x0 + d0, c_x1 + d1]), (c_y + d2)",
        "# Contrast rank-1 x_offsets, where the index address is 1-D.",
    ])
    def test_gather_2d_indices_two_indirect_axes(self):
        """Pin that the indirect base dim carries a 2-D index address for a
        rank-2 ``x_offsets``.

        This is the load-bearing difference from the rank-1 N-D gather, where
        the indirect address is 1-D (see
        ``test_gather_3d_subscript_kinds_pin_offset_axis``).  ``x_offsets`` adds
        index-grid axes to the *address* of a single indirect base dim — NOT
        extra indirect base dims: the KTDP op encodes the K-D index lookup as
        one ``ind(...)`` subscript whose inner ``[...]`` lists K coordinates
        (the verifier requires the indirect map's result count to equal the
        index-view rank).  So a rank-2 ``x_offsets`` over a ``<1x64>`` block
        prints **two** subscripts — ``ind(idx[c_x0 + d0, c_x1 + d1])`` then the
        direct ``(c_y + d2)`` — with both index-grid coordinates inside the
        first.  A regression that dropped an index axis would shrink the inner
        address, so inspect it textually.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %M = arith.constant 1024 : i32
            %K = arith.constant 128 : i32
            %stride_row = arith.constant 128 : i64
            %stride_col = arith.constant 1 : i64
            %idx_rows = arith.constant 8 : i32
            %idx_cols = arith.constant 4 : i32
            %idx_srow = arith.constant 4 : i64
            %idx_scol = arith.constant 1 : i64
            %c0_i32   = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_rows, %idx_cols],
                            [%idx_srow, %idx_scol]
                : <i32>, <8x4xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32, %c0_i32]
                : !tt.tensordesc<8x4xi32> -> tensor<8x4xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x64xf16>, tensor<8x4xi32>, i32) -> tensor<8x4x64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_indirect_access_tile")
        subscripts = _parse_indirect_subscripts(str(self.mod))
        # base = source matrix memref<1024x128> → rank 2 → 2 base-dim subscripts
        # (one per base dim). The K=2 index grid lives in the *address* of the
        # single indirect base dim, not as extra base dims.
        assert len(subscripts) == 2, (
            f"2-D-index gather over a <1x64> block must have 2 subscripts (one "
            f"per base dim); got {len(subscripts)}: {subscripts}"
        )
        # Base dim 0: the single indirect axis, carrying a 2-D index address.
        assert subscripts[0].startswith("ind("), (
            f"base dim 0 must be indirect (the page axis); got `{subscripts[0]}`"
        )
        # The inner `idx[...]` address must list BOTH index-grid coordinates:
        # a top-level comma inside the indirect subscript's `[...]`. This is the
        # K=2 distinction from rank-1 gather (whose inner address is 1-D).
        inner = subscripts[0][subscripts[0].find("[") + 1:subscripts[0].rfind("]")]
        assert inner.count(",") == 1, (
            "rank-2 x_offsets: the indirect address must carry 2 coordinates "
            f"(`ind(idx[c_x0 + d0, c_x1 + d1])`); got inner `{inner}`"
        )
        assert " + " in inner, (
            "the indirect address must capture the x_offset anchors "
            f"(`c_x{{j}} + d_j`); got inner `{inner}`"
        )
        # Base dim 1: the direct y_offset column axis (moved past the K=2 grid).
        assert not subscripts[1].startswith("ind("), (
            f"base dim 1 (block columns) must be direct; got `{subscripts[1]}`"
        )
        assert " + " in subscripts[1], (
            "base dim 1 direct subscript must capture y_offset "
            f"(form `(<y_offset> + <iv>)`); got `{subscripts[1]}`"
        )

    @pytest.mark.parametrize("R,C,COLS", [(8, 4, 64), (16, 2, 32), (4, 8, 128)])
    def test_gather_2d_indices_result_shape(self, R, C, COLS):
        """Result shape is ``[*x_offsets.shape, block_cols]`` for any 2-D grid.

        Parametrised over a few (rows, cols, block_cols) triples to pin that
        the index grid's *both* dims flow through to the result, in order,
        ahead of the block columns.
        """
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {{
            %M = arith.constant 4096 : i32
            %K = arith.constant 256 : i32
            %stride_row = arith.constant 256 : i64
            %stride_col = arith.constant 1 : i64
            %idx_rows = arith.constant {R} : i32
            %idx_cols = arith.constant {C} : i32
            %idx_srow = arith.constant {C} : i64
            %idx_scol = arith.constant 1 : i64
            %c0_i32   = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_rows, %idx_cols],
                            [%idx_srow, %idx_scol]
                : <i32>, <{R}x{C}xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32, %c0_i32]
                : !tt.tensordesc<{R}x{C}xi32> -> tensor<{R}x{C}xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x{COLS}xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x{COLS}xf16>, tensor<{R}x{C}xi32>, i32)
                  -> tensor<{R}x{C}x{COLS}xf16>
            tt.return
          }}
        }}
        """)
        self.assert_absent("tt.descriptor_gather")
        self.assert_result("ktdp.load", shape=[R, C, COLS], elem_type="f16")

    @pattern("descriptor-scatter-2d-indices", category="memory", example=[
        "# Scatter a 2-D grid of pages — mirror of the 2-D-index gather:",
        "x_offsets = tl.descriptor_load(idx_desc, [0, 0])   # tensor<8x4xi32>",
        "tl.descriptor_scatter(desc, x_offsets, y_offset, value)  # value: <8x4x64xf16>",
    ])
    def test_scatter_2d_indices_lowered(self):
        """Scatter mirror of ``test_gather_2d_indices_lowered``.

        Same 2-D index grid drives an 8×4 grid of row writes; ``%src`` carries
        the ``tensor<8x4x64xf16>`` payload.  Lowers via
        ``ConvertDescriptorScatter`` to ``ktdp.store`` over the same
        two-indirect-axis access tile.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32,
                     %src: tensor<8x4x64xf16>) {
            %M = arith.constant 1024 : i32
            %K = arith.constant 128 : i32
            %stride_row = arith.constant 128 : i64
            %stride_col = arith.constant 1 : i64
            %idx_rows = arith.constant 8 : i32
            %idx_cols = arith.constant 4 : i32
            %idx_srow = arith.constant 4 : i64
            %idx_scol = arith.constant 1 : i64
            %c0_i32   = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_rows, %idx_cols],
                            [%idx_srow, %idx_scol]
                : <i32>, <8x4xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32, %c0_i32]
                : !tt.tensordesc<8x4xi32> -> tensor<8x4xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>
            tt.descriptor_scatter %desc[%x_offsets, %y_offset], %src
                : !tt.tensordesc<1x64xf16>, tensor<8x4xi32>, i32, tensor<8x4x64xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_indirect_access_tile", "ktdp.store")
        self.assert_absent("tt.descriptor_scatter")
        self.assert_absent("unrealized_conversion_cast")
        self.assert_has_region("ktdp.construct_indirect_access_tile")
        self.assert_count("ktdp.construct_memory_view", 2, cmp="eq")
        self.assert_integer_set("ktdp.construct_indirect_access_tile",
                                "variables_space_set",
                                num_dims=3, num_symbols=0, num_constraints=6)

    @pattern("descriptor-gather-2d-indices-3d-block", category="memory", example=[
        "# 2-D index grid combined with a rank-3 block (block_shape=[1, B, D]):",
        "x_offsets = tl.descriptor_load(idx_desc, [0, 0])   # tensor<8x4xi32>",
        "data = tl.descriptor_gather(desc, x_offsets, y_offset)",
        "# → tensor<8 x 4 x B x D x f16>  (2 indirect axes + block[1:] = 2 trailing)",
    ])
    def test_gather_2d_indices_3d_block(self):
        """Rank-2 ``x_offsets`` × rank-3 descriptor block → rank-4 result.

        Combines both relaxations: the 2-D index grid contributes dims 0–1
        (both indirect), ``y_offset`` lands on dim 2, and the descriptor's
        trailing inner dim (``block_shape[2]``) is a direct, no-offset dim 3.
        Pins that the two generalisations compose: K indirect axes (K=2) plus
        the existing rank-N trailing-dim handling.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32) {
            %P = arith.constant 1024 : i32     // NUM_BLOCKS
            %B = arith.constant 16   : i32     // in-block rows (y_offset axis)
            %D = arith.constant 128  : i32     // INNER_DIM (trailing, full extent)
            %s0 = arith.constant 2048 : i64
            %s1 = arith.constant 128  : i64
            %s2 = arith.constant 1    : i64
            %idx_rows = arith.constant 8 : i32
            %idx_cols = arith.constant 4 : i32
            %idx_srow = arith.constant 4 : i64
            %idx_scol = arith.constant 1 : i64
            %c0_i32   = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_rows, %idx_cols],
                            [%idx_srow, %idx_scol]
                : <i32>, <8x4xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32, %c0_i32]
                : !tt.tensordesc<8x4xi32> -> tensor<8x4xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
                : <f16>, <1x16x128xf16>
            %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                : (!tt.tensordesc<1x16x128xf16>, tensor<8x4xi32>, i32)
                  -> tensor<8x4x16x128xf16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_indirect_access_tile", "ktdp.load")
        self.assert_absent("tt.descriptor_gather")
        # Rank-4 result: 2 index-grid dims + 2 block dims.
        # variables_space_set: 4 dims × 2 = 8 constraints.
        self.assert_integer_set("ktdp.construct_indirect_access_tile",
                                "variables_space_set",
                                num_dims=4, num_symbols=0, num_constraints=8)
        self.assert_result("ktdp.load", shape=[8, 4, 16, 128], elem_type="f16")


# =========================================================================
# tt.descriptor_gather — N-D limits (kernel-author-facing rejections)
# =========================================================================

class TestDescriptorGatherNDLimits(LowerDescMemoryTester):
    """Negative tests highlighting kernel-author-facing limits of N-D gather.

    The relaxed verifier (``lib/Dialect/Triton/IR/Ops.cpp``) keeps two
    invariants — the gather's *load-bearing* contract — that an author
    coming from rank-2 gather might miss when writing an N-D kernel:

    * ``block.shape[0] == 1`` at *every* rank.  The leading-1 is the
      gather contract: dim 0 is fanned out by the index buffer one
      page at a time, so the block can carry exactly one page.
    * ``indices`` is a 1-D ``tensor<Nxi32>``.  Multi-D index tensors
      are not supported — the index buffer addresses dim 0, period.

    These tests pin both rules at rank 3, where the loop body of the
    new per-dim trailing-shape check actually runs.  The existing
    rank-2 ``test_gather_block_dim0_not_one_fails`` covers the rank-2
    path; this class extends the same invariants to the N-D
    extension.

    Diagnostics fire at MLIR parse time (verifiers run during
    ``ir.parse_mlir_module``), so the raised error is
    ``"Parse MLIR file failed"`` rather than ``"PassManager::run failed"``.
    """

    @pattern("descriptor-gather-nd-block-dim0", category="memory", negative=True,
             example=[
                 "# REJECTED: leading dim of block_shape is not 1.",
                 "# The N-D relaxation widened the rank rule but kept the",
                 "# leading-1 rule — dim 0 specifically must be 1.",
                 "desc = tl.make_tensor_descriptor(ptr, shape=[P, B, D],",
                 "                                 strides=[B*D, D, 1],",
                 "                                 block_shape=[2, 16, 128])  # leading 2 — REJECTED",
                 "data = tl.descriptor_gather(desc, x_offsets, y_offset)",
                 "# 'descriptor block must have exactly 1 row'",
                 "# Workaround: keep block_shape[0]=1; pair blocks via",
                 "# block_shape=[1, 2*16, 128] with a paired index buffer,",
                 "# OR issue two gathers and concatenate.",
             ])
    def test_gather_3d_block_dim0_not_one_fails(self, capfd):
        """Rank-3 block ``<2x16x128>``: the leading-1 rule still applies.

        The N-D relaxation widened the rank rule but kept the
        leading-1 rule.  At rank 2 the same rule is pinned by
        ``test_gather_block_dim0_not_one_fails``; this rank-3 mirror
        exists because a kernel author writing their first N-D gather
        is likely to assume the relaxation also widened *which* dim
        must be 1 (e.g. "any dim that's 1 is fine") — it didn't.  Dim
        0 specifically.
        """
        with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
            self.run("""
            module {
              tt.func @k(%ptr: !tt.ptr<f16>,
                         %x_offsets: tensor<32xi32>, %y_offset: i32) {
                %P = arith.constant 1024 : i32
                %B = arith.constant 16   : i32
                %D = arith.constant 128  : i32
                %s0 = arith.constant 2048 : i64
                %s1 = arith.constant 128  : i64
                %s2 = arith.constant 1    : i64
                %desc = tt.make_tensor_descriptor %ptr, [%P, %B, %D], [%s0, %s1, %s2]
                    : <f16>, <2x16x128xf16>
                %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
                    : (!tt.tensordesc<2x16x128xf16>, tensor<32xi32>, i32) -> tensor<32x16x128xf16>
                tt.return
              }
            }
            """)
        self.assert_stderr(capfd, "descriptor block must have exactly 1 row")


# =========================================================================
# tt.descriptor_scatter → construct_memory_view + construct_indirect_access_tile + store
# =========================================================================

class TestDescriptorScatter(LowerDescMemoryTester):
    # tt.descriptor_scatter → ktdp.store via indirect access tile.
    # Scatter writes non-contiguous rows — the mirror of gather.
    # Same size constraint: block must have exactly 1 row; the number of
    # scattered rows comes from x_offsets, not the block shape.
    #
    # Example: a 1024×128 tensor, scattering 32 non-contiguous rows,
    # writing a 64-wide column slice starting at y_offset:
    #   %desc = tt.make_tensor_descriptor %ptr, [%M=1024, %K=128], [128, 1]
    #               : <f16>, <1x64xf16>          // block: 1 row × 64 cols
    #   tt.descriptor_scatter %desc[%x_offsets, %y_offset], %data
    #               : ..., tensor<32xi32>, i32, tensor<32x64xf16>
    #   → memory view covers the full 1024×128 tensor (memref<1024x128xf16>)
    #   → x_offsets (tensor<32xi32>) names the 32 row indices to write
    #   → y_offset is the starting column for the 64-wide tile within each row
    #   → %data shape [32, 64] = [len(x_offsets), block_cols]
    #
    # tt.make_tensor_descriptor: shape operands are i32, strides are i64.
    #
    # test_scatter_2d[M,K]         — row-scatter, parametrized over (M,K) pairs
    # test_scatter_structural      — affine attrs and region present
    # test_multi_descriptor        — two independent descriptors in one function

    @pytest.mark.parametrize("M,K", [(512, 128), (1024, 128), (2048, 256)])
    def test_scatter_2d(self, M, K):
        # Row-scatter: x_offsets (tensor<32xi32>) names 32 non-contiguous rows.
        # y_offset is the starting column for the 64-wide tile within each row.
        # %data shape [32, 64] = [len(x_offsets), block_cols] is what gets written.
        #
        # x_offsets is staged via descriptor_load — same calling convention
        # as gather (and the only legal provenance after the fallback removal).
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32,              // starting column for the 64-wide tile
                     %data: tensor<32x64xf16>) {{ // data to scatter
            %M = arith.constant {M} : i32         // full tensor rows — compile-time
            %K = arith.constant {K} : i32         // full tensor cols — compile-time
            %stride_row = arith.constant {K} : i64  // row stride = K (row-major)
            %stride_col = arith.constant 1 : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1 : i64
            %c0_i32     = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>               // block: 1 row × 64 cols (scatter constraint)
            tt.descriptor_scatter %desc[%x_offsets, %y_offset], %data
                : !tt.tensordesc<1x64xf16>, tensor<32xi32>, i32, tensor<32x64xf16>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_indirect_access_tile", "ktdp.store")
        self.assert_absent("tt.descriptor_scatter")
        # No cast: trace path reuses the index descriptor's memory view.
        self.assert_absent("unrealized_conversion_cast")
        # Two memory views: index buffer + base table.  Regression to the
        # fallback would build a third.
        self.assert_count("ktdp.construct_memory_view", 2, cmp="eq")
        # Base-table memory view: shape [M, K] (assert_result matches by shape=
        # so it picks the 2-D view, not the 1-D index view of shape [32]).
        self.assert_result("ktdp.construct_memory_view", shape=[M, K], elem_type="f16")
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=2, num_symbols=0, num_constraints=4)

    def test_scatter_structural(self):
        # ktdp.construct_indirect_access_tile carries:
        #   - a region with per-row subscript affine maps
        #   - variables_space_set / variables_space_order attrs that describe
        #     the iteration space over x_offsets
        # The memory view must cover the full tensor, not just the block row.
        #
        # x_offsets is staged via descriptor_load (the only legal provenance
        # post-fallback-removal); the assertions here pin the *base* table
        # memory view by shape= so they remain unambiguous despite the second
        # 1-D index-buffer view that descriptor_load lowering creates.
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>,
                     %idx_ptr: !tt.ptr<i32>,
                     %y_offset: i32,
                     %data: tensor<32x64xf16>) {
            %M = arith.constant 1024 : i32
            %K = arith.constant 128 : i32         // full tensor cols
            %stride_row = arith.constant 128 : i64
            %stride_col = arith.constant 1 : i64
            %idx_count  = arith.constant 32 : i32
            %idx_stride = arith.constant 1 : i64
            %c0_i32     = arith.constant 0 : i32

            %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%idx_count], [%idx_stride]
                : <i32>, <32xi32>
            %x_offsets = tt.descriptor_load %idx_desc[%c0_i32]
                : !tt.tensordesc<32xi32> -> tensor<32xi32>

            %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%stride_row, %stride_col]
                : <f16>, <1x64xf16>
            tt.descriptor_scatter %desc[%x_offsets, %y_offset], %data
                : !tt.tensordesc<1x64xf16>, tensor<32xi32>, i32, tensor<32x64xf16>
            tt.return
          }
        }
        """)
        self.assert_has_region("ktdp.construct_indirect_access_tile")
        # Full tensor memory view: shape [1024, 128].  assert_result matches by
        # shape= so it picks the 2-D base view, not the 1-D index view.
        self.assert_result("ktdp.construct_memory_view", shape=[1024, 128], elem_type="f16")
        # 2 dims, 0 symbols, 4 constraints describes the base view.  The 1-D
        # index view has only 2 constraints, so assert_integer_set's "any
        # match" semantics pick the base view.
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=2, num_symbols=0, num_constraints=4)
        # variables_space_set: [0,31]×[0,63] = [len(x_offsets)−1, block_cols−1]
        self.assert_integer_set("ktdp.construct_indirect_access_tile", "variables_space_set",
                                num_dims=2, num_symbols=0, num_constraints=4)

    def test_multi_descriptor(self):
        # Two independent tt.make_tensor_descriptor ops in one function.
        # Each must produce its own ktdp.construct_memory_view and access tile;
        # both descriptor ops must be erased after lowering.
        self.run("""
        module {
          tt.func @k(%ptr_a: !tt.ptr<f16>, %ptr_b: !tt.ptr<f16>,
                     %m: i32, %k: i32) {
            %M = arith.constant 1024 : i32
            %K = arith.constant 64 : i32
            %stride_row = arith.constant 64 : i64
            %stride_col = arith.constant 1 : i64
            // Two independent descriptors pointing to different tensors
            %desc_a = tt.make_tensor_descriptor %ptr_a, [%M, %K],
                          [%stride_row, %stride_col] : <f16>, <32x64xf16>
            %desc_b = tt.make_tensor_descriptor %ptr_b, [%M, %K],
                          [%stride_row, %stride_col] : <f16>, <32x64xf16>
            %data = tt.descriptor_load %desc_a[%m, %k]
                : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16>
            tt.descriptor_store %desc_b[%m, %k], %data
                : !tt.tensordesc<32x64xf16>, tensor<32x64xf16>
            tt.return
          }
        }
        """)
        self.assert_absent("tt.descriptor_load", "tt.descriptor_store",
                           "tt.make_tensor_descriptor")
        # One memory view and one access tile per descriptor
        self.assert_count("ktdp.construct_memory_view", 2, cmp="eq")
        self.assert_count("ktdp.construct_access_tile", 2, cmp="eq")


# =========================================================================
# Descriptor placement: view inherits the descriptor's insertion point.
# =========================================================================

class TestDescriptorPlacement(LowerDescMemoryTester):
    """``ktdp.construct_memory_view`` is emitted at the descriptor's site.

    ``LowerDescriptorMemory`` rewrites every ``tt.make_tensor_descriptor``
    in place — the resulting ``ktdp.construct_memory_view`` lands in
    the same region as the descriptor it replaces, regardless of where
    the access op (load / store / gather / scatter) that consumes it
    lives.  Three cases are covered:

    * **Top-level descriptor.** Descriptor at function top, access op
      inside ``scf.for`` → view stays at function top.  Built once
      and reused across every loop iteration.

    * **Nested descriptor.** Descriptor *inside* ``scf.for`` → view
      is emitted inside the loop body, alongside the access op.
      Rebuilt once per iteration.  This pass does not try to move
      the view out of the loop; that is left to the user (or a
      later optimization pass).

    * **Conditional descriptor.** Descriptor inside an ``scf.if``
      then-branch → view is emitted inside the same branch, so it
      is only built when the branch is taken.
    """

    @pattern("descriptor-placement-top-level", category="memory", example=[
        "desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1],",
        "                                 block_shape=[BLOCK])  # at function top",
        "for off in range(0, N, BLOCK):",
        "    tile = tl.descriptor_load(desc, [off])  # inside the loop",
    ])
    def test_top_level_descriptor_view_outside_loop(self):
        """Descriptor at function top, load inside ``scf.for`` → view at function top.

        Covers the common case where the user defines a descriptor
        once at the start of the kernel and then loads tiles from it
        inside a loop.  Checks that:

        * exactly one ``ktdp.construct_memory_view`` is emitted (not
          one per loop iteration);
        * its immediate parent in the IR is ``tt.func``, not
          ``scf.for`` — i.e. the view sits at function top, so it is
          built once and reused across every iteration;
        * the per-tile ops (``ktdp.construct_access_tile``,
          ``ktdp.load``) do live inside the loop, since they depend
          on the loop variable.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>) {
            %N = arith.constant 1024 : i32
            %stride = arith.constant 1 : i64
            // Descriptor at function top: view should land at function
            // top too, so it's built once and reused every iteration.
            %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                : <f16>, <64xf16>

            %lo = arith.constant 0 : index
            %hi = arith.constant 1024 : index
            %step = arith.constant 64 : index
            scf.for %iv = %lo to %hi step %step {
              %off = arith.index_cast %iv : index to i32
              %data = tt.descriptor_load %desc[%off]
                  : !tt.tensordesc<64xf16> -> tensor<64xf16>
            }
            tt.return
          }
        }
        """)
        # Exactly one view, and it is at function top — not in the loop.
        self.assert_count("ktdp.construct_memory_view", 1, cmp="eq")
        self.assert_present("ktdp.construct_memory_view", parent="tt.func")
        self.assert_count("ktdp.construct_memory_view", 0, cmp="eq",
                          parent="scf.for")
        # The access tile is per-iteration (driven by %iv) so it lives
        # inside the loop, alongside ktdp.load. Pin this so a regression
        # that hoists the access tile out of the loop also flags here.
        self.assert_present("ktdp.construct_access_tile", parent="scf.for")
        self.assert_present("ktdp.load", parent="scf.for")

    @pattern("descriptor-placement-nested", category="memory", example=[
        "for i in range(0, N, BLOCK):",
        "    desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1],",
        "                                     block_shape=[BLOCK])  # inside loop",
        "    tile = tl.descriptor_load(desc, [i])",
    ])
    def test_nested_descriptor_view_inside_loop(self):
        """Descriptor written inside ``scf.for`` is lowered correctly.

        Covers the case where ``tt.make_tensor_descriptor`` appears
        inside a loop body (instead of at function top).  Checks that:

        * ``tt.make_tensor_descriptor`` and ``tt.descriptor_load`` are
          both removed from the IR;
        * the corresponding ``ktdp.construct_memory_view``,
          ``ktdp.construct_access_tile``, and ``ktdp.load`` are
          emitted by the pass;
        * the new ``ktdp.construct_memory_view`` is placed inside the
          ``scf.for`` body — same region as the descriptor it replaces
          — and not lifted out to function top.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>) {
            %N = arith.constant 1024 : i32
            %stride = arith.constant 1 : i64

            %lo = arith.constant 0 : index
            %hi = arith.constant 1024 : index
            %step = arith.constant 64 : index
            scf.for %iv = %lo to %hi step %step {
              // Descriptor *inside* the loop: view should land inside too.
              %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                  : <f16>, <64xf16>
              %off = arith.index_cast %iv : index to i32
              %data = tt.descriptor_load %desc[%off]
                  : !tt.tensordesc<64xf16> -> tensor<64xf16>
            }
            tt.return
          }
        }
        """)
        # Lowering succeeds end-to-end: descriptor erased, ktdp ops emitted.
        self.assert_absent("tt.make_tensor_descriptor", "tt.descriptor_load")
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_access_tile", "ktdp.load")
        # The view is inside the loop body, not lifted to the function top.
        self.assert_present("ktdp.construct_memory_view", parent="scf.for")
        self.assert_count("ktdp.construct_memory_view", 0, cmp="eq",
                          parent="tt.func")

    @pattern("descriptor-placement-conditional", category="memory", example=[
        "if cond:",
        "    desc = tl.make_tensor_descriptor(ptr, shape=[N], strides=[1],",
        "                                     block_shape=[BLOCK])  # inside if",
        "    tile = tl.descriptor_load(desc, [off])",
    ])
    def test_descriptor_inside_scf_if_view_inside_branch(self):
        """Descriptor written inside ``scf.if`` is lowered correctly.

        Covers the case where ``tt.make_tensor_descriptor`` appears
        inside the then-branch of an ``scf.if`` (instead of at function
        top).  Checks that:

        * ``tt.make_tensor_descriptor`` and ``tt.descriptor_load`` are
          both removed from the IR;
        * the corresponding ``ktdp.construct_memory_view``,
          ``ktdp.construct_access_tile``, and ``ktdp.load`` are
          emitted by the pass;
        * the new ``ktdp.construct_memory_view`` is placed inside the
          ``scf.if`` branch — same region as the descriptor it
          replaces — and not lifted out to function top.  Zero
          ``ktdp.construct_memory_view`` ops are emitted directly under
          ``tt.func``.

        ``%cond`` is a function argument; the test never binds it to a
        concrete value because this is a structural check on where the
        view lands in the IR, not an execution test.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %cond: i1, %off: i32) {
            %N = arith.constant 1024 : i32
            %stride = arith.constant 1 : i64

            scf.if %cond {
              // Descriptor *inside* the then-branch: view should land
              // inside scf.if too, not at function top.
              %desc = tt.make_tensor_descriptor %ptr, [%N], [%stride]
                  : <f16>, <64xf16>
              %data = tt.descriptor_load %desc[%off]
                  : !tt.tensordesc<64xf16> -> tensor<64xf16>
            }
            tt.return
          }
        }
        """)
        # Lowering succeeds end-to-end: descriptor erased, ktdp ops emitted.
        self.assert_absent("tt.make_tensor_descriptor", "tt.descriptor_load")
        self.assert_present("ktdp.construct_memory_view",
                            "ktdp.construct_access_tile", "ktdp.load")
        # The view is inside the conditional branch, not at function top.
        self.assert_present("ktdp.construct_memory_view", parent="scf.if")
        self.assert_count("ktdp.construct_memory_view", 0, cmp="eq",
                          parent="tt.func")


# =========================================================================
# tt.addptr feeding tt.make_tensor_descriptor — NOT YET LOWERED
# =========================================================================

class TestAddptrIntoDescriptor(LowerDescMemoryTester):
    """Per-iteration descriptors built from ``ptr + offset``: not yet supported.

    Note: batched matmul itself *does* compile today via 3D descriptors
    whose base is the raw buffer pointer (``fixtures/matmul/kernel.py::
    bmm_matmul_kernel``).  The gap here is the *per-batch-offset* idiom
    where a ``tt.addptr`` result feeds ``tt.make_tensor_descriptor`` —
    e.g. ``bmm_matmul_kernel_addptr`` which computes
    ``a_ptr + b_idx * stride_batch`` as the descriptor base. In TTIR this
    becomes ``tt.addptr`` feeding ``tt.make_tensor_descriptor``.

    Current state: ``LowerDescriptorMemory`` lowers the descriptor ops
    via ``getBasePtrAsIndex``, which casts the base ``!tt.ptr`` operand
    to ``index``. The ``tt.addptr`` that computes that base is left
    intact. When ``ConvertFunctions`` subsequently rewrites function
    signatures (``!tt.ptr`` args → ``index``), ``tt.addptr``'s operand
    becomes ``index`` but its verifier still demands ``!tt.ptr`` —
    verification fails.

    Fix plan: ``LowerDescriptorMemory`` should fold ``tt.addptr`` into
    the base/offset it passes to ``construct_memory_view`` (or
    canonicalize the pointer arithmetic to index arithmetic before
    ``ConvertFunctions`` runs). Once fixed, delete this test in favor
    of a positive descriptor-with-offset test.
    """

    def _build_passes(self, pm):
        # Failure surfaces in ConvertFunctions, not LowerDescriptorMemory —
        # we need both passes in the pipeline to reproduce it. Mirrors
        # the two-pass harness used by DistributeWork tests.
        from triton._C.libtriton import spyre
        spyre.passes.ttir_to_ktdp.add_lower_descriptor_memory(pm)
        spyre.passes.ttir_to_ktdp.add_convert_functions(pm)

    @pattern("descriptor-offset-base", category="memory", negative=True, example=[
        "# NOT supported: tt.addptr result as descriptor base (e.g. batched matmul)",
        "base = a_ptr + b_idx * stride_batch   # tt.addptr",
        "desc = tl.make_tensor_descriptor(base, shape=[M, K], strides=[K, 1],",
        "                                 block_shape=[BLOCK_M, BLOCK_K])",
    ])
    def test_addptr_into_descriptor_fails(self, capfd):
        """`tt.addptr` result feeding `tt.make_tensor_descriptor`.

        When a `tt.addptr` result is the base pointer for a tensor
        descriptor, `ConvertFunctions` rewrites the `!tt.ptr` argument
        to `index` before `LowerDescriptorMemory` can fold the pointer
        arithmetic, and the op's verifier then rejects the now-illegal
        operand type. This is the underlying reason batched matmul is
        currently disabled: each batch step wants to offset the base
        pointer before constructing the descriptor.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%a_ptr: !tt.ptr<f32>, %offset: i32) {
                %c0_i32 = arith.constant 0 : i32
                %M = arith.constant 1024 : i32
                %K = arith.constant 64 : i32
                %stride_row = arith.constant 64 : i64
                %stride_col = arith.constant 1 : i64
                // Per-batch base: a_ptr + offset. The !tt.ptr result of
                // tt.addptr feeds tt.make_tensor_descriptor — this is
                // the unsupported shape.
                %base = tt.addptr %a_ptr, %offset : !tt.ptr<f32>, i32
                %desc = tt.make_tensor_descriptor %base, [%M, %K],
                            [%stride_row, %stride_col]
                          : <f32>, <16x16xf32>
                %data = tt.descriptor_load %desc[%c0_i32, %c0_i32]
                          : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
                tt.return
              }
            }
            """)
        # Pin the exact verifier diagnostic: tt.addptr's !tt.ptr operand
        # was rewritten to index by ConvertFunctions, but the op's
        # verifier still demands ptr. A drift in the error text or in
        # which pass raises it will flag here.
        self.assert_stderr(capfd,
                           "tt.addptr",
                           "must be ptr",
                           "got 'index'")
