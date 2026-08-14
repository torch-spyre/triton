#!/usr/bin/env python3
"""
Unit tests for the LowerScalarLoad conversion pattern.

Organization: one test class (TestScalarLoad) covering the scalar `tt.load`
lowering — pointer operand `!tt.ptr<ElemT>`, scalar result `ElemT`, as
opposed to tensor-of-pointers `tt.load` (out of scope for this pass; see
`[LowerPointerChainMemory]` in Passes.td).

Each positive test asserts the rank-0 chain this pass emits:
    ktdp.construct_memory_view -> ktdp.construct_access_tile -> ktdp.load
    -> tensor.extract
For masked loads, the mask must be a compile-time constant (a materialized
`arith.constant` i1) — Spyre has no runtime control-flow divergence, so a
mask that might depend on runtime data is refused with a diagnostic rather
than lowered to a runtime branch.
"""

import pytest
from conftest import SinglePassTester
from utils_pattern import pattern


class LowerScalarLoadTester(SinglePassTester):
    """Shared base for all LowerScalarLoad pattern tests."""
    PASS = "add_lower_scalar_load"


# =========================================================================
# tt.load (scalar) -> construct_memory_view + construct_access_tile + load
#                     + tensor.extract
# =========================================================================

class TestScalarLoad(LowerScalarLoadTester):
    # tt.load %ptr : !tt.ptr<ElemT> -> ElemT lowers to a rank-0 ktdp read.
    # No tile/stride/shape logic — every intermediate op is rank 0, the
    # structural floor of the ktdp.load contract.
    #
    # test_bare_pointer                       — no addptr chain, no mask
    # test_addptr_chain_folded                — scalar tt.addptr offsets folded, no scaling
    # test_addptr_chain_multi_hop             — two chained tt.addptr ops fold into one chain
    # test_elem_type[i32/i16/f16/f32/bf16]    — pass is not element-type-specific
    # test_masked_constant_true                — constant-true mask, unconditional read
    # test_masked_constant_false_with_other    — constant-false mask, yields `other`, no read
    # test_masked_constant_false_without_other — constant-false mask, yields zero, no read
    # test_masked_runtime_rejected              — non-constant mask, pass refuses to lower
    # test_masked_cmpi_rejected                 — cmpi of two constants, still refused
    # test_tensor_of_pointers_untouched         — out of scope, stays legal
    # test_ptr_to_ptr_untouched                 — non-numeric pointee, stays legal

    @pattern("scalar-load-bare", category="memory", example=[
        "x = tl.load(ptr)  # ptr: tl.pointer_type(tl.float16), scalar load",
    ])
    def test_bare_pointer(self):
        """Load a scalar through a bare pointer (no addptr chain, no mask).

        The pointer is cast straight to `index` (via `getBasePtrAsIndex`, an
        `unrealized_conversion_cast` that survives until `ConvertFunctions`)
        and used directly as the rank-0 memory view's offset — no
        `arith.addi` folding is needed since there's no `tt.addptr` chain.

        The loaded value is stored back out via `tt.store` (left untouched
        by this pass — raw scalar stores are out of scope), matching how a
        scalar load is typically consumed in a real kernel.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
            %v = tt.load %ptr : !tt.ptr<f16>
            tt.store %out, %v : !tt.ptr<f16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view",
                             "ktdp.construct_access_tile",
                             "ktdp.load", "tensor.extract")
        self.assert_absent("tt.load")
        # Rank 0: memref<f16> / tensor<f16>, no dims.
        self.assert_result_type("ktdp.construct_memory_view", "memref<f16>")
        self.assert_result("ktdp.load", shape=[], elem_type="f16")
        # A rank-0 IntegerSet still needs one constraint to be constructible
        # (IntegerSet::get derives its context from constraints[0], so a
        # truly empty constraint list isn't valid) — the pass emits a
        # trivially-true `0 >= 0` constraint as that placeholder.
        self.assert_integer_set("ktdp.construct_memory_view", "coordinate_set",
                                num_dims=0, num_symbols=0, num_constraints=1)
        self.assert_integer_set("ktdp.construct_access_tile", "access_tile_set",
                                num_dims=0, num_symbols=0, num_constraints=1)
        # Base pointer -> index via unrealized_conversion_cast, consumed
        # directly by construct_memory_view (no addptr chain to fold).
        self.assert_operand("ktdp.construct_memory_view", 0,
                            defined_by="builtin.unrealized_conversion_cast",
                            type_substr="index")

    @pattern("scalar-load-addptr-chain", category="memory", example=[
        "p = ptr + idx        # tt.addptr, offset in elements — no scaling",
        "x = tl.load(p)        # scalar load through the shifted pointer",
    ])
    def test_addptr_chain_folded(self):
        """A scalar `tt.addptr` chain feeding the load is folded into a
        single `index` offset with plain `arith.addi` — no element-size
        scaling (striding is the kernel author's responsibility).

        The `tt.addptr` op itself becomes dead once the fold consumes it
        and the `tt.load` is replaced; `cleanupDeadOps` sweeps it at the
        end of the pass, so it must be absent from the result.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %idx: i32, %out: !tt.ptr<f16>) {
            %p = tt.addptr %ptr, %idx : !tt.ptr<f16>, i32
            %v = tt.load %p : !tt.ptr<f16>
            tt.store %out, %v : !tt.ptr<f16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.load")
        self.assert_absent("tt.load", "tt.addptr")
        # Offset folded via arith.index_cast (i32 -> index) + arith.addi —
        # no multiplication anywhere (no element-size scaling).
        self.assert_present("arith.index_cast", "arith.addi")
        self.assert_absent("arith.muli")
        self.assert_operand("ktdp.construct_memory_view", 0,
                            defined_by="arith.addi", type_substr="index")

    @pattern("scalar-load-addptr-chain-multi", category="memory", example=[
        "p = ptr + a         # tt.addptr, offset in elements",
        "p = p + b            # a second tt.addptr on the shifted pointer",
        "x = tl.load(p)        # scalar load through the twice-shifted pointer",
    ])
    def test_addptr_chain_multi_hop(self):
        """`resolveScalarAddress` loops over an arbitrary-length `tt.addptr`
        chain, not just a single hop: two chained `tt.addptr` ops must fold
        into one `arith.addi` chain (two adds), with both `tt.addptr` ops
        swept as dead by `cleanupDeadOps`.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %a: i32, %b: i32, %out: !tt.ptr<f16>) {
            %p0 = tt.addptr %ptr, %a : !tt.ptr<f16>, i32
            %p1 = tt.addptr %p0, %b : !tt.ptr<f16>, i32
            %v = tt.load %p1 : !tt.ptr<f16>
            tt.store %out, %v : !tt.ptr<f16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.load")
        self.assert_absent("tt.load", "tt.addptr")
        self.assert_present("arith.index_cast", "arith.addi")
        self.assert_absent("arith.muli")
        self.assert_count("arith.addi", 2, cmp="eq")
        self.assert_operand("ktdp.construct_memory_view", 0,
                            defined_by="arith.addi", type_substr="index")

    @pytest.mark.parametrize("elem", ["i32", "i16", "f16", "f32", "bf16"])
    def test_elem_type(self, elem):
        """The pass is not element-type-specific: integer and float types
        of various widths all lower to the same rank-0 chain, just with a
        different element type.
        """
        self.run(f"""
        module {{
          tt.func @k(%ptr: !tt.ptr<{elem}>, %out: !tt.ptr<{elem}>) {{
            %v = tt.load %ptr : !tt.ptr<{elem}>
            tt.store %out, %v : !tt.ptr<{elem}>
            tt.return
          }}
        }}
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.load")
        self.assert_absent("tt.load")
        self.assert_result("ktdp.construct_memory_view", shape=[], elem_type=elem)
        self.assert_result("ktdp.load", shape=[], elem_type=elem)

    @pattern("scalar-load-masked-constant-true", category="memory", example=[
        "x = tl.load(ptr, mask=True)  # compile-time-true guard, unconditional read",
    ])
    def test_masked_constant_true(self):
        """A masked scalar load whose mask is a materialized `arith.constant
        true` drops straight through to the unconditional read — no runtime
        branch is ever emitted, since Spyre has no runtime control-flow
        divergence to lower one to.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %other: f16, %out: !tt.ptr<f16>) {
            %mask = arith.constant true
            %v = tt.load %ptr, %mask, %other : !tt.ptr<f16>
            tt.store %out, %v : !tt.ptr<f16>
            tt.return
          }
        }
        """)
        self.assert_present("ktdp.construct_memory_view", "ktdp.load")
        self.assert_absent("tt.load", "scf.if")

    @pattern("scalar-load-masked-constant-false-other", category="memory", example=[
        "x = tl.load(ptr, mask=False, other=0.0)  # compile-time-false guard, explicit fallback",
    ])
    def test_masked_constant_false_with_other(self):
        """A masked scalar load whose mask is a materialized `arith.constant
        false`, with an explicit `other`, performs no read at all — the op
        is replaced directly by `other`.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %other: f16, %out: !tt.ptr<f16>) {
            %mask = arith.constant false
            %v = tt.load %ptr, %mask, %other : !tt.ptr<f16>
            tt.store %out, %v : !tt.ptr<f16>
            tt.return
          }
        }
        """)
        self.assert_absent("ktdp.load", "scf.if", "tt.load")
        # No zero fallback constant is materialized — %other is used
        # directly. The one surviving `arith.constant` is the now-dead mask
        # value (%mask), left for the pipeline's later canonicalize/CSE
        # rather than swept here — see LowerScalarLoad.cpp's narrowed cleanup.
        self.assert_count("arith.constant", 1, cmp="eq")

    @pattern("scalar-load-masked-constant-false-no-other", category="memory", example=[
        "x = tl.load(ptr, mask=False)  # compile-time-false guard, implicit zero fallback",
    ])
    def test_masked_constant_false_without_other(self):
        """A masked scalar load whose mask is a materialized `arith.constant
        false`, with no `other`, performs no read and yields a materialized
        zero of the element type instead.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
            %mask = arith.constant false
            %v = tt.load %ptr, %mask : !tt.ptr<f16>
            tt.store %out, %v : !tt.ptr<f16>
            tt.return
          }
        }
        """)
        self.assert_absent("ktdp.load", "scf.if", "tt.load")
        self.assert_present("arith.constant")
        self.assert_operand("tt.store", 1, defined_by="arith.constant")

    @pattern("scalar-load-masked-runtime", category="memory", negative=True, example=[
        "# NOT supported: mask depends on a runtime value.",
        "# Spyre has no runtime control-flow divergence, so a masked",
        "# scalar load's mask must be a compile-time constant.",
        "mask = idx < n_elements  # REJECTED — not a materialized constant",
        "x = tl.load(ptr, mask=mask, other=0.0)",
    ])
    def test_masked_runtime_rejected(self, capfd):
        """A mask that isn't a materialized `arith.constant` — here a
        function argument, but the same refusal applies to e.g. a `cmpi` of
        two constants, which this pass deliberately does not fold — cannot
        be lowered: Spyre has no user-programmable control-flow divergence,
        so there is no way to guard the read on a runtime-dependent value.
        The pass refuses to lower it and fails with a diagnostic, mirroring
        the descriptor path's `test_descriptor_from_arg_fails`.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%ptr: !tt.ptr<f16>, %mask: i1, %other: f16, %out: !tt.ptr<f16>) {
                %v = tt.load %ptr, %mask, %other : !tt.ptr<f16>
                tt.store %out, %v : !tt.ptr<f16>
                tt.return
              }
            }
            """)
        self.assert_stderr(capfd, "mask must be a compile-time constant")

    @pattern("scalar-load-masked-runtime", category="memory", negative=True, example=[
        "# NOT supported: a comparison of two constants is not folded —",
        "# only a literal materialized arith.constant counts as a",
        "# compile-time-constant mask.",
        "mask = 3 < 5  # REJECTED even though both operands are constants",
        "x = tl.load(ptr, mask=mask, other=0.0)",
    ])
    def test_masked_cmpi_rejected(self, capfd):
        """A mask built from `arith.cmpi` of two constants is *not* folded
        by `getConstantMask` — only a literal `arith.constant` counts. This
        pins the "we don't fold comparisons, only literal constants"
        behavior documented on `getConstantMask`, distinct from the
        function-argument case covered by `test_masked_runtime_rejected`.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%ptr: !tt.ptr<f16>, %other: f16, %out: !tt.ptr<f16>) {
                %a = arith.constant 3 : i32
                %b = arith.constant 5 : i32
                %mask = arith.cmpi slt, %a, %b : i32
                %v = tt.load %ptr, %mask, %other : !tt.ptr<f16>
                tt.store %out, %v : !tt.ptr<f16>
                tt.return
              }
            }
            """)
        self.assert_stderr(capfd, "mask must be a compile-time constant")

    @pattern("scalar-load-tensor-of-ptrs", category="memory", negative=True, example=[
        "# NOT handled by this pass: pointer operand is a tensor of",
        "# pointers, not a bare scalar !tt.ptr — that path belongs to",
        "# the separate, still-unimplemented [LowerPointerChainMemory].",
        "x = tl.load(ptr + tl.arange(0, BLOCK))  # tensor-of-pointers load",
    ])
    def test_tensor_of_pointers_untouched(self):
        """A tensor-of-pointers `tt.load` (pointer operand shaped as a
        tensor of `!tt.ptr<ElemT>`, not a bare scalar pointer) is out of
        scope for this pass and must remain legal/untouched — the dynamic
        legality predicate must not fire for it.
        """
        self.run("""
        module {
          tt.func @k(%ptrs: tensor<4x!tt.ptr<f16>>) {
            %v = tt.load %ptrs : tensor<4x!tt.ptr<f16>>
            tt.store %ptrs, %v : tensor<4x!tt.ptr<f16>>
            tt.return
          }
        }
        """)
        self.assert_present("tt.load")
        self.assert_absent("ktdp.construct_memory_view", "ktdp.construct_access_tile")

    def test_ptr_to_ptr_untouched(self):
        """A pointer-to-pointer `tt.load` (`!tt.ptr<!tt.ptr<f16>>` — valid
        Triton IR, since `PointerType::verify` only rejects tensor
        pointees, not pointer pointees) must be left legal/untouched, not
        matched by `isScalarPtr`. Only integer/float pointees are
        lowerable: the rank-0 `memref`/`tensor` chain requires a
        `MemRefElementTypeInterface` element type, and the zero-fallback
        path requires `getZeroAttr` to have a case for it — neither holds
        for `PointerType`.
        """
        self.run("""
        module {
          tt.func @k(%ptr: !tt.ptr<!tt.ptr<f16>>, %out: !tt.ptr<!tt.ptr<f16>>) {
            %v = tt.load %ptr : !tt.ptr<!tt.ptr<f16>>
            tt.store %out, %v : !tt.ptr<!tt.ptr<f16>>
            tt.return
          }
        }
        """)
        self.assert_present("tt.load")
        self.assert_absent("ktdp.construct_memory_view")
