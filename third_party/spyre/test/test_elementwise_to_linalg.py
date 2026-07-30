#!/usr/bin/env python3
"""
Tests for the elementwise-tensor-arith -> linalg.generic conversion.

The Spyre dataflow scheduler accepts only ``linalg`` operations; KTIR that
still carries tensor-typed ``arith`` or ``math`` operations cannot be
scheduled. Upstream MLIR's ``convert-elementwise-to-linalg`` pass closes that
gap. It rewrites every operation carrying the ``ElementwiseMappable`` trait
(which covers all of ``arith`` and ``math``) that has at least one
ranked-tensor operand into a ``linalg.generic`` — a ``linalg`` operation whose
per-element computation is spelled out as a small block of IR rather than
implied by the operation's name. The original scalar operation ends up inside
that block.

Two properties are pinned here because they are load-bearing and easy to break:

1. Operations whose operands are all scalars are left alone. This is required:
   ``linalg.reduce`` combiner regions (the block of IR saying how to merge two
   partial reduction results), index arithmetic, and ``ktdp`` address
   computation are all built from scalar ``arith``. Converting those would
   produce malformed IR, because a tensor-typed operation cannot appear inside
   a combiner region.

2. ``arith.constant`` with a dense (whole-tensor) value is left alone. It does
   not carry the ``ElementwiseMappable`` trait, so the pass never sees it.
   Rewriting dense constants into ``tensor.empty`` + ``linalg.fill`` is
   deferred; see ``docs/designs/linalg-compute-normalization.md``.

The pass itself is upstream code and is not re-tested here. These tests pin the
*integration*: that Spyre invokes it, that it is reachable from the Python
bindings, and that the two carve-outs above hold for the IR shapes this backend
actually produces.
"""

import pytest

from conftest import SinglePassTester


# Dialect-qualified names of the operations the conversion must consume.
# Parametrized rather than hardcoded per-test so a new row is one line.
#
# Each entry is (op_name, mlir_body_line, extra_operand_types).
# The body line is written against SSA values %a and %b, both of type
# tensor<TILE>xf32 (or the integer form where the operation demands it).
ELEMENTWISE_CASES = [
    # Floating-point binary
    ("arith.addf", "%r = arith.addf %a, %b : {T}", "f32"),
    ("arith.mulf", "%r = arith.mulf %a, %b : {T}", "f32"),
    ("arith.subf", "%r = arith.subf %a, %b : {T}", "f32"),
    ("arith.divf", "%r = arith.divf %a, %b : {T}", "f32"),
    # maxnumf is the operation softmax's row-max reduction produces, so it is
    # the single most important row here.
    ("arith.maxnumf", "%r = arith.maxnumf %a, %b : {T}", "f32"),
    ("arith.minnumf", "%r = arith.minnumf %a, %b : {T}", "f32"),
    # Floating-point unary
    ("arith.negf", "%r = arith.negf %a : {T}", "f32"),
    # math dialect — a separate dialect from arith, so it needs its own
    # coverage even though the same trait drives the conversion.
    ("math.exp", "%r = math.exp %a : {T}", "f32"),
    ("math.sqrt", "%r = math.sqrt %a : {T}", "f32"),
    ("math.log", "%r = math.log %a : {T}", "f32"),
    # Integer binary. MLIR's i32 is signless, so signed and unsigned division
    # are distinct operations; both must convert.
    ("arith.addi", "%r = arith.addi %a, %b : {T}", "i32"),
    ("arith.divsi", "%r = arith.divsi %a, %b : {T}", "i32"),
    ("arith.divui", "%r = arith.divui %a, %b : {T}", "i32"),
    # Bitwise — no named linalg equivalent exists, which is exactly why
    # linalg.generic is the right target.
    ("arith.andi", "%r = arith.andi %a, %b : {T}", "i32"),
    ("arith.xori", "%r = arith.xori %a, %b : {T}", "i32"),
]

# Comparisons are separated out because their result element type (i1) differs
# from their operand element type, which exercises a different code path in the
# conversion (the destination tensor cannot reuse an operand).
COMPARE_CASES = [
    ("arith.cmpf", '%r = arith.cmpf ogt, %a, %b : {T}', "f32", "i1"),
    ("arith.cmpi", '%r = arith.cmpi sgt, %a, %b : {T}', "i32", "i1"),
]

# Tile widths. 1 is the smallest legal tensor extent and collapses several
# indexing-map cases; 3 is a small non-power-of-two; 7 does not divide any
# natural vector width, which is where a shape-assuming implementation breaks.
TILE_WIDTHS = [1, 3, 7, 64]


class ElementwiseToLinalgTester(SinglePassTester):
    """Runs upstream convert-elementwise-to-linalg in isolation.

    Uses the dedicated ``add_convert_elementwise_to_linalg`` binding rather
    than the full pipeline so a failure localises to this conversion instead of
    to whichever Spyre pass ran alongside it.
    """
    PASS = "add_convert_elementwise_to_linalg"

    def assert_no_tensor_typed(self, op_name: str):
        """Assert no instance of *op_name* produces a ranked-tensor result.

        This is the property the conversion actually establishes, and it is not
        the same as "the operation is gone". A successful conversion *keeps* the
        operation — it moves a scalar copy into the ``linalg.generic`` body,
        where the operands and result are element-typed rather than
        tensor-typed. Asserting plain absence would fail on correct output,
        because the assertion helpers walk nested regions.
        """
        survivors = [
            o for o in self._find(op_name)
            if any("tensor<" in t for t in o.result_types)
        ]
        assert not survivors, (
            f"Op '{op_name}' still produces a tensor result: "
            f"{[o.result_types for o in survivors]}"
        )

    @staticmethod
    def _module(body: str, elem: str, width: int, result_elem: str = ""):
        """Wraps *body* in a tt.func taking two tensor<{width}x{elem}> args.

        Uses ``tt.func`` rather than ``func.func`` because the test context
        registers the Triton and KTDP dialects but not ``func`` (see
        ``load_dialects`` in ``third_party/spyre/triton_spyre.cc``). Parsing a
        ``func.func`` here fails with "Dialect `func' not found".

        *body* must define ``%r`` from ``%a`` and ``%b`` and may use the
        ``{T}`` placeholder for the operand tensor type. ``%r`` is returned so
        the operation is not dead — dead-code elimination would otherwise erase
        the very operation under test and make the assertions vacuous.

        *result_elem* overrides the result element type, for comparisons whose
        result element type (i1) differs from their operands'.
        """
        operand_ty = f"tensor<{width}x{elem}>"
        result_ty = f"tensor<{width}x{result_elem}>" if result_elem else operand_ty
        return f"""
        module {{
          tt.func @k(%a: {operand_ty}, %b: {operand_ty}) -> {result_ty} {{
            {body.format(T=operand_ty)}
            tt.return %r : {result_ty}
          }}
        }}
        """


@pytest.mark.parametrize("width", TILE_WIDTHS)
class TestElementwiseConverted(ElementwiseToLinalgTester):
    """Every tensor-typed arith/math operation becomes a linalg.generic."""

    @pytest.mark.parametrize(
        "op_name,body,elem",
        ELEMENTWISE_CASES,
        ids=[c[0] for c in ELEMENTWISE_CASES],
    )
    def test_converted(self, op_name, body, elem, width):
        """A linalg.generic replaces the operation, with it moved into the body.

        All three assertions matter. The generic must exist; no tensor-typed
        instance of the original operation may remain; and a scalar instance
        must be inside the generic's body — that last one rules out the
        conversion having simply deleted the computation.
        """
        self.run(self._module(body, elem, width))
        self.assert_present("linalg.generic")
        self.assert_no_tensor_typed(op_name)
        self.assert_present(op_name, parent="linalg.generic")

    @pytest.mark.parametrize(
        "op_name,body,elem,result_elem",
        COMPARE_CASES,
        ids=[c[0] for c in COMPARE_CASES],
    )
    def test_compare_converted(self, op_name, body, elem, result_elem, width):
        """Comparisons convert even though result and operand element types differ.

        A comparison on tensor<Nxf32> yields tensor<Nxi1>. The conversion
        cannot reuse an operand as the destination tensor here, so it must
        create a fresh one — a path the same-type cases never exercise.
        """
        self.run(self._module(body, elem, width, result_elem=result_elem))
        self.assert_present("linalg.generic")
        self.assert_no_tensor_typed(op_name)
        self.assert_present(op_name, parent="linalg.generic")
        # The generic's result carries the i1 element type, not the operands'.
        self.assert_result_type("linalg.generic", f"tensor<{width}xi1>")

    def test_scalar_operation_untouched(self, width):
        """Scalar arith is left alone even when tensor arith is present nearby.

        Guards the per-operand-type carve-out. If the conversion keyed on the
        dialect instead, the scalar addf would be rewritten into a rank-0
        linalg.generic and every linalg.reduce combiner region in the backend
        would become malformed.

        *width* is unused; the class-level parametrize applies to all tests.
        """
        self.run(f"""
        module {{
          tt.func @k(%s: f32, %t: tensor<{width}xf32>) -> (f32, tensor<{width}xf32>) {{
            %x = arith.addf %s, %s : f32
            %y = arith.mulf %t, %t : tensor<{width}xf32>
            tt.return %x, %y : f32, tensor<{width}xf32>
          }}
        }}
        """)
        # The tensor operation converted.
        self.assert_present("linalg.generic")
        self.assert_no_tensor_typed("arith.mulf")
        # The scalar addf is still at function level, NOT pulled into a generic.
        # Checking the parent is the point: a plain assert_present would also
        # pass if it had been moved inside a rank-0 generic, which is the exact
        # failure this test exists to catch.
        self.assert_present("arith.addf", parent="tt.func")

    def test_mixed_scalar_and_tensor_operands(self, width):
        """An operation with one scalar and one tensor operand still converts.

        The scalar operand is passed straight through as an input and read via a
        rank-reducing indexing map, rather than being splatted into a tensor
        first. tt.splat produces exactly this shape, so it is not a synthetic
        case.
        """
        self.run(f"""
        module {{
          tt.func @k(%s: f32, %t: tensor<{width}xf32>) -> tensor<{width}xf32> {{
            %z = tt.splat %s : f32 -> tensor<{width}xf32>
            %r = arith.addf %t, %z : tensor<{width}xf32>
            tt.return %r : tensor<{width}xf32>
          }}
        }}
        """)
        self.assert_present("linalg.generic")
        self.assert_no_tensor_typed("arith.addf")
        self.assert_present("arith.addf", parent="linalg.generic")


@pytest.mark.parametrize("width", TILE_WIDTHS)
class TestDeferredDenseConstant(ElementwiseToLinalgTester):
    """Dense constants are deliberately left untouched.

    Pins the deferral rather than the eventual behaviour: rewriting these into
    ``tensor.empty`` + ``linalg.fill`` is planned but out of scope. When that
    work lands these tests fail loudly, which is the intent — they mark the
    boundary so it cannot move silently.
    """

    def test_splat_constant_survives(self, width):
        """A splat dense constant (every element identical) is not converted.

        arith.constant does not carry the ElementwiseMappable trait, so the
        conversion never matches it regardless of its result type.
        """
        self.run(f"""
        module {{
          tt.func @k() -> tensor<{width}xf32> {{
            %c = arith.constant dense<1.0> : tensor<{width}xf32>
            tt.return %c : tensor<{width}xf32>
          }}
        }}
        """)
        self.assert_present("arith.constant")
        self.assert_absent("linalg.fill", "linalg.generic")

    def test_dense_constant_feeding_converted_op(self, width):
        """A dense constant survives while its consumer converts.

        The realistic shape: the constant stays an arith.constant and is read
        as an input by the linalg.generic that replaced its consumer.
        """
        self.run(f"""
        module {{
          tt.func @k(%t: tensor<{width}xf32>) -> tensor<{width}xf32> {{
            %c = arith.constant dense<2.0> : tensor<{width}xf32>
            %r = arith.mulf %t, %c : tensor<{width}xf32>
            tt.return %r : tensor<{width}xf32>
          }}
        }}
        """)
        self.assert_present("arith.constant", "linalg.generic")
        self.assert_no_tensor_typed("arith.mulf")
        # The constant is still tensor-typed: it was NOT converted, and it is
        # read as an input by the generic that replaced its consumer.
        self.assert_result_type("arith.constant", f"tensor<{width}xf32>")

    def test_non_splat_constant_survives(self, width):
        """A non-splat dense constant is not converted and not rejected.

        Non-splat constants cannot be expressed as a linalg.fill, which takes a
        single scalar. Today they pass through untouched. Guarding this
        separately from the splat case because the eventual implementation will
        treat the two differently: splats become linalg.fill, non-splats need a
        diagnostic.
        """
        elems = ", ".join(f"{float(i)}" for i in range(width))
        self.run(f"""
        module {{
          tt.func @k() -> tensor<{width}xf32> {{
            %c = arith.constant dense<[{elems}]> : tensor<{width}xf32>
            tt.return %c : tensor<{width}xf32>
          }}
        }}
        """)
        self.assert_present("arith.constant")
        self.assert_absent("linalg.fill", "linalg.generic")
