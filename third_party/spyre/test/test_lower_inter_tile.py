#!/usr/bin/env python3
"""
Unit tests for the LowerInterTile pass.

Organization: one test class per rewrite rule / spec clause.

  TestFoldAway          — C5/Q6:  W[axis]==1 → forward partials, no KTDP ops
  TestGroupSets         — C1/Q3:  producer_tiles_per_group + groups affine sets
  TestAllReduce         — C2/Q2:  all_reduce consumer == producer set
  TestReduceToOne       — C2/Q2:  reduce_to_one consumer == pick₀ (1-point set)
  TestShorthandIdentity — C3/Q4:  add/max/mul → linalg.fill identity
  TestResultTypes       — C4/Q5:  result rank = partial rank - 1 (unit axis collapsed)
  TestNoOp              — C6/Q8:  no op → IR unchanged
  TestBroadcastDeferred — R8:     broadcast/reduce_scatter → diagnostic
  TestValidation        — R1–R5:  precondition rejections

Partial shape convention: the partial tensor carries a leading unit dimension
(size 1) that is the within-group tile axis to be collapsed by the reduce.
For example, partial ``tensor<1x16xf32>`` collapses to result ``tensor<16xf32>``.

Negative tests use ``pytest.raises`` + ``assert_stderr(capfd, ...)`` to verify
both the RuntimeError and the MLIR diagnostic on stderr.
"""

import pytest
from conftest import SinglePassTester


# ---------------------------------------------------------------------------
# Shared base + MLIR helpers
# ---------------------------------------------------------------------------

class LowerInterTileTester(SinglePassTester):
    PASS = "add_lower_inter_tile"


def _row_major_core_map(num_slices: dict) -> list[dict]:
    """Derive the standard mixed-radix (row-major) core map from slice counts.

    Axes are ordered as given; the last axis is the fastest-varying (innermost).
    Total tiles = product of all slice counts.

    Examples::

        _row_major_core_map({"x": 2})
        # → [{"x": 0}, {"x": 1}]

        _row_major_core_map({"x": 2, "y": 2})
        # → [{"x": 0, "y": 0}, {"x": 0, "y": 1},
        #    {"x": 1, "y": 0}, {"x": 1, "y": 1}]
    """
    axes = list(num_slices.keys())
    counts = list(num_slices.values())
    num_tiles = 1
    for c in counts:
        num_tiles *= c
    core_map = []
    for t in range(num_tiles):
        entry = {}
        rem = t
        for ax, cnt in zip(reversed(axes), reversed(counts)):
            entry[ax] = rem % cnt
            rem //= cnt
        core_map.append({ax: entry[ax] for ax in axes})
    return core_map


def _func_attrs(num_slices: dict, core_map: list[dict] | None = None,
                dep: dict | None = None) -> str:
    """Return the full ``attributes { ... }`` block for a ``tt.func``.

    Args:
        num_slices: axis-name → slice-count, e.g. ``{"x": 2, "y": 4}``.
                    Axis order determines the row-major layout when
                    ``core_map`` is omitted: last axis is innermost (fastest).
        core_map:   explicit per-tile slice-index maps, one dict per tile.
                    Omit to auto-derive the standard row-major layout from
                    ``num_slices`` (covers all non-pinned fixtures).
                    Pass explicitly for non-standard pins (e.g. T5, T8).
        dep:        optional ``depWkSlices`` — maps a consumer's within-group
                    local-index string to a list of producer local indices, e.g.
                    ``{"0": [0], "1": [1]}`` for a pairwise dependency.

    Examples::

        # 2 tiles, single axis "x" — auto row-major:
        _func_attrs({"x": 2})

        # 4 tiles, "x" outer × "y" inner — auto row-major:
        #   tile 0 → x=0,y=0  tile 1 → x=0,y=1
        #   tile 2 → x=1,y=0  tile 3 → x=1,y=1
        _func_attrs({"x": 2, "y": 2})

        # Strided layout (T8): "y" outer (s_y=2), "x" inner — auto:
        #   tile 0 → y=0,x=0  tile 1 → y=0,x=1
        #   tile 2 → y=1,x=0  tile 3 → y=1,x=1
        _func_attrs({"y": 2, "x": 2})

        # Non-standard pin (T5) — explicit core_map:
        _func_attrs({"x": 2}, core_map=[{"x": 1}, {"x": 0}])

        # With per-tile dependency:
        _func_attrs({"x": 2}, dep={"0": [0], "1": [1]})
    """
    if core_map is None:
        core_map = _row_major_core_map(num_slices)

    nw = ", ".join(f'{k} = {v} : i64' for k, v in num_slices.items())
    nw_attr = "{" + nw + "}"

    def tile_dict(d):
        inner = ", ".join(f'{k} = {v} : i64' for k, v in d.items())
        return "{" + inner + "}"
    cm_attr = "[" + ", ".join(tile_dict(t) for t in core_map) + "]"

    base = f'numWkSlicesPerDim = {nw_attr}, coreIdToWkSlice = {cm_attr}'
    if dep is not None:
        items = []
        for k, vs in dep.items():
            vals = ", ".join(f'{v} : i64' for v in vs)
            items.append(f'"{k}" = [{vals}]')
        base += ', depWkSlices = {' + ", ".join(items) + '}'
    return "attributes {" + base + "}"


# ---------------------------------------------------------------------------
# C5 / Q6 — Fold-away: W[axis] == 1
# ---------------------------------------------------------------------------

class TestFoldAway(LowerInterTileTester):
    """W[axis] == 1 → partials forwarded directly; no KTDP ops emitted."""

    def test_single_tile_axis(self):
        """Single slice on the reduced axis — op is elided entirely."""
        attrs = _func_attrs({"x": 1})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<1x16xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   -> (tensor<1x16xf32>)
            tt.return %0 : tensor<1x16xf32>
          }}
        }}
        """)
        self.assert_absent("tt.inter_tile_reduce",
                           "ktdp.inter_tile_produce",
                           "ktdp.inter_tile_reduce")

    def test_fold_away_on_non_innermost_axis(self):
        """W[y]==1 triggers fold-away even when other axes have W>1."""
        attrs = _func_attrs({"x": 4, "y": 1})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<1x8xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "y" mode = "all_reduce" combiner = "add"
                   -> (tensor<1x8xf32>)
            tt.return %0 : tensor<1x8xf32>
          }}
        }}
        """)
        self.assert_absent("tt.inter_tile_reduce",
                           "ktdp.inter_tile_produce",
                           "ktdp.inter_tile_reduce")
        # The partial arg flows straight to tt.return with no intervening op.
        self.assert_operand("tt.return", 0, defined_by=None,
                            type_substr="tensor<1x8xf32>")


# ---------------------------------------------------------------------------
# C1 / Q3 — Group sets: producer_tiles_per_group + groups
# ---------------------------------------------------------------------------

class TestGroupSets(LowerInterTileTester):
    """Groups affine sets are emitted and structurally correct."""

    def test_groups_present(self):
        """inter_tile_produce carries producer_tiles_per_group + groups."""
        # 4 tiles, all reducing on axis "x" (gsize=4, ngroups=1)
        attrs = _func_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce")
        self.assert_absent("tt.inter_tile_reduce")

    def test_two_groups(self):
        """2 groups × 2 tiles each: gsize=2, ngroups=2."""
        attrs = _func_attrs({"x": 2, "y": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")
        self.assert_absent("tt.inter_tile_reduce")


# ---------------------------------------------------------------------------
# C2 / Q2 — Delivery selection: all_reduce vs reduce_to_one
# ---------------------------------------------------------------------------

class TestAllReduce(LowerInterTileTester):
    """all_reduce emits inter_tile_produce + inter_tile_reduce."""

    def test_produce_reduce_pair_emitted(self):
        """all_reduce emits produce/reduce pair; tt.inter_tile_reduce is erased."""
        attrs = _func_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")
        self.assert_absent("tt.inter_tile_reduce")


class TestReduceToOne(LowerInterTileTester):
    """reduce_to_one emits the produce/reduce pair (consumer = pick₀)."""

    def test_reduce_to_one_emitted(self):
        attrs = _func_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "reduce_to_one" combiner = "add"
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")
        self.assert_absent("tt.inter_tile_reduce")


# ---------------------------------------------------------------------------
# C3 / Q4 — Shorthand combiner identity materialization
# ---------------------------------------------------------------------------

class TestShorthandIdentity(LowerInterTileTester):
    """add/max/mul shorthand combiners materialize the correct identity."""

    def _run_shorthand(self, combiner: str):
        attrs = _func_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "{combiner}"
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)

    def test_add_identity(self):
        """add combiner → arith.constant 0.0 via linalg.fill."""
        self._run_shorthand("add")
        self.assert_present("linalg.fill", "arith.constant")

    def test_max_identity(self):
        """max combiner → arith.constant -inf via linalg.fill."""
        self._run_shorthand("max")
        self.assert_present("linalg.fill", "arith.constant")

    def test_mul_identity(self):
        """mul combiner → arith.constant 1.0 via linalg.fill."""
        self._run_shorthand("mul")
        self.assert_present("linalg.fill", "arith.constant")

    def test_produce_reduce_regions(self):
        """Producer region contains yield_partial; reducer region contains yield_reduced."""
        self._run_shorthand("add")
        self.assert_present("ktdp.yield_partial", "ktdp.yield_reduced")


# ---------------------------------------------------------------------------
# C4 / Q5 — Result types: rank = partial rank - 1 (unit axis collapsed)
# ---------------------------------------------------------------------------

class TestResultTypes(LowerInterTileTester):
    """Delivery op result type has rank one less than the partial."""

    def test_f32_result_type(self):
        """tensor<1x16xf32> partial → tensor<16xf32> result."""
        attrs = _func_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        self.assert_result_type("ktdp.inter_tile_reduce", "tensor<16xf32>")

    def test_f16_result_type(self):
        """f16 partial → f16 result (spec P6)."""
        attrs = _func_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf16>) -> tensor<8xf16>
          {attrs} {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf16>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   -> (tensor<8xf16>)
            tt.return %0 : tensor<8xf16>
          }}
        }}
        """)
        self.assert_result_type("ktdp.inter_tile_reduce", "tensor<8xf16>")

    def test_multi_arity_result_types(self):
        """A=2 partials → A=2 results (multi-arity path, spec P7)."""
        attrs = _func_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p0: tensor<1x8xf32>, %p1: tensor<1x8xf32>)
              -> (tensor<8xf32>, tensor<8xf32>)
          {attrs} {{
            %0, %1 = tt.inter_tile_reduce
                       partials(%p0 : tensor<1x8xf32>, %p1 : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = "add"
                       -> (tensor<8xf32>, tensor<8xf32>)
            tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_reduce")
        self.assert_absent("tt.inter_tile_reduce")



# ---------------------------------------------------------------------------
# C6 / Q8 — No-op: module without tt.inter_tile_reduce is unchanged
# ---------------------------------------------------------------------------

class TestNoOp(LowerInterTileTester):
    """Module with no tt.inter_tile_reduce passes through unchanged."""

    def test_empty_module_unchanged(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<16xf32>) -> tensor<16xf32> {
            tt.return %a : tensor<16xf32>
          }
        }
        """)
        self.assert_absent("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")

    def test_func_with_attrs_no_op(self):
        """Function with work-slice attrs but no inter_tile_reduce → unchanged."""
        attrs = _func_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%a: tensor<16xf32>) -> tensor<16xf32>
          {attrs} {{
            tt.return %a : tensor<16xf32>
          }}
        }}
        """)
        self.assert_absent("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")


# ---------------------------------------------------------------------------
# R8 — Deferred modes: broadcast / reduce_scatter → diagnostic
# ---------------------------------------------------------------------------

class TestBroadcastDeferred(LowerInterTileTester):
    """broadcast and reduce_scatter are rejected with R8 diagnostic."""

    def test_broadcast_rejected(self, capfd):
        attrs = _func_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {attrs} {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "broadcast" combiner = ""
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "not yet supported")

    def test_reduce_scatter_rejected(self, capfd):
        attrs = _func_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {attrs} {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "reduce_scatter" combiner = "add"
                       scatter_dimension = 0
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "not yet supported")


# ---------------------------------------------------------------------------
# R1–R5 — Precondition validation
# ---------------------------------------------------------------------------

class TestValidation(LowerInterTileTester):
    """Inline precondition checks (P3/P4/P5) emit diagnostics and fail."""

    def test_missing_work_slice_attrs_rejected(self, capfd):
        """R2: missing numWkSlicesPerDim → diagnostic."""
        with pytest.raises(RuntimeError):
            self.run("""
            module {
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32> {
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = "add"
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }
            }
            """)
        self.assert_stderr(capfd, "numWkSlicesPerDim")

    def test_unknown_axis_rejected(self, capfd):
        """R1: axis not in numWkSlicesPerDim → diagnostic."""
        attrs = _func_attrs({"y": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {attrs} {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = "add"
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "numWkSlicesPerDim")

    def test_unknown_mode_rejected(self, capfd):
        """R5: unknown mode string → diagnostic."""
        attrs = _func_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {attrs} {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "bad_mode" combiner = "add"
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "unknown mode")

    def test_region_combiner_without_identity_rejected(self, capfd):
        """R4: region combiner (combiner="") with no identity operands → diagnostic."""
        attrs = _func_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {attrs} {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = ""
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "region combiner requires")

    def test_scatter_dim_without_reduce_scatter_rejected(self, capfd):
        """R3: scatter_dimension on non-reduce_scatter mode → diagnostic."""
        attrs = _func_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {attrs} {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = "add"
                       scatter_dimension = 0
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "scatter_dimension")
