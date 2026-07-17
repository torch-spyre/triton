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
from utils import compile_to_ttir
from utils_pattern import pattern


# ---------------------------------------------------------------------------
# Shared base + MLIR helpers
# ---------------------------------------------------------------------------

class LowerInterTileTester(SinglePassTester):
    PASS = "add_lower_inter_tile"


def _row_major_core_map(num_slices: dict) -> list[dict]:
    """Derive a row-major core map from slice counts.

    For a multi-axis map the LAST key is the reduction axis (varies within the
    group); the preceding keys are the group-key axes (constant within each group).
    Total tiles = product of all slice counts.

    Single-axis case: tiles get sequential indices (0..N-1) → 1 group of N,
    W[axis]=N, gsize=N.
    Multi-axis case: row-major ordering where the first key varies slowest.

    Examples::

        _row_major_core_map({"x": 4})
        # → [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}]  — 1 group of 4

        _row_major_core_map({"y": 2, "x": 2})
        # → [{"y": 0, "x": 0}, {"y": 0, "x": 1},
        #    {"y": 1, "x": 0}, {"y": 1, "x": 1}]  — 2 groups of 2 (axis="x")
    """
    axes = list(num_slices.keys())
    counts = list(num_slices.values())
    num_tiles = 1
    for c in counts:
        num_tiles *= c

    if len(axes) == 1:
        # Single-axis: sequential slice indices → W[axis]=N, gsize=N, 1 group.
        return [{axes[0]: t} for t in range(num_tiles)]

    # Multi-axis: first axis is outermost / slowest (group key); last is within-group.
    # Row-major: last axis varies fastest.
    core_map = []
    for t in range(num_tiles):
        entry = {}
        rem = t
        for ax, cnt in zip(reversed(axes), reversed(counts)):
            entry[ax] = rem % cnt
            rem //= cnt
        core_map.append({ax: entry[ax] for ax in axes})
    return core_map


def _op_attrs(num_slices: dict, core_map: list[dict] | None = None,
              dep: dict | None = None) -> str:
    """Return the inline ``{ ... }`` attr-dict for a ``tt.inter_tile_reduce`` op.

    The work-slice metadata is attached directly to the op (not to the enclosing
    tt.func), so each op carries its own W/C/D.

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
        _op_attrs({"x": 2})

        # 4 tiles, "y" outer (group key) × "x" inner (reduction axis) — auto row-major:
        #   tile 0 → y=0,x=0  tile 1 → y=0,x=1
        #   tile 2 → y=1,x=0  tile 3 → y=1,x=1
        _op_attrs({"y": 2, "x": 2})  # axis="x"

        # Same layout, axis="y": "x" outer (group key), "y" inner (reduction axis):
        #   tile 0 → x=0,y=0  tile 1 → x=0,y=1
        #   tile 2 → x=1,y=0  tile 3 → x=1,y=1
        _op_attrs({"x": 2, "y": 2})  # axis="y"

        # Non-standard pin (T5) — explicit core_map:
        _op_attrs({"x": 2}, core_map=[{"x": 1}, {"x": 0}])

        # With per-tile dependency:
        _op_attrs({"x": 2}, dep={"0": [0], "1": [1]})
    """
    if core_map is None:
        core_map = _row_major_core_map(num_slices)

    # Derive W from the actual core_map (max value per axis + 1).
    all_axes = list(num_slices.keys())
    W = {ax: max((t.get(ax, 0) for t in core_map), default=0) + 1
         for ax in all_axes}
    nw = ", ".join(f'{k} = {W[k]} : i64' for k in all_axes)
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
    return "{" + base + "}"


# ---------------------------------------------------------------------------
# C5 / Q6 — Fold-away: W[axis] == 1
# ---------------------------------------------------------------------------

class TestFoldAway(LowerInterTileTester):
    """W[axis] == 1 → partials forwarded directly; no KTDP ops emitted."""

    @pattern("fold-away", category="inter-tile", example=[
        "# W[axis] == 1: reduction is a no-op, partial forwarded directly",
        "result = tl.inter_tile(partial, axis='x', combiner='add', mode='all_reduce')",
        "# (axis 'x' has a single slice — no cross-tile communication)",
    ])
    def test_single_tile_axis(self):
        """Single slice on the reduced axis — op is elided entirely."""
        attrs = _op_attrs({"x": 1})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<1x16xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<1x16xf32>)
            tt.return %0 : tensor<1x16xf32>
          }}
        }}
        """)
        self.assert_absent("tt.inter_tile_reduce",
                           "ktdp.inter_tile_produce",
                           "ktdp.inter_tile_reduce")

    def test_fold_away_on_non_innermost_axis(self):
        """W[y]==1 on axis 'y' triggers fold-away (gsize=1, no cross-tile work)."""
        # 4 tiles; W[y]=1 → gsize=1 → each tile is its own group → fold-away.
        attrs = _op_attrs({"x": 4, "y": 1},
                          core_map=[{"x": 0, "y": 0}, {"x": 1, "y": 0},
                                    {"x": 2, "y": 0}, {"x": 3, "y": 0}])
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<1x8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "y" mode = "all_reduce" combiner = "add"
                   {attrs}
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

    @pattern("group-sets", category="inter-tile", example=[
        "# Groups affine sets encode which tiles cooperate in a reduction",
        "result = tl.inter_tile(partial, axis='x', combiner='add', mode='all_reduce')",
        "# producer_tiles_per_group + groups attrs emitted on inter_tile_produce",
    ])
    def test_groups_present(self):
        """inter_tile_produce carries producer_tiles_per_group + groups."""
        # 4 tiles, 1 group (no non-axis dims), W[x]=4, gsize=4, ngroups=1.
        attrs = _op_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce")
        self.assert_absent("tt.inter_tile_reduce")

    def test_two_groups(self):
        """2 groups × 2 tiles each: axis="x" (reduction), y is group key; gsize=2, ngroups=2."""
        attrs = _op_attrs({"y": 2, "x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")
        self.assert_absent("tt.inter_tile_reduce")

    def test_groups_partition(self):
        """Groups affine sets have the right shape for contiguous groups.

        For gsize=4, ngroups=1 (single axis, no group-key dims):
          producer_tiles_per_group = (i)[g] : 2 constraints (≥ lower, ≤ upper)
          groups = (g) : 2 constraints (g >= 0, g <= ngroups-1)
        Checks C1/Q3: the standard contiguous form used by all current fixtures.
        """
        attrs = _op_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        # producer_tiles_per_group: 1 dim (i), 1 symbol (g), 2 constraints (≥/≤)
        self.assert_integer_set("ktdp.inter_tile_produce", "producer_tiles_per_group",
                                num_dims=1, num_symbols=1, num_constraints=2)
        # groups: 1 dim (g), 0 symbols, 2 constraints (g >= 0, ngroups-1-g >= 0)
        self.assert_integer_set("ktdp.inter_tile_produce", "groups",
                                num_dims=1, num_symbols=0, num_constraints=2)

    def test_multi_group_partition(self):
        """Multi-group case: 2 groups × 2 tiles; verify affine set shapes.

        gsize=2, ngroups=2 (axis="x" is reduction dim, "y" is group key).
        producer_tiles_per_group: 1 dim, 1 symbol, 2 constraints.
        groups: 1 dim, 0 symbols, 2 constraints.
        T12 (C7): groups partition producers disjointly.
        """
        attrs = _op_attrs({"y": 2, "x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)
        self.assert_integer_set("ktdp.inter_tile_produce", "producer_tiles_per_group",
                                num_dims=1, num_symbols=1, num_constraints=2)
        self.assert_integer_set("ktdp.inter_tile_produce", "groups",
                                num_dims=1, num_symbols=0, num_constraints=2)
        # The reduce op also carries consumer_tiles_per_group and groups.
        self.assert_affine_attr("ktdp.inter_tile_reduce", "consumer_tiles_per_group")
        self.assert_affine_attr("ktdp.inter_tile_reduce", "groups")


# ---------------------------------------------------------------------------
# C2 / Q2 — Delivery selection: all_reduce vs reduce_to_one
# ---------------------------------------------------------------------------

class TestAllReduce(LowerInterTileTester):
    """all_reduce emits inter_tile_produce + inter_tile_reduce."""

    @pattern("all-reduce", category="inter-tile", example=[
        "result = tl.inter_tile(partial, axis='x', combiner='add', mode='all_reduce')",
        "# Every tile in the group receives the fully reduced value",
    ])
    def test_produce_reduce_pair_emitted(self):
        """all_reduce emits produce/reduce pair; tt.inter_tile_reduce is erased."""
        attrs = _op_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")
        self.assert_absent("tt.inter_tile_reduce")

    def test_consumer_equals_producer(self):
        """all_reduce: consumer_tiles_per_group has same affine shape as producer_tiles_per_group (C2/Q2).

        For all_reduce the delivery op's consumer set = producer set — every tile
        in the group receives the result. Verified by checking both affine sets
        have 1 dim, 1 symbol, 2 constraints (the contiguous group form).
        """
        attrs = _op_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)
        self.assert_integer_set("ktdp.inter_tile_reduce", "consumer_tiles_per_group",
                                num_dims=1, num_symbols=1, num_constraints=2)

    def test_future_single_use(self):
        """The tile_future produced by inter_tile_produce has exactly one use (C7/Q9)."""
        attrs = _op_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)
        # tile_future result type appears exactly once as an operand (in the reduce op)
        self.assert_count("ktdp.inter_tile_produce", 1, cmp="eq")
        self.assert_count("ktdp.inter_tile_reduce", 1, cmp="eq")


class TestReduceToOne(LowerInterTileTester):
    """reduce_to_one emits the produce/reduce pair (consumer = pick₀)."""

    @pattern("reduce-to-one", category="inter-tile", example=[
        "result = tl.inter_tile(partial, axis='x', combiner='add', mode='reduce_to_one')",
        "# Only the designated tile (pick₀ per group) receives the reduced value",
    ])
    def test_reduce_to_one_emitted(self):
        attrs = _op_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "reduce_to_one" combiner = "add"
                   {attrs}
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
        attrs = _op_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "{combiner}"
                   {attrs}
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)

    @pattern("shorthand-identity", category="inter-tile", example=[
        "result = tl.inter_tile(partial, axis='x', combiner='add', mode='all_reduce')",
        "# Shorthand 'add' → identity 0.0 filled via linalg.fill + arith.constant",
    ])
    def test_add_identity(self):
        """add combiner → scalar arith.constant fed into linalg.fill (C3/Q4)."""
        self._run_shorthand("add")
        self.assert_present("linalg.fill", "arith.constant")
        # The fill input is the arith.constant scalar identity value.
        self.assert_operand("linalg.fill", 0, defined_by="arith.constant")

    def test_max_identity(self):
        """max combiner → scalar arith.constant fed into linalg.fill (C3/Q4)."""
        self._run_shorthand("max")
        self.assert_present("linalg.fill", "arith.constant")
        self.assert_operand("linalg.fill", 0, defined_by="arith.constant")

    def test_mul_identity(self):
        """mul combiner → scalar arith.constant fed into linalg.fill (C3/Q4)."""
        self._run_shorthand("mul")
        self.assert_present("linalg.fill", "arith.constant")
        self.assert_operand("linalg.fill", 0, defined_by="arith.constant")

    def test_produce_reduce_regions(self):
        """Producer region contains yield_partial; reducer region contains yield_reduced."""
        self._run_shorthand("add")
        self.assert_present("ktdp.yield_partial", "ktdp.yield_reduced")


# ---------------------------------------------------------------------------
# C4 / Q5 — Result types: rank = partial rank - 1 (unit axis collapsed)
# ---------------------------------------------------------------------------

class TestResultTypes(LowerInterTileTester):
    """Delivery op result type has rank one less than the partial."""

    @pattern("result-types", category="inter-tile", example=[
        "partial: tensor<1xNxf32>  # leading unit axis = within-group tile axis",
        "result = tl.inter_tile(partial, axis='x', combiner='add', mode='all_reduce')",
        "# result: tensor<Nxf32>  — unit axis collapsed, rank decremented by 1",
    ])
    def test_f32_result_type(self):
        """tensor<1x16xf32> partial → tensor<16xf32> result."""
        attrs = _op_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        self.assert_result_type("ktdp.inter_tile_reduce", "tensor<16xf32>")

    def test_f16_result_type(self):
        """f16 partial → f16 result (spec P6)."""
        attrs = _op_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf16>) -> tensor<8xf16>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf16>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<8xf16>)
            tt.return %0 : tensor<8xf16>
          }}
        }}
        """)
        self.assert_result_type("ktdp.inter_tile_reduce", "tensor<8xf16>")

    def test_multi_arity_result_types(self):
        """A=2 partials → A=2 results (multi-arity path, spec P7)."""
        attrs = _op_attrs({"x": 2})
        self.run(f"""
        module {{
          tt.func @k(%p0: tensor<1x8xf32>, %p1: tensor<1x8xf32>)
              -> (tensor<8xf32>, tensor<8xf32>)
          {{
            %0, %1 = tt.inter_tile_reduce
                       partials(%p0 : tensor<1x8xf32>, %p1 : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = "add"
                       {attrs}
                       -> (tensor<8xf32>, tensor<8xf32>)
            tt.return %0, %1 : tensor<8xf32>, tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_reduce")
        self.assert_absent("tt.inter_tile_reduce")

    def test_result_rank_one_less_than_partial(self):
        """Result rank = partial rank - 1: the unit axis is always collapsed (C4/Q5).

        Tests the invariant directly: rank(partial)=2, rank(result)=1.
        For T1: tensor<1x16xf32> → tensor<16xf32>.
        """
        attrs = _op_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x16xf32>) -> tensor<16xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x16xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<16xf32>)
            tt.return %0 : tensor<16xf32>
          }}
        }}
        """)
        # The collapsed result must NOT contain "1x" (the unit axis is gone).
        self.assert_result_type("ktdp.inter_tile_reduce", "tensor<16xf32>")
        self.assert_absent("tt.inter_tile_reduce")


# ---------------------------------------------------------------------------
# work_slices encoding — multi-axis contiguous group layout
# ---------------------------------------------------------------------------

class TestWorkSlices(LowerInterTileTester):
    """work_slices must be laid out so that tiles cooperating on axis are contiguous."""

    @pattern("work-slices", category="inter-tile", example=[
        "# Multi-axis work_slices: group-key axis (first) varies slowest.",
        "# tile_id = pid_out * NUM_MB_TILES + pid_mb",
        "# work_slices[tile_id] = {'out': pid_out, 'mb': pid_mb}",
        "# axis='mb': tiles differing only on 'mb' (same 'out') cooperate.",
        "# Groups must be contiguous in flat tile ordering for LowerInterTile.",
        "result = tl.inter_tile(partial, axis='mb', combiner='add', mode='all_reduce',",
        "                       work_slices=work_slices)",
    ])
    def test_multi_axis_contiguous_groups(self):
        """Multi-axis work_slices: reduction axis last (within-group), groups contiguous.

        2 mb-groups × 2 out-tiles (4 tiles total):
          tile 0 → mb=0, out=0   tile 1 → mb=0, out=1
          tile 2 → mb=1, out=0   tile 3 → mb=1, out=1

        axis="out" (reduction dim): group 0 = tiles {0,1}, group 1 = {2,3}.
        Both groups are contiguous, so LowerInterTile emits the standard
        producer_tiles_per_group (2 constraints) + groups (2 constraints) form.
        """
        # Row-major: "mb" outer (group key), "out" inner (reduction axis).
        attrs = _op_attrs({"mb": 2, "out": 2})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "out" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")
        self.assert_integer_set("ktdp.inter_tile_produce", "producer_tiles_per_group",
                                num_dims=1, num_symbols=1, num_constraints=2)
        self.assert_integer_set("ktdp.inter_tile_produce", "groups",
                                num_dims=1, num_symbols=0, num_constraints=2)

    def test_single_axis_work_slices(self):
        """Single-axis work_slices: all tiles share value 0 → one group (gsize = W[axis])."""
        attrs = _op_attrs({"x": 4})
        self.run(f"""
        module {{
          tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
          {{
            %0 = tt.inter_tile_reduce
                   partials(%p : tensor<1x8xf32>)
                   axis = "x" mode = "all_reduce" combiner = "add"
                   {attrs}
                   -> (tensor<8xf32>)
            tt.return %0 : tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce")
        self.assert_integer_set("ktdp.inter_tile_produce", "groups",
                                num_dims=1, num_symbols=0, num_constraints=2)


# ---------------------------------------------------------------------------
# Double all_reduce — two sequential tl.inter_tile calls in one kernel
# ---------------------------------------------------------------------------

class TestDoubleAllReduce(LowerInterTileTester):
    """Two consecutive tt.inter_tile_reduce ops lower independently."""

    @pattern("double-all-reduce", category="inter-tile", example=[
        "# Two sequential all-reduces sharing the same work_slices (softmax pattern).",
        "rowmax = tl.inter_tile(partial_max, axis='mb', combiner='max', mode='all_reduce',",
        "                       work_slices=work_slices)",
        "rowsum = tl.inter_tile(partial_sum, axis='mb', combiner='add', mode='all_reduce',",
        "                       work_slices=work_slices)",
        "# Both lower to independent produce/reduce pairs (2 × produce + 2 × reduce).",
    ])
    def test_two_all_reduces_emitted(self):
        """Two tt.inter_tile_reduce ops → 2 produce + 2 reduce pairs."""
        attrs = _op_attrs({"mb": 2, "out": 2})
        self.run(f"""
        module {{
          tt.func @k(%pmax: tensor<1x8xf32>, %psum: tensor<1x8xf32>)
              -> (tensor<8xf32>, tensor<8xf32>)
          {{
            %rowmax = tt.inter_tile_reduce
                        partials(%pmax : tensor<1x8xf32>)
                        axis = "out" mode = "all_reduce" combiner = "max"
                        {attrs}
                        -> (tensor<8xf32>)
            %rowsum = tt.inter_tile_reduce
                        partials(%psum : tensor<1x8xf32>)
                        axis = "out" mode = "all_reduce" combiner = "add"
                        {attrs}
                        -> (tensor<8xf32>)
            tt.return %rowmax, %rowsum : tensor<8xf32>, tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")
        self.assert_count("ktdp.inter_tile_produce", 2)
        self.assert_count("ktdp.inter_tile_reduce", 2)
        self.assert_absent("tt.inter_tile_reduce")

    def test_two_all_reduces_independent(self):
        """The two reduce ops carry independent combiner regions (max and add)."""
        attrs = _op_attrs({"mb": 2, "out": 2})
        self.run(f"""
        module {{
          tt.func @k(%pmax: tensor<1x8xf32>, %psum: tensor<1x8xf32>)
              -> (tensor<8xf32>, tensor<8xf32>)
          {{
            %rowmax = tt.inter_tile_reduce
                        partials(%pmax : tensor<1x8xf32>)
                        axis = "out" mode = "all_reduce" combiner = "max"
                        {attrs}
                        -> (tensor<8xf32>)
            %rowsum = tt.inter_tile_reduce
                        partials(%psum : tensor<1x8xf32>)
                        axis = "out" mode = "all_reduce" combiner = "add"
                        {attrs}
                        -> (tensor<8xf32>)
            tt.return %rowmax, %rowsum : tensor<8xf32>, tensor<8xf32>
          }}
        }}
        """)
        self.assert_present("linalg.max", parent="ktdp.inter_tile_reduce")
        self.assert_present("linalg.add", parent="ktdp.inter_tile_reduce")


# ---------------------------------------------------------------------------
# C6 / Q8 — No-op: module without tt.inter_tile_reduce is unchanged
# ---------------------------------------------------------------------------

class TestNoOp(LowerInterTileTester):
    """Module with no tt.inter_tile_reduce passes through unchanged."""

    @pattern("no-op", category="inter-tile", example=[
        "# No tl.inter_tile call — pass does not transform the module",
        "result = x + y  # plain arithmetic; no inter-tile reduction",
    ])
    def test_empty_module_unchanged(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<16xf32>) -> tensor<16xf32> {
            tt.return %a : tensor<16xf32>
          }
        }
        """)
        self.assert_absent("ktdp.inter_tile_produce", "ktdp.inter_tile_reduce")



# ---------------------------------------------------------------------------
# R8 — Deferred modes: broadcast / reduce_scatter → diagnostic
# ---------------------------------------------------------------------------

class TestBroadcastDeferred(LowerInterTileTester):
    """broadcast and reduce_scatter are rejected with R8 diagnostic."""

    def test_broadcast_rejected(self, capfd):
        attrs = _op_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "broadcast" combiner = ""
                       {attrs}
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "not yet supported")

    def test_reduce_scatter_rejected(self, capfd):
        attrs = _op_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "reduce_scatter" combiner = "add"
                       scatter_dimension = 0
                       {attrs}
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
        """R2: op without work-slice attrs → op verifier rejects (requires coreIdToWkSlice)."""
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
        self.assert_stderr(capfd, "requires attribute")

    def test_unknown_axis_rejected(self, capfd):
        """R1: axis not in numWkSlicesPerDim → diagnostic."""
        attrs = _op_attrs({"y": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = "add"
                       {attrs}
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "numWkSlicesPerDim")

    def test_unknown_mode_rejected(self, capfd):
        """R5: unknown mode string → diagnostic."""
        attrs = _op_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "bad_mode" combiner = "add"
                       {attrs}
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "unknown mode")

    def test_region_combiner_without_identity_rejected(self, capfd):
        """R4: region combiner (combiner="") with no identity operands → diagnostic."""
        attrs = _op_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = ""
                       {attrs}
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "region combiner requires")

    def test_scatter_dim_without_reduce_scatter_rejected(self, capfd):
        """R3: scatter_dimension on non-reduce_scatter mode → diagnostic."""
        attrs = _op_attrs({"x": 2})
        with pytest.raises(RuntimeError):
            self.run(f"""
            module {{
              tt.func @k(%p: tensor<1x8xf32>) -> tensor<8xf32>
              {{
                %0 = tt.inter_tile_reduce
                       partials(%p : tensor<1x8xf32>)
                       axis = "x" mode = "all_reduce" combiner = "add"
                       scatter_dimension = 0
                       {attrs}
                       -> (tensor<8xf32>)
                tt.return %0 : tensor<8xf32>
              }}
            }}
            """)
        self.assert_stderr(capfd, "scatter_dimension")


# ---------------------------------------------------------------------------
# E4 — tl.wk_slice_coord: runtime tile coordinate lookup
#
# Unlike the other classes here, wk_slice_coord is a *frontend emission*
# feature, not a KTIR pass rewrite: it lowers to plain TTIR (a program_id
# fed through a constant-folded arith.cmpi/arith.select chain).  So these
# tests compile a real @triton.jit kernel to TTIR and inspect the text,
# rather than running the LowerInterTile pass on inline MLIR.
# ---------------------------------------------------------------------------

class TestWkSliceCoord:
    """E4: tl.wk_slice_coord(work_slices, axis) → runtime i32 coordinate."""

    # 4 tiles, "out" outer / "in" inner — the split-K topology.
    WORK_SLICES = [
        {"out": 0, "in": 0}, {"out": 0, "in": 1},
        {"out": 1, "in": 0}, {"out": 1, "in": 1},
    ]

    def _compile(self):
        from fixtures.inter_tile_reduce import kernel
        sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "i32", "K": "i32", "N": "i32"}
        cx = {"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 32,
              "NUM_IN_TILES": 2, "WORK_SLICES": self.WORK_SLICES}
        return compile_to_ttir(kernel.matmul_splitk_kernel, sig, cx)

    @pattern("wk-slice-coord", category="inter-tile", example=[
        "# Recover this tile's slice coordinates from work_slices itself,",
        "# instead of the manual pid // NUM_IN_TILES radix (spec E4).",
        "pid_out = tl.wk_slice_coord(work_slices, 'out')   # runtime i32",
        "pid_in  = tl.wk_slice_coord(work_slices, 'in')    # runtime i32",
        "if pid_in == 0:",
        "    c_desc.store([pid_out * BLOCK_M, 0], result)",
    ])
    def test_lowers_to_program_id_indexed_select_chain(self):
        """wk_slice_coord emits a program_id-indexed select chain, no new op.

        Each call folds the constexpr per-axis column into a chain of
        ``arith.cmpi eq`` (pid == i) + ``arith.select`` over the runtime
        ``tt.get_program_id``.  No bespoke IR op is introduced.
        """
        ttir = self._compile()
        # The two wk_slice_coord calls (out, in) read program_id.
        assert ttir.count("tt.get_program_id") >= 1
        # Coordinate lookup composes cmpi-eq + select (constant-folded; the
        # in-column [0,1,0,1] and out-column [0,0,1,1] yield several of each).
        assert "arith.cmpi eq" in ttir
        assert "arith.select" in ttir
        # No new dedicated op was introduced for the lookup.
        assert "wk_slice_coord" not in ttir

    def test_result_is_i32_runtime_scalar(self):
        """The selected coordinate is a runtime i32 scalar (not a constexpr)."""
        ttir = self._compile()
        # The select chain operates on i32 scalars (the result type ": i32"
        # appears before the trailing MLIR `loc(...)` annotation).
        assert "arith.select" in ttir
        for line in ttir.splitlines():
            if "arith.select" in line:
                assert ": i32" in line, \
                    f"wk_slice_coord select not i32-scalar: {line!r}"

    def test_invalid_axis_raises_at_compile_time(self):
        """An axis absent from work_slices is a compile-time error."""
        from fixtures.inter_tile_reduce import kernel
        sig = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "i32", "K": "i32", "N": "i32"}
        # work_slices entries carry only "out"/"in"; "bogus" is missing.
        cx = {"BLOCK_M": 32, "BLOCK_K": 32, "BLOCK_N": 32,
              "NUM_IN_TILES": 2, "WORK_SLICES": self.WORK_SLICES}

        # Patch a kernel that asks for a missing axis by compiling a thin
        # wrapper is overkill; instead, drive the semantic helper directly is
        # not possible without a builder.  So assert via a dedicated kernel.
        bad_ws = [{"out": 0}, {"out": 0}, {"out": 1}, {"out": 1}]
        cx_bad = dict(cx, WORK_SLICES=bad_ws)
        with pytest.raises(Exception) as ei:
            compile_to_ttir(kernel.matmul_splitk_kernel, sig, cx_bad)
        # Root cause mentions the missing axis.
        msgs, cur = [], ei.value
        while cur is not None:
            msgs.append(str(cur))
            cur = getattr(cur, "__cause__", None)
        joined = " ".join(msgs)
        assert "wk_slice_coord" in joined and "missing" in joined and "'in'" in joined
