#!/usr/bin/env python3
"""
Unit tests for the DistributeWork pass.

DistributeWork replaces every ``tt.get_program_id`` with a shared
``ktdp.get_compute_tile_id`` (variadic: one index per grid dim) plus
per-axis ``arith.index_cast`` to i32, and stamps a ``grid`` attribute
on the enclosing function. It does NOT synthesize a wrapping
``scf.for`` — kernels must distribute work themselves via an explicit
per-core loop.

The pass walks ``tt.get_program_id`` directly and uses
``FunctionOpInterface`` to find the enclosing function, so it works on
both ``tt.func`` (pre-ConvertFunctions) and ``func.func`` (post).
"""

import pytest
from conftest import SinglePassTester
from utils_pattern import pattern


class DistributeWorkTester(SinglePassTester):
    """Base for DistributeWork tests.

    Prepends ConvertFunctions by default (most tests want the
    post-conversion shape so they can assert on ``func.func`` etc.).
    Override ``SKIP_CONVERT_FUNCTIONS = True`` to test running on
    ``tt.func`` directly.
    """

    PASS = "add_distribute_work"
    # Default 1D grid across all 32 cores. Override per test for multi-
    # axis kernels (e.g. TestMultiAxisPid uses [32, 1]).
    GRID = [32]
    SKIP_CONVERT_FUNCTIONS = False

    def _build_passes(self, pm):
        from triton._C.libtriton import spyre
        if not self.SKIP_CONVERT_FUNCTIONS:
            # ConvertFunctions turns tt.func → func.func. Not required by
            # DistributeWork any more, but most tests want func.func in
            # the output so their assertions read cleanly.
            spyre.passes.ttir_to_ktdp.add_convert_functions(pm)
        spyre.passes.ttir_to_ktdp.add_distribute_work(pm, self.GRID)


# ---------------------------------------------------------------------------
# TestReplacePidOnly — kernel with a single 1D pid and no kernel-side loop.
#
# The pass replaces pid with ktdp.get_compute_tile_id and does NOT
# synthesize a wrapping scf.for. Kernels that actually want "one program
# per block" distribution must express it as an explicit per-core loop.
# ---------------------------------------------------------------------------

class TestReplacePidOnly(DistributeWorkTester):

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            %c1024 = arith.constant 1024 : i32
            %offset = arith.muli %pid, %c1024 : i32
            tt.return
          }
        }
        """)

    @pattern("program-id-1d", category="distribution", example=[
        "pid = tl.program_id(0)",
        "# use pid to compute this core's slice of work",
        "offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
    ])
    def test_no_scf_for_synthesized(self):
        """Replace a 1-D ``tt.get_program_id`` with a compute tile id.

        A 1-D kernel (single ``program_id x``) gets exactly one
        ``ktdp.get_compute_tile_id`` and one ``arith.index_cast`` (index → i32).
        The pass does **not** synthesize a wrapping ``scf.for`` — kernels must
        express their own per-core work loop.
        """
        self.assert_absent("scf.for")

    def test_compute_tile_id_present(self):
        self.assert_present("ktdp.get_compute_tile_id")

    def test_single_compute_tile_id(self):
        """Exactly one ktdp.get_compute_tile_id regardless of pid count."""
        self.assert_count("ktdp.get_compute_tile_id", 1, cmp="eq")

    def test_program_id_removed(self):
        self.assert_absent("tt.get_program_id")

    def test_index_cast_present(self):
        """Pass inserts arith.index_cast (index → i32) for each axis used."""
        self.assert_present("arith.index_cast")


# ---------------------------------------------------------------------------
# TestExplicitLoop — kernel already contains its own distribution loop.
#
# Pass should replace pid and leave the existing scf.for alone.
# ---------------------------------------------------------------------------

class TestExplicitLoop(DistributeWorkTester):

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            %c64 = arith.constant 64 : i32
            %c1024 = arith.constant 1024 : i32
            %start = arith.muli %pid, %c64 : i32
            scf.for %i = %c0 to %c64 step %c1 : i32 {
              %offset = arith.addi %start, %i : i32
              %byte_off = arith.muli %offset, %c1024 : i32
              scf.yield
            }
            tt.return
          }
        }
        """)

    def test_single_loop(self):
        """Exactly one scf.for — the kernel's own loop, untouched."""
        self.assert_count("scf.for", 1, cmp="eq")

    def test_compute_tile_id_present(self):
        self.assert_present("ktdp.get_compute_tile_id")

    def test_program_id_removed(self):
        self.assert_absent("tt.get_program_id")


# ---------------------------------------------------------------------------
# TestRunsOnTtFunc — the pass works on tt.func, not just func.func.
#
# Walks tt.get_program_id directly and stamps grid via
# FunctionOpInterface, so ConvertFunctions can run in either order
# (or not at all).
# ---------------------------------------------------------------------------

class TestRunsOnTtFunc(DistributeWorkTester):
    SKIP_CONVERT_FUNCTIONS = True   # leave tt.func as tt.func

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            tt.return
          }
        }
        """)

    def test_pid_rewritten_on_tt_func(self):
        self.assert_present("ktdp.get_compute_tile_id")
        self.assert_absent("tt.get_program_id")

    def test_grid_on_tt_func(self):
        ir_text = str(self.mod)
        assert "grid = [32]" in ir_text, (
            f"expected 'grid = [32]' on tt.func; got:\n{ir_text}"
        )


# ---------------------------------------------------------------------------
# TestNoProgramId — function without tt.get_program_id.
#
# Single-program kernel: no ktdp.get_compute_tile_id, but the function
# still gets a grid attribute — `grid = [1]` — so downstream consumers
# don't have to distinguish "no grid attr yet" from "grid attr explicitly
# set to single-program".
# ---------------------------------------------------------------------------

class TestNoProgramId(DistributeWorkTester):

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %c0 = arith.constant 0 : i32
            tt.return
          }
        }
        """)

    def test_no_compute_tile_id(self):
        """No pids → no ktdp.get_compute_tile_id emitted."""
        self.assert_absent("ktdp.get_compute_tile_id")

    def test_single_program_grid(self):
        """Grid is stamped as [1] — single program, single core."""
        assert "grid = [1]" in str(self.mod), (
            f"expected 'grid = [1]' on function; got:\n{str(self.mod)}"
        )


# ---------------------------------------------------------------------------
# TestGridAttribute — grid attribute reflects the list passed to the pass.
#
# Parametrized over a few 1D grids (the rank must match the kernel's pid
# dimensionality; this kernel reads only axis 0, so grids are 1-element).
# ---------------------------------------------------------------------------

class TestGridAttribute(DistributeWorkTester):

    @pytest.mark.parametrize("grid", [[1], [32], [64]])
    def test_grid_reflects_option(self, grid):
        self.GRID = grid
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            tt.return
          }
        }
        """)
        expected = "grid = [" + ", ".join(str(x) for x in grid) + "]"
        assert expected in str(self.mod), (
            f"expected '{expected}' in IR; got:\n{str(self.mod)}"
        )


# ---------------------------------------------------------------------------
# TestMultiAxisPid — 2D kernel reading pid on axes 0 AND 1.
#
# The variadic ktdp.get_compute_tile_id returns one index per grid dim;
# one shared op + per-axis i32 casts replace both pids.
# ---------------------------------------------------------------------------

class TestMultiAxisPid(DistributeWorkTester):
    # 2D kernel: caller must pass a 2-element grid (prod = 32 cores).
    GRID = [32, 1]

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %px = tt.get_program_id x : i32
            %py = tt.get_program_id y : i32
            %sum = arith.addi %px, %py : i32
            tt.return
          }
        }
        """)

    @pattern("program-id-2d", category="distribution", example=[
        "pid_x = tl.program_id(0)",
        "pid_y = tl.program_id(1)",
        "# both axes share one underlying tile-id op after lowering",
        "row_offset = pid_x * BLOCK_M",
        "col_offset = pid_y * BLOCK_N",
    ])
    def test_single_ktdp_op(self):
        """Replace 2-D ``tt.get_program_id`` (x and y) with a single tile id op.

        When a kernel reads both ``program_id x`` and ``program_id y``, the
        pass emits one variadic ``ktdp.get_compute_tile_id`` that returns two
        index values — one per grid dimension — and one ``arith.index_cast``
        per axis to recover the i32 program id.
        """
        self.assert_count("ktdp.get_compute_tile_id", 1, cmp="eq")

    def test_both_pids_removed(self):
        self.assert_absent("tt.get_program_id")

    def test_two_index_casts(self):
        """One arith.index_cast per axis — two casts for x and y."""
        self.assert_count("arith.index_cast", 2, cmp="eq")

    def test_ktdp_emits_two_results(self):
        """The variadic op's result type list has two index entries."""
        ir_text = str(self.mod)
        # Match "ktdp.get_compute_tile_id : index, index" (allowing for SSA
        # name / whitespace variance). The key property is two ": index"
        # components in the result type list.
        assert "ktdp.get_compute_tile_id" in ir_text
        # Extract the single line holding the op to avoid matching elsewhere.
        line = next(
            l for l in ir_text.split("\n") if "ktdp.get_compute_tile_id" in l
        )
        assert "index, index" in line, (
            f"expected variadic result 'index, index' in op line; got:\n{line}"
        )

    def test_grid_verbatim(self):
        """Grid attribute matches the per-axis list passed in verbatim."""
        assert "grid = [32, 1]" in str(self.mod)


# ---------------------------------------------------------------------------
# TestSameAxisTwice — two pid reads on the same axis collapse to one cast.
#
# Rare Triton pattern (the compiler usually CSEs it) but legal. Both
# uses should be rewired to the same i32 cast — i.e. exactly one cast
# emitted despite two pid ops, because both are on axis 0.
# ---------------------------------------------------------------------------

class TestSameAxisTwice(DistributeWorkTester):

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %pid_a = tt.get_program_id x : i32
            %pid_b = tt.get_program_id x : i32
            %sum = arith.addi %pid_a, %pid_b : i32
            tt.return
          }
        }
        """)

    def test_single_ktdp_op(self):
        self.assert_count("ktdp.get_compute_tile_id", 1, cmp="eq")

    def test_single_index_cast(self):
        """Two pids on the same axis share one cast."""
        self.assert_count("arith.index_cast", 1, cmp="eq")

    def test_both_pids_removed(self):
        self.assert_absent("tt.get_program_id")


# ---------------------------------------------------------------------------
# TestReplacePidOnlyArity — 1D pid → single-result ktdp op
#
# Complement to TestMultiAxisPid::test_ktdp_emits_two_results. For a
# kernel that reads only axis 0, the emitted variadic op has a single
# index result (prints as ": index"), not "index, index".
# ---------------------------------------------------------------------------

class TestReplacePidOnlyArity(DistributeWorkTester):

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            tt.return
          }
        }
        """)

    def test_ktdp_single_result(self):
        ir_text = str(self.mod)
        line = next(
            l for l in ir_text.split("\n") if "ktdp.get_compute_tile_id" in l
        )
        # Exactly one " : index" after the op, not "index, index".
        assert ": index" in line and "index, index" not in line, (
            f"expected single-result ': index'; got:\n{line}"
        )


# ---------------------------------------------------------------------------
# TestAxisZ — kernel reads tl.program_id(2) (axis z).
#
# numDims = max axis + 1 = 3. Caller must pass a 3-element grid. The
# variadic ktdp op produces three index results; the cast for axis 0
# and axis 1 are emitted but unused (no pid reads them). The grid is
# [1, 1, 32] so all cores partition axis z.
# ---------------------------------------------------------------------------

class TestAxisZ(DistributeWorkTester):
    GRID = [1, 1, 32]

    def setup_method(self):
        # All three axes must be read — the pass enforces dense axes
        # from 0 (see TestAxesNonDense). x and y are read even though
        # this kernel's grid is 1x1x32, because skipping them would
        # trip invariant (b).
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %px = tt.get_program_id x : i32
            %py = tt.get_program_id y : i32
            %pz = tt.get_program_id z : i32
            tt.return
          }
        }
        """)

    def test_ktdp_three_results(self):
        ir_text = str(self.mod)
        line = next(
            l for l in ir_text.split("\n") if "ktdp.get_compute_tile_id" in l
        )
        assert "index, index, index" in line, (
            f"expected three-result variadic op; got:\n{line}"
        )

    def test_grid_verbatim(self):
        assert "grid = [1, 1, 32]" in str(self.mod)

    def test_pid_removed(self):
        self.assert_absent("tt.get_program_id")


# ---------------------------------------------------------------------------
# TestGridRankMismatchTooShort — 2D kernel with 1D grid → error.
#
# The pass can't guess how to partition a 2D kernel given a 1D grid;
# it emits a diagnostic and fails the run. Caught via pytest.raises
# and the assert_stderr capfd helper (same pattern as
# test_lower_compute_ops / test_lower_desc_memory negative tests).
# ---------------------------------------------------------------------------

class TestGridRankMismatchTooShort(DistributeWorkTester):
    GRID = [32]  # 1D, but kernel reads 2 axes → should error

    def test_errors(self, capfd):
        """Grid rank lower than the kernel's dimensionality.

        The caller supplied a 1D grid, but the kernel reads two
        program-id axes. There is no coherent way to partition a 2D
        kernel across a 1D grid — the pass refuses rather than guess.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @kernel(%arg0: !tt.ptr<f32>) {
                %px = tt.get_program_id x : i32
                %py = tt.get_program_id y : i32
                tt.return
              }
            }
            """)
        # Pin to invariant (c): DistributeWork must name both the
        # caller's grid rank and the kernel's dimensionality, and tell
        # the user which axes the kernel is reading.
        self.assert_stderr(capfd,
                           "DistributeWork",
                           "grid rank 1",
                           "does not match",
                           "dimensionality 2",
                           "axes 0..1")


# ---------------------------------------------------------------------------
# TestGridRankMismatchTooLong — 1D kernel with 2D grid → error.
#
# Symmetric case: caller declared more axes than the kernel reads.
# Same error path; different numbers.
# ---------------------------------------------------------------------------

class TestGridRankMismatchTooLong(DistributeWorkTester):
    GRID = [32, 1]  # 2D, but kernel reads 1 axis → should error

    def test_errors(self, capfd):
        """Grid rank higher than the kernel's dimensionality.

        Symmetric to the too-short case: the caller declared more grid
        axes than the kernel actually reads. Accepting this would
        silently leave axes unused — the pass refuses so the mismatch
        is visible at compile time.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @kernel(%arg0: !tt.ptr<f32>) {
                %px = tt.get_program_id x : i32
                tt.return
              }
            }
            """)
        # Pin to invariant (c): caller's grid rank > kernel's
        # dimensionality. Diagnostic should name both numbers and
        # tell the user the kernel only touches axis 0.
        self.assert_stderr(capfd,
                           "DistributeWork",
                           "grid rank 2",
                           "does not match",
                           "dimensionality 1",
                           "axes 0..0")


# ---------------------------------------------------------------------------
# TestMultipleFunctions — module with two functions sharing one grid.
#
# The pass takes one grid per run, so both functions must agree on
# dimensionality. Each function gets its own ktdp op and its own grid
# attribute. Per-function grid override is a follow-up.
# ---------------------------------------------------------------------------

class TestMultipleFunctions(DistributeWorkTester):

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel_a(%arg0: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            tt.return
          }
          tt.func @kernel_b(%arg0: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            tt.return
          }
        }
        """)

    def test_both_functions_have_grid(self):
        """Both tt.func ops get the grid attribute stamped."""
        ir_text = str(self.mod)
        # Two tt.func declarations, each carrying the same grid.
        assert ir_text.count("grid = [32]") == 2, (
            f"expected two 'grid = [32]' attributes; got:\n{ir_text}"
        )

    def test_two_ktdp_ops(self):
        """One ktdp.get_compute_tile_id per function (not shared across)."""
        self.assert_count("ktdp.get_compute_tile_id", 2, cmp="eq")

    def test_all_pids_removed(self):
        self.assert_absent("tt.get_program_id")


# ---------------------------------------------------------------------------
# TestMultiAxisOnTtFunc — 2D kernel inside tt.func (no ConvertFunctions).
#
# Pairs TestMultiAxisPid (func.func) with TestRunsOnTtFunc (1D tt.func).
# Confirms multi-axis handling is function-op-agnostic.
# ---------------------------------------------------------------------------

class TestMultiAxisOnTtFunc(DistributeWorkTester):
    SKIP_CONVERT_FUNCTIONS = True
    GRID = [32, 1]

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %px = tt.get_program_id x : i32
            %py = tt.get_program_id y : i32
            %sum = arith.addi %px, %py : i32
            tt.return
          }
        }
        """)

    def test_ktdp_two_results_on_tt_func(self):
        ir_text = str(self.mod)
        line = next(
            l for l in ir_text.split("\n") if "ktdp.get_compute_tile_id" in l
        )
        assert "index, index" in line

    def test_grid_on_tt_func(self):
        assert "grid = [32, 1]" in str(self.mod)

    def test_pids_removed(self):
        self.assert_absent("tt.get_program_id")


# ---------------------------------------------------------------------------
# TestNumProgramsFold1D — tt.get_num_programs folds to arith.constant.
#
# The kernel reads both tl.program_id(0) and tl.num_programs(0). The
# pass folds the num_programs to arith.constant grid[0] : i32 using the
# same grid it stamps on the function.
# ---------------------------------------------------------------------------

class TestNumProgramsFold1D(DistributeWorkTester):
    GRID = [32]

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %pid = tt.get_program_id x : i32
            %n = tt.get_num_programs x : i32
            %sum = arith.addi %pid, %n : i32
            tt.return
          }
        }
        """)

    def test_num_programs_removed(self):
        self.assert_absent("tt.get_num_programs")

    @pattern("num-programs-fold", category="distribution", example=[
        "pid = tl.program_id(0)",
        "num_cores = tl.num_programs(0)  # folded to grid size constant at compile time",
        "num_tiles = tl.cdiv(N, BLOCK_SIZE)",
        "tiles_per_core = tl.cdiv(num_tiles, num_cores)",
        "start = pid * tiles_per_core",
    ])
    def test_addi_rhs_is_constant_32(self):
        """``tt.get_num_programs`` is folded to the grid size constant.

        ``tt.get_num_programs x`` returns the number of cores on axis 0.
        With a 1-D grid of 32, the pass folds the op away and replaces
        every use with ``arith.constant 32 : i32``.
        """
        self.assert_operand("arith.addi", 1, value=32,
                            defined_by="arith.constant")


# ---------------------------------------------------------------------------
# TestNumProgramsFold2D — one num_programs fold per axis.
#
# 2D kernel reads num_programs on both axes with different grid sizes;
# each axis folds to its own constant value.
# ---------------------------------------------------------------------------

class TestNumProgramsFold2D(DistributeWorkTester):
    GRID = [16, 2]

    def setup_method(self):
        # Two muli ops wire pid×nprog per axis; each muli's RHS is the
        # folded constant for its axis, so we can read the axis-0 fold
        # and axis-1 fold off the two distinct muli ops.
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %px = tt.get_program_id x : i32
            %py = tt.get_program_id y : i32
            %nx = tt.get_num_programs x : i32
            %ny = tt.get_num_programs y : i32
            %a = arith.muli %px, %nx : i32
            %b = arith.muli %py, %ny : i32
            tt.return
          }
        }
        """)

    def test_num_programs_removed(self):
        self.assert_absent("tt.get_num_programs")

    def test_axis0_folded(self):
        """Some muli has a constant-16 RHS — the axis-0 nprog fold."""
        self.assert_operand("arith.muli", 1, value=16,
                            defined_by="arith.constant")

    def test_axis1_folded(self):
        """Some muli has a constant-2 RHS — the axis-1 nprog fold."""
        self.assert_operand("arith.muli", 1, value=2,
                            defined_by="arith.constant")


# ---------------------------------------------------------------------------
# TestNumProgramsWithoutPidErrors — invariant (a): nprog without pid.
#
# A kernel that reads num_programs but never program_id is almost
# certainly a bug (no per-core branch can use the answer). The pass
# emits a diagnostic and fails.
# ---------------------------------------------------------------------------

class TestNumProgramsWithoutPidErrors(DistributeWorkTester):
    GRID = [32]

    @pattern("num-programs-fold", category="distribution", negative=True, example=[
        "# Missing tl.program_id — tl.num_programs alone has nothing to act on",
        "num_cores = tl.num_programs(0)",
        "result = do_something(num_cores)  # per-core location is unknown",
    ])
    def test_errors(self, capfd):
        """Kernel reads `tt.get_num_programs` without any `tt.program_id`.

        A kernel that asks how many programs there are but never
        locates itself in the grid has nothing to do with the answer —
        no per-core branch can use it. The pass flags this as a likely
        bug rather than silently generating unreachable code.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @kernel(%arg0: !tt.ptr<f32>) {
                %n = tt.get_num_programs x : i32
                tt.return
              }
            }
            """)
        # Pin the diagnostic to invariant (a) specifically — the pass
        # must name both ops in the message, identify DistributeWork,
        # and explain the reason. Loose substrings here would happily
        # accept a rank-mismatch or non-dense diagnostic instead.
        self.assert_stderr(capfd,
                           "DistributeWork",
                           "reads tt.get_num_programs",
                           "never tt.get_program_id",
                           "without locating itself in the grid")


# ---------------------------------------------------------------------------
# TestAxesNonDense — invariant (b): axes read must be dense from 0.
#
# Kernel reads axes 0 and 2 but skips 1 → pass rejects it. Previously
# such a kernel would have been accepted with axis 1 silently unused.
# ---------------------------------------------------------------------------

class TestAxesNonDense(DistributeWorkTester):
    GRID = [32, 1, 1]

    def test_errors(self, capfd):
        """Grid axes read non-densely — a gap in the sequence.

        The kernel reads axes 0 and 2 but skips axis 1. Dense indexing
        from 0 is required so the per-axis partition has a single
        continuous meaning; a gap would make axis 1 silently unused
        while the user believes the work is spread across three axes.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @kernel(%arg0: !tt.ptr<f32>) {
                %px = tt.get_program_id x : i32
                %pz = tt.get_program_id z : i32
                tt.return
              }
            }
            """)
        # Pin the diagnostic to invariant (b): the pass must identify
        # DistributeWork, name the specific skipped axis, and report
        # the highest axis read so the user can see the gap.
        self.assert_stderr(capfd,
                           "DistributeWork",
                           "reads grid axes non-densely",
                           "axis 1 is skipped",
                           "highest axis read is 2")


# ---------------------------------------------------------------------------
# TestNumProgramsBumpsDimensionality — num_programs counts toward rank.
#
# A kernel that reads pid on axis 0 only but num_programs on axis 1
# has kernel dimensionality 2 (both ops contribute). GRID must be 2D
# to match; a 1D GRID produces the rank-mismatch error.
# ---------------------------------------------------------------------------

class TestNumProgramsBumpsDimensionality(DistributeWorkTester):
    GRID = [16, 2]

    def setup_method(self):
        self.run("""
        module {
          tt.func @kernel(%arg0: !tt.ptr<f32>) {
            %px = tt.get_program_id x : i32
            %ny = tt.get_num_programs y : i32
            %r = arith.muli %px, %ny : i32
            tt.return
          }
        }
        """)

    def test_ny_folded_to_grid1(self):
        """muli's RHS is the axis-1 nprog fold → grid[1] = 2."""
        self.assert_operand("arith.muli", 1, value=2,
                            defined_by="arith.constant")

    def test_num_programs_removed(self):
        self.assert_absent("tt.get_num_programs")

    def test_ktdp_two_results(self):
        """numDims is 2 (pid axis 0 + nprog axis 1) → ktdp has two results."""
        ir_text = str(self.mod)
        line = next(
            l for l in ir_text.split("\n") if "ktdp.get_compute_tile_id" in l
        )
        assert "index, index" in line
