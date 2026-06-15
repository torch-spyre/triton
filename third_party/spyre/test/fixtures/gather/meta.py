"""SIGNATURE + VARIANTS + reference oracle + input generators for gather.

Exercises ``tl.descriptor_gather`` — indirect row-indexed loads via a
1D index tensor.  The compile path lowers to
``ktdp.construct_indirect_access_tile`` + ``ktdp.load`` (see
``docs/gather.md``).

Vocabulary (matches ``kernel.py`` and ``docs/tiling_concepts.md``):

  - ``M``, ``N``         — source matrix dims (rows, cols)
  - ``K_INDICES``        — number of rows to gather
  - ``BLOCK_COLS``       — column slice width per gathered row
  - ``BLOCK_ROWS``       — row-tile size, 2D variants only (single-program
                           variants gather all ``K_INDICES`` rows in one
                           shot, so they have no row-tile size)
  - ``y_offset``         — starting column of the slice, single-program
                           variants only; the kernel reads
                           ``source[idx[i], y_offset : y_offset + BLOCK_COLS]``

Two kernel families share this fixture:

  - **single-program** (``gather_kernel``) — one kernel invocation
    consumes the whole index array. ``parallel: False`` — no
    ``tl.program_id``, DistributeWork is a no-op. Variants:
      - ``default`` — sanity case with non-zero ``y_offset``.
      - **edge-case set** — six variants that each pin a specific
        edge-case bug class (zero offset, full row, minimum legal sizes,
        slice ending at row edge, wider slice, larger K_INDICES).

  - **2D-tiled** (``gather_2d_kernel``) — tiled across a 2D core grid.
    ``tl.program_id(0)`` and ``tl.program_id(1)`` both active; each
    core runs an inner ``scf.for`` over its row-tile chunk. Variants:
      - ``2d``             — small source matrix (M=1024, N=128).
      - ``2d_large_table`` — same distribution at larger source dims
                              (M=4096, N=256), per-core tile count
                              unchanged.
"""

import functools

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input makers
# ---------------------------------------------------------------------------

def _make_inputs(
    M: int, N: int, K_INDICES: int, BLOCK_COLS: int, y_offset: int,
    *, seed: int, allow_duplicates: bool,
) -> dict:
    """Shared input builder for both variants.

    Generates a random source matrix and a K_INDICES-element row-index
    array.  ``allow_duplicates`` controls whether the same row may appear
    twice (the realistic case for batched embedding lookup / paged KV;
    the unique-index case stresses the indirect-load mechanic without
    aliasing).

    ``y_offset`` does not influence input *generation* but is stashed
    in the returned dict so the oracle can read it without needing a
    separate kwarg path through ``test_numerical``. The same value is
    also threaded through ``run_cpu`` as a kernel runtime arg via the
    ``params``/``runtime_scalars`` flow.
    """
    rng = np.random.default_rng(seed)
    in_data = rng.standard_normal((M, N)).astype(np.float32)
    if allow_duplicates:
        idx_data = rng.integers(0, M, size=(K_INDICES,)).astype(np.int32)
    else:
        # Sample without replacement to force unique indices (sanity case).
        idx_data = rng.choice(M, size=K_INDICES, replace=False).astype(np.int32)
    out_data = np.zeros((K_INDICES, BLOCK_COLS), dtype=np.float32)
    return {
        "in_ptr":   in_data,
        "out_ptr":  out_data,
        "idx_ptr":  idx_data,
        "y_offset": y_offset,
    }


def make_inputs(
    M: int, N: int, K_INDICES: int, BLOCK_COLS: int, y_offset: int,
    **_unused,
) -> dict:
    """Default variant inputs: unique indices, fixed seed."""
    return _make_inputs(
        M, N, K_INDICES, BLOCK_COLS, y_offset,
        seed=42, allow_duplicates=False,
    )


# Per-variant input makers for the edge-case variants below. Each gets a
# distinct seed so failures stay reproducible and the variants exercise
# disjoint random matrices (a bug that depends on a particular value
# distribution will not be hidden by all variants reusing seed=42).

def make_inputs_y_offset_zero(M, N, K_INDICES, BLOCK_COLS, y_offset,
                              **_unused) -> dict:
    return _make_inputs(M, N, K_INDICES, BLOCK_COLS, y_offset,
                        seed=1001, allow_duplicates=False)


def make_inputs_full_row(M, N, K_INDICES, BLOCK_COLS, y_offset,
                         **_unused) -> dict:
    return _make_inputs(M, N, K_INDICES, BLOCK_COLS, y_offset,
                        seed=1002, allow_duplicates=False)


def make_inputs_min_block_cols(M, N, K_INDICES, BLOCK_COLS, y_offset,
                               **_unused) -> dict:
    return _make_inputs(M, N, K_INDICES, BLOCK_COLS, y_offset,
                        seed=1003, allow_duplicates=True)


def make_inputs_slice_at_end(M, N, K_INDICES, BLOCK_COLS, y_offset,
                             **_unused) -> dict:
    return _make_inputs(M, N, K_INDICES, BLOCK_COLS, y_offset,
                        seed=1004, allow_duplicates=False)


def make_inputs_wide_slice(M, N, K_INDICES, BLOCK_COLS, y_offset,
                           **_unused) -> dict:
    return _make_inputs(M, N, K_INDICES, BLOCK_COLS, y_offset,
                        seed=1005, allow_duplicates=False)


def make_inputs_large_k(M, N, K_INDICES, BLOCK_COLS, y_offset,
                        **_unused) -> dict:
    return _make_inputs(M, N, K_INDICES, BLOCK_COLS, y_offset,
                        seed=1006, allow_duplicates=True)


def make_inputs_spyre(M, N, K_INDICES, BLOCK_COLS, y_offset,
                      **_unused) -> dict:
    """fp16 inputs for the Spyre physical-layout variant."""
    rng = np.random.default_rng(seed=42)
    in_data = rng.standard_normal((M, N)).astype(np.float16)
    idx_data = rng.choice(M, size=K_INDICES, replace=False).astype(np.int32)
    out_data = np.zeros((K_INDICES, N), dtype=np.float16)
    return {
        "in_ptr":   in_data,
        "out_ptr":  out_data,
        "idx_ptr":  idx_data,
        "y_offset": y_offset,
    }


def run_spyre(inputs: dict) -> np.ndarray:
    """NumPy oracle for the Spyre variant: gather full rows, fp16."""
    in_data = inputs["in_ptr"]
    idx = inputs["idx_ptr"]
    return in_data[idx, :]


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: gather rows from ``in_ptr`` at ``idx_ptr`` positions.

    Mirrors the kernel exactly:
        result[i, :] = in[idx[i], y_offset : y_offset + BLOCK_COLS]

    ``y_offset`` and ``BLOCK_COLS`` come from ``inputs`` — ``y_offset``
    is stashed there by ``_make_inputs``, and ``BLOCK_COLS`` is
    recovered from the output buffer's column count. The oracle thus
    works for every variant without per-variant plumbing.
    """
    in_data = inputs["in_ptr"]
    idx = inputs["idx_ptr"]
    y_offset = inputs["y_offset"]
    block_cols = inputs["out_ptr"].shape[1]
    return in_data[idx, y_offset:y_offset + block_cols]


# ---------------------------------------------------------------------------
# 2D-tiled variants — input makers + oracle
#
# The 2D kernel has no ``y_offset`` argument and writes the full row
# width of every gathered row, so the oracle and input maker differ
# from the single-program variants. The output buffer is shaped
# [K_INDICES, N], not [K_INDICES, BLOCK_COLS].
# ---------------------------------------------------------------------------

def _make_inputs_2d(
    M: int, N: int, K_INDICES: int,
    *, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    in_data = rng.standard_normal((M, N)).astype(np.float32)
    idx_data = rng.integers(0, M, size=(K_INDICES,)).astype(np.int32)
    out_data = np.zeros((K_INDICES, N), dtype=np.float32)
    return {
        "in_ptr":  in_data,
        "out_ptr": out_data,
        "idx_ptr": idx_data,
    }


def make_inputs_2d(
    M: int, N: int, K_INDICES: int,
    BLOCK_ROWS: int, BLOCK_COLS: int,
) -> dict:
    """Inputs for the ``2d`` variant. Block kwargs are part of the
    param set so the framework can pass them uniformly; they do not
    shape the data."""
    del BLOCK_ROWS, BLOCK_COLS
    return _make_inputs_2d(M, N, K_INDICES, seed=42)


def make_inputs_2d_large_table(
    M: int, N: int, K_INDICES: int,
    BLOCK_ROWS: int, BLOCK_COLS: int,
) -> dict:
    """Inputs for the ``2d_large_table`` variant. Distinct seed from
    ``2d`` so the two variants don't exercise literally identical data."""
    del BLOCK_ROWS, BLOCK_COLS
    return _make_inputs_2d(M, N, K_INDICES, seed=123)


def run_2d(inputs: dict) -> np.ndarray:
    """NumPy oracle for the 2D-tiled kernel: gather full rows of ``in``
    at ``idx`` positions. Result shape: ``[len(idx), N]``."""
    return inputs["in_ptr"][inputs["idx_ptr"]]


# ---------------------------------------------------------------------------
# SIGNATURE
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "idx_ptr":    "*i32",
    "y_offset":   "i32",
    "M":          "i32",
    "N":          "i32",
    "K_INDICES":  "i32",
    "BLOCK_COLS": "i32",
}

_DTYPE_MAP = {"fp32": np.float32, "fp16": np.float16}

def _np_dtype(sig, key):
    return _DTYPE_MAP[sig[key].lstrip("*")]

def _sticksize(sig, key):
    return 128 // np.dtype(_np_dtype(sig, key)).itemsize

_SIG_SPYRE = {
    "in_ptr":      "*fp16",
    "out_ptr":     "*fp16",
    "idx_ptr":     "*i32",
    "y_offset":    "i32",
    "M":           "i32",
    "N":           "i32",
    "K_INDICES":   "i32",
    "BLOCK_ROWS":  "constexpr",
    "BLOCK_COLS":  "constexpr",
    "IN_LAYOUT":   "constexpr",
    "OUT_LAYOUT":  "constexpr",
}
_SS = functools.partial(_sticksize, _SIG_SPYRE)

# 2D kernel has no ``y_offset`` argument and adds ``BLOCK_ROWS``.
_SIG_2D = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "idx_ptr":    "*i32",
    "M":          "i32",
    "N":          "i32",
    "K_INDICES":  "i32",
    "BLOCK_ROWS": "i32",
    "BLOCK_COLS": "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

# Structural check shared across all variants: the traced-memory-view
# path in LowerDescriptorMemory must not emit an unrealized_conversion_cast
# for descriptor-loaded indices (the presence of the cast would block
# ktir_cpu execution). This pins the behaviour at the fixture level
# in addition to the single-pass test
# ``test_lower_desc_memory.py::TestDescriptorGather::test_gather_from_descriptor_load_emits_no_cast``.
_EXTRA_CHECKS = lambda t: (
    t.assert_absent("unrealized_conversion_cast"),
    t.assert_present("ktdp.construct_indirect_access_tile"),
)


VARIANTS = {
    "default": {
        # Sanity case: small source matrix, unique indices, non-zero
        # y_offset to exercise the column-slicing path.  Non-zero
        # y_offset matters because the lowered indirect access tile
        # uses y_offset as a captured variable in the direct-dimension
        # subscript map (``col = y_offset + d1``); a y_offset=0 test
        # would mask any bug in that subscript.
        #
        # ``BLOCK_COLS`` must be a power of two (Triton frontend
        # constraint on descriptor block shapes).  With BLOCK_COLS=32
        # and y_offset=16 we read columns [16, 48) of each gathered
        # row — strictly inside [0, N=64).
        "kernel_fn":  kernel.gather_kernel,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_COLS"],
        "params": {
            "M":          [1024],
            "N":          [64],
            "K_INDICES":  [32],
            "BLOCK_COLS": [32],
            "y_offset":   [16],
        },
        "tags":       ["descriptor-gather"],
        "grid":       [32],
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    # ------------------------------------------------------------------
    # Edge-case variants.  Each pins one specific bug class that the
    # default+embedding pair does not cover.  See README.md (the
    # "Variants" and "Constraints" sections) for the rules these have
    # to obey:
    #   * BLOCK_COLS is a power of two and >= 8 for f32
    #   * K_INDICES >= 8
    #   * y_offset + BLOCK_COLS <= N (slice fits in the source row)
    # ------------------------------------------------------------------
    "y_offset_zero": {
        # y_offset=0 case.  The lowered indirect access tile uses
        # ``col = y_offset + d1`` for the direct dimension; if y_offset
        # were dropped (or treated as a baked-in zero) the bug would
        # only show up when y_offset != 0.  This variant pins the
        # y_offset=0 path so a regression that special-cases zero
        # doesn't slip through.
        "kernel_fn":  kernel.gather_kernel,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_COLS"],
        "params": {
            "M":          [256],
            "N":          [32],
            "K_INDICES":  [16],
            "BLOCK_COLS": [16],
            "y_offset":   [0],
        },
        "tags":       ["descriptor-gather"],
        "grid":       [32],
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs_y_offset_zero,
        "output_key": "out_ptr",
    },
    "full_row": {
        # BLOCK_COLS == N (full-row gather).  This is the embedding
        # lookup *special case* the README calls out: y_offset=0,
        # BLOCK_COLS=embedding_dim — gather entire rows, no column
        # slicing.  Catches a bug where the column slice machinery
        # incorrectly assumes a strict subset (e.g. an off-by-one in
        # the slice-end check).
        "kernel_fn":  kernel.gather_kernel,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_COLS"],
        "params": {
            "M":          [128],
            "N":          [16],
            "K_INDICES":  [16],
            "BLOCK_COLS": [16],
            "y_offset":   [0],
        },
        "tags":       ["descriptor-gather"],
        "grid":       [32],
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs_full_row,
        "output_key": "out_ptr",
    },
    "min_block_cols": {
        # Smallest legal sizes per the verifier: K_INDICES=8 (verifier
        # minimum) and BLOCK_COLS=8 (verifier minimum for f32, since
        # 32/bitwidth*8 = 8 when bitwidth=32).  Probes the lower
        # boundary of the legal region.  Per the test rule "test
        # endpoints of a range, not just the interior" — interior cases
        # alone wouldn't catch a bug that triggers only at the
        # smallest legal block.  Allows duplicates so an aliasing bug
        # at the minimum size shows up.
        "kernel_fn":  kernel.gather_kernel,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_COLS"],
        "params": {
            "M":          [64],
            "N":          [64],
            "K_INDICES":  [8],
            "BLOCK_COLS": [8],
            "y_offset":   [32],
        },
        "tags":       ["descriptor-gather"],
        "grid":       [32],
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs_min_block_cols,
        "output_key": "out_ptr",
    },
    "slice_at_end": {
        # y_offset + BLOCK_COLS == N exactly — slice ends at the
        # right edge of the source row.  Catches off-by-one bugs in
        # the column subscript map: an inclusive/exclusive endpoint
        # mix-up would read column N (out of bounds) only at this
        # extreme.  The default variant has y_offset+BLOCK_COLS=48 < N=64,
        # so it leaves slack and would not detect the bug.
        "kernel_fn":  kernel.gather_kernel,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_COLS"],
        "params": {
            "M":          [256],
            "N":          [64],
            "K_INDICES":  [16],
            "BLOCK_COLS": [16],
            "y_offset":   [48],
        },
        "tags":       ["descriptor-gather"],
        "grid":       [32],
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs_slice_at_end,
        "output_key": "out_ptr",
    },
    "wide_slice": {
        # Wider source row (N=256) and larger BLOCK_COLS=128.  Probes
        # whether the gather lowering scales beyond the small
        # BLOCK_COLS values used by default+embedding.  A bug in the
        # tile-shape computation that only manifests above some size
        # (e.g. an int16 truncation, an alignment assumption) would
        # show up here.
        "kernel_fn":  kernel.gather_kernel,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_COLS"],
        "params": {
            "M":          [128],
            "N":          [256],
            "K_INDICES":  [16],
            "BLOCK_COLS": [128],
            "y_offset":   [64],
        },
        "tags":       ["descriptor-gather"],
        "grid":       [32],
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs_wide_slice,
        "output_key": "out_ptr",
    },
    "large_k": {
        # Larger K_INDICES (128) than any other variant.  The kernel
        # is single-program, so this stresses the descriptor_load on
        # the index buffer + the descriptor_gather fan-out at higher
        # row count, without changing the lowering path.  Duplicates
        # allowed so the larger fan-out also exercises aliasing.
        "kernel_fn":  kernel.gather_kernel,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_COLS"],
        "params": {
            "M":          [512],
            "N":          [64],
            "K_INDICES":  [128],
            "BLOCK_COLS": [32],
            "y_offset":   [16],
        },
        "tags":       ["descriptor-gather"],
        "grid":       [32],
        "parallel":   False,
        "reference":  run,
        "inputs":     make_inputs_large_k,
        "output_key": "out_ptr",
    },
    # ------------------------------------------------------------------
    # 2D-tiled variants.  Use ``gather_2d_kernel`` (separate kernel_fn,
    # different SIGNATURE — no ``y_offset``, adds ``BLOCK_ROWS``).
    #
    # Both variants use a 2D grid whose axis sizes multiply to 32 (the
    # hardware core count). The kernel reads them via
    # ``tl.num_programs(0/1)`` — DistributeWork folds those to compile-time
    # constants against the variant's ``grid`` entry here, so no
    # GRID_M/GRID_N constexpr is needed.
    #
    # Axis partition is picked so each core owns an integer number of
    # tiles along each axis (one tile-column per core on N, two
    # row-tiles per core on M — keeps an scf.for inner loop in the IR,
    # which ``test_work_distribution`` requires for parallel kernels).
    #
    # Shape picked so every tile is in-bounds without masking:
    #   m_blocks  = cdiv(K_INDICES, BLOCK_ROWS) = grid[0] * rows_per_core
    #   n_blocks  = cdiv(N, BLOCK_COLS)         = grid[1] * cols_per_core
    # ------------------------------------------------------------------
    "2d": {
        # 64 indices x 128 cols, tiled 8x16 across [4, 8] cores:
        #   m_blocks=8 / grid[0]=4 = 2 row tiles per core
        #   n_blocks=8 / grid[1]=8 = 1 col tile per core
        # x_offsets loaded from idx_desc has shape [BLOCK_ROWS]; Triton
        # frontend requires x_offsets.shape[0] >= 8 for tt.descriptor_gather
        # (see python/triton/language/semantic.py:1144), so BLOCK_ROWS >= 8
        # here.
        #
        # ``parallel: True`` overrides the inherited ``False`` from the
        # ``default`` variant (variants do a shallow {**default, **delta}
        # merge in conftest._build_registry, so unset keys here would
        # otherwise inherit single-program flags). The 2D kernel is
        # multi-program, so test_work_distribution must run.
        "kernel_fn":    kernel.gather_2d_kernel,
        "SIGNATURE":    _SIG_2D,
        "constexpr":    [
            "M", "N", "K_INDICES",
            "BLOCK_ROWS", "BLOCK_COLS",
        ],
        "params": {
            "M":          [1024],
            "N":          [128],
            "K_INDICES":  [64],
            "BLOCK_ROWS": [8],
            "BLOCK_COLS": [16],
        },
        "tags":         ["descriptor-gather", "program-id-2d", "num-programs-fold"],
        "grid":         [4, 8],
        "parallel":     True,
        "reference":    run_2d,
        "inputs":       make_inputs_2d,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "2d_serial": {
        # Same kernel source as ``2d`` but on a 1-core grid. With
        # grid=[1, 1] the kernel's tiling math degenerates: a single
        # program covers all m_blocks * n_blocks output tiles via the
        # inner ``scf.for`` loops (rows_per_core = m_blocks,
        # cols_per_core = n_blocks). ``parallel: False`` skips
        # ``test_work_distribution`` (no multi-program lowering to pin),
        # but ``test_numerical`` still runs and pins that the degenerate
        # tiling path produces the correct output.
        #
        # Same data shape as ``2d`` so the numerical comparison reuses
        # the same NumPy oracle without per-variant plumbing.
        "kernel_fn":    kernel.gather_2d_kernel,
        "SIGNATURE":    _SIG_2D,
        "constexpr":    [
            "M", "N", "K_INDICES",
            "BLOCK_ROWS", "BLOCK_COLS",
        ],
        "params": {
            "M":          [1024],
            "N":          [128],
            "K_INDICES":  [64],
            "BLOCK_ROWS": [8],
            "BLOCK_COLS": [16],
        },
        "tags":         ["descriptor-gather"],
        "grid":         [1, 1],
        "parallel":     False,
        "reference":    run_2d,
        "inputs":       make_inputs_2d,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "2d_large_table": {
        # Same distribution at larger source dims: 64 indices x 256 cols,
        # tiled 8x32 across [4, 8] cores. Same per-core tile counts
        # (2 row tiles, 1 col tile) as ``2d`` — we vary data shape, not
        # the distribution pattern. Confirms the lowering is robust to
        # ~4x larger M and 2x wider N+BLOCK_COLS without changing the
        # number of tiles per core.
        "kernel_fn":    kernel.gather_2d_kernel,
        "SIGNATURE":    _SIG_2D,
        "constexpr":    [
            "M", "N", "K_INDICES",
            "BLOCK_ROWS", "BLOCK_COLS",
        ],
        "params": {
            "M":          [4096],
            "N":          [256],
            "K_INDICES":  [64],
            "BLOCK_ROWS": [8],
            "BLOCK_COLS": [32],
        },
        "tags":         ["descriptor-gather", "program-id-2d", "num-programs-fold"],
        "grid":         [4, 8],
        "parallel":     True,
        "reference":    run_2d,
        "inputs":       make_inputs_2d_large_table,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "2d_large_table_serial": {
        # ``2d_large_table`` data shape on a 1-core grid. Same intent
        # as ``2d_serial``: pin the degenerate tiling path numerically
        # at the larger source dims, no work-distribution check.
        "kernel_fn":    kernel.gather_2d_kernel,
        "SIGNATURE":    _SIG_2D,
        "constexpr":    [
            "M", "N", "K_INDICES",
            "BLOCK_ROWS", "BLOCK_COLS",
        ],
        "params": {
            "M":          [4096],
            "N":          [256],
            "K_INDICES":  [64],
            "BLOCK_ROWS": [8],
            "BLOCK_COLS": [32],
        },
        "tags":         ["descriptor-gather"],
        "grid":         [1, 1],
        "parallel":     False,
        "reference":    run_2d,
        "inputs":       make_inputs_2d_large_table,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    # ------------------------------------------------------------------
    # Spyre physical-layout variant.
    # in_desc [M, N] annotated stick-on-N; out_desc [K_INDICES, BLOCK_COLS]
    # annotated stick-on-N. idx_desc is not annotated.
    # Loops over all N columns in BLOCK_COLS-wide sticks so the full row
    # is gathered into out_ptr[K_INDICES, N].
    # ------------------------------------------------------------------
    "spyre_stick": {
        "kernel_fn":  kernel.gather_kernel_spyre,
        "SIGNATURE":  _SIG_SPYRE,
        "constexpr":  ["M", "N", "K_INDICES", "BLOCK_ROWS", "BLOCK_COLS",
                       "IN_LAYOUT", "OUT_LAYOUT"],
        "params": {
            "M":          [256],
            # N spans multiple source N-sticks (256/64 = 4) so the physical
            # in_desc first dim is > 1 and the gather's floordiv/mod actually
            # selects a source stick rather than collapsing to stick 0.
            "N":          [256],
            "K_INDICES":  [32],
            "BLOCK_ROWS": [32],
            # BLOCK_COLS == STICK_SIZE (64): one stick per gather call;
            # the col_stick loop runs N//BLOCK_COLS = 4 times.
            "BLOCK_COLS": [128],
            # y_offset unused by the kernel (full-row gather), but kept in
            # the signature as a runtime scalar to satisfy run_cpu plumbing.
            "y_offset":   [0],
            # in_ptr [M, N]: stick-on-N → [N//_S, M, _S]
            "IN_LAYOUT":  [[(1, "floordiv", _SS("in_ptr")), 0, (1, "mod", _SS("in_ptr"))]],
            # out_ptr [K_INDICES, N]: stick-on-N → [N//_S, K_INDICES, _S]
            "OUT_LAYOUT": [[(1, "floordiv", _SS("out_ptr")), 0, (1, "mod", _SS("out_ptr"))]],
        },
        "tags":       ["descriptor-gather", "spyre-tensor-layout"],
        "grid":       [1],
        "parallel":   False,
        "reference":  run_spyre,
        "inputs":     make_inputs_spyre,
        "output_key": "out_ptr",
        "extra_checks": lambda t: (
            t.assert_absent("tt.spyre_tensor_layout"),
            # in_desc is physicalized stick-on-N → rank-3 [N//stick, M, stick]
            # view = [4, 256, 64] (4 source N-sticks).
            t.assert_result_type(
                "ktdp.construct_memory_view", "4x256x64xf16"),
            # Each gather reads BLOCK_COLS=128 (2 sticks) for K_INDICES=32
            # rows, so the indirect access tile is rank-3 [2, 32, 64].
            t.assert_result(
                "ktdp.construct_indirect_access_tile", shape=[2, 32, 64]),
            # The rank-3 load result flows straight into the rank-3 store sink,
            # so no insert_slice is synthesized (ranks already match).
            t.assert_absent("tensor.insert_slice"),
        ),
    },
}
