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
    twice (the realistic case for batched embedding lookup; the
    unique-index case stresses the indirect-load mechanic without
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
# 1D-source gather — input maker + oracle
#
# The 1D kernel takes a 1D source vector and produces a 1D output. The
# rank-2 ``[K, 1]`` shape used internally by the kernel descriptor is
# an implementation detail to satisfy the gather verifier; the user-facing
# buffers are 1D.
# ---------------------------------------------------------------------------

def make_inputs_1d(K: int, K_INDICES: int, BLOCK_ROWS: int) -> dict:
    """Inputs for the ``1d`` variant. Source and output are 1D ``[K]``
    and ``[K_INDICES]`` respectively. Distinct seed (2001) so this
    variant doesn't share random data with any other gather variant."""
    del BLOCK_ROWS
    rng = np.random.default_rng(2001)
    in_data = rng.standard_normal((K,)).astype(np.float32)
    idx_data = rng.integers(0, K, size=(K_INDICES,)).astype(np.int32)
    out_data = np.zeros((K_INDICES,), dtype=np.float32)
    return {
        "in_ptr":  in_data,
        "out_ptr": out_data,
        "idx_ptr": idx_data,
    }


def run_1d(inputs: dict) -> np.ndarray:
    """NumPy oracle for the 1D-source kernel: ``out[i] = in[idx[i]]``."""
    return inputs["in_ptr"][inputs["idx_ptr"]]


# ---------------------------------------------------------------------------
# Rank-N (N ≥ 3) input makers and oracles
# ---------------------------------------------------------------------------

def _check_rank_n_preconditions(
    K_INDICES: int, *, M: int,
    group_idx: int = None, NUM_GROUPS: int = None,
) -> None:
    """Guard fixture-only preconditions that have no upstream backstop.

    Two preconditions documented in the kernel docstrings — ``K_INDICES >= 8``
    and inner dim is a power of two — are already enforced by the Triton
    frontend (``descriptor_gather`` verifier in ``semantic.py`` and
    ``validate_block_shape`` in ``_utils.py``), so a violating variant
    fails at trace time with a clear message; we don't re-check them here.

    The two below are not enforced anywhere else:

    - ``K_INDICES <= M`` is needed because every rank-N input maker draws
      indices via ``np.random.choice(M, size=K_INDICES, replace=False)``;
      ``K_INDICES > M`` raises a generic numpy ``ValueError`` that does
      not name the fixture-config mistake.
    - ``0 <= group_idx < NUM_GROUPS`` has no backstop at all: ``group_idx``
      flows straight through to ``tt.descriptor_gather``'s ``y_offset``
      capture, with no verifier or lowering-pass bounds check. A violating
      value silently reads out of bounds in the source tensor.

    A partial caller — passing ``group_idx`` without ``NUM_GROUPS`` or vice
    versa — would silently skip the bounds check, so we reject that shape
    explicitly rather than letting it pass.
    """
    assert K_INDICES <= M, (
        f"K_INDICES ({K_INDICES}) must be <= M ({M}) for unique-index "
        f"sampling without replacement"
    )
    assert (group_idx is None) == (NUM_GROUPS is None), (
        f"group_idx and NUM_GROUPS must be passed together or both omitted; "
        f"got group_idx={group_idx}, NUM_GROUPS={NUM_GROUPS}"
    )
    if group_idx is not None:
        assert 0 <= group_idx < NUM_GROUPS, (
            f"group_idx must satisfy 0 <= group_idx < NUM_GROUPS; "
            f"got group_idx={group_idx}, NUM_GROUPS={NUM_GROUPS}"
        )


def make_inputs_3d(
    M: int, BLOCK_SIZE: int, HEAD_DIM: int, K_INDICES: int,
    **_unused,
) -> dict:
    """Inputs for ``gather_3d_kernel``: source ``[M, BLOCK_SIZE, HEAD_DIM]``."""
    _check_rank_n_preconditions(K_INDICES, M=M)
    rng = np.random.default_rng(2001)
    in_data = rng.standard_normal((M, BLOCK_SIZE, HEAD_DIM)).astype(np.float32)
    idx_data = rng.choice(M, size=K_INDICES, replace=False).astype(np.int32)
    out_data = np.zeros((K_INDICES, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
    return {"in_ptr": in_data, "out_ptr": out_data, "idx_ptr": idx_data}


def run_3d(inputs: dict) -> np.ndarray:
    """NumPy oracle for the 3D gather kernel: result[i] = in[idx[i], :, :]."""
    return inputs["in_ptr"][inputs["idx_ptr"]]


def make_inputs_3d_group(
    M: int, NUM_GROUPS: int, HEAD_DIM: int, K_INDICES: int, group_idx: int,
    **_unused,
) -> dict:
    """Inputs for ``gather_3d_group_kernel``: source ``[M, NUM_GROUPS, HEAD_DIM]``.

    ``group_idx`` is stashed in the returned dict so the oracle can read it.
    """
    _check_rank_n_preconditions(
        K_INDICES, M=M, group_idx=group_idx, NUM_GROUPS=NUM_GROUPS,
    )
    rng = np.random.default_rng(2002)
    in_data = rng.standard_normal((M, NUM_GROUPS, HEAD_DIM)).astype(np.float32)
    idx_data = rng.choice(M, size=K_INDICES, replace=False).astype(np.int32)
    out_data = np.zeros((K_INDICES, 1, HEAD_DIM), dtype=np.float32)
    return {
        "in_ptr":    in_data,
        "out_ptr":   out_data,
        "idx_ptr":   idx_data,
        "group_idx": group_idx,
    }


def run_3d_group(inputs: dict) -> np.ndarray:
    """NumPy oracle for the 3D group-gather kernel: result[i, 0, :] = in[idx[i], group_idx, :]."""
    in_data = inputs["in_ptr"]
    idx = inputs["idx_ptr"]
    g = inputs["group_idx"]
    return in_data[idx, g:g + 1, :]


def make_inputs_4d(
    NUM_BLOCKS: int, NUM_GROUPS: int, BLOCK_SIZE: int, INNER_DIM: int,
    K_INDICES: int, group_idx: int,
    **_unused,
) -> dict:
    """Inputs for ``gather_4d_kernel``:
    source ``[NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM]``."""
    _check_rank_n_preconditions(
        K_INDICES, M=NUM_BLOCKS,
        group_idx=group_idx, NUM_GROUPS=NUM_GROUPS,
    )
    rng = np.random.default_rng(2003)
    in_data = rng.standard_normal(
        (NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM)
    ).astype(np.float32)
    idx_data = rng.choice(NUM_BLOCKS, size=K_INDICES, replace=False).astype(np.int32)
    out_data = np.zeros((K_INDICES, 1, BLOCK_SIZE, INNER_DIM), dtype=np.float32)
    return {
        "in_ptr":    in_data,
        "out_ptr":   out_data,
        "idx_ptr":   idx_data,
        "group_idx": group_idx,
    }


def run_4d(inputs: dict) -> np.ndarray:
    """NumPy oracle for the 4D gather kernel: result[i, 0, :, :] = in[idx[i], group_idx, :, :]."""
    in_data = inputs["in_ptr"]
    idx = inputs["idx_ptr"]
    g = inputs["group_idx"]
    return in_data[idx, g:g + 1, :, :]


def make_inputs_scatter_3d(
    M: int, BLOCK_SIZE: int, HEAD_DIM: int, K_INDICES: int,
    **_unused,
) -> dict:
    """Inputs for ``scatter_3d_kernel``.

    ``dst_ptr`` is a zero-initialised destination buffer; ``data_ptr``
    holds the blocks to scatter.  Unique indices so the oracle is
    deterministic (no two blocks write the same destination row).
    """
    _check_rank_n_preconditions(K_INDICES, M=M)
    rng = np.random.default_rng(2004)
    dst = np.zeros((M, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
    src = rng.standard_normal((K_INDICES, BLOCK_SIZE, HEAD_DIM)).astype(np.float32)
    idx_data = rng.choice(M, size=K_INDICES, replace=False).astype(np.int32)
    return {"dst_ptr": dst, "data_ptr": src, "idx_ptr": idx_data}


def run_scatter_3d(inputs: dict) -> np.ndarray:
    """NumPy oracle for the 3D scatter kernel: dst[idx[i], :, :] = data[i, :, :] for each i."""
    dst = inputs["dst_ptr"].copy()
    dst[inputs["idx_ptr"]] = inputs["data_ptr"]
    return dst


def _check_partial_preconditions(
    K_INDICES: int, M: int, NUM_TOKENS: int, TOKEN_BLOCK: int,
) -> None:
    """Guard ``3d_partial`` / ``scatter_3d_partial`` preconditions.

    Augments :func:`_check_rank_n_preconditions` with the divisibility
    invariant the partial-extent fixtures rely on: ``TOKEN_BLOCK`` must
    divide ``NUM_TOKENS`` so the ``scf.for`` sweep over
    ``NUM_BLOCKS = NUM_TOKENS // TOKEN_BLOCK`` windows covers every dim-1
    element exactly once. NUM_TOKENS divisibility has no upstream
    verifier — a violating variant would either skip elements at the tail
    (oracle mismatch) or read out of bounds.
    """
    _check_rank_n_preconditions(K_INDICES, M=M)
    assert NUM_TOKENS % TOKEN_BLOCK == 0, (
        f"TOKEN_BLOCK ({TOKEN_BLOCK}) must divide NUM_TOKENS ({NUM_TOKENS}) "
        f"for exact tiling along dim 1"
    )


def make_inputs_3d_partial(
    M: int, NUM_TOKENS: int, TOKEN_BLOCK: int, HEAD_DIM: int,
    K_INDICES: int,
    **_unused,
) -> dict:
    """Inputs for ``gather_3d_partial_kernel``.

    Output buffer is ``[K_INDICES, NUM_TOKENS, HEAD_DIM]`` — the kernel's
    ``scf.for`` sweep covers every window along dim 1 in order, so the
    full output reconstructs ``in[idx, :, :]``.
    """
    _check_partial_preconditions(K_INDICES, M, NUM_TOKENS, TOKEN_BLOCK)
    rng = np.random.default_rng(2005)
    in_data = rng.standard_normal((M, NUM_TOKENS, HEAD_DIM)).astype(np.float32)
    idx_data = rng.choice(M, size=K_INDICES, replace=False).astype(np.int32)
    out_data = np.zeros((K_INDICES, NUM_TOKENS, HEAD_DIM), dtype=np.float32)
    return {"in_ptr": in_data, "out_ptr": out_data, "idx_ptr": idx_data}


def run_3d_partial(inputs: dict) -> np.ndarray:
    """Oracle for ``gather_3d_partial_kernel``: full-sweep reconstruction.

    Each loop iteration writes one ``[K_INDICES, TOKEN_BLOCK, HEAD_DIM]``
    window of ``in[idx, b*G:(b+1)*G, :]``; concatenating all windows
    yields ``in[idx, :, :]``. Numerical equality with this oracle proves
    every iteration's ``y_offset`` capture landed at the correct dim-1
    window — a regression that pinned ``y_offset`` to a single iteration's
    value would scramble the output.
    """
    return inputs["in_ptr"][inputs["idx_ptr"]]


def make_inputs_scatter_3d_partial(
    M: int, NUM_TOKENS: int, TOKEN_BLOCK: int, HEAD_DIM: int,
    K_INDICES: int,
    **_unused,
) -> dict:
    """Inputs for ``scatter_3d_partial_kernel``.

    ``data_ptr`` is the full ``[K_INDICES, NUM_TOKENS, HEAD_DIM]`` source;
    the kernel sweeps every ``TOKEN_BLOCK``-wide window along dim 1 and
    scatters it into ``dst[M, NUM_TOKENS, HEAD_DIM]`` at the rows named by
    ``idx_ptr``. Unique indices keep the oracle deterministic.
    """
    _check_partial_preconditions(K_INDICES, M, NUM_TOKENS, TOKEN_BLOCK)
    rng = np.random.default_rng(2006)
    dst = np.zeros((M, NUM_TOKENS, HEAD_DIM), dtype=np.float32)
    src = rng.standard_normal((K_INDICES, NUM_TOKENS, HEAD_DIM)).astype(np.float32)
    idx_data = rng.choice(M, size=K_INDICES, replace=False).astype(np.int32)
    return {"dst_ptr": dst, "data_ptr": src, "idx_ptr": idx_data}


def run_scatter_3d_partial(inputs: dict) -> np.ndarray:
    """Oracle for ``scatter_3d_partial_kernel``: ``dst[idx, :, :] = data``.

    The full-sweep loop writes every dim-1 window in order, so the final
    ``dst`` equals the source ``data`` indexed back into the destination
    rows.
    """
    dst = inputs["dst_ptr"].copy()
    dst[inputs["idx_ptr"]] = inputs["data_ptr"]
    return dst

# ---------------------------------------------------------------------------
# rank-2 x_offsets (index-grid) variants — input makers + oracles
#
# These drive ``gather_2d_index_kernel`` / ``gather_scatter_2d_index_kernel``,
# which take a 2-D (S0 x S1) index *grid* rather than a 1-D index list. The
# index buffer is shaped [S0, S1]; the gather result is [S0, S1, BLOCK_COLS].
# See docs/impl-strategy-2d-x-offsets.md.
# ---------------------------------------------------------------------------

def _make_inputs_2d_index(
    M: int, N: int, S0: int, S1: int, BLOCK_COLS: int, y_offset: int,
    *, seed: int, out_shape,
) -> dict:
    """Source matrix + an S0xS1 grid of UNIQUE row indices.

    Unique indices keep the scatter round-trip oracle unambiguous (no
    two grid cells write the same destination row). ``y_offset`` is
    stashed for the oracle and threaded to the kernel as a runtime arg,
    mirroring the 1-D ``_make_inputs``.
    """
    rng = np.random.default_rng(seed)
    in_data = rng.standard_normal((M, N)).astype(np.float32)
    idx_data = rng.choice(M, size=S0 * S1, replace=False)
    idx_data = idx_data.reshape(S0, S1).astype(np.int32)
    out_data = np.zeros(out_shape, dtype=np.float32)
    return {
        "in_ptr":   in_data,
        "out_ptr":  out_data,
        "idx_ptr":  idx_data,
        "y_offset": y_offset,
    }


def make_inputs_2d_index_gather(
    M, N, S0, S1, BLOCK_COLS, y_offset, **_unused,
) -> dict:
    """Gather variant: result tile is [S0, S1, BLOCK_COLS]."""
    return _make_inputs_2d_index(
        M, N, S0, S1, BLOCK_COLS, y_offset,
        seed=2025, out_shape=(S0, S1, BLOCK_COLS),
    )


def run_2d_index_gather(inputs: dict) -> np.ndarray:
    """Oracle for the rank-2-index gather:
        result[i, j, :] = in[idx[i, j], y_offset : y_offset + BLOCK_COLS]

    ``BLOCK_COLS`` is recovered from the output tile's trailing dim, so
    no per-variant plumbing is needed (mirrors the 1-D ``run`` oracle).
    """
    in_data = inputs["in_ptr"]
    idx = inputs["idx_ptr"]                 # [S0, S1]
    y = inputs["y_offset"]
    bc = inputs["out_ptr"].shape[2]         # [S0, S1, BLOCK_COLS]
    # Advanced index on axis 0 by the 2-D grid, basic slice on axis 1:
    # result shape = idx.shape + (bc,) = [S0, S1, bc].
    return in_data[idx, y:y + bc]


def make_inputs_2d_index_roundtrip(
    M, N, S0, S1, BLOCK_COLS, y_offset, **_unused,
) -> dict:
    """Round-trip variant: ``out`` is a zeroed [M, N] table written by the
    scatter. Uses full-row (``BLOCK_COLS == N``, ``y_offset == 0``) so the
    oracle needs no BLOCK_COLS plumbing."""
    return _make_inputs_2d_index(
        M, N, S0, S1, BLOCK_COLS, y_offset,
        seed=2026, out_shape=(M, N),
    )


def run_2d_index_roundtrip(inputs: dict) -> np.ndarray:
    """Oracle for the gather->scatter round-trip (full-row, y_offset=0):
        out[idx[i, j], :] = in[idx[i, j], :]   for every grid cell,
        out stays zero on rows no index selects.
    """
    in_data = inputs["in_ptr"]
    idx = inputs["idx_ptr"]
    out = np.zeros_like(inputs["out_ptr"])  # [M, N]
    flat = idx.reshape(-1)
    out[flat, :] = in_data[flat, :]
    return out


# ---------------------------------------------------------------------------
# rank-2 index grid x rank-3 source block -> rank-4 output (f16).
#
# Drives ``gather_2d_index_3d_block_kernel``: a 2-D (S0 x S1) index grid
# gathers from a rank-3 source [M, D1, D2] with block [1, C1, D2] (leading 1 =
# the fanned-out page), producing a rank-4 [S0, S1, C1, D2] result stored into
# the [0,0,0,0] corner of a [IS0, IS1, D1, D2] output table. ``y_offset`` is
# non-zero, exercising the direct subscript on the inner (D1) axis. f16 source
# (a pure indexed copy, so the oracle compares bit-exactly).
# See docs/impl-strategy-2d-x-offsets.md.
# ---------------------------------------------------------------------------

def _make_inputs_2d_index_4d_out(
    M, D1, D2, IS0, IS1, S0, S1, C1, h_offset, *, seed,
) -> dict:
    """Rank-3 f16 source + a full [IS0, IS1] index buffer of page indices.

    The index buffer matches the *descriptor* full shape [IS0, IS1]; the
    kernel loads only the [0:S0, 0:S1] block from it (so the buffer must
    be the full extent — a block-sized buffer would put the strided reads
    of later index rows past its end). Indices are sampled in ``[0, M)``
    (duplicates allowed — gather has no aliasing concern). ``y_offset`` is
    stashed for the oracle; ``y_offset + C1 <= D1`` must hold (the variant
    params ensure it). The output buffer is the full [IS0, IS1, D1, D2]
    table; the kernel writes only the [S0, S1, C1, D2] corner.
    """
    rng = np.random.default_rng(seed)
    in_data = rng.standard_normal((M, D1, D2)).astype(np.float16)
    idx_data = rng.integers(0, M, size=(IS0, IS1)).astype(np.int32)
    out_data = np.zeros((IS0, IS1, D1, D2), dtype=np.float16)
    return {
        "in_ptr":   in_data,
        "out_ptr":  out_data,
        "idx_ptr":  idx_data,
        "h_offset": h_offset,
    }


def make_inputs_2d_index_3d_block(
    CACHE_SZ, HEAD, D, B, L, BLOCK_B, BLOCK_L, BLOCK_H, h_offset, **_unused,
) -> dict:
    return _make_inputs_2d_index_4d_out(
        CACHE_SZ, HEAD, D, B, L, BLOCK_B, BLOCK_L, BLOCK_H, h_offset, seed=4001,
    )


def make_inputs_2d_index_3d_block_large(
    CACHE_SZ, HEAD, D, B, L, BLOCK_B, BLOCK_L, BLOCK_H, h_offset, **_unused,
) -> dict:
    return _make_inputs_2d_index_4d_out(
        CACHE_SZ, HEAD, D, B, L, BLOCK_B, BLOCK_L, BLOCK_H, h_offset, seed=4002,
    )


def run_2d_index_3d_block(inputs: dict, *, BLOCK_B: int, BLOCK_L: int, BLOCK_H: int) -> np.ndarray:
    """Oracle:
        result[i, j, c, d] = in[idx[i, j], h_offset + c, d]
    written into the [0:BLOCK_B, 0:BLOCK_L, 0:BLOCK_H, 0:D] corner of a zeroed
    [B, L, HEAD, D] table.

    The kernel loads only the [0:BLOCK_B, 0:BLOCK_L] block of the full [B, L]
    index buffer, so the oracle slices the same corner. ``BLOCK_B``/``BLOCK_L``/``BLOCK_H``
    are block (compile-time) sizes baked per-variant via
    ``functools.partial`` — not recoverable from the runtime ``inputs``
    buffers (the index buffer is the full [B, L], the output the full
    [BLOCK_B, BLOCK_L, HEAD, D]).
    """
    in_data = inputs["in_ptr"]               # [CACHE_SZ, HEAD, D]
    idx = inputs["idx_ptr"][:BLOCK_B, :BLOCK_L]         # [BLOCK_B, BLOCK_L] block of the full grid
    h = inputs["h_offset"]
    out = np.zeros_like(inputs["out_ptr"])    # [B, L, BLOCK_B, BLOCK_L]
    D = in_data.shape[2]
    # Advanced index dim 0 by the 2-D grid, slice the inner D1 axis,
    # full D -> tile shape [BLOCK_B, BLOCK_L, BLOCK_H, D].
    out[:BLOCK_B, :BLOCK_L, :BLOCK_H, :D] = in_data[idx, h:h + BLOCK_H, :]
    return out


# 1D-source gather — input maker + oracle
#
# The 1D kernel takes a 1D source vector and produces a 1D output. The
# rank-2 ``[K, 1]`` shape used internally by the kernel descriptor is
# an implementation detail to satisfy the gather verifier; the user-facing
# buffers are 1D.
# ---------------------------------------------------------------------------

def make_inputs_1d(K: int, K_INDICES: int, BLOCK_ROWS: int) -> dict:
    """Inputs for the ``1d`` variant. Source and output are 1D ``[K]``
    and ``[K_INDICES]`` respectively. Distinct seed (2001) so this
    variant doesn't share random data with any other gather variant."""
    del BLOCK_ROWS
    rng = np.random.default_rng(2001)
    in_data = rng.standard_normal((K,)).astype(np.float32)
    idx_data = rng.integers(0, K, size=(K_INDICES,)).astype(np.int32)
    out_data = np.zeros((K_INDICES,), dtype=np.float32)
    return {
        "in_ptr":  in_data,
        "out_ptr": out_data,
        "idx_ptr": idx_data,
    }


def run_1d(inputs: dict) -> np.ndarray:
    """NumPy oracle for the 1D-source kernel: ``out[i] = in[idx[i]]``."""
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

# Rank-N (N ≥ 3) kernel signatures.
_SIG_3D = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "idx_ptr":    "*i32",
    "M":          "i32",
    "BLOCK_SIZE": "i32",
    "HEAD_DIM":   "i32",
    "K_INDICES":  "i32",
}

_SIG_3D_GROUP = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "idx_ptr":    "*i32",
    "group_idx":  "i32",
    "M":          "i32",
    "NUM_GROUPS": "i32",
    "HEAD_DIM":   "i32",
    "K_INDICES":  "i32",
}

_SIG_4D = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "idx_ptr":    "*i32",
    "group_idx":  "i32",
    "NUM_BLOCKS": "i32",
    "NUM_GROUPS": "i32",
    "BLOCK_SIZE": "i32",
    "INNER_DIM":  "i32",
    "K_INDICES":  "i32",
}

_SIG_SCATTER_3D = {
    "dst_ptr":    "*fp32",
    "data_ptr":   "*fp32",
    "idx_ptr":    "*i32",
    "M":          "i32",
    "BLOCK_SIZE": "i32",
    "HEAD_DIM":   "i32",
    "K_INDICES":  "i32",
}

# Partial-extent variants: dim 1 is read/written at TOKEN_BLOCK < NUM_TOKENS,
# in an scf.for sweep with y_offset = b * TOKEN_BLOCK computed inside the loop.
_SIG_3D_PARTIAL = {
    "in_ptr":      "*fp32",
    "out_ptr":     "*fp32",
    "idx_ptr":     "*i32",
    "M":           "i32",
    "NUM_TOKENS":  "i32",
    "TOKEN_BLOCK": "i32",
    "HEAD_DIM":    "i32",
    "K_INDICES":   "i32",
}

_SIG_SCATTER_3D_PARTIAL = {
    "dst_ptr":     "*fp32",
    "data_ptr":    "*fp32",
    "idx_ptr":     "*i32",
    "M":           "i32",
    "NUM_TOKENS":  "i32",
    "TOKEN_BLOCK": "i32",
    "HEAD_DIM":    "i32",
    "K_INDICES":   "i32",
}

# rank-2 x_offsets kernels: the index buffer is an S0 x S1 grid (no
# K_INDICES; adds S0/S1). Both the gather and the round-trip kernel share it.
_SIG_2D_INDEX = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "idx_ptr":    "*i32",
    "y_offset":   "i32",
    "M":          "i32",
    "N":          "i32",
    "S0":         "i32",
    "S1":         "i32",
    "BLOCK_COLS": "i32",
}

# rank-2 index grid x rank-3 source block -> rank-4 output (f16 source).
# Runtime args (in/out/idx pointers + h_offset) first; the rest are constexpr.
_SIG_2D_INDEX_4D = {
    "in_ptr":   "*fp16",
    "out_ptr":  "*fp16",
    "idx_ptr":  "*i32",
    "h_offset": "i32",
    "CACHE_SZ": "i32",
    "HEAD":     "i32",
    "D":        "i32",
    "B":        "i32",
    "L":        "i32",
    "BLOCK_B":  "i32",
    "BLOCK_L":  "i32",
    "BLOCK_H":  "i32",
}

# 1D kernel: 1D source ``in[K]`` and 1D output ``out[K_INDICES]``.
# No ``y_offset`` (width-1 rows leave nothing to slice), no ``M``/``N``/
# ``BLOCK_COLS``. ``K`` is the source length.
_SIG_1D = {
    "in_ptr":     "*fp32",
    "out_ptr":    "*fp32",
    "idx_ptr":    "*i32",
    "K":          "i32",
    "K_INDICES":  "i32",
    "BLOCK_ROWS": "i32",
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


# Partial-extent variants: layered on top of _EXTRA_CHECKS to pin the in-loop
# y_offset capture. The kernel computes ``y_offset = b * TOKEN_BLOCK`` inside
# an ``scf.for``, so the lowered ``ktdp.construct_indirect_access_tile``
# (whose ``c_y`` operand SSA-depends on the loop-variant ``arith.muli``) must
# land inside the loop body. The descriptor itself is built outside the loop,
# so its ``ktdp.construct_memory_view`` should stay at function top — same
# hoisting rule as ``descriptor-placement-top-level``, but for the gather/
# scatter path.
#
# A regression that hoisted the access tile out of the loop would either
# constant-fold y_offset to a single window (numerical mismatch) or fail
# to build at all.
_PARTIAL_EXTRA_CHECKS = lambda t: (
    t.assert_absent("unrealized_conversion_cast"),
    t.assert_present("ktdp.construct_indirect_access_tile", parent="scf.for"),
    t.assert_present("ktdp.construct_memory_view", parent="func.func"),
    t.assert_count("ktdp.construct_indirect_access_tile", 0, cmp="eq",
                   parent="func.func"),
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
    # "Variants" and "Preconditions" sections) for the rules these have
    # to obey:
    #   * BLOCK_COLS is a power of two (validate_block_shape)
    #   * K_INDICES >= 8 (descriptor_gather verifier)
    #   * y_offset + BLOCK_COLS <= N (slice fits in the source row)
    # The TMA-only ``BLOCK_COLS >= 32 / bitwidth * 8`` minimum does not
    # apply on Spyre (see kernel.py docstring), but every rank-2
    # variant below picks BLOCK_COLS >= 8 anyway for portability.
    # ------------------------------------------------------------------
    "y_offset_zero": {
        # y_offset=0 case.  The lowered indirect access tile uses
        # ``col = y_offset + d1`` for the direct dimension; if y_offset
        # were dropped (or treated as a baked-in zero) the bug would
        # only show up when y_offset != 0.  This variant pins the
        # y_offset=0 path so a regression that special-cases zero
        # doesn't slip through.
        "base":   "default",
        "params": {
            "M":          [256],
            "N":          [32],
            "K_INDICES":  [16],
            "BLOCK_COLS": [16],
            "y_offset":   [0],
        },
        "inputs": make_inputs_y_offset_zero,
    },
    "full_row": {
        # BLOCK_COLS == N (full-row gather).  This is the embedding
        # lookup *special case* the README calls out: y_offset=0,
        # BLOCK_COLS=embedding_dim — gather entire rows, no column
        # slicing.  Catches a bug where the column slice machinery
        # incorrectly assumes a strict subset (e.g. an off-by-one in
        # the slice-end check).
        "base":   "default",
        "params": {
            "M":          [128],
            "N":          [16],
            "K_INDICES":  [16],
            "BLOCK_COLS": [16],
            "y_offset":   [0],
        },
        "inputs": make_inputs_full_row,
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
        "base":   "default",
        "params": {
            "M":          [64],
            "N":          [64],
            "K_INDICES":  [8],
            "BLOCK_COLS": [8],
            "y_offset":   [32],
        },
        "inputs": make_inputs_min_block_cols,
    },
    "slice_at_end": {
        # y_offset + BLOCK_COLS == N exactly — slice ends at the
        # right edge of the source row.  Catches off-by-one bugs in
        # the column subscript map: an inclusive/exclusive endpoint
        # mix-up would read column N (out of bounds) only at this
        # extreme.  The default variant has y_offset+BLOCK_COLS=48 < N=64,
        # so it leaves slack and would not detect the bug.
        "base":   "default",
        "params": {
            "M":          [256],
            "N":          [64],
            "K_INDICES":  [16],
            "BLOCK_COLS": [16],
            "y_offset":   [48],
        },
        "inputs": make_inputs_slice_at_end,
    },
    "wide_slice": {
        # Wider source row (N=256) and larger BLOCK_COLS=128.  Probes
        # whether the gather lowering scales beyond the small
        # BLOCK_COLS values used by default+embedding.  A bug in the
        # tile-shape computation that only manifests above some size
        # (e.g. an int16 truncation, an alignment assumption) would
        # show up here.
        "base":   "default",
        "params": {
            "M":          [128],
            "N":          [256],
            "K_INDICES":  [16],
            "BLOCK_COLS": [128],
            "y_offset":   [64],
        },
        "inputs": make_inputs_wide_slice,
    },
    "large_k": {
        # Larger K_INDICES (128) than any other variant.  The kernel
        # is single-program, so this stresses the descriptor_load on
        # the index buffer + the descriptor_gather fan-out at higher
        # row count, without changing the lowering path.  Duplicates
        # allowed so the larger fan-out also exercises aliasing.
        "base":   "default",
        "params": {
            "M":          [512],
            "N":          [64],
            "K_INDICES":  [128],
            "BLOCK_COLS": [32],
            "y_offset":   [16],
        },
        "inputs": make_inputs_large_k,
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
        # (see ``descriptor_gather`` in python/triton/language/semantic.py,
        # the ``x_offsets.shape[0] >= 8`` assertion), so BLOCK_ROWS >= 8
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
        "base":     "2d",
        "tags":     ["descriptor-gather"],
        "grid":     [1, 1],
        "parallel": False,
    },
    "2d_large_table": {
        # Same distribution at larger source dims: 64 indices x 256 cols,
        # tiled 8x32 across [4, 8] cores. Same per-core tile counts
        # (2 row tiles, 1 col tile) as ``2d`` — we vary data shape, not
        # the distribution pattern. Confirms the lowering is robust to
        # ~4x larger M and 2x wider N+BLOCK_COLS without changing the
        # number of tiles per core.
        "base":   "2d",
        "params": {
            "M":          [4096],
            "N":          [256],
            "K_INDICES":  [64],
            "BLOCK_ROWS": [8],
            "BLOCK_COLS": [32],
        },
        "inputs": make_inputs_2d_large_table,
    },
    "1d": {
        # 1D-source gather distributed across grid=[32].
        # K_INDICES=256, BLOCK_ROWS=8 → m_blocks = 256/8 = 32, so each
        # of the 32 cores owns exactly one gather call (blocks_per_core
        # = cdiv(32, 32) = 1). BLOCK_ROWS=8 is the frontend gather
        # verifier minimum (semantic.py: x_offsets.shape[0] >= 8).
        #
        # Internally the kernel describes the 1D source as [K, 1] with
        # block_shape=[1, 1] — see gather_1d_kernel docstring and the
        # paired test_lower_desc_memory.py negative+positive pair
        # (test_gather_rank1_block_rejected /
        #  test_gather_1d_source_via_rank2_reshape_lowers).
        "kernel_fn":    kernel.gather_1d_kernel,
        "SIGNATURE":    _SIG_1D,
        "constexpr":    ["K", "K_INDICES", "BLOCK_ROWS"],
        "params": {
            "K":          [1024],
            "K_INDICES":  [256],
            "BLOCK_ROWS": [8],
        },
        "tags":         ["descriptor-gather", "1d-source"],
        "grid":         [32],
        "parallel":     True,
        "reference":    run_1d,
        "inputs":       make_inputs_1d,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "2d_large_table_serial": {
        # ``2d_large_table`` data shape on a 1-core grid. Same intent
        # as ``2d_serial``: pin the degenerate tiling path numerically
        # at the larger source dims, no work-distribution check.
        "base":     "2d_large_table",
        "tags":     ["descriptor-gather"],
        "grid":     [1, 1],
        "parallel": False,
    },
    # ------------------------------------------------------------------
    # Rank-N (N ≥ 3) gather / scatter variants.
    # ------------------------------------------------------------------
    "3d": {
        # Block-fetch gather: rank-3 source [M, BLOCK_SIZE, HEAD_DIM].
        # Each index names a block (dim 0); the full [BLOCK_SIZE, HEAD_DIM]
        # extent of that block is gathered.  y_offset=0 because dim 1 covers
        # the full BLOCK_SIZE — no sub-block slicing.
        #
        # Chosen sizes: M=256 keeps the index pool large enough for
        # K_INDICES=32 unique-index draws; BLOCK_SIZE=16 and HEAD_DIM=64
        # are powers of two and match typical attention block dimensions.
        "kernel_fn":    kernel.gather_3d_kernel,
        "SIGNATURE":    _SIG_3D,
        "constexpr":    ["M", "BLOCK_SIZE", "HEAD_DIM", "K_INDICES"],
        "params": {
            "M":          [256],
            "BLOCK_SIZE": [16],
            "HEAD_DIM":   [64],
            "K_INDICES":  [32],
        },
        "tags":         ["descriptor-gather-nd"],
        "grid":         [32],
        "parallel":     False,
        "reference":    run_3d,
        "inputs":       make_inputs_3d,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "3d_group": {
        # Rank-3 with non-trivial y_offset: source [M, NUM_GROUPS, HEAD_DIM],
        # block [1, 1, HEAD_DIM].  group_idx selects dim 1 (the group axis);
        # result shape is [K_INDICES, 1, HEAD_DIM].
        #
        # group_idx=3 (non-zero) makes the c_y capture load-bearing — a bug
        # that dropped c_y from the subscript would pass group_idx=0 but
        # fail here.
        #
        # Note: this variant does NOT pin the descriptor-gather-nd-subscripts
        # pattern (partial extent on dim 1 vs. full extent on dim ≥ 2). Block
        # dim 1 is 1, so the role contrast is invisible — see
        # ``gather_3d_group_kernel`` docstring. The ``3d_partial`` /
        # ``scatter_3d_partial`` variants below cover that pattern.
        "kernel_fn":    kernel.gather_3d_group_kernel,
        "SIGNATURE":    _SIG_3D_GROUP,
        "constexpr":    ["M", "NUM_GROUPS", "HEAD_DIM", "K_INDICES"],
        "params": {
            "M":          [256],
            "NUM_GROUPS": [8],
            "HEAD_DIM":   [64],
            "K_INDICES":  [32],
            "group_idx":  [3],
        },
        "tags":         ["descriptor-gather-nd"],
        "grid":         [32],
        "parallel":     False,
        "reference":    run_3d_group,
        "inputs":       make_inputs_3d_group,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "4d": {
        # Rank-4: source [NUM_BLOCKS, NUM_GROUPS, BLOCK_SIZE, INNER_DIM].
        # Block [1, 1, BLOCK_SIZE, INNER_DIM] — two leading 1s for the
        # indirect and group dims; trailing two dims at full extent.
        # group_idx=1 (non-zero) for the same reason as 3d_group.
        "kernel_fn":    kernel.gather_4d_kernel,
        "SIGNATURE":    _SIG_4D,
        "constexpr":    ["NUM_BLOCKS", "NUM_GROUPS", "BLOCK_SIZE", "INNER_DIM", "K_INDICES"],
        "params": {
            "NUM_BLOCKS": [64],
            "NUM_GROUPS": [4],
            "BLOCK_SIZE": [16],
            "INNER_DIM":  [64],
            "K_INDICES":  [32],
            "group_idx":  [1],
        },
        "tags":         ["descriptor-gather-4d"],
        "grid":         [32],
        "parallel":     False,
        "reference":    run_4d,
        "inputs":       make_inputs_4d,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "scatter_3d": {
        # Write-back mirror of the "3d" variant.  Reads K_INDICES blocks from
        # data_ptr and scatters them into dst_ptr[M, BLOCK_SIZE, HEAD_DIM].
        # Uses unique indices (no aliasing) so the oracle is deterministic.
        "base":       "3d",
        "kernel_fn":  kernel.scatter_3d_kernel,
        "SIGNATURE":  _SIG_SCATTER_3D,
        "tags":       ["descriptor-scatter-nd"],
        "reference":  run_scatter_3d,
        "inputs":     make_inputs_scatter_3d,
        "output_key": "dst_ptr",
    },
    # ------------------------------------------------------------------
    # Partial-extent rank-3 variants: block dim 1 is a strict divisor of
    # source dim 1 (``TOKEN_BLOCK | NUM_TOKENS``, ``TOKEN_BLOCK < NUM_TOKENS``);
    # an ``scf.for`` sweeps the windows along dim 1 with
    # ``y_offset = b * TOKEN_BLOCK`` computed inside the loop body. Each
    # iteration performs a windowed gather/scatter along dim 1:
    #
    #     dst[idx[i], b*G:(b+1)*G, :] = data[i, b*G:(b+1)*G, :]   # G | shape[1]
    #
    # The full sweep reconstructs ``in[idx, :, :]`` numerically, so the
    # oracle is the same as the ``3d`` variant. The partial-extent + in-
    # loop y_offset capture properties live in IR — pinned by
    # ``_PARTIAL_EXTRA_CHECKS`` (access tile inside ``scf.for``, view at
    # function top).
    #
    # Sizes: NUM_TOKENS=64, TOKEN_BLOCK=16 → 4 windows. K_INDICES=32 keeps
    # the per-iteration gather identical in shape to the ``3d`` variant
    # (``[32, 16, 64]``) for easy cross-reference.
    # ------------------------------------------------------------------
    "3d_partial": {
        "kernel_fn":    kernel.gather_3d_partial_kernel,
        "SIGNATURE":    _SIG_3D_PARTIAL,
        "constexpr":    ["M", "NUM_TOKENS", "TOKEN_BLOCK", "HEAD_DIM", "K_INDICES"],
        "params": {
            "M":           [256],
            "NUM_TOKENS":  [64],
            "TOKEN_BLOCK": [16],
            "HEAD_DIM":    [64],
            "K_INDICES":   [32],
        },
        "tags":         ["descriptor-gather-nd"],
        "grid":         [32],
        "parallel":     False,
        "reference":    run_3d_partial,
        "inputs":       make_inputs_3d_partial,
        "output_key":   "out_ptr",
        "extra_checks": _PARTIAL_EXTRA_CHECKS,
    },
    "scatter_3d_partial": {
        # Write-back mirror of "3d_partial": same shape and sweep
        # structure, so gather/scatter exercise symmetric partial-extent
        # windowed access. Mirror's _PARTIAL_EXTRA_CHECKS pins that the
        # scatter's indirect access tile also lands inside the scf.for.
        "base":       "3d_partial",
        "kernel_fn":  kernel.scatter_3d_partial_kernel,
        "SIGNATURE":  _SIG_SCATTER_3D_PARTIAL,
        "tags":       ["descriptor-scatter-nd"],
        "reference":  run_scatter_3d_partial,
        "inputs":     make_inputs_scatter_3d_partial,
        "output_key": "dst_ptr",
    },
    # ------------------------------------------------------------------
    # rank-2 x_offsets (index-grid) variants.  Use the single-program
    # ``gather_2d_index_kernel`` / ``gather_scatter_2d_index_kernel`` with
    # an S0 x S1 index grid (no K_INDICES; adds S0/S1).  These are the
    # numerical oracles for the rank-K x_offsets relaxation — they confirm
    # the K-D indirect read (and scatter write) executes with correct
    # numerics on ktir_cpu.  See docs/impl-strategy-2d-x-offsets.md.
    # ------------------------------------------------------------------
    "2d_index_gather": {
        # 8x4 index grid → gather a [8, 4, 32] tile from a [1024, 64]
        # source. Non-zero y_offset (16) + BLOCK_COLS=32 < N=64 exercises
        # the direct y_offset column-slice subscript alongside the two
        # indirect index-grid axes.
        "kernel_fn":    kernel.gather_2d_index_kernel,
        "SIGNATURE":    _SIG_2D_INDEX,
        "constexpr":    ["M", "N", "S0", "S1", "BLOCK_COLS"],
        "params": {
            "M":          [1024],
            "N":          [64],
            "S0":         [8],
            "S1":         [4],
            "BLOCK_COLS": [32],
            "y_offset":   [16],
        },
        "tags":         ["descriptor-gather", "descriptor-gather-2d-indices"],
        "grid":         [32],
        "parallel":     False,
        "reference":    run_2d_index_gather,
        "inputs":       make_inputs_2d_index_gather,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "2d_index_roundtrip": {
        # Gather→scatter round-trip over a shared 8x4 index grid. Full-row
        # (BLOCK_COLS == N == 64, y_offset == 0) so the scatter writes
        # whole rows back into a zeroed [1024, 64] table; the oracle then
        # checks out[idx] == in[idx]. This is the only numerical coverage
        # of the rank-2 scatter (indirect store) path.
        "kernel_fn":    kernel.gather_scatter_2d_index_kernel,
        "SIGNATURE":    _SIG_2D_INDEX,
        "constexpr":    ["M", "N", "S0", "S1", "BLOCK_COLS"],
        "params": {
            "M":          [1024],
            "N":          [64],
            "S0":         [8],
            "S1":         [4],
            "BLOCK_COLS": [64],
            "y_offset":   [0],
        },
        "tags":         ["descriptor-gather", "descriptor-scatter-2d-indices"],
        "grid":         [32],
        "parallel":     False,
        # All memory ops are indirect (gather load + scatter store); the index
        # descriptor_load's direct tile is traced away, so no direct
        # construct_access_tile survives. See test_ktdp_ops_present.
        "direct_access_tile": False,
        "reference":    run_2d_index_roundtrip,
        "inputs":       make_inputs_2d_index_roundtrip,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    # ------------------------------------------------------------------
    # rank-2 index grid x rank-3 source block -> rank-4 output.  Both
    # generalisations at once (2-D x_offsets AND a rank-3 block), with a
    # non-zero h_offset on the inner axis.  f16 source; the gather is a
    # pure indexed copy so the oracle compares bit-exactly.
    # ------------------------------------------------------------------
    "2d_index_3d_block": {
        # 2x4 index grid into a [16, 6, 8] source, block [1, 2, 8]:
        # reads in[idx[i,j], 2:4, :] -> [2, 4, 2, 8] tile, stored into the
        # corner of a [4, 8, 6, 8] output. h_offset=2 (non-zero) exercises
        # the direct subscript on the inner D1=6 axis.
        "kernel_fn":    kernel.gather_2d_index_3d_block_kernel,
        "SIGNATURE":    _SIG_2D_INDEX_4D,
        "constexpr":    ["CACHE_SZ", "HEAD", "D", "B", "L", "BLOCK_B", "BLOCK_L", "BLOCK_H"],
        "params": {
            "CACHE_SZ": [16],
            "HEAD":     [6],
            "D":        [8],
            "B":        [4],
            "L":        [8],
            "BLOCK_B":  [2],
            "BLOCK_L":  [4],
            "BLOCK_H":  [2],
            "h_offset": [2],
        },
        "tags":         ["descriptor-gather", "descriptor-gather-2d-indices-3d-block"],
        "grid":         [32],
        "parallel":     False,
        "reference":    functools.partial(run_2d_index_3d_block, BLOCK_B=2, BLOCK_L=4, BLOCK_H=2),
        "inputs":       make_inputs_2d_index_3d_block,
        "output_key":   "out_ptr",
        "extra_checks": _EXTRA_CHECKS,
    },
    "2d_index_3d_block_large": {
        # Paged-KV scale: 2x64 index grid into a [32768, 32, 128] page pool,
        # block [1, 4, 128]: reads in[idx[i,j], 8:12, :] -> [2, 64, 4, 128]
        # tile, stored into the corner of a [12, 256, 32, 128] output.
        # h_offset=8 (non-zero) on the inner D1=32 axis.
        "base":      "2d_index_3d_block",
        "params": {
            "CACHE_SZ": [32768],
            "HEAD":     [32],
            "D":        [128],
            "B":        [12],
            "L":        [256],
            "BLOCK_B":  [2],
            "BLOCK_L":  [64],
            "BLOCK_H":  [4],
            "h_offset": [8],
        },
        "reference": functools.partial(run_2d_index_3d_block, BLOCK_B=2, BLOCK_L=64, BLOCK_H=4),
        "inputs":    make_inputs_2d_index_3d_block_large,
    },
}
