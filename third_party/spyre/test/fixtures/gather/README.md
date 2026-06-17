# gather

End-to-end test fixture for **`tl.descriptor_gather`** — Triton's
indirect row-indexed load. The fixture carries two `@triton.jit`
functions sharing one source file:

- **`gather_kernel`** — single-program. One kernel invocation pulls
  `K_INDICES` rows from a 2D source matrix into a contiguous
  `[K_INDICES, BLOCK_COLS]` output tile. Implements
  `out[i, :] = in[idx[i], y_offset : y_offset + BLOCK_COLS]`.
- **`gather_2d_kernel`** — tiled across a 2D Spyre core grid. Each
  program instance `(pid_m, pid_n)` produces one
  `BLOCK_ROWS × BLOCK_COLS` tile of the output; together the cores
  materialize the full `[K_INDICES, N]` result. Implements
  `out[i, j] = in[idx[i], j]`.

Both back the same downstream pattern: **embedding lookups** and
**indirect row-gather access into a 2D source**.

### Pythonic semantics

The two kernels differ in *what* they gather, not in the underlying
`tt.descriptor_gather` mechanism. The mechanism takes two coords —
`x_offsets` (row indices, fanned out across `BLOCK_ROWS`) and
`y_offset` (a scalar column start) — and returns a
`[BLOCK_ROWS, BLOCK_COLS]` tile from the source. The two kernels
expose that primitive in different ways.

```python
# gather_kernel  — single-program, fixed column slice
for i in range(K_INDICES):
    out[i, :] = in[idx[i], y_offset : y_offset + BLOCK_COLS]
# out shape: [K_INDICES, BLOCK_COLS]
# y_offset is a runtime kernel argument; chosen once per launch.
```

`gather_kernel` takes a fixed `[y_offset, y_offset + BLOCK_COLS)`
column window of every gathered row — so `BLOCK_COLS` is part of the
output shape and `y_offset` flows directly through to the underlying
gather op as a kernel arg.

`gather_2d_kernel` writes the **full row width** of every gathered
row, but it builds that row by walking `BLOCK_COLS`-wide column tiles
in an inner loop — i.e. it calls the same gather primitive multiple
times per row, varying `y_offset` to cover all `N` columns. There is
no `y_offset` *kernel argument*; the role of `y_offset` is played by
each tile's `offset_n`.

```python
# gather_2d_kernel — 2D-tiled, full row width via column-tile walk
# Row-level contract:
for i in range(K_INDICES):
    out[i, :] = in[idx[i], :]
# out shape: [K_INDICES, N]

# Equivalent tile-level form (closer to the IR — make the inner
# column walk visible, and where x_offsets / y_offset come in):
for m_block in range(K_INDICES // BLOCK_ROWS):
    for n_block in range(N // BLOCK_COLS):                  # ← inner column walk
        offset_m = m_block * BLOCK_ROWS
        offset_n = n_block * BLOCK_COLS                     # ← becomes y_offset
        x_offsets = idx[offset_m : offset_m + BLOCK_ROWS]   # ← rows to gather
        y_offset  = offset_n                                # ← column start
        out[offset_m : offset_m + BLOCK_ROWS,
            offset_n : offset_n + BLOCK_COLS] = \
            in[x_offsets, y_offset : y_offset + BLOCK_COLS]
```

Notice that the body of the inner loop is *exactly* what
`gather_kernel` does in one shot — same `x_offsets`/`y_offset` call
into the same underlying `tt.descriptor_gather`. The 2D kernel
schedules many such calls (one per `(m_block, n_block)` tile, sharded
across cores) so that the union of their outputs covers the full
`[K_INDICES, N]` matrix.

## Why this fixture exists

`gather_kernel` pins the `tt.descriptor_gather` →
`ktdp.construct_indirect_access_tile` lowering at the simplest possible
shape — no `tl.program_id`, one descriptor_gather over the whole
output, `DistributeWork` is a no-op. Seven variants cover edge cases
of the column-slice machinery (`y_offset = 0`, full-row, minimum legal
block sizes, slice ending at `N`, wider slice, larger fan-out).

`gather_2d_kernel` adds coverage the single-program kernel cannot:

1. **Two-axis `tl.program_id`.** Most other fixtures use a 1D grid;
   this is the first to exercise a 2D grid (`[4, 8]`) with
   `tl.program_id(0)` and `tl.program_id(1)` both active. The
   DistributeWork pass synthesizes one `ktdp.get_compute_tile_id` with
   two results and stamps a 2-element `grid` attribute.
2. **`BLOCK_COLS < N`.** The gather affine map
   `base[idx[d0], y_offset + d1]` runs with non-trivial `y_offset` and
   a strict-subset column block — the kernel column-tiles its way
   across the full row width, using each tile's `offset_n` as the
   gather's column offset.
3. **Multi-tile per core.** Each core runs an inner `scf.for` over
   its row-tile chunk, matching the per-core-loop idiom of `vector_add`
   and `softmax`.

Both 2D variants are also paired with a `_serial` flavour on a
`[1, 1]` grid that runs the same kernel as a single program — pinning
the degenerate path where `rows_per_core = m_blocks` and
`cols_per_core = n_blocks` produce the right answer without
`DistributeWork` having anything to distribute.

## Variants

### Single-program (`gather_kernel`)

| Variant            | M    | N   | K_INDICES | BLOCK_COLS | y_offset | y_off+BLOCK | dups | Pinned bug class                                |
|--------------------|------|-----|-----------|------------|----------|-------------|------|-------------------------------------------------|
| `default`          | 1024 | 64  | 32        | 32         | 16       | 48          | no   | sanity, non-zero offset, slice strictly inside  |
| `y_offset_zero`    |  256 | 32  | 16        | 16         |  0       | 16          | no   | `y_offset = 0` path                             |
| `full_row`         |  128 | 16  | 16        | 16         |  0       | 16          | no   | `BLOCK_COLS == N` (full-row gather)             |
| `min_block_cols`   |   64 | 64  |  8        |  8         | 32       | 40          | yes  | verifier minimums (`K=8`, `BLOCK_COLS=8`)       |
| `slice_at_end`     |  256 | 64  | 16        | 16         | 48       | 64          | no   | slice ends exactly at column `N` (off-by-one)   |
| `wide_slice`       |  128 | 256 | 16        | 128        | 64       | 192         | no   | larger `BLOCK_COLS` (size-dependent bugs)       |
| `large_k`          |  512 | 64  | 128       | 32         | 16       | 48          | yes  | larger fan-out + duplicate indices              |

All seven set `parallel: False` — `DistributeWork` is a no-op, so
`test_work_distribution` is skipped. The whole index array is consumed
in one `descriptor_gather`, so `K_INDICES` is the *total* number of
rows gathered (not a tile size).

### 2D-tiled (`gather_2d_kernel`)

| Variant                 | M    | N   | K_INDICES | BLOCK_ROWS × BLOCK_COLS | grid   | parallel | What it pins                                       |
|-------------------------|------|-----|-----------|--------------------------|--------|----------|----------------------------------------------------|
| `2d`                    | 1024 | 128 | 64        | 8 × 16                   | [4, 8] | True     | 2D `program_id` path, multi-tile-per-core loop     |
| `2d_serial`             | 1024 | 128 | 64        | 8 × 16                   | [1, 1] | False    | degenerate 1-core path of the same kernel          |
| `2d_large_table`        | 4096 | 256 | 64        | 8 × 32                   | [4, 8] | True     | same distribution at ~4× larger M and 2× wider N   |
| `2d_large_table_serial` | 4096 | 256 | 64        | 8 × 32                   | [1, 1] | False    | degenerate 1-core path at large source dims        |

The two `parallel: True` variants run on all 32 Spyre cores
(`prod(grid) = 32`); per-core work is identical (2 row tiles ×
1 column tile). `2d_large_table` is a scale-up sanity check —
per-core *tile count* is unchanged, only the data dimensions grow.
The two `_serial` flavours skip `test_work_distribution` (no
multi-program lowering to pin) but still run `test_numerical` against
the NumPy oracle, and reuse the data shape + input generator of the
corresponding parallel variant.

## Descriptor-based index loading

The natural Python idiom for loading the index array does **not** lower
through Spyre today:

```python
# emits tt.splat + tt.addptr + tt.load on tensor<K x !tt.ptr<i32>>
# — LowerComputeOps cannot lower this (linalg.fill rejects !tt.ptr)
idx = tl.load(idx_ptr + tl.arange(0, K_INDICES))
```

Both kernels therefore load the index tensor via a 1D
`tl.make_tensor_descriptor`, which lowers cleanly through
`LowerDescriptorMemory`.

## Preconditions

### Single-program kernel (unchecked → enforce in variant params)

Verifier rules from `tt.descriptor_gather` (the Triton frontend at
`python/triton/language/semantic.py:descriptor_gather`):

- The source descriptor's `block_shape` leading dim is exactly **1**.
- `K_INDICES ≥ 8`.
- `BLOCK_COLS ≥ 32 / bitwidth * 8` (i.e. ≥ **8** for f32, ≥ **16** for f16).
- `BLOCK_COLS` is a power of two.
- `y_offset + BLOCK_COLS ≤ N` (slice fits in the source row — the
  kernel does not zero-pad).

### 2D kernel (unchecked → enforce in variant params)

In addition to the verifier rules above (with `BLOCK_COLS`'s size and
power-of-two constraints), the 2D kernel assumes:

- `K_INDICES % BLOCK_ROWS == 0` and `N % BLOCK_COLS == 0` (tiles
  exactly cover the output — no masking).
- `cdiv(K_INDICES, BLOCK_ROWS) % grid[0] == 0` and
  `cdiv(N, BLOCK_COLS) % grid[1] == 0` (each core owns an integer
  number of tiles along each axis).
- `max(idx) < M` (no out-of-range row gather).

All variants satisfy these. A new variant violating them would
read/write out of bounds with no diagnostic — add masking first if
you need to exercise non-divisible shapes.

## Deeper reading

- **`CHEATSHEET.md`** — full walkthrough: array-level semantics with
  diagrams, motivating use cases, the descriptor-load workaround, the
  KTDP lowering, the per-test invariants each variant pins, and the
  resolved `unrealized_conversion_cast` history.
- **`docs/gather_lowering_walkthrough.md`** — step-by-step
  `LowerDescriptorMemory.cpp::buildIndirectAccessTile` walkthrough.
