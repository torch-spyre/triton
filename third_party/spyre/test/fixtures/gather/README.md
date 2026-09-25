# gather

End-to-end test fixture for **`tl.descriptor_gather`** — Triton's
indirect row-indexed load. Several `@triton.jit` functions share this
fixture; the ones below form the foundation the rest build on:

- **`gather_kernel`** — the fixture's `default`. Fixed
  `[y_offset, y_offset + BLOCK_COLS)` column slice, same row-level
  contract as `gather_kernel_1core` below, but the `K_INDICES` rows are
  tiled `BLOCK_ROWS` at a time and distributed across a 1D Spyre core
  grid. Implements
  `out[i, :] = in[idx[i], y_offset : y_offset + BLOCK_COLS]`.
- **`gather_kernel_1core`** — single-program. One kernel invocation pulls
  `K_INDICES` rows from a 2D source matrix into a contiguous
  `[K_INDICES, BLOCK_COLS]` output tile in one `descriptor_gather` call,
  no `tl.program_id`. Same row-level contract as `gather_kernel`, minus
  the distribution — the simplest kernel in the file.
- **`gather_2d_kernel`** — tiled across a 2D Spyre core grid. Each
  program instance `(pid_m, pid_n)` produces one
  `BLOCK_ROWS × BLOCK_COLS` tile of the output; together the cores
  materialize the full `[K_INDICES, N]` result. Implements
  `out[i, j] = in[idx[i], j]` — no `y_offset`; gathers the full row
  width by column-tiling instead.

All three back the same downstream pattern: **embedding lookups** and
**indirect row-gather access into a 2D source**.

## Variant hierarchy

```
Level A  shape/distribution   fp32 data, i32 indices (gather has one
                               operation, so there is no OP axis to pin)
         default, y_offset_zero, full_row, slice_large_row,
         min_block_cols, slice_at_end, wide_slice, 1core, large_k,
         2d, 2d_serial, 2d_large_table, 2d_large_table_serial, 1d,
         3d, 3d_large_k, 3d_group, 3d_group_end, 4d, 4d_boundary,
         3d_partial, scatter_3d, scatter_3d_partial,
         2d_index_gather, 2d_index_roundtrip,
         2d_index_3d_block, 2d_index_3d_block_large           27 keys

Level B  compute correctness   DTYPE sweep on gather_kernel_1core, the
                                simplest legal shape (still no OP axis)
         1core_compute[DTYPE=fp16|fp32|i32]                    3 keys
```

Level D has no gather variant: no variant reaches a Spyre binary. The
layout-carrying trio (`spyre_stick`, `spyre_stick_output_only`,
`4d_spyre_stick_output`) sits outside Level A, deliberately
unclassified — see the `Unclassified` section under `## Variants`
below.

### Pythonic semantics

The three kernels differ in *what* they gather, not in the underlying
`tt.descriptor_gather` mechanism. The mechanism takes two coords —
`x_offsets` (row indices, fanned out across `BLOCK_ROWS`) and
`y_offset` (a scalar column start) — and returns a
`[BLOCK_ROWS, BLOCK_COLS]` tile from the source. The three kernels
expose that primitive in different ways.

```python
# gather_kernel_1core — single-program, fixed column slice
for i in range(K_INDICES):
    out[i, :] = in[idx[i], y_offset : y_offset + BLOCK_COLS]
# out shape: [K_INDICES, BLOCK_COLS]
# y_offset is a runtime kernel argument; chosen once per launch.
```

`gather_kernel_1core` takes a fixed `[y_offset, y_offset + BLOCK_COLS)`
column window of every gathered row — so `BLOCK_COLS` is part of the
output shape and `y_offset` flows directly through to the underlying
gather op as a kernel arg.

`gather_kernel` implements the *exact same* row-level loop body
— same fixed column window, same `y_offset` kernel argument — but
splits the `range(K_INDICES)` loop into `BLOCK_ROWS`-sized chunks and
assigns chunks to cores via `tl.program_id(0)`, so each core's
`descriptor_gather` call only ever touches its own row tile:

```python
# gather_kernel — same row-level contract as gather_kernel_1core,
# distributed across cores in BLOCK_ROWS-row chunks
for m_block in range(m_start, m_end):           # ← this core's chunk
    offset_m = m_block * BLOCK_ROWS
    out[offset_m : offset_m + BLOCK_ROWS, :] = \
        in[idx[offset_m : offset_m + BLOCK_ROWS],
           y_offset : y_offset + BLOCK_COLS]
# out shape: [K_INDICES, BLOCK_COLS] — same as gather_kernel_1core
```

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
`gather_kernel_1core` does in one shot — same `x_offsets`/`y_offset` call
into the same underlying `tt.descriptor_gather`. The 2D kernel
schedules many such calls (one per `(m_block, n_block)` tile, sharded
across cores) so that the union of their outputs covers the full
`[K_INDICES, N]` matrix.

## Why this fixture exists

`gather_kernel_1core` (variant `1core`) pins the `tt.descriptor_gather`
→ `ktdp.construct_indirect_access_tile` lowering at the simplest possible
shape — no `tl.program_id`, one `descriptor_gather` over the whole
output, `DistributeWork` is a no-op. Its only sibling, `large_k`, pins a
larger single-shot fan-out; it is kept single-program rather than moved
onto the distributed kernel below, since row-tiling would turn its
per-gather fan-out into `BLOCK_ROWS` and silently change what it tests.

`gather_kernel` (the `default`) pins the same lowering
*distributed*: `tl.program_id(0)` tiles `K_INDICES` into `BLOCK_ROWS`-row
chunks across the core grid, so `DistributeWork` has real work to do —
unlike `gather_kernel_1core`'s no-op case — while every gathered row still
takes the same fixed `y_offset` column slice. Six variants cover edge
cases of the column-slice machinery (`y_offset = 0`, full-row, a wider
full-row, minimum legal block sizes, slice ending at `N`, wider slice);
all sit on this kernel, so the column-slice machinery is exercised on
the row-tiled path rather than only the one-shot one.

`gather_2d_kernel` adds coverage neither `gather_kernel_1core` nor
`gather_kernel` can:

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
   its row-tile chunk, matching the per-core-loop idiom of `elementwise`
   and `softmax`.

Both 2D variants are also paired with a `_serial` flavour on a
`[1, 1]` grid that runs the same kernel as a single program — pinning
the degenerate path where `rows_per_core = m_blocks` and
`cols_per_core = n_blocks` produce the right answer without
`DistributeWork` having anything to distribute.

## Variants

### Level A — shape and distribution (fp32 data, i32 indices)

No `OP` axis: gather has exactly one operation, so nothing here sweeps
a combiner. `4d` / `4d_boundary` carry `IN_LAYOUT`/`OUT_LAYOUT`
constexprs but pin them to `None` — layout plumbing with no
annotation — so nothing at this level carries a real layout annotation
either; that would be Level C, which this fixture does not cover.

#### Distributed (`gather_kernel`)

| Variant           | M    | N   | K_INDICES | BLOCK_ROWS | BLOCK_COLS | y_offset | y_off+BLOCK | dups | Pinned bug class                                     |
|-------------------|------|-----|-----------|------------|------------|----------|-------------|------|-------------------------------------------------------|
| `default`         | 1024 | 64  | 256       | 8          | 32         | 16       | 48          | no   | sanity, non-zero offset, slice strictly inside         |
| `y_offset_zero`   |  256 | 32  |  16       | 8          | 16         |  0       | 16          | no   | `y_offset = 0` path                                    |
| `full_row`        |  128 | 16  |  16       | 8          | 16         |  0       | 16          | no   | `BLOCK_COLS == N` (full-row gather)                    |
| `slice_large_row` |  128 | 256 |  16       | 8          | 256        |  0       | 256         | no   | `full_row` at a 16× wider embedding dim                |
| `min_block_cols`  |   64 | 64  |   8       | 8          | 8          | 32       | 40          | yes  | verifier minimums (`BLOCK_ROWS=8`, `BLOCK_COLS=8`)     |
| `slice_at_end`    |  256 | 64  |  16       | 8          | 16         | 48       | 64          | no   | slice ends exactly at column `N` (off-by-one)          |
| `wide_slice`      |  128 | 256 |  16       | 8          | 128        | 64       | 192         | no   | larger `BLOCK_COLS` (size-dependent bugs)              |

All seven run on `grid=[32]`, each core gathering its own
`BLOCK_ROWS`-row chunk via `tl.program_id(0)`, so `K_INDICES` is the
total number of rows gathered and `BLOCK_ROWS` the per-gather tile size.
`min_block_cols` reduces to one busy core and 31 idle
(`K_INDICES == BLOCK_ROWS`, so `m_blocks = 1`) — the smallest legal
shape the kernel accepts, and what makes the `m_end` clamp in
`gather_kernel` load-bearing rather than decorative.

#### Single-program (`gather_kernel_1core`)

| Variant   | M    | N  | K_INDICES | BLOCK_COLS | y_offset | y_off+BLOCK | dups | Pinned bug class                        |
|-----------|------|----|-----------|------------|----------|-------------|------|--------------------------------------------|
| `1core`   | 1024 | 64 | 32        | 32         | 16       | 48          | no   | sanity — no `scf.for`, one-shot gather      |
| `large_k` |  512 | 64 | 128       | 32         | 16       | 48          | yes  | larger fan-out + duplicate indices          |

Neither reads `tl.program_id`, so `DistributeWork` is a no-op on both.
The whole index array is consumed in one `descriptor_gather`, so here
`K_INDICES` is the *total* number of rows gathered and there is no tile
size at all. `large_k` stays on this kernel rather than moving onto
`gather_kernel`: row-tiling would turn its per-gather fan-out into
`BLOCK_ROWS`, silently changing what it pins — at K_INDICES=128 it is
the largest single-shot fan-out in the fixture.

#### 2D-tiled (`gather_2d_kernel`)

| Variant                 | M    | N   | K_INDICES | BLOCK_ROWS × BLOCK_COLS | grid   | What it pins                                       |
|-------------------------|------|-----|-----------|--------------------------|--------|----------------------------------------------------|
| `2d`                    | 1024 | 128 | 64        | 8 × 16                   | [4, 8] | 2D `program_id` path, multi-tile-per-core loop     |
| `2d_serial`             | 1024 | 128 | 64        | 8 × 16                   | [1, 1] | degenerate 1-core path of the same kernel          |
| `2d_large_table`        | 4096 | 256 | 64        | 8 × 32                   | [4, 8] | same distribution at ~4× larger M and 2× wider N   |
| `2d_large_table_serial` | 4096 | 256 | 64        | 8 × 32                   | [1, 1] | degenerate 1-core path at large source dims        |

The two multi-core variants run on all 32 Spyre cores
(`prod(grid) = 32`); per-core work is identical (2 row tiles ×
1 column tile). `2d_large_table` is a scale-up sanity check —
per-core *tile count* is unchanged, only the data dimensions grow.
The two `_serial` flavours drop to `grid = [1, 1]`; they run
`test_numerical` against the NumPy oracle and reuse the data shape +
input generator of the corresponding multi-core variant.

#### 1D-source (`gather_1d_kernel`)

| Variant | K    | K_INDICES | BLOCK_ROWS | Pinned bug class                                    |
|---------|------|-----------|------------|------------------------------------------------------|
| `1d`    | 1024 | 256       | 8          | 1D source/output gather, one core per gather call     |

Distributed across `grid=[32]`; `K_INDICES=256`, `BLOCK_ROWS=8` gives
`m_blocks = 32`, so each core owns exactly one gather call.
`BLOCK_ROWS=8` is the frontend gather verifier's minimum
(`x_offsets.shape[0] >= 8`). Internally the 1D source is described as
`[K, 1]` with `block_shape=[1, 1]`.

#### Rank-3 block-fetch (`gather_3d_kernel`)

| Variant      | M   | BLOCK_SIZE | HEAD_DIM | K_INDICES | Pinned bug class                              |
|--------------|-----|------------|----------|-----------|------------------------------------------------|
| `3d`         | 256 | 16         | 64       | 32        | each index names a full `[BLOCK_SIZE, HEAD_DIM]` block |
| `3d_large_k` | 256 | 16         | 64       | 128       | same path at a 4× larger fan-out               |

Source is rank-3 `[M, BLOCK_SIZE, HEAD_DIM]`; each index selects dim 0
and the full extent of dims 1–2 is gathered (`y_offset=0`, no
sub-block slicing).

#### Rank-3 group-indexed (`gather_3d_group_kernel`)

| Variant        | M   | NUM_GROUPS | HEAD_DIM | K_INDICES | group_idx | Pinned bug class                                        |
|----------------|-----|------------|----------|-----------|-----------|-----------------------------------------------------------|
| `3d_group`     | 256 | 8          | 64       | 32        | 3         | non-zero, mid-range `group_idx` makes the `c_y` capture load-bearing |
| `3d_group_end` | 256 | 8          | 64       | 32        | 7         | `group_idx` at the last valid group (boundary condition)   |

Source is `[M, NUM_GROUPS, HEAD_DIM]`, block `[1, 1, HEAD_DIM]`;
`group_idx` selects dim 1. Block dim 1 is 1, so neither variant pins
the partial-vs-full-extent contrast on a middle dim — `3d_partial` /
`scatter_3d_partial` below cover that.

#### Rank-4 (`gather_4d_kernel`)

| Variant       | NUM_BLOCKS | NUM_GROUPS | BLOCK_SIZE | INNER_DIM | K_INDICES | group_idx | Pinned bug class                                                          |
|---------------|------------|------------|------------|-----------|-----------|-----------|-----------------------------------------------------------------------------|
| `4d`          | 64         | 4          | 16         | 64        | 32        | 1         | rank-4 gather with a leading block-index axis on top of the group axis      |
| `4d_boundary` | 64         | 4          | 16         | 64        | 64        | 3         | `group_idx` at the last group + `K_INDICES == NUM_BLOCKS` (every block selected) |

`IN_LAYOUT`/`OUT_LAYOUT` are declared as constexprs on both variants
but pinned to `None` — layout plumbing with no annotation, not layout
coverage.

#### Rank-3 partial-extent (`gather_3d_partial_kernel`)

| Variant       | M   | NUM_TOKENS | TOKEN_BLOCK | HEAD_DIM | K_INDICES | Pinned bug class                                                          |
|---------------|-----|------------|-------------|----------|-----------|-----------------------------------------------------------------------------|
| `3d_partial`  | 256 | 64         | 16          | 64       | 32        | windowed gather along dim 1 via an `scf.for`, `y_offset` computed in-loop  |

`TOKEN_BLOCK` is a strict divisor of `NUM_TOKENS` (`16 | 64`, 4
windows); the full sweep reconstructs `in[idx, :, :]`, so the oracle
is the same as `3d`'s.

#### Rank-3 scatter (`scatter_3d_kernel` / `scatter_3d_partial_kernel`)

| Variant               | Base         | Pinned bug class                                                  |
|-----------------------|--------------|---------------------------------------------------------------------|
| `scatter_3d`          | `3d`         | write-back mirror: reads `K_INDICES` blocks and scatters into `dst_ptr` |
| `scatter_3d_partial`  | `3d_partial` | write-back mirror of the windowed partial-extent sweep              |

Both use unique indices (no aliasing), so the oracle is deterministic.
Neither carries a layout constexpr: no annotation direction compiles
today for the scatter side — see the comment on `scatter_3d` in
`meta.py`.

#### Rank-2 index grid (`gather_2d_index_kernel` / `gather_scatter_2d_index_kernel`)

| Variant               | M    | N  | S0 | S1 | BLOCK_COLS | y_offset | Pinned bug class                                                         |
|-----------------------|------|----|----|----|------------|----------|-----------------------------------------------------------------------------|
| `2d_index_gather`     | 1024 | 64 | 8  | 4  | 32         | 16       | 2D `[S0, S1]` index grid instead of a 1D index list                        |
| `2d_index_roundtrip`  | 1024 | 64 | 8  | 4  | 64         | 0        | gather→scatter round-trip over a shared index grid; the only numerical coverage of the rank-2 scatter path |

Numerical oracles for the rank-K `x_offsets` relaxation: they confirm
the K-D indirect read (and scatter write) executes with correct
numerics on `ktir_cpu`.

#### Rank-2 index grid × rank-3 block (`gather_2d_index_3d_block_kernel`)

| Variant                    | CACHE_SZ | HEAD | D   | B  | L   | BLOCK_B | BLOCK_L | BLOCK_H | h_offset | Pinned bug class                                             |
|----------------------------|----------|------|-----|----|-----|---------|---------|---------|----------|---------------------------------------------------------------|
| `2d_index_3d_block`        | 16       | 6    | 8   | 4  | 8   | 2       | 4       | 2       | 2        | both relaxations at once: 2D index grid *and* rank-3 source block, non-zero `h_offset` on the inner axis |
| `2d_index_3d_block_large`  | 32768    | 32   | 128 | 12 | 256 | 2       | 64      | 4       | 8        | same path at paged-KV-cache scale                            |

### Level B — compute correctness

| Variant                     | M  | N  | K_INDICES | BLOCK_COLS | y_offset | DTYPE            | Pinned bug class                                    |
|------------------------------|----|----|-----------|------------|----------|------------------|------------------------------------------------------|
| `1core_compute[DTYPE=fp16]` | 16 | 16 | 8         | 16         | 0        | fp16             | descriptor_gather correctness at fp16                |
| `1core_compute[DTYPE=fp32]` | 16 | 16 | 8         | 16         | 0        | fp32             | descriptor_gather correctness at fp32                |
| `1core_compute[DTYPE=i32]`  | 16 | 16 | 8         | 16         | 0        | i32              | descriptor_gather correctness on an integer payload   |

Reuses `gather_kernel_1core` unchanged — no dtype-specific code in the
kernel, so this variant exists purely to sweep `in_ptr`/`out_ptr`'s
element type through `SIGNATURE`. `idx_ptr` is pinned `i32` in every
key: index dtype is a fixed contract, not a compute axis, the same way
`reduce`'s axis being reduced is fixed while `OP`/`DTYPE` sweep. There
is no `OP` axis here either — gather has exactly one operation, so the
sweep is DTYPE alone.

The shape is the smallest `gather_kernel_1core`'s preconditions allow
across all three dtypes: `BLOCK_COLS ≥ 32 / bitwidth * 8` needs 16 for
fp16 and only 8 for fp32/i32, so `BLOCK_COLS = 16` is the smallest value
that satisfies every arm with one shared shape row (see
`## Preconditions` below). `y_offset = 0` with `N = BLOCK_COLS` reads
the full row, the simplest case. Because gather has no arithmetic —
it is pure indexed data movement — all three dtypes are bit-exact
against the NumPy oracle; unlike `reduce`/`elementwise`'s compute
sweeps, no `rtol`/`atol` override is needed.

### Unclassified — layout-carrying (level deliberately unstated)

| Variant                    | Base kernel           | Layout annotation                     |
|-----------------------------|------------------------|-----------------------------------------|
| `spyre_stick`               | `gather_kernel_spyre` | `in_desc` and `out_desc` both stick-on-N |
| `spyre_stick_output_only`   | `gather_kernel_spyre` | `out_desc` alone, stick-on-N            |
| `4d_spyre_stick_output`     | `gather_4d_kernel`    | `out_desc` alone, stick-on-INNER_DIM    |

None of these is labelled Level C: stick physicalization moved from
the `ktir` stage to `spyrecode`, and this suite's numerical tier runs
only `ktir`, so the annotation here is carried but inert — these three
execute the same IR an unannotated variant would. A Level C label
would claim layout coverage that isn't actually happening at this
tier.

## Descriptor-based index loading

The natural Python idiom for loading the index array does **not** lower
through Spyre today:

```python
# emits tt.splat + tt.addptr + tt.load on tensor<K x !tt.ptr<i32>>
# — LowerComputeOps cannot lower this (linalg.fill rejects !tt.ptr)
idx = tl.load(idx_ptr + tl.arange(0, K_INDICES))
```

All three kernels therefore load the index tensor via a 1D
`tl.make_tensor_descriptor`, which lowers cleanly through
`LowerDescriptorMemory`.

## Preconditions

### Single-program kernel — `gather_kernel_1core` (unchecked → enforce in variant params)

Verifier rules from `tt.descriptor_gather` (the Triton frontend at
`python/triton/language/semantic.py:descriptor_gather`):

- The source descriptor's `block_shape` leading dim is exactly **1**.
- `K_INDICES ≥ 8` — the whole index array is loaded as one `x_offsets`
  tile, so the verifier's minimum binds directly on `K_INDICES`.
- `BLOCK_COLS ≥ 32 / bitwidth * 8` (i.e. ≥ **8** for f32, ≥ **16** for f16).
- `BLOCK_COLS` is a power of two.
- `y_offset + BLOCK_COLS ≤ N` (slice fits in the source row — the
  kernel does not zero-pad).

### Parallel kernel — `gather_kernel` (unchecked → enforce in variant params)

Same verifier rules apply, but row-tiling shifts where the `x_offsets`
minimum binds:

- `BLOCK_ROWS ≥ 8` — each core loads a `[BLOCK_ROWS]` index tile per
  gather call, so the verifier's minimum binds on `BLOCK_ROWS`, not
  `K_INDICES`.
- `K_INDICES % BLOCK_ROWS == 0` (tiles exactly cover the index array —
  no masking).
- `BLOCK_COLS ≥ 32 / bitwidth * 8` and a power of two (same as the
  single-program kernel).
- `y_offset + BLOCK_COLS ≤ N` (same as the single-program kernel).

### 2D kernel — `gather_2d_kernel` (unchecked → enforce in variant params)

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
