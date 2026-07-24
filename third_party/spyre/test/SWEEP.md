# Sweep coverage plan

Planned additional test variants, grouped by priority.
None of these are implemented yet.

P1–P3 are correctness gaps (code paths that could miscompile silently).
P4–P5 are coverage improvements.

Entries marked **[base]** can be added now using the `"base"` key already
implemented — they need only a `constexpr` or `params` change relative to
a sibling.  Entries marked **[sweep]** require the Cartesian-product sweep
grammar (see `DESIGN.md`) and should be added after that lands.

---

## P1 — Clamp path (highest priority)

The `tl.minimum(start + blocks_per_core, total_blocks)` guard in every
distribution loop is never exercised because every active variant picks
shapes where `total_tiles % num_cores == 0`.  A miscompile of the clamp
would be invisible.

| New variant | Base | Changed params | What fires |
|---|---|---|---|
| `matmul__nonaligned` | `default` | `M=520` | m_blocks=33, clamp on last core |
| `matmul__dynamic_nonaligned` | `dynamic` | `M=500` | same, runtime bounds |
| `matmul__bmm_nonaligned` | `bmm` | `B=5` | bm_blocks=45, clamp fires |
| `matmul__2d_grid_nonaligned` | `2d_grid` | `M=260` | m_blocks=17 across 4 cores |
| `matmul__bmm_3d_grid_nonaligned` | `bmm_3d_grid` | `B=5` | b_blocks=5 across 2 B-cores |
| `vector_add__nonaligned` | `default` | `n_elements=2097153` | num_blocks=2049 |
| `vector_add__2d_nonaligned` | `2d` | `M=520` | m_blocks=33 |
| `vector_add__3d_nonaligned` | `3d` | `M=65` | m_blocks=9 across 32 cores |
| `softmax__nonaligned` | `default` | `M=1000` | rows_per_core clamp |
| `softmax__multi_tile_nonaligned` | `multi_tile` | `M=1000` | same + N-tile inner loop |

All are **[base]** — each changes only `params` relative to the named base.

---

## P2 — Zero-work / single-tile edge cases

Shapes where `total_tiles < num_cores`: some cores produce a zero-trip
`scf.for`.  No active variant tests this.

| New variant | Base | Changed params | What fires |
|---|---|---|---|
| `matmul__single_tile` | `default` | `M=16` | 1 M-tile total; 31 cores idle |
| `vector_add__single_block` | `default` | `n_elements=1024` | 1 block; 31 cores idle |
| `softmax__few_rows` | `default` | `M=16` | 16 rows across 32 cores |
| `vector_add__3d_active_cores` | `3d` | `M=256` | all 32 cores active (currently only 8 are) |

All are **[base]**.

---

## P3 — Dynamic variants tested at a different shape than their static sibling

The dynamic kernel is compiled once and should run at any shape.  Every
current dynamic variant uses the same param values as its static sibling,
so the "runs unchanged for any shape" claim is never verified.

| New variant | Base | Changed params | What it verifies |
|---|---|---|---|
| `matmul__dynamic_small` | `dynamic` | `M=128, K=32, N=64` | different shape on same compiled kernel |
| `matmul__dynamic_large` | `dynamic` | `M=1024, K=128, N=512` | larger shape |
| `vector_add__dynamic_small` | `dynamic` | `n_elements=4096` | smaller shape |
| `vector_add__2d_dynamic_alt` | `2d_dynamic` | `M=256, N=64` | different N than static sibling |

All are **[base]**.

---

## P4 — Multi-iteration distribution loops

Several variants have per-core trip count of 1 on at least one axis,
meaning the loop body runs but the loop itself doesn't iterate.

| New variant | Base | Changed params | What it exercises |
|---|---|---|---|
| `matmul__2d_grid_both_axes` | `2d_grid` | `N=256, grid=[4,4]` | n_blocks_per_core=4; both axes multi-iter |
| `matmul__bmm_multi_bm` | `bmm` | `B=8, M=256` | bm_blocks=128; 4 per core |
| `softmax__multi_tile_small_block` | `multi_tile` | `BLOCK_N=32` | n_tiles=32; more inner iterations |

All are **[base]**.

---

## P5 — Gather boundary conditions

Gather-specific gaps in boundary conditions for the indirect-access-tile
lowering.

| New variant | Base | Changed params | What it exercises |
|---|---|---|---|
| `gather__slice_large_row` | `full_row` | `N=256, BLOCK_COLS=256` | full-row gather at wider embedding dim |
| `gather__3d_large_k` | `3d` | `K_INDICES=128` | larger K_INDICES for rank-3 gather |
| `gather__3d_group_end` | `3d_group` | `group_idx=NUM_GROUPS-1` | slice at end of group axis |
| `gather__4d_boundary` | `4d` | `group_idx=3, K_INDICES=64` | group and K_INDICES boundary |

All are **[base]**.  Note: `group_idx=NUM_GROUPS-1` requires that
`NUM_GROUPS` be a fixed number in `params`, not computed — write it as
a literal (e.g. `group_idx=7` with `NUM_GROUPS=8`).

---

## Future — sweep grammar (after DESIGN.md is implemented)

Once Cartesian expansion lands, several of the above named variants
collapse into sweep values on an existing variant.  For example:

- `matmul__nonaligned` + `matmul__single_tile` become
  `"M": [16, 512, 520]` on `default`.
- `matmul__dynamic_small` + `matmul__dynamic_large` become
  `"M": [128, 512, 1024]` on `dynamic`.
- The entire P1 clamp column for `vector_add` becomes
  `"n_elements": [1024, 2097152, 2097153]` on `default`.

The named-variant form added in P1–P4 will stay valid after the sweep
grammar lands (single-element lists are the degenerate sweep case), so
there is no migration cost.
