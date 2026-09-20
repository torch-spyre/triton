# elementwise

Elementwise kernels `C = A OP B` with tensor descriptors. Exercises the **one
program per core** idiom: each of the 32 Spyre cores runs one program that
loops over its share of the sequence.

Four operations are supported via `OP: tl.constexpr`: `add`, `sub`, `mul`, `div`.

## Variant hierarchy

```
Level D  device              fp16/fp32 × {add,sub,mul}         compiles_to_binary
         1d_device            2 keys (fp16, 1 layout)              [Stage 3]

Level C  layout               fp16 annotated (physical form: lit + Level D)
         2d_spyre_stick        1 key (fp16)

Level B  compute              fp16/fp32/i32 × {add,sub,mul,div}
         1d_compute           12 keys (ktir_cpu only; div+i32 stop here)

Level A  shape                fp32, add  (OP and DTYPE pinned)
         default, dynamic, 2d, 2d_dynamic, 2d_grid,
         3d, 3d_grid, + dynamic/scalar-dim siblings   ≈ 27 keys
```

## Variants

### Level A — shape coverage (OP="add", DTYPE=fp32)

#### 1D

- **default** (`elementwise`) — `n_elements` is a `constexpr`, baked into
  the TTIR as a literal. Descriptor shapes are fully static (`memref<Nxf32>`).
- **dynamic** (`elementwise__dynamic`) — `n_elements` is a runtime `i32`.
  Descriptor shapes are dynamic (`memref<?xf32>`). Exercises the
  dynamic-shape path through `LowerDescriptorMemory`.

#### 2D

- **2d** (`elementwise__2d`) — `shape=[M, N]` with all dimensions as
  `constexpr`. Nested M × N tile loops with `cdiv` block counts and
  bounds-clamping across both dimensions. Static descriptor shapes
  (`memref<MxNxf32>`).
- **2d_dynamic** (`elementwise__2d_dynamic`) — Same kernel with `M` and `N`
  as runtime `i32`. Dynamic descriptor shapes (`memref<?x?xf32>`).

#### 3D

- **3d** (`elementwise__3d`) — `shape=[M, N, P]` with all dimensions as
  `constexpr`. Nested M × N × P tile loops with explicit stride computation
  (`stride_m = N * P`, `stride_n = P`). Static descriptor shapes
  (`memref<MxNxPxf32>`).
- **3d_dynamic** (`elementwise__3d_dynamic`) — Same kernel with `M`, `N`,
  and `P` as runtime `i32`. Dynamic descriptor shapes (`memref<?x?x?xf32>`).

### Level B — compute correctness (OP × DTYPE sweep)

- **1d_compute** (`elementwise__1d_compute[DTYPE=..., OP=...]`) — all 12 cells of
  `fp16/fp32/i32` × `add/sub/mul/div`. The simplest possible shape (1D, 128
  elements, single core) so only the arithmetic varies. ktir_cpu only; `div` and
  `i32` are refused by dbo-opt's scheduler and so do not appear at Level D.
  The `(i32, div)` cell is the one that pins a **ktir-cpu floor**: an i32 division
  is not an integer op in Triton — it goes through float — so the kernel carries
  `arith.sitofp`, `divf` and `fptosi` on *tensors*, and those reach the interpreter
  un-wrapped now that `convert-elementwise-to-linalg` is in the `spyrecode` stage
  while the numerical tier reads the `ktir` one. ktir-cpu resolved a cast's result
  type through a scalar-only path until recently, so a tree predating that fix
  fails this one cell and nothing else.

### Level D — device launch (compiles_to_binary)

- **1d_device** (`elementwise__1d_device[LAYOUT=stick]`) — Elementwise add
  over a single fp16 tile with no distribution loop. The only variant that
  dbo-opt can lower all the way to a Spyre binary.
- **1d_device_grid2** (`elementwise__1d_device_grid2[LAYOUT=stick]`) — Same
  as `1d_device` but distributed over two cores (one stick each).

#### Declared buffers

The variants above compute one operation. These compute several in sequence, and
every value passed between them makes a round trip through HBM (high-bandwidth
memory — the accelerator's main off-chip memory) rather than flowing directly from
one operation into the next.

Nothing in the compiler decides that. The kernel author writes it out: each
intermediate gets its own pointer argument, its own tensor descriptor, its own
`spyre_tensor_layout` annotation, and an explicit store followed by an explicit
load. Written normally, `sqrt(exp(x))` would pass `exp(x)` straight to `sqrt`;
here it is stored and read back.

Two things follow, and they are what these fixtures test:

- Every compute sits between a load and a store, so each becomes its own
  schedule (one unit of work the device runs start to finish).
- A buffer holding an intermediate is an ordinary descriptor, so it
  physicalizes under `LAYOUT` — gets laid out in the accelerator's stick format —
  exactly like an input or output.

##### One buffer per intermediate

- **1d_device_dag_buffers** — `out = exp(x) * sqrt(exp(x)) + sqrt(x)`, with four
  named intermediates, each in its own buffer:

  ```
  e = exp(x)      s = sqrt(e)      m = e * s      r = sqrt(x)
  out = m + r
  ```

  The kernel takes `e_ptr`, `s_ptr`, `m_ptr`, `r_ptr` as separate arguments, so
  the launcher allocates four full-size buffers.

  This expression is chosen because it **branches**: `e` is read by both the
  `sqrt` and the multiply. A straight chain like `sqrt(exp(x))` has no value with
  two consumers, so it cannot exercise what happens next. A value read twice
  becomes **one store and two loads of the same descriptor** — not two buffers,
  and not one load whose result is reused. `x` is likewise loaded twice, once for
  the `exp` and once for the other `sqrt`. That is the load-per-consumer rule.

##### One pooled buffer, divided into regions

Four separate allocations is the plainest way to write the above and the most
wasteful: each intermediate is dead within a statement or two, yet occupies a
whole tensor for the kernel's lifetime. The pooled variants take a **single**
`pool_ptr` instead and give each intermediate a *region* of it — a fixed-size
slice, addressed by adding a base offset to the descriptor index.

The regions are offsets into one descriptor covering the whole pool, rather than a
separate descriptor per region built over a shifted pointer (`pool_ptr + OFF`).
The kernels note this as forced rather than preferred, because the pointer
arithmetic the per-region form needs does not survive the descriptor lowering; see
`chain3_pooled2_1d_device`'s docstring in `kernel.py`.

What varies across these variants is **how many regions the intermediates need**,
and whether one region can hold two of them at different times. Two intermediates
can share a region only if their live ranges do not overlap — a live range running
from the store that writes a value to the last load that reads it.

- **1d_device_chain_pooled** — `out = sqrt(exp(x))`. One intermediate, so one
  region, and `pool_ptr` is used whole with no offsets. Nothing is shared here,
  which is the point: it is the baseline. If this behaves differently from
  `dag_buffers`-style separate allocation, the cause is pooling itself, since
  there is no sharing present to blame.

- **1d_device_chain_pooled_grid2** — the same kernel across two cores. Each core
  offsets into its own block of the one region, so the two never collide.

- **1d_device_chain3_pooled** — `out = exp(sqrt(exp(x)))`. Two intermediates,
  `t0 = exp(x)` and `t1 = sqrt(t0)`. Because `t0`'s last reader is the very `sqrt`
  that produces `t1`, their live ranges do not overlap and **one** region suffices.
  The middle compute therefore reads and writes the same region:
  `load R0 → sqrt → store R0`. This is reuse **within a single schedule**.

- **1d_device_chain3_pooled2** — the same expression given **two** regions, so no
  schedule reads and writes the same one. It is the comparison for the variant
  above: if the one-region version misbehaves and this one does not, the cause is
  the in-place reuse rather than the pooling.

- **1d_device_dag_pooled** — the same expression as `dag_buffers`, with its four
  intermediates in **three** regions. Three is provably the minimum. Numbering the
  statements, each intermediate's live range is:

  ```
  1  store e                          e : [1, 3]
  2  load e,  store s                 s : [2, 3]
  3  load e,  load s,  store m        m : [3, 5]
  4  store r                          r : [4, 5]
  5  load m,  load r,  store out
  ```

  All three of `e`, `s`, `m` are live at statement 3, so no two of them can share
  and three regions are needed. Three are also enough: `r` is written at statement
  4, after `e`'s last read at 3, so `r` takes over `e`'s region:

  ```
  R0 : e, then r          R1 : s          R2 : m
  ```

  Here the reuse is **across** schedules — the compute that last reads `e` and the
  compute that writes `r` are different ones. That is a weaker claim than
  `chain3_pooled`'s reuse inside one schedule, and the two are kept in separate
  fixtures so a failure points at one of them.
