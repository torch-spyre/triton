# elementwise

Elementwise kernels `C = A OP B` with tensor descriptors. Exercises the **one
program per core** idiom: each of the 32 Spyre cores runs one program that
loops over its share of the sequence.

Four operations are supported via `OP: tl.constexpr`: `add`, `sub`, `mul`, `div`.

## Variant hierarchy

```
Level D  device              fp16/fp32 × {add,sub,mul}         compiles_to_binary
         1d_device            2 keys (fp16, 1 layout)              [Stage 3]

Level C  layout               fp16 stick physicalization
         2d_spyre_stick        1 key (fp16)

Level B  compute              fp16/fp32/i32 × {add,sub,mul,div}
         1d_compute           12 keys (ktir_cpu only; div+i32 stop here: #107)

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

- **1d_compute** (`elementwise__1d_compute[DTYPE=..., OP=...]`) — 12 keys
  sweeping `fp16/fp32/i32` × `add/sub/mul/div`. The simplest possible shape
  (1D, 128 elements, single core) so only the arithmetic varies. ktir_cpu only;
  `div` and `i32` are refused by dbo-opt (#107) and do not appear at Level D.

### Level D — device launch (compiles_to_binary)

- **1d_device** (`elementwise__1d_device[LAYOUT=stick]`) — Elementwise add
  over a single fp16 tile with no distribution loop. The only variant that
  dbo-opt can lower all the way to a Spyre binary.
- **1d_device_grid2** (`elementwise__1d_device_grid2[LAYOUT=stick]`) — Same
  as `1d_device` but distributed over two cores (one stick each).
