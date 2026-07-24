# Rewriting Descriptor Layout to Physical (Stick-Tiled) KTIR

This document walks through three concrete examples of how
`RewriteDescriptorLayout` transforms annotated tensor descriptors into
physical (stick-tiled) KTIR, and then describes the three-phase design of
the pass.

## Example 1: Matmul

> For readability: A stick-on-K, B stick-on-N, both with `STICK_SIZE=64`.
> `phys A = [K//64, M, 64] = [2, 64, 64]`, `phys B = [N//64, K, 64] = [4, 128, 64]`,
> `phys C = [N//64, M, 64] = [4, 64, 64]`.
> `BLOCK_M = BLOCK_N = 64` and `BLOCK_K = 64`.

### Triton

Written in logical layouts; descriptors are annotated with
`tl.spyre_tensor_layout(desc, LAYOUT)`, where `A_LAYOUT = [[(0, "floordiv", STICK_SIZE), 1, (0, "mod", STICK_SIZE)]]`:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, K, N,
    BLOCK_M: tl.constexpr,
    A_LAYOUT: tl.constexpr,
    # ...
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)
    # ...

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
    )
    tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    # ...
    m_start = pid * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)

    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            # ...
            for k in range(k_tiles):
                a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
                b_tile = b_desc.load([k * BLOCK_K, n * BLOCK_N])
                # ...
```

### KTIR: logical view (before the rewrite)

After `LowerDescriptorMemory` + `LowerComputeOps`, no m/n/k loops in this
minimal example — a single load of the whole `[M,K]`/`[K,N]` block, one
matmul, one store:

```mlir
%A_view = ktdp.construct_memory_view %a, sizes: [64, 128], strides: [128, 1] {...} : memref<64x128xf16>
%B_view = ktdp.construct_memory_view %b, sizes: [128, 256], strides: [256, 1] {...} : memref<128x256xf16>
%C_view = ktdp.construct_memory_view %c, sizes: [64, 256], strides: [256, 1] {...} : memref<64x256xf16>

%A_tile = ktdp.construct_access_tile %A_view[%m, %c0] {...} : memref<64x128xf16> -> !ktdp.access_tile<64x128xindex>
%A = ktdp.load %A_tile : <64x128xindex> -> tensor<64x128xf16>
%B_tile = ktdp.construct_access_tile %B_view[%c0, %n] {...} : memref<128x256xf16> -> !ktdp.access_tile<128x64xindex>
%B = ktdp.load %B_tile : <128x64xindex> -> tensor<128x64xf16>

%acc = linalg.matmul ins(%A, %B) outs(%cst) -> tensor<64x64xf32>
%ch  = arith.truncf %acc : tensor<64x64xf32> to tensor<64x64xf16>
%C_tile = ktdp.construct_access_tile %C_view[%m, %n] {...} : memref<64x256xf16> -> !ktdp.access_tile<64x64xindex>
ktdp.store %ch, %C_tile : tensor<64x64xf16>, <64x64xindex>
```

### Task 1: Rewriting KTIR logical to physical view (memory ops)

The pass rewrites each annotated memory view + its access tiles + loads to
the physical (stick-tiled) layout:

- A `[64,128]` stick-on-K → physical view `[2, 64, 64]` (2 K-sticks of `64`);
  access tile becomes `<2x64x64xindex>` — the whole K dim (both sticks) is
  loaded in one tile.
- B `[128,256]` stick-on-N → physical view `[4, 128, 64]` (4 N-sticks); the
  access tile for one N-block is `<1x128x64xindex>`.
- C `[64,256]` stick-on-N → physical view `[4, 64, 64]` (4 N-sticks); the
  access tile for one output block is `<1x64x64xindex>`.

Logical offsets are converted to physical `[stick, row, lane]` coordinates
via `divsi`/`remsi` on the stick size (`64`):

```mlir
%A_view = ktdp.construct_memory_view %a, sizes: [2, 64, 64], strides: [64, 128, 1] {...} : memref<2x64x64xf16>
%B_view = ktdp.construct_memory_view %b, sizes: [4, 128, 64], strides: [64, 256, 1] {...} : memref<4x128x64xf16>
%C_view = ktdp.construct_memory_view %c, sizes: [4, 64, 64], strides: [64, 256, 1] {...} : memref<4x64x64xf16>

// A offset [%m, 0] -> [0//64, %m, 0%64]
%a_div = arith.divsi %c0, %c64 : index
%a_mod = arith.remsi %c0, %c64 : index
%A_tile = ktdp.construct_access_tile %A_view[%a_div, %m, %a_mod] {...} : memref<2x64x64xf16> -> !ktdp.access_tile<2x64x64xindex>
%A = ktdp.load %A_tile : <2x64x64xindex> -> tensor<2x64x64xf16>

// B offset [0, %n] -> [%n//64, 0, %n%64]
%b_div = arith.divsi %n, %c64 : index
%b_mod = arith.remsi %n, %c64 : index
%B_tile = ktdp.construct_access_tile %B_view[%b_div, %c0, %b_mod] {...} : memref<4x128x64xf16> -> !ktdp.access_tile<1x128x64xindex>
%B = ktdp.load %B_tile : <1x128x64xindex> -> tensor<1x128x64xf16>
```

### Task 2: Handling new tile shapes (compute ops)

After Task 1 the loaded tiles are no longer matmul-compatible:
- `%A`: `tensor<2x64x64xf16>` — 2 K-sticks, each `[64, 64]`
- `%B`: `tensor<1x128x64xf16>` — 1 N-stick, K flat as `128`

The pass synthesizes a single `scf.for` over A's K-sticks (`0..2`),
extracting one `[64,64]` slice from each tile per iteration and accumulating
into the matmul's `iter_arg`. A's stick index drives the loop IV; B is
sliced along its flat-K dim at `%k * 64`:

```mlir
%acc = scf.for %k = %c0 to %c2 step %c1 iter_args(%acc0 = %cst) -> (tensor<64x64xf32>) {
    %A_slice = tensor.extract_slice %A[%k, 0, 0]  [1, 64, 64] [1, 1, 1] : tensor<2x64x64xf16> to tensor<64x64xf16>
    %koff = arith.muli %k, %c64 : index
    %B_slice = tensor.extract_slice %B[0, %koff, 0] [1, 64, 64] [1, 1, 1] : tensor<1x128x64xf16> to tensor<64x64xf16>
    %m12 = linalg.matmul ins(%A_slice, %B_slice) outs(%acc0) -> tensor<64x64xf32>
    scf.yield %m12 : tensor<64x64xf32>
}
```

The store sink is also physicalized because C is annotated stick-on-N — the
`linalg.matmul` result `tensor<64x64xf32>` is inserted into a
`tensor<1x64x64xf32>` container matching the output stick shape, then stored
through the physical `<1x64x64xindex>` access tile:

```mlir
// C offset [%m, %n] -> [%n//64, %m, %n%64]
%c_div = arith.divsi %n, %c64 : index
%c_mod = arith.remsi %n, %c64 : index
%C_tile = ktdp.construct_access_tile %C_view[%c_div, %m, %c_mod] {...} : memref<4x64x64xf16> -> !ktdp.access_tile<1x64x64xindex>
%C_container = tensor.empty() : tensor<1x64x64xf32>
%C_stick = tensor.insert_slice %acc into %C_container[0, 0, 0] [1, 64, 64] [1, 1, 1] : tensor<64x64xf32> into tensor<1x64x64xf32>
ktdp.store %C_stick, %C_tile : tensor<1x64x64xf32>, <1x64x64xindex>
```

If the op did not care about element-block shape (e.g. `linalg.add`), no
loop would be synthesized — the rewritten tile shapes flow through unchanged.
The extra loop is emitted only when an op (like matmul) constrains operand
shapes.


## Example 2: Triple-chain matmul

`D = A @ (B @ C)`. All three inputs `A[M,K1]`, `B[K1,K2]`, `C[K2,N]` are
annotated physical (stick-tiled, `STICK_SIZE=64`); `D` is annotated too.
The inner product `B @ C` accumulates into an **intermediate tile** that is
*not* backed by a descriptor.

### Triton

The intermediate `bc` is a plain `tl.zeros` accumulator — no
`make_tensor_descriptor`, no `tl.spyre_tensor_layout`. It is consumed
directly by the next `tl.dot`:

```python
for k1 in range(k1_tiles):
    bc = tl.zeros([BLOCK_K1, BLOCK_N], dtype=tl.float16)
    for k2 in range(k2_tiles):
        b_tile = b_desc.load([k1 * BLOCK_K1, k2 * BLOCK_K2])
        c_tile = c_desc.load([k2 * BLOCK_K2, n * BLOCK_N])
        bc = tl.dot(b_tile, c_tile, bc)        # B @ C  -> intermediate
    a_tile = a_desc.load([m * BLOCK_M, k1 * BLOCK_K1])
    acc = tl.dot(a_tile, bc, acc)              # A @ bc -> output
```

### Correct lowering

`A`, `B`, `C`, `D` lower to physical stick-tiled views exactly as in
Example 1 (loads become rank-3, sliced + transposed into canonical matmul
orientation). The intermediate `bc` is the only difference: because it has
no descriptor it **stays logical** — a rank-2 register value, never
physicalized, never written to memory. It is already in the `(k, n)`
orientation the next matmul wants, so it flows straight into the outer
`linalg.matmul` with no slice and no transpose:

```mlir
// inner B @ C: both physical, each sliced to [64,64]; result is the
// logical intermediate (rank-2, no stick dim, no physical view)
%bc = linalg.matmul ins(%B_slice, %C_slice) outs(%cst) -> tensor<64x64xf16>

// outer A @ bc: A is physical -> slice + transpose (stick-on-M);
// bc is the logical intermediate -> passed through unchanged
%A_t = linalg.transpose ins(%A_slice) outs(%empty) permutation = [1, 0]
%acc = linalg.matmul ins(%A_t, %bc) outs(%cst) -> tensor<64x64xf16>
```

`D` is physicalized on the store side just like `C` in Example 1.

The takeaway: a descriptor-less intermediate is left in its logical form and
fed directly to the consuming op — the physical-layout rewrite only touches
values that came from (or go to) an annotated descriptor.

**Scratchpad recognition is by type, not by producer.** A non-physical operand
is a scratchpad iff it is already logical — a ranked tensor of the operand's
logical rank — regardless of what op produced it. `bc` is not always a direct
`linalg.matmul`: when an enclosing loop survives (`n_blocks > 1`) the inner
`B @ C` stays wrapped in its `scf.for`, so `bc` is the loop result. The pass
processes ops inside-out, so whatever produced `bc` has already been lowered to
its logical shape — which is exactly the shape the outer matmul expects.


## Example 3: tl.gather

`gather_kernel_spyre` from `test/fixtures/gather/kernel.py` (`spyre_stick`
variant). The kernel gathers `K_INDICES=32` full rows from a `[M=256, N=256]`
source matrix into a `[K_INDICES, N] = [32, 256]` output, tiling the column dim
into `N // BLOCK_COLS = 2` chunks of `BLOCK_COLS=128`. Both `in_desc` and
`out_desc` are annotated stick-on-N (`STICK_SIZE=64`, fp16); `idx_desc` is not
annotated.

> `phys in  = [N//64, M, 64] = [4, 256, 64]` (four source N-sticks of width 64)
> `phys out = [N//64, K_INDICES, 64] = [4, 32, 64]` (four output N-sticks)

Because `BLOCK_COLS=128` is **two sticks** wide, each gather/store touches two
sticks at once, and the `col_stick` loop is rescaled from `range(0, 2)` over
128-wide blocks to `range(0, 4, 2)` over 64-wide sticks (see *Loop rescaling*
under Phase 1).

### Triton

```python
@triton.jit
def gather_kernel_spyre(
    in_ptr, out_ptr, idx_ptr, y_offset,
    M: tl.constexpr, N: tl.constexpr, K_INDICES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
    IN_LAYOUT: tl.constexpr, OUT_LAYOUT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    n_blocks = N // BLOCK_COLS
    rows_per_core = tl.cdiv(tl.cdiv(K_INDICES, BLOCK_ROWS), tl.num_programs(0))

    idx_desc = tl.make_tensor_descriptor(
        idx_ptr, shape=[K_INDICES], strides=[1], block_shape=[BLOCK_ROWS],
    )
    in_desc = tl.make_tensor_descriptor(
        in_ptr, shape=[M, N], strides=[N, 1], block_shape=[1, BLOCK_COLS],
    )
    if IN_LAYOUT is not None and IN_LAYOUT != 0:
        tl.spyre_tensor_layout(in_desc, IN_LAYOUT)   # stick-on-N annotation

    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[K_INDICES, N], strides=[N, 1],
        block_shape=[BLOCK_ROWS, BLOCK_COLS],
    )
    if OUT_LAYOUT is not None and OUT_LAYOUT != 0:
        tl.spyre_tensor_layout(out_desc, OUT_LAYOUT)  # stick-on-N annotation

    m_start = pid_m * rows_per_core
    for m_sub in range(0, rows_per_core):
        for col_stick in range(n_blocks):       # this loop is rescaled below
            offset_m = (m_start + m_sub) * BLOCK_ROWS
            col_offset = col_stick * BLOCK_COLS
            idx = idx_desc.load([offset_m])
            result = in_desc.gather(idx, col_offset)
            out_desc.store([offset_m, col_offset], result)
```

### KTIR (after full lowering)

```mlir
func.func @gather_kernel_spyre(%arg0: index, %arg1: index, %arg2: index, %arg3: i32)
    attributes {grid = [1]} {
  %pid_m = ktdp.get_compute_tile_id : index
  %pid_m_0 = arith.index_cast %pid_m : index to i32

  // idx_desc: no annotation → rank-1, logical.
  %idx_desc = ktdp.construct_memory_view %arg2, sizes: [32], strides: [1] {...}
                : memref<32xsi32>
  // in_desc:  IN_LAYOUT  → rank-3 physical [N//64, M, 64] = [4, 256, 64].
  %in_desc = ktdp.construct_memory_view %arg0, sizes: [4, 256, 64], strides: [64, 256, 1] {...}
               : memref<4x256x64xf16>
  // out_desc: OUT_LAYOUT → rank-3 physical [N//64, K_INDICES, 64] = [4, 32, 64].
  %out_desc = ktdp.construct_memory_view %arg1, sizes: [4, 32, 64], strides: [64, 256, 1] {...}
                : memref<4x32x64xf16>

  // col_stick loop, RESCALED to stick granularity: range(0,2) over 128-wide
  // blocks became range(0,4,2) over 64-wide sticks. col_offset's multiplier
  // dropped from BLOCK_COLS=128 to STICK_SIZE=64 in lockstep (see below).
  scf.for %arg4 = %c0_i32 to %c4_i32 step %c2_i32 : i32 {
    %offset_m  = arith.muli %pid_m_0, %c32_i32 : i32
    %col_offset = arith.muli %arg4, %c64_i32 : i32     // arg4 * 64, not * 128
    %idx    = arith.index_cast %offset_m  : i32 to index
    %result = arith.index_cast %col_offset : i32 to index

    // Physicalized gather. The output column j within a tile is reconstructed
    // from the stick + lane iteration variables (j = d_stick*64 + d_lane); the
    // source column is col_offset + j, split into the source stick/lane via
    // floordiv/mod. The tile is 2 sticks wide → access_tile<2x32x64>.
    //   dim 0 (stick): direct,   (col_offset + d_stick*64 + d_lane) floordiv 64
    //   dim 1 (row):   indirect, idx[d_row]
    //   dim 2 (lane):  direct,   (col_offset + d_stick*64 + d_lane) mod 64
    %gather_tile = ktdp.construct_indirect_access_tile
                     intermediate_variables(%d_stick, %d_row, %d_lane)
                     %in_desc[
                       ((%result + %d_stick * 64 + %d_lane) floordiv 64),
                       ind(%idx_desc[%idx + %d_row]),
                       ((%result + %d_stick * 64 + %d_lane) mod 64)
                     ] {...}
                     : memref<4x256x64xf16>, memref<32xsi32> -> !ktdp.access_tile<2x32x64xindex>
    %loaded = ktdp.load %gather_tile : <2x32x64xindex> -> tensor<2x32x64xf16>

    // out_desc store tile: the stick coord is the (rescaled) IV directly; the
    // lane coord is col_offset mod 64 (= 0, sticks are 64-aligned here).
    %stick = arith.index_cast %arg4 : i32 to index
    %lane  = arith.remsi %result, %c64 : index
    %out_tile = ktdp.construct_access_tile %out_desc[%stick, %idx, %lane] {...}
                  : memref<4x32x64xf16> -> !ktdp.access_tile<2x32x64xindex>

    // Both load result and output tile are rank-3 → store directly, no insert_slice.
    ktdp.store %loaded, %out_tile : tensor<2x32x64xf16>, <2x32x64xindex>
  }
  return
}
```

### What happened

`IN_LAYOUT` physicalizes the gather **source**: `in_desc` becomes the rank-3
stick-on-N view `memref<4x256x64xf16>`, and the
`ktdp.construct_indirect_access_tile` is rewritten to address it. Because a
stick-tiled column dim splits into two physical dims (stick + lane), the
indirect tile gains a third intermediate variable: the logical column within a
tile `j = d_stick*64 + d_lane` is recombined, the source column `col_offset + j`
is formed, and that single expression is split back into the source's stick and
lane coordinates with `floordiv`/`mod`. This carries correctly across a source
stick boundary, so a non-stick-aligned `col_offset` would still be correct. The
indirect (row) subscript — `idx[d_row]` — is unaffected; only the direct column
dim is stick-split.

`OUT_LAYOUT` physicalizes the gather **output**: `out_desc` is the rank-3
`memref<4x32x64xf16>`. Because the gather tile is now also rank-3, its load
result matches the output tile rank directly — the store sink needs **no**
`tensor.insert_slice` (contrast the matmul case in Example 1, where the logical
rank-2 result must be scattered into the rank-3 stick container).

The `col_stick` loop comes out **rescaled** to stick granularity
(`range(0,2,1)` → `range(0,4,2)`, `col_offset = iv*128` → `iv*64`) — a Phase 1
mechanism shared by both descriptors over that loop. See *Loop rescaling* under
Phase 1 below.


## Example 4: Row-sum reduce

`reduce_spyre` from `test/fixtures/reduce/kernel.py` (`spyre_stick` variant).
The kernel computes `out[m] = sum(in[m, :])` over a `[M=64, N=256]` input,
distributing `M`-blocks across the grid. `in_desc` is annotated stick-on-N
(`STICK_SIZE=64`, fp16); `out_desc` is annotated with a 1D stick layout
(`STICK_SIZE=64`).

> `phys in  = [N//64, M, 64] = [4, 64, 64]` (four N-sticks)
> `phys out = [M//64, 64]    = [1, 64]`      (one output M-stick)

### Triton

```python
@triton.jit
def reduce_spyre(
    in_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
    IN_LAYOUT: tl.constexpr, OUT_LAYOUT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    grid_m = tl.num_programs(0)
    m_blocks = tl.cdiv(M, BLOCK_M)
    rows_per_core = tl.cdiv(m_blocks, grid_m)

    in_desc = tl.make_tensor_descriptor(
        in_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, N],
    )
    tl.spyre_tensor_layout(in_desc, IN_LAYOUT)   # stick-on-N annotation

    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M], strides=[1], block_shape=[BLOCK_M],
    )
    tl.spyre_tensor_layout(out_desc, OUT_LAYOUT) # 1D stick annotation

    m_start = pid_m * rows_per_core
    m_end   = tl.minimum(m_start + rows_per_core, m_blocks)
    for m_sub in range(m_start, m_end):
        a_tile = in_desc.load([m_sub * BLOCK_M, 0])
        out_desc.store([m_sub * BLOCK_M], a_tile.sum(1))
```

### KTIR (after full lowering)

```mlir
func.func @reduce_spyre(%arg0: index, %arg1: index) attributes {grid = [1]} {
  %pid_m = ktdp.get_compute_tile_id : index

  // in_desc: IN_LAYOUT → rank-3 physical [N//64, M, 64] = [4, 64, 64]
  %in_desc = ktdp.construct_memory_view %arg0, sizes: [4, 64, 64], strides: [64, 256, 1] {...}
               : memref<4x64x64xf16>
  // out_desc: OUT_LAYOUT → rank-2 physical [M//64, 64] = [1, 64]
  %out_desc = ktdp.construct_memory_view %arg1, sizes: [1, 64], strides: [64, 1] {...}
                : memref<1x64xf16>

  scf.for %m_sub = %pid_m to ... {
    %m_off = arith.muli %m_sub, %c64 : index

    // Physicalized load: four N-sticks → tensor<4x64x64xf16>
    %A_tile = ktdp.construct_access_tile %in_desc[%c0, %m_off, %c0] {...}
                : memref<4x64x64xf16> -> !ktdp.access_tile<4x64x64xindex>
    %A = ktdp.load %A_tile : <4x64x64xindex> -> tensor<4x64x64xf16>

    // Source stage (dispatchReduce): scf.for over 4 N-sticks, accumulate into [64].
    // The stick count (4) comes from the N-stick FLOOR dim; the lane (dim 2) is
    // the opInnerDim the reduce consumes.
    %init = linalg.fill ins(%cst : f16) outs(%empty : tensor<64xf16>) -> tensor<64xf16>
    %acc = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc_in = %init)
               -> tensor<64xf16> {
      // extract one N-stick slice: tensor<4x64x64xf16> → tensor<64x64xf16>
      %slice = tensor.extract_slice %A[%k, 0, 0] [1, 64, 64] [1, 1, 1]
                 : tensor<4x64x64xf16> to tensor<64x64xf16>
      // reduce over dim=1 (the lane). outs(%acc_in) carries the running sum:
      // linalg.reduce combines each row's sum into the accumulator init.
      %partial = linalg.reduce ins(%slice : tensor<64x64xf16>)
                               outs(%acc_in : tensor<64xf16>) dimensions = [1]
        (%in: f16, %out: f16) {
          %s = arith.addf %in, %out : f16
          linalg.yield %s : f16
        }
      scf.yield %partial : tensor<64xf16>
    }

    // Sink stage (dispatchSink): scatter [64] into physical [1, 64] container
    %stick_idx = arith.index_cast %m_sub : ...
    %lane      = arith.remsi %m_off, %c64 : index
    %out_tile  = ktdp.construct_access_tile %out_desc[%stick_idx, %lane] {...}
                   : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
    %container = tensor.empty() : tensor<1x64xf16>
    %out_phys  = tensor.insert_slice %acc into %container[0, 0] [1, 64] [1, 1]
                   : tensor<64xf16> into tensor<1x64xf16>
    ktdp.store %out_phys, %out_tile : tensor<1x64xf16>, <1x64xindex>
  }
}
```

### What happened

**Source stage (`dispatchReduce`)**: the physical load produces
`tensor<4x64x64xf16>` (4 N-sticks × M-rows × lane). The reduction axis N is
stickified, so it splits into two physical dims: the N-stick **floor** dim
(`physBlock[0]=4`, a `loopDim`) and the **lane** (`physBlock[2]=64`, the
`opInnerDim` consumed directly by the reduce). `dispatchSource` reads the
loop trip count from the floor `loopDim` (`stickFactor=4`) and emits one
`scf.for` over the 4 N-sticks: each iteration calls `extractOpSlice` to peel
one `tensor<64x64xf16>` slice (M-rows × lane), then `linalg.reduce` reduces
over `dim=1` (the lane) and accumulates into the running `tensor<64xf16>`
row-sum via the loop's `iter_args`. The accumulator init comes from
`op.getDpsInits()[0]` — the linalg DPS interface shared by all source ops
(matmul, reduce, …). Compare the K-stick loop in Example 1: the machinery
is identical; only the emitted inner op differs.

**Sink stage (`dispatchSink`)**: the output descriptor is 1D `[M]` with a
2-element physical layout `[M//64, 64]` — rank 2, one floor dim and one lane.
`dispatchSink` derives `logRank` from the data tile type (rank 1) rather than
hardcoding 2, so the scatter loop correctly indexes the logical `[64]`
result by its single dimension. The `tensor.insert_slice` scatters the
`tensor<64xf16>` result into a `tensor<1x64xf16>` container before the store.

The reduce case exercises three generalizations over the matmul path:
1. **`dispatchReduce`** — a new source handler that builds a single-input
   `SourceOpSpec` with `canonicalAxes = {0, -1}` (M parallel, N reduction)
   and emits `linalg.reduce` instead of `linalg.matmul`. The stick loop from
   `dispatchSource` is reused unchanged; only the `emitOp` lambda differs.
2. **Stick loop from a floor `loopDim`** — for matmul's stick-on-K the
   reduction `opInnerDim` spans multiple sticks (`StickifiedBlock`); for
   reduce the lane is a single stick and the *floor* dim carries the stick
   count. `stickFactor` is derived from whichever floor dim of the shared
   logical reduction axis spans `>1` stick, covering both shapes.
3. **Rank-1 sink** — `dispatchSink` now derives logical rank from the
   data tile, enabling 1D output layouts (the 2-element `[M//S, S]` form).


## Pass Design

`RewriteDescriptorLayout` runs in three phases.

### Phase 1 — physicalize each annotated descriptor

For every `tt.spyre_tensor_layout` marker the pass:

1. Rebuilds `ktdp.construct_memory_view` with physical shape, strides, and
   memref type (floordiv dims get extent `⌈N/div⌉`, mod dims get the
   modulus, identity dims are unchanged).
2. Rebuilds each `ktdp.construct_access_tile` with the physical block shape
   and remapped index operands (`identity` / `divsi` / `remsi`). For a
   `ktdp.construct_indirect_access_tile` (gather), the affine subscript maps
   are rewritten instead: a stick-split column dim adds an iteration variable
   and its subscript becomes `floordiv`/`mod` over the recombined logical
   index (see Example 3).
3. Retypes `ktdp.load` result tensors and forward-retypes the elementwise
   compute chain up to the first contraction (stops at multi-tensor ops like
   `linalg.matmul`).
4. Redirects `ktdp.store`'s access tile operand to the physical tile.

Phase 1 is the only phase a purely pointwise kernel exercises. The
`tt.spyre_tensor_layout` marker is kept alive for Phase 2.

#### Loop rescaling

A loop written over logical blocks (`for i in range(n_blocks)`,
`offset = i*BLOCK`) addresses memory in stick units once physicalized. When a
`FloorDiv` (stick-index) tile coordinate traces back to an enclosing `scf.for`
IV, `rewriteAccessTile` rescales that loop by `factor = physBlock` (sticks per
tile): `ub *= factor`, `step = factor`, and every `muli(iv, C)` in the body has
`C` divided by `factor` (so `offset = i*BLOCK` → `i*STICK`). The IV is then wired
directly as the stick coordinate. With `physBlock=1` (single-stick tile) this is
a no-op.

Only the direct-tile path rescales: `rewriteIndirectAccessTile` (gather)
reconstructs its coordinates inside the affine map and leaves the loop alone.
Multiple descriptors can share one loop, so a pass-level `rescaledLoops` set
ensures only the first to physicalize a `FloorDiv` dim rescales it; the rest skip.

### Phase 2 — synthesize loop nests (fixpoint loop)

Phase 2 is the only op-aware part. `synthesizeContractions` walks all
contraction and store ops in a `while (changed)` loop:

- **source (e.g. `linalg.matmul`, `linalg.reduce`)**: when any input is
  rank-3 (physical), the dispatch method (`dispatchMatmul` or `dispatchReduce`)
  slices operands into canonical 2D orientation and emits an `scf.for` loop
  for any reduction (K-stick / N-stick) dim; the result is always rank-≤2 /
  logical. Both handlers share `dispatchSource` / `emitSourceStage`; they
  differ only in the `SourceOpSpec` (operand axes) and the `emitOp` lambda
  (which op to build). `emitSourceStage` retrieves the accumulator init via
  `getDpsInits()[0]` — the linalg DPS interface common to all source ops.
- **sink (`ktdp.store`)**: when `data_tile` rank ≠ `access_tile` rank, the
  sink stage emits `tensor.insert_slice` to scatter the logical value into a
  physical stick container. `logRank` is derived from the data tile type, so
  1D outputs (e.g. the reduce `[M]` case) work alongside the 2D matmul case.

To add a new contraction type: add a dispatch method and hook it to
`dispatchOne`. The method's responsibility is to construct a
`SourceOperandSpec` appropriately and call `emitOp`; `dispatchSource` then
checks whether it was a roundtrip or scratchpad, reconciles the plans for
all operands, builds `OperandPlan` objects with the mechanical instructions,
and passes them to `emitSourceStage` which rewrites the op body.

#### Canonical orientation as the inter-op interface

Phase 1's elementwise retyping deliberately stops at the first contraction.
Every contraction Phase 2 resolves uniformly:

- **consumes** each operand normalized to canonical 2D orientation: a
  physical operand (traced to an annotated load by `walkToLoad`) is sliced +
  transposed (`classify`); any other operand is a logical intermediate
  (scratchpad), already canonical and passed through whole
  (`classifyScratchpad`). The scratchpad test is purely the operand's *type*
  (a ranked tensor of the expected logical rank), so it covers a direct
  contraction result and one carried out of an `scf.for` alike (see Example 2).
- **emits** its result as a rank-2 *canonical* tensor — logical, carrying no
  stick dim and no physical layout.

So for a chain `input(phys) → op1 → op2 → output(phys)`: op1 leaves a
logical canonical result; op2 sees one physical input and one logical
(scratchpad) input, and re-normalizes **both** to canonical before emitting.
Canonical orientation (pure `[M,N]` math) is the shared contract between
chained ops — no cross-op layout reconciliation is needed.

#### Fixpoint trace: `D = A @ (B @ C)`

Post-Phase-1 IR: A/B/C/D loads are all physicalized to rank-3
`tensor<1x64x64>`; both `linalg.matmul`s are still fed those rank-3
operands. `synthesizeContractions` drives to a fixpoint:

| sweep | inner `B@C` | outer `A@bc` | store `D` | `changed` |
|-------|-------------|--------------|-----------|-----------|
| **1** | fires: both operands rank-3 → sliced → `%bc : tensor<64x64>` | fires: A rank-3 (slice+transpose) × `%bc` rank-2 (passthrough) → `%acc : tensor<64x64>` | fires: `data` rank-2 / `tile` rank-3 → `insert_slice` → `data` now rank-3 | `true` |
| **2** | skip: rank-2 × rank-2 | skip: rank-2 × rank-2 | skip: `data` rank-3 == `tile` rank-3 | **`false`** → exit |

The chain collapses in one productive sweep here because `module.walk`
visits the inner matmul before the outer (def-before-use pre-order), so
when the outer fires its `bc` operand is already the inner's rank-2 result.
A deeper or differently-ordered chain takes more sweeps — correctness
depends only on reaching the fixpoint, not on walk order. Sweep 2 is the
confirming pass: every op re-walked, every guard skips, `changed = false`.

### Phase 3 — erase markers

Erase the `tt.spyre_tensor_layout` ops and the now-dead
`UnrealizedConversionCast` bridges.
