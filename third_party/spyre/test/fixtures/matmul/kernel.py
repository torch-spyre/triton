"""Matmul kernels: 2D and batched (BMM).

Five @triton.jit functions exercising tt.dot -> linalg.matmul with tensor
descriptors. Two grid styles:

1D-grid kernels — one axis distributes work via an explicit loop:
- matmul_kernel          -- 2D: C[M,N] = A[M,K] @ B[K,N], pid distributes M
- bmm_matmul_kernel      -- batched: C[B,M,N] = A[B,M,K] @ B[B,K,N], pid
                            distributes B×M flattened

Multi-axis grid kernels — each grid axis distributes work via a loop:
- matmul_kernel_2d_grid  -- 2D grid: pid_0 loops M-tiles, pid_1 loops N-tiles
- bmm_matmul_kernel_3d_grid -- 3D grid: pid_0 loops B-tiles, pid_1 loops M-tiles,
                               pid_2 loops N-tiles
"""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    A_LAYOUT: tl.constexpr,
    B_LAYOUT: tl.constexpr,
    C_LAYOUT: tl.constexpr,
):
    """2D tiled matmul: C[M,N] = A[M,K] @ B[K,N].

    M is the distributed axis -- rows of C are partitioned across the
    SpyreOptions.grid cores (read via tl.num_programs(0), folded by
    DistributeWork). Each core iterates over its assigned M-blocks, and
    for each block iterates all N-blocks (output columns). The K-dimension
    is the innermost reduction loop that accumulates partial dot products
    into an [BLOCK_M, BLOCK_N] accumulator.

    One tensor descriptor per operand covers the full matrix; block_shape
    controls the tile granularity for loads and stores.

    ``A_LAYOUT`` / ``B_LAYOUT`` / ``C_LAYOUT`` are optional Spyre physical
    stick-tiling layouts (OpSpec ``device_coordinates`` form). When supplied
    (as constexprs) they annotate the matching descriptor via
    ``tl.spyre_tensor_layout`` so RewriteDescriptorLayout physicalizes it;
    left ``0`` the kernel lowers logically (non-Spyre variants pass ``0``).
    Passed as constexprs so the inline-literal requirement of
    ``tl.spyre_tensor_layout`` is met without binding to a local.
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    if A_LAYOUT is not None and A_LAYOUT != 0:
        tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    if B_LAYOUT is not None and B_LAYOUT != 0:
        tl.spyre_tensor_layout(b_desc, B_LAYOUT)
    if C_LAYOUT is not None and C_LAYOUT != 0:
        tl.spyre_tensor_layout(c_desc, C_LAYOUT)

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    k_tiles  = tl.cdiv(K, BLOCK_K)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores)
    m_start = pid * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)

    for m in range(m_start, m_end):
        for n in range(n_blocks):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(k_tiles):
                a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
                b_tile = b_desc.load([k * BLOCK_K, n * BLOCK_N])
                acc = tl.dot(a_tile, b_tile, acc)
            c_desc.store([m * BLOCK_M, n * BLOCK_N], acc)

@triton.jit
def bmm_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    K,
    N,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    A_LAYOUT: tl.constexpr = 0,
    B_LAYOUT: tl.constexpr = 0,
    C_LAYOUT: tl.constexpr = 0,
):
    """Batched matmul (BMM): C[B,M,N] = A[B,M,K] @ B[B,K,N].

    Flattens batch * M-blocks into a single work dimension and distributes
    it across the SpyreOptions.grid cores. Each iteration recovers the batch
    index and M-block index via divmod.

    ``A_LAYOUT`` / ``B_LAYOUT`` / ``C_LAYOUT`` are optional Spyre physical
    stick-tiling layouts (same mechanism as ``matmul_kernel``). When non-zero
    they annotate the matching descriptor via ``tl.spyre_tensor_layout``.
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    b_blocks    = tl.cdiv(B, BLOCK_B)
    m_blocks    = tl.cdiv(M, BLOCK_M)
    n_blocks    = tl.cdiv(N, BLOCK_N)
    k_tiles     = tl.cdiv(K, BLOCK_K)
    bm_blocks   = b_blocks * m_blocks
    bm_per_core = tl.cdiv(bm_blocks, num_cores)
    bm_start = pid * bm_per_core
    bm_end   = tl.minimum(bm_start + bm_per_core, bm_blocks)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[B, M, K], strides=[M*K, K, 1], block_shape=[BLOCK_B, BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[B, K, N], strides=[K*N, N, 1], block_shape=[BLOCK_B, BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[B, M, N], strides=[M*N, N, 1], block_shape=[BLOCK_B, BLOCK_M, BLOCK_N],
    )

    if A_LAYOUT is not None and A_LAYOUT != 0:
        tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    if B_LAYOUT is not None and B_LAYOUT != 0:
        tl.spyre_tensor_layout(b_desc, B_LAYOUT)
    if C_LAYOUT is not None and C_LAYOUT != 0:
        tl.spyre_tensor_layout(c_desc, C_LAYOUT)

    for bm in range(bm_start, bm_end):
        b = bm // m_blocks
        m     = bm % m_blocks

        for n in range(n_blocks):
            acc = tl.zeros([BLOCK_B, BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(k_tiles):
                a_tile = a_desc.load([b * BLOCK_B, m * BLOCK_M, k * BLOCK_K])
                b_tile = b_desc.load([b * BLOCK_B, k * BLOCK_K, n * BLOCK_N])
                acc = tl.dot(a_tile, b_tile, acc)
            c_desc.store([b * BLOCK_B, m * BLOCK_M, n * BLOCK_N], acc)

@triton.jit
def matmul_kernel_2d_grid(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D grid matmul: pid_0 distributes M-tiles, pid_1 distributes N-tiles.

    Each grid axis loops over its assigned tiles. The K dimension is the
    innermost reduction loop accumulating into a [BLOCK_M, BLOCK_N] accumulator.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_cores_m = tl.num_programs(0)
    num_cores_n = tl.num_programs(1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    k_tiles  = tl.cdiv(K, BLOCK_K)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores_m)
    n_blocks_per_core = tl.cdiv(n_blocks, num_cores_n)
    m_start = pid_m * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)
    n_start = pid_n * n_blocks_per_core
    n_end   = tl.minimum(n_start + n_blocks_per_core, n_blocks)

    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(k_tiles):
                a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
                b_tile = b_desc.load([k * BLOCK_K, n * BLOCK_N])
                acc = tl.dot(a_tile, b_tile, acc)
            c_desc.store([m * BLOCK_M, n * BLOCK_N], acc)


@triton.jit
def bmm_matmul_kernel_3d_grid(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    K,
    N,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """3D grid BMM: pid_0 distributes B-tiles, pid_1 M-tiles, pid_2 N-tiles.

    Each grid axis loops over its assigned tiles. The K dimension is the
    innermost reduction loop accumulating into a [BLOCK_B, BLOCK_M, BLOCK_N]
    accumulator.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    num_cores_b = tl.num_programs(0)
    num_cores_m = tl.num_programs(1)
    num_cores_n = tl.num_programs(2)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[B, M, K], strides=[M*K, K, 1], block_shape=[BLOCK_B, BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[B, K, N], strides=[K*N, N, 1], block_shape=[BLOCK_B, BLOCK_K, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[B, M, N], strides=[M*N, N, 1], block_shape=[BLOCK_B, BLOCK_M, BLOCK_N],
    )

    b_blocks = tl.cdiv(B, BLOCK_B)
    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    k_tiles  = tl.cdiv(K, BLOCK_K)
    b_blocks_per_core = tl.cdiv(b_blocks, num_cores_b)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores_m)
    n_blocks_per_core = tl.cdiv(n_blocks, num_cores_n)
    b_start = pid_b * b_blocks_per_core
    b_end   = tl.minimum(b_start + b_blocks_per_core, b_blocks)
    m_start = pid_m * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)
    n_start = pid_n * n_blocks_per_core
    n_end   = tl.minimum(n_start + n_blocks_per_core, n_blocks)

    for b in range(b_start, b_end):
        for m in range(m_start, m_end):
            for n in range(n_start, n_end):
                acc = tl.zeros([BLOCK_B, BLOCK_M, BLOCK_N], dtype=tl.float32)
                for k in range(k_tiles):
                    a_tile = a_desc.load([b * BLOCK_B, m * BLOCK_M, k * BLOCK_K])
                    b_tile = b_desc.load([b * BLOCK_B, k * BLOCK_K, n * BLOCK_N])
                    acc = tl.dot(a_tile, b_tile, acc)
                c_desc.store([b * BLOCK_B, m * BLOCK_M, n * BLOCK_N], acc)


@triton.jit
def chained_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    K1,
    K2,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    A_LAYOUT: tl.constexpr,
    B_LAYOUT: tl.constexpr,
    C_LAYOUT: tl.constexpr,
    D_LAYOUT: tl.constexpr,
):
    """Chained matmul: D[M,N] = A[M,K1] @ (B[K1,K2] @ C[K2,N]).

    The inner loop accumulates B_tiles @ C_tiles into a logical scratchpad
    tile bc[BLOCK_K1, BLOCK_N].  The outer dot then contracts A_tile against
    that scratchpad.  A, B, C, D all carry physical layout annotations; bc
    is the only logical intermediate (pure register value, no descriptor).
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K1], strides=[K1, 1], block_shape=[BLOCK_M, BLOCK_K1],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K1, K2], strides=[K2, 1], block_shape=[BLOCK_K1, BLOCK_K2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[K2, N], strides=[N, 1], block_shape=[BLOCK_K2, BLOCK_N],
    )
    d_desc = tl.make_tensor_descriptor(
        d_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
    )

    if A_LAYOUT is not None and A_LAYOUT != 0:
        tl.spyre_tensor_layout(a_desc, A_LAYOUT)
    if B_LAYOUT is not None and B_LAYOUT != 0:
        tl.spyre_tensor_layout(b_desc, B_LAYOUT)
    if C_LAYOUT is not None and C_LAYOUT != 0:
        tl.spyre_tensor_layout(c_desc, C_LAYOUT)
    if D_LAYOUT is not None and D_LAYOUT != 0:
        tl.spyre_tensor_layout(d_desc, D_LAYOUT)

    m_blocks = tl.cdiv(M, BLOCK_M)
    n_blocks = tl.cdiv(N, BLOCK_N)
    k1_tiles = tl.cdiv(K1, BLOCK_K1)
    k2_tiles = tl.cdiv(K2, BLOCK_K2)
    m_blocks_per_core = tl.cdiv(m_blocks, num_cores)
    m_start = pid * m_blocks_per_core
    m_end   = tl.minimum(m_start + m_blocks_per_core, m_blocks)

    for m in range(m_start, m_end):
        for n in range(n_blocks):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
            for k1 in range(k1_tiles):
                bc = tl.zeros([BLOCK_K1, BLOCK_N], dtype=tl.float16)
                for k2 in range(k2_tiles):
                    b_tile = b_desc.load([k1 * BLOCK_K1, k2 * BLOCK_K2])
                    c_tile = c_desc.load([k2 * BLOCK_K2, n * BLOCK_N])
                    bc = tl.dot(b_tile, c_tile, bc, out_dtype=tl.float16)
                a_tile = a_desc.load([m * BLOCK_M, k1 * BLOCK_K1])
                acc = tl.dot(a_tile, bc, acc, out_dtype=tl.float16)
            d_desc.store([m * BLOCK_M, n * BLOCK_N], acc)


@triton.jit
def bmm_matmul_kernel_addptr(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Batched matmul (BMM): C[B,M,N] = A[B,M,K] @ B[B,K,N].

    Flattens batch * M-blocks into a single work dimension and distributes
    it across the SpyreOptions.grid cores (read via tl.num_programs(0),
    folded by DistributeWork). Each iteration recovers the batch index and
    M-block index via divmod.

    Because each batch slice lives at a different memory offset, fresh 2D
    tensor descriptors are created per batch element using pointer arithmetic
    (e.g. a_ptr + b_idx * stride_a_b). This exercises the tt.addptr path
    into tl.make_tensor_descriptor.
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    stride_a_b = M * K
    stride_b_b = K * N
    stride_c_b = M * N

    m_blocks    = tl.cdiv(M, BLOCK_M)
    n_blocks    = tl.cdiv(N, BLOCK_N)
    k_tiles     = tl.cdiv(K, BLOCK_K)
    bm_blocks   = B * m_blocks
    bm_per_core = tl.cdiv(bm_blocks, num_cores)
    bm_start = pid * bm_per_core
    bm_end   = tl.minimum(bm_start + bm_per_core, bm_blocks)

    for bm in range(bm_start, bm_end):
        b_idx = bm // m_blocks
        m     = bm % m_blocks

        a_desc = tl.make_tensor_descriptor(
            a_ptr + b_idx * stride_a_b,
            shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr + b_idx * stride_b_b,
            shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
        )
        c_desc = tl.make_tensor_descriptor(
            c_ptr + b_idx * stride_c_b,
            shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N],
        )

        for n in range(n_blocks):
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k in range(k_tiles):
                a_tile = a_desc.load([m * BLOCK_M, k * BLOCK_K])
                b_tile = b_desc.load([k * BLOCK_K, n * BLOCK_N])
                acc = tl.dot(a_tile, b_tile, acc)
            c_desc.store([m * BLOCK_M, n * BLOCK_N], acc)
