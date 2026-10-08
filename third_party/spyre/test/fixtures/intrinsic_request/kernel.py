"""A ``tl.spyre_op`` request over a 1D vector, distributed across the grid.

One kernel for every intrinsic: ``OP`` names it, and the kernel body is the same
call whatever it is. What ``ktir_cpu`` runs is the request's FALLBACK -- the
``ktir`` artifact holds the backend's registered body for ``OP``, inlined and
tagged -- so the numbers this fixture checks are the fallback's, which is what
any KTIR reader computes for the request.
"""

import triton
import triton.language as tl


@triton.jit
def intrinsic_request_1d(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,
):
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    num_cores = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, num_cores)
    start = pid * blocks_per_core
    end = tl.minimum(start + blocks_per_core, num_blocks)
    for i in range(start, end):
        offset = i * BLOCK_SIZE
        x = x_desc.load([offset])
        out_desc.store([offset], tl.spyre_op(OP, x))
