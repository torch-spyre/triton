"""spyreop kernels: 1D shape, distribution loop and single-tile, OP dispatch.

Both kernels take ``OP: tl.constexpr`` and dispatch among the unary scalar
float ops ``LowerSpyreOps.cpp`` converts unconditionally: ``sqrt``, ``rsqrt``,
``exp`` (``math.sqrt``/``math.rsqrt``/``math.exp`` -> ``spyreop.sqrt``/
``spyreop.rsqrt``/``spyreop.exp``). See ``meta.py``'s module docstring for why
``arith.divf`` (binary, not unary) and the ``arith.addi``/``arith.muli``
integer patterns (gated on being inside a ``linalg.generic``, not on type
alone) are out of scope for this fixture.

- :func:`spyreop_1d` -- 1D grid (``tl.program_id(0)`` only); each core loops
  over its share of tiles in a distribution loop.
- :func:`spyreop_1d_device` -- no grid at all: one tile, no distribution loop.
  The only variant here dbo-opt can lower all the way to a binary; every
  other kernel outlines an ``scf.for`` from its program-id distribution,
  which dbo-opt's scheduler rejects.

"""

import triton
import triton.language as tl


@triton.jit
def spyreop_1d(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OP: tl.constexpr,
):
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[n_elements],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[n_elements],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )

    # Each core loops over its chunk of the sequence. tl.num_programs(0)
    # reports the grid's axis-0 size -- folded to a compile-time constant
    # by DistributeWork against SpyreOptions.grid.
    num_cores = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    blocks_per_core = tl.cdiv(num_blocks, num_cores)
    start = pid * blocks_per_core
    end = tl.minimum(start + blocks_per_core, num_blocks)
    for i in range(start, end):
        offset = i * BLOCK_SIZE
        x = x_desc.load([offset])
        if OP == "sqrt":
            result = tl.sqrt(x)
        elif OP == "rsqrt":
            result = tl.rsqrt(x)
        else:
            result = tl.exp(x)
        out_desc.store([offset], result)


@triton.jit
def spyreop_1d_device(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
    OP: tl.constexpr,
):
    """Unary spyreop over exactly one tile, with no distribution loop.

    No ``tl.program_id``, no ``tl.num_programs``, no loop -- one
    ``BLOCK_SIZE``-wide tile that is the whole tensor. That absence is what
    lets dbo-opt schedule it onto a binary: every looped variant outlines an
    ``scf.for`` from its program-id distribution, and dbo-opt's scheduler
    rejects that loop.
    """
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        output_ptr, shape=[n_elements], strides=[1], block_shape=[BLOCK_SIZE],
    )
    tl.spyre_tensor_layout(x_desc, LAYOUT)
    tl.spyre_tensor_layout(out_desc, LAYOUT)

    offset = pid * BLOCK_SIZE
    x = x_desc.load([offset])
    if OP == "sqrt":
        result = tl.sqrt(x)
    elif OP == "rsqrt":
        result = tl.rsqrt(x)
    else:
        result = tl.exp(x)
    out_desc.store([offset], result)
