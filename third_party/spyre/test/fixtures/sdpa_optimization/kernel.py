"""Scaled dot-product attention three ways, over one measured case.

Same shape of comparison as ``softmax_optimization``, and a much larger one: here the
intermediate that matters does **not** fit in the scratchpad, so the traffic removed by
fusing is HBM traffic rather than on-chip passes.

  sdpa_opspec       the 22 OpSpecs the SDSC path emits, every intermediate materialized
                    where that path materializes it
  sdpa_fused        the same attention with the single-block accumulator dance dropped
                    and the one fusible pair fused: 9 groups
  sdpa_fused_auto   the same, with every placement left to the compiler

DRAFT, written for discussion. It does not run and the suite does not discover it:
``tl.spyre_pin`` does not exist, ``tl.spyre_tensor_layout`` takes neither a memory space
nor an element arrangement yet, ``tl.trans`` on this backend is assumed rather than
known, and there is no ``meta.py``, so ``conftest.py`` never imports this file.

  case          torch.nn.functional.scaled_dot_product_attention
  from          op-level-tests/Mistral-Small-3.2-24B-Instruct-2506/SDSC/
                torch.nn.functional.scaled_dot_product_attention-1x32x855x128
  Q             [1, 32, 855, 128] fp16
  K, V          [1, 32, 2048, 128] fp16
  QK^T          [1, 32, 855, 2048] -- 56,033,280 elements, 106.9 MiB
  grid          [32];  32 heads over 32 cores, so one head per core
  per core      q [855, 128], k/v [2048, 128], qk [855, 2048] = 3.34 MiB

**3.34 MiB against 2 MiB of scratchpad.** The attention matrix does not fit, which is
why the recorded allocations put it in `hbm_pool` at two offsets 112,066,560 bytes
apart -- exactly one attention matrix -- ping-ponged between stages. Everything the
fused version saves on that tensor is saved in HBM.

The scale is `128 ** -0.25 == 0.29730177875068026`, applied to **both** Q and K so their
product carries `1/sqrt(128)`. Splitting it keeps both operands in fp16 range; the
kernels below follow that rather than scaling once.

A **group** is a maximal region of dataflow sharing one iteration space -- the same
extents and the same iterator types -- bounded by materializations. See
``softmax_optimization`` for the longer statement.
"""

import triton
import triton.language as tl

SCALE = 0.29730177875068026  # 128 ** -0.25, applied to Q and to K

# Logical [s, d] with the stick on d: physically [s, d // 64, d % 64]. One head per
# core, so the head dim is the grid rather than a tensor dim.
ROWS = [0, (1, "floordiv", 64), (1, "mod", 64)]


@triton.jit
def sdpa_opspec(q_ptr, k_ptr, v_ptr, out_ptr):
    """The 22 OpSpecs the SDSC path emits, with every intermediate pinned where it lands.

    The decomposition is multi-block flash attention applied to a **single** block, so
    eight of the twenty-two are an accumulator prologue and epilogue that carry no
    information here: the running max starts at `-inf`, the running sum at `0`, the
    running output at `0`, and combining with them is the identity. They are written out
    because they are what the path emits.

    Numbering follows the recorded OpSpec order in `sdsc_output_code.py`.
    """
    pid = tl.program_id(0)  # grid is [32]: one head per core

    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[32, 855, 128], strides=[109440, 128, 1],
        block_shape=[1, 855, 128],
    )
    tl.spyre_tensor_layout(q_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[32, 2048, 128], strides=[262144, 128, 1],
        block_shape=[1, 2048, 128],
    )
    tl.spyre_tensor_layout(k_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[32, 2048, 128], strides=[262144, 128, 1],
        block_shape=[1, 2048, 128],
    )
    tl.spyre_tensor_layout(v_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[32, 855, 128], strides=[109440, 128, 1],
        block_shape=[1, 855, 128],
    )
    tl.spyre_tensor_layout(out_desc, [0] + ROWS, element_arrangements=["standard"] * 4)

    q = q_desc.load([pid, 0, 0])
    k = k_desc.load([pid, 0, 0])
    v = v_desc.load([pid, 0, 0])

    # --- the accumulator prologue: eight OpSpecs that say nothing here -------------
    # 0 -- full_default, the running output, zeros [855, 128].
    o_run = tl.zeros([855, 128], dtype=tl.float16)
    tl.spyre_pin(o_run, memory_space="global")

    # 1 -- full_default_1, the running max seed, -inf [855, 64]. The 64 is one stick,
    # present only so the reduction below has an axis to eliminate.
    neg_inf = tl.full([855, 64], float("-inf"), dtype=tl.float16)
    tl.spyre_pin(neg_inf, memory_space="global")

    # 2 -- amax over that: -inf per row.
    m_run = tl.max(neg_inf, axis=1, keep_dims=True)
    tl.spyre_pin(m_run, memory_space="ct_local")

    # 16 -- full_default_2, the running sum seed, zeros [855, 64].
    zeros_s = tl.zeros([855, 64], dtype=tl.float16)
    tl.spyre_pin(zeros_s, memory_space="global")

    # 17 -- amax over that: 0 per row. A max used to reduce a constant.
    s_run = tl.max(zeros_s, axis=1, keep_dims=True)
    tl.spyre_pin(s_run, memory_space="ct_local")

    # --- the attention itself -------------------------------------------------------
    # 3 -- mul_1, Q by the split scale.
    qs = q * SCALE
    tl.spyre_pin(qs, memory_space="ct_local")

    # 4 -- mul, K by the split scale.
    ks = k * SCALE
    tl.spyre_pin(ks, memory_space="global")

    # 5 -- ReStickifyOpHBM, the permute of K. A layout change, so a boundary of its own.
    kt = tl.trans(ks)
    tl.spyre_pin(kt, memory_space="global")

    # 6 -- batchmatmul, Q @ K^T. 106.9 MiB, which is 3.34 MiB per core and so does not
    # fit in the scratchpad: the recorded allocation is hbm_pool.
    qk = tl.dot(qs, kt)
    tl.spyre_pin(qk, memory_space="global")

    # 7 -- amax_2, this block's row max.
    m_blk = tl.max(qk, axis=1, keep_dims=True)
    tl.spyre_pin(m_blk, memory_space="ct_local")

    # 8 -- maximum, combining with the running max. `maximum(-inf, x) == x`.
    m = tl.maximum(m_run, m_blk)
    tl.spyre_pin(m, memory_space="ct_local")

    # 9 -- sub_1, the running-max correction exponent. `-inf - x` is -inf.
    corr = m_run - m
    tl.spyre_pin(corr, memory_space="ct_local")

    # 10 -- exp_1, the correction factor. `exp(-inf) == 0`.
    c = tl.exp(corr)
    tl.spyre_pin(c, memory_space="ct_local")

    # 11 -- mul_3, rescaling the running output by it. Zero times zero.
    o_scaled = o_run * c
    tl.spyre_pin(o_scaled, memory_space="ct_local")

    # 12 -- sub. The second 106.9 MiB tensor, and the one the fused version removes.
    p = qk - m
    tl.spyre_pin(p, memory_space="global")

    # 13 -- exp. Written back into the slot qk occupied.
    e = tl.exp(p)
    tl.spyre_pin(e, memory_space="global")

    # 14 -- batchmatmul, exp @ V.
    ov = tl.dot(e, v)
    tl.spyre_pin(ov, memory_space="ct_local")

    # 15 -- add_1, accumulating into the running output. Adding zero.
    o = o_scaled + ov
    tl.spyre_pin(o, memory_space="global")

    # 18 -- mul_2, rescaling the running sum. Zero times zero.
    s_scaled = s_run * c
    tl.spyre_pin(s_scaled, memory_space="ct_local")

    # 19 -- sum_1, this block's row sum.
    s_blk = tl.sum(e, axis=1, keep_dims=True)
    tl.spyre_pin(s_blk, memory_space="ct_local")

    # 20 -- add, accumulating into the running sum. Adding zero.
    s = s_scaled + s_blk
    tl.spyre_pin(s, memory_space="global")

    # 21 -- div.
    out_desc.store([pid, 0, 0], o / s)


@triton.jit
def sdpa_fused(q_ptr, k_ptr, v_ptr, out_ptr):
    """The same attention in nine groups.

    Two changes, of very different size.

    **The accumulator dance goes.** With one K/V block the running max is `-inf`, the
    running sum is `0` and the running output is `0`, so eight of the twenty-two
    OpSpecs compute values that are already known. Dropping them costs nothing and is
    not a scheduling decision -- it is arithmetic the single-block case does not need.
    Note this is the one change that would be **wrong** for multi-block attention.

    **`sub` and `exp` fuse.** They are pointwise on one iteration space, so with no pin
    between them they are one ``linalg.generic``. That removes a 106.9 MiB HBM write and
    the read that follows it, which is the largest single saving in either fixture.

    What stays: both matmuls and both reductions are boundaries, the permute of K is a
    boundary, `qk` is materialized because `max` and the subtraction both read it, and
    `e` is materialized because `sum` and the second matmul both read it. Both of those
    are 106.9 MiB and neither fits on-chip.

    Nine groups: scale Q, scale K, permute K, QK^T, max, sub+exp, sum, exp@V, divide.
    """
    pid = tl.program_id(0)  # grid is [32]: one head per core

    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[32, 855, 128], strides=[109440, 128, 1],
        block_shape=[1, 855, 128],
    )
    tl.spyre_tensor_layout(q_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[32, 2048, 128], strides=[262144, 128, 1],
        block_shape=[1, 2048, 128],
    )
    tl.spyre_tensor_layout(k_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[32, 2048, 128], strides=[262144, 128, 1],
        block_shape=[1, 2048, 128],
    )
    tl.spyre_tensor_layout(v_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[32, 855, 128], strides=[109440, 128, 1],
        block_shape=[1, 855, 128],
    )
    tl.spyre_tensor_layout(out_desc, [0] + ROWS, element_arrangements=["standard"] * 4)

    q = q_desc.load([pid, 0, 0])
    k = k_desc.load([pid, 0, 0])
    v = v_desc.load([pid, 0, 0])

    # groups 1 and 2 -- the split scale on each operand. Siblings rather than a chain,
    # so they do not fuse: elementwise fusion is producer-into-consumer.
    qs = q * SCALE
    ks = k * SCALE

    # group 3 -- the permute of K. A layout change, so its own group.
    kt = tl.trans(ks)

    # group 4 -- QK^T. A matmul, so a boundary; `qk` is 106.9 MiB and pinned to HBM
    # because 3.34 MiB per core does not fit in the scratchpad.
    qk = tl.dot(qs, kt)
    tl.spyre_pin(qk, memory_space="global")

    # group 5 -- the row max.
    m = tl.max(qk, axis=1, keep_dims=True)

    # group 6 -- sub and exp together. This is the fusion, and it removes a 106.9 MiB
    # write plus the read after it.
    e = tl.exp(qk - m)
    tl.spyre_pin(e, memory_space="global")

    # group 7 -- the row sum.
    s = tl.sum(e, axis=1, keep_dims=True)

    # group 8 -- exp @ V.
    o = tl.dot(e, v)

    # group 9 -- the divide.
    out_desc.store([pid, 0, 0], o / s)


@triton.jit
def sdpa_fused_auto(q_ptr, k_ptr, v_ptr, out_ptr):
    """``sdpa_fused`` with every placement left to the compiler.

    Two pins fewer, and they are the two that matter: `qk` and `e` are each 106.9 MiB,
    3.34 MiB per core, and so cannot live in the scratchpad. Naming that here is
    stating a capacity fact the compiler can compute -- it knows the extents, it knows
    the scratchpad size, and `hbm_pool_planning.py` exists to place what does not fit.

    So this is the version that says only what the attention is. It is also the version
    that depends most on the backend: if the planner does not spill, it fails on
    capacity rather than falling back, and that is a difference between the two worth
    measuring rather than arguing.
    """
    pid = tl.program_id(0)  # grid is [32]: one head per core

    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[32, 855, 128], strides=[109440, 128, 1],
        block_shape=[1, 855, 128],
    )
    tl.spyre_tensor_layout(q_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[32, 2048, 128], strides=[262144, 128, 1],
        block_shape=[1, 2048, 128],
    )
    tl.spyre_tensor_layout(k_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[32, 2048, 128], strides=[262144, 128, 1],
        block_shape=[1, 2048, 128],
    )
    tl.spyre_tensor_layout(v_desc, [0] + ROWS, element_arrangements=["standard"] * 4)
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[32, 855, 128], strides=[109440, 128, 1],
        block_shape=[1, 855, 128],
    )
    tl.spyre_tensor_layout(out_desc, [0] + ROWS, element_arrangements=["standard"] * 4)

    q = q_desc.load([pid, 0, 0])
    k = k_desc.load([pid, 0, 0])
    v = v_desc.load([pid, 0, 0])

    qs = q * SCALE
    ks = k * SCALE
    kt = tl.trans(ks)
    qk = tl.dot(qs, kt)
    m = tl.max(qk, axis=1, keep_dims=True)
    e = tl.exp(qk - m)
    s = tl.sum(e, axis=1, keep_dims=True)
    o = tl.dot(e, v)
    out_desc.store([pid, 0, 0], o / s)
