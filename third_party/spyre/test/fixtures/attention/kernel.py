"""Paged-attention kernel: batched gather over the KV cache + 4-D batched ``tl.dot``.

This is the attention counterpart to the ``gather`` fixture's indirect-load
kernels.  Instead of copying gathered rows straight to the output, it folds
them into an online-softmax attention computation: the KV cache is a flat page
pool indexed by a per-(request, token) slot table, and ``tl.descriptor_gather``
pulls the physical K/V rows for a block of requests in one shot.

Layout (all f16 except the int32 slot table):

  - ``Q``     ``[B, Lq, H, D]``         — queries
  - ``K``     ``[CACHE, H, D]``         — key cache (page pool, ``CACHE`` rows)
  - ``V``     ``[CACHE, H, D]``         — value cache (same shape as ``K``)
  - ``SLOTS`` ``[B, Lk]`` ``i32``       — absolute physical slot index per
                                          (request, key token)
  - ``Out``   ``[B, H, Lq, D]``         — attention output

The single ``@triton.jit`` function below vectorises ``BLK_B`` requests and
``BLK_H`` heads per inner step so a single ``descriptor_gather`` fetches both
requests' rows at once and ``tl.dot`` batches over ``(BLK_B, BLK_H)`` to a 4-D
contraction.  ``grid = (1,)`` — the whole problem runs in one program; the
explicit ``b_start`` / ``lq_start`` / ``h_start`` loops sweep the work.

K/V are viewed 2-D as ``[CACHE, H*D]`` so a gather of ``BLK_B*KV_BLOCK`` rows
with a ``BLK_H*D``-wide column slice starting at ``h_start*D`` lands the
``BLK_H`` heads for the gathered slots in one descriptor_gather call.  This is
the same single-row-template gather contract the ``gather`` fixture pins:
``K``/``V`` descriptors carry ``block_shape=[1, BLK_H, BLOCK_D]`` (leading 1 = the fanned-
out page), and the row index list is the flattened slot table.

The ``desc.gather(rows, y_offset)`` spelling (rather than the bare
``descriptor_gather(desc, rows, y_offset)`` of the upstream GPU sketch) matches
the idiom every other Spyre fixture uses and lowers cleanly through
``LowerDescriptorMemory`` to ``ktdp.construct_indirect_access_tile`` + load.
"""

import triton
import triton.language as tl


@triton.jit
def paged_attn_NHD_kernel(
    Q,      # [B, Lq, H, D]
    K,      # [CACHE, H, D]
    V,      # [CACHE, H, D]
    SLOTS,  # [B, Lk] absolute physical slot index per (request, token)
    Out,    # [B, H, Lq, D]
    scale,
    B: tl.constexpr, H: tl.constexpr, Lq: tl.constexpr, Lk: tl.constexpr,
    CACHE: tl.constexpr, Tk: tl.constexpr,
    KV_BLOCK: tl.constexpr,  # KV page width swept per inner step
    BLOCK_Q: tl.constexpr,   # query tile width
    BLOCK_D: tl.constexpr,   # head dim D
    BLK_B: tl.constexpr,     # B blocking factor (requests per gather)
    BLK_H: tl.constexpr,     # H blocking factor (heads per gather)
):
    # --- tensor descriptors (no pointer arithmetic) ---
    # Q tile spans BLK_B requests x BLOCK_Q queries x BLK_H heads x D.
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, Lq, H, BLOCK_D],
        strides=[Lq * H * BLOCK_D, H * BLOCK_D, BLOCK_D, 1],
        block_shape=[BLK_B, BLOCK_Q, BLK_H, BLOCK_D],
    )
    # K/V as rank-3 [CACHE, H, D]; gather BLK_B*KV_BLOCK rows (rank-2 index
    # fans across leading two dims), BLK_H heads at h_start. Leading block
    # dim is 1 — the gather fans it out per row.
    k_desc = tl.make_tensor_descriptor(
        K, shape=[CACHE, H, BLOCK_D], strides=[H * BLOCK_D, BLOCK_D, 1], block_shape=[1, BLK_H, BLOCK_D],
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[CACHE, H, BLOCK_D], strides=[H * BLOCK_D, BLOCK_D, 1], block_shape=[1, BLK_H, BLOCK_D],
    )
    # SLOTS for both requests at once: (BLK_B, KV_BLOCK).
    s_desc = tl.make_tensor_descriptor(
        SLOTS, shape=[B, Lk], strides=[Lk, 1], block_shape=[BLK_B, KV_BLOCK],
    )
    # Out tile spans BLK_B requests x BLK_H heads x BLOCK_Q queries x D.
    o_desc = tl.make_tensor_descriptor(
        Out, shape=[B, H, Lq, BLOCK_D],
        strides=[H * Lq * BLOCK_D, Lq * BLOCK_D, BLOCK_D, 1],
        block_shape=[BLK_B, BLK_H, BLOCK_Q, BLOCK_D],
    )

    # ===== request loop, blocked by BLK_B =====
    for b_start in range(0, B, BLK_B):
        # ===== query-block loop =====
        for lq_start in range(0, Lq, BLOCK_Q):
            # ===== head loop, blocked by BLK_H =====
            for h_start in range(0, H, BLK_H):

                # load Q (BLK_B, BLOCK_Q, BLK_H, D) -> (BLK_B, BLK_H, BLOCK_Q, D); scale
                q = q_desc.load([b_start, lq_start, h_start, 0])
                q = tl.permute(q, (0, 2, 1, 3))             # (BLK_B, BLK_H, BLOCK_Q, D)
                q = (q.to(tl.float32) * scale).to(tl.float16)

                # online-softmax state, batched over (BLK_B, BLK_H)
                m_i = tl.full([BLK_B, BLK_H, BLOCK_Q], float("-inf"), tl.float32)
                l_i = tl.zeros([BLK_B, BLK_H, BLOCK_Q], tl.float32)
                acc = tl.zeros([BLK_B, BLK_H, BLOCK_Q, BLOCK_D], tl.float32)

                # ===== KV-page loop (Tk explicit) =====
                for j in range(0, Tk):
                    # both requests' absolute slot indices: (BLK_B, KV_BLOCK).
                    # The 2-D index grid is fed to the gather DIRECTLY (no
                    # reshape): the Spyre gather lowering traces x_offsets
                    # provenance back to the descriptor_load, and any
                    # intervening op (e.g. a reshape to a flat row list)
                    # breaks that trace and fails to legalize. The rank-2
                    # index fans out across the gather result's leading two
                    # dims instead.
                    slots = s_desc.load([b_start, j * KV_BLOCK])  # (BLK_B, KV_BLOCK)

                    # ONE batched gather over both requests, rank-2 index ->
                    #   (BLK_B, KV_BLOCK, BLK_H, D)
                    k_g = k_desc.gather(slots, h_start)# * BLOCK_D)
                    v_g = v_desc.gather(slots, h_start)# * BLOCK_D)
                    k_g = (k_g.to(tl.float32) * scale).to(tl.float16)

                    kT = tl.permute(k_g, (0, 2, 3, 1))      # (BLK_B, BLK_H, D, KV_BLOCK)
                    vv = tl.permute(v_g, (0, 2, 1, 3))      # (BLK_B, BLK_H, KV_BLOCK, D)

                    # 4-D batched matmul over (BLK_B, BLK_H)
                    scores = tl.dot(q, kT)                  # (BLK_B, BLK_H, BLOCK_Q, KV_BLOCK)

                    block_max = tl.max(scores, axis=3)      # (BLK_B, BLK_H, BLOCK_Q)
                    m_new = tl.maximum(m_i, block_max)
                    correction = tl.exp(m_i - m_new)
                    p = tl.exp(scores - m_new[:, :, :, None])

                    l_i = l_i * correction + tl.sum(p, axis=3)
                    acc = acc * correction[:, :, :, None] + tl.dot(p.to(tl.float16), vv)
                    m_i = m_new

                acc = acc / l_i[:, :, :, None]              # (BLK_B, BLK_H, BLOCK_Q, D)
                o_desc.store([b_start, h_start, lq_start, 0], acc.to(tl.float16))
