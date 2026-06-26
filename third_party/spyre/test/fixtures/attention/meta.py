"""SIGNATURE + VARIANTS + reference oracle + input generators for paged attention.

Exercises the batched-gather attention kernel in ``kernel.py``: a flat KV
page pool ``[CACHE, H, D]`` indexed by a per-(request, token) slot table,
read via ``tl.descriptor_gather`` and folded into an online-softmax
``tl.dot`` attention.  This is the attention-shaped consumer of the same
indirect-load lowering the ``gather`` fixture pins in isolation.

Vocabulary (matches ``kernel.py``):

  - ``B``, ``H``         — batch (requests) and head count
  - ``Lq``, ``Lk``       — query / key sequence lengths
  - ``D``                — head dim (``BLOCK_D``)
  - ``CACHE``            — page-pool size (rows of the K/V cache)
  - ``Tk``               — KV pages per request (``Lk // KV_BLOCK``)
  - ``KV_BLOCK``         — KV page width swept per inner step
  - ``BLOCK_Q``          — query tile width
  - ``BLK_B`` / ``BLK_H``— request / head blocking factors (vectorised per
                           gather + per ``tl.dot`` batch dim)
  - ``scale``            — runtime scalar; applied to BOTH Q and K, so the
                           QK product carries ``scale**2`` (the standard
                           ``1/sqrt(D)`` softmax scaling expressed as a
                           ``1/sqrt(sqrt(D))`` factor on each operand)

``scale`` is a runtime arg (not constexpr); it is stashed in the input dict so
the oracle reads the exact value the kernel runs with, and threaded to the
kernel via the ``params`` \\ ``constexpr`` runtime-scalar flow in
``test_numerical`` (same mechanism as gather's ``y_offset``).
"""

import math

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def _make_inputs(
    B: int, H: int, Lq: int, Lk: int, D: int,
    CACHE: int, KV_BLOCK: int,
    *, seed: int,
) -> dict:
    """Build Q/K/V caches, a unique slot table, and a zeroed output buffer.

    The slot table maps each (request, key token) to a distinct physical row
    of the ``[CACHE, H, D]`` page pool.  Slots are sampled **without
    replacement** across the whole table so no two (request, token) pairs
    alias the same cache row — that keeps the oracle's gather unambiguous
    (an aliasing slot would make the result order-dependent only if the
    kernel wrote the cache, which it does not, but unique slots also make
    the reference a clean ``K[slots]`` advanced index).

    All tensors are f16 to match the kernel's ABI; ``scale`` is the
    ``1/sqrt(sqrt(D))`` factor applied to both Q and K.
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, Lq, H, D)).astype(np.float16)
    k = rng.standard_normal((CACHE, H, D)).astype(np.float16)
    v = rng.standard_normal((CACHE, H, D)).astype(np.float16)

    # Unique physical slots for every (request, key token) pair.
    n_slots = B * Lk
    assert n_slots <= CACHE, (
        f"B*Lk ({n_slots}) must be <= CACHE ({CACHE}) for unique slot sampling"
    )
    slots = rng.choice(CACHE, size=n_slots, replace=False)
    slots = slots.reshape(B, Lk).astype(np.int32)

    out = np.zeros((B, H, Lq, D), dtype=np.float16)
    scale = np.float32(1.0 / math.sqrt(math.sqrt(D)))
    return {
        "Q":     q,
        "K":     k,
        "V":     v,
        "SLOTS": slots,
        "Out":   out,
        "scale": scale,
    }


def make_inputs(
    B, H, Lq, Lk, CACHE, Tk, KV_BLOCK, BLOCK_Q, BLOCK_D, BLK_B, BLK_H,
    **_unused,
) -> dict:
    """Default variant inputs. The head dim is the kernel's ``BLOCK_D``;
    ``Tk``/``BLOCK_Q``/block factors don't shape the data — they're part of
    the param set so the framework can pass them uniformly."""
    del Tk, BLOCK_Q, BLK_B, BLK_H  # not needed to build the data
    return _make_inputs(B, H, Lq, Lk, BLOCK_D, CACHE, KV_BLOCK, seed=7)


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: standard attention over the gathered KV rows.

    Mirrors ``kernel.py`` exactly:

      1. Gather the physical K/V rows for each (request, key token) from the
         page pool: ``Kg[b, t] = K[SLOTS[b, t]]`` (shape ``[B, Lk, H, D]``),
         likewise for V.
      2. Scale Q and K each by ``scale`` (so the QK product carries
         ``scale**2``), compute scores ``[B, H, Lq, Lk]``, full (non-causal)
         softmax over the key axis, and weight V.
      3. Lay the result out as ``[B, H, Lq, D]``.

    The online-softmax loop in the kernel is mathematically equal to a single
    full softmax over all ``Lk`` keys, so the oracle computes the dense form.
    Accumulation is done in f32 (matching the kernel's f32 accumulators) and
    cast back to f16 for the comparison.
    """
    q = inputs["Q"].astype(np.float32)          # [B, Lq, H, D]
    k = inputs["K"].astype(np.float32)          # [CACHE, H, D]
    v = inputs["V"].astype(np.float32)          # [CACHE, H, D]
    slots = inputs["SLOTS"]                      # [B, Lk]
    scale = float(inputs["scale"])

    B, Lq, H, D = q.shape
    Lk = slots.shape[1]

    kg = k[slots]                                # [B, Lk, H, D]
    vg = v[slots]                                # [B, Lk, H, D]

    # Move head to a leading batch axis: [B, H, Lq, D] / [B, H, Lk, D].
    qh = np.transpose(q, (0, 2, 1, 3)) * scale
    kh = np.transpose(kg, (0, 2, 1, 3)) * scale
    vh = np.transpose(vg, (0, 2, 1, 3))

    scores = np.einsum("bhqd,bhkd->bhqk", qh, kh)        # [B, H, Lq, Lk]
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out = np.einsum("bhqk,bhkd->bhqd", p, vh)            # [B, H, Lq, D]
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE
# ---------------------------------------------------------------------------

SIGNATURE = {
    "Q":        "*fp16",
    "K":        "*fp16",
    "V":        "*fp16",
    "SLOTS":    "*i32",
    "Out":      "*fp16",
    "scale":    "fp32",
    "B":        "i32",
    "H":        "i32",
    "Lq":       "i32",
    "Lk":       "i32",
    "CACHE":    "i32",
    "Tk":       "i32",
    "KV_BLOCK": "i32",
    "BLOCK_Q":  "i32",
    "BLOCK_D":  "i32",
    "BLK_B":    "i32",
    "BLK_H":    "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

# Structural check shared across variants: the gather path must lower to an
# indirect access tile with no unrealized_conversion_cast left over (same
# invariant the gather fixture pins). Two gathers (K and V) are emitted.
_EXTRA_CHECKS = lambda t: (
    t.assert_absent("unrealized_conversion_cast"),
    t.assert_present("ktdp.construct_indirect_access_tile"),
)


VARIANTS = {
    "default": {
        # Small but representative paged-attention shape:
        #   B=4 requests, H=4 heads, Lq=Lk=64 tokens, D=64.
        #   KV_BLOCK=32 -> Tk=2 KV pages; BLOCK_Q=64 (single query tile).
        #   BLK_B=2, BLK_H=2 -> each gather pulls 2 requests' rows and the
        #   tl.dot batches over (2, 2). CACHE=4096 >> B*Lk=256 so unique
        #   slot sampling has plenty of room.
        #
        # All block factors evenly divide their axes (B%BLK_B==0,
        # H%BLK_H==0, Lq%BLOCK_Q==0, Lk%KV_BLOCK==0) so every tile is
        # in-bounds with no masking — same no-mask discipline as the
        # gather 2D variants. KV_BLOCK and D are powers of two
        # (validate_block_shape). The gather row count per step is
        # BLK_B*KV_BLOCK = 64 >= 8 (descriptor_gather verifier minimum).
        "kernel_fn":  kernel.paged_attn_NHD_kernel,
        "constexpr":  [
            "B", "H", "Lq", "Lk", "CACHE", "Tk",
            "KV_BLOCK", "BLOCK_Q", "BLOCK_D", "BLK_B", "BLK_H",
        ],
        "params": {
            "B":        [4],
            "H":        [4],
            "Lq":       [64],
            "Lk":       [64],
            "CACHE":    [4096],
            "Tk":       [2],
            "KV_BLOCK": [32],
            "BLOCK_Q":  [64],
            "BLOCK_D":  [64],
            "BLK_B":    [2],
            "BLK_H":    [2],
        },
        "tags":         ["descriptor-gather", "attention"],
        "grid":         [1],
        "parallel":     False,
        "reference":    run,
        "inputs":       make_inputs,
        "output_key":   "Out",
        # f16 accumulation through a softmax + two matmuls accumulates more
        # error than the pure-copy gather variants; loosen the tolerance.
        "rtol":         2e-2,
        "atol":         2e-2,
        "extra_checks": _EXTRA_CHECKS,
    },
}
