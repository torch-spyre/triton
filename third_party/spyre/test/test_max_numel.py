#!/usr/bin/env python3
"""Maximum-tensor-numel relaxation on the Spyre backend.

Triton's frontend caps a block shape's total element count at
``TRITON_MAX_TENSOR_NUMEL`` (2**20) in ``validate_block_shape``
(``triton/_utils.py``). Every ``block_type`` construction goes through it, so
``tl.make_tensor_descriptor(block_shape=...)``, ``tl.zeros``, ``tl.full`` and
``tl.arange`` all inherit the cap.

The cap is a GPU register / shared-memory budget: there a tensor is a
register-resident distributed value, so its element count tracks physical
capacity. On Spyre a ``block_shape`` describes an HBM region through a
KTIR/KTDP descriptor that the lowering re-tiles into access tiles streaming
through scratchpad, so it is not a resident footprint and the cap measures the
wrong thing. It rejected kernels Spyre can lower -- concretely matmul's B
operand in device stick layout (``[N//64, K, 64]``, 8Mi elements at
K=2048/N=4096), which is why the ``matmul`` fixture's stick-layout variants run
on toy M/K/N.

Two independent gates enforced the cap, and both are relaxed for Spyre:

  * ``validate_block_shape`` (Python) — skipped when ``target_info.is_spyre()``.
  * the ``verifyTensorSize`` MLIR trait (``lib/Dialect/Triton/IR/Traits.cpp``),
    carried by every op with ``TensorSizeTrait`` — the whole trait is a no-op
    under ``TRITON_BUILD_TTIR_ONLY``, since its other limit (power-of-two) is a
    GPU artifact too. Relaxing only the Python gate left an over-cap descriptor
    rejected during TTIR generation, which is why the end-to-end tests below
    exist rather than only unit tests of the gate.

No replacement gate exists: ``SpyreOptions.lx_size`` (2 MB scratchpad per core)
has no reader. The real limit is scratchpad bytes, which is not knowable from a
block shape alone and belongs to the backend.

These tests pin the relaxation and guard that it does not leak to the GPU
backends. The sibling pow2 relaxation in the same function lives in
``test_non_pow2.py``.
"""

import math
import tempfile

import pytest
import triton
import triton.language as tl
from triton._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape
from triton.backends.compiler import GPUTarget
from triton.language import target_info
from utils import compile_to_ttir, make_ktir_mod


def _target(backend):
    return GPUTarget(backend=backend, arch=1, warp_size=1)


@pytest.fixture
def as_backend(monkeypatch):
    """Pin ``current_target()`` to a chosen backend (or ``None`` = no driver)."""

    def _set(backend):
        target = None if backend is None else _target(backend)
        monkeypatch.setattr(target_info, "current_target", lambda: target)

    return _set


class TestBlockShapeNumelCap:

    # matmul's B operand in device stick layout -- [N//64, K, 64] at K=2048,
    # N=4096, the shape from the motivating issue. 8,388,608 elements = 8x the
    # cap. Every dim is a power of two on purpose: the pow2 check runs first and
    # is itself Spyre-gated, so an all-pow2 shape isolates the numel cap as the
    # only gate under test on every backend.
    OVER_CAP = [64, 2048, 64]
    # Exactly at the cap. The check is `>`, so this is accepted everywhere.
    AT_CAP = [TRITON_MAX_TENSOR_NUMEL]

    def test_spyre_allows_over_cap(self, as_backend):
        as_backend("spyre")
        assert validate_block_shape(self.OVER_CAP) == math.prod(self.OVER_CAP)

    @pytest.mark.parametrize("backend", ["cuda", "hip", None])
    def test_non_spyre_rejects_over_cap(self, as_backend, backend):
        # The relaxation must not leak: GPU/HIP (and no-driver) still reject.
        # Matching the message proves the numel cap fired, not the pow2 check.
        as_backend(backend)
        with pytest.raises(ValueError, match="maximum tensor numel"):
            validate_block_shape(self.OVER_CAP)

    @pytest.mark.parametrize("backend", ["spyre", "cuda", "hip"])
    def test_at_cap_allowed_everywhere(self, as_backend, backend):
        as_backend(backend)
        assert validate_block_shape(self.AT_CAP) == math.prod(self.AT_CAP)

    def test_constexpr_int_check_still_applies_on_spyre(self, as_backend):
        # Relaxing the numel cap must not relax the element-type check.
        as_backend("spyre")
        with pytest.raises(TypeError, match=r"constexpr\[int\]"):
            validate_block_shape([64, "2048", 64])


# ---------------------------------------------------------------------------
# End to end — the Python gate is not the only place the cap lives
# ---------------------------------------------------------------------------


@triton.jit
def _desc_copy(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Descriptor copy whose block shape carries the element count under test."""
    src = tl.make_tensor_descriptor(in_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    dst = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    dst.store([0, 0], src.load([0, 0]))


_SIG = {
    "in_ptr": "*fp16",
    "out_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
}

# (BLOCK_M, BLOCK_N) — at the cap, and the motivating issue's 8x-cap matmul B
# operand expressed as a block the descriptor path takes directly.
_BLOCKS = [
    pytest.param(1024, 1024, id="1024x1024_at_cap"),
    pytest.param(2048, 4096, id="2048x4096_8x_cap"),
]


class TestOverCapKernelLowers:
    """Drive an over-cap element count through TTIR generation and the KTDP passes.

    ``validate_block_shape`` is only the Python gate. ``verifyTensorSize``
    (``lib/Dialect/Triton/IR/Traits.cpp``) is an MLIR trait carried by every op
    with ``TensorSizeTrait`` and enforced the same cap independently, so a
    unit test of the gate alone passes while an over-cap descriptor is still
    rejected during TTIR generation. These tests exercise both gates plus the
    lowering, which is what the motivating kernel actually needs.
    """

    @pytest.mark.parametrize("block_m, block_n", _BLOCKS)
    def test_ttir_generation_accepts(self, block_m, block_n):
        ttir = compile_to_ttir(_desc_copy, _SIG, {"BLOCK_M": block_m, "BLOCK_N": block_n})
        assert f"tensor<{block_m}x{block_n}xf16>" in ttir

    @pytest.mark.parametrize("block_m, block_n", _BLOCKS)
    def test_ktdp_pipeline_accepts(self, block_m, block_n):
        ttir = compile_to_ttir(_desc_copy, _SIG, {"BLOCK_M": block_m, "BLOCK_N": block_n})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete_on_close=False) as f:
            f.write(ttir)
            f.flush()
            mod = make_ktir_mod(f.name, grid=[1])
        # The block survives as one device access of the same extent. A silent
        # re-tile or truncation would still lower without raising, so the shape
        # is what pins the contract, not the absence of an exception.
        assert f"access_tile<{block_m}x{block_n}xindex>" in str(mod)
