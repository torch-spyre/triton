#!/usr/bin/env python3
"""Non-power-of-2 shape support on the Spyre backend.

The power-of-2 requirement is a GPU ``LinearLayout`` / warp-tiling artifact, not
a Triton IR rule: the Spyre backend lowers tensors to KTIR/KTDP descriptors that
handle arbitrary sizes (no ``isPowerOf2`` anywhere in the Spyre C++ lowering).
Three gates enforced pow2; two are relaxed for Spyre here (the third is handled
in torch-spyre codegen and so has no ../triton test):

  * **Issue 1 — ``validate_block_shape`` (Python, ``triton/_utils.py``).**
    Per-element pow2 check reached via ``make_tensor_descriptor(block_shape=...)``.
    Skipped when ``target_info.is_spyre()``; the numel cap still applies.
  * **Issue 2 — ``verifyTensorSize`` trait (C++, ``lib/Dialect/Triton/IR/Traits.cpp``).**
    Total-numel pow2 check on every op operand/result carrying ``TensorSizeTrait``.
    Compiled out under ``TRITON_BUILD_TTIR_ONLY`` (the Spyre build), so a non-pow2
    tensor op verifies. Exercised by parsing MLIR (parse runs the verifier).
  * Issue 3 — the reduction ``tl.arange`` range check (``semantic.py``) — is
    avoided in torch-spyre by not emitting the reduction preamble.

These tests pin the Spyre relaxations and, for Issue 1, guard that the GPU/HIP pow2
rejection is left intact (the relaxation must not leak to other backends).
"""

import tempfile

import pytest
from triton._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape
from triton.backends.compiler import GPUTarget
from triton.language import target_info


def _target(backend):
    return GPUTarget(backend=backend, arch=1, warp_size=1)


@pytest.fixture
def as_backend(monkeypatch):
    """Pin ``current_target()`` to a chosen backend (or ``None`` = no driver)."""

    def _set(backend):
        target = None if backend is None else _target(backend)
        monkeypatch.setattr(target_info, "current_target", lambda: target)

    return _set


def _parse_mlir(mlir_text: str):
    """Parse (and thus verify) an MLIR module under the Spyre backend.

    ``parse_mlir_module`` runs the MLIR verifier, so a tensor op whose total
    element count is non-pow2 would be rejected by the ``verifyTensorSize`` trait
    unless it is compiled out (``TRITON_BUILD_TTIR_ONLY``). Returns the module.
    """
    from backend.compiler import SpyreBackend
    from triton._C.libtriton import ir

    backend = SpyreBackend(_target("spyre"))
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete_on_close=False
    ) as f:
        f.write(mlir_text)
        f.flush()
        return ir.parse_mlir_module(f.name, ctx)


# ---------------------------------------------------------------------------
# Issue 1 — validate_block_shape (Python frontend gate)
# ---------------------------------------------------------------------------


class TestBlockShapePow2:

    # 192 = 3 * 64: stick-aligned but not a power of 2 (the sum_non_pow2 reduced
    # dim); [4, 192, 64] is a full non-pow2 descriptor block shape.
    NON_POW2 = [4, 192, 64]
    POW2 = [4, 128, 64]

    def test_spyre_allows_non_pow2(self, as_backend):
        as_backend("spyre")
        assert validate_block_shape(self.NON_POW2) == 4 * 192 * 64

    @pytest.mark.parametrize("backend", ["cuda", "hip", None])
    def test_non_spyre_rejects_non_pow2(self, as_backend, backend):
        # The relaxation must not leak: GPU/HIP (and no-driver) still reject.
        as_backend(backend)
        with pytest.raises(ValueError, match="power of 2"):
            validate_block_shape(self.NON_POW2)

    @pytest.mark.parametrize("backend", ["spyre", "cuda", "hip"])
    def test_pow2_allowed_everywhere(self, as_backend, backend):
        as_backend(backend)
        assert validate_block_shape(self.POW2) == 4 * 128 * 64

    def test_numel_cap_still_enforced_on_spyre(self, as_backend):
        # Relaxing pow2 must not relax the maximum-numel cap.
        as_backend("spyre")
        with pytest.raises(ValueError, match="maximum tensor numel"):
            validate_block_shape([TRITON_MAX_TENSOR_NUMEL + 1])


# ---------------------------------------------------------------------------
# Issue 2 — verifyTensorSize trait (C++ op verifier, TRITON_BUILD_TTIR_ONLY)
# ---------------------------------------------------------------------------


class TestVerifyTensorSizeNonPow2:

    def test_non_pow2_pointwise_verifies(self):
        # tt.splat -> tensor<192xf32> (numel 192, non-pow2). tt.splat carries
        # TensorSizeTrait; parse+verify must accept it on the Spyre build.
        mod = _parse_mlir(
            """
        module {
          tt.func @k(%s: f32) -> tensor<192xf32> {
            %0 = tt.splat %s : f32 -> tensor<192xf32>
            tt.return %0 : tensor<192xf32>
          }
        }
        """
        )
        assert mod is not None

    def test_non_pow2_reduce_verifies(self):
        # tt.reduce over tensor<4x192x64xf32> (numel 49152, non-pow2) -> the exact
        # op sum_non_pow2 emits. The operand carries TensorSizeTrait; must verify.
        mod = _parse_mlir(
            """
        module {
          tt.func @k(%arg0: tensor<4x192x64xf32>) -> tensor<4x192xf32> {
            %0 = "tt.reduce"(%arg0) <{axis = 2 : i32}> ({
            ^bb0(%a: f32, %b: f32):
              %1 = arith.addf %a, %b : f32
              "tt.reduce.return"(%1) : (f32) -> ()
            }) : (tensor<4x192x64xf32>) -> tensor<4x192xf32>
            tt.return %0 : tensor<4x192xf32>
          }
        }
        """
        )
        assert mod is not None
