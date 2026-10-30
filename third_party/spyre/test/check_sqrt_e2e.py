"""Standalone end-to-end check for tl.sqrt -> spyreop.sqrt on real Spyre hardware.

Not a pytest test (no test_*.py name): a manual driver script for checking one
thing end to end -- TTIR -> KTIR -> spyrecode (lower_spyre_ops + dbo-opt) ->
binary -> real-device launch -- for the math.sqrt -> spyreop.sqrt conversion
in LowerSpyreOps.cpp specifically.

Run:
    source /home/erio/dt-inductor/torch-spyre-docs/scripts/dev-env.sh
    python third_party/spyre/test/check_sqrt_e2e.py

Requires `dbo-opt` on PATH (or TRITON_SPYRE_DBO_OPT set) and a reachable
Spyre device (TRITON_SPYRE_DEVICE if it isn't the default one).

The kernel below mirrors fixtures/elementwise/kernel.py's `elementwise_1d_device`
shape exactly (single tile, no distribution loop, static tensor-descriptor
layout) -- that shape is the only one in the suite dbo-opt lowers all the way
to a binary rather than stopping at KTIR; anything with an scf.for over
tl.program_id gets rejected by dbo-opt's scheduler.
"""

import glob
import os
import tempfile
import time
from pathlib import Path

import numpy as np

# torch BEFORE torch_spyre: torch auto-loads torch_spyre as a device-backend
# extension, so importing torch_spyre first re-enters a half-built module.
# See backend/driver.py's _import_torch_spyre for the same trap.
import torch
import torch_spyre  # noqa: F401 -- registers the "spyre" torch device

import triton
import triton.language as tl

N_ELEMENTS = 128
BLOCK_SIZE = 128
# fp32 stick layout: STICK_BYTES(128) // itemsize(4) = 32 lanes per stick.
# Same tuple _stick_1d("fp32") builds in fixtures/elementwise/meta.py, with the
# "stick" label stripped -- that label is only a test-ID marker in the fixture
# registry, not part of what tl.spyre_tensor_layout actually takes.
LAYOUT = ((0, "floordiv", 32), (0, "mod", 32))


@triton.jit
def sqrt_kernel_1d_device(x_ptr, output_ptr, n_elements: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr, LAYOUT: tl.constexpr):
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
    out_desc.store([offset], tl.sqrt(x))


def find_fresh_kernel_ktir(after: float):
    """The kernel.ktir this compile just handed to dbo-opt, if any.

    compiler.py's _make_spyrecode writes it under a fresh
    tempfile.TemporaryDirectory(prefix="spyrecode_", delete=not dbo_debug) --
    dbo_debug defaults True, so the dir survives the call, but nothing returns
    its path. Glob the system tempdir for the newest one created since `after`.
    Returns None (not an error) on a cache hit that skipped recompilation, or
    if dbo_debug was disabled -- callers should treat that as inconclusive,
    not as proof the conversion didn't happen.
    """
    candidates = [
        p for p in glob.glob(os.path.join(tempfile.gettempdir(), "spyrecode_*", "kernel.ktir"))
        if os.path.getmtime(p) >= after - 1.0
    ]
    return Path(max(candidates, key=os.path.getmtime)) if candidates else None


def main():
    rng = np.random.default_rng(0)
    # +0.1 keeps every value comfortably away from 0, where fp32 sqrt's
    # relative error grows fastest.
    x_host = (rng.random(N_ELEMENTS, dtype=np.float64).astype(np.float32) + 0.1)
    expected = np.sqrt(x_host)

    x = torch.from_numpy(x_host).to("spyre")
    out = torch.zeros(N_ELEMENTS, dtype=torch.float32, device="spyre")

    t0 = time.time()
    sqrt_kernel_1d_device[(1,)](x, out, N_ELEMENTS, BLOCK_SIZE, LAYOUT)

    ktir_path = find_fresh_kernel_ktir(after=t0)
    if ktir_path is None:
        print("WARN: no fresh kernel.ktir found (cache hit, or "
              "TRITON_SPYRE_DBO_DEBUG=0) -- skipping the spyreop.sqrt check; "
              "the numeric check below is still authoritative.")
    else:
        ktir_text = ktir_path.read_text()
        assert "spyreop.sqrt" in ktir_text, (
            f"lower_spyre_ops did not fire -- no spyreop.sqrt in {ktir_path}")
        assert "math.sqrt" not in ktir_text, (
            f"math.sqrt survived lower_spyre_ops in {ktir_path}")
        print(f"OK: spyreop.sqrt present, math.sqrt gone ({ktir_path})")

    result = out.cpu().numpy()
    assert np.count_nonzero(result) > 0, (
        "device wrote nothing: output is still all zeros")
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
    print(f"OK: device result matches NumPy sqrt oracle "
          f"(max abs err {np.max(np.abs(result - expected)):.2e})")


if __name__ == "__main__":
    main()
