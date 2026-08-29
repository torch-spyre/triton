# RUN: %python %s
"""An over-cap block shape reaches KTIR with its extent intact.

Both halves of the relaxation are unit-tested by
test/Triton/tensor-size-trait-roundtrip.mlir, which feeds the verifier directly.
What no unit test can show is the pair working together: with only
validate_block_shape relaxed, every test in the suite passed while an over-cap
descriptor was still rejected during TTIR generation, by the MLIR trait. So this
runs the shape a kernel author actually writes through TTIR and the KTDP passes.

Two things are asserted per shape, and the second is why this is not just "did it
raise": an over-cap block could lower without complaint while some pass quietly
re-tiled it, which compiles fine and computes the wrong thing. The extent recorded
in ktdp.access_tile is the contract.

A plain script rather than pytest, following dataflow-scheduler's
test/python/register_everything.py: no pytest, no fixtures, no classes, no
monkeypatching, and an exception fails the test. Nothing here needs dbo-opt or a
device, so the file carries no feature requirement. The kernel comes from the
example registry the rest of the suite uses, like backend-options-test.py.

The hyphen in the filename is load-bearing: pytest collects test_*.py and
*_test.py, so an underscore would have the pytest suite re-run this alongside lit.
"""

import tempfile

from triton._utils import TRITON_MAX_TENSOR_NUMEL, validate_block_shape
from triton.language import target_info

from conftest import EXAMPLES
from utils import compile_to_ttir, make_ktir_mod

# One entry per block shape to drive through TTIR -> KTIR. Adding a row is the
# whole edit; the guard below keeps a row that does not exercise the relaxation,
# or an empty list, from passing vacuously.
SHAPES = [
    (2048, 4096),   # 8Mi, 8x the cap. Powers of two, so only the cap is tripped
    (3000, 3000),   # 9M, over the cap *and* a non-pow2 total: both limits at once
]

# The *dynamic* variant specifically: M and N are runtime i32 while BLOCK_M and
# BLOCK_N are constexprs, so the block can grow past the cap without the shape
# arguments having to follow. A static variant would describe a block larger than
# its own tensor. No registered variant is over-cap on its own -- the largest
# tensor across all 106 is 131,072, an eighth of the cap.
_EXAMPLE = EXAMPLES["elementwise__2d_dynamic[M=520]"]


def assert_over_cap_block_survives_lowering(block_m, block_n):
    numel = block_m * block_n
    assert numel > TRITON_MAX_TENSOR_NUMEL, (
        f"{block_m}x{block_n} is {numel} elements, not over the "
        f"{TRITON_MAX_TENSOR_NUMEL} cap -- this shape does not reach the "
        f"relaxation and would pass with or without it")

    constexprs = {**_EXAMPLE["constexprs"], "BLOCK_M": block_m, "BLOCK_N": block_n}
    ttir = compile_to_ttir(
        _EXAMPLE["kernel_fn"], _EXAMPLE["signature"], constexprs)

    want_tensor = f"tensor<{block_m}x{block_n}xf32>"
    assert want_tensor in ttir, (
        f"{block_m}x{block_n}: {want_tensor} not in TTIR -- the block was "
        f"rejected or reshaped before KTIR")

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete_on_close=False) as f:
        f.write(ttir)
        f.flush()
        ktir = str(make_ktir_mod(f.name, grid=[1]))

    want_tile = f"access_tile<{block_m}x{block_n}xindex>"
    assert want_tile in ktir, (
        f"{block_m}x{block_n}: {want_tile} not in KTIR -- the extent did not "
        f"survive lowering")


def assert_frontend_gate_relaxed(block_m, block_n):
    """validate_block_shape directly, which the lowering above only reaches
    incidentally. No target faking is needed: this build has one backend, so
    current_target() already reports spyre and is_spyre() is already True.
    """
    assert target_info.is_spyre(), (
        "current_target() is not spyre, so the relaxation under test is not the "
        "branch being taken -- this file assumes a Spyre-only build")
    assert validate_block_shape([block_m, block_n]) == block_m * block_n


assert SHAPES, "SHAPES is empty: nothing would be checked"
for m, n in SHAPES:
    assert_frontend_gate_relaxed(m, n)
    assert_over_cap_block_survives_lowering(m, n)
    print(f"ok {m}x{n}")
