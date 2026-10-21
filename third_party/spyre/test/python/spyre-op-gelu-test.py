# RUN: %python -m pytest %s -q
"""``tl.spyre_op("gelu", x)`` end to end: trace, device-neutral cache, Spyre lowering.

The whole point of ``ktdp.<name>`` as a layer is that the artifact cached at the
``ktir`` stage names *no* device-specific op -- a second backend reading that cache
would see a request, not an x86/Spyre-specific implementation. This file is the
one place that claim is checked directly, by inspecting the IR text at each stage
rather than only the final numeric result (which would pass even if ``ktdp.gelu``
leaked through unconverted, or converted too early).

Three stages, three assertions, matching ``third_party/spyre/docs/ktdp-spyre-intrinsics.md``'s
``tl.spyre_op(name, x) -> ktdp.<name> -> {spyreop.<name> | decomposition}`` chain:

- ``ttir`` (``_make_ttir``, via ``compile_to_ttir``): ``tts.spyre_op "gelu"`` --
  the TTIR-level staging op exists so a ``tl.spyre_op`` call is still visible as a
  *request* in a TTIR dump, before any KTIR-dialect op has been introduced.
- ``ktir`` (``_make_ktir``, via ``make_ktir_mod``): ``ktdp.gelu``, and neither
  ``spyreop.*`` nor ``tts.spyre_op`` -- the cached artifact is device-neutral.
- ``spyrecode`` (``_make_spyrecode``): the module it hands to ``dbo-opt`` --
  recorded in the archive's ``debug/00_ktir.mlir`` regardless of outcome, per
  ``knobs.spyre.dbo_debug`` -- has ``ConvertElementwiseToLinalg`` scalarized
  ``ktdp.gelu`` into a ``linalg.generic`` body and ``LowerSpyreOps`` rewritten
  that into ``spyreop.gelu``; no ``ktdp.gelu`` survives.

The last assertion needs ``dbo-opt`` (``_make_spyrecode`` shells out to it), so it
is gated on the ``dbo_opt`` fixture, same as ``spyrecode-compile-test.py``. The
first two need neither a tool nor a device and always run.
"""

import io
import zipfile

import triton
import triton.language as tl

from utils import compile_to_ttir, make_ktir_mod, spyre_target


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    x = x_desc.load([0])
    y = tl.spyre_op("gelu", x)
    out_desc.store([0], y)


_SIGNATURE = {"x_ptr": "*fp16", "out_ptr": "*fp16", "N": "constexpr", "BLOCK": "constexpr"}
_CONSTEXPRS = {"N": 128, "BLOCK": 128}


def _ttir_text(tmp_path):
    text = compile_to_ttir(_gelu_kernel, signature=_SIGNATURE, constexprs=_CONSTEXPRS)
    ttir_path = tmp_path / "gelu_kernel.ttir.mlir"
    ttir_path.write_text(text)
    return text, ttir_path


def test_ttir_names_the_request(tmp_path):
    text, _ = _ttir_text(tmp_path)
    assert 'tts.spyre_op "gelu"' in text


def test_ktir_is_device_neutral(tmp_path):
    _, ttir_path = _ttir_text(tmp_path)
    ktir_text = str(make_ktir_mod(str(ttir_path), grid=[1]))
    assert "ktdp.gelu" in ktir_text
    assert "spyreop" not in ktir_text
    assert "tts.spyre_op" not in ktir_text


def test_spyrecode_lowers_to_spyreop(dbo_opt, tmp_path):
    from backend.compiler import SpyreBackend

    _, ttir_path = _ttir_text(tmp_path)
    ktir_mod = make_ktir_mod(str(ttir_path), grid=[1])

    backend = SpyreBackend(spyre_target())
    options = backend.parse_options({"grid": [1]})
    zip_bytes = backend._make_spyrecode(ktir_mod, {}, options)

    debug_text = zipfile.ZipFile(io.BytesIO(zip_bytes)).read("debug/00_ktir.mlir").decode("utf-8")
    assert "spyreop.gelu" in debug_text
    assert "ktdp.gelu" not in debug_text
