#!/usr/bin/env python3
"""``tl.spyre_op`` from the kernel line to the module dbo-opt receives.

Three stages, one claim each:

* tracing builds a ``tts.spyre_op`` holding the registered fallback;
* the ``ktir`` artifact holds that fallback alone, inlined and tagged
  ``tts.spyreop_hint`` -- no ``tts`` op, which a KTIR reader could not parse;
* the ``spyrecode`` pipeline replaces the tagged body with exactly one
  ``spyreop.<name>``, and no tag is left for dbo-opt.

The last is reproduced with the registered pipeline rather than through the
``spyrecode`` stage itself, which shells out to dbo-opt -- CLAUDE.md's
"Reproducing that module". And the refusals the frontend owns: a name the backend
does not register, a dtype the intrinsic does not take, and the wrong arity, each
at the kernel line.

Numerical coverage of the fallbacks, through ``ktir_cpu``, is the
``intrinsic_request`` fixture's.
"""

import pytest
import triton
import triton.language as tl

from utils import compile_to_ttir, make_ktir_mod

_REQUESTS = [("gelu", "fp16"), ("silu", "fp16"), ("silu", "fp32"),
             ("sigmoid", "fp16"), ("sigmoid", "fp32")]


@triton.jit
def _request(x_ptr, out_ptr, N: tl.constexpr, OP: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[N], strides=[1], block_shape=[N])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[N])
    out_desc.store([0], tl.spyre_op(OP, x_desc.load([0])))


def _trace(kernel, op, dtype):
    signature = {"x_ptr": f"*{dtype}", "out_ptr": f"*{dtype}", "N": "constexpr", "OP": "constexpr"}
    return compile_to_ttir(kernel, signature, {"N": 128, "OP": op})


def _ktir(ttir, tmp_path):
    path = tmp_path / "kernel.ttir"
    path.write_text(ttir)
    return make_ktir_mod(path, grid=[1])


def _spyrecode(mod):
    """The module the ``spyrecode`` stage hands dbo-opt, in symbolic mode."""
    from triton._C.libtriton import ir, spyre
    pm = ir.pass_manager(mod.context)
    spyre.passes.ttir_to_ktdp.add_spyrecode_pipeline(pm)
    pm.run(mod, "spyrecode")
    return str(mod)


@pytest.mark.parametrize("op,dtype", _REQUESTS, ids=[f"{o}-{d}" for o, d in _REQUESTS])
def test_request_reaches_one_intrinsic(op, dtype, tmp_path):
    ttir = _trace(_request, op, dtype)
    # The request, holding the fallback as a call the `ttir` stage inlines.
    assert f'tts.spyre_op "{op}"' in ttir
    assert "tts.spyreop_yield" in ttir

    mod = _ktir(ttir, tmp_path)
    ktir = str(mod)
    assert "tts.spyre_op" not in ktir and "tts.spyreop_yield" not in ktir
    assert f'tts.spyreop_hint = {{id = 0 : i64, name = "{op}"}}' in ktir
    assert "tt.call" not in ktir

    spyrecode = _spyrecode(mod)
    elem = {"fp16": "f16", "fp32": "f32"}[dtype]
    assert spyrecode.count(f"spyreop.{op} ") == 1
    assert f"spyreop.{op} %" in spyrecode and f": {elem}" in spyrecode
    # The fallback went whole, casts included, and so did every tag.
    assert "tts.spyreop_hint" not in spyrecode
    assert "math." not in spyrecode
    assert "arith.extf" not in spyrecode and "arith.truncf" not in spyrecode


def _raises(kernel, op, dtype, match):
    from triton.compiler.errors import CompilationError
    with pytest.raises(CompilationError) as exc:
        _trace(kernel, op, dtype)
    assert match in str(exc.value)


def test_unregistered_name_is_refused():
    _raises(_request, "softplus", "fp16",
            "tl.spyre_op: no intrinsic named 'softplus'; the Spyre backend registers "
            "gelu, sigmoid, silu")


def test_dtype_the_intrinsic_does_not_take_is_refused():
    # spyreop.gelu is f16 only; LowerSpyreOps would refuse the same request later.
    _raises(_request, "gelu", "fp32", "tl.spyre_op('gelu'): the intrinsic takes ['fp16'], not fp32")


def test_wrong_arity_is_refused():

    @triton.jit
    def two_operands(x_ptr, out_ptr, N: tl.constexpr, OP: tl.constexpr):
        x_desc = tl.make_tensor_descriptor(x_ptr, shape=[N], strides=[1], block_shape=[N])
        out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[N])
        x = x_desc.load([0])
        out_desc.store([0], tl.spyre_op(OP, x, x))

    _raises(two_operands, "silu", "fp32", "tl.spyre_op('silu'): takes 1 tensor operand(s), got 2")


@pytest.mark.parametrize("backend", ["cuda", "hip", None])
def test_raises_off_backend(monkeypatch, backend):
    # The guard fires before the builtin touches ``_semantic`` or its operands.
    from triton.backends.compiler import GPUTarget
    from triton.language import target_info
    target = None if backend is None else GPUTarget(backend=backend, arch=1, warp_size=1)
    monkeypatch.setattr(target_info, "current_target", lambda: target)
    with pytest.raises(ValueError, match="only supported on the 'spyre' backend"):
        tl.spyre_op("gelu", None, _semantic=object())
