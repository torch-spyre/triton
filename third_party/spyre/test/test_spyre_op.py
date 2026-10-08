#!/usr/bin/env python3
"""``tl.spyre_op`` from the kernel line to the module dbo-opt receives.

Every case names the intrinsic table's test-only entry, ``test_mock``, and binds
its own fallback to it, so nothing here changes when an intrinsic is added. The
claims, one per stage:

* tracing builds a ``tts.spyre_op`` holding the bound fallback, which its region
  verifier admits;
* the ``ktir`` artifact holds that fallback alone, inlined and hinted
  ``tts.spyreop_hint`` -- no ``tts`` op, which a KTIR reader could not parse;
* the ``spyrecode`` pipeline fuses the hinted ops into one body and replaces it
  with exactly one spyreop op, and no hint is left for dbo-opt.

The last is reproduced with the registered pipeline rather than through the
``spyrecode`` stage itself, which shells out to dbo-opt -- CLAUDE.md's
"Reproducing that module". And the refusals the frontend owns: a name the table
does not have, a dtype the intrinsic does not take, the wrong arity, a fallback
returning other types than declared, and ``test_mock`` with no fallback
registered.

Numerical coverage of the real fallbacks, through ``ktir_cpu``, is the
``spyreop`` fixture's.
"""

import re

import pytest
import triton
import triton.language as tl

# The module compile_to_ttir's SpyreBackend reads its fallbacks from.
from backend import intrinsics
from utils import compile_to_ttir, make_ktir_mod


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


@pytest.fixture
def register(monkeypatch):
    """``spyre_intrinsic``, over a throw-away copy of the fallback table, so a
    test's registrations are gone when it ends."""
    monkeypatch.setattr(intrinsics, "FALLBACKS", dict(intrinsics.FALLBACKS))
    return intrinsics.spyre_intrinsic


@triton.jit
def func1(x):
    xf = x.to(tl.float32)
    return (xf * xf + xf).to(x.dtype)


@triton.jit
def func2(x):
    xf = x.to(tl.float32)
    return (tl.exp(xf) - 1.0).to(x.dtype)


@pytest.mark.parametrize("fallback", [func1, func2], ids=["func1", "func2"])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_call_site_reaches_one_intrinsic(register, fallback, dtype, tmp_path):
    register("test_mock")(fallback)
    ttir = _trace(_request, "test_mock", dtype)
    # The call site, verified, holding the fallback as a call the `ttir` stage
    # inlines.
    assert 'tts.spyre_op "test_mock"' in ttir
    assert "tts.spyreop_yield" in ttir

    mod = _ktir(ttir, tmp_path)
    ktir = str(mod)
    assert "tts.spyre_op" not in ktir and "tts.spyreop_yield" not in ktir
    assert 'tts.spyreop_hint = {id = 0 : i64, name = "test_mock"}' in ktir
    assert "tt.call" not in ktir

    spyrecode = _spyrecode(mod)
    elem = {"fp16": "f16", "fp32": "f32"}[dtype]
    # Fused into one body and selected: one spyreop op, at the operand's type.
    selected = re.findall(r"(spyreop\.\w+) %\S+ : (\w+)", spyrecode)
    assert len(selected) == 1 and selected[0][1] == elem, selected
    # The fallback went whole, casts included, and so did every hint.
    assert "tts.spyreop_hint" not in spyrecode
    assert "math." not in spyrecode
    assert "arith.extf" not in spyrecode and "arith.truncf" not in spyrecode


def _raises(kernel, op, dtype, match):
    from triton.compiler.errors import CompilationError
    with pytest.raises(CompilationError) as exc:
        _trace(kernel, op, dtype)
    assert match in str(exc.value)


@triton.jit
def _constant_request(out_ptr, N: tl.constexpr, OP: tl.constexpr):
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[N])
    out_desc.store([0], tl.spyre_op(OP, tl.full([N], 1.0, tl.float32)))


def test_constant_operand_is_refused(register, tmp_path, capfd):
    register("test_mock")(func1)
    ttir = compile_to_ttir(_constant_request, {"out_ptr": "*fp32", "N": "constexpr", "OP": "constexpr"},
                           {"N": 128, "OP": "test_mock"})
    with pytest.raises(Exception):
        _ktir(ttir, tmp_path)
    assert 'tl.spyre_op("test_mock"): operand 0 is a constant' in capfd.readouterr().err


def test_unregistered_name_is_refused():
    _raises(_request, "softplus", "fp16", "tl.spyre_op: no intrinsic named 'softplus'; the Spyre backend registers")


def test_test_mock_without_a_fallback_is_refused():
    assert intrinsics.TEST_MOCK not in intrinsics.FALLBACKS
    _raises(_request, "test_mock", "fp32", "tl.spyre_op: no fallback is registered for 'test_mock'")


def test_dtype_the_intrinsic_does_not_take_is_refused(register):
    register("test_mock")(func1)
    _raises(_request, "test_mock", "bf16", "tl.spyre_op('test_mock'): the intrinsic takes ['fp16', 'fp32'], not bf16")


def test_wrong_arity_is_refused(register):
    register("test_mock")(func1)

    @triton.jit
    def two_operands(x_ptr, out_ptr, N: tl.constexpr, OP: tl.constexpr):
        x_desc = tl.make_tensor_descriptor(x_ptr, shape=[N], strides=[1], block_shape=[N])
        out_desc = tl.make_tensor_descriptor(out_ptr, shape=[N], strides=[1], block_shape=[N])
        x = x_desc.load([0])
        out_desc.store([0], tl.spyre_op(OP, x, x))

    _raises(two_operands, "test_mock", "fp32", "tl.spyre_op('test_mock'): takes 1 tensor operand(s), got 2")


def test_fallback_returning_other_types_is_refused(register):

    @register("test_mock")
    @triton.jit
    def widens(x):
        return x.to(tl.float32)

    _raises(_request, "test_mock", "fp16", "tl.spyre_op('test_mock'): the registered fallback returns")


def test_intrinsic_table_names_are_unique():
    # The C++ table is checked where it is defined; this pins that what the
    # frontend reads from it has one entry per name.
    from triton._C.libtriton import spyre
    names = [entry["name"] for entry in spyre.intrinsics.table()]
    assert len(names) == len(set(names))


def test_every_table_name_has_one_fallback():
    table = intrinsics._read_table()
    intrinsics._check_names(intrinsics.FALLBACKS, table)
    missing = dict(list(intrinsics.FALLBACKS.items())[1:])
    with pytest.raises(RuntimeError, match="the C\\+\\+ intrinsic table names"):
        intrinsics._check_names(missing, table)


def test_a_name_the_table_does_not_define_is_refused(register):
    with pytest.raises(ValueError, match="'softplus' is not in the C\\+\\+ intrinsic table"):
        register("softplus")(func1)


def test_a_second_fallback_for_one_name_is_refused(register):
    register("test_mock")(func1)
    with pytest.raises(ValueError, match="intrinsic 'test_mock' has two fallbacks"):
        register("test_mock")(func2)


@pytest.mark.parametrize("backend", ["cuda", "hip", None])
def test_raises_off_backend(monkeypatch, backend):
    # The guard fires before the builtin touches ``_semantic`` or its operands.
    from triton.backends.compiler import GPUTarget
    from triton.language import target_info
    target = None if backend is None else GPUTarget(backend=backend, arch=1, warp_size=1)
    monkeypatch.setattr(target_info, "current_target", lambda: target)
    with pytest.raises(ValueError, match="only supported on the 'spyre' backend"):
        tl.spyre_op("test_mock", None, _semantic=object())
