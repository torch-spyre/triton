#!/usr/bin/env python3
"""Unit tests for the reusable Spyre frontend backend guard, and the two
upstream dtype refusals forked behind it.

``triton.language.target_info`` gains two helpers (next to ``is_cuda`` /
``is_hip``) that let frontend code diverge by backend at *runtime* — needed
because frontend op construction runs before any pass, so the C++
``TRITON_BUILD_TTIR_ONLY`` compile flag can't reach it:

  * :func:`is_spyre` — a predicate, for behavior that *forks* by backend
    (``if is_spyre(): <relaxed> else: <strict>``), e.g. the
    descriptor_gather/scatter rank-N relaxation.
  * :func:`requires_backend` — a decorator that raises on the wrong backend,
    for ops that *only exist* for one backend, e.g. ``tl.spyre_tensor_layout``.

Both resolve the backend through ``current_target()``, so these tests drive
them by monkeypatching ``current_target`` to return a chosen ``GPUTarget`` —
no active driver required.

The second half of this file covers the two places ``is_spyre()`` is used to
relax an upstream *dtype* refusal, which is what makes an fp16 softmax
traceable at all:

  * ``language/standard.py``'s ``max``/``min`` widen a sub-32-bit float to fp32
    before reducing. On Spyre that emits ``arith.extf``/``arith.truncf``, which
    no pass in this tree lowers.
  * ``language/math.py``'s ``_check_dtype`` refuses anything but fp32/fp64 for
    ``exp`` and its siblings, as a ``ValueError`` while tracing.

fp16 is not a preference on Spyre, it is the only width that works: a reduce's
statistic is written replicated across a stick and read back one lane wide then
splatted, and that splat is an fp16-only path in the backend. See the rationale
at each fork for the accuracy trade being accepted.
"""

import pytest
from triton.backends.compiler import GPUTarget
from triton.language import core, math, standard, target_info


def _target(backend):
    return GPUTarget(backend=backend, arch=1, warp_size=1)


@pytest.fixture
def as_backend(monkeypatch):
    """Return a setter that pins ``current_target()`` to a given backend (or
    ``None`` for "no active driver")."""

    def _set(backend):
        target = None if backend is None else _target(backend)
        monkeypatch.setattr(target_info, "current_target", lambda: target)

    return _set


# ---------------------------------------------------------------------------
# is_spyre — predicate
# ---------------------------------------------------------------------------

class TestIsSpyre:

    def test_true_on_spyre(self, as_backend):
        as_backend("spyre")
        assert target_info.is_spyre() is True

    @pytest.mark.parametrize("backend", ["cuda", "hip"])
    def test_false_on_other_backends(self, as_backend, backend):
        as_backend(backend)
        assert target_info.is_spyre() is False

    def test_false_when_no_active_target(self, as_backend):
        # current_target() returns None when there is no active driver.
        as_backend(None)
        assert target_info.is_spyre() is False


# ---------------------------------------------------------------------------
# requires_backend — decorator
# ---------------------------------------------------------------------------

class TestRequiresBackend:

    @staticmethod
    @target_info.requires_backend("spyre")
    def _spyre_only(x):
        return x + 1

    def test_runs_on_matching_backend(self, as_backend):
        as_backend("spyre")
        assert self._spyre_only(41) == 42

    @pytest.mark.parametrize("backend", ["cuda", "hip"])
    def test_raises_on_other_backend(self, as_backend, backend):
        as_backend(backend)
        with pytest.raises(ValueError, match="only supported on the 'spyre' backend"):
            self._spyre_only(0)

    def test_raises_when_no_active_target(self, as_backend):
        as_backend(None)
        with pytest.raises(ValueError, match="only supported on the 'spyre' backend"):
            self._spyre_only(0)

    def test_resolves_backend_at_call_time_not_decoration_time(self, as_backend):
        # The same decorated callable passes under spyre and fails otherwise —
        # proving the check is deferred to each call, not bound at decoration.
        as_backend("spyre")
        assert self._spyre_only(1) == 2
        as_backend("cuda")
        with pytest.raises(ValueError):
            self._spyre_only(1)

    def test_error_names_the_function_and_actual_backend(self, as_backend):
        as_backend("cuda")
        with pytest.raises(ValueError) as exc:
            self._spyre_only(0)
        msg = str(exc.value)
        assert "_spyre_only" in msg
        assert "'cuda'" in msg


# ---------------------------------------------------------------------------
# standard.py's max/min — the narrow-float widening fork
# ---------------------------------------------------------------------------

class TestNarrowFloatMinMaxFork:
    """``standard._widens_narrow_float_reduce`` is the predicate ``max``/``min``
    consult before widening a sub-32-bit float to fp32.

    It is a ``constexpr_function``, so inside a ``@jit`` body the code generator
    threads ``_semantic`` in and it yields a ``constexpr``; called from ordinary
    Python as here, it returns a plain bool. Asserted with ``is`` on purpose —
    a ``constexpr`` leaking out would be truthy and a looser assertion would not
    notice.
    """

    def test_spyre_does_not_widen(self, as_backend):
        as_backend("spyre")
        assert standard._widens_narrow_float_reduce() is False

    @pytest.mark.parametrize("backend", ["cuda", "hip"])
    def test_every_other_backend_still_widens(self, as_backend, backend):
        as_backend(backend)
        assert standard._widens_narrow_float_reduce() is True

    def test_no_active_target_still_widens(self, as_backend):
        # Upstream behaviour is the default: absent a target, nothing is relaxed.
        as_backend(None)
        assert standard._widens_narrow_float_reduce() is True


# ---------------------------------------------------------------------------
# math.py's _check_dtype — the fp16 admission fork
# ---------------------------------------------------------------------------

class _FakeTensor(core.tensor):
    """A ``core.tensor`` carrying a dtype and nothing else.

    ``_check_dtype`` reads exactly two things off each argument: that it *is* a
    ``core.tensor`` (a real ``isinstance``, so a mock will not do) and
    ``arg.type.scalar.name``. Building a real tensor would need an MLIR builder
    and a value handle; subclassing and setting ``type`` needs neither.
    """

    def __init__(self, dtype):
        self.type = core.block_type(dtype, [4])


class TestCheckDtypeFork:

    # Stand-ins for the two shapes of upstream list, decorated here rather than
    # calling tl.exp so the test measures the decorator and not a builtin's body.
    @staticmethod
    @math._check_dtype(dtypes=["fp32", "fp64"])
    def _float_op(x):
        return "ran"

    @staticmethod
    @math._check_dtype(dtypes=["int32", "int64", "uint32", "uint64"])
    def _int_op(x):
        return "ran"

    def test_fp16_admitted_on_spyre(self, as_backend):
        as_backend("spyre")
        assert self._float_op(_FakeTensor(core.float16)) == "ran"

    @pytest.mark.parametrize("backend", ["cuda", "hip", None])
    def test_fp16_still_refused_elsewhere(self, as_backend, backend):
        as_backend(backend)
        with pytest.raises(ValueError, match="Expected dtype"):
            self._float_op(_FakeTensor(core.float16))

    def test_declared_dtypes_still_admitted_on_spyre(self, as_backend):
        # The fork only ever *adds*; fp32/fp64 must not be lost on the way.
        as_backend("spyre")
        assert self._float_op(_FakeTensor(core.float32)) == "ran"
        assert self._float_op(_FakeTensor(core.float64)) == "ran"

    def test_bf16_deliberately_not_admitted_on_spyre(self, as_backend):
        # Not an oversight: nothing in this tree has run bf16 through these ops,
        # and standard.py's _promote_bfloat16_to_float32 still widens bf16
        # reduces, so admitting it here would claim support the frontend as a
        # whole does not have.
        as_backend("spyre")
        with pytest.raises(ValueError, match="Expected dtype"):
            self._float_op(_FakeTensor(core.bfloat16))

    def test_integer_op_untouched_on_spyre(self, as_backend):
        # Gated on the declared list containing fp32, so umulhi's integer list
        # is exactly what upstream wrote. A float relaxation must not leak into
        # an op that never accepted floats.
        as_backend("spyre")
        with pytest.raises(ValueError, match="Expected dtype"):
            self._int_op(_FakeTensor(core.float16))
        assert self._int_op(_FakeTensor(core.int32)) == "ran"

    def test_error_reports_the_set_actually_accepted(self, as_backend):
        # On Spyre the message has to name fp16 too, or it tells the reader to
        # cast to a type that was already allowed.
        as_backend("spyre")
        with pytest.raises(ValueError) as exc:
            self._float_op(_FakeTensor(core.bfloat16))
        msg = str(exc.value)
        assert "fp16" in msg and "bf16" in msg

    def test_resolves_backend_at_call_time_not_decoration_time(self, as_backend):
        # There is no active driver when language/math.py is imported, so the
        # relaxation cannot be baked in at decoration; the same callable has to
        # answer differently per call.
        as_backend("cuda")
        with pytest.raises(ValueError):
            self._float_op(_FakeTensor(core.float16))
        as_backend("spyre")
        assert self._float_op(_FakeTensor(core.float16)) == "ran"


# ---------------------------------------------------------------------------
# Both forks, in the IR they actually produce
#
# The unit tests above pin the predicates; this pins the consequence. It traces
# for real, so it needs the active driver to be Spyre rather than a
# monkeypatched target -- ``is_spyre()`` asks the *driver*, not the compile
# target handed to ``make_ir``.
# ---------------------------------------------------------------------------

class TestForkedGuardsInTracedIR:

    @staticmethod
    def _ttir():
        import triton
        import triton.language as tl
        from utils import compile_to_ttir

        @triton.jit
        def fp16_max_and_exp(x_ptr, stat_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            """The two forked guards in one kernel: an fp16 row max, and an fp16
            exp. Descriptors rather than pointer arithmetic, matching the
            fixtures -- nothing here goes past TTIR, so no layout is needed."""
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            stat_desc = tl.make_tensor_descriptor(
                stat_ptr, shape=[M], strides=[1], block_shape=[M])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            x = x_desc.load([0, 0])
            stat_desc.store([0], tl.max(x, axis=1))
            out_desc.store([0, 0], tl.exp(x))

        signature = {"x_ptr": "*fp16", "stat_ptr": "*fp16", "out_ptr": "*fp16",
                     "M": "constexpr", "N": "constexpr"}
        return compile_to_ttir(fp16_max_and_exp, signature, {"M": 64, "N": 64})

    @pytest.fixture(scope="class")
    def ttir(self):
        if not target_info.is_spyre():
            pytest.skip("needs the Spyre driver active: the fork reads the "
                        "driver's target, not the one make_ir is given")
        return self._ttir()

    def test_max_reduces_in_fp16(self, ttir):
        # The combiner, not the reduce's result type: a widened reduce would
        # still be stored back as f16 by a truncf, so f16 in the tt.reduce body
        # is what says the widening did not happen.
        assert "arith.maxnumf %a, %b : f16" in ttir

    def test_no_unlowerable_conversions_emitted(self, ttir):
        # The concrete reason the widening cannot stand on Spyre. Both spellings,
        # because the promotion emits an extf going in and a truncf coming out.
        assert "arith.extf" not in ttir
        assert "arith.truncf" not in ttir

    def test_exp_traces_at_fp16(self, ttir):
        # Upstream this call raises ValueError during tracing, so reaching TTIR
        # at all is the assertion; the type is checked so a silent cast would
        # not pass either.
        assert "math.exp %" in ttir
        assert "tensor<64x64xf16>" in ttir.split("math.exp %")[1].split("\n")[0]
