#!/usr/bin/env python3
"""Unit tests for the reusable Spyre frontend backend guard, and the one
upstream dtype refusal forked behind it.

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

The second half covers a different mechanism, which is why it does not use those
helpers: ``language/math.py``'s ``_check_dtype`` admits the extra dtypes a target
declares through the ``extra_math_dtypes`` codegen hook, alongside
``min_dot_size``. The table of which ops qualify lives in the backend, so these
tests drive the hook rather than a backend predicate.

The last two sections are the argument checking of two Spyre-only ops,
``tl.spyre_pin`` and ``tl.make_distributed_descriptor``.
It belongs with the guards rather than with the op's own tests because what it
measures is which refusals the FRONTEND owns: a misspelled memory space and an
address of the wrong Python type are reported at the kernel line, while the
address's admissible *shape* is the op verifier's, over IR the frontend has
already built. Driven through real tracing, since the backend guard resolves the
target first and so a monkeypatched one never reaches the checks under test.

The same division holds for the compose: a ragged table or an ``axes`` of the wrong
length is the frontend's, while whether the table has one entry per tile is the
lowering's, since only a pass sees the grid.
"""

import pytest
from triton.backends.compiler import GPUTarget
from triton.language import core, math, target_info


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
# math.py's _check_dtype — the extra_math_dtypes hook
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


class _FakeSemantic:
    """The one thing ``_check_dtype`` reads off ``_semantic``: ``builder.codegen_fns``."""

    def __init__(self, codegen_fns):
        self.builder = type("_B", (), {"codegen_fns": codegen_fns})()


def _spyre_codegen_fns():
    """The real hook, from the backend, so these tests cannot drift from it."""
    from backend.compiler import SpyreBackend

    backend = SpyreBackend(_target("spyre"))
    return backend.get_codegen_implementation(backend.parse_options({}))


def _standin(name, dtypes=("fp32", "fp64")):
    """A ``_check_dtype``-decorated callable standing in for the math op ``name``.

    Decorated here rather than calling ``tl.exp`` so the test measures the
    decorator and not a builtin's body. The name is set explicitly because the
    hook is keyed on it, so it is part of what is under test.
    """

    def op(*args, **kwargs):
        return "ran"

    op.__name__ = name
    return math._check_dtype(dtypes=list(dtypes))(op)


def _call(op, dtype, codegen_fns):
    return op(_FakeTensor(dtype), _semantic=_FakeSemantic(codegen_fns))


#: The ops the hook admits fp16 into, and ops that declare fp32 but are not in it.
_LOWERABLE = ("exp", "sqrt", "rsqrt")
_NOT_LOWERABLE = ("log", "sin", "erf", "exp2", "floor")


@pytest.fixture(scope="module")
def spyre_fns():
    return _spyre_codegen_fns()


class TestExtraMathDtypes:

    def test_the_backend_supplies_the_hook(self, spyre_fns):
        assert callable(spyre_fns.get("extra_math_dtypes"))

    def test_the_admitted_set_is_the_ops_lowerspyreops_converts(self, spyre_fns):
        # LowerSpyreOps converts math.exp, sqrt and rsqrt at f16 and no other math
        # op, so a change to either side has to be a change to both.
        hook = spyre_fns["extra_math_dtypes"]
        assert {n for n in _LOWERABLE + _NOT_LOWERABLE if hook(n)} == set(_LOWERABLE)

    @pytest.mark.parametrize("name", _LOWERABLE)
    def test_fp16_admitted(self, spyre_fns, name):
        assert _call(_standin(name), core.float16, spyre_fns) == "ran"

    def test_no_hook_leaves_the_upstream_table(self):
        # Every other backend, and the interpreter, whose codegen_fns omit the key.
        with pytest.raises(ValueError, match="Expected dtype"):
            _call(_standin("exp"), core.float16, {"min_dot_size": None})

    @pytest.mark.parametrize("name", _NOT_LOWERABLE)
    def test_fp16_refused_for_an_op_with_no_f16_intrinsic(self, spyre_fns, name):
        # These declare fp32 exactly as the three above do, so only the name
        # separates them.
        with pytest.raises(ValueError, match="Expected dtype"):
            _call(_standin(name), core.float16, spyre_fns)

    def test_declared_dtypes_still_admitted(self, spyre_fns):
        # The hook only ever adds; fp32/fp64 must not be lost on the way.
        assert _call(_standin("exp"), core.float32, spyre_fns) == "ran"
        assert _call(_standin("exp"), core.float64, spyre_fns) == "ran"

    def test_bf16_not_admitted(self, spyre_fns):
        with pytest.raises(ValueError, match="Expected dtype"):
            _call(_standin("exp"), core.bfloat16, spyre_fns)

    def test_integer_op_untouched(self, spyre_fns):
        # A float addition must not leak into an op that never accepted floats.
        umulhi = _standin("umulhi", dtypes=("int32", "int64", "uint32", "uint64"))
        with pytest.raises(ValueError, match="Expected dtype"):
            _call(umulhi, core.float16, spyre_fns)
        assert _call(umulhi, core.int32, spyre_fns) == "ran"

    def test_error_reports_the_set_actually_accepted(self, spyre_fns):
        # The message has to name fp16 too, or it tells the reader to cast to a
        # type that was already allowed.
        with pytest.raises(ValueError) as exc:
            _call(_standin("exp"), core.bfloat16, spyre_fns)
        msg = str(exc.value)
        assert "fp16" in msg and "bf16" in msg

    def test_read_per_call_not_per_decoration(self, spyre_fns):
        # codegen_fns belongs to the compilation, so one callable has to answer
        # differently for two of them.
        exp = _standin("exp")
        with pytest.raises(ValueError):
            _call(exp, core.float16, {})
        assert _call(exp, core.float16, spyre_fns) == "ran"


# ---------------------------------------------------------------------------
# The hook, in the IR it actually produces
#
# The tests above pin the decorator against a stub; this traces for real. No
# active driver is needed: compile_to_ttir takes codegen_fns from the backend it
# constructs, which is the same route a compile uses.
# ---------------------------------------------------------------------------

class TestExtraMathDtypesInTracedIR:

    @staticmethod
    def _ttir():
        import triton
        import triton.language as tl
        from utils import compile_to_ttir

        @triton.jit
        def fp16_exp(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            """An fp16 ``tl.exp``. Descriptors rather than pointer arithmetic,
            matching the fixtures -- nothing here goes past TTIR, so no layout is
            needed."""
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            x = x_desc.load([0, 0])
            out_desc.store([0, 0], tl.exp(x))

        signature = {"x_ptr": "*fp16", "out_ptr": "*fp16",
                     "M": "constexpr", "N": "constexpr"}
        return compile_to_ttir(fp16_exp, signature, {"M": 64, "N": 64})

    @pytest.fixture(scope="class")
    def ttir(self):
        return self._ttir()

    def test_no_conversion_emitted(self, ttir):
        # The op stays at the width it was handed, rather than being widened
        # around a fp32-only intrinsic.
        assert "arith.extf" not in ttir
        assert "arith.truncf" not in ttir

    def test_exp_traces_at_fp16(self, ttir):
        # Upstream this call raises ValueError during tracing, so reaching TTIR
        # at all is the assertion; the type is checked so a silent cast would
        # not pass either.
        assert "math.exp %" in ttir
        assert "tensor<64x64xf16>" in ttir.split("math.exp %")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# tl.spyre_pin — the refusals the frontend owns, and the op it builds
# ---------------------------------------------------------------------------

class TestSpyrePin:

    SIGNATURE = {"x_ptr": "*fp16", "out_ptr": "*fp16",
                 "M": "constexpr", "N": "constexpr"}
    CONSTANTS = {"M": 64, "N": 64}

    @staticmethod
    def _trace(kernel):
        from utils import compile_to_ttir
        return compile_to_ttir(kernel, TestSpyrePin.SIGNATURE,
                               TestSpyrePin.CONSTANTS)

    @staticmethod
    def _raises(kernel, match):
        from triton.compiler.errors import CompilationError
        with pytest.raises(CompilationError) as exc:
            TestSpyrePin._trace(kernel)
        assert match in str(exc.value)

    def test_constant_address_reaches_the_ir(self):
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            e = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(e, "ct_local", address=4096)
            out_desc.store([0, 0], e)

        ttir = self._trace(k)
        # The int becomes a scalar constant, and the space is carried as the
        # string the kernel wrote — the ktdp attribute is built by the lowering.
        assert "tts.pin" in ttir
        assert 'memory_space = "ct_local"' in ttir
        assert "arith.constant 4096 : i32" in ttir

    def test_affine_address_traces_as_arithmetic(self):
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            pid = tl.program_id(0)
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            e = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(e, "ct_local", address=4096 + pid * 256)
            out_desc.store([0, 0], e)

        ttir = self._trace(k)
        # Nothing folds it: the address is an operand, and the shape restriction
        # is checked over this arithmetic by the op's verifier.
        assert "tt.get_program_id" in ttir
        assert "arith.muli" in ttir and "arith.addi" in ttir
        assert "tts.pin" in ttir

    def test_no_address_traces(self):
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            e = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(e, "ct_local")
            out_desc.store([0, 0], e)

        ttir = self._trace(k)
        # An absent optional operand, not a sentinel: the op prints no `address`
        # at all, which is what keeps "stated none" and "stated 0" apart.
        assert "tts.pin" in ttir
        assert "address" not in ttir.split("tts.pin")[1].split("\n")[0]

    def test_unknown_memory_space_is_refused_at_the_kernel_line(self):
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            e = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(e, "lx", address=4096)
            out_desc.store([0, 0], e)

        # Restated in the frontend so this is a traceback at the pin, rather than
        # the op verifier's failure after the whole function has been traced.
        self._raises(k, "memory_space must be 'ct_local'")

    def test_global_is_refused_and_says_why(self):
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            e = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(e, "global")
            out_desc.store([0, 0], e)

        # `global` is a real ktdp memory space and the design's prose names it, so
        # the message has to say a pin cannot place an HBM intermediate rather than
        # imply the spelling is wrong — otherwise the author goes looking for a
        # third name.
        self._raises(k, "'global' cannot be pinned")
        self._raises(k, "tl.make_tensor_descriptor")

    def test_non_integer_address_is_refused(self):
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            e = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(e, "ct_local", address="4096")
            out_desc.store([0, 0], e)

        self._raises(k, "address must be an int or an int32 scalar")


# ---------------------------------------------------------------------------
# tl.make_distributed_descriptor — the surface, and the refusals it owns
# ---------------------------------------------------------------------------

class TestMakeDistributedDescriptor(TestSpyrePin):
    """Reuses TestSpyrePin's tracing helpers: the two ops are checked the same way,
    through a real trace, and inheriting says so rather than copying them."""

    @staticmethod
    def _kernel(work_slices, axes, block_shape):
        """A relayout kernel, parameterised on the three things under test.

        The pin is not optional decoration: it is what gives the share an address,
        and the compose refuses a share without one. Its absence is the lowering's
        diagnostic rather than the frontend's, so it is tested in
        Conversion/TritonToKTIR/LowerInterTile/distributed-view-invalid.mlir.
        """
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, WS: tl.constexpr,
              AX: tl.constexpr, BS: tl.constexpr):
            pid = tl.program_id(0)
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            share = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(share, "ct_local", address=0)
            whole = tl.make_distributed_descriptor(share, WS, AX, BS)
            mine = whole.load([0, pid * N])
            out_desc.store([0, 0], mine)

        from utils import compile_to_ttir
        signature = {"x_ptr": "*fp16", "out_ptr": "*fp16", "M": "constexpr",
                     "N": "constexpr", "WS": "constexpr", "AX": "constexpr",
                     "BS": "constexpr"}
        constants = {"M": 64, "N": 32, "WS": work_slices, "AX": axes,
                     "BS": block_shape}
        return lambda: compile_to_ttir(k, signature, constants)

    @staticmethod
    def _refuses(run, match):
        from triton.compiler.errors import CompilationError
        with pytest.raises(CompilationError) as exc:
            run()
        assert match in str(exc.value)

    def test_it_traces(self):
        ttir = self._kernel([{"n": 0}, {"n": 1}], [None, "n"], [64, 32])()
        assert "tts.make_distributed_descriptor" in ttir
        # None becomes "" in the IR: an attribute array cannot hold a Python None,
        # and "" is what the op's verifier reads as "not divided on this dim".
        assert 'axes = ["", "n"]' in ttir
        assert "work_slices = [{n = 0 : i64}, {n = 1 : i64}]" in ttir
        # The result is a descriptor, so the read through it is the ordinary
        # tt.descriptor_load and needed no new surface.
        assert "tt.descriptor_load" in ttir

    def test_the_result_is_a_descriptor_of_the_block_shape(self):
        # A block smaller than the share: the descriptor follows block_shape, not
        # the share's extents, so the two being different is what proves which one
        # the type came from. Its own kernel, because the output descriptor has to
        # match what is stored and the shared kernel's is the share's shape.
        import triton
        import triton.language as tl
        from utils import compile_to_ttir

        # BM/BN as scalars rather than a list, because make_tensor_descriptor needs
        # a real Python list and a constexpr wrapping one has no len().
        @triton.jit
        def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, WS: tl.constexpr,
              AX: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
            x_desc = tl.make_tensor_descriptor(
                x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
            out_desc = tl.make_tensor_descriptor(
                out_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
            share = tl.exp(x_desc.load([0, 0]))
            tl.spyre_pin(share, "ct_local", address=0)
            whole = tl.make_distributed_descriptor(share, WS, AX, [BM, BN])
            out_desc.store([0, 0], whole.load([0, 0]))

        ttir = compile_to_ttir(
            k,
            {"x_ptr": "*fp16", "out_ptr": "*fp16", "M": "constexpr",
             "N": "constexpr", "WS": "constexpr", "AX": "constexpr",
             "BM": "constexpr", "BN": "constexpr"},
            {"M": 64, "N": 32, "WS": [{"n": 0}, {"n": 1}], "AX": [None, "n"],
             "BM": 64, "BN": 16})
        assert "tensordesc<64x16xf16>" in ttir

    def test_ragged_table_is_refused(self):
        self._refuses(self._kernel([{"n": 0}, {"m": 1}], [None, "n"], [64, 32]),
                      "work_slices[1] has keys ['m'], expected ['n']")

    def test_empty_table_is_refused(self):
        self._refuses(self._kernel([], [None, "n"], [64, 32]),
                      "work_slices must not be empty")

    def test_table_entry_must_be_a_dict(self):
        self._refuses(self._kernel([0], [None, "n"], [64, 32]),
                      "work_slices[0] must be a dict")

    def test_axes_length_is_the_rank(self):
        # Per tensor DIMENSION, not per key — so a one-entry axes for a rank-2
        # share is wrong even though there is one key.
        self._refuses(self._kernel([{"n": 0}, {"n": 1}], ["n"], [64, 32]),
                      "axes has 1 entries for a rank-2 share")

    def test_block_shape_length_is_the_rank(self):
        self._refuses(self._kernel([{"n": 0}, {"n": 1}], [None, "n"], [64]),
                      "block_shape has 1 entries for a rank-2 share")

    def test_axis_must_be_a_string_or_none(self):
        self._refuses(self._kernel([{"n": 0}, {"n": 1}], [None, 1], [64, 32]),
                      "axes[1] must be a string or None")

    def test_the_table_check_is_shared_with_inter_tile(self):
        # One checker, two callers, and the message names which one rejected the
        # table — so a third caller cannot quietly get different rules.
        from triton.language.semantic import TritonSemantic
        check = TritonSemantic._check_work_slices
        with pytest.raises(ValueError, match="tl.inter_tile: work_slices"):
            check([{"n": 0}, {"m": 0}], "tl.inter_tile")
        with pytest.raises(ValueError,
                           match="tl.make_distributed_descriptor: work_slices"):
            check([{"n": 0}, {"m": 0}], "tl.make_distributed_descriptor")
