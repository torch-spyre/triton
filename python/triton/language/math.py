from . import core
from functools import wraps
from typing import List
# --- added for spyre
from . import target_info

T = core.TypeVar('T')

# --- START --- added for spyre
#: Narrower floats the Spyre backend admits into any math op that declares fp32.
#:
#: Spyre needs fp16 for these, and it is not a convenience. A reduce's statistic
#: is stored replicated across a stick and read back one lane wide then splatted,
#: and that splat is an fp16-only path in the backend — so a chain that shifts a
#: tile by a row statistic and then exponentiates it (softmax's numerator) has to
#: be fp16 end to end. The alternative is not a less precise softmax, it is no
#: softmax: at fp32 the read-back fails instruction selection.
#:
#: The precision trade is deliberate. fp16 exp/log over a shifted tile is less
#: accurate than fp32, and the surrounding chain accumulates in fp16 anyway
#: because there is no wider datapath for the statistic round-trip; refusing fp16
#: here would not buy accuracy, it would only move the failure earlier. The
#: fixture tolerances under ``third_party/spyre/test/fixtures`` record the
#: measured cost.
#:
#: fp16 only, and bf16 deliberately not: nothing in this tree has run bf16
#: through these ops, and ``standard.py``'s ``_promote_bfloat16_to_float32``
#: still widens bf16 reduces, so admitting it here would claim support the rest
#: of the frontend does not.
#:
#: Ops Spyre cannot lower at all are *not* this decorator's business to refuse —
#: ``LowerSpyreOps`` converts sqrt/rsqrt/exp and the pipeline rejects the rest
#: with a lowering diagnostic naming the op, which is a better error than a
#: dtype table's.
_SPYRE_EXTRA_FLOAT_DTYPES = ("fp16", )
# --- END --- added for spyre


def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """

    def wrapper(fn):

        @wraps(fn)
        def check(*args, **kwargs):
            # --- START --- added for spyre
            # Resolved per call, not per decoration: there is no active driver
            # when this module is imported. ``is_spyre()`` is a
            # ``constexpr_function``, but this wrapper runs as ordinary Python at
            # trace time — outside the builtin, so no ``_semantic`` is threaded
            # through it — and called that way it returns a plain bool.
            #
            # Gated on fp32 being declared so this only ever relaxes a *float*
            # op: ``umulhi``'s integer list is left exactly as upstream wrote it.
            accepted = list(dtypes)
            if "fp32" in dtypes and target_info.is_spyre():
                accepted += [d for d in _SPYRE_EXTRA_FLOAT_DTYPES if d not in accepted]
            # --- END --- added for spyre
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, core.tensor)]:
                # --- added for spyre: ``accepted``, upstream reads ``dtypes``
                if arg.type.scalar.name not in accepted:
                    raise ValueError(f"Expected dtype {accepted} but got {arg.type.scalar.name}")
            return fn(*args, **kwargs)

        return check

    return wrapper


def _add_math_1arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`.

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_2arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x` and :code:`y`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


def _add_math_3arg_docstr(name: str) -> core.Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`, :code:`y`, and :code:`z`.

    :param x: the input values
    :type x: Block
    :param y: the input values
    :type y: Block
    :param z: the input values
    :type z: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@core.builtin
@_check_dtype(dtypes=["int32", "int64", "uint32", "uint64"])
@_add_math_2arg_docstr("most significant N bits of the 2N-bit product")
def umulhi(x, y, _semantic=None):
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    return core.tensor(_semantic.builder.create_umulhi(x.handle, y.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential")
@core._tensor_member_fn
def exp(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_exp(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("exponential (base 2)")
@core._tensor_member_fn
def exp2(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_exp2(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("natural logarithm")
@core._tensor_member_fn
def log(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_log(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("logarithm (base 2)")
@core._tensor_member_fn
def log2(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_log2(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("cosine")
@core._tensor_member_fn
def cos(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_cos(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("sine")
@core._tensor_member_fn
def sin(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_sin(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("fast square root")
@core._tensor_member_fn
def sqrt(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_sqrt(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_1arg_docstr("precise square root (rounding to nearest wrt the IEEE standard)")
@core._tensor_member_fn
def sqrt_rn(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_precise_sqrt(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("inverse square root")
@core._tensor_member_fn
def rsqrt(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_rsqrt(x.handle), x.type)


@core._tensor_member_fn
@core.builtin
@_add_math_1arg_docstr("absolute value")
def abs(x, _semantic=None):
    x = _semantic.to_tensor(x)
    dtype = x.dtype
    if dtype.is_fp8e4b15():
        mask = core.full(x.shape, 0x7F, core.int8, _semantic=_semantic)
        return core.tensor(_semantic.builder.create_and(x.handle, mask.handle), x.type)
    elif dtype.is_floating():
        return core.tensor(_semantic.builder.create_fabs(x.handle), x.type)
    elif dtype.is_int_signed():
        return core.tensor(_semantic.builder.create_iabs(x.handle), x.type)
    elif dtype.is_int_unsigned():
        return x  # no-op
    else:
        assert False, f"Unexpected dtype {dtype}"


@core.builtin
@_add_math_2arg_docstr("fast division")
def fdiv(x, y, ieee_rounding=False, _semantic=None):
    ieee_rounding = core._unwrap_if_constexpr(ieee_rounding)
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    return _semantic.fdiv(x, y, ieee_rounding)


@core.builtin
@_check_dtype(dtypes=["fp32"])
@_add_math_2arg_docstr("precise division (rounding to nearest wrt the IEEE standard)")
def div_rn(x, y, _semantic=None):
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    return core.tensor(_semantic.builder.create_precise_divf(x.handle, y.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("error function")
@core._tensor_member_fn
def erf(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_erf(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("floor")
@core._tensor_member_fn
def floor(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_floor(x.handle), x.type)


@core.builtin
@_check_dtype(dtypes=["fp32", "fp64"])
@_add_math_1arg_docstr("ceil")
@core._tensor_member_fn
def ceil(x, _semantic=None):
    x = _semantic.to_tensor(x)
    return core.tensor(_semantic.builder.create_ceil(x.handle), x.type)


@core.builtin
@_add_math_3arg_docstr("fused multiply-add")
def fma(x, y, z, _semantic=None):
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    z = _semantic.to_tensor(z)
    x, y = core.binary_op_type_legalization(x, y, _semantic)
    z, x = core.binary_op_type_legalization(z, x, _semantic)
    z, y = core.binary_op_type_legalization(z, y, _semantic)
    return core.tensor(_semantic.builder.create_fma(x.handle, y.handle, z.handle), x.type)
