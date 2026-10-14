"""The device ops a kernel may request with ``tl.spyre_intrinsic``, and the fallback each
one is traced as.

``tl.spyre_intrinsic(target, *args)`` is a request that ``args`` be computed by the
device op ``target``, named in full -- ``"spyreop.sigmoid"`` -- which takes them
in that order. The kernel author writes only
the target; what is traced is the FALLBACK registered here -- the same
computation in plain ``tl`` ops -- inside a ``tts.spyre_intrinsic`` region. So a request
means the same thing at every stage:

* the ``ktir`` artifact holds the fallback, inlined and hinted with the
  ``tts.spyre_intrinsic`` attribute, which any KTIR reader runs as written;
* the ``spyrecode`` stage replaces the fallback, whole, with ``target``.

This module is the one registry. Each entry is registered with
``@register_intrinsic(target, dtypes=...)`` on its fallback, and says everything the
frontend checks when the call is traced: the operand count (the fallback's) and
the dtypes every operand may have. The result is operand 0's type, for every
target. The IR carries only the target's name, so the compiler needs no copy of
this table: ``LowerSpyreIntrinsicCalls`` builds the op by that name, and the op's own
verifier checks the operands. The one thing checked against the compiler here is
that the name is a registered op, at registration, so a misspelled target is
refused where it is written rather than at its first trace.

The decorator is the backend's: it is not exported through ``triton.language``,
so kernel authors cannot register targets.

Every fallback widens to fp32 and narrows back. ``tl.exp`` and ``tl.erf`` are
fp32-only in the frontend, and the casts cost nothing on the device path, since
``LowerSpyreIntrinsicCalls`` replaces the body they are part of, casts included. At fp32
the casts are no-ops and are not traced.

The frontend reaches this registry through the ``spyre_intrinsic_lookup`` codegen hook
(see ``SpyreBackend.get_codegen_implementation``), the channel
``extra_math_dtypes`` already uses, so ``triton.language`` names no backend
module.
"""

import functools
from dataclasses import dataclass
from typing import Callable, Tuple

import triton
import triton.language as tl


def target_name(target):
    """``target`` as an op name: a string as given, or the ``OPERATION_NAME`` of
    an MLIR Python op class such as ``mlir_ktdp.dialects.spyreop.Sigmoid``."""
    if isinstance(target, str):
        return target
    name = getattr(target, "OPERATION_NAME", None)
    if isinstance(name, str):
        return name
    raise TypeError(f"tl.spyre_intrinsic: a target is an op name or an op class, got {target!r}")


@functools.cache
def _context():
    """A context set up as a compile's is -- Triton's dialects and the Spyre
    backend's in its registry -- made once and kept, since it is only read."""
    from triton._C.libtriton import ir, spyre
    context = ir.context()
    ir.load_dialects(context)
    spyre.load_dialects(context)
    return context


def is_registered_op(name):
    """Whether ``name`` is an op the compiler registers: the answer
    ``tts.spyre_intrinsic``'s region verifier gives for the same name, asked of a
    context set up as a compile's is."""
    from triton._C.libtriton import spyre
    return spyre.ir_utils.is_registered_op(_context(), name)


@dataclass(frozen=True)
class Intrinsic:
    """One registered target: the op, its fallback, and what it takes.

    ``fallback`` is a ``@triton.jit`` function of ``arity`` tensors, traced at the
    operands' types. ``dtypes`` is what every operand may be. The one result has
    operand 0's type: the op is created before its body is traced, so its result
    type cannot be read off the body, and the frontend refuses a fallback that
    returns anything else.
    """

    target: str
    fallback: Callable
    arity: int
    dtypes: Tuple[tl.dtype, ...]


#: Target name -> Intrinsic, filled by ``register_intrinsic``.
REGISTRY = {}


def register_intrinsic(target, *, dtypes):
    """Register the decorated ``@triton.jit`` function as the fallback for the op
    ``target``, taking operands of ``dtypes`` and returning operand 0's type.
    Raises on a target that is not a registered op, and on one that already has
    a fallback."""
    name = target_name(target)
    if not is_registered_op(name):
        raise ValueError(f"tl.spyre_intrinsic: target {name!r} is not a registered op; a "
                         f"target is named in full, e.g. 'spyreop.sigmoid'")
    dtypes = tuple(dtypes)
    for dtype in dtypes:
        if not isinstance(dtype, tl.dtype):
            raise TypeError(f"tl.spyre_intrinsic: {name!r} takes dtypes as tl dtypes, got {dtype!r}")

    def register(fallback):
        if name in REGISTRY:
            raise ValueError(f"tl.spyre_intrinsic: target {name!r} has two fallbacks")
        if not fallback.arg_names:
            raise ValueError(f"tl.spyre_intrinsic: the fallback for {name!r} takes no operand, "
                             f"and the result is operand 0's type")
        REGISTRY[name] = Intrinsic(
            target=name,
            fallback=fallback,
            arity=len(fallback.arg_names),
            dtypes=dtypes,
        )
        return fallback

    return register


def lookup(target):
    """The registration for ``target``. Raises ``ValueError`` naming the
    registered targets when there is none."""
    name = target_name(target)
    entry = REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"tl.spyre_intrinsic: no fallback is registered for {name!r}; the Spyre "
                         f"backend registers {', '.join(sorted(REGISTRY)) or 'none'}")
    return entry


@register_intrinsic("spyreop.gelu", dtypes=(tl.float16, ))
@triton.jit
def gelu(x):
    # The exact form, x * CDF(x), which is what spyreop.gelu is documented as.
    xf = x.to(tl.float32)
    return (0.5 * xf * (1.0 + tl.erf(xf * 0.7071067811865476))).to(x.dtype)


@register_intrinsic("spyreop.silu", dtypes=(tl.float16, tl.float32))
@triton.jit
def silu(x):
    xf = x.to(tl.float32)
    return (xf / (1.0 + tl.exp(-xf))).to(x.dtype)


@register_intrinsic("spyreop.sigmoid", dtypes=(tl.float16, tl.float32))
@triton.jit
def sigmoid(x):
    xf = x.to(tl.float32)
    return (1.0 / (1.0 + tl.exp(-xf))).to(x.dtype)
