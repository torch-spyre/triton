"""The spyreop intrinsics a kernel may request with ``tl.spyre_op``, and the
fallback each one is traced as.

``tl.spyre_op(name, x)`` is a request that ``x`` be computed by the device
intrinsic ``name``. The kernel author writes only the name; what is traced is the
FALLBACK registered here -- the same computation in plain ``tl`` ops -- inside a
``tts.spyre_op`` region. So a request means the same thing at every stage:

* the ``ktir`` artifact holds the fallback, inlined and tagged ``tts.hint``, which
  any KTIR reader runs as written;
* the ``spyrecode`` stage replaces the fallback, whole, with the intrinsic.

The registry is the backend's and not the author's. A name is admitted only when
three things agree on it, and all three are in this tree: this table,
``LowerSpyreOps``' table of the intrinsics it builds, and the spyreop dialect's
operand constraints, which ``dtypes`` restates so that a dtype the intrinsic does
not take is refused at the kernel line rather than in ``spyrecode``.

Every fallback widens to fp32 and narrows back. ``tl.exp`` and ``tl.erf`` are
fp32-only in the frontend, and the casts cost nothing on the device path, since
``LowerSpyreOps`` replaces the body they are part of, casts included. At fp32
the casts are no-ops and are not traced.

The frontend reaches this table through the ``spyre_intrinsic`` codegen hook (see
``SpyreBackend.get_codegen_implementation``), the channel ``extra_math_dtypes``
already uses, so ``triton.language`` names no backend module.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    # The exact form, x * CDF(x), which is what spyreop.gelu is documented as.
    xf = x.to(tl.float32)
    return (0.5 * xf * (1.0 + tl.erf(xf * 0.7071067811865476))).to(x.dtype)


@triton.jit
def _silu(x):
    xf = x.to(tl.float32)
    return (xf / (1.0 + tl.exp(-xf))).to(x.dtype)


@triton.jit
def _sigmoid(x):
    xf = x.to(tl.float32)
    return (1.0 / (1.0 + tl.exp(-xf))).to(x.dtype)


@dataclass(frozen=True)
class Intrinsic:
    """One registered intrinsic.

    ``fallback`` is a ``@triton.jit`` function of ``arity`` tensors, traced at the
    operands' types. ``dtypes`` is what the intrinsic takes, by Triton dtype name.
    ``result_types`` maps the operand types to the declared result types: the op is
    created before its body is traced, so the result types cannot be read off the
    body, and the frontend refuses a fallback that returns anything else.
    """

    fallback: Callable
    arity: int
    dtypes: Tuple[str, ...]
    result_types: Callable

    @staticmethod
    def elementwise(fallback, dtypes):
        """A unary intrinsic whose result has its operand's type."""
        return Intrinsic(fallback, 1, dtypes, lambda arg_types: [arg_types[0]])


#: Name -> intrinsic. The names are spyreop's op mnemonics.
INTRINSICS = {
    "gelu": Intrinsic.elementwise(_gelu, ("fp16", )),
    "silu": Intrinsic.elementwise(_silu, ("fp16", "fp32")),
    "sigmoid": Intrinsic.elementwise(_sigmoid, ("fp16", "fp32")),
}


def lookup(name):
    """The intrinsic registered as ``name``. Raises ``ValueError`` naming the
    registered ones when there is none."""
    entry = INTRINSICS.get(name)
    if entry is None:
        raise ValueError(f"tl.spyre_op: no intrinsic named {name!r}; the Spyre backend "
                         f"registers {', '.join(sorted(INTRINSICS))}")
    return entry
