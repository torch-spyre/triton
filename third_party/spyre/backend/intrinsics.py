"""The spyreop intrinsics a kernel may request with ``tl.spyre_op``, and the
fallback each one is traced as.

``tl.spyre_op(name, x)`` is a request that ``x`` be computed by the device
intrinsic ``name``. The kernel author writes only the name; what is traced is the
FALLBACK registered here -- the same computation in plain ``tl`` ops -- inside a
``tts.spyre_op`` region. So a request means the same thing at every stage:

* the ``ktir`` artifact holds the fallback, inlined and hinted ``tts.spyreop_hint``, which
  any KTIR reader runs as written;
* the ``spyrecode`` stage replaces the fallback, whole, with the intrinsic.

The registry is the backend's and not the author's. What each intrinsic takes
-- its operand count, the dtypes of its operands and its result rule -- is the
C++ intrinsic table's (``Dialect/TTS/IR/Intrinsics.h``), read here through the
``spyre.intrinsics`` pybind module; the same table is what the ``tts.spyre_op``
verifier and ``LowerSpyreOps`` read. This module holds only the fallbacks, keyed
by name, and checks at import that their names are the table's.

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
    """One registered intrinsic: its fallback, and what the C++ table says it takes.

    ``fallback`` is a ``@triton.jit`` function of ``arity`` tensors, traced at the
    operands' types. ``dtypes`` is what every operand may be, by Triton dtype name.
    ``result_types`` maps the operand types to the declared result types: the op is
    created before its body is traced, so the result types cannot be read off the
    body, and the frontend refuses a fallback that returns anything else.
    """

    fallback: Callable
    arity: int
    dtypes: Tuple[str, ...]
    result_types: Callable


def _by_name(*pairs):
    """``{name: fallback}`` from ``(name, fallback)`` pairs, refusing a name given
    twice."""
    fallbacks = {}
    for name, fallback in pairs:
        if name in fallbacks:
            raise ValueError(f"tl.spyre_op: intrinsic {name!r} has two fallbacks")
        fallbacks[name] = fallback
    return fallbacks


#: Name -> fallback. The names are the C++ table's, which are spyreop's op
#: mnemonics.
FALLBACKS = _by_name(
    ("gelu", _gelu),
    ("silu", _silu),
    ("sigmoid", _sigmoid),
)


def _read_table():
    """The C++ intrinsic table, as ``{name: entry}``."""
    from triton._C.libtriton import spyre
    entries = spyre.intrinsics.table()
    table = {entry["name"]: entry for entry in entries}
    if len(table) != len(entries):
        raise RuntimeError("tl.spyre_op: the C++ intrinsic table names an intrinsic twice")
    return table


def _check_names(fallbacks, table):
    """Raise unless ``fallbacks`` has exactly the names of ``table``."""
    if set(fallbacks) != set(table):
        raise RuntimeError(f"tl.spyre_op: the fallbacks name {sorted(fallbacks)}, but the C++ "
                           f"intrinsic table names {sorted(table)}")


_TABLE = _read_table()
_check_names(FALLBACKS, _TABLE)


def lookup(name):
    """The intrinsic registered as ``name``. Raises ``ValueError`` naming the
    registered ones when there is none."""
    entry = _TABLE.get(name)
    if entry is None:
        raise ValueError(f"tl.spyre_op: no intrinsic named {name!r}; the Spyre backend "
                         f"registers {', '.join(sorted(FALLBACKS))}")
    result_operands = list(entry["result_operands"])
    return Intrinsic(
        fallback=FALLBACKS[name],
        arity=entry["num_operands"],
        dtypes=tuple(entry["dtypes"]),
        result_types=lambda arg_types: [arg_types[i] for i in result_operands],
    )
