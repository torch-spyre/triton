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
verifier and ``LowerSpyreOps`` read. This module holds only the fallbacks, each
registered under its name with ``@spyre_intrinsic(name)``, and checks at import
that every name in the table has exactly one. The decorator is the backend's: it
is not exported through ``triton.language``, so kernel authors cannot register
names. The table's reserved entry, ``test_mock``, has no fallback here: a test
registers its own through the same decorator, and a kernel naming it otherwise
is refused for want of one.

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


def _read_table():
    """The C++ intrinsic table, as ``{name: entry}``."""
    from triton._C.libtriton import spyre
    entries = spyre.intrinsics.table()
    table = {entry["name"]: entry for entry in entries}
    if len(table) != len(entries):
        raise RuntimeError("tl.spyre_op: the C++ intrinsic table names an intrinsic twice")
    return table


_TABLE = _read_table()

#: The table's entry reserved for tests, which register its fallback.
TEST_MOCK = "test_mock"

#: Name -> fallback, filled by ``spyre_intrinsic``.
FALLBACKS = {}


def spyre_intrinsic(name):
    """Register the decorated ``@triton.jit`` function as the fallback for the
    intrinsic ``name``. Raises on a name the C++ table does not define, and on a
    name that already has a fallback."""

    def register(fallback):
        if name not in _TABLE:
            raise ValueError(f"tl.spyre_op: {name!r} is not in the C++ intrinsic table, "
                             f"which names {', '.join(sorted(_TABLE))}")
        if name in FALLBACKS:
            raise ValueError(f"tl.spyre_op: intrinsic {name!r} has two fallbacks")
        FALLBACKS[name] = fallback
        return fallback

    return register


@spyre_intrinsic("gelu")
@triton.jit
def gelu(x):
    # The exact form, x * CDF(x), which is what spyreop.gelu is documented as.
    xf = x.to(tl.float32)
    return (0.5 * xf * (1.0 + tl.erf(xf * 0.7071067811865476))).to(x.dtype)


@spyre_intrinsic("silu")
@triton.jit
def silu(x):
    xf = x.to(tl.float32)
    return (xf / (1.0 + tl.exp(-xf))).to(x.dtype)


@spyre_intrinsic("sigmoid")
@triton.jit
def sigmoid(x):
    xf = x.to(tl.float32)
    return (1.0 / (1.0 + tl.exp(-xf))).to(x.dtype)


def _check_names(fallbacks, table):
    """Raise unless every name in ``table`` but ``TEST_MOCK`` has a fallback in
    ``fallbacks``, and ``fallbacks`` has no other. ``spyre_intrinsic`` has already
    refused a second fallback for one name."""
    wanted = set(table) - {TEST_MOCK}
    if set(fallbacks) != wanted:
        raise RuntimeError(f"tl.spyre_op: the fallbacks name {sorted(fallbacks)}, but the C++ "
                           f"intrinsic table names {sorted(wanted)}")


_check_names(FALLBACKS, _TABLE)


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


def lookup(name):
    """The intrinsic registered as ``name``. Raises ``ValueError`` naming the
    registered ones when there is none."""
    entry = _TABLE.get(name)
    if entry is None:
        raise ValueError(f"tl.spyre_op: no intrinsic named {name!r}; the Spyre backend "
                         f"registers {', '.join(sorted(FALLBACKS))}")
    if name not in FALLBACKS:
        raise ValueError(f"tl.spyre_op: no fallback is registered for {name!r}")
    result_operands = list(entry["result_operands"])
    return Intrinsic(
        fallback=FALLBACKS[name],
        arity=entry["num_operands"],
        dtypes=tuple(entry["dtypes"]),
        result_types=lambda arg_types: [arg_types[i] for i in result_operands],
    )
