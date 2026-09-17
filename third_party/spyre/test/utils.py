"""Shared utilities for Spyre KTIR tests.

Hoisted out of ``conftest.py`` so that standalone scripts (``../scripts/gen_ttir.py``,
ad-hoc notebooks) can import them without pulling in pytest fixtures.

Contents
--------
- :func:`compile_to_ttir`      — ``@triton.jit`` → TTIR text
- :func:`make_ktir_mod`         — TTIR → KTIR pipeline, returns live ir.module
- :func:`np_dtype`              — SIGNATURE type string → NumPy dtype
- :func:`sticksize`             — Spyre stick size (elements per stick) for an arg
"""

import numpy as np


# ---------------------------------------------------------------------------
# Spyre layout helpers — dtype / stick-size derivation from a SIGNATURE
# ---------------------------------------------------------------------------

#: Triton element-type spelling (as it appears in a fixture ``SIGNATURE``,
#: with any leading ``*`` pointer marker stripped) → NumPy dtype.
DTYPE_MAP = {
    "fp32": np.float32,
    "fp16": np.float16,
    "i32":  np.int32,
}

#: Spyre stick width in bytes. A "stick" is the hardware's contiguous
#: innermost memory unit, so the number of *elements* per stick depends on the
#: element size: 32 for fp32, 64 for fp16.
STICK_BYTES = 128


def spyre_target():
    """The Spyre compile target, built on demand.

    A function rather than a module constant because this module imports
    ``GPUTarget`` inside its functions, not at the top -- so a constant here would
    need a module-level triton import that the rest of the file avoids.

    ``warp_size=1`` is load-bearing: CompiledKernel compares
    ``num_warps * warp_size`` against ``n_max_threads``, and Spyre has no warps.
    """
    from triton.backends.compiler import GPUTarget
    return GPUTarget(backend="spyre", arch=1, warp_size=1)


def np_dtype(signature, key):
    """Map a ``SIGNATURE`` entry to its NumPy dtype.

    ``signature`` is a fixture's arg-name → Triton-type-string dict (e.g.
    ``{"a_ptr": "*fp16"}``); ``key`` is the arg name. A leading ``*``
    (pointer) marker is ignored, so ``"*fp16"`` and ``"fp16"`` both map to
    ``np.float16``.

    Raises ``KeyError`` if the element type is not in :data:`DTYPE_MAP`.
    """
    triton_type = signature[key].lstrip("*")
    if triton_type not in DTYPE_MAP:
        raise KeyError(
            f"{key!r} has element type {triton_type!r}, which has no NumPy "
            f"mapping; known types: {sorted(DTYPE_MAP)}"
        )
    return DTYPE_MAP[triton_type]


def sticksize(signature, key):
    """Number of elements in one Spyre stick for the arg named ``key``.

    Derived as :data:`STICK_BYTES` / element-size, so it tracks the dtype
    declared in ``signature`` rather than being hardcoded per fixture:
    fp32 → 32, fp16 → 64.

    Fixtures typically bind the signature once and reuse the result::

        _SS = functools.partial(sticksize, _SIG_SPYRE)
        ...
        "A_LAYOUT": [[(1, "floordiv", _SS("a_ptr")), 0, (1, "mod", _SS("a_ptr"))]]
    """
    return STICK_BYTES // np.dtype(np_dtype(signature, key)).itemsize


# ---------------------------------------------------------------------------
# compile_to_ttir — Triton kernel → TTIR text
# ---------------------------------------------------------------------------

def compile_to_ttir(kernel_fn, signature, constexprs):
    """Compile a ``@triton.jit`` function to TTIR text.

    Parameters
    ----------
    kernel_fn  : a ``@triton.jit`` decorated function (``triton.JITFunction``)
    signature  : dict mapping arg names to type strings (e.g. ``"*fp32"``)
    constexprs : dict mapping constexpr names to values
    """
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from backend.compiler import SpyreBackend

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    src = ASTSource(fn=kernel_fn, signature=signature, constexprs=constexprs)

    backend = SpyreBackend(target)
    options = backend.parse_options({})

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = (backend.get_codegen_implementation(options)
                   if hasattr(backend, "get_codegen_implementation") else {})
    module_map = backend.get_module_map()

    mod = src.make_ir(target, options, codegen_fns, module_map, context)
    return str(mod)


# ---------------------------------------------------------------------------
# make_ktir_mod — full TTIR → KTIR pipeline, returns live ir.module
# ---------------------------------------------------------------------------

def make_ktir_mod(ttir_path, *, grid=None, metadata=None, **options):
    """Parse *ttir_path*, run TTIR and KTIR passes, return the live module.

    *metadata* is an out-parameter, and the only way to observe what the stages
    recorded: pass a dict and it comes back holding ``name``, ``stage`` and (in the
    address-binding mode) ``base_addresses``. Omitted, a throwaway is used, which is
    what every caller wanting only the IR wants. Not a return value, because that
    would change the shape for all eight existing call sites to serve one.

    Every keyword is forwarded verbatim as a ``SpyreOptions`` field, so a caller
    reaches the whole option surface without this helper growing a parameter per
    option. See ``SpyreOptions`` for the fields and what they mean; they are
    deliberately not restated here, since a second copy of that list would drift
    as options are added.

    ``grid`` is the one named parameter, and only because ``None`` has to mean
    "leave the default alone": callers pass ``grid=grid`` with a value that may
    be ``None``, and forwarding that would override the dataclass default with
    ``None`` instead of falling back to it. No coercion happens here —
    ``SpyreOptions.__post_init__`` normalizes list → tuple for the fields that
    need it, so doing it for ``grid`` alone would just be an inconsistency.

    Unknown keys raise instead of being dropped. ``SpyreBackend.parse_options``
    silently filters anything it doesn't recognize, so a typo'd option would
    otherwise become an invisible no-op — the same failure mode ``_make_ktir``
    guards against by raising on a missing pass binding.
    """
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from backend.compiler import SpyreBackend, SpyreOptions

    unknown = sorted(k for k in options if k not in SpyreOptions.__dataclass_fields__)
    if unknown:
        raise ValueError(
            f"unknown SpyreOptions field(s): {', '.join(unknown)}; "
            f"valid fields are {', '.join(sorted(SpyreOptions.__dataclass_fields__))}"
        )

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    backend = SpyreBackend(target)
    opts = dict(options)
    if grid is not None:
        opts["grid"] = grid
    options = backend.parse_options(opts)

    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)

    mod = ir.parse_mlir_module(str(ttir_path), ctx)
    mod.context = ctx

    metadata = {} if metadata is None else metadata
    mod = backend._make_ttir(mod, metadata, options)
    return backend._make_ktir(mod, metadata, options)
