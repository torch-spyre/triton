import functools

from triton.runtime import driver
from triton.runtime.jit import constexpr_function

__all__ = ["current_target", "is_spyre", "requires_backend"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


# --- START --- added for spyre
@constexpr_function
def is_spyre():
    target = current_target()
    return target is not None and target.backend == "spyre"


def requires_backend(name):
    """Decorate a frontend op as ``name``-backend-only.

    The wrapped function raises ``ValueError`` if invoked under any other
    backend. The backend is resolved at *call* time (``current_target()``),
    not at decoration time — there is no active driver when the class body is
    evaluated. Use this for ops that only exist for one backend (e.g.
    ``tl.spyre_tensor_layout``); for behavior that merely *diverges* by
    backend, branch on the :func:`is_spyre`-style predicate instead.
    """

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            target = current_target()
            backend = target.backend if target is not None else None
            if backend != name:
                raise ValueError(f"{fn.__name__} is only supported on the "
                                 f"'{name}' backend, not '{backend}'")
            return fn(*args, **kwargs)

        return wrapper

    return decorator
# --- END --- added for spyre


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"


@constexpr_function
def is_hip_gfx1250():
    target = current_target()
    return target is not None and target.arch == "gfx1250"
