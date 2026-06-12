from ._runtime import GluonJITFunction, constexpr_function, jit
from triton import aggregate, must_use_result
# --- START --- added for spyre
# GPU-only arch shims; absent from a spyre-only wheel. Guard so import succeeds.
try:
    from . import nvidia
except ImportError:
    nvidia = None
try:
    from . import amd
except ImportError:
    amd = None
# --- END --- added for spyre

__all__ = ["aggregate", "amd", "constexpr_function", "GluonJITFunction", "jit", "must_use_result", "nvidia"]
