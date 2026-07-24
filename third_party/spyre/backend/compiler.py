import hashlib

from triton.backends.compiler import BaseBackend, GPUTarget
from dataclasses import dataclass
from typing import Dict, Tuple
from types import ModuleType


@dataclass
class SpyreOptions:
    # Per-axis partition of the Spyre hardware grid. One entry per
    # tl.program_id axis the kernel reads; prod(grid) is the total
    # physical core count. Default covers the common 1D-on-32-cores
    # case. A 2D kernel with grid = (16, 2) would partition the same
    # 32 cores as 16x2 across axes x and y.
    grid: Tuple[int, ...] = (32,)
    lx_size: int = 2 * 1024 * 1024  # 2 MB scratchpad per core
    # HBM data layout mode: "logical" (default) derives physical strides from
    # logical strides via the coord map; "physical" uses row-major strides over
    # the physical shape.
    hbm_data_layout: str = "logical"
    # Required by Triton code generator
    sanitize_overflow: bool = False
    debug: bool = False
    allowed_dot_input_precisions: tuple = ("ieee",)

    def __post_init__(self):
        # Normalize list → tuple for hashability / dataclass equality.
        if isinstance(self.grid, list):
            self.grid = tuple(self.grid)

    def hash(self):
        key = "_".join(f"{name}-{val}" for name, val in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SpyreBackend(BaseBackend):
    """Spyre AI accelerator backend for Triton.

    Compiles Triton TTIR to KTIR (KTDP dialect IR) for the IBM Spyre accelerator.
    """

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "spyre"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "ktir"

    def hash(self) -> str:
        return f"spyre-{self.target.arch}"

    def parse_options(self, options: dict) -> SpyreOptions:
        return SpyreOptions(**{k: v for k, v in options.items()
                               if k in SpyreOptions.__dataclass_fields__})

    def add_stages(self, stages: dict, options: SpyreOptions, language=None) -> None:
        stages["ttir"] = lambda src, metadata: self._make_ttir(src, metadata, options)
        stages["ktir"] = lambda src, metadata: self._make_ktir(src, metadata, options)

    def load_dialects(self, context) -> None:
        from triton._C.libtriton import spyre
        spyre.load_dialects(context)

    def get_codegen_implementation(self, options):
        """Return codegen hooks queried by the Triton frontend.

        Called from ``python/triton/compiler/compiler.py::compile_to_ttir``
        (line 304). The returned dict is threaded through to the
        language builder and consumed by individual frontend helpers;
        today only ``min_dot_size`` is required, enforced around the
        ``tl.dot`` shape check in
        ``python/triton/language/semantic.py`` (lines 1453-1458).

        ``min_dot_size(lhs_type, rhs_type)`` returns ``(min_M, min_N,
        min_K)`` lower bounds that a ``tl.dot`` call's operand shapes
        must satisfy. Upstream backends pick:

          - NVIDIA (``third_party/nvidia/backend/compiler.py:19``) —
            ``(1, 1, 16)`` for fp16/bf16, ``(1, 1, 32)`` for int8/fp8.
            Only K is constrained; small M/N are padded into tensor
            cores.
          - AMD (``third_party/amd/backend/compiler.py:16``) —
            ``(1, 1, 1)``, falling back to FMA for configurations not
            natively supported by its matrix cores.
          - Interpreter (``python/triton/runtime/interpreter.py:290``) —
            ``(1, 1, 1)``.

        Spyre has no ``tl.dot`` shape floor today (``linalg.matmul``
        handles arbitrary tile sizes), so we return ``(1, 1, 1)``
        matching AMD and the interpreter. Revisit this if a future
        KTIR matmul path needs a minimum.
        """
        return {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def pack_metadata(self, metadata):
        return ()

    def _make_ttir(self, mod, metadata, options):
        """Run standard Triton TTIR optimization passes."""
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")

        metadata["stage"] = "ttir"
        return mod

    def _make_ktir(self, mod, metadata, options):
        """Lower optimized TTIR to KTIR using C++ MLIR passes.

        Pipeline (add_convert_ttir_to_ktdp expands to steps 1-4):
        1. LowerDescriptorMemory: tt.descriptor_load/store/gather/scatter -> ktdp.*
        2. LowerScalarLoad: scalar tt.load (+ addptr chain) -> ktdp.* rank-0 read
        3. LowerComputeOps: tt.reduce/broadcast/expand_dims -> linalg/tensor + dead op sweep
        4. ConvertFunctions: tt.func/return -> func.func/return, !tt.ptr -> index
           (must run last — memory passes consume !tt.ptr args via getBasePtrAsIndex)
        5. DistributeWork: tt.get_program_id -> ktdp.get_compute_tile_id
        6. canonicalize + CSE
        """
        from triton._C.libtriton import ir, passes, spyre

        grid = list(options.grid)
        hw_layout = (options.hbm_data_layout == "physical")

        pm = ir.pass_manager(mod.context)
        spyre.passes.ttir_to_ktdp.add_convert_ttir_to_ktdp(pm, hw_layout)
        spyre.passes.ttir_to_ktdp.add_distribute_work(pm, grid)
        # Clean up redundant arithmetic (fold muli x,1; simplify cast chains)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ktir")

        metadata["name"] = mod.get_entry_func_name()
        metadata["stage"] = "ktir"
        return mod
