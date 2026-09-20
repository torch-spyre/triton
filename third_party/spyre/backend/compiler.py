import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

from triton import knobs
from triton.backends.compiler import BaseBackend, GPUTarget
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from types import ModuleType

# ---------------------------------------------------------------------------
# The artifact layout, named once
#
# These names are torch-spyre's rather than ours, and nothing here can choose
# them: ``SpyreSDSCKernelRunner`` appends SPYRE_CODE_DIR to the directory it is
# given, and ``prepare_kernel`` then opens SPYRECODE_JSON inside it and the
# ``init_bin_file`` that names, both by name with no directory scan.
#
# Here rather than beside either use site, because there are two and they are in
# different modules: ``_make_spyrecode`` writes the layout and checks it, and
# ``SpyreUtils.load_binary`` (driver.py) reads it back to decide whether the
# unpacked directory is complete. Spelled twice, the pair drifts and the symptom
# is a launch that cannot find a file nobody misspelled on purpose.
#
# In the producer, so the dependency runs consumer → producer. Both modules are
# imported together anyway (``triton/backends/__init__.py`` imports
# ``<backend>.compiler`` and ``<backend>.driver`` for every discovered backend),
# so the import costs nothing that was not already paid.
# ---------------------------------------------------------------------------

#: Sub-directory of the export directory holding the loadable program.
SPYRE_CODE_DIR = "spyreCodeDir"

#: The job execution plan, inside SPYRE_CODE_DIR. The one file whose presence
#: means the artifact unpacked completely.
SPYRECODE_JSON = "spyrecode.json"

#: The initialization payload, named by SPYRECODE_JSON's ``init_bin_file`` field.
#: Written as a sibling of it, which is where that field's relative path resolves.
INIT_BINARY = "init_binary.bin"

#: The per-stage artifacts, a sibling of SPYRE_CODE_DIR rather than a child. The
#: one name here that no code *constructs*: the archive takes it along because it
#: walks the whole export directory, and nothing on the launch path opens it. Named
#: anyway, because it is part of the layout and the tests assert it is carried.
DEBUG_DIR = "debug"

#: The compile stage, its artifact's file extension, and the value recorded in
#: metadata["stage"] -- one name in three roles, and they have to agree:
#: ``binary_ext`` is how CompiledKernel picks which cached file to read as the
#: kernel, and it picks it by matching this extension against the stage names.
SPYRECODE_STAGE = "spyrecode"


def resolve_dbo_opt(required: bool = True) -> Optional[str]:
    """Absolute path to the ``dbo-opt`` named by ``knobs.spyre.dbo_opt``.

    A value containing a path separator is taken literally; a bare name is
    looked up on ``PATH``. Returns ``None`` instead of raising when
    ``required`` is False, which is what :meth:`SpyreBackend.hash` wants: a
    missing tool must still produce a cache key (and a distinct one), with the
    actionable error raised later by the stage that actually needs the tool.
    """
    tool = knobs.spyre.dbo_opt
    path = tool if os.sep in tool else shutil.which(tool)
    if path is None or not os.path.isfile(path):
        if not required:
            return None
        raise RuntimeError(
            f"dbo-opt not found: {tool!r}. Set TRITON_SPYRE_DBO_OPT (or "
            "knobs.spyre.dbo_opt) to the dbo-opt binary that should compile "
            "KTIR to SpyreCode."
        )
    return os.path.realpath(path)


def resolve_device(required: bool = True) -> Optional[str]:
    """Absolute path to the device description named by ``knobs.spyre.device``.

    Returns ``None`` when the knob is unset, which is a legitimate configuration
    and not an error: dbo-opt then falls back to its own default device under
    ``$DEEPTOOLS_PATH``. A knob pointing at a file that does not exist *is* an
    error, and gets an actionable one -- unless ``required`` is False, which lets a
    caller that only needs to describe the setting (rather than read the file)
    proceed.

    The counterpart of :func:`resolve_dbo_opt`, and for the same reason: these two
    settings are what a caller has to get right together, so turning each knob into
    a usable value belongs in one place that the stage and the tests can share.
    """
    device = knobs.spyre.device
    if not device:
        return None
    path = Path(device)
    if not path.is_file():
        if not required:
            return None
        raise FileNotFoundError(
            f"knobs.spyre.device / TRITON_SPYRE_DEVICE points at {device!r}, "
            "which does not exist. Leave it unset to let dbo-opt use its own "
            "default device under $DEEPTOOLS_PATH."
        )
    return str(path.resolve())


# Pointer argument i is based at i * 16 GiB, matching the assignment
# torch-spyre's Inductor path makes, so the two producers of a Spyre binary agree
# by construction. Segment 7 holds the program, which is why only 7 pointers fit.
_SEGMENT_BYTES = 16 * 1024 ** 3
_MAX_POINTER_ARGS = 7

# Width of an MLIR element type, read off its own spelling: i8/i32, f16/f32,
# bf16, and the f8 family (f8E4M3FN, ...) all name it after the leading letters.
# BITWIDTH_DICT is keyed by *Triton* spellings ("fp16") and the IR hands us MLIR
# ones, with no reverse map in tree — hence a regex rather than a second table.
_MLIR_ELEM_BITS = re.compile(r"^(?:bf|[fiu])(\d+)")


def _elem_bytes(pointee: str) -> int:
    match = _MLIR_ELEM_BITS.match(pointee)
    bits = int(match.group(1)) if match else None
    if not bits or bits % 8:
        raise ValueError(
            f"pointer element type {pointee!r} has no usable byte width "
            f"(read as {bits!r} bits); Spyre base addresses are element "
            "indices, so the width must be a whole number of bytes"
        )
    return bits // 8


def _segment_addresses(signature_types) -> Tuple[int, ...]:
    """Segment addresses in ELEMENTS for the ``*``-prefixed entries, in order."""
    ptr_types = [ty for ty in signature_types
                 if isinstance(ty, str) and ty.startswith("*")]
    if len(ptr_types) > _MAX_POINTER_ARGS:
        raise ValueError(
            f"Spyre supports at most {_MAX_POINTER_ARGS} pointer arguments "
            f"(segment {_MAX_POINTER_ARGS} holds the program itself); this kernel "
            f"has {len(ptr_types)}: {ptr_types}"
        )
    return tuple(i * _SEGMENT_BYTES // _elem_bytes(ty[1:])
                 for i, ty in enumerate(ptr_types))


def entry_func_name(mod) -> str:
    """The kernel entry function's name, refusing the empty string.

    ``get_entry_func_name`` (``python/src/ir.cc``) scans the module's top-level ops
    for a ``tt.func`` that ``triton::isKernel`` accepts, and returns ``""`` when it
    finds none. That answer conflates two different situations — a module with no
    kernel in it, and a module whose entry point is no longer a ``tt.func`` — and
    the second one is *reachable in this pipeline*: ``ConvertFunctions`` rewrites the
    entry point to a ``func.func``, and the binding's ``dyn_cast`` then misses every
    op.

    So the empty string is refused here rather than passed on. It is the difference
    between a name and a silence, and everything downstream that takes a name
    (``metadata["name"]``, the base-address inference) treats what it is given as
    one.

    Callers must therefore read this *before* the core pipeline runs. Both do.
    """
    name = mod.get_entry_func_name()
    if not name:
        raise RuntimeError(
            "module has no kernel entry function that get_entry_func_name can "
            "see. Either there is no kernel in it, or its entry point is no "
            "longer a tt.func -- ConvertFunctions rewrites it to a func.func, "
            "and this must be read before that runs."
        )
    return name


def infer_base_addresses_from_ptr_types(mod) -> Tuple[int, ...]:
    """The default base addresses, from ``mod``'s entry-function pointer types.

    "Infer" because this is the fallback when nobody set
    ``SpyreOptions.base_addresses``: it applies the fixed segment policy rather than
    reporting where any buffer really lives.

    Both the *count* and the *widths* matter. The count decides how many segments
    are handed out; the address of segment i is ``i * 16 GiB`` in *elements*, so it
    also depends on that pointer's own pointee type — ``["*f32", "*f32"]`` and
    ``["*f16", "*f16"]`` have the same number of arguments and different
    addresses.

    Must run while the entry function still takes ``!tt.ptr<...>`` arguments,
    i.e. before or at the very start of the TTIR→KTIR pipeline: ConvertFunctions
    rewrites every pointer to a bare ``index`` and the widths are then gone.
    """
    entry = entry_func_name(mod)
    return _segment_addresses(mod.get_function_signature(mod.get_function(entry)))


@dataclass
class SpyreOptions:
    # Per-axis partition of the Spyre hardware grid. One entry per
    # tl.program_id axis the kernel reads; prod(grid) is the total
    # physical core count. Default covers the common 1D-on-32-cores
    # case. A 2D kernel with grid = (16, 2) would partition the same
    # 32 cores as 16x2 across axes x and y.
    grid: Tuple[int, ...] = (32,)

    # Explicit override for the kernel's HBM base addresses, positionally: entry
    # i is the address for the i-th `index` argument of the lowered function,
    # which ConvertFunctions produced from the i-th !tt.ptr argument. Empty (the
    # default) means the backend derives them from the TTIR pointer types with
    # the fixed i * 16 GiB segment policy — that is the right answer for
    # a Triton launch, where the runtime binds buffers to those segments.
    #
    # It stays an option because a caller may want to specify addresses.
    #
    # Values are ELEMENT indices, not byte addresses.
    #
    # SOON TO BE DEPRECATED, along with baked addresses generally. A baked address
    # is only correct if torch-spyre binds the buffers to those segments at launch,
    # which it does only in the non-symbolic mode of the BUNDLE_SYMBOLIC_ARGS
    # environment variable. How long that variable stays respected is not settled --
    # the SDSC codegen path has already stopped requiring it to be off -- and once
    # it is not, addresses have to arrive symbolically and be patched through the
    # correction table (see symbolic_args). Treat this field as the current
    # mechanism, not the intended one.
    base_addresses: Tuple[int, ...] = ()

    # How the kernel's buffer addresses reach the entry function.
    #
    # True: leave the addresses symbolic, for a runtime that patches them in
    # through the correction table. SpyreLauncher names one tensor per symbol in
    # the SymbolicArg payload it passes to launch_jobplan.
    #
    # False: the pointer arguments are replaced by base addresses —
    # base_addresses if set, otherwise the derived ones — and the runtime binds
    # the buffers to those segments instead.
    #
    # The dataclass default is False, but it is not the effective one: every
    # launch goes through parse_options, which defaults the field from
    # BUNDLE_SYMBOLIC_ARGS — set to "1" by importing torch_spyre, so a launching
    # process compiles symbolic unless it says otherwise. The launcher refuses a
    # launch where the two disagree; see SpyreLauncher._check_argument_mode_agrees.
    #
    # This is a *compile* option, not an environment read at pass-install time:
    # it changes the emitted artifact, so it has to be in options.hash() and
    # therefore in the cache key. Its default comes from the BUNDLE_SYMBOLIC_ARGS
    # environment variable, read once in parse_options.
    symbolic_args: bool = False

    # ---- Required by Triton code generator -----
    sanitize_overflow: bool = False
    debug: bool = False
    allowed_dot_input_precisions: tuple = ("ieee",)

    # num_warps and shared are deliberately NOT options. Nothing reads them from
    # here -- the only reader is CompiledKernel._init_handles, which takes them
    # from metadata, so _make_spyrecode reports them there. That comment explains
    # the values.
    #
    # Two things follow from leaving them out. They cannot reach options.hash(),
    # so they cannot make the cache key differ for an artifact that would be
    # identical either way. And `kernel[grid](..., num_warps=4)` now fails with
    # "Keyword argument num_warps was specified but unrecognised", rather than
    # being accepted and quietly having no effect -- which is the truthful answer
    # on a device with no warps.
    #
    # debug and instrumentation_mode cannot move the same way: JITFunction.run
    # adds both to the launch kwargs every time, and _pack_args rejects any kwarg
    # that is not an option, so they have to stay fields.

    # JITFunction.run injects instrumentation_mode into kwargs unconditionally
    # (python/triton/runtime/jit.py) and _pack_args rejects any launch kwarg
    # that is not a field of the parsed options, so every launch fails
    # that check unless the field exists. Spyre runs no instrumentation passes;
    # the value is accepted and ignored.
    instrumentation_mode: str = ""

    def __post_init__(self):
        # Normalize list → tuple for hashability / dataclass equality.
        if isinstance(self.grid, list):
            self.grid = tuple(self.grid)
        if isinstance(self.base_addresses, list):
            self.base_addresses = tuple(self.base_addresses)
        # Symbolic mode leaves the pointer arguments un-materialized, so there is
        # nothing for supplied addresses to be baked into. Silently honouring one
        # and dropping the other would pick a mode the caller did not ask for.
        if self.symbolic_args and self.base_addresses:
            raise ValueError(
                "base_addresses and symbolic_args=True are mutually exclusive: "
                "symbolic mode does not materialize the pointer arguments, so "
                f"there is nothing for base_addresses={self.base_addresses} to be "
                "baked into. Drop one."
            )
        # Validated here rather than at a launch site because __post_init__ is the
        # one funnel both entry points cross: compile_time_launch_options runs only
        # for kernel[grid](...), not for a direct triton.compile().
        if self.instrumentation_mode:
            raise ValueError(
                f"instrumentation_mode={self.instrumentation_mode!r} is not "
                "supported: Spyre runs no instrumentation passes, so the value "
                "would be accepted and quietly do nothing. Unset "
                "TRITON_INSTRUMENTATION_MODE (knobs.compilation."
                "instrumentation_mode), which JITFunction.run injects into every "
                "launch."
            )

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
        # Set here, not in add_stages: CompiledKernel constructs a *fresh*
        # backend via make_backend() (python/triton/compiler/compiler.py),
        # so an attribute assigned on the compiling instance is not the one it
        # reads. It also decides bytes-vs-text per artifact — the file
        # whose extension matches binary_ext is read as bytes.
        self.binary_ext = SPYRECODE_STAGE

    def hash(self) -> str:
        """Backend identity folded into the on-disk cache key.

        Includes the ``dbo-opt`` binary and the device description, because both
        change the emitted artifact while leaving the source, the signature and
        the options untouched. Without them, repointing or rebuilding dbo-opt
        silently reuses a stale binary. NVIDIA closes the same hole by folding
        ``get_ptxas_version(arch)`` into its options
        (``third_party/nvidia/backend/compiler.py``).
        """
        return "-".join([
            f"spyre-{self.target.arch}",
            self._dbo_opt_identity(),
            self._device_identity(),
            f"dbo_debug-{int(bool(knobs.spyre.dbo_debug))}",
        ])

    @staticmethod
    def _dbo_opt_identity() -> str:
        """``dbo-opt`` identity: resolved path plus size and mtime.

        dbo-opt has no meaningful ``--version`` (it reports the LLVM version it
        was linked against, which is identical across rebuilds), so size+mtime
        of the resolved path is what actually distinguishes two builds. A
        missing tool gets its own key rather than raising — see resolve_dbo_opt.
        """
        path = resolve_dbo_opt(required=False)
        if path is None:
            return f"dbo_opt-missing-{knobs.spyre.dbo_opt}"
        st = os.stat(path)
        return f"dbo_opt-{path}-{st.st_size}-{st.st_mtime_ns}"

    @staticmethod
    def _device_identity() -> str:
        """Digest of the device description, or an explicit "no file" marker.

        ``knobs.spyre.device`` is optional: unset, dbo-opt falls back to its own
        default under ``$DEEPTOOLS_PATH``. In that case there is no file to
        digest, so we say so instead of hashing an empty string as though it
        were a device — a real device file must never collide with "default".

        Reads the knob directly rather than going through :func:`resolve_device`,
        which cannot serve this: the resolver returns one value for both "unset"
        and "set but missing", and those two have to key differently here.
        """
        device = knobs.spyre.device
        if not device:
            return "device-dbo_opt_default"
        path = Path(device)
        if not path.is_file():
            # Report it rather than hash nothing; _make_spyrecode raises with a
            # usable message when it actually needs the file.
            return f"device-{device}-missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return f"device-{device}-{digest}"

    def parse_options(self, options: dict) -> SpyreOptions:
        """Build :class:`SpyreOptions` from the launch/compile kwargs.

        The one environment read on this path lives here. ``BUNDLE_SYMBOLIC_ARGS``
        is torch-spyre's spelling of the argument-passing mode, so it is honoured,
        but it is turned into an *option* at this single point instead of being
        consulted where the pass is installed: only then does it reach
        ``SpyreOptions.hash()`` and so the cache key. Read later, a compile under
        one value could be silently served from the cache under the other.

        Polarity follows torch-spyre's ``prepare_kernel.cpp``, where
        ``bind_io_addresses_ = (env == nullptr || env != "1")``: only the literal
        ``"1"`` selects symbolic arguments; unset or ``"0"`` means the runtime
        binds the addresses from the tensor list, which is the supported mode. An
        explicitly passed ``symbolic_args`` wins over the environment.
        """
        parsed = {k: v for k, v in options.items()
                  if k in SpyreOptions.__dataclass_fields__}
        # Read from the environment rather than through knobs, unlike every
        # setting this backend owns. BUNDLE_SYMBOLIC_ARGS is torch-spyre's
        # variable: it describes how *that* runtime will bind buffers at launch,
        # and declaring it as a Triton knob would have Triton publishing another
        # project's contract as though it were stable and ours to define.
        #
        # This expression is copied from torch_spyre/_inductor/config.py, and the
        # default of "1" is the load-bearing part: importing torch_spyre does
        # `os.environ.setdefault("BUNDLE_SYMBOLIC_ARGS", "1")`, so any process that
        # launches has it set, and prepare_kernel.cpp then does NOT bind addresses
        # from the tensor list. Defaulting the compile to baked instead would emit
        # addresses nobody corrects and nobody binds -- wrong data, no diagnostic.
        #
        # `== "1"` and not a truthy test, because that is torch-spyre's own rule
        # (prepare_kernel.cpp: bind_io_addresses_ = env != "1"). Accepting
        # "true"/"on" here would disagree with the runtime in the other direction.
        parsed.setdefault("symbolic_args",
                          os.environ.get("BUNDLE_SYMBOLIC_ARGS", "1") == "1")

        # Supplying base_addresses selects the baked mode. They only mean anything
        # there, and with symbolic the default, a caller that passes addresses and
        # nothing else would otherwise trip the mutual-exclusion check against a
        # mode it never asked for -- dataflow-test-framework's `dft triton-lower`
        # passes exactly that. An EXPLICIT symbolic_args=True alongside them is
        # still a contradiction, and __post_init__ still refuses it.
        if parsed.get("base_addresses") and "symbolic_args" not in options:
            parsed["symbolic_args"] = False

        # Nothing else is injected here. Every pass either stage needs is in that
        # stage's own list in third_party/spyre/lib/Pipeline.cpp, which is the
        # whole of what a caller can ask for: the pipelines take a fixed ordered
        # list and there is no knob to add to it.
        return SpyreOptions(**parsed)

    def compile_time_launch_options(self, grid, specialization) -> dict:
        """Contribute the grid as a *compile* input.

        The grid is baked into the artifact, so it must be part of the options
        before the cache key is computed, and this hook is the only way to get
        it there: ``JITFunction.run`` takes ``grid`` as its own keyword-only
        parameter, so it never reaches the ``**options`` the argument binder
        collects, and ``parse_options`` runs after the key. Without it
        ``kernel[(2,)](...)`` and ``kernel[(4,)](...)`` have identical
        specializations and identical kwargs: they collide, and the second
        launch silently runs the binary baked for the first grid.

        The grid is the *only* thing that needs the hook. ``base_addresses``
        used to be contributed here too, but it is a pure function of the
        specialized signature, and ``specialization`` is already in the cache
        key — so it is derived inside the backend instead, in ``_make_ktir``.
        ``specialization`` stays in the signature because the hook is a general
        upstream capability and another backend may want it.

        See the hook's default in ``python/triton/backends/compiler.py`` and its
        single call site in ``python/triton/runtime/jit.py``.
        """
        if grid is None:
            # warmup=True: there is no grid to bake, leave the option default.
            return {}
        if callable(grid):
            raise ValueError(
                "Spyre bakes the grid into the compiled artifact, so a callable "
                "grid cannot be used: the tile count must be known before the "
                "kernel is compiled. Pass a concrete tuple, e.g. kernel[(32,)]."
            )
        return {"grid": tuple(grid)}

    def add_stages(self, stages: dict, options: SpyreOptions, language=None) -> None:
        stages["ttir"] = lambda src, metadata: self._make_ttir(src, metadata, options)
        stages["ktir"] = lambda src, metadata: self._make_ktir(src, metadata, options)
        stages[SPYRECODE_STAGE] = lambda src, metadata: self._make_spyrecode(src, metadata, options)

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

          - NVIDIA (``third_party/nvidia/backend/compiler.py``) —
            ``(1, 1, 16)`` for fp16/bf16, ``(1, 1, 32)`` for int8/fp8.
            Only K is constrained; small M/N are padded into tensor
            cores.
          - AMD (``third_party/amd/backend/compiler.py``) —
            ``(1, 1, 1)``, falling back to FMA for configurations not
            natively supported by its matrix cores.
          - Interpreter (``python/triton/runtime/interpreter.py``) —
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
        # MLIR_ENABLE_DUMP, honoured for the first time on this backend: the GPU
        # backends call this on every pass manager they build and ours never did,
        # so the variable was silently inert here. One call per pass manager, and
        # every stage needs its own.
        pm.enable_debug()
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
        """Lower optimized TTIR to KTIR: build the stage's pipeline, run it.

        The pass list is ``buildTTIRToKTIRPipeline``, in
        third_party/spyre/lib/Pipeline.cpp, which is also what
        ``spyre-triton-opt --spyre-ttir-to-ktir`` runs. It is deliberately not
        restated here: it was stated in two places before this, and the two
        drifted without any test being able to see it.

        The default pointer base addresses are also inferred here, into
        metadata["base_addresses"], and consumed by _make_spyrecode. It has to
        happen *before* this pipeline runs: ConvertFunctions rewrites every
        !tt.ptr<f16> argument to a bare index, and the element widths the address
        policy needs are gone from the IR after that. Nothing about them belongs
        in the launch path, since they are a pure function of the specialized
        signature, which the cache key already covers.

        Inferring them does not install MaterializeBaseAddresses — _make_spyrecode
        does that, so the cached .ktir keeps its arguments. A caller wanting a
        zero-argument entry function against addresses of its own sets
        options.base_addresses and reads the *spyrecode* stage's module; the
        materialization does not happen in this stage, and cannot be asked for
        here.
        """
        from triton._C.libtriton import ir, spyre

        # Read before the pipeline, for the same reason as the base addresses
        # below: the binding behind this name only matches a tt.func, and
        # ConvertFunctions -- the last conversion -- has rewritten the entry point
        # to a func.func by the end of this method. Read afterwards it comes back
        # as the empty string for every kernel, and silently, because "no match"
        # and "no kernel" are the same answer there.
        #
        # A name is not load-bearing for execution, which is what let a blank one
        # survive: it is what CompiledKernel reports as `.name`, what reaches
        # utils.load_binary, and what torch-spyre puts in its log lines, profiler
        # event names and failure reports. 
        metadata["name"] = entry_func_name(mod)

        # Only the address-binding mode has any use for these. Inferring them in
        # symbolic mode would also mean reporting a pointer-width or pointer-count
        # problem in place of the NotImplementedError the caller is actually about
        # to get, which buries the real answer.
        if not options.symbolic_args:
            metadata["base_addresses"] = infer_base_addresses_from_ptr_types(mod)

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()  # MLIR_ENABLE_DUMP
        # The grid is passed as a list because the binding takes a
        # std::vector<int64_t>; SpyreOptions keeps it a tuple to stay hashable.
        spyre.passes.ttir_to_ktdp.add_ttir_to_ktir_pipeline(
            pm,
            grid=list(options.grid),
        )
        pm.run(mod, "make_ktir")

        metadata["stage"] = "ktir"
        return mod

    def _make_spyrecode(self, mod, metadata, options):
        """Lower KTIR to a loadable Spyre binary by running ``dbo-opt``.

        Returns the exported directory as **ZIP bytes**. A compile stage yields
        one artifact, but the export is a ``SPYRE_CODE_DIR`` holding
        ``SPYRECODE_JSON`` and ``INIT_BINARY``, with a ``debug/`` tree of
        per-stage artifacts beside it, so the archive is the single artifact and
        ``SpyreUtils.load_binary`` unpacks it.

        Member names are relative to the export directory, so unpacking
        reproduces the layout as written rather than a second convention of our
        own. That is what ``torch_spyre``'s ``SpyreSDSCKernelRunner`` expects: it
        is handed a directory and appends ``/spyreCodeDir`` itself before calling
        ``prepare_kernel``.

        Two steps. First a KTIR → KTIR round trip through
        ``buildSpyrecodePipeline`` (third_party/spyre/lib/Pipeline.cpp, also
        reachable as ``spyre-triton-opt --spyre-prepare-spyrecode``), whose pass
        list and admission rule are stated there. Then ``dbo-opt --from-ktir
        --kEmitSpyreCode``, whose scheduler + codegen stages write the
        spyreCodeDir. ``--kEmitSpyreCode`` is a pass pipeline that has to be
        requested explicitly; ``--export-dir`` alone makes dbo-opt exit 0 having
        written nothing.

        The only decision left here is the one the round trip cannot make for
        itself, because it depends on values this method has and the IR does not:
        whether to bind the base addresses, and which. That is where
        ``options.symbolic_args`` is honoured — the one place in the backend that
        branches on the mode.
        """
        from triton._C.libtriton import ir, spyre

        bind = not options.symbolic_args
        base_addresses = ()
        if bind:
            base_addresses = options.base_addresses or metadata.get("base_addresses")
            if base_addresses is None:
                # A compile that starts from a .ktir source skips _make_ktir, and
                # by then the element widths are no longer in the IR to recover.
                # Distinct from an empty override, which means "no pointers".
                raise RuntimeError(
                    "no base addresses were derived for this kernel. They come "
                    "from the TTIR pointer types in _make_ktir, so a compile "
                    "entered at the .ktir stage cannot produce them; compile "
                    "from the @triton.jit source, or set "
                    "SpyreOptions.base_addresses explicitly."
                )

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()  # MLIR_ENABLE_DUMP
        spyre.passes.ttir_to_ktdp.add_spyrecode_pipeline(
            pm,
            bind_base_addresses=bind,
            base_addresses=list(base_addresses),
        )
        pm.run(mod, "make_spyrecode")

        dbo_opt = resolve_dbo_opt()
        device = resolve_device()

        # Debug mode: leave the directory on disk so kernel.ktir and the
        # per-stage artifacts under export/debug/ survive the call, whether it
        # succeeds or fails.  Non-debug cleans up automatically.
        with tempfile.TemporaryDirectory(
            prefix="spyrecode_", delete=not knobs.spyre.dbo_debug,
        ) as _tmp_str:
            tmp = Path(_tmp_str)
            ktir_path = tmp / "kernel.ktir"
            ktir_path.write_text(str(mod))
            export_dir = tmp / "export"
            export_dir.mkdir()

            argv = [dbo_opt, "--from-ktir", "--kEmitSpyreCode",
                    f"--export-dir={export_dir}"]
            if device:
                argv.append(f"--device={device}")
            argv.append(str(ktir_path))

            env = dict(os.environ)
            if knobs.spyre.dbo_debug:
                env["DBO_DEBUG"] = "1"
            # dbo-opt prints the optimized module on stdout; only the files it
            # writes under --export-dir matter here.
            result = subprocess.run(argv, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                # Say which dbo-opt this was and how it was chosen. A bare knob
                # value is looked up on PATH, so an old system install wins
                # without anyone selecting it, and its dialect errors read as a
                # broken branch rather than as a stale tool.
                configured = knobs.spyre.dbo_opt
                origin = (f"PATH, since knobs.spyre.dbo_opt is the default "
                          f"{configured!r}" if os.sep not in configured
                          else f"knobs.spyre.dbo_opt={configured!r}")
                debug_note = (f"\n  debug files at: {tmp}"
                              if knobs.spyre.dbo_debug else "")
                raise RuntimeError(
                    f"dbo-opt failed (exit {result.returncode}):\n"
                    f"  tool: {dbo_opt}\n"
                    f"  from: {origin}\n"
                    f"  argv: {' '.join(argv)}{debug_note}\n"
                    "If the diagnostics below name an unknown attribute or "
                    "dialect, this dbo-opt is older than the KTIR this backend "
                    "emits: set TRITON_SPYRE_DBO_OPT to a newer one.\n"
                    f"{result.stderr}"
                )

            code_dir = export_dir / SPYRE_CODE_DIR
            # dbo-opt can exit 0 having written nothing, so check rather than
            # trust the exit status.
            missing = [name for name in (SPYRECODE_JSON, INIT_BINARY)
                       if not (code_dir / name).is_file()]
            if missing:
                raise RuntimeError(
                    f"dbo-opt exited 0 but did not write {', '.join(missing)} "
                    f"under {code_dir}\n  argv: {' '.join(argv)}\n{result.stderr}"
                )

            # Relative to export_dir, not to code_dir: the archive keeps the
            # layout as exported, spyreCodeDir/ and debug/ side by side, because
            # that is the layout the launch path consumes.
            members = [(path, path.relative_to(export_dir).as_posix())
                       for path in sorted(export_dir.rglob("*")) if path.is_file()]

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for path, name in members:
                    # Fixed timestamp and mode: the artifact digest is the key
                    # SpyreUtils.load_binary unpacks under, so identical inputs
                    # must give identical bytes. Real mtimes would make every
                    # recompile look like a new binary.
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.external_attr = 0o644 << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())

        metadata["stage"] = SPYRECODE_STAGE

        # Before the first launch, CompiledKernel._init_handles refuses to run a
        # kernel that asks for more than the device has. It makes two such checks,
        # and needs a number from us for each:
        #
        #   shared > max_shared_mem                  how much shared memory the
        #                                            kernel needs, vs how much
        #                                            the device reports
        #   num_warps * warp_size > n_max_threads    how many threads it wants,
        #                                            vs how many are allowed
        #
        # Spyre has neither warps nor Triton's shared memory. Its LX scratchpad is
        # a different thing: the device description sizes it and the scheduler
        # allocates it, with nothing for Triton to bound. So we report zero need,
        # both comparisons come out false, and the launch goes ahead. Reporting
        # anything larger would invent a limit that does not exist and refuse a
        # kernel that runs perfectly well.
        #
        # Reported here, and not carried as options a caller could set, because
        # nothing about a Spyre kernel changes them. In *this* stage specifically
        # because it is the only one certain to run: a compile may start partway
        # down the pipeline (first_stage = stages.index(src.ext), +1 for an
        # IRSource), so a .ttir input skips _make_ttir and a .ktir input skips
        # _make_ktir as well. Setting them in either would leave some compiles
        # without them, and _init_handles would then fail with AttributeError at
        # the first launch. This stage produces the artifact, so if there is a
        # kernel to launch at all, this ran.
        #
        # If _init_handles ever changes upstream, these go stale silently -- hence
        # the reasoning here and beside each value the driver reports.
        metadata["num_warps"] = 1
        metadata["shared"] = 0
        return buffer.getvalue()
