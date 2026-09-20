import hashlib
import io
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

from . import tensor_layout
from .compiler import SPYRE_CODE_DIR, SPYRECODE_JSON


# ---------------------------------------------------------------------------
# SpyreUtils
#
# ``utils`` and ``launcher_cls`` are NOT declared on DriverBase — it declares
# only is_active / map_python_to_cpp_type / get_current_target /
# get_active_torch_device / get_benchmarker / allocate_default_profile_scratch.
# They are an undeclared convention that ``CompiledKernel`` reaches for by name:
#
#   driver.active.utils.get_device_properties(device)["max_shared_mem"]
#   driver.active.utils.unload_module(self.module)
#   driver.active.launcher_cls(self.src, self.metadata)
#   driver.active.utils.load_binary(name, kernel, shared, device)
#
# plus get_current_device() / get_current_stream(device) unconditionally at the
# top of JITFunction.run (jit.py). Every in-tree backend implements the same
# shape (CudaUtils / CudaLauncher), so this is the local spelling of an existing
# Triton role — but an upstream refactor moves the contract with no deprecation.
# ---------------------------------------------------------------------------

class SpyreUtils:
    """The ``driver.active.utils`` members Triton's runtime calls.

    Deliberately not a singleton. ``CudaUtils`` is one (``__new__`` plus a
    process-wide ``__init__``) only so that its ``driver.c`` is compiled once;
    there is nothing to compile here.
    """

    #: Sub-directory of ``knobs.cache.dir`` holding unpacked export directories.
    MODULE_CACHE = "spyre-modules"

    def get_device_properties(self, device):
        """One key, because one key is what Triton actually reads.

        ``CompiledKernel._init_handles`` uses ``max_shared_mem`` to bound
        ``metadata.shared``. Spyre's LX scratchpad is not Triton shared memory —
        it is sized by the device description and allocated by the scheduler — so
        0 is reported against ``shared = 0`` and the check is a no-op rather than
        a false floor.

        CUDA reports seven keys. The other six are read only by
        ``python/triton/testing.py``; inventing them would make that module
        print nonsense instead of failing, so they stay absent.
        """
        del device
        return {"max_shared_mem": 0}

    def load_binary(self, name, kernel, shared, device):
        """Unpack the compiled artifact and return Triton's 5-tuple.

        ``kernel`` is the ZIP produced by ``SpyreBackend._make_spyrecode``. It is
        extracted, content-addressed, under ``knobs.cache.dir``, and the
        directory is returned as **both** the module and the function: there is
        no loaded-module / entry-point split to mirror, because on Spyre the
        program *is* the directory. ``module`` is what ``unload_module``
        receives; ``function`` is what reaches the launcher.

        The unpack reproduces dbo-opt's ``--export-dir`` layout, so the returned
        directory holds ``spyreCodeDir/`` with ``debug/`` beside it rather than
        the spyreCodeDir's contents at the top. It is the *parent* that is
        returned, because that is what ``torch_spyre``'s
        ``SpyreSDSCKernelRunner`` takes as its ``code_dir``: it appends
        ``/spyreCodeDir`` itself, and ``prepare_kernel`` then opens exactly
        ``<code_dir>/spyreCodeDir/spyrecode.json`` and the ``init_bin_file``
        that names, both by name and with no directory scan anywhere.

        Keyed on the **artifact digest**, not on ``name``. A name is not an
        identity: one ``@triton.jit`` function compiles to as many different
        binaries as it has specializations, grids and option sets, and every one of
        them arrives here under the same name — so keying on it would serve the
        first artifact to whoever compiled the second. What *would* be an identity,
        ``CompiledKernel.hash``, is not passed in. The bytes are, so they are the
        key.

        ``n_regs`` / ``n_spills`` are meaningless here and reported as 0.
        ``n_max_threads`` is 1, which keeps ``_init_handles``' check
        ``num_warps * warp_size > n_max_threads`` (1 * 1 > 1) false.

        ``prepare_kernel`` is deliberately **not** called here, despite being
        once-per-kernel work. It needs an initialized Spyre runtime and SIGSEGVs
        without one, and ``_init_handles`` runs at subscript time — before this
        path has seen a tensor, and allocating a device tensor is what
        initializes the runtime. It happens on the launch path instead, in
        ``SpyreSDSCKernelRunner.jobplan``, which is lazy for that same reason and
        is cached per CompiledKernel — per-kernel just the same, and it keeps
        loading device-free.
        """
        del name, shared, device  # see the docstring: none of the three is usable
        digest = hashlib.sha256(kernel).hexdigest()
        root = Path(knobs.cache.dir) / self.MODULE_CACHE / digest
        if not (root / SPYRE_CODE_DIR / SPYRECODE_JSON).is_file():
            # Extract into a private directory and then move it into place, so a
            # concurrent load never observes a half-written export directory.
            staging = root.parent / f"tmp.pid_{os.getpid()}_{uuid.uuid4().hex}"
            staging.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(kernel)) as archive:
                archive.extractall(staging)
            try:
                os.rename(staging, root)
            except OSError:
                # Another process won the race. Its copy is equivalent, because
                # the directory name is the digest of these very bytes.
                shutil.rmtree(staging, ignore_errors=True)
        return (str(root), str(root), 0, 0, 1)

    def unload_module(self, module):
        """Keep the unpacked directory. Load-bearing, not an omission.

        CUDA calls ``cuModuleUnload`` here; there is no Spyre analogue to undo.
        The directory is content-addressed, so keeping it is a cache rather than
        a leak, and the ``debug/dfir.mlir`` inside it is the only on-disk record
        of what ran.
        """
        del module


# ---------------------------------------------------------------------------
# SpyreLauncher
# ---------------------------------------------------------------------------

def _import_torch_spyre():
    """Import torch-spyre, torch first — the order is not a style choice.

    ``torch_spyre`` is a torch device-backend extension, so ``import torch``
    auto-loads it through an entry point. Reaching for ``torch_spyre`` in a
    process that has not imported torch yet therefore re-enters a half-built
    module: torch's ``_import_device_backends`` looks for ``_autoload`` on a
    ``torch_spyre`` still executing its own line 20, and the caller gets
    ``Failed to load the backend extension`` (or, further along, a duplicate
    ``TORCH_LIBRARY`` for the ``triton`` namespace) rather than anything about a
    launch.

    Not at module scope at all, either: importing this module is how Triton
    discovers the backend, and that happens on every machine with the wheel,
    including ones with no torch-spyre and no device.
    """
    import torch  # noqa: F401  -- must precede torch_spyre; see above
    import torch_spyre
    return torch_spyre


class SpyreLauncher:
    """Pure-Python launcher over the nine-positional ABI ``jit.py`` calls.

    Pure Python is not a departure: no in-tree backend generates per-kernel C
    (``make_launcher`` does not exist in this tree) and CUDA compiles one static
    ``driver.c`` once, process-wide.

    ``__init__`` must touch **no** option field that ``SpyreOptions`` does not
    declare. ``CudaLauncher`` reads ``global_scratch_size``,
    ``profile_scratch_*``, ``launch_cooperative_grid`` and ``launch_pdl`` as hard
    attributes; any of those here is an immediate AttributeError.

    Of the nine positionals (``jit.py``) only ``function`` and ``*args``
    carry information: ``stream`` is a stub, ``packed_metadata`` is ``()``
    (``SpyreBackend.pack_metadata``) and both hooks are ``None`` unless a
    profiler installed them.

    The launch itself is torch-spyre's — ``SpyreSDSCKernelRunner``, which owns
    ``prepare_kernel(<dir>/spyreCodeDir)`` for the JobPlan and
    ``launch_jobplan(plan, tensors, symbolic_args)`` for the launch. Three things
    come with it that this backend did not have: the runtime initialization
    ``prepare_kernel`` requires, profiler events around both calls, and
    first-failure data capture — on a failed launch torch-spyre writes a JSON
    report naming the exception, the kernel and its artifact directory, so the
    failure does not have to be reproduced to be diagnosed (opt-in, via
    ``TORCH_SPYRE_FFDC=1``).

    What is left here is the part that is Triton's: deciding which launch
    arguments carry an address, and in which order.

    This process is still the one that makes those calls, so the process running
    Triton is also the process holding the device, which a Spyre device grants to
    exactly one opener — from the first ``.to("spyre")`` until that process exits,
    not until the tensors are dropped. What it costs the test suite: the launch
    test lives in the pytest suite (``test/test_device_launch.py``) rather than
    under lit, because pytest runs one process sequentially and that is what
    serializes device access, where lit runs one process per file in parallel.
    """

    def __init__(self, src, metadata):
        self.src = src
        self.metadata = metadata
        #: torch-spyre's runner for this artifact, built on first launch.
        #: ``_init_handles`` makes one launcher per CompiledKernel, so an
        #: instance attribute already has the right lifetime: the runner's
        #: jobplan is lazy, so prepare_kernel runs once per compiled kernel
        #: rather than once per launch.
        self._runner = None

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata,
                 launch_metadata, enter_hook, exit_hook, *args):
        # The grid is baked into the artifact (see
        # ``SpyreBackend.compile_time_launch_options``), so the tile count
        # arrives here only because the ABI has a slot for it; it was already
        # consumed at compile time and a different one is a different artifact.
        del gridX, gridY, gridZ
        del stream, packed_metadata, launch_metadata, enter_hook, exit_hook

        # First, and before torch-spyre is imported (see _import_torch_spyre), so
        # a wrong call says what is wrong about it even on a machine that could
        # not have launched anyway.
        tensors = self._address_args(args)

        runner = self._runner_for(function)
        runner.run(*tensors, symbolic_args=self._symbolic_args(len(tensors)))

    def _address_args(self, args):
        """The launch arguments that carry an address, in kernel order.

        ``args`` is ``*bound_args.values()`` from ``JITFunction.run`` — every
        declared parameter, constexprs included — and ``src.signature`` is keyed
        by the same names in the same order, with ``"constexpr"`` where a value
        was baked in. So the pointers are the entries whose type starts with
        ``*``, and dropping the rest leaves the scalars behind on purpose: a
        constexpr is already in the binary, and a *runtime* scalar has no
        launch-time channel at all — ``SymbolicArgKind::kDimension`` is where one
        would go, and it raises "not yet implemented" downstream. Passing one
        therefore comes out as an address-count disagreement, which
        ``launch_jobplan`` rejects (see ``_symbolic_args``).

        Order is the binding, not merely a convention: the correction flit is
        built by walking these positionally, so segment *i* belongs to argument
        *i* and a reordering silently patches the wrong segments.

        That ordering is also what makes this the place the device-layout claims
        are checked. The compile recorded them keyed by pointer *ordinal*, because
        the IR carries no argument names; this loop produces exactly that sequence,
        and it is the one place that has both the ordinal and the name a diagnostic
        needs. See ``backend/tensor_layout.py`` for what the claim is and why the
        two numbers it compares never met before.
        """
        names = list(self.src.signature)
        if len(args) != len(names):
            raise RuntimeError(
                f"SpyreLauncher: {len(args)} launch argument(s) for a signature "
                f"of {len(names)} ({names}). These are positionally paired, so "
                "there is no safe way to guess which is which."
            )
        # Absent, for a kernel compiled before this key existed or one entered at
        # the .ktir stage, means no claim rather than no annotation -- so the
        # lookup below simply finds nothing and every argument passes.
        claims = tensor_layout.entries_by_ptr_index(
            getattr(self.metadata, "device_layouts", None))

        tensors = []
        for name, arg in zip(names, args):
            if not str(self.src.signature[name]).startswith("*"):
                continue
            # launch_jobplan requires every argument to be a Spyre tensor
            # (SpyreStream::launch checks is_privateuseone). Said here, with the
            # parameter name, because from C++ it is an index into a list the
            # caller never built.
            if getattr(getattr(arg, "device", None), "type", None) != "spyre":
                raise TypeError(
                    f"SpyreLauncher: argument {name!r} must be a Spyre tensor, "
                    f"got {type(arg).__name__} on device "
                    f"{getattr(arg, 'device', None)!r}. Move it with "
                    '.to("spyre") — there is no implicit host staging.'
                )
            # After the device check and before the launch: the claim is about a
            # Spyre tensor's device storage, and a host tensor has none to measure.
            claim = claims.get(len(tensors))
            if claim is not None:
                tensor_layout.check_fits(claim, name, arg)
            tensors.append(arg)
        return tensors

    def _runner_for(self, code_dir):
        """torch-spyre's runner for this artifact, built once and kept.

        *code_dir* is the unpacked export directory (``SpyreUtils.load_binary``
        returns it as both module and function), passed as-is: the runner appends
        ``/spyreCodeDir`` itself.

        Nothing is prepared here. ``SpyreSDSCKernelRunner.jobplan`` is lazy and
        calls ``torch.spyre._impl._lazy_init()`` before ``prepare_kernel``, which
        needs an initialized Spyre runtime and SIGSEGVs without one — the reason
        this cannot happen at load time, and a step this backend used to be
        missing rather than doing.

        The name is only ever reported — torch-spyre's log lines, its profiler
        event names, its failure reports — and it comes from ``metadata["name"]``,
        the name the compile recorded, so that a report and an IR dump of the same
        kernel agree. It is guaranteed non-empty: ``_make_ktir`` reads it before
        ``ConvertFunctions`` can erase it and refuses an empty answer, which it
        used to return for every kernel and which would have made every one of
        those reports unattributable.
        """
        if self._runner is None:
            # Imported before the mode check below, not merely before the runner:
            # importing torch_spyre does
            # `os.environ.setdefault("BUNDLE_SYMBOLIC_ARGS", "1")`, so reading the
            # variable first would read a value prepare_kernel will not see.
            _import_torch_spyre()
            from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

            self._check_argument_mode_agrees()
            self._runner = SpyreSDSCKernelRunner(self.metadata.name, str(code_dir))
        return self._runner

    def _check_argument_mode_agrees(self):
        """Refuse a launch whose runtime mode is not the artifact's.

        ``prepare_kernel`` takes the mode from the environment at call time
        (``prepare_kernel.cpp``: ``bind_io_addresses_ = (env == nullptr || env !=
        "1")``, so only the exact string ``"1"`` means symbolic and unset is *not*
        neutral), while the artifact was compiled in the mode ``parse_options``
        recorded in ``SpyreOptions.symbolic_args``. They normally agree, because
        both read ``BUNDLE_SYMBOLIC_ARGS``; they disagree when the option was
        decided some other way — ``base_addresses`` selects baked — or when an
        artifact cached under one value is launched under the other.

        Reported rather than corrected, and that is the choice worth stating.
        Setting the variable around the call would mean reaching past
        ``SpyreSDSCKernelRunner``'s lazy ``jobplan`` to force preparation under an
        environment of our own — reimplementing the launch path this backend
        delegates — and the variable is torch-spyre's, read by other code in this
        process. Left alone it is silent: the runtime binds addresses from the
        tensor list while the binary expects them patched in, or the reverse, and
        either way the kernel reads somebody else's memory.
        """
        ambient = os.environ.get("BUNDLE_SYMBOLIC_ARGS")
        if (ambient == "1") == bool(self.metadata.symbolic_args):
            return
        compiled_for = "symbolic" if self.metadata.symbolic_args else "baked"
        raise RuntimeError(
            f"SpyreLauncher: this artifact was compiled for {compiled_for} "
            f"addresses (symbolic_args={self.metadata.symbolic_args}), but "
            f"BUNDLE_SYMBOLIC_ARGS={ambient!r} in this process, which is how "
            "torch-spyre's prepare_kernel picks the mode it prepares for (only "
            '"1" means symbolic). Launching across that disagreement patches or '
            "binds the wrong addresses without failing, so set "
            f'BUNDLE_SYMBOLIC_ARGS={"1" if self.metadata.symbolic_args else "0"} '
            "for this process, or recompile with the value it already has."
        )

    def _symbolic_args(self, count):
        """The typed payload saying which tensor patches which symbol.

        One ``kAddress`` entry per address argument, in the order
        ``_address_args`` returned them — which is the order the artifact's
        symbols were compiled in, so entry *i* resolves correction-vector slot
        *i*.

        This is what replaced counting the artifact's own symbols here. With a
        payload passed, ``JobPlanStepHostCompute::construct`` TORCH_CHECKs its
        length against the compiled symbol count
        (``hcm_->vdci.inputSym_.size()``) and a disagreement is loud. With an
        empty one it takes the legacy path — walk every context tensor as an
        address source, patch however many it finds — which is exactly the silent
        wrong-segment failure the old ``_artifact_address_count`` check existed to
        catch from outside.

        ``None`` in baked mode, where there are no symbols to patch: the
        addresses are ``arith.constant`` in the binary, and the runner then takes
        ``launch_jobplan``'s two-argument form.
        """
        if not self.metadata.symbolic_args:
            return None
        torch_spyre = _import_torch_spyre()
        kind = torch_spyre._C.SymbolicArgKind.kAddress
        # dim_index and value stay at their -1 defaults: they belong to
        # kDimension, the runtime-scalar kind, which is not implemented
        # downstream.
        return [torch_spyre._C.SymbolicArg(kind=kind, tensor_id=index)
                for index in range(count)]


class SpyreDriver(DriverBase):
    """Spyre device driver.

    Sits on ``DriverBase`` rather than ``GPUDriver`` because
    ``GPUDriver.__init__`` hard-imports ``torch.cuda``
    (``python/triton/backends/driver.py``).

    Absent rather than stubbed: ``get_device_interface``,
    ``allocate_default_profile_scratch`` (inherits the raising base),
    ``get_empty_cache_for_benchmark``, ``clear_cache``, ``set_current_device``;
    ``get_benchmarker`` keeps raising. The cost, stated so it is a choice and not
    a discovery: ``triton.autotune``, ``do_bench``, Proton profiling and
    everything in ``python/triton/testing.py`` do not work.
    """

    def __init__(self) -> None:
        super().__init__()
        # Mirrors CudaDriver.__init__ (nvidia/backend/driver.py): the two
        # members Triton's runtime reaches for by name are assigned here.
        self.utils = SpyreUtils()
        self.launcher_cls = SpyreLauncher

    @classmethod
    def is_active(cls) -> bool:
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        mapping = {
            "i32": "int32_t",
            "f16": "half",
            "fp8": "fp8",
        }
        return mapping.get(ty, ty)

    def get_current_target(self) -> GPUTarget:
        # warp_size = 1 keeps num_warps * warp_size at 1, which
        # SpyreUtils.load_binary's n_max_threads = 1 does not exceed.
        return GPUTarget(backend="spyre", arch=1, warp_size=1)

    def get_active_torch_device(self):
        return None

    def get_current_device(self) -> int:
        """Single device. Called unconditionally at the top of JITFunction.run
        (``jit.py``) and used to index the per-device compile caches."""
        return 0

    def get_current_stream(self, device) -> int:
        """Streams are not modelled. The value is threaded through to the
        launcher's ``stream`` positional, which ignores it."""
        del device
        return 0

    def get_benchmarker(self):
        raise NotImplementedError("Spyre does not support local benchmarking")
