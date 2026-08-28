import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


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

    #: Sub-directory of ``knobs.cache.dir`` holding unpacked spyreCodeDirs.
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

        The unpack is flat — ``spyrecode.json`` and ``init_binary.bin`` at the
        top with dbo-opt's ``debug/`` beside them — because ``prepare_kernel``
        opens exactly ``<dir>/spyrecode.json`` and the ``init_bin_file`` that
        names, both by name and with no directory scan anywhere.

        Keyed on the **artifact digest**, not on ``name``: ``name`` is ``""``
        (issue #104 — ``get_entry_func_name`` dyn_casts to ``tt.func`` after
        ``convert_functions`` has made it ``func.func``), and
        ``CompiledKernel.hash`` is not passed in.

        ``n_regs`` / ``n_spills`` are meaningless here and reported as 0.
        ``n_max_threads`` is 1, which keeps ``_init_handles``' check
        ``num_warps * warp_size > n_max_threads`` (1 * 1 > 1) false.

        ``prepare_kernel`` is deliberately **not** called here, despite being
        once-per-kernel work. It needs an initialized Spyre runtime and SIGSEGVs
        without one, and ``_init_handles`` runs at subscript time — before this
        path has seen a tensor, and allocating a device tensor is what
        initializes the runtime. It belongs in the launcher, built lazily and
        cached per CompiledKernel, which is per-kernel just the same and keeps
        loading device-free.
        """
        del name, shared, device  # see the docstring: none of the three is usable
        digest = hashlib.sha256(kernel).hexdigest()
        root = Path(knobs.cache.dir) / self.MODULE_CACHE / digest
        if not (root / "spyrecode.json").is_file():
            # Extract into a private directory and then move it into place, so a
            # concurrent load never observes a half-written spyreCodeDir.
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


def _artifact_address_count(directory):
    """How many launch addresses the artifact expects, per its own plan.

    A symbolic artifact patches its addresses in through a correction flit built
    by a ``ComputeOnHost`` step, and that step's ``ishape`` is the number of input
    symbols it substitutes — ``["3"]`` for a three-pointer kernel, one symbol per
    address, the same count as ``len(hcm.vdci.inputSym_)``.

    Nothing downstream checks it: ``JobPlanStepHostCompute::construct`` walks the
    launch arguments in order and hands the patcher however many addresses it was
    given. So a disagreement does not fail — it patches the wrong segments and
    the kernel reads somebody else's memory.

    Returns ``None`` when there is no count to compare against: no
    ``ComputeOnHost`` step (a baked artifact carries its addresses already), or
    the ``["0"]`` ishape that means fake symbols.
    """
    plan = json.loads((Path(directory) / "spyrecode.json").read_text())
    for command in plan.get("JobExecPlan", []):
        if command.get("command") != "ComputeOnHost":
            continue
        ishape = command.get("properties", {}).get("ishape", [])
        if len(ishape) != 1 or int(ishape[0]) == 0:
            return None
        return int(ishape[0])
    return None


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

    A launch is two torch-spyre calls — ``prepare_kernel(spyreCodeDir)`` for the
    JobPlan, then ``launch_jobplan(plan, tensors)`` — and this process is the one
    that makes them. That means the process running Triton is also the process
    holding the device, which a Spyre device grants to exactly one opener — from the
    first ``.to("spyre")`` until that process exits, not until the tensors are
    dropped. What it costs the test suite: the launch test lives in the pytest suite
    (``test/test_device_launch.py``) rather than under lit, because pytest runs one
    process sequentially and that is what serializes device access, where lit runs
    one process per file in parallel.
    """

    def __init__(self, src, metadata):
        self.src = src
        self.metadata = metadata
        #: The prepared JobPlan, built on first launch. ``_init_handles`` makes
        #: one launcher per CompiledKernel, so an instance attribute already has
        #: the right lifetime: prepare_kernel runs once per compiled kernel, not
        #: once per launch.
        self._plan = None

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata,
                 launch_metadata, enter_hook, exit_hook, *args):
        # The grid is baked into the artifact (see
        # ``SpyreBackend.compile_time_launch_options``), so the tile count
        # arrives here only because the ABI has a slot for it; it was already
        # consumed at compile time and a different one is a different artifact.
        del gridX, gridY, gridZ
        del stream, packed_metadata, launch_metadata, enter_hook, exit_hook

        # Both checks run before torch-spyre is imported (see
        # _import_torch_spyre), so a wrong call says what is wrong about it even
        # on a machine that could not have launched anyway.
        tensors = self._address_args(args)
        plan = self._prepare(function, len(tensors))

        torch_spyre = _import_torch_spyre()
        torch_spyre._C.launch_jobplan(plan, tensors)

    def _address_args(self, args):
        """The launch arguments that carry an address, in kernel order.

        ``args`` is ``*bound_args.values()`` from ``JITFunction.run`` — every
        declared parameter, constexprs included — and ``src.signature`` is keyed
        by the same names in the same order, with ``"constexpr"`` where a value
        was baked in. So the pointers are the entries whose type starts with
        ``*``, and dropping the rest leaves the scalars behind on purpose: a
        constexpr is already in the binary, and a *runtime* scalar has no
        launch-time channel at all (the artifact would have to declare it, and
        the address count check below is what notices).

        Order is the binding, not merely a convention: the correction flit is
        built by walking these positionally, so segment *i* belongs to argument
        *i* and a reordering silently patches the wrong segments.
        """
        names = list(self.src.signature)
        if len(args) != len(names):
            raise RuntimeError(
                f"SpyreLauncher: {len(args)} launch argument(s) for a signature "
                f"of {len(names)} ({names}). These are positionally paired, so "
                "there is no safe way to guess which is which."
            )
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
            tensors.append(arg)
        return tensors

    def _prepare(self, function, num_addresses):
        """The artifact's JobPlan, prepared once and kept.

        *function* is the unpacked spyreCodeDir (``SpyreUtils.load_binary``
        returns it as both module and function), passed as-is: it is flat, with
        ``spyrecode.json`` at its root, which is what ``prepare_kernel`` opens.

        ``prepare_kernel`` needs an initialized Spyre runtime and SIGSEGVs
        without one, which is why this is not done at load time — a device
        allocation is what initializes it, and by here the caller has necessarily
        made one to get the Spyre tensors above.
        """
        if self._plan is not None:
            return self._plan

        expected = _artifact_address_count(function)
        if expected is not None and expected != num_addresses:
            raise RuntimeError(
                f"SpyreLauncher: the artifact at {function} patches {expected} "
                f"address(es) but this launch supplies {num_addresses}. The "
                "correction flit is built positionally, so a count mismatch "
                "would silently patch the wrong segments."
            )

        # prepare_kernel.cpp reads BUNDLE_SYMBOLIC_ARGS with getenv at call time:
        #     bind_io_addresses_ = (env == nullptr || env != "1")
        # so only the exact string "1" means symbolic and unset is NOT neutral —
        # it binds addresses. The artifact records the mode it was compiled in
        # (``symbolic_args`` is a SpyreOptions field, so it is in metadata), so
        # take it from there: the ambient value cannot then disagree with the
        # binary, which is stronger than asserting that it does not.
        #
        # Set for the call and put back, because the variable is torch-spyre's
        # and other code in this process reads it. Restoring is not enough on its
        # own, though: ``torch_spyre._inductor.config`` snapshots it into a
        # module-level ``bundle_symbolic_args`` at import, so a mutation after
        # import is invisible there. That is fine for this path — nothing in
        # prepare_kernel / launch_jobplan consults it, only the Inductor codegen
        # does — and it is why the mutation is kept as narrow as possible.
        torch_spyre = _import_torch_spyre()

        previous = os.environ.get("BUNDLE_SYMBOLIC_ARGS")
        os.environ["BUNDLE_SYMBOLIC_ARGS"] = (
            "1" if self.metadata.symbolic_args else "0")
        try:
            self._plan = torch_spyre._C.prepare_kernel(str(function))
        finally:
            if previous is None:
                del os.environ["BUNDLE_SYMBOLIC_ARGS"]
            else:
                os.environ["BUNDLE_SYMBOLIC_ARGS"] = previous
        return self._plan


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
