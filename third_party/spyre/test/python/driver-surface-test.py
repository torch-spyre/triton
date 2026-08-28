# RUN: %python -m pytest %s -q

"""The Spyre driver's surface — ``SpyreUtils`` / ``SpyreLauncher``.

``utils`` and ``launcher_cls`` are an *undeclared* convention: they do not
appear on ``DriverBase``, but ``compiler.py`` reaches for them by name at four
sites, and ``JITFunction.run`` calls ``get_current_device()`` /
``get_current_stream()`` unconditionally. Before this skeleton existed,
``kernel[grid](...)`` died with ``AttributeError: launcher_cls``.

Nothing here needs ``dbo-opt`` or a device, so this file carries no ``REQUIRES``
line and runs everywhere, CI included. That is the whole reason it is a lit test
rather than a pytest module: a test that needs nothing should not sit in the suite
that serializes the device.

What that costs, stated rather than discovered: the launcher is constructed
directly here, so nothing in this file proves that a subscript call *arrives* at
it through real ``jit.py`` plumbing — the one thing the compile-driven end-to-end
test it replaced did prove. ``test/test_device_launch.py`` walks that path with
real tensors, subscript through launcher to device, so the coverage exists; but it
is gated on hardware, so on a machine without a device that link is not checked at
all. The decomposition is not free.
"""

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from triton import knobs
from triton.backends.driver import DriverBase

from backend.driver import SpyreDriver, SpyreLauncher, SpyreUtils


# ---------------------------------------------------------------------------
# A duck-typed tensor that is not on the Spyre device: what a plain torch CPU
# tensor presents to the launcher. .dtype must str() to something
# triton._utils.canonicalize_dtype knows, so the rejection is about the *device*
# and not about a member the object failed to have.
# ---------------------------------------------------------------------------

class _FakeDType:

    def __str__(self):
        return "torch.float16"

    def __hash__(self):
        return hash("float16")


class FakeTensor:
    dtype = _FakeDType()

    def __init__(self, address):
        self._address = address

    def data_ptr(self):
        return self._address


class _Src:
    """The one member ``SpyreLauncher`` reads off its source: ``signature``.

    Keyed by parameter name in declaration order, with ``"constexpr"`` where a
    value was baked in — the shape ``JITFunction._pack_args`` builds, which is
    what makes it positionally paired with the launch arguments.
    """

    def __init__(self, signature):
        self.signature = signature


class _SpyreTensor:
    """Enough of a Spyre tensor for the launcher's device check: ``device.type``.

    Not a FakeTensor subclass — that one stands in for a tensor on the wrong
    device and carries ``data_ptr`` / ``dtype``; this one never leaves the
    launcher.
    """

    class _Device:
        type = "spyre"

    device = _Device()


def _zip_bytes(entries):
    """Deterministic ZIP, matching what ``_make_spyrecode`` emits."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    return buffer.getvalue()


_ARTIFACT = {
    "spyrecode.json": b'{"init_bin_file": "init_binary.bin"}',
    "init_binary.bin": b"\x00\x01\x02\x03",
    "debug/dfir.mlir": b"module {}\n",
}


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    """Point knobs.cache.dir at a temp dir so load_binary is observable."""
    monkeypatch.setattr(knobs.cache, "dir", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# The members Triton reaches for by name
# ---------------------------------------------------------------------------

class TestDriverSurface:

    def test_utils_and_launcher_cls_assigned_in_init(self):
        driver = SpyreDriver()
        assert isinstance(driver.utils, SpyreUtils)
        assert driver.launcher_cls is SpyreLauncher

    def test_device_and_stream(self):
        driver = SpyreDriver()
        assert driver.get_current_device() == 0
        assert driver.get_current_stream(0) == 0

    def test_device_properties_has_exactly_one_key(self):
        # The other six CUDA keys are read only by python/triton/testing.py;
        # inventing them would make it print nonsense instead of failing.
        assert SpyreUtils().get_device_properties(0) == {"max_shared_mem": 0}

    def test_utils_is_not_a_singleton(self):
        # CudaUtils is one only to avoid recompiling driver.c; nothing here is
        # compiled, so a plain instance is correct.
        assert SpyreUtils() is not SpyreUtils()

    @pytest.mark.parametrize("member", [
        "get_device_interface",
        "get_empty_cache_for_benchmark",
        "clear_cache",
        "set_current_device",
    ])
    def test_members_stay_absent(self, member):
        # Absent, not stubbed: a fabricated value is worse than an
        # AttributeError, because it fails somewhere else and later.
        assert not hasattr(SpyreDriver(), member)

    def test_benchmarker_and_profile_scratch_still_raise(self):
        driver = SpyreDriver()
        with pytest.raises(NotImplementedError):
            driver.get_benchmarker()
        with pytest.raises(NotImplementedError):
            driver.allocate_default_profile_scratch(0, 0, 0)

    def test_base_class_is_driver_base_not_gpu_driver(self):
        # GPUDriver.__init__ hard-imports torch.cuda, which is not available and
        # would not mean anything here.
        from triton.backends.driver import GPUDriver
        assert issubclass(SpyreDriver, DriverBase)
        assert not issubclass(SpyreDriver, GPUDriver)


# ---------------------------------------------------------------------------
# load_binary / unload_module
# ---------------------------------------------------------------------------

class TestLoadBinary:

    def test_returns_the_five_tuple(self, cache_dir):
        artifact = _zip_bytes(_ARTIFACT)
        module, function, n_regs, n_spills, n_max_threads = \
            SpyreUtils().load_binary("", artifact, 0, 0)
        # Both handles are the same directory: the program *is* the directory,
        # so there is no loaded-module / entry-point split to mirror.
        assert module == function
        assert (n_regs, n_spills) == (0, 0)
        # Must be >= 1 so num_warps * warp_size (1) does not exceed it.
        assert n_max_threads >= 1

    def test_unpacks_flat_with_debug_beside(self, cache_dir):
        module, _, _, _, _ = SpyreUtils().load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        root = cache_dir / SpyreUtils.MODULE_CACHE / hashlib.sha256(
            _zip_bytes(_ARTIFACT)).hexdigest()
        assert str(root) == module
        # prepare_kernel opens <dir>/spyrecode.json and the init_bin_file it
        # names, both by name with no directory scan, so debug/ beside them is
        # invisible to it.
        assert (root / "spyrecode.json").is_file()
        assert (root / "init_binary.bin").is_file()
        assert (root / "debug" / "dfir.mlir").is_file()

    def test_keyed_on_the_artifact_digest(self, cache_dir):
        artifact = _zip_bytes(_ARTIFACT)
        module, _, _, _, _ = SpyreUtils().load_binary("", artifact, 0, 0)
        assert hashlib.sha256(artifact).hexdigest() == module.rsplit("/", 1)[-1]

    def test_name_is_not_part_of_the_key(self, cache_dir):
        # metadata["name"] is "" (issue #104), so keying on it would collide
        # every kernel into one directory.
        artifact = _zip_bytes(_ARTIFACT)
        first, _, _, _, _ = SpyreUtils().load_binary("", artifact, 0, 0)
        second, _, _, _, _ = SpyreUtils().load_binary("something_else", artifact, 0, 0)
        assert first == second

    def test_same_artifact_reuses_its_directory(self, cache_dir):
        utils = SpyreUtils()
        artifact = _zip_bytes(_ARTIFACT)
        first, _, _, _, _ = utils.load_binary("", artifact, 0, 0)
        # A marker survives the second load, proving nothing was re-extracted.
        marker = cache_dir / SpyreUtils.MODULE_CACHE / first.rsplit("/", 1)[-1] / "marker"
        marker.write_text("kept")
        second, _, _, _, _ = utils.load_binary("", artifact, 0, 0)
        assert second == first
        assert marker.read_text() == "kept"

    def test_different_artifacts_do_not_collide(self, cache_dir):
        utils = SpyreUtils()
        other = dict(_ARTIFACT, init_binary=b"different")
        other["init_binary.bin"] = b"\x04\x05\x06\x07"
        first, _, _, _, _ = utils.load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        second, _, _, _, _ = utils.load_binary("", _zip_bytes(other), 0, 0)
        assert first != second
        assert (cache_dir / SpyreUtils.MODULE_CACHE / first.rsplit("/", 1)[-1]).is_dir()
        assert (cache_dir / SpyreUtils.MODULE_CACHE / second.rsplit("/", 1)[-1]).is_dir()

    def test_no_partial_directory_is_left_behind(self, cache_dir):
        SpyreUtils().load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        staging = [p.name for p in (cache_dir / SpyreUtils.MODULE_CACHE).iterdir()
                   if p.name.startswith("tmp.")]
        assert staging == []

    def test_unload_module_keeps_the_directory(self, cache_dir):
        utils = SpyreUtils()
        module, _, _, _, _ = utils.load_binary("", _zip_bytes(_ARTIFACT), 0, 0)
        utils.unload_module(module)
        # Deliberate: content-addressed, so a cache rather than a leak, and
        # debug/dfir.mlir is the only on-disk record of what ran.
        assert (Path(module) / "spyrecode.json").is_file()


# ---------------------------------------------------------------------------
# SpyreLauncher
# ---------------------------------------------------------------------------

class TestSpyreLauncher:

    def test_init_touches_no_option_fields(self):
        # CudaLauncher reads global_scratch_size / profile_scratch_* /
        # launch_cooperative_grid / launch_pdl as hard attributes. SpyreOptions
        # has none of them, so reading any would be an immediate AttributeError.
        class Bare:

            def __getattr__(self, name):
                raise AssertionError(f"launcher read metadata.{name}")

        SpyreLauncher(object(), Bare())

    def test_call_rejects_a_non_spyre_tensor_by_name(self):
        # Every launch argument must be a Spyre tensor -- SpyreStream::launch
        # checks is_privateuseone -- and this is where that gets said, while the
        # parameter still has a name. From C++ it is an index into a list the
        # caller never built. Rejected before torch_spyre is imported, so the
        # message is the same on a machine that has no torch-spyre at all.
        launcher = SpyreLauncher(_Src({"x_ptr": "*fp16", "BLOCK": "constexpr"}),
                                 object())
        with pytest.raises(TypeError, match="'x_ptr'"):
            launcher(1, 1, 1, 0, "/some/dir", (), None, None, None,
                     FakeTensor(0), 64)

    def test_call_rejects_an_argument_count_the_signature_does_not_match(self):
        # The pairing is positional, so a length disagreement has no safe guess.
        launcher = SpyreLauncher(_Src({"x_ptr": "*fp16", "y_ptr": "*fp16"}),
                                 object())
        with pytest.raises(RuntimeError, match="1 launch argument"):
            launcher(1, 1, 1, 0, "/some/dir", (), None, None, None,
                     _SpyreTensor())

    def test_address_args_keeps_pointers_in_kernel_order(self):
        # Only the pointers carry an address, and their order IS the binding: the
        # correction flit is built by walking them positionally. Constexprs and
        # runtime scalars are dropped -- a constexpr is already in the binary.
        x, y = _SpyreTensor(), _SpyreTensor()
        launcher = SpyreLauncher(
            _Src({"x_ptr": "*fp16", "n": "i32", "y_ptr": "*fp16",
                  "BLOCK": "constexpr"}), object())
        assert launcher._address_args((x, 4096, y, 64)) == [x, y]


# ---------------------------------------------------------------------------
# _address_args, called directly
#
# The launcher's only decision: which arguments carry an address, and in which
# order. Reached without a compile because it reads nothing but ``src.signature``
# and the arguments handed to it, so a stub source is the whole setup.
# ---------------------------------------------------------------------------

class TestAddressArgs:

    def test_names_the_parameter_of_a_tensor_on_the_wrong_device(self):
        # The name and the type are both in the message: the name says which
        # argument to move, the type says what was passed instead.
        launcher = SpyreLauncher(_Src({"x_ptr": "*fp16"}), object())
        with pytest.raises(TypeError) as excinfo:
            launcher._address_args((FakeTensor(0), ))
        message = str(excinfo.value)
        assert "'x_ptr'" in message
        assert "FakeTensor" in message

    def test_names_the_offending_pointer_not_the_first_one(self):
        # A hardcoded first name would pass the test above and mislead every
        # caller whose second buffer is the one still on the host.
        launcher = SpyreLauncher(
            _Src({"x_ptr": "*fp16", "y_ptr": "*fp16"}), object())
        with pytest.raises(TypeError, match="'y_ptr'"):
            launcher._address_args((_SpyreTensor(), FakeTensor(0)))

    def test_a_scalar_where_a_pointer_is_declared_is_rejected(self):
        # An int has no .device either, and the check is on the declared type, so
        # this must be the device error rather than a silent drop.
        launcher = SpyreLauncher(_Src({"x_ptr": "*fp16"}), object())
        with pytest.raises(TypeError, match="'x_ptr'"):
            launcher._address_args((4096, ))

    def test_too_few_arguments_for_the_signature(self):
        launcher = SpyreLauncher(
            _Src({"x_ptr": "*fp16", "n": "i32", "y_ptr": "*fp16"}), object())
        with pytest.raises(RuntimeError, match="2 launch argument"):
            launcher._address_args((_SpyreTensor(), 4096))

    def test_too_many_arguments_for_the_signature(self):
        # Checked in both directions: the length is compared, not bounded, so an
        # extra argument is as unguessable as a missing one.
        launcher = SpyreLauncher(_Src({"x_ptr": "*fp16"}), object())
        with pytest.raises(RuntimeError, match="2 launch argument"):
            launcher._address_args((_SpyreTensor(), _SpyreTensor()))

    def test_constexprs_and_runtime_scalars_leave_nothing_behind(self):
        # A kernel with no pointers at all patches no addresses. The empty list
        # is the answer, not an error: _prepare compares it against the count the
        # artifact declares, and that is where a disagreement is caught.
        launcher = SpyreLauncher(
            _Src({"n": "i32", "BLOCK": "constexpr"}), object())
        assert launcher._address_args((4096, 64)) == []

    def test_a_constexpr_pointer_is_not_an_address(self):
        # The filter is on the signature entry, not on the argument: "constexpr"
        # does not start with "*", so a pointer baked in at compile time is
        # already in the binary and is not patched again.
        tensor = _SpyreTensor()
        launcher = SpyreLauncher(
            _Src({"x_ptr": "constexpr", "y_ptr": "*fp16"}), object())
        assert launcher._address_args((object(), tensor)) == [tensor]

    def test_declaration_order_is_preserved_not_sorted(self):
        # Segment i belongs to argument i, so the returned order has to be the
        # signature's. Names that sort the other way, to catch a stray sorted().
        z, a = _SpyreTensor(), _SpyreTensor()
        launcher = SpyreLauncher(
            _Src({"z_ptr": "*fp16", "a_ptr": "*fp16"}), object())
        assert launcher._address_args((z, a)) == [z, a]
