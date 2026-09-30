# RUN: %python -m pytest %s -q
# REQUIRES: dbo-opt

"""The ``spyrecode`` stage through a real compile — KTIR to a loadable binary.

The ``REQUIRES`` line above is the gate, and it is at that level on purpose. These
tests run pytest, and pytest exits 0 when it skips, so a gate living only in the
fixtures would report a run that compiled nothing as an ordinary lit ``Passed``. As
a lit feature, the absence is visible: lit reports ``Unsupported``. The feature
itself is defined in ``test/lit.cfg.py``.

(Do not write that directive's name followed by a colon anywhere in this docstring.
lit scans the entire file for its directives, prose included, so a mention becomes a
second and malformed one, and the test turns up ``Unresolved`` rather than skipped.)

The fixtures still gate too, which is not redundancy worth removing: this file is
also runnable as plain ``pytest <file>``, where no lit directive applies, and the
fixture skip is what keeps that from erroring. lit's feature is the *visible* gate;
the fixture is the one that holds under direct invocation.

The kernel comes from whichever fixture variants declare ``compiles_to_binary`` in
their ``meta.py`` -- the loop-free ones, because dbo-opt refuses the loop the
others outline from their program-id distribution. Nothing below is specific to
which kernel that is.

Every test gets its binary from a fixture; none builds an ASTSource of its own. Two
do call triton_compile a second time, because recompiling is precisely what they
assert, and take ``spyrecode_options`` to do it with the same options the first
compile used. The conftest fixtures are a chain, not a checklist -- each depends on
the one above, so asking for a later one brings the earlier:

======================= =============================================
``compilable_example``  the variant key; ``params=COMPILES_TO_BINARY``,
                        so requesting it, directly or not, parametrizes
``spyrecode_options``   compile options for that variant. Pure data --
                        it never skips
``dbo_opt``             the resolved tool path, or ``pytest.skip``. This,
                        and only this, is what makes a test skip
``binary_source``       an ASTSource for the variant, not yet compiled
``compiled``            that source through every stage, symbolic
======================= =============================================

So ``compiled`` alone is already both parametrized and gated:
one input, not three. Ask for nothing the body does not use.

Subsets are safe, with one trap. Because these are module-scoped and pytest caches
each per parameter, a partial request can never hand back a different variant than
its siblings -- there is one ``compilable_example`` value per run either way. The
trap is taking ``spyrecode_options`` (or ``binary_source``) and compiling *without*
``dbo_opt``: parametrized but not gated, so under direct pytest with no tool it
fails instead of skipping. Anything that compiles takes ``dbo_opt``, or takes a
fixture that already did.

Every setting these need is a knob -- ``knobs.spyre.dbo_opt`` and
``knobs.spyre.device`` -- and the backend reads them only through ``knobs``, never
from the environment. ``test/lit.cfg.py`` forwards the two ``TRITON_SPYRE_*``
variables so that a knob configured in the shell reaches a lit run at all.
"""

import hashlib
import io
import zipfile

import pytest
from triton import knobs
from triton.compiler.compiler import compile as triton_compile

from conftest import EXAMPLES
from backend.compiler import (
    DEBUG_DIR,
    INIT_BINARY,
    SPYRE_CODE_DIR,
    SPYRECODE_JSON,
)
from utils import spyre_target


def test_artifact_holds_the_spyre_code_dir(compiled):
    # The export layout, member names relative to the export directory: the
    # spyreCodeDir stays a directory rather than being flattened, because
    # torch-spyre's SpyreSDSCKernelRunner is handed the parent and appends
    # /spyreCodeDir itself.
    #
    # Names from the constants, so this cannot drift from the module that writes
    # them; driver-surface-test.py's TestArtifactLayoutNames is what pins the
    # constants themselves to the names torch-spyre opens.
    names = set(zipfile.ZipFile(io.BytesIO(compiled.kernel)).namelist())
    assert {f"{SPYRE_CODE_DIR}/{SPYRECODE_JSON}",
            f"{SPYRE_CODE_DIR}/{INIT_BINARY}"} <= names
    assert any(n.startswith(f"{DEBUG_DIR}/") for n in names), sorted(names)


def test_the_kernel_name_reaches_the_compiled_kernel(compiled, compilable_example):
    # ``backend-options-test.py`` pins the recording against the stage; this reads
    # the far end of it, off a full compile, where the name is what CompiledKernel
    # reports as ``.name`` and what SpyreLauncher hands to torch-spyre for its log
    # lines, profiler events and failure reports. Read at the end of _make_ktir it
    # was "" here, for every kernel, with nothing to notice it.
    assert compiled.name == EXAMPLES[compilable_example]["kernel_fn"].__name__


def test_artifact_is_bytes(compiled):
    # binary_ext decides bytes-vs-text when CompiledKernel reads the cache back;
    # the spyrecode artifact must come back as bytes.
    assert isinstance(compiled.kernel, bytes)


def test_cache_files_include_the_artifact(compiled):
    exts = {p.rsplit(".", 1)[-1] for p in compiled.metadata_group}
    assert {"ttir", "ktir", "spyrecode", "json"} <= exts


def test_kernel_dump_writes_every_stage(dbo_opt, binary_source, spyrecode_options,
                                        monkeypatch, tmp_path):
    """TRITON_KERNEL_DUMP, and the warm-cache trap that makes it look broken.

    Here rather than with the ``MLIR_ENABLE_DUMP`` tests because it needs a whole
    compile, and a whole compile needs the tool -- so it belongs behind this file's
    feature gate, where the absence is reported as ``Unsupported`` rather than
    vanishing into a skip.

    Takes ``dbo_opt`` for that gate even though it never names the path: the two
    compiles below are the reason, and a test that compiles must be gated or it
    fails instead of skipping under direct pytest.
    """
    monkeypatch.setattr(knobs.cache, "dump_dir", str(tmp_path))
    monkeypatch.setattr(knobs.compilation, "dump_ir", True)

    # A cache hit returns before any stage runs, so there is nothing to dump and
    # nothing says so. This half is the trap; it is documented, so pin it.
    monkeypatch.setattr(knobs.compilation, "always_compile", False)
    triton_compile(binary_source, target=spyre_target(),
                   options=spyrecode_options)
    assert not list(tmp_path.rglob("*.ktir")), "a cache hit dumped something"

    monkeypatch.setattr(knobs.compilation, "always_compile", True)
    triton_compile(binary_source, target=spyre_target(),
                   options=spyrecode_options)
    dumped = {p.suffix[1:] for p in tmp_path.rglob("*") if p.is_file()}
    assert {"ttir", "ktir", "spyrecode"} <= dumped, sorted(dumped)


def test_the_archive_carries_the_module_dbo_opt_was_handed(compiled):
    """Why nothing of ours needs to capture the boundary module.

    ``knobs.spyre.dbo_debug`` is on by default, so every compile already ships
    dbo-opt's own per-stage tree inside the artifact, and the module it was handed
    is in there. ``test_artifact_holds_the_spyre_code_dir`` asserts the tree is
    carried; this asserts what is *in* it, which is the part that makes a separate
    artifact of our own unnecessary.
    """
    names = zipfile.ZipFile(io.BytesIO(compiled.kernel)).namelist()
    debug = [n for n in names if n.startswith(f"{DEBUG_DIR}/")]
    assert debug, sorted(names)
    ktir = [n for n in debug if n.endswith(".mlir") or n.endswith(".ktir")]
    assert ktir, f"no IR under {DEBUG_DIR}/: {sorted(debug)}"


def test_recompile_hits_the_cache(compiled, spyrecode_options):
    again = triton_compile(compiled.src, target=spyre_target(),
                           options=spyrecode_options)
    assert again.hash == compiled.hash
    assert again.kernel == compiled.kernel


def test_artifact_bytes_are_deterministic(compiled, spyrecode_options, monkeypatch):
    # The artifact digest is what SpyreUtils.load_binary unpacks under, so identical
    # inputs must give identical bytes. Real ZIP mtimes would make every recompile
    # look like a new binary.
    monkeypatch.setattr(knobs.compilation, "always_compile", True)
    rebuilt = triton_compile(compiled.src, target=spyre_target(),
                             options=spyrecode_options)
    assert hashlib.sha256(rebuilt.kernel).hexdigest() == \
        hashlib.sha256(compiled.kernel).hexdigest()


def test_missing_device_file_raises(dbo_opt, binary_source, spyrecode_options,
                                    monkeypatch, tmp_path):
    # The one test here that does not take a compile, and cannot: it asserts a
    # compile *fails*, so a fixture returning one that succeeded is the wrong input
    # -- taking it would compile twice and assert on neither. It takes the source
    # instead, and ``dbo_opt`` directly for the skip that ``compiled`` would
    # otherwise have carried.
    monkeypatch.setattr(knobs.spyre, "device", str(tmp_path / "nope.mlir"))
    with pytest.raises(FileNotFoundError, match="TRITON_SPYRE_DEVICE"):
        triton_compile(binary_source, target=spyre_target(),
                       options=spyrecode_options)
