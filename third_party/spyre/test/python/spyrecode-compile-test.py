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
their ``meta.py`` -- today one, the loop-free single-tile add, because dbo-opt
refuses the loop the others outline from their program-id distribution.

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

from utils import spyre_target


def test_artifact_holds_the_spyre_code_dir(compiled):
    # metadata["name"] is "" (issue #104), so the artifact is keyed by the source
    # function name; what matters is the ZIP's contents.
    names = set(zipfile.ZipFile(io.BytesIO(compiled.kernel)).namelist())
    assert {"spyrecode.json", "init_binary.bin"} <= names
    assert any(n.startswith("debug/") for n in names), sorted(names)


def test_artifact_is_bytes(compiled):
    # binary_ext decides bytes-vs-text when CompiledKernel reads the cache back;
    # the spyrecode artifact must come back as bytes.
    assert isinstance(compiled.kernel, bytes)


def test_cache_files_include_the_artifact(compiled):
    exts = {p.rsplit(".", 1)[-1] for p in compiled.metadata_group}
    assert {"ttir", "ktir", "spyrecode", "json"} <= exts


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
