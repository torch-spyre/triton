# RUN: %python -m pytest %s -q

"""``MLIR_ENABLE_DUMP`` — the debug hook that was inert on this backend.

It is honoured by ``pass_manager.enable_debug()``, which the NVIDIA and AMD
backends call on every pass manager they build and the Spyre stages never did, so
the variable was accepted and did nothing. That is exactly the failure a test is
for: nothing raised, nothing printed, and no way to tell "this pipeline is quiet"
from "this hook is not wired".

Scope. The mechanism is one ``enable_debug()`` call per pass manager, and the
``ttir`` and ``ktir`` stages exercise it fully -- there is nothing about the
``spyrecode`` stage's pass manager that differs, so covering two covers the
mechanism. Lowering through those two needs neither ``dbo-opt`` nor a device,
which is why this file carries no feature requirement; the tests that do need the
tool declare one on the ``dbo-opt`` lit feature, in ``spyrecode-compile-test.py``.

The other two dump variables are upstream Triton's and are not retested here:
``TRITON_KERNEL_DUMP`` writes the per-stage artifacts, which needs a whole compile
and so lives with the gated tests, and ``DBO_DEBUG`` belongs to ``dbo-opt``.

``capfd`` and not ``capsys``: the pass manager's printing goes to LLVM's own
``errs()`` stream, a C++ file descriptor, which Python-level capture does not see.
"""

import tempfile

import pytest

from conftest import EXAMPLES
from utils import compile_to_ttir, make_ktir_mod

# Three *fp32 pointers and one constexpr; nothing here depends on which kernel it
# is, only that lowering it runs some passes.
_EXAMPLE = EXAMPLES["elementwise__2d[M=512]"]


@pytest.fixture
def ttir_file():
    """The example as TTIR on disk, which is what ``make_ktir_mod`` takes."""
    text = compile_to_ttir(_EXAMPLE["kernel_fn"], _EXAMPLE["signature"],
                           _EXAMPLE["constexprs"])
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete_on_close=False) as f:
        f.write(text)
        f.flush()
        yield f.name


class TestMlirEnableDump:

    def test_it_prints_before_every_pass(self, ttir_file, monkeypatch, capfd):
        monkeypatch.setenv("MLIR_ENABLE_DUMP", "1")
        make_ktir_mod(ttir_file, grid=_EXAMPLE["grid"])
        err = capfd.readouterr().err
        # "Before" only. enable_debug() passes printAfterOnlyOnFailure=true, so a
        # pipeline that succeeds prints one dump per pass and no "after" -- shared
        # with the GPU backends, since the flags are upstream's. An "after" dump
        # means the pass failed.
        assert "IR Dump Before" in err
        assert "IR Dump After" not in err
        # Both stages, and each pass manager has to be told separately: the ttir
        # stage runs upstream Triton's passes, the ktir stage runs ours. One
        # enable_debug() call covers a pass manager, not a compile.
        assert "IR Dump Before Inliner" in err
        assert "IR Dump Before LowerDescriptorMemory" in err

    def test_the_whole_module_is_printed(self, ttir_file, monkeypatch, capfd):
        # printModuleScope=true, so each dump is the module and not just the op the
        # pass ran on -- which is what makes the output usable as IR rather than as
        # a fragment.
        monkeypatch.setenv("MLIR_ENABLE_DUMP", "1")
        make_ktir_mod(ttir_file, grid=_EXAMPLE["grid"])
        err = capfd.readouterr().err
        assert "module {" in err or "module attributes" in err

    def test_unset_prints_nothing(self, ttir_file, monkeypatch, capfd):
        # The other half of the claim. Without this, a test asserting the dump
        # appears would also pass against a backend that printed unconditionally.
        monkeypatch.delenv("MLIR_ENABLE_DUMP", raising=False)
        make_ktir_mod(ttir_file, grid=_EXAMPLE["grid"])
        assert "IR Dump Before" not in capfd.readouterr().err

    def test_a_function_name_narrows_it(self, ttir_file, monkeypatch, capfd):
        # A non-boolean value is read as the kernel to dump, so pointing it at a
        # name no module defines is the cheap way to pin that the value is read at
        # all rather than treated as truthy.
        monkeypatch.setenv("MLIR_ENABLE_DUMP", "no_such_kernel")
        make_ktir_mod(ttir_file, grid=_EXAMPLE["grid"])
        assert "IR Dump Before" not in capfd.readouterr().err

    def test_the_named_kernel_is_dumped(self, ttir_file, monkeypatch, capfd):
        # The matching half: the kernel's own name selects it, so the narrowing
        # above is a filter rather than an off switch.
        monkeypatch.setenv("MLIR_ENABLE_DUMP", _EXAMPLE["kernel_fn"].__name__)
        make_ktir_mod(ttir_file, grid=_EXAMPLE["grid"])
        assert "IR Dump Before" in capfd.readouterr().err
