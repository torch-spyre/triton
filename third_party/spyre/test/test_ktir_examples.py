#!/usr/bin/env python3
"""
Numerical KTIR tests for every kernel variant discovered from
``fixtures/*/meta.py``.

One class :class:`TestExample` drives everything via pytest parametrize:
``test_numerical`` runs the kernel through ``ktir_cpu`` and compares to
the variant's NumPy oracle, with per-variant ``xfail_numerical`` marks.

Structural IR checking used to live here too. It now lives in the
lit/FileCheck suite under ``test/Conversion/*.mlir``, which pins the exact
lowered IR per pass rather than asserting op presence over a compiled
fixture.
"""

import pytest
from conftest import EXAMPLES, KTIRCpuTester

# ---------------------------------------------------------------------------
# Discovered variants — subset of EXAMPLES that came in via meta.py discovery
# (anything with ``kernel_fn`` is compiled from Triton source; path-based
# legacy entries are excluded).
# ---------------------------------------------------------------------------

DISCOVERED = sorted(k for k, v in EXAMPLES.items() if "kernel_fn" in v)


def _keys_with_numerical_xfail():
    """Param list for the numerical test — attaches each variant's mark.

    ``disabled`` variants skip (they can't compile, so there is nothing to
    run numerically). For the rest, ``xfail_numerical`` in meta.py is
    either a short reason string or a dict forwarded to
    ``pytest.mark.xfail(**d)`` (so ``raises=ValueError`` etc. work); it
    is built at collection time so failures are reported as proper
    XFAIL, not SKIP.
    """
    params = []
    for k in DISCOVERED:
        entry = EXAMPLES[k]
        marks = []
        disabled = entry.get("disabled")
        if disabled is not None:
            marks.append(pytest.mark.skip(reason=disabled["reason"]))
        else:
            xfm = entry.get("xfail_numerical")
            if xfm is not None:
                kw = xfm if isinstance(xfm, dict) else {"reason": xfm, "strict": True}
                marks.append(pytest.mark.xfail(**kw))
        params.append(pytest.param(k, marks=marks, id=k))
    return params


def test_disabled_variants_tracking_tests_exist():
    """Every ``disabled`` entry must carry a non-blank ``tracking_test``.

    A ``disabled`` variant in ``meta.py`` carries a ``tracking_test``
    string naming where the underlying gap is pinned — normally a lit
    file, e.g. ``"Conversion/lower-descriptor-memory-addptr-invalid.mlir"``.
    This meta-test checks only that the field is present and non-blank.

    It used to parse ``"<file>.py::<ClassName>"``, import the module and
    look for a ``test_*`` attribute on the class. That shape assumed the
    tracking test was a *Python* test; once the single-pass suites became
    lit files (#55), no ``.mlir`` could satisfy any of those three checks.

    So the rot this catches is now narrow — an empty or missing pointer.
    A renamed or deleted tracking file is **not** caught: grep for the
    filename if you move one. When a gap closes, either re-enable the
    variant or remove the ``disabled`` block.

    Adding more ``disabled`` rules later
    -----------------------------------
    If the ``disabled`` schema grows (extra sub-fields, new invariants),
    the current shape — one big function with a shared ``failures``
    list — stays readable for a handful of independent rules. Each new
    rule appends to ``failures`` and the assert at the end dumps all of
    them. Keep rules *independent* (no ``continue`` cascades) so a
    single variant with two issues surfaces both at once rather than
    only the first.

    Refactor into per-rule methods once the rules start interacting
    (e.g. rule B only applies when rule A passes) or once per-variant
    failure visibility in CI matters enough to want separate pytest
    node IDs per rule. The shape would be a ``TestDisabledField`` class
    with one ``@pytest.mark.parametrize``'d method per rule, so failing
    tests report as e.g. ``test_tracking_test_resolves[matmul__bmm]``.
    """
    failures = []
    for key, entry in EXAMPLES.items():
        disabled = entry.get("disabled")
        if disabled is None:
            continue
        tracking = disabled.get("tracking_test")
        if not tracking:
            failures.append(f"{key}: 'disabled' has no 'tracking_test'")
            continue

        # ``tracking_test`` is free text naming where the gap is pinned —
        # validated as present and non-blank, nothing more.
        #
        # So the rot this rule catches is narrow: an empty or missing
        # pointer. A renamed lit file will not be caught here — grep for the
        # filename if you move one.
        if not isinstance(tracking, str) or not tracking.strip():
            failures.append(
                f"{key}: tracking_test must be a non-empty string, got "
                f"{tracking!r}"
            )

    assert not failures, (
        "Disabled variants have broken tracking_test references:\n  "
        + "\n  ".join(failures)
    )


class TestExample(KTIRCpuTester):
    """Parametrized numerical suite for every kernel variant under ``fixtures/``.

    ``test_numerical`` runs the kernel on ``ktir_cpu`` and compares against
    the NumPy oracle in ``reference.py``. Per-variant ``xfail_numerical``
    marks in ``meta.py`` express known execution-layer gaps (e.g.
    ``ktir_cpu`` cannot parse dynamic memref).

    New kernels added under ``fixtures/`` are picked up automatically by
    discovery; they supply their oracle in ``meta.py``.

    Structural claims about the lowered IR belong in the lit suite under
    ``test/Conversion/``, not here.
    """

    @pytest.mark.parametrize("key", _keys_with_numerical_xfail())
    def test_numerical(self, key):
        """Execute the kernel on ``ktir_cpu`` and compare to the NumPy oracle.

        Per-variant ``xfail_numerical`` (in ``meta.py``) is attached as a
        ``pytest.param`` mark so the failure mode shows as ``XFAIL``,
        strict, with the declared reason — e.g. "ktir_cpu regex parser
        cannot parse memref<?xf32>" for the dynamic variant.

        Skips the variant if ``reference`` is not declared (structure-only
        variants with no numerical oracle).
        """
        import numpy as np

        entry = EXAMPLES[key]
        if entry.get("reference") is None:
            pytest.skip(f"{key}: no numerical oracle")
        self.EXAMPLE = key
        self.setup_method()

        # ``inputs(**param_values)`` returns the kernel's buffer kwargs plus any
        # runtime scalars the oracle needs to see (e.g. gather's ``y_offset``).
        # The rest of the kernel's runtime args are filled from ``params``.
        #
        # Taken from ``signature``, which is the runtime ABI, rather than as
        # ``params`` minus ``constexprs``: a param need not be an argument of the
        # kernel at all. ``DTYPE`` is one -- it drives ``signature()`` and
        # ``inputs()`` and no kernel reads it -- so subtracting left it in and
        # ``run_cpu`` rejected it as an unknown kwarg. That is what made the dead
        # ``DTYPE: tl.constexpr`` parameters undeletable; asking the signature
        # what the kernel takes cannot invent an argument it does not have.
        #
        # The ``not in inputs`` filter keeps the ``run_cpu`` kwarg merge
        # collision-free when ``make_inputs`` stashes a scalar beside the buffers.
        param_values = entry["param_values"]
        inputs = entry["inputs"](**param_values)
        runtime_scalars = {
            k: param_values[k] for k in entry["signature"]
            if k not in inputs and k in param_values
        }
        func_name = entry.get("func_name") or entry["kernel_fn"].__name__
        outputs = self.run_cpu(
            func_name, kernel_fn=entry["kernel_fn"],
            **inputs, **runtime_scalars,
        )

        ref = entry["reference"](inputs)
        output_key = entry["output_key"]
        np.testing.assert_allclose(outputs[output_key], ref,
                                   rtol=entry.get("rtol", 1e-6),
                                   atol=entry.get("atol", 0))
