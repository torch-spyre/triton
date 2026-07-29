#!/usr/bin/env python3
"""
Structural and numerical KTIR tests for every kernel variant discovered
from ``fixtures/*/meta.py``.

One class :class:`TestExample` drives everything via pytest parametrize.
Each test method falls into one of three categories (see the class
docstring for detail):

  a. **Pipeline invariants** — kernel-agnostic properties of the final
     KTIR (no ``tt.*`` ops, ``ktdp.*`` ops present, DistributeWork ran,
     ...). Parametrized over every variant uniformly.
  b. **Per-variant structural hook** — ``test_extra_checks`` runs the
     variant's own ``extra_checks`` callable from ``meta.py`` for
     claims that depend on variant shape.
  c. **Numerical** — ``test_numerical`` runs the kernel through
     ``ktir_cpu`` and compares to the variant's NumPy oracle, with
     per-variant ``xfail_numerical`` marks.
"""

import pytest
from conftest import EXAMPLES, KTIRCpuTester, KTIRStructuralTester

# ---------------------------------------------------------------------------
# Discovered variants — subset of EXAMPLES that came in via meta.py discovery
# (anything with ``kernel_fn`` is compiled from Triton source; path-based
# legacy entries are excluded).
# ---------------------------------------------------------------------------

DISCOVERED = sorted(k for k, v in EXAMPLES.items() if "kernel_fn" in v)


def _keys():
    """Build pytest params for structural tests.

    Every discovered variant becomes a ``pytest.param``. Variants that
    declare ``"disabled"`` in their ``meta.py`` entry get a
    ``pytest.mark.skip`` with the declared reason, so the structural
    tests surface a visible ``SKIPPED`` line explaining why the variant
    is off (e.g. a known compile-time gap pinned by a single-pass test).

    The ``disabled`` field shape is documented in ``fixtures/README.md``
    and consumed here plus in :func:`_keys_with_numerical_xfail` plus in
    the meta-test :meth:`test_disabled_variants_tracking_tests_exist`.
    """
    params = []
    for key in DISCOVERED:
        disabled = EXAMPLES[key].get("disabled")
        if disabled is None:
            params.append(pytest.param(key, id=key))
        else:
            mark = pytest.mark.skip(reason=disabled["reason"])
            params.append(pytest.param(key, marks=[mark], id=key))
    return params


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
    """Every ``disabled`` entry's ``tracking_test`` must resolve.

    A ``disabled`` variant in ``meta.py`` carries a ``tracking_test``
    string of the form ``"<file>::<ClassName>"`` pointing at a
    single-pass test that pins the underlying gap. This meta-test
    parses each string, imports the file, and checks that the class
    exists and contains at least one ``test_*`` method.

    The point: when the gap closes and someone deletes the tracking
    test, this test fails — flagging the stale ``disabled`` entry that
    now points into the void. Either re-enable the variant (the
    compile succeeded) or update/remove the ``disabled`` block.

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
    import importlib
    import pathlib

    test_dir = pathlib.Path(__file__).parent
    failures = []
    for key, entry in EXAMPLES.items():
        disabled = entry.get("disabled")
        if disabled is None:
            continue
        tracking = disabled.get("tracking_test")
        if not tracking:
            failures.append(f"{key}: 'disabled' has no 'tracking_test'")
            continue

        # Split "file.py::ClassName" — we only validate file + class.
        # Per-method resolution would require pytest's collector; the
        # class-plus-at-least-one-test_ check catches the rot we
        # actually worry about (file renamed, class deleted).
        try:
            file_part, class_part = tracking.split("::", 1)
        except ValueError:
            failures.append(
                f"{key}: tracking_test {tracking!r} not in "
                f"'file.py::ClassName' form"
            )
            continue

        module_name = file_part.removesuffix(".py")
        module_path = test_dir / file_part
        if not module_path.is_file():
            failures.append(
                f"{key}: tracking_test file {file_part} not found in "
                f"{test_dir}"
            )
            continue

        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            failures.append(
                f"{key}: failed to import tracking_test module "
                f"{module_name}: {e}"
            )
            continue

        cls = getattr(mod, class_part, None)
        if cls is None:
            failures.append(
                f"{key}: tracking_test class {class_part!r} not found "
                f"in {file_part}"
            )
            continue

        if not any(name.startswith("test_") for name in dir(cls)):
            failures.append(
                f"{key}: tracking_test {tracking} has no test_* methods"
            )

    assert not failures, (
        "Disabled variants have broken tracking_test references:\n  "
        + "\n  ".join(failures)
    )


class TestExample(KTIRCpuTester, KTIRStructuralTester):
    """Parametrized suite for every kernel variant under ``fixtures/``.

    The test methods split into three groups:

    1. **Pipeline invariants** — properties of the final KTIR that hold for
       every descriptor-based Triton kernel regardless of what the kernel
       computes. If one of these fails, a pipeline pass has regressed, not
       the kernel itself. Examples: no ``tt.*`` ops survived lowering,
       ``ktdp.construct_memory_view`` was produced, etc.

    2. **Per-variant structural hook** (``test_extra_checks``) — runs the
       variant's own ``extra_checks`` callable from ``meta.py``. This is
       where kernel-specific / variant-specific structural claims live
       (e.g. "the dynamic variant produces ``memref<?x...>``").

    3. **Numerical** (``test_numerical``) — runs the kernel on ``ktir_cpu``
       and compares against the NumPy oracle in ``reference.py``.
       Per-variant ``xfail_numerical`` marks in ``meta.py`` express known
       execution-layer gaps (e.g. ``ktir_cpu`` cannot parse dynamic memref).

    New kernels added under ``fixtures/`` automatically pick up
    group (1) from discovery; they add group (2) / (3) content in their
    ``meta.py``.
    """

    # ------------------------------------------------------------------
    # Group 1: pipeline invariants (kernel-agnostic)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("key", _keys())
    def test_no_tt_ops(self, key):
        """No Triton dialect ops should survive the TTIR→KTIR lowering.

        Every descriptor-era op must be rewritten during the pipeline:
          - ``tt.descriptor_load`` / ``tt.descriptor_store`` /
            ``tt.make_tensor_descriptor`` → ``ktdp.load`` / ``ktdp.store`` /
            ``ktdp.construct_memory_view`` (LowerDescriptorMemory)
          - ``tt.get_program_id`` → ``ktdp.get_compute_tile_id``
            (DistributeWork)
          - ``tt.func`` / ``tt.return`` → ``func.func`` / ``func.return``
            (ConvertFunctions)

        A survivor indicates a pass regression, not a kernel-specific issue.
        """
        self.EXAMPLE = key
        self.setup_method()
        self.assert_absent(
            "tt.descriptor_load", "tt.descriptor_store",
            "tt.make_tensor_descriptor", "tt.get_program_id",
            "tt.func", "tt.return",
        )

    @pytest.mark.parametrize("key", _keys())
    def test_no_raw_ptr_ops(self, key):
        """Raw-pointer Triton ops (pre-descriptor idiom) should be absent.

        Our example kernels use ``tl.make_tensor_descriptor`` end to end —
        the raw-pointer ops below should never appear. This is a belt-and-
        suspenders check against upstream regressions that might re-
        introduce them. A variant that deliberately uses the raw-pointer
        idiom (none today; see ``scripts/gen_ttir.py`` for prior examples
        for the shape) would need to opt out.
        """
        self.EXAMPLE = key
        self.setup_method()
        self.assert_absent("tt.splat", "tt.addptr", "tt.load", "tt.store")

    @pytest.mark.parametrize("key", _keys())
    def test_no_tt_ptr_type(self, key):
        """No SSA value in the final KTIR should have a ``!tt.ptr`` type.

        Follows from the descriptor-based flow: ``!tt.ptr`` is a Triton-
        specific type, not part of Spyre's KTIR target. Any surviving
        occurrence means a lowering pass missed a rewrite.
        """
        self.EXAMPLE = key
        self.setup_method()
        for op in self.ops:
            for t in op.result_types:
                assert "!tt.ptr" not in t, (
                    f"Found !tt.ptr in result type of '{op.name}': {t}"
                )

    @pytest.mark.parametrize("key", _keys())
    def test_ktdp_ops_present(self, key):
        """The lowering should emit the expected KTDP memory-access ops.

        Positive counterpart to ``test_no_tt_ops``: not only must the
        ``tt.*`` ops be gone, but the ``ktdp.*`` replacements must exist.

        Kernels whose every memory op is *indirect* (gather/scatter only,
        e.g. the rank-2-index round-trip) emit no direct
        ``ktdp.construct_access_tile`` — the index descriptor_load's tile is
        traced away by the gather/scatter lowering and DCE'd. Such variants
        opt out via ``"direct_access_tile": False`` in ``meta.py``; their
        indirect tiles are pinned by ``extra_checks`` and the single-pass
        tests in ``test_lower_desc_memory.py``.
        """
        entry = EXAMPLES[key]
        self.EXAMPLE = key
        self.setup_method()
        expected = ["ktdp.construct_memory_view", "ktdp.load", "ktdp.store"]
        if entry.get("direct_access_tile", True):
            expected.append("ktdp.construct_access_tile")
        self.assert_present(*expected)

    @pytest.mark.parametrize("key", _keys())
    def test_work_distribution(self, key):
        """DistributeWork must lower ``tl.program_id`` and wrap the body.

        Applies only to variants that use ``tl.program_id`` — single-
        program kernels (e.g. ``gather``) produce neither
        ``ktdp.get_compute_tile_id`` nor the distribution ``scf.for``
        because DistributeWork has nothing to rewrite. Such variants opt
        out by setting ``"parallel": False`` in ``meta.py`` (default is
        ``True``).

        When ``parallel`` is True, both ops must be present. If the
        compute_tile_id op is missing, DistributeWork was skipped or
        silently no-oped (see ``TestRequiresConvertFunctions`` in
        ``test_distribute_work.py`` for that failure mode).

        ``scf.for`` is always required: Spyre-compliant kernels must handle
        any input size for a fixed grid size and tile size, so every parallel
        kernel distributes work via a loop even when the grid maps one axis
        per output dimension.
        """
        entry = EXAMPLES[key]
        if not entry.get("parallel", True):
            pytest.skip(f"{key}: not a parallel kernel (no tl.program_id)")
        self.EXAMPLE = key
        self.setup_method()
        self.assert_present("ktdp.get_compute_tile_id")
        self.assert_present("scf.for")

    @pytest.mark.parametrize("key", _keys())
    def test_access_tile_type(self, key):
        """``ktdp.construct_access_tile`` result must be index-typed.

        The access-tile shape is indexed into the memory view — a non-
        ``index`` result type indicates the op builder picked the wrong
        element type.

        Skipped for all-indirect kernels (``"direct_access_tile": False``),
        which emit no direct ``construct_access_tile``; the indirect tile's
        index typing is covered by ``test_lower_desc_memory.py``.
        """
        entry = EXAMPLES[key]
        if not entry.get("direct_access_tile", True):
            pytest.skip(f"{key}: no direct construct_access_tile (all-indirect kernel)")
        self.EXAMPLE = key
        self.setup_method()
        self.assert_result_type("ktdp.construct_access_tile", "xindex")

    @pytest.mark.parametrize("key", _keys())
    def test_memref_type(self, key):
        """``ktdp.construct_memory_view`` must produce a ``memref<...>`` type.

        Whether the dims are static (``memref<Nxf32>``) or dynamic
        (``memref<?xf32>``) is a variant-specific concern handled by
        ``extra_checks``; here we only check that it *is* a memref.
        """
        self.EXAMPLE = key
        self.setup_method()
        self.assert_result_type("ktdp.construct_memory_view", "memref<")

    # ------------------------------------------------------------------
    # Group 2: per-variant structural hook
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("key", _keys())
    def test_extra_checks(self, key):
        """Run the variant's ``extra_checks`` callable from ``meta.py``.

        Use this for claims that depend on variant-specific structure —
        e.g. "the dynamic variant produces ``memref<?x...>``", "the static
        variant's memory view shape is not the block shape". Skips the
        variant if no ``extra_checks`` is declared.
        """
        entry = EXAMPLES[key]
        if entry.get("extra_checks") is None:
            pytest.skip(f"{key}: no extra_checks")
        self.EXAMPLE = key
        self.setup_method()
        entry["extra_checks"](self)

    # ------------------------------------------------------------------
    # Group 3: numerical (ktir_cpu execution + NumPy oracle)
    # ------------------------------------------------------------------

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

        # ``inputs(**param_values)`` returns the kernel's buffer kwargs
        # plus any runtime scalars the oracle needs to see (e.g.
        # gather's ``y_offset``). Runtime scalars not already in
        # ``inputs`` come from ``params`` \ ``constexpr`` and are
        # merged in here — the ``not in inputs`` filter keeps the
        # ``run_cpu`` kwarg merge collision-free when ``make_inputs``
        # chooses to stash a scalar alongside the buffers.
        param_values = entry["param_values"]
        inputs = entry["inputs"](**param_values)
        runtime_scalars = {
            k: v for k, v in param_values.items()
            if k not in entry["constexprs"] and k not in inputs
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
