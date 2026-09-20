"""Launch a compiled fixture on the Spyre device and check it numerically.

The last link, and it is an ordinary Triton launch: inputs go over with
``.to("spyre")`` where CUDA writes ``.cuda()``, ``kernel[grid](...)`` runs, and
``.cpu()`` brings the answer back. Nothing here reaches for a driver, a directory
or a subprocess -- ``SpyreLauncher`` does that, in this process.

Nothing here is a second convention either: inputs, ``reference``,
``output_key``, ``rtol`` and ``atol`` all come from the variant's ``meta.py``, and
they are the same ones the CPU numerical tests use, so the two are the same test
against two backends.

The one place ``.to("spyre")`` is not enough is a buffer whose device layout
REPLICATES, where the host element count is short of what the kernel addresses. That
buffer's allocation comes from the compiled kernel's own recorded layout rather than
from anything in ``meta.py`` -- see :func:`stage`. A variant declares no allocation,
because the layout it already declares is where the answer comes from.

The variants covered are whatever declares ``compiles_to_binary`` -- the
``compilable_example`` fixture parametrizes over them, so a second one extends this
with no change here.

The consequence, accepted: a plain ``pytest third_party/spyre/test`` on a machine
with a device opens it and holds it for the session, the same posture upstream has
with a GPU.
"""

import numpy as np
import pytest

from conftest import EXAMPLES

# torch BEFORE torch_spyre, and that order is a finding rather than a style
# choice: torch auto-loads torch_spyre as a device-backend extension, so reaching
# for it first re-enters a half-built module and the caller gets ``Failed to load
# the backend extension`` -- or, further along, a duplicate ``TORCH_LIBRARY`` for
# the ``triton`` namespace -- rather than anything about a launch. See
# ``_import_torch_spyre`` in backend/driver.py, which documents the same trap.
# importorskip preserves the order because it is two statements, in order.
torch = pytest.importorskip("torch")
pytest.importorskip("torch_spyre")  # registers the "spyre" device with torch


def claims_by_arg_name(compiled, entry):
    """``{argument name: device-layout claim}`` for one compiled kernel.

    ``metadata["device_layouts"]`` is keyed by pointer *ordinal*, because the IR
    carries no argument names (see ``backend/tensor_layout.py``). Ordinal *i* is the
    *i*-th ``*``-typed entry of the signature, and resolving it that way is not a
    convention this file invents: it is the rule ``SpyreLauncher._address_args``
    uses to pair the launch arguments with the same claims, so a harness resolving
    it differently would allocate against one buffer and be checked against another.
    """
    from backend.tensor_layout import entries_by_ptr_index

    ptr_names = [name for name, ty in entry["signature"].items()
                 if str(ty).startswith("*")]
    return {ptr_names[index]: claim
            for index, claim in entries_by_ptr_index(
                getattr(compiled.metadata, "device_layouts", None)).items()}


def stage(host_tensor, claim):
    """*host_tensor* on the device, allocated large enough for *claim*.

    ``.to("spyre")`` unless that is too small, which is a *derived* decision and the
    point of doing this here rather than per variant. That path allocates for the
    host element count; a layout that REPLICATES needs more — an ``[M]`` statistic
    stored stick-wide occupies ``M*S`` elements on the device, so statistic *m* lands
    at *m·S* and only *m = 0* is inside a dense ``[M]`` allocation, an out-of-bounds
    write rather than merely a wrong answer.

    Asked as a question about bytes rather than answered per variant. The fixture
    used to carry a ``device_alloc`` restating ``OUT_LAYOUT`` as a
    ``(device_size, stride_map)`` pair with nothing deriving one from the other, and
    softmax's two statistic buffers made that two more chances to disagree. Now the
    compiled kernel is asked, through the same ``fits`` predicate the launcher
    refuses on — so the harness cannot allocate against a rule the launcher does not
    check.

    Staging first and re-staging on a short answer, rather than predicting what
    ``.to("spyre")`` will give: the prediction would be a third sizing rule. The
    discarded buffer is harmless, nothing having been written to it yet.
    """
    from backend.tensor_layout import empty_with_device_layout, fits

    staged = host_tensor.to("spyre")
    if claim is None or fits(claim, staged):
        return staged
    return empty_with_device_layout(claim, host_tensor)


class TestDeviceLaunch:
    """The compilable variants, launched on hardware and compared to the oracle."""

    def test_launches_and_matches_the_oracle(self, dbo_opt, compilable_example,
                                             spyrecode_options, compiled):
        entry = EXAMPLES[compilable_example]
        inputs = entry["inputs"](**entry["param_values"])
        # Zeroed on the host before it goes over, so that a kernel which never
        # writes cannot be mistaken for one that wrote the right answer. The
        # nonzero assertion below is what reads this. It reaches the device either
        # way: a derived allocation copies the host bytes in, exactly as
        # ``.to("spyre")`` does.
        inputs[entry["output_key"]] = np.zeros_like(inputs[entry["output_key"]])

        # ``compiled`` is the same artifact the launch below will hit in the cache
        # -- same source, same signature, same grid -- taken purely for the layouts
        # it recorded. The alternative is a second statement of them in meta.py,
        # which is the duplication this replaces.
        claims = claims_by_arg_name(compiled, entry)
        staged = {name: stage(torch.from_numpy(array.copy()), claims.get(name))
                  for name, array in inputs.items()}

        # Kernel order comes from the kernel. The registry's ``signature`` holds
        # only the runtime args and its ``constexprs`` is keyed off a set, so
        # neither is an argument order; ``arg_names`` is the declaration order,
        # which is what a positional call needs.
        args = [staged.get(name, entry["param_values"].get(name))
                for name in entry["kernel_fn"].arg_names]

        # Grid comes from spyrecode_options. The fixes dbo-opt requires are
        # injected automatically by parse_options.
        entry["kernel_fn"][spyrecode_options["grid"]](*args)

        output = staged[entry["output_key"]].cpu().numpy()

        # An all-zero output would pass assert_allclose against an all-zero
        # reference while proving only that nothing ran, so the write is asserted
        # separately from the values.
        assert np.count_nonzero(output) > 0, (
            "device wrote nothing: the output buffer is still the zeros it was "
            f"launched with (shape {output.shape})")

        np.testing.assert_allclose(output, entry["reference"](inputs),
                                   rtol=entry.get("rtol", 1e-6),
                                   atol=entry.get("atol", 0.0))
