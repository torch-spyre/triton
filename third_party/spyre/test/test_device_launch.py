"""Launch a compiled fixture on the Spyre device and check it numerically.

The last link, and it is an ordinary Triton launch: inputs go over with
``.to("spyre")`` where CUDA writes ``.cuda()``, ``kernel[grid](...)`` runs, and
``.cpu()`` brings the answer back. Nothing here reaches for a driver, a directory
or a subprocess -- ``SpyreLauncher`` does that, in this process.

Nothing here is a second convention either: inputs, ``reference``,
``output_key``, ``rtol`` and ``atol`` all come from the variant's ``meta.py``, and
they are the same ones the CPU numerical tests use, so the two are the same test
against two backends.

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


class TestDeviceLaunch:
    """The compilable variants, launched on hardware and compared to the oracle."""

    def test_launches_and_matches_the_oracle(self, dbo_opt, compilable_example,
                                             spyrecode_options):
        entry = EXAMPLES[compilable_example]
        inputs = entry["inputs"](**entry["param_values"])
        # Zeroed on the host before it goes over, so that a kernel which never
        # writes cannot be mistaken for one that wrote the right answer. The
        # nonzero assertion below is what reads this.
        inputs[entry["output_key"]] = np.zeros_like(inputs[entry["output_key"]])
        staged = {name: torch.from_numpy(array.copy()).to("spyre")
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
