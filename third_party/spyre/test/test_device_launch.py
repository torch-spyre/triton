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


#: torch dtype -> the name of the ``torch_spyre._C.DataFormats`` member
#: SpyreTensorLayout wants. Only the dtypes a replicating layout is used at need an
#: entry; anything else raises rather than guessing, because a wrong device format
#: is a silently wrong reinterpretation of the same bytes.
#:
#: The two names do not follow one convention, so neither can be derived from the
#: other: fp16 is Spyre's own 1-6-9 float and is spelled ``SEN169_FP16``, while
#: fp32 is plain IEEE and is ``IEEE_FP32`` -- there is no ``SEN169_FP32``. Looked
#: up by name rather than held as a value so importing this module does not need
#: the extension.
_DEVICE_FORMAT = {"torch.float16": "SEN169_FP16", "torch.float32": "IEEE_FP32"}


def _empty_with_device_layout(host_array, device_size, stride_map):
    """A Spyre tensor with *host_array*'s shape but *device_size*'s storage.

    ``_lazy_init`` is required and is not implicit: ``spyre_empty_with_layout``
    goes through the allocator, which needs a live C++ RuntimeContext. Without it
    the call raises ``RAS::RUNTIMECONTEXT::ContextNotCreated`` -- the same message
    that otherwise appears as harmless noise on every launch.
    """
    from torch_spyre._C import (DataFormats, SpyreTensorLayout,
                                spyre_empty_with_layout)
    torch.spyre._impl._lazy_init()

    key = str(host_array.dtype)
    torch_dtype = getattr(torch, host_array.dtype.name)
    fmt = _DEVICE_FORMAT.get(str(torch_dtype))
    if fmt is None:
        raise AssertionError(
            f"device_alloc: no SpyreTensorLayout device format known for "
            f"{torch_dtype}; add one to _DEVICE_FORMAT rather than guessing")

    layout = SpyreTensorLayout(device_size=list(device_size),
                               stride_map=list(stride_map),
                               device_dtype=getattr(DataFormats, fmt))
    buf = spyre_empty_with_layout(tuple(host_array.shape),
                                  tuple(s // host_array.itemsize
                                        for s in host_array.strides),
                                  torch_dtype, layout)
    # Zeroed for the same reason the dense path is: a kernel that never writes
    # must not be mistaken for one that wrote the right answer.
    buf.zero_()
    return buf


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
                  for name, array in inputs.items()
                  if name not in (entry.get("device_alloc") or {})}

        # A buffer whose device layout REPLICATES cannot be staged by `.to("spyre")`.
        # That path allocates the host element count, and a broadcast layout needs
        # more: an [M] statistic stored stick-wide occupies M*S elements on the
        # device, so statistic m lands at m*S and only m=0 is inside a dense [M]
        # allocation -- an out-of-bounds write, not merely a wrong answer.
        #
        # torch-spyre's own generated wrappers do not use `.to(...)` here either;
        # they call spyre_empty_with_layout with an explicit SpyreTensorLayout, and
        # the -1 entries in stride_map are what mark the device axes the host tensor
        # does not address. The runner cannot do this for us: it resolves each
        # argument to a base address and never allocates, so the tensor has to be
        # right before it is handed over.
        for name, (device_size, stride_map) in (entry.get("device_alloc") or {}).items():
            staged[name] = _empty_with_device_layout(
                inputs[name], device_size, stride_map)

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
