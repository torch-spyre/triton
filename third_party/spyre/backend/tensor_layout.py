"""A kernel buffer's device (stick-tiled) layout, for the two readers that need torch.

A descriptor annotated with ``tl.spyre_tensor_layout`` can ask for a device layout
that occupies **more elements than the host tensor holds**. A splat physical dim is
the case: a ``[M]`` statistic becomes ``[M, S]`` on device, so statistic *m* lands at
*m·S* and only *m = 0* is inside a ``.to("spyre")`` allocation. That is an
out-of-bounds write, not a wrong answer, and until now nothing could see it —
three parties each hold one of the two numbers it takes:

* the **compiler** knows the physical extent, from the annotation plus the
  descriptor's logical shape;
* the **allocator** (``.to("spyre")``) sizes from the host shape alone;
* the **launcher** resolves tensors to base addresses and checks only residency.

The compiler's number is written down in ``metadata["device_layouts"]``, one entry
per annotated buffer, by the ``spyre.ir_utils.get_descriptor_layouts`` query that
``_make_ktir`` calls — see that binding in ``third_party/spyre/triton_spyre.cc`` for
the entry shape and for what absence and ``None`` mean. This module is the other two
parties' side of it: ``check_fits`` is what ``SpyreLauncher`` refuses a launch with,
and ``empty_with_device_layout`` is what a caller allocates with instead of
``.to("spyre")``.

What is here, in the order a caller meets it: ``as_spyre_tensor_layout`` turns an
entry into a ``SpyreTensorLayout``; ``device_bytes`` sizes it; ``fits`` asks whether a
tensor is big enough and ``check_fits`` refuses it if not;
``empty_with_device_layout`` allocates one that is; ``entries_by_ptr_index`` looks an
entry up by the pointer ordinal it is keyed by. Nothing here computes a layout —
everything here needs ``torch_spyre``, and that is the whole membership rule. The
reasoning behind the split is in ``PLAN_LAUNCHER_LAYOUTS.md``.
"""

# ---------------------------------------------------------------------------
# The consumers
#
# Everything below needs torch-spyre, so the import is per-call rather than at
# module scope: this module is imported by backend/driver.py, which Triton imports
# on every machine with the wheel, including ones with no device.
# ---------------------------------------------------------------------------

def as_spyre_tensor_layout(entry: dict, torch_dtype):
    """*entry* as a ``SpyreTensorLayout`` for a tensor of *torch_dtype*.

    The device format comes from ``get_device_dtype``, which is why no dtype table
    lives here: it is the same function torch-spyre's own allocator uses, so a
    dtype it supports is one this supports. A local dtype→format table is the
    specific thing not to build — the one the fixture harness used to carry named
    ``SEN169_FP32``, which does not exist in ``DataFormats`` (fp32 is
    ``IEEE_FP32``), so it would have raised the first time an fp32 variant reached
    it.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype
    return SpyreTensorLayout(device_size=list(entry["device_size"]),
                             stride_map=list(entry["stride_map"]),
                             device_dtype=get_device_dtype(torch_dtype))


def device_bytes(entry: dict, torch_dtype) -> int:
    """Bytes of device storage *entry* needs at *torch_dtype*.

    Through ``get_device_size_in_bytes``, not multiplied out here, because the rule
    is not the obvious one: the **trailing** device extent is ignored and the
    innermost axis always costs a whole stick, so a layout's footprint does not
    shrink when its last extent does. Reimplementing that would be a second
    allocation rule, and the interesting failures are exactly where two rules
    disagree.
    """
    from torch_spyre._C import get_device_size_in_bytes
    return int(get_device_size_in_bytes(as_spyre_tensor_layout(entry,
                                                               torch_dtype)))


def fits(entry: dict, tensor) -> bool:
    """Is *tensor*'s device storage at least what *entry* claims it must be?

    Both sides go through ``get_device_size_in_bytes``: the *need* from the layout
    the compiler recorded, the *have* from the layout the tensor actually carries.
    Same function, so a difference is a real difference and not two sizing rules
    disagreeing.

    ``have > need`` is True. An over-allocated buffer wastes device memory and
    corrupts nothing, and calling it a failure would reject a tensor deliberately
    padded or sliced from a larger one.

    True, vacuously, when there is nothing to compare: a ``None`` footprint, or a
    tensor carrying no device layout at all (a host tensor, or a FakeTensor).
    Vacuous rather than False because the caller's own device check is what should
    report a host tensor, with a message about staging.

    Separate from :func:`check_fits` because two callers want the two shapes:
    ``SpyreLauncher`` wants the refusal, and a caller deciding *how to allocate*
    wants the question — and it must be the same question, or a harness would
    allocate on one rule while the launcher checked another.
    """
    if entry["device_size"] is None:
        return True
    have_layout = tensor.device_tensor_layout()
    if have_layout is None:
        return True

    from torch_spyre._C import get_device_size_in_bytes
    return int(get_device_size_in_bytes(have_layout)) >= device_bytes(
        entry, tensor.dtype)


def check_fits(entry: dict, name: str, tensor) -> None:
    """Refuse *tensor* if its device storage is smaller than *entry* claims.

    Checked for **every** annotated argument, not only the stored ones: a splat
    layout on an input reads out of bounds just as a splat on an output writes out
    of bounds, and the read is the quieter of the two.
    """
    if fits(entry, tensor):
        return
    have_layout = tensor.device_tensor_layout()

    from torch_spyre._C import get_device_size_in_bytes
    need = device_bytes(entry, tensor.dtype)
    have = int(get_device_size_in_bytes(have_layout))

    raise RuntimeError(
        f"SpyreLauncher: argument {name!r} is allocated too small for the device "
        f"layout this kernel was compiled with, so the kernel would "
        f"{_reads_or_writes(entry['access'])} past the end of it.\n"
        f"  declared device_size {list(entry['device_size'])} "
        f"stride_map {list(entry['stride_map'])} -> {need} bytes\n"
        f"  tensor's device_size {list(have_layout.device_size)} "
        f"stride_map {list(have_layout.stride_map)} -> {have} bytes\n"
        f'  .to("spyre") sizes from the host shape alone, which is short whenever '
        "the layout replicates. Stage it with the layout instead — the numbers "
        "are the declared ones above:\n"
        "    import torch\n"
        "    from torch_spyre._C import SpyreTensorLayout, get_device_dtype\n"
        "    torch.spyre._impl._lazy_init()  # this path does not self-initialize\n"
        f"    {name} = host_{name}.to(\"spyre\", device_layout=SpyreTensorLayout(\n"
        f"        device_size={list(entry['device_size'])}, "
        f"stride_map={list(entry['stride_map'])},\n"
        f"        device_dtype=get_device_dtype(host_{name}.dtype)))")


def _reads_or_writes(access: str) -> str:
    return {"load": "read", "store": "write",
            "load_store": "read and write"}.get(access, "address")


def empty_with_device_layout(entry: dict, host_tensor):
    """*host_tensor* staged to the device under *entry*'s layout.

    The replacement for ``.to("spyre")`` on a buffer whose layout replicates.
    Through the patched ``Tensor.to(device_layout=...)``, which allocates with
    ``spyre_empty_with_layout`` — sizing the storage as
    ``max(layout bytes, numel × itemsize)`` — and then copies the host bytes in.
    So the host contents survive, which is what an input needs and what an output
    zeroed on the host relies on: a buffer the kernel never writes must not be
    mistakable for one that wrote the right answer.

    The ``"spyre"`` argument is passed for legibility. The layout branch of the
    patched ``to`` ignores it and takes the destination device from
    ``spyre_empty_with_layout`` regardless.
    """
    if entry["device_size"] is None:
        raise ValueError(
            "this kernel recorded no device layout for the buffer (its physical "
            "extents depend on a runtime argument), so there is nothing to "
            'allocate from; stage it with .to("spyre") and accept that a '
            "replicating layout would overrun it")
    layout = as_spyre_tensor_layout(entry, host_tensor.dtype)
    return host_tensor.to("spyre", device_layout=layout)


def entries_by_ptr_index(entries) -> dict:
    """``{ptr_index: entry}``, tolerating the JSON round trip.

    ``metadata`` goes through ``json.dumps``/``loads``, which turns every tuple
    into a list and every integer dict key into a string — the reason this is
    carried as a ``list`` of dicts and re-keyed here rather than being stored as a
    dict. ``None`` survives unchanged, which is what the dynamic-extent entries
    rely on.

    The key is the ordinal of the entry function's pointer argument, not the
    parameter name: Triton records no argument names in the IR, so the compile
    stage that wrote these had none to use. It is the key the rest of the launch
    ABI already uses anyway — ``_segment_addresses`` hands segment *i* to pointer
    *i* and the correction flit is walked positionally — and the launcher, which
    does have the signature, resolves the ordinal to a name for its diagnostic.
    """
    return {int(e["ptr_index"]): e for e in (entries or [])}
