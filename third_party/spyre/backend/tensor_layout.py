"""A kernel buffer's device (stick-tiled) layout: captured once, read three times.

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

This module is where the compiler's number is written down, in a form the other two
can read. ``capture_device_layouts`` runs in the ``ktir`` stage and produces
``metadata["device_layouts"]``; ``check_fits`` is what ``SpyreLauncher`` refuses a
launch with; ``empty_with_device_layout`` is what a caller allocates with instead of
``.to("spyre")``.

**Why one module and not three.** The three consumers sit in three places — the
compile stage, the launcher, the test harness — and what they share is one
convention, torch-spyre's ``SpyreTensorLayout``, which is easy to spell two ways
that differ only under a splat. The fixture harness hand-wrote its half and the plan
this closes named that as the defect: ``device_alloc`` duplicated what ``OUT_LAYOUT``
already stated with nothing deriving one from the other.

**No dtype table and no byte-size rule.** ``get_device_dtype`` and
``get_device_size_in_bytes`` are already bound on ``torch_spyre._C``, and both sides
of every comparison here go through them. A local dtype→format table is the specific
thing not to build: the one in the fixture harness named ``SEN169_FP32``, which does
not exist in ``DataFormats`` (fp32 is ``IEEE_FP32``), so it would have raised the
first time an fp32 variant reached it.

**Why the metadata is keyed by pointer ordinal and not by parameter name.** A name
would be the better key, and it is not available: Triton records no argument names in
the IR, and a compile stage is handed ``(module, metadata)`` and never the source. So
the key is the ordinal of the entry function's pointer argument — the same key the
rest of the launch ABI already uses, since ``_segment_addresses`` hands segment *i* to
pointer *i* and the correction flit is walked positionally. The launcher does have the
signature, and resolves the ordinal to a name for the diagnostic.
"""

# ---------------------------------------------------------------------------
# The coordinate ops, by the numbering the IR carries
#
# Mirrors mlir::triton::tts::CoordOp and the frontend's own ``_COORD_OPS`` table
# in python/triton/language/semantic.py. Three copies of a four-entry enum is two
# too many, and this one exists because the numbers arrive here as plain ints in a
# JSON-round-tripped dict: the C++ side cannot hand over an enum, and the frontend
# table maps the other way (keyword -> code).
#
# Only SPLAT is actually branched on below. The other three are named so that a
# reader of ``_stride_map`` can see which cases the fall-through covers.
# ---------------------------------------------------------------------------

IDENTITY, FLOORDIV, MOD, SPLAT = 0, 1, 2, 3


def capture_device_layouts(mod) -> list:
    """``metadata["device_layouts"]`` for *mod*, one entry per annotated buffer.

    Must be called while the entry point is still a ``tt.func`` taking ``!tt.ptr``
    arguments and while the ``tts.tensor_layout`` markers are still ops — so
    before the TTIR→KTIR pipeline. ``LowerTTSMarkers`` turns each marker into an
    attribute and ``ConvertFunctions`` retypes the pointers; after either, this
    returns an empty list and says nothing about why.

    Each entry is a plain dict of JSON-survivable values::

        {"ptr_index": 1, "device_size": [1, 64, 64], "stride_map": [-1, 1, -1],
         "access": "store"}

    ``device_size`` and ``stride_map`` are ``None`` together when any physical
    extent is not known at compile time — a descriptor taking its shape from a
    runtime ``i32`` argument, which ``matmul__spyre_stick_parallel_dynamic`` does.
    Such a kernel is unlaunchable today anyway; the point of recording the entry
    at all is that its absence and its emptiness mean different things.

    A **missing** entry for a pointer argument is not a fault: it means the kernel
    made no claim about that buffer, which is every unannotated descriptor.
    """
    from triton._C.libtriton import spyre
    return [_normalise(raw) for raw in spyre.ir_utils.get_descriptor_layouts(mod)]


def _normalise(raw: dict) -> dict:
    """One raw marker reading, in torch-spyre's convention.

    Pure integer arithmetic, deliberately: nothing here imports torch or
    torch_spyre, because this runs at compile time on machines that have neither.
    What it produces is what ``SpyreTensorLayout``'s
    ``(device_size, stride_map, device_dtype)`` constructor takes, minus the dtype,
    which is the tensor's to supply.
    """
    device_size = raw["device_size"]
    if device_size is None:
        stride_map = None
    else:
        device_size, stride_map = _to_torch_spyre_layout(
            device_size, raw["phys_src"], raw["phys_op"], raw["phys_arg"],
            raw["logical_size"], raw["logical_stride"])

    if raw["is_load"] and raw["is_store"]:
        access = "load_store"
    elif raw["is_store"]:
        access = "store"
    elif raw["is_load"]:
        access = "load"
    else:
        # An annotated descriptor nothing reads or writes. Recorded rather than
        # dropped: the buffer still has to be big enough if the kernel is later
        # edited to use it, and "no access" is a more useful thing for a
        # diagnostic to say than a missing entry.
        access = "none"

    return {"ptr_index": raw["ptr_index"], "device_size": device_size,
            "stride_map": stride_map, "access": access}


def _to_torch_spyre_layout(device_size, phys_src, phys_op, phys_arg,
                           host_size, host_stride):
    """``(device_size, stride_map)`` for a coordinate map, rank-normalized."""
    stride_map = _stride_map(device_size, phys_src, phys_op, host_size,
                             host_stride)
    return _normalise_rank(device_size, stride_map, phys_src, phys_op)


def _stride_map(device_size, phys_src, phys_op, host_size, host_stride):
    """The host element stride each device axis advances by, or ``-1``.

    This is ``dim_map_to_stride_map`` (``spyre_tensor_impl.cpp``) re-derived, and
    it is their loop rather than a table of cases on purpose. The loop runs
    **innermost to outermost** and carries a per-host-dim running stride, which is
    what makes a repeated host dim come out right: a stick split's inner (mod)
    half takes ``host_stride[d]`` and its outer (floordiv) half then takes
    ``host_stride[d] × (the inner half's device extent)``. A table would have to
    state that product as ``host_stride[d] × phys_arg``, which is the same number
    only while the mod half is the innermost axis — true of every layout in tree
    and not a rule anything checks.

    ``-1`` marks a device axis the host tensor does not address, in two cases that
    torch-spyre treats identically (same branch, ``spyre_tensor_impl.cpp``):

      * a **splat**, which replicates rather than partitions, so there is no host
        stride to advance by;
      * a host dim of extent **1**, which has nothing to advance over.

    Neither updates the running stride, so an outer axis over the same host dim is
    unaffected by a splat inside it.
    """
    stride_map = [0] * len(device_size)
    running = {}
    for j in range(len(device_size) - 1, -1, -1):
        d = phys_src[j]
        if phys_op[j] == SPLAT or host_size[d] == 1:
            stride_map[j] = -1
            continue
        stride_map[j] = running.get(d, host_stride[d])
        running[d] = stride_map[j] * device_size[j]
    return stride_map


def _normalise_rank(device_size, stride_map, phys_src, phys_op):
    """Insert a unit device axis where torch-spyre's DMA setup needs one.

    **This is not "pad to rank 3"**, and the difference is the whole hazard.
    ``get_dim_map`` (``spyre_mem.cpp``) computes

        stick_dim_index = device_rank > 2 ? device_rank - 3 : 0

    and then, having matched each device axis to a host dim by a greedy stride
    scan, **forcibly overwrites that one entry**::

        if (dim_map[stick_dim_index] != -1)
          dim_map[stick_dim_index] = dim_map[device_rank - 1];

    The assumption is that the axis at ``device_rank - 3`` is the tile half of the
    same host dim as the innermost axis — true of every layout their own
    constructor builds, where ``get_generic_stick_layout`` duplicates the stick
    host dim into exactly those two positions. It is not true of a splat layout,
    where the innermost axis addresses no host dim at all: feed the rank-2
    ``device_size [64, 64]`` / ``stride_map [1, -1]`` in unnormalized and the scan
    finds ``dim_map = [0, -1]``, the forcing turns it into ``[-1, -1]``, and every
    host dim is then skipped downstream — ``dcsi_sizes`` stays all ones and the
    DMA moves **one element instead of 64**, silently, with no check firing.

    So the rule is about making that forced assignment harmless, and there are
    three ways it already is:

      1. ``stride_map[p] == -1`` — the scan never matches an axis with a negative
         stride, so ``dim_map[p]`` is already ``-1`` and the forcing is guarded out;
      2. ``device_size[p] == 1`` — the scan skips unit axes, same conclusion;
      3. ``p`` really is the floordiv half of the same logical dim the innermost
         axis takes modulo — the canonical stick split, where the forcing assigns
         the value that was already there.

    Otherwise a unit axis goes in, and it goes in at ``device_rank - 2`` rather
    than at ``p``: inserting shifts every later index by one, so the position that
    must end up holding the unit axis is the *new* ``stick_dim_index``,
    ``(device_rank + 1) - 3``. Putting it at the old ``p`` would leave the new
    ``stick_dim_index`` pointing at a real axis and change nothing.

    Rank 1 needs nothing: ``stick_dim_index`` and ``device_rank - 1`` are both 0
    there, so the forcing is a self-assignment.

    Worth raising upstream. Their assumption has no test on their side, and
    nothing in the layout they are handed lets them detect its violation.
    """
    rank = len(device_size)
    if rank < 2:
        return list(device_size), list(stride_map)

    p = rank - 3 if rank > 2 else 0
    last = rank - 1
    harmless = (
        stride_map[p] == -1
        or device_size[p] == 1
        or (phys_src[p] == phys_src[last]
            and phys_op[p] == FLOORDIV and phys_op[last] == MOD)
    )
    if harmless:
        return list(device_size), list(stride_map)

    at = rank - 2
    return (list(device_size[:at]) + [1] + list(device_size[at:]),
            list(stride_map[:at]) + [-1] + list(stride_map[at:]))


# ---------------------------------------------------------------------------
# The consumers
#
# Everything below needs torch-spyre, so the import is per-call rather than at
# module scope: this module is imported by backend/compiler.py, which Triton
# imports on every machine with the wheel, including ones with no device.
# ---------------------------------------------------------------------------

def as_spyre_tensor_layout(entry: dict, torch_dtype):
    """*entry* as a ``SpyreTensorLayout`` for a tensor of *torch_dtype*.

    The device format comes from ``get_device_dtype``, which is why no dtype table
    lives here: it is the same function torch-spyre's own allocator uses, so a
    dtype it supports is one this supports.
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
        "the layout replicates. Allocate it from the compiled kernel's own claim "
        "instead:\n"
        "    from triton.backends.spyre.tensor_layout import "
        "empty_with_device_layout\n"
        f"    {name} = empty_with_device_layout(entry, host_{name})\n"
        "  where `entry` is the matching element of "
        'kernel.metadata.device_layouts.')


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
    """
    return {int(e["ptr_index"]): e for e in (entries or [])}
