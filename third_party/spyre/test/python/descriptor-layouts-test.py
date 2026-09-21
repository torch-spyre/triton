# RUN: %python -m pytest %s -q
"""``metadata["device_layouts"]``: what the compile claims a buffer needs on device.

A descriptor annotated with ``tl.spyre_tensor_layout`` can ask for a device layout
that occupies more elements than its host tensor holds — a splat physical dim is the
case. Nothing could see that before: the compiler knows the physical extent, the
allocator sizes from the host shape, and the launcher checks only device residency.
This key is the compiler's number written down, and this file is what pins its shape.

Four claims, and they are separable.

**The extents are the coordinate map's, not a transcription of it.** They come from
``tts::applyCoordMap`` through a pybind query, the same evaluator
``rewrite-descriptor-layout-generic`` builds the physical memref with, over the same
logical extents (``getDescriptorLogicalLayout``, shared with
``lower-descriptor-memory``). So what is checked here is the *reduction* — that a
floordiv rounds up and a splat contributes its own width — rather than the
arithmetic, which has one implementation and cannot disagree with itself.

**The key is the pointer ordinal.** Not the parameter name, which would be the better
key and is not available: Triton records no argument names in the IR and a compile
stage is handed ``(module, metadata)``, never the source. Ordinal *i* is the *i*-th
``!tt.ptr`` argument, which is what ``_segment_addresses`` and ``_address_args``
already agree on, so a test that pinned names would be pinning something the artifact
cannot carry.

**The rank normalization happens, and it is not padding to rank 3.** This is the
subtle one and the reason the key exists in a torch-spyre convention rather than the
coordinate map's own. ``get_dim_map`` (``spyre_mem.cpp``) takes
``stick_dim_index = device_rank > 2 ? device_rank - 3 : 0`` and then *overwrites* that
entry with the host dim it found for the innermost axis — sound for every layout their
own constructor builds, where the stick host dim occupies exactly those two positions,
and wrong for a splat, whose innermost axis addresses no host dim. Unnormalized, the
rank-2 splat layout comes out of that function as all ``-1`` and the DMA moves one
element instead of 64, silently. So a unit axis goes in, at ``device_rank - 2`` so
that it lands on the *new* ``stick_dim_index`` — and the canonical stick split, which
already satisfies the assumption, must NOT get one. Both directions are asserted,
because a rule that fired everywhere would be padding and a rule that fired nowhere
would be absent.

**An unknowable footprint is ``None``, and an unannotated descriptor is absent.** Two
different things: a kernel taking its shape from a runtime ``i32`` has a footprint no
compile can know, while an unannotated descriptor made no claim at all. Both leave the
launcher's check inert, and only the first says so.

The hyphen in the filename is load-bearing — pytest collects ``test_*.py`` and
``*_test.py``, so an underscore would have the pytest suite re-run this alongside lit.
"""

import pytest

import triton
import triton.language as tl

from utils import compile_to_ttir, make_ktir_mod

#: Lanes to an fp16 stick. Spelled once; every layout below is stated against it.
S = 64


@triton.jit
def reduce_to_stick(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                    IN_LAYOUT: tl.constexpr, OUT_LAYOUT: tl.constexpr):
    """A row reduce whose two descriptors are annotated independently.

    Deliberately a reduce and not an elementwise op: the output is rank-1 while the
    input is rank-2, so the two claims cannot be confused for each other, and a
    surviving extent is what a splat output has to hold a whole stick of.
    """
    src = tl.make_tensor_descriptor(in_ptr, shape=[M, N], strides=[N, 1],
                                   block_shape=[M, N])
    tl.spyre_tensor_layout(src, IN_LAYOUT)
    dst = tl.make_tensor_descriptor(out_ptr, shape=[M], strides=[1],
                                    block_shape=[M])
    tl.spyre_tensor_layout(dst, OUT_LAYOUT)
    dst.store([0], tl.sum(src.load([0, 0]), axis=1))


@triton.jit
def unannotated(in_ptr, out_ptr, M: tl.constexpr):
    """The same shape with no annotation anywhere."""
    src = tl.make_tensor_descriptor(in_ptr, shape=[M], strides=[1],
                                    block_shape=[M])
    dst = tl.make_tensor_descriptor(out_ptr, shape=[M], strides=[1],
                                    block_shape=[M])
    dst.store([0], src.load([0]))


@triton.jit
def dynamic_extent(in_ptr, out_ptr, n, BLOCK: tl.constexpr,
                   LAYOUT: tl.constexpr):
    """A descriptor whose extent is a runtime argument, so no footprint is knowable.

    ``BLOCK`` is a constexpr argument rather than the module-level stick width,
    because a ``@triton.jit`` body cannot read a plain global.
    """
    src = tl.make_tensor_descriptor(in_ptr, shape=[n], strides=[1],
                                    block_shape=[BLOCK])
    tl.spyre_tensor_layout(src, LAYOUT)
    dst = tl.make_tensor_descriptor(out_ptr, shape=[n], strides=[1],
                                    block_shape=[BLOCK])
    dst.store([0], src.load([0]))


#: ``[M]`` -> ``[ceil(M/S), S]``: the statistic SPLIT across a stick. The canonical
#: form, and the one torch-spyre's own constructor produces.
SPLIT_1D = ((0, "floordiv", S), (0, "mod", S))

#: ``[M]`` -> ``[M, S]``: the statistic REPLICATED across a stick. What a stick-axis
#: reduce has to store, and what no host-shaped allocation is big enough for.
SPLAT_1D = (0, (0, "splat", S))

#: ``[M, N]`` -> ``[ceil(N/S), M, S]``: stick on the reduced axis.
STICK_ON_N = ((1, "floordiv", S), 0, (1, "mod", S))


def capture(tmp_path, kernel_fn, signature, constexprs):
    """``metadata["device_layouts"]`` for one kernel, through the real ``ktir`` stage.

    Through ``make_ktir_mod`` rather than by calling the pybind query directly, so
    what is under test is the whole capture: the walk, the normalization, and the
    placement of the read before the pipeline that erases the markers.
    """
    path = tmp_path / "kernel.ttir"
    path.write_text(compile_to_ttir(kernel_fn, signature, constexprs))
    metadata = {}
    make_ktir_mod(path, grid=(1, ), metadata=metadata)
    return metadata["device_layouts"]


def test_one_entry_per_annotated_descriptor(tmp_path):
    """Two annotations, two entries, keyed by pointer ordinal in argument order."""
    entries = capture(tmp_path, reduce_to_stick,
                      {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                      {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                       "OUT_LAYOUT": SPLIT_1D})
    assert [e["ptr_index"] for e in entries] == [0, 1]
    assert [e["access"] for e in entries] == ["load", "store"]


def test_split_output_is_not_rank_normalized(tmp_path):
    """The canonical stick split already satisfies ``get_dim_map``'s assumption.

    ``[64]`` -> ``[1, 64]``: the axis at ``stick_dim_index`` IS the floordiv half of
    the same logical dim the last axis takes modulo, so the forced assignment writes
    back the value already there and no unit axis is needed. Asserting this is what
    stops the normalization from being padding.
    """
    _, out = capture(tmp_path, reduce_to_stick,
                     {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                     {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                      "OUT_LAYOUT": SPLIT_1D})
    assert out["device_size"] == [1, S]
    assert out["stride_map"] == [S, 1]


def test_splat_output_gains_a_unit_axis(tmp_path):
    """The splat layout, which is the whole reason this key exists.

    The coordinate map says ``[64] -> [64, 64]``. Handed to ``get_dim_map`` at rank 2
    that is destroyed — the greedy scan finds ``dim_map = [0, -1]`` and the forced
    assignment turns it into ``[-1, -1]`` — so the recorded form is rank 3 with the
    unit axis at ``device_rank - 2`` of the original, i.e. index 0.

    ``[1, 64, 64]`` / ``[-1, 1, -1]`` is also exactly what the fixture harness used to
    hand-write beside ``OUT_LAYOUT``, with nothing deriving one from the other. This
    is the derivation.
    """
    _, out = capture(tmp_path, reduce_to_stick,
                     {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                     {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                      "OUT_LAYOUT": SPLAT_1D})
    assert out["device_size"] == [1, S, S]
    assert out["stride_map"] == [-1, 1, -1]


def test_input_extents_round_a_partial_stick_up(tmp_path):
    """A floordiv coordinate is a CEILING divide, and a ragged extent shows it.

    ``N = 129`` over a 64-lane stick is three sticks, not two: the coordinate of an
    element is ``i floordiv 64`` and there are three distinct such coordinates. A
    floor here would under-claim by a whole stick, which is the direction that does
    not fail loudly.
    """
    src, _ = capture(tmp_path, reduce_to_stick,
                     {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                     {"M": 64, "N": 129, "IN_LAYOUT": STICK_ON_N,
                      "OUT_LAYOUT": SPLIT_1D})
    assert src["device_size"] == [3, 64, S]
    assert src["stride_map"] == [S, 129, 1]


def test_no_annotation_means_no_entry(tmp_path):
    """An unannotated kernel makes no claim, and says so by absence.

    Not an empty-ish entry per pointer: the launcher's rule is that a missing entry
    is silence rather than a fault, and every kernel without a layout annotation
    relies on it.
    """
    assert capture(tmp_path, unannotated, {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                   {"M": 64}) == []


def test_a_runtime_extent_records_a_null_footprint(tmp_path):
    """A shape from a runtime argument is recorded, as ``None``, not dropped.

    The distinction the launcher reads: absent means no claim was made, ``None`` means
    a claim was made and cannot be evaluated at compile time. Both leave the check
    inert; only one of them is a thing a diagnostic can mention.
    """
    entries = capture(tmp_path, dynamic_extent,
                      {"in_ptr": "*fp16", "out_ptr": "*fp16", "n": "i32"},
                      {"BLOCK": S, "LAYOUT": SPLIT_1D})
    assert len(entries) == 1
    assert entries[0]["ptr_index"] == 0
    assert entries[0]["device_size"] is None
    assert entries[0]["stride_map"] is None


def test_the_key_survives_the_metadata_json_round_trip(tmp_path):
    """``json.dumps(metadata, default=vars)`` is what the cache stores.

    Tuples come back as lists and integer dict keys come back as strings, which is
    why this is a ``list`` of dicts keyed by an ``int`` *value* rather than a dict
    keyed by the ordinal. ``None`` survives, which the null footprint above relies on.
    """
    import json
    entries = capture(tmp_path, reduce_to_stick,
                      {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                      {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                       "OUT_LAYOUT": SPLAT_1D})
    assert json.loads(json.dumps(entries, default=vars)) == entries


def test_lookup_is_by_ptr_index_not_by_position(tmp_path):
    """``entries_by_ptr_index`` re-keys the list, tolerating the round trip.

    The launcher looks a claim up by the ordinal of the pointer it has just collected,
    so the mapping has to be by the recorded value and not by the entry's position:
    a kernel whose first pointer is unannotated has no entry at position 0.
    """
    import json
    from backend.tensor_layout import entries_by_ptr_index
    entries = capture(tmp_path, reduce_to_stick,
                      {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                      {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                       "OUT_LAYOUT": SPLAT_1D})
    revived = json.loads(json.dumps(entries, default=vars))
    assert sorted(entries_by_ptr_index(revived)) == [0, 1]
    assert entries_by_ptr_index(None) == {}


@pytest.mark.parametrize("entries", [None, []])
def test_an_absent_key_is_not_a_fault(entries):
    """Nothing recorded means no claim, for a kernel compiled before the key existed
    or one entered at the ``.ktir`` stage, where the markers are already gone."""
    from backend.tensor_layout import entries_by_ptr_index
    assert entries_by_ptr_index(entries) == {}


# ---------------------------------------------------------------------------
# The padding convention, held to torch-spyre's own generated wrappers
#
# The unit-axis padding is a convention we MATCH, not one we invented, and that is
# what this section is for. Every test above it pins the padding to our own
# reasoning about what ``get_dim_map`` needs -- sound reasoning, but a closed loop:
# our evaluator and our reading of their DMA are the only two parties to it, and
# they agree by construction. There is an outside witness, though. torch-spyre's
# own SDSC codegen states device layouts explicitly, the same way we do, and the
# layouts it emits satisfy the same rule. So the rule is checkable against
# something other than ourselves, which is the whole point of the comparison.
#
# Checked as a PROPERTY, not by parsing their wrappers. The wrapper files are
# generated artifacts outside this repository (one of them is
# ``gemma4-route-scalar-relayout-s2-20260902/accepted/generated.py`` under the
# dataflow-test-framework working tree), so a test that read them would fail on
# any machine without that tree -- which is every CI machine -- and would be
# pinned to the spelling of a generated file nobody promises to keep. Their two
# distinct layouts are recorded below as literals instead, and go through the
# same predicate as ours: what is shared is the rule, and a rule both sides
# satisfy is what the comparison was for.
# ---------------------------------------------------------------------------

#: The two distinct explicit ``SpyreTensorLayout``s torch-spyre's SDSC codegen
#: emits across the wrapper files on hand, with the host tensor each was built
#: for. They satisfy ``stick_axis_is_harmless`` in the two different ways it
#: allows, which is why both are here rather than one:
#:
#: * the rank-4 one for host ``(1, 2816, 512)`` has the TILE HALF third from the
#:   end (8 sticks of 64 over the innermost host dim) and its unit axis -- host
#:   dim 0, extent 1 -- elsewhere, carrying ``-1``;
#: * the rank-3 one for host ``(512, 2816)`` is the canonical stick split, tile
#:   half third from the end again.
#:
#: Ours satisfy it the third way, with a unit axis third from the end. The
#: predicate is what the three have in common; none of the three positions is.
WRAPPER_LAYOUTS = [
    ([2816, 8, 1, 64], [512, 64, -1, 1]),
    ([44, 512, 64], [64, 2816, 1]),
]


def stick_axis_is_harmless(device_size, stride_map):
    """Will torch-spyre's DMA move this layout's whole extent, or one element of it?

    The DMA reads the axis **third from the end** as the tile-count half of the
    stick split, whose lanes are the **last** axis, and overwrites what it matched
    there with the last axis's host dim (``get_dim_map``, ``spyre_mem.cpp``). If
    that axis is something else, a real dimension gets clobbered and the transfer
    silently shrinks to one element. So a usable device layout has to make that one
    assignment harmless, and this is the predicate for it — the single rule the
    unit-axis padding exists to satisfy, and the one both our layouts and
    torch-spyre's own can be held to.

    Harmless three ways: the third-from-last axis is skipped by their scan (stride
    ``-1``, or extent 1), or it genuinely is the tile half of the same host dim as
    the last axis, so the value written back is the one already there.

    That third clause is stated in LAYOUT terms rather than the coordinate map's,
    deliberately: the tile half of the dim the last axis takes modulo advances by
    one whole stick of it, ``stride_map[-1] * device_size[-1]``. That is all a
    metadata consumer can see — nothing downstream of the compile has the
    coordinate map — and it is what lets the same predicate judge a torch-spyre
    wrapper's layout, which never had one.
    """
    rank = len(device_size)
    if rank < 2:
        return True  # The two positions coincide; the assignment is a self-assignment.
    p = rank - 3 if rank > 2 else 0
    return (stride_map[p] == -1 or device_size[p] == 1
            or stride_map[p] == stride_map[-1] * device_size[-1])


@pytest.mark.parametrize("device_size, stride_map", WRAPPER_LAYOUTS)
def test_the_wrappers_own_layouts_satisfy_the_convention(device_size, stride_map):
    """The claim this section rests on: it is their rule, not just our reading.

    If this fails, the predicate below is measuring something torch-spyre does not
    do, and the padding it justifies needs re-deriving -- not the emissions.
    """
    assert stick_axis_is_harmless(device_size, stride_map)


def test_the_convention_is_discriminating():
    """The unpadded rank-2 splat FAILS it, which is what makes the rest worth asserting.

    ``[64, 64]`` / ``[1, -1]`` is the coordinate map's own answer for a splat
    output, before the unit axis goes in. At rank 2 the axis third from the end is
    the first one, it addresses host dim 0 with a real stride, and the last axis
    addresses no host dim at all — so the DMA's forced assignment destroys the map.
    A predicate every layout satisfied would say nothing about the padding.
    """
    assert not stick_axis_is_harmless([64, 64], [1, -1])


#: ``[M, N]`` -> ``[ceil(M/8), 8, ceil(N/S), S]``: both logical dims split, so the
#: axis third from the end is the mod half of the OUTER dim while the last is the
#: lane of the inner one. Neither exemption applies and the tile-half test fails, so
#: this is where the padding fires at a rank the splat case cannot reach.
DOUBLE_SPLIT_2D = ((0, "floordiv", 8), (0, "mod", 8), (1, "floordiv", S),
                   (1, "mod", S))


@pytest.mark.parametrize("in_layout, out_layout", [
    (STICK_ON_N, SPLIT_1D),
    (STICK_ON_N, SPLAT_1D),
    (DOUBLE_SPLIT_2D, SPLIT_1D),
    (DOUBLE_SPLIT_2D, SPLAT_1D),
])
def test_every_emitted_pair_satisfies_the_convention(tmp_path, in_layout,
                                                     out_layout):
    """What the evaluator emits is held to the rule the wrappers are held to.

    Four combinations rather than one because the padding decision is per layout
    and its three exemptions are reached by different shapes: the split output
    takes the tile-half exemption, the splat output is padded at rank 2, and
    ``DOUBLE_SPLIT_2D`` is padded at rank 4 -- the rank where the axis inserted and
    the axis being neutralized stop being the same one.
    """
    entries = capture(tmp_path, reduce_to_stick,
                      {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                      {"M": 64, "N": 128, "IN_LAYOUT": in_layout,
                       "OUT_LAYOUT": out_layout})
    assert entries
    for entry in entries:
        assert stick_axis_is_harmless(entry["device_size"],
                                      entry["stride_map"]), entry


@pytest.mark.parametrize("which, in_layout", [
    # The splat output, padded at rank 2 -- entry 1, the `out_ptr` claim.
    (1, STICK_ON_N),
    # The double-split input, padded at rank 4 -- entry 0, where the axis inserted
    # and the axis being neutralized are not the same one.
    (0, DOUBLE_SPLIT_2D),
])
def test_an_inserted_axis_is_spelled_the_way_theirs_is(tmp_path, which,
                                                      in_layout):
    """A padded layout's unit axis carries extent 1 and ``stride_map`` ``-1`` together.

    The same spelling the wrappers use for an axis the host tensor does not
    address (their ``device_size=[2816, 8, 1, 64]`` / ``stride_map=[512, 64, -1,
    1]``). Asserted separately from the predicate above because the predicate is
    satisfied by extent 1 ALONE: a unit axis emitted with a real stride would pass
    it and still be a second spelling of the same thing.

    Both padded ranks are covered, because the insertion goes second from the end
    of the unpadded layout, which is the axis being neutralized only at rank 2.
    """
    entry = capture(tmp_path, reduce_to_stick,
                    {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                    {"M": 64, "N": 128, "IN_LAYOUT": in_layout,
                     "OUT_LAYOUT": SPLAT_1D})[which]
    p = len(entry["device_size"]) - 3
    assert (entry["device_size"][p], entry["stride_map"][p]) == (1, -1), entry


# ---------------------------------------------------------------------------
# The check the launcher makes
#
# Byte figures come from torch-spyre's own ``get_device_size_in_bytes``, so these
# need ``torch_spyre`` importable -- but not a device: constructing a
# ``SpyreTensorLayout`` and sizing it are pure arithmetic in spyre_tensor_impl.cpp,
# and nothing here allocates. The tensor is a stub for the same reason: a real one
# would open the device, which this file has no business doing and lit could not
# serialize if it did.
# ---------------------------------------------------------------------------

class FakeTensor:
    """Just the two members ``check_fits`` reads off a launch argument."""

    def __init__(self, dtype, layout):
        self.dtype = dtype
        self._layout = layout

    def device_tensor_layout(self):
        return self._layout


def torch_bits():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_spyre")  # must follow torch; see driver.py
    from torch_spyre._C import SpyreTensorLayout
    return torch, SpyreTensorLayout


def test_the_splat_claim_is_64x_what_a_host_shaped_allocation_gives(tmp_path):
    """The measurement the whole key exists for.

    An ``[64]`` fp16 statistic under a splat layout needs 8192 bytes; the layout
    ``.to("spyre")`` builds for a ``[64]`` fp16 tensor is worth 128. Both numbers come
    from ``get_device_size_in_bytes``, so the factor of 64 is torch-spyre's own
    arithmetic disagreeing with itself across two descriptions of one buffer -- which
    is why neither party could see it alone.
    """
    torch, SpyreTensorLayout = torch_bits()
    from backend.tensor_layout import device_bytes

    _, out = capture(tmp_path, reduce_to_stick,
                     {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                     {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                      "OUT_LAYOUT": SPLAT_1D})
    from torch_spyre._C import get_device_size_in_bytes
    need = device_bytes(out, torch.float16)
    have = get_device_size_in_bytes(SpyreTensorLayout([64], torch.float16))
    assert (need, have) == (8192, 128)


def test_check_fits_refuses_a_host_shaped_allocation(tmp_path):
    """The refusal, and what it has to say.

    Named in the message: the argument, both device sizes, both byte counts, what the
    kernel would do out of bounds, and the call that fixes it. A message missing any
    of those leaves the reader with a number and no way to act on it -- the state this
    replaces, where there was no message at all.
    """
    torch, SpyreTensorLayout = torch_bits()
    from backend.tensor_layout import check_fits

    _, out = capture(tmp_path, reduce_to_stick,
                     {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                     {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                      "OUT_LAYOUT": SPLAT_1D})
    host_shaped = FakeTensor(torch.float16, SpyreTensorLayout([64], torch.float16))
    with pytest.raises(RuntimeError) as excinfo:
        check_fits(out, "out_ptr", host_shaped)
    message = str(excinfo.value)
    # The suggested call is copy-pasteable: the declared numbers are interpolated
    # into it, so a reader fixes the allocation without looking anything up.
    for fragment in ("out_ptr", "[1, 64, 64]", "[-1, 1, -1]", "8192 bytes",
                     "128 bytes", "write past the end",
                     'device_layout=SpyreTensorLayout(',
                     "device_size=[1, 64, 64]", "_lazy_init"):
        assert fragment in message, f"missing {fragment!r} from:\n{message}"


def test_check_fits_accepts_the_layout_it_asked_for(tmp_path):
    """No false alarm on the allocation the claim itself describes.

    The direction that matters for adoption: a check that fired on a correctly
    allocated buffer would be worse than no check, because the workaround would be to
    stop passing layouts.
    """
    torch, _ = torch_bits()
    from backend.tensor_layout import as_spyre_tensor_layout, check_fits

    _, out = capture(tmp_path, reduce_to_stick,
                     {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                     {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                      "OUT_LAYOUT": SPLAT_1D})
    exact = FakeTensor(torch.float16, as_spyre_tensor_layout(out, torch.float16))
    check_fits(out, "out_ptr", exact)


def test_check_fits_accepts_an_over_allocation(tmp_path):
    """``have > need`` passes, deliberately.

    An over-allocated buffer wastes device memory and corrupts nothing, and a tensor
    sliced from a larger one is a normal thing to hand a kernel.
    """
    torch, SpyreTensorLayout = torch_bits()
    from backend.tensor_layout import check_fits
    from torch_spyre._C import get_device_dtype

    _, out = capture(tmp_path, reduce_to_stick,
                     {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                     {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                      "OUT_LAYOUT": SPLAT_1D})
    roomy = FakeTensor(torch.float16, SpyreTensorLayout(
        device_size=[2, S, S], stride_map=[-1, 1, -1],
        device_dtype=get_device_dtype(torch.float16)))
    check_fits(out, "out_ptr", roomy)


class FakeSource:
    """Just ``src.signature``, which is all ``_address_args`` reads off the source."""

    def __init__(self, signature):
        self.signature = signature


class FakeMetadata:
    """``metadata.device_layouts`` as attribute access, the way KernelMetadata
    presents the JSON the compile wrote."""

    def __init__(self, device_layouts):
        self.device_layouts = device_layouts


def fake_device_tensor(dtype, layout):
    """A stub that passes ``_address_args``' Spyre-tensor check.

    A real tensor would open the device, which this file must not do -- lit runs one
    process per file in parallel and a Spyre device admits one opener per process
    for that process's whole life.
    """
    import types
    tensor = FakeTensor(dtype, layout)
    tensor.device = types.SimpleNamespace(type="spyre")
    return tensor


def test_the_launcher_checks_the_claim_it_was_compiled_with(tmp_path):
    """The wiring, not just the predicate.

    ``_address_args`` is where this has to happen: it is the one place that has both
    the pointer ordinal the claim is keyed by and the parameter name a diagnostic
    needs, and it already walks the signature in the order the artifact's segments
    and correction-vector slots were compiled in. A check anywhere else would have to
    reconstruct that pairing.

    Driven through ``_address_args`` directly rather than through a launch, because
    the launch is what needs hardware and the pairing is not.
    """
    torch, SpyreTensorLayout = torch_bits()
    from backend.driver import SpyreLauncher

    entries = capture(tmp_path, reduce_to_stick,
                      {"in_ptr": "*fp16", "out_ptr": "*fp16"},
                      {"M": 64, "N": 128, "IN_LAYOUT": STICK_ON_N,
                       "OUT_LAYOUT": SPLAT_1D})
    signature = {"in_ptr": "*fp16", "out_ptr": "*fp16", "M": "constexpr",
                 "N": "constexpr", "IN_LAYOUT": "constexpr",
                 "OUT_LAYOUT": "constexpr", "OP": "constexpr", "AXIS": "constexpr"}
    launcher = SpyreLauncher(FakeSource(signature), FakeMetadata(entries))

    from backend.tensor_layout import as_spyre_tensor_layout
    good_in = fake_device_tensor(
        torch.float16, as_spyre_tensor_layout(entries[0], torch.float16))
    host_shaped_out = fake_device_tensor(
        torch.float16, SpyreTensorLayout([64], torch.float16))
    args = (good_in, host_shaped_out, 64, 128, STICK_ON_N, SPLAT_1D, "sum", 1)

    with pytest.raises(RuntimeError, match="out_ptr"):
        launcher._address_args(args)

    # And the same launch with the derived allocation passes, returning the two
    # pointers in kernel order and dropping the constexprs.
    good_out = fake_device_tensor(
        torch.float16, as_spyre_tensor_layout(entries[1], torch.float16))
    assert launcher._address_args(
        (good_in, good_out) + args[2:]) == [good_in, good_out]


def test_the_launcher_makes_no_claim_without_the_key(tmp_path):
    """An artifact with no recorded layouts leaves every argument unchecked.

    The compatibility rule, and it has to hold for a kernel cached before this key
    existed as well as for every unannotated kernel.
    """
    torch, SpyreTensorLayout = torch_bits()
    from backend.driver import SpyreLauncher

    signature = {"in_ptr": "*fp16", "out_ptr": "*fp16"}
    tiny = fake_device_tensor(torch.float16, SpyreTensorLayout([1], torch.float16))
    for metadata in (FakeMetadata(None), FakeMetadata([])):
        launcher = SpyreLauncher(FakeSource(signature), metadata)
        assert launcher._address_args((tiny, tiny)) == [tiny, tiny]


def test_a_null_footprint_makes_the_check_inert(tmp_path):
    """A claim that could not be evaluated must not be turned into a refusal."""
    torch, SpyreTensorLayout = torch_bits()
    from backend.tensor_layout import check_fits, empty_with_device_layout

    entries = capture(tmp_path, dynamic_extent,
                      {"in_ptr": "*fp16", "out_ptr": "*fp16", "n": "i32"},
                      {"BLOCK": S, "LAYOUT": SPLIT_1D})
    tiny = FakeTensor(torch.float16, SpyreTensorLayout([1], torch.float16))
    check_fits(entries[0], "in_ptr", tiny)
    # Allocating from it, on the other hand, cannot be silently inert: there is no
    # size to allocate, so saying so is the only answer.
    with pytest.raises(ValueError, match="recorded no device layout"):
        empty_with_device_layout(entries[0], tiny)
