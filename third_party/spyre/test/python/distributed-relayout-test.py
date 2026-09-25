# RUN: %python -m pytest %s -q
"""A relayout, from Python to KTIR: `tl.spyre_pin` and `tl.make_distributed_descriptor`.

The two ops only mean something together, and this file is where that is checked.
The lit tests beside it drive each pass on hand-written IR, which is what keeps a
diagnostic honest; none of them shows that a kernel an author could actually write
reaches the composed form, because each starts from the shape the previous pass was
supposed to produce.

The kernel is the smallest thing that is a real redistribution:

    share = exp(x)                                  produced on-chip
    pin(share, ct_local, 0)                         where my share lives
    whole = make_distributed_descriptor(share, ...)  every core's share, composed
    mine  = whole.load([0, pid * N])                 my region under the new division
    pin(mine, ct_local, 8192)                        where the received tile lands

Both pins are load-bearing and for different reasons, which is the point of testing
them here rather than apart. The first is the only thing that gives a partition an
address — the compose refuses a share it cannot trace to a buffer. The second is the
landing the design requires before a compute unit may read a received tile, and it
is emitted by `PlacePinnedValues` rather than by the compose, so nothing but an
end-to-end run shows the two passes agreeing about it.

What is NOT checked here is numerical: this cannot reach a device. dbo's V1 legality
check rejects `construct_distributed_memory_view` outright, and a hand-written
`ct_local` memory view makes the scheduler abort — not diagnose — on a missing
`applicable_units` attribute. So the artifact is read instead, at the `ktir` stage,
which is also where `PLAN-DistDesc.md` puts this layer's acceptance.

Three claims, and they are separable.

**The partitions differ only in coordinate_set and ct_id.** Design section 3 says
those two attributes are "the entire content of the distribution", so the offsets
being EQUAL across partitions is as much the claim as the sets being different: every
core runs the same program text, so a share sits at the same offset in its own core's
scratchpad as every other share does in its. A lowering that made the offsets differ
would look plausible and address the wrong memory on 31 cores out of 32.

**The composed extent is the union, not a share.** The share is [64, 32] and the
composition of two of them is [64, 64]. That number appears nowhere in the kernel —
the slice count is derived as one more than the largest index in the table — so it is
the derivation being checked and not a transcription.

**The read is per-instance.** The access tile's offset is `pid * N`, which is what
makes this a redistribution rather than 32 copies of the same read. It stays an SSA
value through the lowering, because `construct_access_tile` takes runtime base
indices.
"""

import pytest

SIGNATURE = {
    "x_ptr": "*fp16",
    "out_ptr": "*fp16",
    "M": "constexpr",
    "N": "constexpr",
    "WORK_SLICES": "constexpr",
    "AXES": "constexpr",
}

#: Two cores, the work divided on the last dimension. The table is the PARTITION
#: table -- one entry per region -- and here it has one entry per tile, which is the
#: only length the lowering admits so far: the holder is then the entry's own index.
CONSTANTS = {
    "M": 64,
    "N": 32,
    "WORK_SLICES": [{"n": 0}, {"n": 1}],
    "AXES": [None, "n"],
}

GRID = [2]

#: Element indices, not byte addresses: what `construct_memory_view`'s offset feeds.
SHARE_ADDRESS = 0
LANDING_ADDRESS = 8192


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def relayout(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                 WORK_SLICES: tl.constexpr, AXES: tl.constexpr):
        pid = tl.program_id(0)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
        out_desc = tl.make_tensor_descriptor(
            out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])

        share = tl.exp(x_desc.load([0, 0]))
        tl.spyre_pin(share, "ct_local", address=0)

        whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES, [M, N])
        mine = whole.load([0, pid * N])
        tl.spyre_pin(mine, "ct_local", address=8192)

        out_desc.store([0, 0], mine)

    return relayout


@pytest.fixture(scope="module")
def ktir():
    """The kernel through the whole TTIR -> KTIR pipeline, as printed IR."""
    import tempfile
    import os
    from utils import compile_to_ttir, make_ktir_mod

    ttir = compile_to_ttir(_kernel(), SIGNATURE, CONSTANTS)
    fd, path = tempfile.mkstemp(suffix=".ttir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(ttir)
        return str(make_ktir_mod(path, grid=GRID))
    finally:
        os.unlink(path)


def _views(ktir, needle):
    return [line.strip() for line in ktir.splitlines()
            if "ktdp.construct_memory_view" in line and needle in line]


class TestTheComposedView:

    def test_one_partition_per_tile(self, ktir):
        # Two cores, two partitions, each naming a different core's scratchpad.
        assert len(_views(ktir, "ct_id = 0")) == 1
        assert len(_views(ktir, "ct_id = 1")) == 1

    def test_the_partitions_share_one_offset(self, ktir):
        # The claim that matters most, and the one a wrong lowering would pass every
        # other check with. Compared as the offset TOKEN rather than a number,
        # because the value is an SSA name.
        offsets = [line.split("construct_memory_view ")[1].split(",")[0]
                   for line in _views(ktir, "ct_id =")]
        assert len(set(offsets)) == 1, offsets

    def test_the_partitions_differ_only_in_set_and_ct_id(self, ktir):
        # Everything except the two distinguishing attributes must be identical.
        # Normalised by naming exactly what may differ -- the coordinate-set
        # reference and the ct_id -- rather than by a pattern over the line, so a
        # future difference shows up as a failure instead of being absorbed.
        def strip(line, ct_id):
            body = line.split(" = ", 1)[1]        # drop the result's SSA name
            body = body.replace(f"ct_id = {ct_id}", "ct_id = _")
            head, _, rest = body.partition("coordinate_set = ")
            _, _, tail = rest.partition(",")
            return head + "coordinate_set = _," + tail

        a, b = _views(ktir, "ct_id = 0")[0], _views(ktir, "ct_id = 1")[0]
        assert strip(a, 0) == strip(b, 1)
        # And they really do differ before normalising, or the assertion above
        # proves nothing.
        assert a != b

    def test_the_two_regions_tile_the_whole(self, ktir):
        # d1 in [0, 32) and d1 in [32, 64): the slice width is the share's extent and
        # the coordinate picks which slice. The `- 32` is what says the second
        # partition owns the upper half.
        sets = {}
        for line in ktir.splitlines():
            if line.strip().startswith("#set"):
                name, body = line.split(" = ", 1)
                sets[name.strip()] = body
        bodies = " ".join(sets.values())
        assert "d1 >= 0" in bodies and "-d1 + 31 >= 0" in bodies
        assert "d1 - 32 >= 0" in bodies and "-d1 + 63 >= 0" in bodies

    def test_the_composed_extent_is_the_union(self, ktir):
        # [64, 32] composed twice is [64, 64], and 64 appears in no kernel line: the
        # slice count is derived from the table.
        assert "ktdp.construct_distributed_memory_view" in ktir
        compose = [line for line in ktir.splitlines()
                   if "construct_distributed_memory_view" in line][0]
        assert ": memref<64x64xf16>" in compose

    def test_the_marker_is_gone(self, ktir):
        assert "tts.make_distributed_descriptor" not in ktir
        assert "tt.descriptor_load" not in ktir


class TestTheTransfer:

    def test_the_read_is_on_the_composed_view(self, ktir):
        compose = [line for line in ktir.splitlines()
                   if "construct_distributed_memory_view" in line][0]
        name = compose.split(" =")[0].strip()
        tiles = [line for line in ktir.splitlines()
                 if "construct_access_tile" in line and f"{name}[" in line]
        assert len(tiles) == 1, tiles
        # memref<64x64xf16> in, a 64x32 tile out: the instance takes its own block
        # from the whole.
        assert "memref<64x64xf16> -> !ktdp.access_tile<64x32xindex>" in tiles[0]

    def test_the_offset_is_per_instance(self, ktir):
        # pid * N survives as arithmetic on the compute-tile id, so the 32 instances
        # read 32 different regions of one view. A folded constant here would mean
        # every core read the same thing.
        assert "ktdp.get_compute_tile_id" in ktir
        assert "arith.muli" in ktir


class TestTheTwoPins:

    def test_the_share_is_stored_before_it_is_composed(self, ktir):
        # Phase 1 needs the share resident: the compose names each core's buffer, so
        # something has to have written it. That is the first pin's store, and it must
        # come before the compose.
        lines = ktir.splitlines()
        store = next(i for i, l in enumerate(lines) if "ktdp.store" in l)
        compose = next(i for i, l in enumerate(lines)
                       if "construct_distributed_memory_view" in l)
        assert store < compose

    def test_the_received_tile_lands(self, ktir):
        # The mandatory landing of design section 3 phase 3, and it is the SECOND
        # pin's store rather than anything the compose emitted -- which is why an
        # end-to-end test is the only place the two passes can be seen agreeing.
        lines = ktir.splitlines()
        compose = next(i for i, l in enumerate(lines)
                       if "construct_distributed_memory_view" in l)
        transfer = next(i for i, l in enumerate(lines)
                        if i > compose and "ktdp.load" in l)
        assert any("ktdp.store" in l for l in lines[transfer:]), \
            "the received tile was never stored"

    def test_the_landing_is_its_own_buffer(self, ktir):
        # Two ct_local views with no ct_id: the share's and the landing's. They must be
        # distinct buffers, or the relayout would overwrite its own source.
        own = _views(ktir, "memory_space = #ktdp.memory_space<ct_local>}")
        assert len(own) == 2, own
        offsets = [line.split("construct_memory_view ")[1].split(",")[0] for line in own]
        assert len(set(offsets)) == 2, offsets
