"""Softmax twice: once as the SDSC path schedules it, once fused.

Three kernels over one measured case, written to be diffed against each other.

  softmax_opspec       the six OpSpecs the SDSC path emits, every intermediate
                       materialized in the scratchpad exactly as that path does
  softmax_fused        the one pair that can fuse, fused: five groups
  softmax_fused_auto   the same, with the input's staging left to the compiler

One group of difference between the first two, and the reason it is only one, is the
point of the set: most of what the SDSC path does here is forced. The third differs
from the second by one line and asks a question rather than answering it -- whether
staging a twice-read input is the author's decision or the compiler's. See README.md.

DRAFT, written for discussion. It does not run and the suite does not discover it:
``tl.spyre_pin`` does not exist, ``tl.spyre_tensor_layout`` takes neither a memory
space nor an element arrangement yet, and there is no ``meta.py``, so
``conftest.py`` never imports this file.

  case          torch.nn.functional.softmax, [1, 64, 11, 2049] fp16
  from          op-level-tests/gpt-oss-20b/SDSC/
                torch.nn.functional.softmax-1x64x11x2049
  elements      1,442,496 -- the largest softmax in that corpus by 11x
  grid          [32];  d0 = 64 split 32 ways, so 2 rows of 64 per core
  per core      [2, 11, 2049] logical, [2, 11, 33, 64] physical, ~91 KiB
  reduction     the last axis, 2049

A **group** here is a maximal region of dataflow whose ops share one iteration space,
bounded by materializations. It lowers to one ``linalg.generic`` after elementwise
fusion, inside one function, which becomes one three-stage pipeline -- and on the SDSC
path, one SDSC. Note this admits several arithmetic ops in one group where they fuse,
which is why ``softmax_fused``'s second group holds two.

The leading extent-1 logical dim is squeezed, so the descriptors are rank 3 where the
record is rank 4. Nothing else about the geometry is changed: the physical form below
is the record's `device_size = [64, 11, 33, 1, 64]` without that dim.

Padding is not a target. 2049 occupies 33 sticks of 64, so 63 lanes per row are
unused; the stick is the unit of transfer, so that is a property of the hardware
rather than something a kernel can spend differently. Both kernels pay it equally.
"""

import triton
import triton.language as tl

# Logical [d0, d1, d2] with the stick on d2, stick size 64: physically
# [d0, d1, d2 // 64, d2 % 64] = [64, 11, 33, 64]. This is the record's device_size
# with its extent-1 dim dropped.
TILE = [0, 1, (2, "floordiv", 64), (2, "mod", 64)]


@triton.jit
def softmax_opspec(x_ptr, out_ptr):
    """The six OpSpecs the SDSC path emits, with every intermediate in scratchpad.

    identity -> max -> sub -> exp -> sum -> realdiv, which is what
    ``ir_post_fusion.txt`` and the six ``sdsc_N.json`` files carry. Each
    ``tl.spyre_pin`` is one of that path's scratchpad buffers, so the store and the
    reload between stages are written out rather than implied.

    Two of the five pins are load-bearing and three are not. Pinning ``x`` chooses
    where its two consumers read it from, and pinning ``d`` is what splits ``sub``
    from ``exp``; ``m`` and ``s`` are reduction outputs and would materialize with or
    without a pin, so theirs are written only to make the six buffers visible.
    """
    pid = tl.program_id(0)  # grid is [32]: 2 of the 64 rows per core

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[64, 11, 2049], strides=[22539, 2049, 1],
        block_shape=[2, 11, 2049],
    )
    tl.spyre_tensor_layout(x_desc, TILE, element_arrangements=["standard"] * 4)

    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[64, 11, 2049], strides=[22539, 2049, 1],
        block_shape=[2, 11, 2049],
    )
    tl.spyre_tensor_layout(out_desc, TILE, element_arrangements=["standard"] * 4)

    x = x_desc.load([pid * 2, 0, 0])

    # sdsc_0 -- identity. Stages the input in the scratchpad. Both `max` and `sub`
    # below need the whole tile and cannot share a group, because `max` eliminates the
    # last axis and so changes the iteration space. So the tile is read twice either
    # way, and this stage makes both reads local: one HBM read plus two scratchpad
    # reads, rather than two HBM reads.
    tl.spyre_pin(x, memory_space="ct_local")

    # sdsc_1 -- max, reducing the last axis. The result is [2, 11, 1] logically and
    # occupies a full stick per row physically, which is the 64x the record shows for
    # its buf0.
    m = tl.max(x, axis=2, keep_dims=True)
    tl.spyre_pin(m, memory_space="ct_local")

    # sdsc_2 -- sub. A full tile stored and reloaded, and the one materialization in
    # this kernel that nothing forces: `d` has a single consumer on the same iteration
    # space, so the pin is what keeps it from fusing into `exp`.
    d = x - m
    tl.spyre_pin(d, memory_space="ct_local")

    # sdsc_3 -- exp.
    e = tl.exp(d)
    tl.spyre_pin(e, memory_space="ct_local")

    # sdsc_4 -- sum, the second reduction. Same 64x as the max.
    s = tl.sum(e, axis=2, keep_dims=True)
    tl.spyre_pin(s, memory_space="ct_local")

    # sdsc_5 -- realdiv, straight to the output descriptor.
    out_desc.store([pid * 2, 0, 0], e / s)


@triton.jit
def softmax_fused(x_ptr, out_ptr):
    """The same softmax in five groups, with one scratchpad round-trip removed.

    One change: ``sub`` and ``exp`` are a pointwise chain on one iteration space, so
    with no pin between them they fuse into a single ``linalg.generic`` and ``x - m``
    never reaches memory.

    Everything else is the same, and deliberately so:

    - ``x`` is still staged, for the reason given in ``softmax_opspec``. Without the
      pin this kernel would read the whole tile from HBM twice, which is worse than
      what the SDSC path does rather than better.
    - both reductions are still boundaries, because each eliminates an axis.
    - ``e`` is still materialized, and needs no pin to be: ``sum`` and the divide both
      read it, and a value cannot cross a group boundary in a register. Recomputing
      ``exp`` in the divide's group is the alternative, and it costs more than storing
      it once.
    - ``m`` and ``s`` are still materialized, being reduction outputs.

    So five groups: stage, max, sub+exp, sum, divide.
    """
    pid = tl.program_id(0)  # grid is [32]: 2 of the 64 rows per core

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[64, 11, 2049], strides=[22539, 2049, 1],
        block_shape=[2, 11, 2049],
    )
    tl.spyre_tensor_layout(x_desc, TILE, element_arrangements=["standard"] * 4)

    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[64, 11, 2049], strides=[22539, 2049, 1],
        block_shape=[2, 11, 2049],
    )
    tl.spyre_tensor_layout(out_desc, TILE, element_arrangements=["standard"] * 4)

    x = x_desc.load([pid * 2, 0, 0])

    # group 1 -- stage the input, so both consumers read it locally.
    tl.spyre_pin(x, memory_space="ct_local")

    # group 2 -- max. A reduction, so a boundary whether or not anything says so.
    m = tl.max(x, axis=2, keep_dims=True)

    # group 3 -- sub and exp together. This is the whole difference from
    # softmax_opspec: no pin, so one generic, and the subtraction's result never
    # reaches memory.
    e = tl.exp(x - m)

    # group 4 -- sum. The second reduction, and the reason `e` is materialized.
    s = tl.sum(e, axis=2, keep_dims=True)

    # group 5 -- the divide, straight to the output descriptor.
    out_desc.store([pid * 2, 0, 0], e / s)


@triton.jit
def softmax_fused_auto(x_ptr, out_ptr):
    """``softmax_fused`` with the staging decision left to the compiler.

    One line different: no ``tl.spyre_pin`` on ``x``. Everything else, including the
    group count, is identical.

    What that changes is *where* the two consumers of ``x`` read it from, and the
    answer is the backend's rather than this kernel's. As things stand nothing promotes
    a twice-read input into the scratchpad -- the planner places buffers that already
    exist rather than creating one -- so today this reads the whole tile from HBM twice
    where ``softmax_fused`` reads it once.

    It is here because that is arguably the correct shape and not a regression. Staging
    a value that already has a backing store is an allocation decision, and allocation
    of intermediates is the compiler's; pinning it by hand is an author working around
    a missing optimization. The SDSC path gets the staging by accident rather than by
    judgement -- its ``identity`` comes from a clone in the traced graph, not from a
    planner that noticed two readers.

    So the pair ``softmax_fused`` / ``softmax_fused_auto`` is the question: should a
    kernel have to say this?
    """
    pid = tl.program_id(0)  # grid is [32]: 2 of the 64 rows per core

    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[64, 11, 2049], strides=[22539, 2049, 1],
        block_shape=[2, 11, 2049],
    )
    tl.spyre_tensor_layout(x_desc, TILE, element_arrangements=["standard"] * 4)

    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[64, 11, 2049], strides=[22539, 2049, 1],
        block_shape=[2, 11, 2049],
    )
    tl.spyre_tensor_layout(out_desc, TILE, element_arrangements=["standard"] * 4)

    x = x_desc.load([pid * 2, 0, 0])

    # No pin. `x` is backed by the caller's tensor, so both consumers below can re-read
    # it from the descriptor; whether either read comes from the scratchpad instead is
    # the compiler's call.

    # group 1 -- max.
    m = tl.max(x, axis=2, keep_dims=True)

    # group 2 -- sub and exp together.
    e = tl.exp(x - m)

    # group 3 -- sum.
    s = tl.sum(e, axis=2, keep_dims=True)

    # group 4 -- the divide, straight to the output descriptor.
    out_desc.store([pid * 2, 0, 0], e / s)
