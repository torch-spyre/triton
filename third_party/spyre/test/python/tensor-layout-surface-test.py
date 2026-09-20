# RUN: %python -m pytest %s -q
"""What ``tl.spyre_tensor_layout`` emits, and what it refuses.

The Python surface of the layout annotation: its entry grammar, the numeric codes
those keywords map to, and the op it builds. Nothing else covers it -- the fixture
kernels exercise it in passing, but a change to the coordinate-op table or to the
builder would show up there as a numerical failure in some other family rather
than as a statement about this surface.

Three claims, and they are separable on purpose.

**It authors ``tts.tensor_layout``.** Not ``tt.spyre_tensor_layout``, which is what
it built before the layout annotation moved into our own dialect, and which is
still in tree for the named pass. Since the two ops carry identical attributes, a
kernel authored against the wrong one lowers quietly down the old path instead of
failing -- so the op name is asserted, not assumed.

**The four coordinate ops round-trip, by keyword and by code.** ``splat`` is the
one worth naming: both verifiers accepted code 3 before the frontend could reach
it, so it was representable in IR and unauthorable from Python. Each entry form is
driven twice, once spelled and once numbered, and the two must produce the same
arrays -- that equality is the test, rather than a transcription of the expected
numbers, because the map from keyword to code lives in exactly one place
(``_COORD_OPS`` in ``python/triton/language/semantic.py``) and a test restating it
would just be a second copy to keep in step.

**The rejections name the valid set.** A message that enumerates an option set goes
stale the moment the set grows, and this one did: it listed three ops while the
verifiers took four. So the enumeration is checked, not just the raising.

The hyphen in the filename is load-bearing -- pytest collects ``test_*.py`` and
``*_test.py``, so an underscore would have the pytest suite re-run this alongside
lit. Lit invokes pytest on it by path, which is how every other file here runs.
"""

import pytest

import triton
import triton.language as tl

from utils import compile_to_ttir


# One rank-1 descriptor and one annotation, which is the least a layout can be
# stated on. Rank is not what is under test here -- the coordinate map is -- and a
# 2-D kernel would need its own shape arguments for no added claim.
@triton.jit
def annotate(p, M: tl.constexpr, LAYOUT: tl.constexpr):
    desc = tl.make_tensor_descriptor(p, shape=[M], strides=[1], block_shape=[M])
    tl.spyre_tensor_layout(desc, LAYOUT)


def emit(layout):
    """The ``tts.tensor_layout`` line for *layout*, minus its location suffix."""
    text = compile_to_ttir(annotate, {"p": "*fp16"}, {"M": 64, "LAYOUT": layout})
    lines = [l.strip() for l in text.splitlines() if "tts.tensor_layout" in l]
    assert len(lines) == 1, (
        f"expected exactly one tts.tensor_layout for {layout!r}, "
        f"got {len(lines)}:\n" + "\n".join(lines))
    return lines[0].split(" loc(")[0]


# One row per coordinate op: a label, the layout spelled with keywords, and the
# same layout with numeric codes. Whole layouts rather than single entries, because
# what a valid layout is depends on the entries TOGETHER -- the shared checker's
# repeated-dim rule is why ``splat`` needs a companion identity while the other
# three stand alone. A bare int is identity's third spelling and is a row of its
# own: it is the form every fixture kernel uses for an untouched dim, so a
# regression there would be the widest.
#
# The expected arrays are spelled per row. They duplicate nothing -- the
# keyword-to-code map is checked by the agreement test below, which is where that
# single source of truth is exercised; here the point is that the op carries the
# coordinate map the author wrote.
#
# phys_arg is 64 on every non-identity entry because the rule is `> 0`, not because
# 64 means anything here; a stick width is the hardware's and this file makes no
# claim about it.
CASES = [
    ("identity_bare", (0,), (0,),
     "phys_arg = array<i64: 0>, phys_op = array<i64: 0>, "
     "phys_src = array<i64: 0>"),
    ("identity", ((0, "identity"),), ((0, 0),),
     "phys_arg = array<i64: 0>, phys_op = array<i64: 0>, "
     "phys_src = array<i64: 0>"),
    ("floordiv", ((0, "floordiv", 64),), ((0, 1, 64),),
     "phys_arg = array<i64: 64>, phys_op = array<i64: 1>, "
     "phys_src = array<i64: 0>"),
    ("mod", ((0, "mod", 64),), ((0, 2, 64),),
     "phys_arg = array<i64: 64>, phys_op = array<i64: 2>, "
     "phys_src = array<i64: 0>"),
    ("splat", (0, (0, "splat", 64)), (0, (0, 3, 64)),
     "phys_arg = array<i64: 0, 64>, phys_op = array<i64: 0, 3>, "
     "phys_src = array<i64: 0, 0>"),
]


@pytest.mark.parametrize("label,spelled,numbered,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_authors_the_coordinate_map(label, spelled, numbered, expected):
    """The op is ours, and it carries the map the author wrote."""
    line = emit(spelled)
    assert line.startswith("tts.tensor_layout"), line
    # And never the Triton-dialect op the named pass still reads. The two carry
    # identical attributes, so a kernel authored against the wrong one would
    # lower quietly down the old path rather than fail.
    assert "tt.spyre_tensor_layout" not in line, line
    assert expected in line, f"{label}: expected {expected!r} in\n  {line}"


@pytest.mark.parametrize("label,spelled,numbered,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_keyword_and_numeric_spellings_agree(label, spelled, numbered,
                                             expected):
    """The keyword-to-code map, exercised rather than transcribed."""
    assert emit(spelled) == emit(numbered), (
        f"{label}: keyword and numeric spellings disagree")


# Both rejections, and both messages have to keep naming the whole valid set --
# which is the part that went stale before.
@pytest.mark.parametrize("label,layout,expected", [
    ("unknown_keyword", (0, (0, "broadcast", 64)),
     ["must be one of ['floordiv', 'identity', 'mod', 'splat']",
      "got 'broadcast'"]),
    ("out_of_range_code", (0, (0, 4, 64)),
     ["['floordiv', 'identity', 'mod', 'splat'] or 0/1/2/3", "got 4"]),
], ids=["unknown_keyword", "out_of_range_code"])
def test_rejects_and_names_the_valid_set(label, layout, expected):
    # A CompilationError wrapping the ValueError, so match on the text rather
    # than the type.
    with pytest.raises(Exception) as excinfo:
        emit(layout)
    message = str(excinfo.value)
    for fragment in expected:
        assert fragment in message, (
            f"{label}: {fragment!r} missing from\n  {message}")
