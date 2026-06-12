"""Tests for the @pattern decorator in utils_pattern.py.

Covers the PatternEmptyBlockError guard (comment-only / blank-only
example= blocks) and the metadata attachment contract.
"""
import pytest

from utils_pattern import PatternEmptyBlockError, pattern

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_TAG = "test-tag"
DUMMY_CATEGORY = "test-category"


def apply_pattern(**kwargs):
    """Return a decorated no-op function using the given @pattern kwargs."""
    @pattern(DUMMY_TAG, category=DUMMY_CATEGORY, **kwargs)
    def fn():
        pass
    return fn


# ---------------------------------------------------------------------------
# No Error section: decorator attaches metadata and does not raise error
# ---------------------------------------------------------------------------

class TestPatternMetadata:
    def test_no_example_attaches_metadata(self):
        """No example= → _pattern dict is attached with example=None."""
        fn = apply_pattern()
        assert fn._pattern["tag"] == DUMMY_TAG
        assert fn._pattern["category"] == DUMMY_CATEGORY
        assert fn._pattern["example"] is None
        assert fn._pattern["negative"] is False

    def test_code_example_string_attaches_metadata(self):
        """Single-string example= with a real code line is accepted."""
        fn = apply_pattern(example="x = tl.load(ptr)")
        assert fn._pattern["example"] == "x = tl.load(ptr)"

    def test_code_example_list_attaches_metadata(self):
        """List-form example= is joined with newlines and stored verbatim."""
        lines = ["x = tl.load(ptr)", "tl.store(out, x)"]
        fn = apply_pattern(example=lines)
        assert fn._pattern["example"] == "\n".join(lines)

    def test_comment_plus_code_line_is_accepted(self):
        """A mix of comments and at least one code line must not raise."""
        fn = apply_pattern(example=["# docstring-style note", "x = tl.load(ptr)"])
        assert fn._pattern["example"] is not None

    def test_negative_flag_is_stored(self):
        """negative=True is reflected in _pattern."""
        fn = apply_pattern(negative=True, example=["x = tl.load(ptr)"])
        assert fn._pattern["negative"] is True

    def test_function_is_returned_unchanged(self):
        """@pattern returns the original function object."""
        def original():
            return 42
        decorated = pattern(DUMMY_TAG, category=DUMMY_CATEGORY)(original)
        assert decorated is original
        assert decorated() == 42

    def test_triple_quoted_string_is_dedented(self):
        """Triple-quoted string example= is dedented and stripped."""
        fn = apply_pattern(example="""
            x = tl.load(ptr)
            tl.store(out, x)
        """)
        assert fn._pattern["example"] == "x = tl.load(ptr)\ntl.store(out, x)"


# ---------------------------------------------------------------------------
# Guard: comment-only / blank-only example= must raise PatternEmptyBlockError.
#
# All cases below fire at decoration time (class-body evaluation), so pytest
# surfaces a collection error before any test runs — that is the entire point
# of the guard.  Every pytest.raises block in this class already verifies that
# property; there is no need for a separate "fires at decoration time" test.
# ---------------------------------------------------------------------------

_EMPTY_BLOCK_CASES = [
    pytest.param(["# only a comment"],          id="comment-only-list"),
    pytest.param("# only a comment",            id="comment-only-string"),
    pytest.param(["", "   "],                   id="blank-only-list"),
    pytest.param(["# first comment", "# second comment"], id="multiple-comments"),
    pytest.param(["  # leading-space comment"], id="leading-space-comment"),
    pytest.param(["", "# blank-plus-comment"],  id="blank-plus-comment"),
    # example= explicitly provided as empty: a typo, not a deliberate
    # prose-only entry (which omits example= entirely → None).
    pytest.param("",                            id="empty-string"),
    pytest.param([],                            id="empty-list"),
]


class TestPatternEmptyBlockError:
    @pytest.mark.parametrize("example", _EMPTY_BLOCK_CASES)
    def test_empty_block_raises(self, example):
        """Any example= without a code line raises PatternEmptyBlockError.

        Fires at decoration time (class-body evaluation), so pytest surfaces
        it as a collection error — not a test failure — making misconfigured
        @pattern calls immediately visible.
        """
        with pytest.raises(PatternEmptyBlockError):
            apply_pattern(example=example)

    def test_negative_flag_does_not_bypass_guard(self):
        """negative=True does not exempt example= from the empty-block guard."""
        with pytest.raises(PatternEmptyBlockError):
            apply_pattern(negative=True, example=["# not a code line"])

    def test_error_message_contains_qualname(self):
        """Error message names the function's qualname for easy grep."""
        with pytest.raises(PatternEmptyBlockError, match="named_fn"):
            @pattern(DUMMY_TAG, category=DUMMY_CATEGORY, example=["# comment only"])
            def named_fn():
                pass
