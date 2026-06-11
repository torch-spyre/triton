"""@pattern decorator — attaches doc-generation metadata to single-pass tests.

Usage::

    from utils_pattern import pattern

    @pattern("program-id-1d", category="distribution", example=[
        "pid = tl.program_id(0)",
        "num_cores = tl.num_programs(0)",
        "offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
    ])
    def test_no_scf_for_synthesized(...): ...

    @pattern("descriptor-load-dynamic", category="memory", negative=True, example=[
        "# Not yet supported: tt.addptr result as descriptor base",
        "base = a_ptr + b_idx * stride",
        "desc = tl.make_tensor_descriptor(base, ...)",
    ])
    def test_addptr_into_descriptor_fails(...): ...

The decorator is a pure no-op at runtime: it attaches a ``_pattern`` dict
to the function and returns it unchanged.  Test collection and execution
are unaffected.

The ``gen_patterns_docs.py`` script harvests these attributes to build
``docs/patterns/{category}.md``.

``example`` is optional.  Entries without one render prose-only (docstring +
diagnostics for negatives).  For negative entries in particular, ``example``
is important — it shows the Python pattern to avoid.  A missing ``example``
on a negative entry is a reminder that it still needs one.
"""
import textwrap


class PatternEmptyBlockError(Exception):
    """Raised when @pattern receives an example= with no code lines.

    Fires at import time (class-body evaluation), so pytest surfaces it as a
    collection error before any test runs. The message names the tag and the
    decorated function's qualname to pinpoint the misconfigured decorator.
    """


def pattern(tag: str, *, category: str, negative: bool = False,
            example: str | None = None):
    """Decorator that marks a test as a documented pattern entry.

    Parameters
    ----------
    tag:
        Identifies the pattern.  Multiple tests may share a tag (e.g. a
        positive and a negative test for the same concept).  The same tag
        string on a fixture variant in ``meta.py`` cross-links to a
        round-trip demonstration.
    category:
        Groups tags into a single ``docs/patterns/{category}.md`` file.
    negative:
        True for tests that verify a pattern is *rejected* by the compiler.
        Positive and negative tests sharing a tag are rendered side-by-side.
    example:
        Optional Triton Python snippet shown in the docs.  For positive
        entries: what to write.  For negative entries: what to avoid.
        Pass a list of strings (one per line) or a single triple-quoted string
        (dedented automatically).
    """
    def decorator(fn):
        if isinstance(example, list):
            ex = "\n".join(example)
        else:
            ex = textwrap.dedent(example).strip() if example else None

        if ex is not None:
            code_lines = [
                line for line in ex.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if not code_lines:
                raise PatternEmptyBlockError(
                    f"@pattern(tag={tag!r}) on {fn.__qualname__}: "
                    f"example= contains no code lines (only comments/blank). "
                    f"Move prose to the docstring; put a tl.* / Triton snippet in example=."
                )

        fn._pattern = {
            "tag": tag,
            "category": category,
            "negative": negative,
            "example": ex,
        }
        return fn
    return decorator
