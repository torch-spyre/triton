#!/usr/bin/env python3
"""Generate third_party/spyre/docs/patterns/ from @pattern-decorated single-pass tests.

For each category found among tagged tests, emits one Markdown file
``third_party/spyre/docs/patterns/{category}.md``.  Within each file, tests are
grouped by tag; positive and negative entries appear side by side.

A top-level ``third_party/spyre/docs/patterns/index.md`` lists all categories.

Usage::

    uv run python scripts/gen_patterns_docs.py
    uv run python scripts/gen_patterns_docs.py --dest third_party/spyre/docs/patterns/
    uv run python scripts/gen_patterns_docs.py --check   # diff, exit 1 if stale

Tags that have no matching fixture variant emit an info note listing
the unverified patterns — not an error.
"""

import argparse
import importlib.util
import sys
import tempfile
import textwrap
import types
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TRITON_ROOT = _HERE.parent.parent.parent
_SPYRE_ROOT = _TRITON_ROOT / "third_party" / "spyre"
_TEST_DIR = _SPYRE_ROOT / "test"
_FIXTURES_DIR = _TEST_DIR / "fixtures"

def _discover_test_files() -> list[Path]:
    """Return test_*.py files under _TEST_DIR that are single-pass test files.

    Detection: load each candidate module and check whether any class
    inherits from ``SinglePassTester``.  That base class is the contract
    for single-pass pattern tests; non-pass test files (fixture tests,
    frontend-guard tests, etc.) do not subclass it and are excluded.
    """
    import inspect
    # Bootstrap must have run before this is called so conftest is importable.
    sys.path.insert(0, str(_TEST_DIR))
    try:
        from conftest import SinglePassTester
    except Exception:
        SinglePassTester = None

    found = []
    for path in sorted(_TEST_DIR.glob("test_*.py")):
        try:
            mod = _load_module(path)
        except Exception:
            continue
        if SinglePassTester is not None and any(
            inspect.isclass(obj)
            and issubclass(obj, SinglePassTester)
            and obj is not SinglePassTester
            for obj in vars(mod).values()
        ):
            found.append(path)
    return found

_DEFAULT_DEST = _SPYRE_ROOT / "docs" / "patterns"


# ---------------------------------------------------------------------------
# Bootstrap — set up sys.path and stub Triton so test/fixture modules import
# cleanly without a full Triton build in scope.
# ---------------------------------------------------------------------------
#
# Why a Triton stub is needed (and when it is not)
# ------------------------------------------------
# This generator only reads *static* metadata: tag/param/docstring info from the
# ``@pattern`` tests and from each fixture's ``VARIANTS`` dict (``tags`` and
# ``params`` keys). It never compiles or runs a kernel.
#
# But it discovers that metadata by *executing* the modules
# (``importlib.exec_module``) and reading their module-level names. Executing a
# fixture ``meta.py`` runs its top level, which does ``from . import kernel`` and
# embeds live kernel objects in ``VARIANTS`` (``"kernel_fn": kernel.foo``).
# ``kernel.py`` in turn does ``import triton``, and Triton's import chain reaches
# ``triton._C.libtriton.getenv`` (via ``triton.knobs``). In a checkout with no
# compiled extension (e.g. CI before the build, or a docs-only job) that import
# raises ``ImportError`` and the whole crawl fails.
#
# The dependency is therefore *incidental*: we don't want anything from Triton,
# we only need ``import triton`` / ``@triton.jit`` / ``tl.constexpr`` to not blow
# up while the module executes. So when the real Triton is unavailable we install
# a minimal fake into ``sys.modules`` *before* exec'ing fixtures:
#
#   - ``triton.jit`` returns its argument unchanged — decoration is a no-op, so
#     ``kernel.foo`` becomes an ordinary (dummy) function object. (Returning the
#     function, not ``None``, is essential: ``VARIANTS`` evaluates
#     ``kernel.foo`` eagerly, so the attribute access must succeed.)
#   - ``triton.language.constexpr`` is a sentinel usable as a parameter
#     annotation (``M: tl.constexpr`` is evaluated at def time).
#
# Nothing else is needed: every other ``tl.*`` call (``tl.dot``,
# ``tl.make_tensor_descriptor``, ...) lives *inside* ``@triton.jit`` function
# bodies, which are compiled but never executed during a plain import.
#
# The stub is intentionally scoped to this generator (installed only inside
# ``_bootstrap()``, only when the real Triton is missing) so the test suite,
# which needs the real ``kernel_fn`` objects, is never affected.
#
# A more robust future alternative: drop ``exec_module`` entirely and read
# ``VARIANTS`` statically with ``ast`` (parse the dict literal, ``literal_eval``
# the ``tags``/``params`` values, and simply skip the ``kernel_fn`` node). That
# would need no Triton at all, fake or real, and no stub here — at the cost of a
# small AST walker. Contributions welcome.


class _AttrStub(types.ModuleType):
    """A module whose every attribute access returns ``None``."""

    def __getattr__(self, name):
        return None


def _stub_triton():
    """Install a minimal fake ``triton`` so fixture modules import without a build.

    Only used when the real Triton extension is not importable. See the module
    comment above for the full rationale.
    """
    triton_mod = types.ModuleType("triton")
    # No-op decorator: `@triton.jit` returns the function unchanged so that
    # `kernel.foo` (referenced eagerly in VARIANTS) resolves to a real object.
    triton_mod.jit = lambda fn=None, **_kw: fn if fn is not None else (lambda f: f)

    lang_mod = types.ModuleType("triton.language")
    lang_mod.constexpr = object()  # usable as a parameter annotation
    triton_mod.language = lang_mod

    sys.modules.setdefault("triton", triton_mod)
    sys.modules.setdefault("triton.language", lang_mod)


def _bootstrap():
    sys.path.insert(0, str(_SPYRE_ROOT))
    sys.path.insert(0, str(_TEST_DIR))
    try:
        from triton._C.libtriton import gluon_ir  # noqa: F401
    except ImportError:
        # No compiled extension in scope. Stub gluon_ir (for the single-pass
        # tests) and the whole triton package (for the fixture kernels).
        sys.modules["triton._C.libtriton.gluon_ir"] = _AttrStub("gluon_ir")
        _stub_triton()


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixture tag resolution — collect tags from fixtures/*/meta.py VARIANTS.
# ---------------------------------------------------------------------------

def _collect_fixture_tags() -> dict[str, list[dict]]:
    """Return {tag: [{fixture_name, variant_key, params}, ...]}."""
    tag_map: dict[str, list[dict]] = defaultdict(list)
    for meta_path in sorted(_FIXTURES_DIR.glob("*/meta.py")):
        fixture_name = meta_path.parent.name
        spec = importlib.util.spec_from_file_location(
            f"fixtures.{fixture_name}.meta", meta_path
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"Warning: skipping fixture '{fixture_name}': {e}")
            continue
        variants = getattr(mod, "VARIANTS", {})
        for key, entry in variants.items():
            for tag in entry.get("tags", []):
                tag_map[tag].append({
                    "fixture": fixture_name,
                    "key": key,
                    "params": entry.get("params", {}),
                    "tags": entry.get("tags", []),
                })
    return dict(tag_map)


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------

def _render_entry(entry, *, heading_level: int = 4) -> str:
    """Render one PatternEntry as Markdown."""
    from _patterns import split_docstring

    prefix = "#" * heading_level
    title, body = split_docstring(entry.docstring)

    lines: list[str] = []
    icon = "❌" if entry.negative else "✅"
    lines.append(f"{prefix} {icon} `{entry.fn_name}`\n")

    if title:
        lines.append(f"_{title}_\n")

    if body:
        lines.append(body + "\n")

    if entry.example:
        lines.append(f"```python\n{entry.example}\n```\n")
    elif entry.negative:
        lines.append("_TODO: add `example=` to show which Python pattern to avoid._\n")

    if entry.negative and entry.stderr_substrings:
        diags = "\n".join(f"- `{s}`" for s in entry.stderr_substrings)
        lines.append(f"Expected diagnostics:\n\n{diags}\n")

    src_rel = entry.source_file.relative_to(_TRITON_ROOT)
    lines.append(
        f"<sup>Source: `{src_rel}:{entry.lineno}` "
        f"(`{entry.class_name}.{entry.fn_name}`)</sup>\n"
    )
    return "\n".join(lines)


_ROUND_TRIP_LIMIT = 5


def _render_round_trip_block(variants: list[dict]) -> str:
    lines = ["**Round-trip evidence**\n"]
    shown = variants[:_ROUND_TRIP_LIMIT]
    remainder = len(variants) - len(shown)
    for v in shown:
        fixture = v["fixture"]
        key = v["key"]
        def _fmt_param(k, vals):
            v0 = vals[0]
            if isinstance(v0, list) and len(v0) > 4:
                return f"{k}=[…{len(v0)} items]"
            return f"{k}={v0!r}"

        params_str = ", ".join(
            _fmt_param(k, vals) for k, vals in v["params"].items() if vals
        ) if v["params"] else ""
        other_tags = [t for t in v["tags"] if t != v.get("_current_tag")]
        suffix = f" (also demonstrates: {', '.join(other_tags)})" if other_tags else ""
        params_note = f" — {params_str}" if params_str else ""
        lines.append(f"- `{fixture}::{key}`{params_note}{suffix}")
    if remainder:
        lines.append(f"\n_+ {remainder} more variant{'s' if remainder > 1 else ''}_")
    return "\n".join(lines) + "\n"


def _render_category(
    category: str,
    entries_by_tag: dict[str, list],
    fixture_tags: dict[str, list[dict]],
) -> str:
    """Render one category's Markdown file content."""
    lines: list[str] = [f"# {category.replace('-', ' ').title()}\n"]
    lines.append(
        "_This file is generated by `scripts/gen_patterns_docs.py`. "
        "Do not edit by hand._\n"
    )

    no_round_trip: list[str] = []

    for tag in sorted(entries_by_tag):
        entries = entries_by_tag[tag]
        positives = [e for e in entries if not e.negative]
        negatives = [e for e in entries if e.negative]

        lines.append(f"## {tag}\n")

        if positives:
            lines.append("### Supported\n")
            for e in positives:
                lines.append(_render_entry(e, heading_level=4))

        if negatives:
            lines.append("### Rejected\n")
            for e in negatives:
                lines.append(_render_entry(e, heading_level=4))

        variants = fixture_tags.get(tag, [])
        if variants:
            for v in variants:
                v["_current_tag"] = tag
            lines.append(_render_round_trip_block(variants))
        elif positives:
            no_round_trip.append(tag)

    if no_round_trip:
        tags_list = ", ".join(f"`{t}`" for t in sorted(no_round_trip))
        lines.append(
            f"\n---\n\n"
            f"_Patterns without round-trip evidence: {tags_list}. "
            f"Add a tagged fixture variant to verify end-to-end._\n"
        )

    return "\n".join(lines)


def _render_index(categories: list[str]) -> str:
    lines = [
        "# Spyre KTIR Pattern Reference\n",
        "_Generated by `scripts/gen_patterns_docs.py`. Do not edit by hand._\n",
        "Patterns are organized by category. "
        "Each section lists supported patterns alongside rejected ones for contrast. "
        "Round-trip evidence links to fixture variants that exercise the pattern end-to-end.\n",
        "## Categories\n",
    ]
    for cat in sorted(categories):
        title = cat.replace("-", " ").title()
        lines.append(f"- [{title}]({cat}.md)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_entries():
    from _patterns import extract_tagged_tests
    entries = []
    for path in _discover_test_files():
        mod = _load_module(path)
        entries.extend(extract_tagged_tests(mod))
    return entries


def generate(dest: Path) -> dict[str, str]:
    """Return {relative_path: content} for all files to write."""
    entries = collect_entries()
    fixture_tags = _collect_fixture_tags()

    # Group by category → tag.
    by_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        by_cat[e.category][e.tag].append(e)

    files: dict[str, str] = {}
    for cat, entries_by_tag in by_cat.items():
        content = _render_category(cat, entries_by_tag, fixture_tags)
        files[f"{cat}.md"] = content

    files["index.md"] = _render_index(list(by_cat.keys()))
    return files


def write_docs(dest: Path, files: dict[str, str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        (dest / rel).write_text(content)
        print(f"  wrote {dest / rel}")


def check_docs(dest: Path, files: dict[str, str]) -> int:
    """Return 0 if on-disk docs match generated content, 1 otherwise."""
    stale = []
    for rel, content in files.items():
        path = dest / rel
        if not path.exists() or path.read_text() != content:
            stale.append(rel)
    if stale:
        print("Stale docs (re-run gen_patterns_docs.py to update):")
        for f in stale:
            print(f"  {f}")
        return 1
    print("docs/patterns/ is up to date.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate third_party/spyre/docs/patterns/ from @pattern-decorated tests."
    )
    parser.add_argument(
        "--dest", type=Path, default=_DEFAULT_DEST,
        help="Output directory (default: third_party/spyre/docs/patterns/)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Diff against on-disk docs; exit 1 if stale.",
    )
    args = parser.parse_args()

    _bootstrap()
    files = generate(args.dest)

    if args.check:
        return check_docs(args.dest, files)
    write_docs(args.dest, files)
    print(f"Generated {len(files)} files in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
