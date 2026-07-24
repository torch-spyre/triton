#!/usr/bin/env python3
"""
Dump a debug tree of compiled kernels for local inspection.

By default walks ``third_party/spyre/test/fixtures/*/meta.py`` and
compiles every variant. With ``--driver`` you can lower a single
external @triton.jit kernel from any Python file — useful for new
kernels (in this repo or elsewhere) without writing a fixture.

Each input produces one folder containing the kernel source, the post-
lowering KTIR, and (with --include-ttir) the intermediate TTIR. Output
is throwaway — not committed anywhere.

    <dest>/
    ├── vector_add__default/
    │   ├── vector_add__default.py
    │   ├── vector_add__default.ktir
    │   └── (vector_add__default.ttir  when --include-ttir)
    ├── vector_add__dynamic/
    │   └── ...
    └── ...

Driver files
------------
A driver is a Python file declaring four module-level globals:

    KERNEL      : @triton.jit function to lower
    SIGNATURE   : dict[str, str] arg name → Triton type ("*fp32", "i32", ...)
    CONSTEXPRS  : dict[str, value] for every constexpr arg
    GRID        : optional tuple, forwarded to SpyreOptions.grid

Drivers are typically dropped next to the kernel they target; the
script adds the driver's parent directory and any ancestor containing
``kernels/`` or ``pyproject.toml`` to ``sys.path`` before importing,
so ``from kernels.foo.spyre import ...`` style imports work.

Usage::

    uv run python scripts/dump_round_trip.py
    uv run python scripts/dump_round_trip.py --dest /tmp/rt/
    uv run python scripts/dump_round_trip.py --filter 'vector_add.*'
    uv run python scripts/dump_round_trip.py --include-ttir
    uv run python scripts/dump_round_trip.py --lint-triton
    uv run python scripts/dump_round_trip.py --lint-ktir
    uv run python scripts/dump_round_trip.py --driver path/to/foo_lower.py
    uv run python scripts/dump_round_trip.py --driver a.py --driver b.py --dest /tmp/rt/
"""

import argparse
import ast
import importlib.util
import inspect
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# IR cleaning and docstring helpers from the shared lib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _patterns import clean_ir, split_docstring  # noqa: E402


_HERE = Path(__file__).resolve().parent
_TRITON_ROOT = _HERE.parent.parent.parent
_SPYRE_ROOT = _TRITON_ROOT / "third_party" / "spyre"
_TEST_DIR = _SPYRE_ROOT / "test"
_FIXTURES_DIR = _TEST_DIR / "fixtures"

# Shipped alongside this script; copied into the distribution root so
# `ruff check` on a checkout picks it up via discovery, and so readers
# can see the exact config the generator used.
_RUFF_CONFIG = _HERE / "ruff.toml"

def _load_conftest():
    """Import ``test/conftest.py`` as a module so we can call its helpers."""
    sys.path.insert(0, str(_SPYRE_ROOT))
    sys.path.insert(0, str(_TEST_DIR))

    spec = importlib.util.spec_from_file_location(
        "conftest", _TEST_DIR / "conftest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Extracting a single @triton.jit function source so each variant's .py
# file stands on its own (instead of copying the whole multi-kernel
# kernel.py). Also rewrites the arg list to annotate constexpr args per
# the variant's resolved SIGNATURE — source-level `n_elements` is plain
# in kernel.py because the static and dynamic variants share the same
# function; the constexpr-ness comes from meta.py's SIGNATURE tuple.
# ---------------------------------------------------------------------------

def extract_kernel_source(
    kernel_fn, constexpr_names: list[str],
    param_values: dict | None = None,
) -> str:
    """Return the standalone source for a single ``@triton.jit`` function
    with ``: tl.constexpr`` annotations applied to the named args.

    When ``param_values`` is given, a header comment lists the concrete
    values the distribution was compiled with (so a reader sees the
    actual shapes without chasing ``meta.py``).
    """
    raw_fn = getattr(kernel_fn, "fn", kernel_fn)
    body = _dedent(inspect.getsource(raw_fn))
    tree = ast.parse(body)

    func_def = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == raw_fn.__name__
    )

    constexpr_set = set(constexpr_names)
    constexpr_annot = ast.parse("tl.constexpr", mode="eval").body
    for arg in func_def.args.args:
        if arg.arg in constexpr_set:
            arg.annotation = constexpr_annot

    rewritten = ast.unparse(tree)
    header = "import triton\nimport triton.language as tl\n\n"
    if param_values:
        params_str = ", ".join(f"{k}={v!r}" for k, v in param_values.items())
        header += f"\n# Compiled with: {params_str}\n\n"
    else:
        header += "\n"
    return header + rewritten + "\n"


def _dedent(src: str) -> str:
    lines = src.splitlines(keepends=True)
    if not lines:
        return src
    first_nonempty = next((l for l in lines if l.strip()), lines[0])
    indent = len(first_nonempty) - len(first_nonempty.lstrip())
    if indent == 0:
        return src
    stripped = []
    for line in lines:
        if line.strip():
            stripped.append(line[indent:])
        else:
            stripped.append(line)
    return "".join(stripped)


# ---------------------------------------------------------------------------
# Main: discover variants, compile, write
# ---------------------------------------------------------------------------

def iter_variants(conftest_mod):
    """Yield ``(key, entry)`` for every variant conftest discovered.

    Uses conftest's populated EXAMPLES registry so discovery semantics
    match the test side exactly. Path-based legacy entries come out
    too; ``main`` filters them by checking for ``kernel_fn``.
    """
    for key, entry in conftest_mod.EXAMPLES.items():
        yield key, entry


def load_driver(path: Path) -> tuple[str, dict]:
    """Import an external driver file and return ``(key, entry)`` shaped like
    a fixture meta.py variant so the rest of the pipeline reuses it.

    The driver may live anywhere on disk. We add its parent directory
    to ``sys.path`` so its own imports resolve, and walk a few levels
    up looking for a ``kernels/`` or ``pyproject.toml`` marker so
    project-rooted imports (``from kernels.foo.spyre import ...``) work
    too.
    """
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    cur = path.parent
    for _ in range(6):
        if (cur / "kernels").is_dir() or (cur / "pyproject.toml").is_file():
            sys.path.insert(0, str(cur))
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    missing = [n for n in ("KERNEL", "SIGNATURE", "CONSTEXPRS") if not hasattr(mod, n)]
    if missing:
        raise SystemExit(
            f"driver {path} is missing required attribute(s): "
            f"{', '.join(missing)}"
        )

    constexpr_names = list(mod.CONSTEXPRS.keys())
    entry = {
        "kernel_fn":   mod.KERNEL,
        "signature":   mod.SIGNATURE,
        "constexprs":  mod.CONSTEXPRS,
        "constexpr":   constexpr_names,   # for extract_kernel_source annotation
        "param_values": mod.CONSTEXPRS,    # for the per-variant header line
    }
    if hasattr(mod, "GRID") and mod.GRID is not None:
        entry["grid"] = tuple(mod.GRID)
    if hasattr(mod, "HBM_DATA_LAYOUT") and mod.HBM_DATA_LAYOUT is not None:
        entry["hbm_data_layout"] = mod.HBM_DATA_LAYOUT
    return path.stem, entry


def _run_make_ttir(conftest_mod, ttir_text: str):
    """Parse raw TTIR text, run SpyreBackend._make_ttir (inliner +
    canonicalizer + combine + reorder_broadcast + CSE + symbol_dce),
    return ``(mod, text)``.

    ``compile_to_ttir`` returns the raw output of ``ASTSource.make_ir``
    — kernels that call `tl.*` helpers still have un-inlined ``tt.call``
    and the callee ``tt.func private @triton.language...`` definitions.
    The distribution output should be post-inlining so reviewers see
    one self-contained ``tt.func``.
    """
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from backend.compiler import SpyreBackend

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    backend = SpyreBackend(target)
    options = backend.parse_options({})

    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete_on_close=False) as f:
        f.write(ttir_text)
        f.flush()
        mod = ir.parse_mlir_module(f.name, ctx)
    mod.context = ctx
    mod = backend._make_ttir(mod, {}, options)
    return mod, str(mod)


def compile_variant(conftest_mod, entry: dict) -> tuple[str, str]:
    """Compile one variant to (ttir_text, ktir_text), both cleaned and
    post-inlining."""
    grid = entry.get("grid")  # None → backend default
    hbm_data_layout = entry.get("hbm_data_layout", "logical")

    raw_ttir = conftest_mod.compile_to_ttir(
        entry["kernel_fn"],
        entry["signature"],
        entry.get("constexprs", {}),
    )
    _, ttir_text = _run_make_ttir(conftest_mod, raw_ttir)

    # KTIR: feed the post-inlined TTIR back through make_ktir_mod.
    # make_ktir_mod runs _make_ttir again internally — that's idempotent
    # after one pass, so it's fine.
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete_on_close=False) as f:
        f.write(ttir_text)
        f.flush()
        mod = conftest_mod.make_ktir_mod(
            f.name, grid=grid, hbm_data_layout=hbm_data_layout
        )
    ktir_text = str(mod)

    return clean_ir(ttir_text), clean_ir(ktir_text)


# ---------------------------------------------------------------------------
# Per-variant and top-level README writers
# ---------------------------------------------------------------------------

def _fixture_base(key: str) -> str:
    """vector_add__dynamic → vector_add."""
    return key.split("__", 1)[0]


def _variant_suffix(key: str) -> str:
    """vector_add__dynamic → dynamic;  vector_add → default."""
    parts = key.split("__", 1)
    return parts[1] if len(parts) == 2 else "default"


def _auto_doc_stub(key: str, entry: dict) -> str:
    """Fallback one-liner when meta.py doesn't supply a ``doc`` field."""
    base = _fixture_base(key)
    suffix = _variant_suffix(key)
    kernel_fn = entry.get("kernel_fn")
    fn_name = getattr(kernel_fn, "fn", kernel_fn).__name__ if kernel_fn else ""
    return (
        f"`{suffix}` variant of the `{base}` fixture, compiled from "
        f"`{fn_name}`."
    )


def _resolve_doc(key: str, entry: dict) -> tuple[str, str]:
    """Return ``(summary, doc)`` for a variant.

    - ``summary`` — one-liner for the top-level TOC.
    - ``doc``     — prose body for the per-variant README.

    Fallbacks when meta.py doesn't set both:
      - only ``doc``: summary = first sentence of doc.
      - only ``summary``: doc = summary.
      - neither: auto-stub for both.
    """
    summary = entry.get("summary")
    doc = entry.get("doc")
    if summary is None and doc is None:
        stub = _auto_doc_stub(key, entry)
        return stub, stub
    if doc is None:
        return summary, summary
    if summary is None:
        first = doc.strip().split(". ", 1)[0].strip()
        if not first.endswith("."):
            first += "."
        # collapse whitespace/newlines for a one-liner TOC row
        summary = " ".join(first.split())
        return summary, doc
    return summary, doc


def write_variant(dest_root: Path, key: str, entry: dict,
                  ttir: str, ktir: str, include_ttir: bool) -> None:
    """Write the per-variant folder with .py, .ktir (and .ttir when --include-ttir)."""
    folder = dest_root / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{key}.py").write_text(
        extract_kernel_source(
            entry["kernel_fn"],
            entry.get("constexpr", []),
            entry.get("param_values"),
        )
    )
    if include_ttir:
        (folder / f"{key}.ttir").write_text(ttir)

    tags = entry.get("tags", [])
    param_values = entry.get("param_values") or {}
    header_lines = [f"// Round-trip variant: {key}"]
    if tags:
        header_lines.append(f"// Demonstrates patterns: {', '.join(tags)}")
    if param_values:
        params_str = ", ".join(f"{k}={v!r}" for k, v in param_values.items())
        header_lines.append(f"// Compiled with: {params_str}")
    header = "\n".join(header_lines) + "\n\n"
    (folder / f"{key}.ktir").write_text(header + ktir)


# ---------------------------------------------------------------------------
# Lints — best-effort sanity checks on the generated tree
# ---------------------------------------------------------------------------

def _resolve_ruff() -> list[str] | None:
    """Return an argv prefix for invoking ruff, or ``None`` if unavailable.

    Prefers a ruff binary on PATH; falls back to ``uvx ruff`` when uv
    is available (downloads and caches the binary on first run).
    """
    if shutil.which("ruff"):
        return ["ruff"]
    if shutil.which("uvx"):
        return ["uvx", "ruff"]
    return None


def lint_triton(written_keys: list[str], dest_root: Path) -> int:
    """Run ``ruff check`` on each generated ``<key>/<key>.py``.

    Advisory — prints failures but does not abort the script. Returns
    the number of files ruff reported issues in (0 means clean).
    """
    argv_prefix = _resolve_ruff()
    if argv_prefix is None:
        print(
            "--lint-triton: skipped — neither `ruff` nor `uvx` on PATH. "
            "Install with `pip install ruff` to enable.",
            flush=True,
        )
        return 0

    # Run once over all files so ruff emits a single grouped report.
    files = [str(dest_root / k / f"{k}.py") for k in written_keys]
    if not files:
        return 0

    # ``--no-cache`` keeps the generated tree self-contained (no
    # .ruff_cache gets dropped into the output dir). ``--output-format=concise``
    # gives one line per finding — easy to scan in a script summary.
    # ``--config`` pins the linter to the shipped ruff.toml so the run
    # matches what a reader would see running ruff on the distribution.
    result = subprocess.run(
        [*argv_prefix, "check", "--no-cache",
         "--config", str(_RUFF_CONFIG),
         "--output-format=concise", *files],
        capture_output=True, text=True,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 0:
        print(f"--lint-triton: clean ({len(files)} file(s))", flush=True)
        return 0

    print(f"--lint-triton: ruff reported issues (exit {result.returncode}):",
          flush=True)
    if stdout:
        print(stdout, flush=True)
    if stderr:
        print(stderr, flush=True)
    # Count distinct files with findings.
    flagged = {
        line.split(":", 1)[0] for line in stdout.splitlines()
        if ":" in line and line[:1] != " "
    }
    return len(flagged)


def _resolve_ktir_opt() -> str | None:
    """Look for the ``ktir-opt`` binary on PATH."""
    return shutil.which("ktir-opt")


def _mlir_ktdp_available() -> bool:
    """True if the ``mlir_ktdp`` Python module is importable."""
    try:
        import importlib
        importlib.import_module("mlir_ktdp")
        return True
    except Exception:  # noqa: BLE001 — any failure means unavailable
        return False


def _lint_ktir_with_ktir_opt(ktir_path: Path) -> tuple[bool, str]:
    """Parse/verify ``ktir_path`` via ``ktir-opt``. Returns ``(ok, stderr)``.

    ``ktir-opt`` (like upstream ``mlir-opt``) sometimes returns exit 0
    even when the verifier emits errors, depending on how input is
    fed in. We treat any stderr output as a failure signal to catch
    that case.
    """
    result = subprocess.run(
        ["ktir-opt", "--verify-each=true", str(ktir_path), "-o", "/dev/null"],
        capture_output=True, text=True,
    )
    ok = result.returncode == 0 and not result.stderr.strip()
    return ok, result.stderr


def _lint_ktir_with_mlir_ktdp(ktir_path: Path) -> tuple[bool, str]:
    """Parse/verify ``ktir_path`` via ``mlir_ktdp`` Python bindings.

    Uses the standard MLIR context + ``Module.parse`` which runs the
    verifier on load. Any exception means the file didn't parse.
    """
    try:
        import mlir_ktdp  # noqa: F401  - registers dialects
        from mlir_ktdp.ir import Context, Module  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return False, f"mlir_ktdp import failed: {exc}"

    try:
        with Context():
            Module.parse(ktir_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, ""


def lint_ktir(written_keys: list[str], dest_root: Path) -> int:
    """Parse + verify each generated ``<key>/<key>.ktir``.

    Tries ``ktir-opt`` on PATH first; falls back to the ``mlir_ktdp``
    Python bindings. Skips with a setup pointer when neither is
    available. Returns the number of files that failed to parse.
    """
    if _resolve_ktir_opt() is not None:
        lint_fn = _lint_ktir_with_ktir_opt
        tool = "ktir-opt"
    elif _mlir_ktdp_available():
        lint_fn = _lint_ktir_with_mlir_ktdp
        tool = "mlir_ktdp"
    else:
        print(
            "--lint-ktir: skipped — neither `ktir-opt` nor the "
            "`mlir_ktdp` Python bindings are available. "
            "See the top-level README (\"KTIR tooling\") for setup.",
            flush=True,
        )
        return 0

    failures: list[tuple[str, str]] = []
    for key in written_keys:
        path = dest_root / key / f"{key}.ktir"
        ok, err = lint_fn(path)
        if not ok:
            failures.append((key, err.strip()))

    if not failures:
        print(
            f"--lint-ktir: clean ({len(written_keys)} file(s) via {tool})",
            flush=True,
        )
        return 0

    print(f"--lint-ktir: {len(failures)} file(s) failed via {tool}:",
          flush=True)
    for key, err in failures:
        first = err.splitlines()[0] if err else "(no stderr)"
        print(f"  - {key}: {first}", flush=True)
    return len(failures)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--dest", type=Path, default=Path(tempfile.mkdtemp(prefix="spyre_rt_")),
        help="destination folder for the debug tree (default: a temp dir)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="regex; only variants whose registry key matches are compiled",
    )
    parser.add_argument(
        "--include-ttir", action="store_true",
        help="also write the intermediate .ttir file for each variant",
    )
    parser.add_argument(
        "--lint-triton", action="store_true",
        help="run `ruff check` on each generated <key>/<key>.py (advisory)",
    )
    parser.add_argument(
        "--lint-ktir", action="store_true",
        help="parse + verify each generated <key>/<key>.ktir via ktir-opt "
             "or mlir_ktdp (skipped if neither is available)",
    )
    parser.add_argument(
        "--driver", type=Path, action="append", default=[],
        help="external driver file declaring KERNEL/SIGNATURE/CONSTEXPRS/[GRID] "
             "(may be passed multiple times). When set, fixtures are skipped.",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    conftest_mod = _load_conftest()
    filt = re.compile(args.filter) if args.filter else None

    # Source of variants: explicit driver list takes precedence over fixture
    # discovery. Mixing the two in one run isn't useful and would force an
    # arbitrary ordering choice for the per-key output folders.
    if args.driver:
        variants = (load_driver(p) for p in args.driver)
    else:
        variants = iter_variants(conftest_mod)

    written_keys: list[str] = []
    disabled_keys: list[str] = []
    for key, entry in variants:
        if filt and not filt.search(key):
            continue
        if "kernel_fn" not in entry:
            continue
        if entry.get("disabled"):
            reason = entry["disabled"].get("reason", "disabled")
            print(f"[{key}] skipped (disabled: {reason})", flush=True)
            disabled_keys.append(key)
            continue
        print(f"[{key}] compiling ...", flush=True)
        ttir, ktir = compile_variant(conftest_mod, entry)
        write_variant(args.dest, key, entry, ttir, ktir,
                      include_ttir=args.include_ttir)
        written_keys.append(key)
        print(f"[{key}] wrote {args.dest / key}", flush=True)

    print(f"\nwrote {len(written_keys)} variant(s) to {args.dest}")
    if disabled_keys:
        print(f"skipped {len(disabled_keys)} disabled: {', '.join(disabled_keys)}")

    if args.lint_triton:
        lint_triton(written_keys, args.dest)
    if args.lint_ktir:
        lint_ktir(written_keys, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
