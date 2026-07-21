#!/usr/bin/env python3
"""Compile Triton kernel modules to TTIR and write .mlir files.

Each input module must export:
  - A ``@triton.jit`` decorated function (the first one found is used)
  - ``SIGNATURE``: dict mapping arg names to type strings (e.g. ``"*fp16"``)
  - ``CONSTEXPRS``: dict mapping constexpr names to values

Usage::

    python scripts/gen_ttir.py test/fixtures/vector_add/kernel.py
    python scripts/gen_ttir.py test/fixtures/*/kernel.py   # regenerate all

Output ``.mlir`` files are written next to each input file.
"""

import importlib.util
import sys
from pathlib import Path

_spyre_dir = Path(__file__).resolve().parents[1]
_triton_root = _spyre_dir.parents[1]
sys.path.insert(0, str(_triton_root / "python"))
sys.path.insert(0, str(_spyre_dir / "test"))

from utils import compile_to_ttir
import triton


def find_kernel(module):
    """Return the first ``@triton.jit`` function in *module*."""
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, triton.JITFunction):
            return obj
    raise ValueError(f"No @triton.jit function found in {module.__name__}")


def load_module(path):
    """Import a Python file as a module."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.py> [kernel2.py ...]",
              file=sys.stderr)
        sys.exit(1)

    for kernel_path in sys.argv[1:]:
        module = load_module(kernel_path)
        kernel_fn = find_kernel(module)
        signature = module.SIGNATURE
        constexprs = module.CONSTEXPRS

        ttir_text = compile_to_ttir(kernel_fn, signature, constexprs)

        out_name = Path(kernel_path).stem + ".mlir"
        out_path = Path(kernel_path).parent / out_name
        out_path.write_text(ttir_text)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
