"""Shared utilities for Spyre KTIR tests.

Hoisted out of ``conftest.py`` so that standalone scripts (``gen_ttir.py``,
ad-hoc notebooks) can import them without pulling in pytest fixtures.

Contents
--------
- :func:`compile_to_ttir`      — ``@triton.jit`` → TTIR text
- :class:`OpInfo`              — structural snapshot of one MLIR op
- :func:`walk_module`           — build flat OpInfo list from a live ir.module
- :func:`make_ktir_mod`         — TTIR → KTIR pipeline, returns live ir.module
- :class:`StructuralAssertions` — query/assertion mixin over a flat ops list
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# compile_to_ttir — Triton kernel → TTIR text
# ---------------------------------------------------------------------------

def compile_to_ttir(kernel_fn, signature, constexprs):
    """Compile a ``@triton.jit`` function to TTIR text.

    Parameters
    ----------
    kernel_fn  : a ``@triton.jit`` decorated function (``triton.JITFunction``)
    signature  : dict mapping arg names to type strings (e.g. ``"*fp32"``)
    constexprs : dict mapping constexpr names to values
    """
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from backend.compiler import SpyreBackend

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    src = ASTSource(fn=kernel_fn, signature=signature, constexprs=constexprs)

    backend = SpyreBackend(target)
    options = backend.parse_options({})

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = (backend.get_codegen_implementation(options)
                   if hasattr(backend, "get_codegen_implementation") else {})
    module_map = backend.get_module_map()

    mod = src.make_ir(target, options, codegen_fns, module_map, context)
    return str(mod)


# ---------------------------------------------------------------------------
# OpInfo — structural snapshot of one MLIR operation
# ---------------------------------------------------------------------------

@dataclass
class OpInfo:
    """Structural snapshot of a single MLIR operation produced by :func:`walk_module`.

    Fields
    ------
    name         : dialect-qualified op name, e.g. ``"ktdp.load"``
    ancestry     : tuple of ancestor op names from root down to the immediate
                   parent, **not** including this op's own name.
                   The root ``builtin.module`` has an empty tuple.

                   Example for ``ktdp.load`` nested inside ``scf.for``::

                       ancestry = ("builtin.module", "func.func", "scf.for")

    result_types : string form of each result type, e.g.
                   ``["!ktdp.access_tile<1024xindex>"]``.
                   Empty for ops with no results (store, return, …).
    num_regions  : number of regions owned by this op (0 for leaf ops).
    _op          : raw ``ir.operation`` handle.
    """
    name: str
    ancestry: tuple
    result_types: list
    num_regions: int
    _op: object = field(repr=False)


# ---------------------------------------------------------------------------
# walk_module — flat OpInfo list from a live ir.module
# ---------------------------------------------------------------------------

def walk_module(mod) -> list:
    """Return a flat list of :class:`OpInfo` for every op in *mod*.

    ``ir.module.walk()`` visits ops in **post-order** — children fire before
    their parent. Ancestry is reconstructed in a **two-pass** approach keyed
    on MLIR region ids (no recursive descent, which the pybind API doesn't
    support for block-indexed access).
    """
    raw = []  # [(op, in_rid, owned_rids, result_types)]

    def _cb(op):
        blk = op.get_block()
        in_rid = blk.get_parent().id() if blk is not None else None
        owned = [op.get_region(i).id() for i in range(op.get_num_regions())]
        rtypes = [
            str(op.get_result(i).get_type())
            for i in range(op.get_num_results())
        ]
        raw.append((op, in_rid, owned, rtypes))

    mod.walk(_cb)

    region_owner: dict = {}
    region_owner_in_rid: dict = {}
    for op, in_rid, owned, _ in raw:
        for rid in owned:
            region_owner[rid] = op.get_name()
            region_owner_in_rid[rid] = in_rid

    def _ancestry(in_rid):
        chain = []
        rid = in_rid
        while rid is not None and rid in region_owner:
            chain.append(region_owner[rid])
            rid = region_owner_in_rid[rid]
        return tuple(reversed(chain))

    return [
        OpInfo(
            name=op.get_name(),
            ancestry=_ancestry(in_rid),
            result_types=rtypes,
            num_regions=len(owned),
            _op=op,
        )
        for op, in_rid, owned, rtypes in raw
    ]


# ---------------------------------------------------------------------------
# make_ktir_mod — full TTIR → KTIR pipeline, returns live ir.module
# ---------------------------------------------------------------------------

def make_ktir_mod(ttir_path, *, grid=None):
    """Parse *ttir_path*, run TTIR and KTIR passes, return the live module.

    ``grid`` is an optional per-axis hardware partition forwarded to the
    DistributeWork pass via SpyreOptions. Defaults to the backend's
    default grid (currently ``(32,)``) when omitted.
    """
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from backend.compiler import SpyreBackend

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    backend = SpyreBackend(target)
    opts = {"grid": tuple(grid)} if grid is not None else {}
    options = backend.parse_options(opts)

    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)

    mod = ir.parse_mlir_module(str(ttir_path), ctx)
    mod.context = ctx

    metadata = {}
    mod = backend._make_ttir(mod, metadata, options)
    return backend._make_ktir(mod, metadata, options)


# ---------------------------------------------------------------------------
# StructuralAssertions — query and assertion helpers over a flat ops list
# ---------------------------------------------------------------------------

class StructuralAssertions:
    """Mixin providing structural query and assertion helpers.

    Expects the subclass to populate ``self.ops`` (list of :class:`OpInfo`)
    and ``self._def_map`` (initially ``None``, built lazily).
    """

    @property
    def def_map(self) -> dict:
        """``{value.id(): OpInfo}`` — maps each SSA result to its defining op.

        Built lazily on first access.
        """
        if self._def_map is None:
            self._def_map = {}
            for o in self.ops:
                for j in range(o._op.get_num_results()):
                    self._def_map[o._op.get_result(j).id()] = o
        return self._def_map

    def get_defining_op(self, op_info, operand_idx: int):
        """Return the :class:`OpInfo` that defines operand *operand_idx* of *op_info*."""
        vid = op_info._op.get_operand(operand_idx).id()
        return self.def_map.get(vid)

    def _find(self, name: str, parent: str = None, idx: int = 0,
              shape=None, elem_type: str = None) -> list:
        """Internal: return all :class:`OpInfo` matching the given filters."""
        result = []
        for o in self.ops:
            if o.name != name:
                continue
            if parent is not None and not (o.ancestry and o.ancestry[-1] == parent):
                continue
            if shape is not None or elem_type is not None:
                info = self.get_result_info(o, idx)
                if info is None:
                    continue
                if shape is not None and info.get("shape") != shape:
                    continue
                if elem_type is not None and info.get("elem_type") != elem_type:
                    continue
            result.append(o)
        return result

    @staticmethod
    def get_result_info(op_info, idx: int = 0) -> dict:
        """Return structured type info for result *idx* of *op_info*."""
        from triton._C.libtriton import spyre
        return spyre.ir_utils.get_result_info(op_info._op, idx)

    def assert_present(self, *op_names: str, parent: str = None):
        """Assert each op in *op_names* appears at least once."""
        for name in op_names:
            assert self._find(name, parent), (
                f"Expected op '{name}'"
                + (f" inside '{parent}'" if parent else "")
                + " not found in KTIR"
            )

    def assert_absent(self, *op_names: str):
        """Assert none of *op_names* appear anywhere in the module."""
        for name in op_names:
            assert not self._find(name), (
                f"Unexpected op '{name}' found in KTIR"
            )

    def assert_count(self, op_name: str, n: int, cmp: str = "ge",
                     parent: str = None):
        """Assert occurrence count of *op_name* satisfies *cmp* vs *n*."""
        c = len(self._find(op_name, parent))
        ok = {"ge": c >= n, "eq": c == n, "gt": c > n}[cmp]
        assert ok, (
            f"Op '{op_name}': expected {cmp} {n}"
            + (f" inside '{parent}'" if parent else "")
            + f", found {c}"
        )

    def assert_attr(self, op_name: str, attr_name: str, parent: str = None):
        """Assert at least one matching op has *attr_name* as a non-None string attribute."""
        matches = self._find(op_name, parent)
        assert matches, f"Op '{op_name}' not found in KTIR"
        assert any(o._op.get_str_attr(attr_name) is not None for o in matches), (
            f"Op '{op_name}' missing attribute '{attr_name}'"
        )

    def assert_affine_attr(self, op_name: str, attr_name: str, parent: str = None):
        """Assert at least one matching op has *attr_name* as an IntegerSet or AffineMap attribute."""
        from triton._C.libtriton import spyre
        matches = self._find(op_name, parent)
        assert matches, f"Op '{op_name}' not found in KTIR"

        def _has(op):
            return (
                spyre.ir_utils.get_integer_set_attr(op._op, attr_name) is not None
                or spyre.ir_utils.get_affine_map_attr(op._op, attr_name) is not None
            )

        assert any(_has(o) for o in matches), (
            f"Op '{op_name}' missing affine attribute '{attr_name}'"
        )

    def assert_integer_set(self, op_name: str, attr_name: str, *,
                           num_symbols: int = None, num_dims: int = None,
                           num_constraints: int = None, parent: str = None):
        """Assert properties of an IntegerSet attribute on a matching op."""
        from triton._C.libtriton import spyre
        matches = self._find(op_name, parent)
        assert matches, f"Op '{op_name}' not found in KTIR"

        def _parse(op):
            s = spyre.ir_utils.get_integer_set_attr(op._op, attr_name)
            if s is None:
                return None
            dm = re.match(r'\(([^)]*)\)', s)
            dims = len(dm.group(1).split(',')) if dm and dm.group(1).strip() else 0
            sm = re.search(r'\[([^\]]*)\]', s)
            syms = len(sm.group(1).split(',')) if sm and sm.group(1).strip() else 0
            cm = re.search(r':\s*\(([^)]*)\)', s)
            cons = len(cm.group(1).split(',')) if cm and cm.group(1).strip() else 0
            return {"dims": dims, "syms": syms, "cons": cons}

        parsed = [_parse(o) for o in matches]
        parsed = [p for p in parsed if p is not None]
        assert parsed, f"Op '{op_name}' has no integer set attribute '{attr_name}'"

        def _matches(p):
            if num_symbols is not None and p["syms"] != num_symbols:
                return False
            if num_dims is not None and p["dims"] != num_dims:
                return False
            if num_constraints is not None and p["cons"] != num_constraints:
                return False
            return True

        assert any(_matches(p) for p in parsed), (
            f"Op '{op_name}' attr '{attr_name}': no matching op satisfied "
            f"num_symbols={num_symbols}, num_dims={num_dims}, "
            f"num_constraints={num_constraints}; got {parsed}"
        )

    def assert_result_type(self, op_name: str, type_substr: str,
                           parent: str = None):
        """Assert at least one result type of a matching op contains *type_substr*."""
        matches = self._find(op_name, parent)
        assert matches, f"Op '{op_name}' not found in KTIR"
        assert any(
            type_substr in t for o in matches for t in o.result_types
        ), f"Op '{op_name}' has no result type containing '{type_substr}'"

    def assert_tile_future_groups(self, op_name: str, *,
                                  num_symbols: int = None, num_dims: int = None,
                                  num_constraints: int = None, parent: str = None):
        """Assert properties of the `groups` integer set carried by an inter-tile op's
        associated `!ktdp.tile_future` type.

        In PR-25 of ktir-mlir-frontend, the `groups` integer set moved off
        `ktdp.inter_tile_produce` / `ktdp.inter_tile_reduce` as an op attribute
        and became a type parameter of `!ktdp.tile_future<(...), groups = #set>`.
        The set now lives on the produce op's result type (single tile_future
        result) or the reduce op's operand 0 type (its future input). This
        helper looks in the correct place based on *op_name*.

        Parameters
        ----------
        op_name          : `"ktdp.inter_tile_produce"` or `"ktdp.inter_tile_reduce"`
        num_symbols, num_dims, num_constraints
                         : same semantics as :meth:`assert_integer_set` —
                           parsed from the affine_set text.
        parent           : optional parent op filter (see :meth:`_find`).
        """
        matches = self._find(op_name, parent)
        assert matches, f"Op '{op_name}' not found in KTIR"

        if op_name == "ktdp.inter_tile_produce":
            # groups lives in the op's single result type (a tile_future).
            def _type_str(op):
                if op._op.get_num_results() == 0:
                    return None
                return str(op._op.get_result(0).get_type())
        elif op_name == "ktdp.inter_tile_reduce":
            # groups lives in the type of operand 0 (the future).
            def _type_str(op):
                if op._op.get_num_operands() == 0:
                    return None
                return str(op._op.get_operand(0).get_type())
        else:
            raise ValueError(
                f"assert_tile_future_groups: expected 'ktdp.inter_tile_produce' "
                f"or 'ktdp.inter_tile_reduce', got '{op_name}'"
            )

        def _parse(op):
            ty = _type_str(op)
            if ty is None or "tile_future" not in ty:
                return None
            # Match: groups = affine_set<(d1, d2)[s1, s2] : (c1, c2, c3)>
            m = re.search(
                r"groups\s*=\s*affine_set<\s*\(([^)]*)\)\s*(?:\[([^\]]*)\])?\s*:\s*\(([^)]*)\)\s*>",
                ty,
            )
            if not m:
                return None
            dims_txt, syms_txt, cons_txt = m.group(1), m.group(2) or "", m.group(3)
            dims = len(dims_txt.split(",")) if dims_txt.strip() else 0
            syms = len(syms_txt.split(",")) if syms_txt.strip() else 0
            cons = len(cons_txt.split(",")) if cons_txt.strip() else 0
            return {"dims": dims, "syms": syms, "cons": cons}

        parsed = [p for p in (_parse(o) for o in matches) if p is not None]
        assert parsed, (
            f"Op '{op_name}' has no !ktdp.tile_future<..., groups = ...> "
            f"in its associated type (looked at "
            f"{'result 0' if op_name == 'ktdp.inter_tile_produce' else 'operand 0'})"
        )

        def _matches(p):
            if num_symbols is not None and p["syms"] != num_symbols:
                return False
            if num_dims is not None and p["dims"] != num_dims:
                return False
            if num_constraints is not None and p["cons"] != num_constraints:
                return False
            return True

        assert any(_matches(p) for p in parsed), (
            f"Op '{op_name}' tile_future groups: no match for "
            f"num_symbols={num_symbols}, num_dims={num_dims}, "
            f"num_constraints={num_constraints}; got {parsed}"
        )

    def assert_has_region(self, op_name: str, parent: str = None):
        """Assert at least one matching op owns at least one region."""
        matches = self._find(op_name, parent)
        assert matches, f"Op '{op_name}' not found in KTIR"
        assert any(o.num_regions > 0 for o in matches), (
            f"Op '{op_name}' has no regions"
        )

    def assert_operand(self, op_name: str, operand_idx: int, *,
                       value=None, defined_by: str = None,
                       type_substr: str = None, parent: str = None):
        """Assert a property of operand *operand_idx* on at least one matching op."""
        matches = self._find(op_name, parent)
        assert matches, f"Op '{op_name}' not found in KTIR"

        def _info(o):
            if operand_idx >= o._op.get_num_operands():
                return None
            v = o._op.get_operand(operand_idx)
            defop = self.get_defining_op(o, operand_idx)
            return {
                "type": str(v.get_type()),
                "const_value": defop._op.get_constant_value() if defop else None,
                "defined_by": defop.name if defop else None,
            }

        infos = [_info(o) for o in matches]
        infos = [i for i in infos if i is not None]
        assert infos, f"Op '{op_name}' has no operand at index {operand_idx}"

        def _matches(i):
            if value is not None and i["const_value"] != value:
                return False
            if defined_by is not None and i["defined_by"] != defined_by:
                return False
            if type_substr is not None and type_substr not in i["type"]:
                return False
            return True

        assert any(_matches(i) for i in infos), (
            f"Op '{op_name}' operand {operand_idx}: no matching op satisfied "
            f"value={value!r}, defined_by={defined_by!r}, type_substr={type_substr!r}; "
            f"got {infos}"
        )

    def assert_result(self, op_name: str, *, idx: int = 0,
                      shape=None, shape_not=None,
                      parent: str = None, elem_type: str = None):
        """Assert a property of result *idx* on at least one matching op."""
        matches = self._find(op_name, parent=parent, elem_type=elem_type)
        assert matches, f"Op '{op_name}' not found in KTIR"
        if shape is not None:
            found = self._find(op_name, parent=parent, elem_type=elem_type,
                               shape=shape, idx=idx)
            assert found, (
                f"Op '{op_name}': no result with shape {shape} found"
            )
        for o in matches:
            info = self.get_result_info(o, idx)
            assert info is not None, (
                f"Op '{o.name}': result {idx} does not exist"
            )
            if shape_not is not None:
                assert info.get("shape") != shape_not, (
                    f"Op '{o.name}': shape must not be {shape_not} "
                    f"(type: {info['type_str']})"
                )

    def assert_stderr(self, capfd, *substrings: str):
        """Assert each *substring* appears in captured stderr.

        ``capfd`` is pytest's C-level fd capture fixture. MLIR diagnostics
        are written to stderr from C++ via ``llvm::errs()``, so fd-level
        capture is required.
        """
        stderr = capfd.readouterr().err
        for s in substrings:
            assert s in stderr, (
                f"Expected '{s}' in stderr, got:\n{stderr}"
            )
