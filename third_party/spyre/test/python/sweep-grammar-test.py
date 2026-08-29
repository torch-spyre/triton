# RUN: %python -m pytest %s -q

# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the ``_load_examples`` helpers in ``conftest``.

A variant's ``params`` maps each argument name to a *list* of values. A param
with more than one value is **swept**: every value becomes its own registry
entry, and each key gets a ``[name=value, ...]`` suffix to tell them apart.

A value is written either plainly (``64``) or as a **labelled** pair
(``("tile64", 64)``), and the label is what goes in the key. Labels exist for
values whose repr would make a key unreadable — a stick layout is a nested
tuple — so the author names them instead.

That gives the rule these tests pin: a param appears in the suffix when it is
swept **or** when it was written with a label. The second half is why a param
with only one value can still show up in a key.

Three helpers, bottom-up: ``_normalise_param_list`` turns one param's list into
``(label, value)`` pairs; ``_expand_params`` takes the Cartesian product across
params; ``_sweep_suffix`` renders one combination's suffix. Then the *factory
protocol*, where a variant's ``"factory"`` key supplies the fields that vary
with the combination — so ``OP`` × ``DTYPE`` is one declaration, not twelve.

A lit test rather than a pytest module: it needs no dbo-opt and no device, and a
test that needs nothing should not sit in the suite that serializes the device.
"""

from dataclasses import dataclass

import pytest

from conftest import (
    VariantFactory,
    _apply_factory,
    _expand_params,
    _normalise_param_list,
    _sweep_suffix,
)


# ---------------------------------------------------------------------------
# _normalise_param_list — one param's value list → [(label, value), ...]
#
# Puts both spellings into one form so nothing downstream has to ask which was
# used. The label it invents for a plain value is what ends up in the key.
# ---------------------------------------------------------------------------

def test_plain_values_are_labelled_with_their_own_repr():
    """``str(value)`` becomes the label, so a plain param needs no naming to
    show up readably in a key."""
    assert _normalise_param_list("M", [128, 256]) == [("128", 128), ("256", 256)]


def test_plain_strings_are_labelled_with_themselves():
    """No quoting is added, so the key reads ``[dtype=f32]`` and not
    ``[dtype='f32']``."""
    assert _normalise_param_list("dtype", ["f32"]) == [("f32", "f32")]


def test_labelled_pairs_keep_the_author_s_label():
    """Already in pair form, so the list passes straight through — choosing the
    label is the author's job, not this function's."""
    values = [("small", 16), ("large", 512)]
    assert _normalise_param_list("N", values) == values


def test_a_lone_labelled_pair_is_legal():
    """One value is still allowed a label. That is the whole point of labels for
    unreadable values, and it is what makes the param show up unswept."""
    assert _normalise_param_list("BLOCK", [("tile64", 64)]) == [("tile64", 64)]


def test_an_empty_value_list_normalises_to_empty():
    """Not an error here. It becomes zero combinations upstream, which is where
    a variant that declares a param but no value for it gets caught."""
    assert _normalise_param_list("X", []) == []


def test_mixing_plain_and_labelled_values_is_refused():
    """Half-labelled lists are almost always a typo, and there is no sensible
    reading of one: the plain value's label would have to be invented while its
    sibling's was chosen."""
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _normalise_param_list("M", [128, ("big", 512)])


def test_the_mixing_error_names_the_param():
    """A fixture sweeps a dozen params, so the message has to say which one or
    the author is left bisecting the dict."""
    with pytest.raises(ValueError, match=r"params\[.M.\]"):
        _normalise_param_list("M", [("a", 1), 2])


def test_the_mixing_error_names_the_fixture_when_given_one():
    """``kernel_name`` is optional — the helper is unit-testable without it — but
    ``_load_examples`` always passes it, so a real failure names the fixture."""
    with pytest.raises(ValueError, match="myfixture"):
        _normalise_param_list("M", [("a", 1), 2], kernel_name="myfixture")


def test_a_non_string_label_is_refused():
    """The label is pasted into a registry key, so it has to be text."""
    with pytest.raises(ValueError, match="must be .str, value."):
        _normalise_param_list("K", [(42, 128)])


def test_a_tuple_that_is_not_a_pair_is_refused():
    """Only a 2-tuple is a pair. A longer one is a tuple-valued param that
    needed wrapping in a pair of its own, which is easy to forget."""
    with pytest.raises(ValueError, match="must be .str, value."):
        _normalise_param_list("K", [("a", 1, 2)])


# ---------------------------------------------------------------------------
# _expand_params — all params → (combinations, names that used labels)
#
# One combination per registry entry. The second return value is the only
# record of which spelling was used, and _sweep_suffix needs it.
# ---------------------------------------------------------------------------

def test_one_value_gives_one_combination():
    """The state every fixture was in before sweeps existed: a single entry, and
    nothing reported as labelled."""
    combos, labelled = _expand_params({"M": [64]})
    assert combos == [{"M": ("64", 64)}]
    assert labelled == set()


def test_each_value_gives_its_own_combination():
    """In the order written, which is what keeps generated keys stable across
    edits to unrelated params."""
    combos, labelled = _expand_params({"M": [1, 2, 3]})
    assert [c["M"][1] for c in combos] == [1, 2, 3]
    assert labelled == set()


def test_params_are_crossed_not_zipped():
    """Two swept params give every pairing rather than the positional ones — so
    ``OP`` × ``DTYPE`` covers all twelve, not four."""
    combos, _ = _expand_params({"M": [1, 2], "K": [3, 4]})
    assert {(c["M"][1], c["K"][1]) for c in combos} == {
        (1, 3), (1, 4), (2, 3), (2, 4)}


def test_a_labelled_param_is_reported_as_labelled():
    """Being in this set is what later lets a one-value param into the key; the
    combination alone no longer shows which spelling produced it."""
    combos, labelled = _expand_params({"BLOCK": [("t64", 64)]})
    assert combos == [{"BLOCK": ("t64", 64)}]
    assert labelled == {"BLOCK"}


def test_only_the_params_that_used_labels_are_reported():
    """``M`` was labelled and ``K`` was not, so only ``M`` earns the right to
    appear in a key when it is down to one value."""
    combos, labelled = _expand_params({
        "M": [("small", 16), ("large", 256)],
        "K": [32, 64],
    })
    assert len(combos) == 4
    assert labelled == {"M"}


def test_no_params_gives_one_empty_combination():
    """One, not zero — a variant that sweeps nothing is still a registry entry,
    and returning zero would silently drop it."""
    assert _expand_params({}) == ([{}], set())


def test_expansion_refuses_a_mixed_list_too():
    """The check itself lives in ``_normalise_param_list``, but it has to fire
    during expansion rather than leaving a malformed pair to surface later as a
    confusing key."""
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _expand_params({"M": [1, ("big", 512)]})


# ---------------------------------------------------------------------------
# _sweep_suffix — one combination → "[k=v, ...]"
#
# Decides which params are worth naming in a key. Everything above exists to
# feed this: a param is named when it is swept, or when it was labelled.
# ---------------------------------------------------------------------------

def _suffix(params: dict, index: int = 0) -> str:
    """The key suffix the *index*-th combination of *params* would get.

    Wires the three helpers together the way ``_load_examples`` does, so each
    test below can state a ``params`` dict and the suffix it should produce.
    """
    combos, labelled = _expand_params(params)
    normalised = {k: _normalise_param_list(k, v) for k, v in params.items()}
    return _sweep_suffix(normalised, combos[index], labelled)


def test_an_unswept_plain_param_adds_no_suffix():
    """One value and no label means there is nothing to disambiguate, so the key
    stays bare however many such params there are."""
    assert _suffix({"M": [64]}) == ""
    assert _suffix({"BLOCK": [64], "M": [128]}) == ""


def test_a_swept_plain_param_shows_its_value():
    """Each combination names its own value — that is what makes the sibling
    keys distinct rather than colliding."""
    assert _suffix({"M": [64, 128]}) == "[M=64]"
    assert _suffix({"M": [64, 128]}, index=1) == "[M=128]"


def test_a_labelled_param_shows_even_unswept():
    """Writing a label is a request to be visible, and having only one value
    does not withdraw it."""
    assert _suffix({"BLOCK": [("tile64", 64)]}) == "[BLOCK=tile64]"


def test_a_swept_labelled_param_shows_its_label_not_its_value():
    """The reason labels exist: a stick layout's repr in a key would be
    unreadable, so the author's short name stands in for it."""
    params = {"N": [("small", 16), ("large", 256)]}
    assert _suffix(params) == "[N=small]"
    assert _suffix(params, index=1) == "[N=large]"


def test_swept_and_labelled_params_both_appear():
    """The two reasons for inclusion are independent, so a key can carry one of
    each — here ``N`` because it is swept and ``TAG`` because it is labelled."""
    params = {"TAG": [("fast", 1)], "N": [4, 8]}
    assert _suffix(params) == "[N=4, TAG=fast]"
    assert _suffix(params, index=1) == "[N=8, TAG=fast]"


def test_suffix_params_are_sorted_by_name():
    """Sorted rather than left in dict order, so reordering the ``params`` dict
    does not rename every key the fixture generates."""
    assert _suffix({"Z": [1, 2], "A": [10, 20]}) == "[A=10, Z=1]"


# ---------------------------------------------------------------------------
# The registry keys the three helpers produce together
#
# What a fixture author actually sees, and what test IDs are made of.
# ---------------------------------------------------------------------------

def _keys(variant_name: str, params: dict) -> list[str]:
    """The registry keys ``_load_examples`` would emit for one variant.

    A re-implementation of its key-building loop, not a call into it — that
    would need the fixtures directory and a Triton build.
    """
    combos, labelled = _expand_params(params, kernel_name=variant_name)
    normalised = {k: _normalise_param_list(k, v) for k, v in params.items()}
    return [variant_name + _sweep_suffix(normalised, combo, labelled)
            for combo in combos]


def test_a_variant_with_nothing_swept_keeps_its_bare_name():
    """The case every fixture predating sweeps is in, so it has to stay exactly
    as it was — a suffix here would rename keys other tests reference."""
    assert _keys("elementwise", {"N": [64], "BLOCK": [32]}) == ["elementwise"]


def test_a_swept_param_gives_one_key_per_value():
    """The registry gains a separate entry per value, so each shape is its own
    test rather than one test looping internally."""
    assert _keys("matmul", {"M": [128, 256]}) == [
        "matmul[M=128]", "matmul[M=256]"]


def test_a_labelled_param_names_itself_in_the_key():
    """Still one entry, but a named one — so a failure report says which layout
    was under test instead of just naming the fixture."""
    assert _keys("softmax", {"BLOCK": [("tile32", 32)]}) == [
        "softmax[BLOCK=tile32]"]


def test_two_swept_params_give_a_key_per_pairing():
    """Entry order is first-param-outermost, from the product; within each key
    the suffix sorts by name. Both are pinned here because test IDs are built
    from them."""
    keys = _keys("kernel", {
        "M": [("s", 16), ("l", 64)],
        "K": [("a", 4), ("b", 8)],
    })
    assert keys == [
        "kernel[K=a, M=s]", "kernel[K=b, M=s]",
        "kernel[K=a, M=l]", "kernel[K=b, M=l]",
    ]


def test_a_malformed_param_list_raises_before_any_key_is_built():
    """Collection fails whole rather than registering the keys it managed to
    build, which would leave a fixture half-present."""
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _keys("bad_kernel", {"X": [1, ("label", 2)]})


# ---------------------------------------------------------------------------
# _apply_factory — the fields a VariantFactory supplies per combination
#
# Runs once per combination, mutating that combination's entry in place, before
# _resolve_variant reads SIGNATURE out of it.
# ---------------------------------------------------------------------------

def _apply(entry: dict, values: dict) -> dict:
    """Run ``_apply_factory`` on *entry* for a combination of plain *values*.

    ``_apply_factory`` consumes ``(label, value)`` pairs, so the values are
    wrapped back up here; what the label is does not matter to it.
    """
    combo = {k: (str(v), v) for k, v in values.items()}
    _apply_factory(entry, combo, kernel_name="fix::variant")
    return entry


def test_an_entry_without_a_factory_is_untouched():
    """Every fixture predating the protocol takes this path, so it is the one
    that must not regress."""
    entry = {"kernel_fn": object(), "params": {"M": [64]}, "reference": len}
    before = dict(entry)
    _apply(entry, {"M": 64})
    assert entry == before


def test_each_hook_sets_the_field_it_is_named_for():
    """``signature()`` fills ``SIGNATURE`` — the one case where hook and field
    names differ, because the field is uppercase by fixture convention."""

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, DTYPE, **_):
            return {"x_ptr": f"*{DTYPE}"}

        def reference(self, OP, **_):
            return f"oracle_{OP}"

        def inputs(self, DTYPE, **_):
            return f"inputs_{DTYPE}"

    entry = _apply({"factory": F()}, {"DTYPE": "fp16", "OP": "add"})
    assert entry["SIGNATURE"] == {"x_ptr": "*fp16"}
    assert entry["reference"] == "oracle_add"
    assert entry["inputs"] == "inputs_fp16"


def test_a_hook_returning_none_leaves_its_field_unset():
    """So a factory can supply one field and stay silent on the others, whether
    by declining for this combination (``inputs``) or by not being overridden at
    all (``reference``). Unset matters: the field then falls back to the
    module-level default rather than being ``None``."""

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, **_):
            return {"x_ptr": "*fp32"}

        def inputs(self, DTYPE, **_):
            return None if DTYPE == "fp32" else "gen"

    entry = _apply({"factory": F()}, {"DTYPE": "fp32"})
    assert "SIGNATURE" in entry
    assert "inputs" not in entry
    assert "reference" not in entry


def test_hooks_are_given_values_not_label_pairs():
    """A hook reasons about ``DTYPE="fp16"``; the label only names the
    combination in a key. Passing pairs through would make every hook unpack
    ``[1]`` before it could use its argument."""
    seen = {}

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, **combo):
            seen.update(combo)
            return {"x_ptr": "*fp32"}

    combo = {"DTYPE": ("f16", "fp16"), "LAYOUT": ("stick", [(1, "floordiv", 64)])}
    _apply_factory({"factory": F()}, combo, kernel_name="fix::variant")
    assert seen == {"DTYPE": "fp16", "LAYOUT": [(1, "floordiv", 64)]}


def test_a_hook_colliding_with_its_literal_field_raises():
    """Refused rather than resolved by precedence: whichever won would be a fact
    about statement order inside ``_apply_factory``, not something the fixture
    author could read off their own variant."""

    @dataclass(frozen=True)
    class F(VariantFactory):
        def reference(self, **_):
            return "from_hook"

    with pytest.raises(ValueError, match="fix::variant.*'reference'.*reference"):
        _apply({"factory": F(), "reference": "literal"}, {"M": 64})


def test_a_literal_field_whose_hook_is_unused_is_kept():
    """The collision is per field, so the two styles mix on one variant — a
    factory for what varies, literals for what does not."""

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, **_):
            return {"x_ptr": "*fp32"}

    entry = _apply({"factory": F(), "reference": "literal"}, {"M": 64})
    assert entry["reference"] == "literal"
    assert entry["SIGNATURE"] == {"x_ptr": "*fp32"}


def test_a_bare_callable_is_not_accepted_as_a_factory():
    """Hooks are looked up as methods, so a plain function has nowhere to put
    them — and accepting one would mean guessing at its parameters instead."""
    with pytest.raises(TypeError, match="must be a VariantFactory"):
        _apply({"factory": lambda **kw: None}, {"M": 64})
