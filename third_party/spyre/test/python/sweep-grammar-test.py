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

A ``params`` key is also allowed to be a **tuple of names**, whose value is a
list of rows rather than of values. Then those names are swept *jointly*: only
the rows written are enumerated, never their product. That states a functional
dependency — the ``N`` a dtype implies sits on the same row as the dtype, so an
fp16 width beside an fp32 dtype is unrepresentable — and it skips an invalid
combination by the same means, which is to not list it.

Four helpers, bottom-up: ``_normalise_param_list`` turns one param's list into
``(label, value)`` pairs; ``_normalise_param_group`` does the same per column of
a tuple key's rows; ``_expand_params`` takes the Cartesian product across keys
and decides which names a key is worth naming for; ``_sweep_suffix`` renders one
combination's suffix. Then the *factory protocol*, where a variant's
``"factory"`` key supplies the fields that vary with the combination — so
``OP`` × ``DTYPE`` is one declaration, not twelve.

A lit test rather than a pytest module: it needs no dbo-opt and no device, and a
test that needs nothing should not sit in the suite that serializes the device.
"""

from dataclasses import dataclass

import pytest

from conftest import (
    VariantFactory,
    _apply_factory,
    _expand_params,
    _normalise_param_group,
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
# _normalise_param_group — a tuple key's rows → the same pairs, per column
#
# Transposes the rows into columns, hands each column to
# _normalise_param_list, transposes back. So a group member is labelled by
# exactly the rule a scalar param is, and the validation here is only about the
# shape a row has to have to be transposable at all.
# ---------------------------------------------------------------------------

def test_a_group_normalises_each_column_the_way_a_scalar_is():
    """Column-wise, not row-wise: labelling is a property of the argument, so
    ``DTYPE`` gets its own repr as a label while ``LAYOUT`` keeps the author's."""
    rows, labelled = _normalise_param_group(
        ("DTYPE", "LAYOUT"),
        [("fp16", ("stick", 64)), ("fp32", ("stick", 32))],
    )
    assert rows == [
        (("fp16", "fp16"), ("stick", 64)),
        (("fp32", "fp32"), ("stick", 32)),
    ]
    assert labelled == {"LAYOUT"}


def test_group_rows_keep_the_order_they_were_written_in():
    """The rows are what the registry enumerates, so their order is the order of
    the keys a fixture generates."""
    rows, _ = _normalise_param_group(("N",), [(16,), (8,), (32,)])
    assert [row[0][1] for row in rows] == [16, 8, 32]


def test_a_group_of_one_row_is_legal():
    """A group with nothing to vary is still the right spelling when the names
    belong together — a sibling variant pinning one row of its parent's sweep."""
    rows, labelled = _normalise_param_group(("DTYPE", "N"), [("fp16", 128)])
    assert rows == [(("fp16", "fp16"), ("128", 128))]
    assert labelled == set()


def test_a_group_with_no_rows_normalises_to_no_rows():
    """Like an empty scalar list: not an error here, but zero combinations
    upstream, which is where a group declared and never populated is caught."""
    assert _normalise_param_group(("A", "B"), []) == ([], set())


def test_a_row_shorter_than_the_group_is_refused():
    """Rows are positional, so a missing value does not leave a hole — it shifts
    every later column onto the wrong name, which no downstream check can see."""
    with pytest.raises(ValueError, match="has 2 value"):
        _normalise_param_group(("DTYPE", "N", "BLOCK_N"), [("fp16", 128)])


def test_a_row_longer_than_the_group_is_refused():
    """The same reason in the other direction, and the more likely typo: a name
    was dropped from the key while its value stayed in the rows."""
    with pytest.raises(ValueError, match="has 3 value"):
        _normalise_param_group(("DTYPE", "N"), [("fp16", 128, 128)])


def test_the_row_length_error_names_the_row():
    """A group is a list of rows that look alike, so naming the group is not
    enough — the message has to say which row is the odd one."""
    with pytest.raises(ValueError, match=r"\('fp32', 64\)"):
        _normalise_param_group(
            ("DTYPE", "N", "BLOCK_N"),
            [("fp16", 128, 128), ("fp32", 64)],
        )


def test_a_row_that_is_not_a_sequence_is_refused():
    """Writing the bare value instead of a one-element row is the easy mistake in
    a group of one name; it would otherwise be transposed character by character
    or not at all."""
    with pytest.raises(ValueError, match="must be a tuple or list"):
        _normalise_param_group(("DTYPE",), ["fp16"])


def test_a_non_string_name_in_a_group_is_refused():
    """A group key's elements are argument names. Anything else means the author
    wrote a row where the key belongs."""
    with pytest.raises(ValueError, match="is not a str"):
        _normalise_param_group(("DTYPE", 64), [("fp16", 128)])


def test_a_name_repeated_inside_a_group_is_refused():
    """Two columns for one argument: the second would win the ``combo.update``
    and the first would vanish silently."""
    with pytest.raises(ValueError, match=r"repeats \['N'\]"):
        _normalise_param_group(("N", "N"), [(128, 64)])


def test_a_group_column_obeys_the_all_or_nothing_labelling_rule():
    """Per column, which is the point of transposing first: a layout is labelled
    in every row or in none, and a half-labelled column is the same typo it is at
    scalar level."""
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _normalise_param_group(
            ("DTYPE", "LAYOUT"),
            [("fp16", ("stick", 64)), ("fp32", 32)],
        )


def test_a_group_error_names_the_fixture_and_variant():
    """``_load_examples`` passes ``fixture::variant``, and a group's failures are
    the ones most in need of it — a bad row says nothing about where it lives."""
    with pytest.raises(ValueError, match="elementwise::2d_device"):
        _normalise_param_group(("DTYPE", "N"), [("fp16",)],
                               kernel_name="elementwise::2d_device")


# ---------------------------------------------------------------------------
# _expand_params — all params → (combinations, names worth naming in a key)
#
# One combination per registry entry. The second return value is the set of
# names _sweep_suffix renders: computed here because only here is it known
# whether a name came from a scalar list or a group column, and the two rules
# differ.
# ---------------------------------------------------------------------------

def test_one_value_gives_one_combination():
    """The state every fixture was in before sweeps existed: a single entry, and
    nothing worth naming."""
    combos, suffixed = _expand_params({"M": [64]})
    assert combos == [{"M": ("64", 64)}]
    assert suffixed == set()


def test_each_value_gives_its_own_combination():
    """In the order written, which is what keeps generated keys stable across
    edits to unrelated params."""
    combos, suffixed = _expand_params({"M": [1, 2, 3]})
    assert [c["M"][1] for c in combos] == [1, 2, 3]
    assert suffixed == {"M"}


def test_params_are_crossed_not_zipped():
    """Two swept params give every pairing rather than the positional ones — so
    ``OP`` × ``DTYPE`` covers all twelve, not four."""
    combos, _ = _expand_params({"M": [1, 2], "K": [3, 4]})
    assert {(c["M"][1], c["K"][1]) for c in combos} == {
        (1, 3), (1, 4), (2, 3), (2, 4)}


def test_a_labelled_param_is_named_even_with_one_value():
    """Being in this set is what lets a one-value param into the key; the
    combination alone no longer shows which spelling produced it."""
    combos, suffixed = _expand_params({"BLOCK": [("t64", 64)]})
    assert combos == [{"BLOCK": ("t64", 64)}]
    assert suffixed == {"BLOCK"}


def test_an_unswept_unlabelled_param_is_not_named():
    """``K`` has one value and no label, so it stays out however many of its
    siblings are swept — a key names what distinguishes it, not the whole dict."""
    combos, suffixed = _expand_params({
        "M": [("small", 16), ("large", 256)],
        "K": [32],
    })
    assert len(combos) == 2
    assert suffixed == {"M"}


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


def test_a_group_enumerates_its_rows_rather_than_their_product():
    """The whole reason for the form. Two rows of two names give two
    combinations, not four — so the pairing the author wrote is the pairing that
    runs, and the two it did not write are unrepresentable."""
    combos, _ = _expand_params({("DTYPE", "N"): [("fp16", 128), ("fp32", 64)]})
    assert [(c["DTYPE"][1], c["N"][1]) for c in combos] == [
        ("fp16", 128), ("fp32", 64)]


def test_a_group_member_lands_as_an_ordinary_param():
    """Flattened into the combination under its own name, with no trace of the
    group. That is what lets the constexpr/runtime split, ``inputs()`` and the
    docs generator stay ignorant of the form entirely."""
    combos, _ = _expand_params({("DTYPE", "N"): [("fp16", 128)]})
    assert combos == [{"DTYPE": ("fp16", "fp16"), "N": ("128", 128)}]


def test_a_group_crosses_with_the_other_keys():
    """Joint *within* a group, Cartesian *across* keys — so a dependent pair
    still sweeps against an independent axis like ``OP``."""
    combos, _ = _expand_params({
        ("DTYPE", "N"): [("fp16", 128), ("fp32", 64)],
        "OP": ["add", "sub", "mul"],
    })
    assert len(combos) == 6
    assert {(c["DTYPE"][1], c["N"][1]) for c in combos} == {
        ("fp16", 128), ("fp32", 64)}


def test_a_varying_group_column_is_named_and_a_constant_one_is_not():
    """The asymmetry against scalars, stated: a group's *row* count says nothing
    about any single column. ``M`` repeated across both rows is not what tells the
    two entries apart, so it must stay out of the key."""
    _, suffixed = _expand_params({
        ("DTYPE", "N", "M"): [("fp16", 128, 64), ("fp32", 64, 64)],
    })
    assert suffixed == {"DTYPE", "N"}


def test_a_group_of_one_row_behaves_like_a_scalar_of_one_value():
    """Nothing distinguishes the single entry, so the key stays bare — the group
    form costs nothing when a sibling variant pins one row of its parent's
    sweep."""
    combos, suffixed = _expand_params({("DTYPE", "N"): [("fp16", 128)]})
    assert len(combos) == 1
    assert suffixed == set()


def test_a_labelled_group_column_is_named_even_when_constant():
    """A label is a request to be visible and holding still does not withdraw it,
    exactly as at scalar level. This is how the stick layouts stay in the key
    while the fp16/fp32 widths behind them differ."""
    _, suffixed = _expand_params({
        ("DTYPE", "LAYOUT"): [("fp16", ("stick", 64)), ("fp32", ("stick", 32))],
    })
    assert suffixed == {"DTYPE", "LAYOUT"}


def test_a_name_under_two_keys_is_refused():
    """``combo.update`` runs key by key, so one of the two values would be
    dropped and which one would be a fact about dict order."""
    with pytest.raises(ValueError, match=r"\['N'\]"):
        _expand_params({("DTYPE", "N"): [("fp16", 128)], "N": [64]})


def test_a_name_under_two_groups_is_refused():
    """Same rule with no scalar involved — an argument belongs to exactly one
    axis."""
    with pytest.raises(ValueError, match=r"\['N'\]"):
        _expand_params({
            ("DTYPE", "N"): [("fp16", 128)],
            ("OP", "N"): [("add", 64)],
        })


# ---------------------------------------------------------------------------
# _sweep_suffix — one combination → "[k=v, ...]"
#
# Renders the names _expand_params picked. It no longer sees ``params`` at all,
# so there is nowhere left for it to re-derive the rule from and disagree.
# ---------------------------------------------------------------------------

def _suffix(params: dict, index: int = 0) -> str:
    """The key suffix the *index*-th combination of *params* would get.

    Wires the helpers together the way ``_load_examples`` does, so each test
    below can state a ``params`` dict and the suffix it should produce.
    """
    combos, suffix_names = _expand_params(params)
    return _sweep_suffix(suffix_names, combos[index])


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
# The registry keys the helpers produce together
#
# What a fixture author actually sees, and what test IDs are made of.
# ---------------------------------------------------------------------------

def _keys(variant_name: str, params: dict) -> list[str]:
    """The registry keys ``_load_examples`` would emit for one variant.

    A re-implementation of its key-building loop, not a call into it — that
    would need the fixtures directory and a Triton build.
    """
    combos, suffix_names = _expand_params(params, kernel_name=variant_name)
    return [variant_name + _sweep_suffix(suffix_names, combo)
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


def test_a_group_gives_one_key_per_row():
    """Two rows, two keys, and both of the group's varying names in each — there
    is no driving axis a group nominates to stand for the rest."""
    assert _keys("dev", {("DTYPE", "N"): [("fp16", 128), ("fp32", 64)]}) == [
        "dev[DTYPE=fp16, N=128]", "dev[DTYPE=fp32, N=64]"]


def test_a_constant_group_column_stays_out_of_the_key():
    """``M`` is 64 on both rows, so naming it would pad every key with something
    that never varies — while ``N``, which does vary, has to be there or the two
    keys collide."""
    assert _keys("dev", {("DTYPE", "N", "M"): [("fp16", 128, 64),
                                              ("fp32", 64, 64)]}) == [
        "dev[DTYPE=fp16, N=128]", "dev[DTYPE=fp32, N=64]"]


def test_a_labelled_row_element_is_labelled_in_the_key():
    """Rows go through ``_normalise_param_list`` per column, so a layout is named
    in the key by the same label a scalar param would give it, rather than
    printing its nested repr."""
    assert _keys("dev", {
        ("DTYPE", "LAYOUT"): [("fp16", ("stick", ((0, "floordiv", 64),))),
                              ("fp32", ("stick", ((0, "floordiv", 32),)))],
    }) == ["dev[DTYPE=fp16, LAYOUT=stick]", "dev[DTYPE=fp32, LAYOUT=stick]"]


def test_a_labelled_row_element_reaches_the_kernel_as_its_bare_value():
    """The other half of the same property, and the class of bug worth pinning: a
    label names a value inside a key and must not survive into the value the
    combination carries, or the kernel gets ``("stick", layout)`` as its
    constexpr. Rows cannot get this wrong, because normalising per column is what
    splits the pair."""
    layout = ((0, "floordiv", 64), (0, "mod", 64))
    combos, _ = _expand_params({("DTYPE", "LAYOUT"): [("fp16", ("stick", layout))]})
    # What ``_load_examples`` writes back as the entry's ``params``.
    params = {k: [v[1]] for k, v in combos[0].items()}
    assert params == {"DTYPE": ["fp16"], "LAYOUT": [layout]}


def test_a_group_crossed_with_a_scalar_names_both():
    """What the device variants generate: a dependent pair against an independent
    ``OP``, product-ordered with the first key outermost."""
    assert _keys("dev", {
        ("DTYPE", "N"): [("fp16", 128), ("fp32", 64)],
        "OP": ["add", "sub"],
    }) == [
        "dev[DTYPE=fp16, N=128, OP=add]", "dev[DTYPE=fp16, N=128, OP=sub]",
        "dev[DTYPE=fp32, N=64, OP=add]", "dev[DTYPE=fp32, N=64, OP=sub]",
    ]


def test_a_malformed_group_row_raises_before_any_key_is_built():
    """Same whole-or-nothing rule as a malformed scalar list, and the message
    names the row so the author is not left bisecting the group."""
    with pytest.raises(ValueError, match=r"\('fp32',\)"):
        _keys("bad_kernel", {("DTYPE", "N"): [("fp16", 128), ("fp32",)]})


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
