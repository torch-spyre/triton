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

"""Unit tests for conftest sweep-grammar helpers.

Tests ``_normalise_param_list``, ``_expand_params``, and ``_sweep_suffix``
in isolation without touching the filesystem or importing Triton.
"""

import pytest

from conftest import _expand_params, _normalise_param_list, _sweep_suffix


# ---------------------------------------------------------------------------
# _normalise_param_list
# ---------------------------------------------------------------------------

def test_normalise_plain_ints():
    result = _normalise_param_list("M", [128, 256, 512])
    assert result == [("128", 128), ("256", 256), ("512", 512)]


def test_normalise_plain_str():
    result = _normalise_param_list("dtype", ["f32", "f16"])
    assert result == [("f32", "f32"), ("f16", "f16")]


def test_normalise_labelled_tuples():
    values = [("small", 16), ("large", 512)]
    result = _normalise_param_list("N", values)
    assert result == [("small", 16), ("large", 512)]


def test_normalise_single_labelled_tuple():
    values = [("tile64", 64)]
    result = _normalise_param_list("BLOCK", values)
    assert result == [("tile64", 64)]


def test_normalise_mixed_raises():
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _normalise_param_list("M", [128, ("big", 512)])


def test_normalise_mixed_raises_includes_param_name():
    with pytest.raises(ValueError, match=r"params\[.M.\]"):
        _normalise_param_list("M", [("a", 1), 2])


def test_normalise_mixed_raises_includes_kernel_name():
    with pytest.raises(ValueError, match="myfixture"):
        _normalise_param_list("M", [("a", 1), 2], kernel_name="myfixture")


def test_normalise_non_string_label_raises():
    with pytest.raises(ValueError, match="must be .str, value."):
        _normalise_param_list("K", [(42, 128)])


def test_normalise_wrong_tuple_length_raises():
    with pytest.raises(ValueError, match="must be .str, value."):
        _normalise_param_list("K", [("a", 1, 2)])


def test_normalise_empty_list():
    result = _normalise_param_list("X", [])
    assert result == []


# ---------------------------------------------------------------------------
# _expand_params
# ---------------------------------------------------------------------------

def test_expand_single_plain_value():
    combos, always_suffixed = _expand_params({"M": [64]})
    assert len(combos) == 1
    assert combos[0] == {"M": ("64", 64)}
    assert always_suffixed == set()


def test_expand_multi_plain_values():
    combos, always_suffixed = _expand_params({"M": [1, 2, 3]})
    assert len(combos) == 3
    assert combos[0] == {"M": ("1", 1)}
    assert combos[2] == {"M": ("3", 3)}
    assert always_suffixed == set()


def test_expand_single_labelled_tuple():
    combos, always_suffixed = _expand_params({"BLOCK": [("t64", 64)]})
    assert len(combos) == 1
    assert combos[0] == {"BLOCK": ("t64", 64)}
    assert "BLOCK" in always_suffixed


def test_expand_multi_labelled_tuples():
    combos, always_suffixed = _expand_params({"N": [("a", 1), ("b", 2)]})
    assert len(combos) == 2
    assert combos[0] == {"N": ("a", 1)}
    assert combos[1] == {"N": ("b", 2)}
    assert "N" in always_suffixed


def test_expand_cartesian_product_two_params():
    combos, always_suffixed = _expand_params({"M": [1, 2], "K": [3, 4]})
    assert len(combos) == 4
    assert always_suffixed == set()
    # Extract (M-value, K-value) pairs from combos
    pairs = {(c["M"][1], c["K"][1]) for c in combos}
    assert pairs == {(1, 3), (1, 4), (2, 3), (2, 4)}


def test_expand_cartesian_product_mixed_label_and_plain():
    combos, always_suffixed = _expand_params({
        "M": [("small", 16), ("large", 256)],
        "K": [32, 64],
    })
    assert len(combos) == 4
    assert "M" in always_suffixed
    assert "K" not in always_suffixed


def test_expand_mixed_raises():
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _expand_params({"M": [1, ("big", 512)]})


def test_expand_empty_params():
    combos, always_suffixed = _expand_params({})
    # itertools.product() of zero sequences yields one empty combo
    assert combos == [{}]
    assert always_suffixed == set()


# ---------------------------------------------------------------------------
# _sweep_suffix
# ---------------------------------------------------------------------------

def _make_combo(params: dict) -> tuple[dict, dict, set]:
    """Helper: normalise params, return (merged_params_normalised, first_combo, always_suffixed)."""
    combos, always_suffixed = _expand_params(params)
    # merged_params for _sweep_suffix is the normalised lists
    from conftest import _normalise_param_list
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}
    return merged, combos[0], always_suffixed


def test_suffix_no_swept_params():
    merged, combo, always_suffixed = _make_combo({"M": [64]})
    assert _sweep_suffix(merged, combo, always_suffixed) == ""


def test_suffix_single_plain_multi_value():
    merged, combo, always_suffixed = _make_combo({"M": [64, 128]})
    # M has len > 1, so it appears in suffix; first combo M=64
    suffix = _sweep_suffix(merged, combo, always_suffixed)
    assert suffix == "[M=64]"


def test_suffix_single_labelled_single_value_always_shown():
    merged, combo, always_suffixed = _make_combo({"BLOCK": [("tile64", 64)]})
    suffix = _sweep_suffix(merged, combo, always_suffixed)
    assert suffix == "[BLOCK=tile64]"


def test_suffix_labelled_multi_value_uses_custom_labels():
    params = {"N": [("small", 16), ("large", 256)]}
    combos, always_suffixed = _expand_params(params)
    from conftest import _normalise_param_list
    merged = {"N": _normalise_param_list("N", params["N"])}

    assert _sweep_suffix(merged, combos[0], always_suffixed) == "[N=small]"
    assert _sweep_suffix(merged, combos[1], always_suffixed) == "[N=large]"


def test_suffix_multiple_swept_params_sorted_alphabetically():
    params = {"Z": [1, 2], "A": [10, 20]}
    combos, always_suffixed = _expand_params(params)
    from conftest import _normalise_param_list
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}

    suffix = _sweep_suffix(merged, combos[0], always_suffixed)
    # "A" before "Z" alphabetically
    assert suffix.startswith("[A=")
    assert "Z=" in suffix
    assert suffix.index("A=") < suffix.index("Z=")


def test_suffix_plain_single_value_no_suffix():
    params = {"BLOCK": [64], "M": [128]}
    merged, combo, always_suffixed = _make_combo(params)
    assert _sweep_suffix(merged, combo, always_suffixed) == ""


def test_suffix_mixed_labelled_and_plain_multi():
    """Labelled single-value + plain multi-value both appear in suffix."""
    params = {"TAG": [("fast", 1)], "N": [4, 8]}
    combos, always_suffixed = _expand_params(params)
    from conftest import _normalise_param_list
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}

    suffix0 = _sweep_suffix(merged, combos[0], always_suffixed)
    suffix1 = _sweep_suffix(merged, combos[1], always_suffixed)
    # N=4 and N=8 differ; TAG always shown as "fast"
    assert "TAG=fast" in suffix0
    assert "N=4" in suffix0
    assert "TAG=fast" in suffix1
    assert "N=8" in suffix1


# ---------------------------------------------------------------------------
# Integration: simulate _load_examples pipeline without filesystem
# ---------------------------------------------------------------------------

def _simulate_registry_keys(kernel_name: str, params: dict) -> list[str]:
    """Simulate the key generation logic in _load_examples for a single variant."""
    combos, always_suffixed = _expand_params(params, kernel_name=kernel_name)
    from conftest import _normalise_param_list
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}

    keys = []
    for combo in combos:
        suffix = _sweep_suffix(merged, combo, always_suffixed)
        keys.append(kernel_name + suffix)
    return keys


def test_integration_single_plain_no_suffix():
    keys = _simulate_registry_keys("vector_add", {"N": [64], "BLOCK": [32]})
    assert keys == ["vector_add"]


def test_integration_multi_plain_suffix_with_value():
    keys = _simulate_registry_keys("matmul", {"M": [128, 256]})
    assert keys == ["matmul[M=128]", "matmul[M=256]"]


def test_integration_labelled_single_always_suffix():
    keys = _simulate_registry_keys("softmax", {"BLOCK": [("tile32", 32)]})
    assert keys == ["softmax[BLOCK=tile32]"]


def test_integration_cartesian_two_labelled():
    keys = _simulate_registry_keys("kernel", {
        "M": [("s", 16), ("l", 64)],
        "K": [("a", 4), ("b", 8)],
    })
    assert len(keys) == 4
    # Suffix params are sorted alphabetically: K before M
    assert "kernel[K=a, M=s]" in keys
    assert "kernel[K=b, M=l]" in keys


def test_integration_mixed_error_propagates():
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _simulate_registry_keys("bad_kernel", {"X": [1, ("label", 2)]})
