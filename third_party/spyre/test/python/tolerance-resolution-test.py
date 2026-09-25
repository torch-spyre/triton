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

"""Unit tests for ``conftest.tolerances`` — one variant's ``rtol``/``atol``.

The two numerical tests compare a kernel's output against the same NumPy oracle,
one through ``ktir_cpu`` and one on the device, and both take their bounds from
the variant. ``tolerances`` is where a variant's ``rtol``/``atol`` become the
kwargs of that comparison, and it is one function because two copies of a
tolerance rule would drift without saying so.

Either key is written two ways. A **scalar** is the bound itself. A **dict keyed
by dtype string** is resolved against the variant's own ``DTYPE`` param, which is
what lets one entry sweep ``DTYPE`` and still give its arms different bounds --
tolerance being a property of the dtype, an fp16 ulp at a given magnitude is
thousands of times an fp32 one.

So the rules pinned here: a scalar passes through, an absent key -- or one
written as an explicit ``None`` -- takes the default, a dict selects by the
variant's dtype, and a dict that does not name that dtype raises. The last is the one worth a test of its own: falling back to
the default would mean an ``atol`` of 0 deciding a pass or a failure for a reason
nothing in the fixture states.

A lit test rather than a pytest module: it needs no dbo-opt and no device, and a
test that needs nothing should not sit in the suite that serializes the device.
"""

import pytest

from conftest import tolerances


def _entry(dtype=None, **fields):
    """A variant entry carrying just what ``tolerances`` reads — the tolerance
    fields, and the resolved ``param_values`` it takes ``DTYPE`` from."""
    entry = dict(fields)
    entry["param_values"] = {} if dtype is None else {"DTYPE": dtype}
    return entry


# ---------------------------------------------------------------------------
# Scalars and defaults — the path every existing fixture is on
# ---------------------------------------------------------------------------

def test_scalars_pass_through():
    """The spelling nearly every fixture uses, and the reason the dict is an
    addition rather than a migration."""
    assert tolerances(_entry(rtol=1e-2, atol=5e-2)) == {"rtol": 1e-2, "atol": 5e-2}


def test_an_absent_key_takes_its_default():
    """``atol`` defaults to 0, making the check purely relative, which is what a
    variant declaring only ``rtol`` is asking for."""
    assert tolerances(_entry(rtol=1e-4)) == {"rtol": 1e-4, "atol": 0.0}
    assert tolerances(_entry()) == {"rtol": 1e-6, "atol": 0.0}


def test_an_explicit_none_is_read_as_absent():
    """``None`` is the absent key written out, not a bound. Forwarding it would
    fail inside ``assert_allclose``, which has no default of its own, two frames
    from the resolver and naming neither the key nor the variant."""
    assert tolerances(_entry(rtol=None, atol=None)) == {"rtol": 1e-6, "atol": 0.0}
    assert tolerances(_entry(rtol=1e-2, atol=None)) == {"rtol": 1e-2, "atol": 0.0}


def test_a_scalar_is_not_resolved_against_the_dtype():
    """A scalar covers every arm of a sweep, so a variant whose dtypes agree on a
    bound keeps writing one number."""
    assert tolerances(_entry(dtype="fp16", atol=5e-2))["atol"] == 5e-2


# ---------------------------------------------------------------------------
# The per-dtype dict — one entry, one bound per arm
# ---------------------------------------------------------------------------

def test_a_dict_is_resolved_against_the_variants_dtype():
    """Both arms of a ``DTYPE`` sweep read the same field and get the bound their
    own dtype names, which is the whole point of the dict."""
    both = {"fp16": 2.5e-1, "fp32": 3e-5}
    assert tolerances(_entry(dtype="fp16", atol=both))["atol"] == 2.5e-1
    assert tolerances(_entry(dtype="fp32", atol=both))["atol"] == 3e-5


def test_the_two_keys_are_resolved_independently():
    """A rule is often relative in one dtype and absolute in the other, so the
    dict is per key rather than per variant."""
    resolved = tolerances(_entry(dtype="fp16", rtol=0.0, atol={"fp16": 2.5e-1}))
    assert resolved == {"rtol": 0.0, "atol": 2.5e-1}


# ---------------------------------------------------------------------------
# A dict that does not name the dtype
# ---------------------------------------------------------------------------

def test_a_dict_missing_the_variants_dtype_raises():
    """Refused rather than defaulted: the default ``atol`` of 0 would turn an
    unstated dtype into a strict check that fails for a reason the fixture never
    wrote down. The message names the variant, the dtype it has, and the dtypes
    the dict offers, because those three are what the author has to reconcile."""
    with pytest.raises(ValueError, match="fix::variant.*'atol'.*fp32.*'fp64'"):
        tolerances(_entry(dtype="fp64", atol={"fp16": 2.5e-1, "fp32": 3e-5}),
                   key="fix::variant")


def test_a_dict_on_a_variant_with_no_dtype_param_raises():
    """A dict states that the bound depends on the dtype, so a variant with no
    ``DTYPE`` to resolve against has nothing the dict could mean."""
    with pytest.raises(ValueError, match="'rtol'.*None"):
        tolerances(_entry(rtol={"fp16": 1e-2}), key="fix::variant")
