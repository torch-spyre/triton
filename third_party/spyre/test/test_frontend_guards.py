#!/usr/bin/env python3
"""Unit tests for the reusable Spyre frontend backend guard.

``triton.language.target_info`` gains two helpers (next to ``is_cuda`` /
``is_hip``) that let frontend code diverge by backend at *runtime* — needed
because frontend op construction runs before any pass, so the C++
``TRITON_BUILD_TTIR_ONLY`` compile flag can't reach it:

  * :func:`is_spyre` — a predicate, for behavior that *forks* by backend
    (``if is_spyre(): <relaxed> else: <strict>``), e.g. the
    descriptor_gather/scatter rank-N relaxation.
  * :func:`requires_backend` — a decorator that raises on the wrong backend,
    for ops that *only exist* for one backend, e.g. ``tl.spyre_tensor_layout``.

Both resolve the backend through ``current_target()``, so these tests drive
them by monkeypatching ``current_target`` to return a chosen ``GPUTarget`` —
no active driver required.
"""

import pytest
from triton.backends.compiler import GPUTarget
from triton.language import target_info


def _target(backend):
    return GPUTarget(backend=backend, arch=1, warp_size=1)


@pytest.fixture
def as_backend(monkeypatch):
    """Return a setter that pins ``current_target()`` to a given backend (or
    ``None`` for "no active driver")."""

    def _set(backend):
        target = None if backend is None else _target(backend)
        monkeypatch.setattr(target_info, "current_target", lambda: target)

    return _set


# ---------------------------------------------------------------------------
# is_spyre — predicate
# ---------------------------------------------------------------------------

class TestIsSpyre:

    def test_true_on_spyre(self, as_backend):
        as_backend("spyre")
        assert target_info.is_spyre() is True

    @pytest.mark.parametrize("backend", ["cuda", "hip"])
    def test_false_on_other_backends(self, as_backend, backend):
        as_backend(backend)
        assert target_info.is_spyre() is False

    def test_false_when_no_active_target(self, as_backend):
        # current_target() returns None when there is no active driver.
        as_backend(None)
        assert target_info.is_spyre() is False


# ---------------------------------------------------------------------------
# requires_backend — decorator
# ---------------------------------------------------------------------------

class TestRequiresBackend:

    @staticmethod
    @target_info.requires_backend("spyre")
    def _spyre_only(x):
        return x + 1

    def test_runs_on_matching_backend(self, as_backend):
        as_backend("spyre")
        assert self._spyre_only(41) == 42

    @pytest.mark.parametrize("backend", ["cuda", "hip"])
    def test_raises_on_other_backend(self, as_backend, backend):
        as_backend(backend)
        with pytest.raises(ValueError, match="only supported on the 'spyre' backend"):
            self._spyre_only(0)

    def test_raises_when_no_active_target(self, as_backend):
        as_backend(None)
        with pytest.raises(ValueError, match="only supported on the 'spyre' backend"):
            self._spyre_only(0)

    def test_resolves_backend_at_call_time_not_decoration_time(self, as_backend):
        # The same decorated callable passes under spyre and fails otherwise —
        # proving the check is deferred to each call, not bound at decoration.
        as_backend("spyre")
        assert self._spyre_only(1) == 2
        as_backend("cuda")
        with pytest.raises(ValueError):
            self._spyre_only(1)

    def test_error_names_the_function_and_actual_backend(self, as_backend):
        as_backend("cuda")
        with pytest.raises(ValueError) as exc:
            self._spyre_only(0)
        msg = str(exc.value)
        assert "_spyre_only" in msg
        assert "'cuda'" in msg
