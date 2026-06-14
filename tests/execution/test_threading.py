"""Tests for the canonical thread-budget helpers.

``cap_threaded_config`` / ``thread_budget`` previously existed as two verbatim
private copies in backend.py and optimizer.py that could drift apart. These
pin both the clamping behaviour and that every call site shares it.
"""

from __future__ import annotations

from calibre.execution import backend, threading
from calibre.tuning import optimizer


def test_every_entry_point_applies_the_canonical_clamp():
    config = {"model": "lightgbm.LGBMRegressor", "n_jobs": -1, "num_threads": 16}
    capped = threading.cap_threaded_config(config, cpu_budget=2.0)
    assert capped["n_jobs"] == 2
    assert capped["num_threads"] == 2

    # The backend and tuner re-export the one implementation, so they must
    # produce the identical clamp rather than a divergent private copy.
    assert backend.cap_threaded_config(config, cpu_budget=2.0) == capped
    assert optimizer.cap_threaded_config(config, cpu_budget=2.0) == capped
    assert optimizer.thread_budget(2.0) == threading.thread_budget(2.0) == 2


def test_clamps_thread_knobs_to_budget():
    capped = threading.cap_threaded_config(
        {"model": "lightgbm.LGBMRegressor", "n_jobs": -1, "num_threads": 16},
        cpu_budget=2.0,
    )
    assert capped["n_jobs"] == 2
    assert capped["num_threads"] == 2


def test_none_budget_returns_config_unchanged():
    config = {"model": "x", "n_jobs": 16}
    assert threading.cap_threaded_config(config, None) is config


def test_adds_n_jobs_for_tree_models_only():
    tree = threading.cap_threaded_config({"model": "xgboost.XGBRegressor"}, cpu_budget=3.0)
    assert tree["n_jobs"] == 3
    plain = threading.cap_threaded_config({"model": "SeasonalNaive"}, cpu_budget=3.0)
    assert "n_jobs" not in plain


def test_thread_budget_floor_and_none():
    assert threading.thread_budget(None) == 1
    assert threading.thread_budget(0.5) == 1
    assert threading.thread_budget(4.0) == 4
