"""Tests for the canonical thread-budget helpers (roadmap P1.2a).

``cap_threaded_config`` / ``thread_budget`` previously existed as two verbatim
private copies in backend.py and optimizer.py. These assert there is now a
single source of truth and that the clamping behaviour is correct.
"""

from __future__ import annotations

from calibre.execution import backend, threading
from calibre.tuning import optimizer


def test_single_source_of_truth():
    """Both the backend and the tuner use the one canonical implementation."""
    assert backend.cap_threaded_config is threading.cap_threaded_config
    assert optimizer.cap_threaded_config is threading.cap_threaded_config
    assert optimizer.thread_budget is threading.thread_budget


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
