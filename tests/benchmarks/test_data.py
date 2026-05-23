"""Tests for benchmarks.vn2.data — data loading and windowing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmarks.vn2.data import (
    _model_uses_cumulative_target,
    _prepare_history,
    _stable_config_key,
    _strip_private,
    _walk_forward_origins,
)
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vn2"


def _small_sales() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    rows = []
    for uid in ("sku_a", "sku_b"):
        for ds in dates:
            rows.append({UNIQUE_ID: uid, DS: ds, Y: 5.0})
    return pd.DataFrame(rows)


def test_prepare_history_returns_uncensored_demand():
    sales = _small_sales()
    history = _prepare_history(sales, instock=None)
    assert set(history.columns) == {UNIQUE_ID, DS, Y}
    assert len(history) == len(sales)


def test_walk_forward_origins_limits_by_history():
    sales = _small_sales()
    origins = _walk_forward_origins(sales, n_origins=3, horizon=4)
    assert len(origins) <= 3
    for ts in origins:
        assert isinstance(ts, pd.Timestamp)


def test_strip_private_removes_underscore_keys():
    config = {"backend": "mlforecast", "_quantile_alpha": 0.5, "_target_mode": "per_horizon"}
    cleaned = _strip_private(config)
    assert "backend" in cleaned
    assert "_quantile_alpha" not in cleaned
    assert "_target_mode" not in cleaned


def test_model_uses_cumulative_target():
    assert _model_uses_cumulative_target({"_target_mode": "cumulative"}) is True
    assert _model_uses_cumulative_target({"_target_mode": "per_horizon"}) is False
    assert _model_uses_cumulative_target({}) is False


def test_stable_config_key_is_deterministic():
    cfg = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    assert _stable_config_key(cfg) == _stable_config_key(cfg)
    cfg2 = {"b": [2, 3], "a": 1, "c": {"d": 4}}
    assert _stable_config_key(cfg) == _stable_config_key(cfg2)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="VN2 data not available")
def test_data_loading_round_trip():
    from calibre.execution.data_loading import load_period

    week0 = load_period(DATA_DIR, 0)
    assert not week0.empty
    assert UNIQUE_ID in week0.columns
    assert DS in week0.columns
    assert Y in week0.columns
    history = _prepare_history(week0, instock=None)
    assert not history.empty
    assert set(history.columns) == {UNIQUE_ID, DS, Y}
