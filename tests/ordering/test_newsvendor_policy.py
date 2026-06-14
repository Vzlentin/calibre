"""Tests for the newsvendor ordering policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.core.order_types import NewsvendorPolicyParameters
from calibre.ordering.newsvendor import apply_newsvendor_policy


def _forecast_frame(
    unique_id: str = "SKU-A",
    lo: float = 10.0,
    hi: float = 30.0,
    horizons: range = range(1, 5),
    coverage: float = 0.9,
) -> pd.DataFrame:
    lower_col, upper_col = interval_column_names(coverage)
    n = len(horizons)
    ds = pd.date_range("2024-02-11", periods=n, freq="W")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: [unique_id] * n,
            DS: ds,
            Y: [np.nan] * n,
            Y_HAT: [(lo + hi) / 2] * n,
            H: list(horizons),
            FORECAST_ORIGIN: [pd.Timestamp("2024-02-04")] * n,
            MODEL_NAME: ["SeasonalNaive"] * n,
            lower_col: [lo] * n,
            upper_col: [hi] * n,
        }
    )
    frame[H] = frame[H].astype("int64")
    return frame


def _newsvendor_params(
    unique_id: str = "SKU-A",
    underage_cost: float = 3.0,
    overage_cost: float = 1.0,
    inventory_position: float = 5.0,
) -> list[NewsvendorPolicyParameters]:
    return [
        NewsvendorPolicyParameters(
            unique_id=unique_id,
            underage_cost=underage_cost,
            overage_cost=overage_cost,
            inventory_position=inventory_position,
        )
    ]


def test_symmetric_costs_picks_midpoint() -> None:
    # Cu=1, Co=1 → critical_ratio=0.5 → demand_quantile = (10+30)/2 = 20
    # order_qty = max(20 - 5, 0) = 15
    frame = _forecast_frame(lo=10.0, hi=30.0)
    params = _newsvendor_params(underage_cost=1.0, overage_cost=1.0, inventory_position=5.0)

    result = apply_newsvendor_policy(frame, params)

    assert len(result) == 1
    assert result.iloc[0]["critical_ratio"] == pytest.approx(0.5)
    assert result.iloc[0]["demand_quantile"] == pytest.approx(20.0)
    assert result.iloc[0]["order_qty"] == pytest.approx(15.0)


def test_high_underage_cost_picks_near_upper_bound() -> None:
    # Cu=9, Co=1 → critical_ratio=0.9 → demand_quantile = 10 + 0.9*(30-10) = 28
    # order_qty = max(28 - 5, 0) = 23
    frame = _forecast_frame(lo=10.0, hi=30.0)
    params = _newsvendor_params(underage_cost=9.0, overage_cost=1.0, inventory_position=5.0)

    result = apply_newsvendor_policy(frame, params)

    assert result.iloc[0]["critical_ratio"] == pytest.approx(0.9)
    assert result.iloc[0]["demand_quantile"] == pytest.approx(28.0)
    assert result.iloc[0]["order_qty"] == pytest.approx(23.0)


def test_high_overage_cost_picks_near_lower_bound() -> None:
    # Cu=1, Co=9 → critical_ratio=0.1 → demand_quantile = 10 + 0.1*(30-10) = 12
    # order_qty = max(12 - 5, 0) = 7
    frame = _forecast_frame(lo=10.0, hi=30.0)
    params = _newsvendor_params(underage_cost=1.0, overage_cost=9.0, inventory_position=5.0)

    result = apply_newsvendor_policy(frame, params)

    assert result.iloc[0]["critical_ratio"] == pytest.approx(0.1)
    assert result.iloc[0]["demand_quantile"] == pytest.approx(12.0)
    assert result.iloc[0]["order_qty"] == pytest.approx(7.0)


def test_order_qty_zero_when_demand_quantile_below_inventory() -> None:
    # Cu=1, Co=1 → critical_ratio=0.5 → demand_quantile=20, inventory_position=50
    # order_qty = max(20 - 50, 0) = 0
    frame = _forecast_frame(lo=10.0, hi=30.0)
    params = _newsvendor_params(underage_cost=1.0, overage_cost=1.0, inventory_position=50.0)

    result = apply_newsvendor_policy(frame, params)

    assert result.iloc[0]["order_qty"] == pytest.approx(0.0)


def test_requires_interval_columns() -> None:
    frame = _forecast_frame()
    _, upper_col = interval_column_names(0.9)
    frame = frame.drop(columns=[upper_col])

    with pytest.raises(ValueError, match="Missing conformal interval columns"):
        apply_newsvendor_policy(frame, _newsvendor_params())


def test_raises_when_period_not_in_frame() -> None:
    # frame has h=1..4, request period=10
    frame = _forecast_frame(horizons=range(1, 5))

    with pytest.raises(ValueError, match="No rows found for period h=10"):
        apply_newsvendor_policy(frame, _newsvendor_params(), period=10)


def test_uses_specified_period() -> None:
    # frame has h=1..4 with different lo/hi per horizon
    lower_col, upper_col = interval_column_names(0.9)
    n = 4
    ds = pd.date_range("2024-02-11", periods=n, freq="W")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU-A"] * n,
            DS: ds,
            Y: [np.nan] * n,
            Y_HAT: [15.0] * n,
            H: [1, 2, 3, 4],
            FORECAST_ORIGIN: [pd.Timestamp("2024-02-04")] * n,
            MODEL_NAME: ["SeasonalNaive"] * n,
            lower_col: [10.0, 20.0, 30.0, 40.0],
            upper_col: [30.0, 40.0, 50.0, 60.0],
        }
    )
    frame[H] = frame[H].astype("int64")

    # Use period=2: lo=20, hi=40, critical_ratio=0.5 → demand_quantile=30
    # order_qty = max(30 - 5, 0) = 25
    params = _newsvendor_params(underage_cost=1.0, overage_cost=1.0, inventory_position=5.0)
    result = apply_newsvendor_policy(frame, params, period=2)

    assert result.iloc[0]["demand_quantile"] == pytest.approx(30.0)
    assert result.iloc[0]["order_qty"] == pytest.approx(25.0)
