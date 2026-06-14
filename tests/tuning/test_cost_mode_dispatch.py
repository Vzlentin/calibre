"""Tests for cost-objective mode dispatch (perhorizon vs cumulative)."""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    CONFORMAL_MODE,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.order_types import CostStruct
from calibre.tuning.objectives import Cost


def _target_from_yhat(frame: pd.DataFrame, costs: CostStruct) -> float:
    del costs
    return float(frame[Y_HAT].sum())


def _order_from_target(target: float, ip: float, *, reorder_point=None) -> float:
    del reorder_point
    return max(float(target) - float(ip), 0.0)


def _frame(mode: str = "perhorizon") -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["sku", "sku", "sku"],
            "ds": pd.date_range("2024-01-14", periods=3, freq="W"),
            Y: [10.0, 20.0, 30.0],
            Y_HAT: [12.0, 18.0, 35.0],
            H: [1, 2, 3],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-07")] * 3,
            MODEL_NAME: ["stub", "stub", "stub"],
            CONFORMAL_MODE: [mode, mode, mode],
        }
    )


def test_perhorizon_sums_per_group() -> None:
    objective = Cost(
        _target_from_yhat,
        _order_from_target,
        CostStruct(underage_cost=3.0, overage_cost=2.0),
    )

    value = objective.evaluate(_frame(), _frame()[Y])

    assert value == 20.0


def test_cumulative_keeps_single_window_semantics() -> None:
    frame = _frame("cumulative")
    objective = Cost(
        _target_from_yhat,
        _order_from_target,
        CostStruct(underage_cost=3.0, overage_cost=2.0),
        mode="cumulative",
    )

    value = objective.evaluate(frame, frame[Y])

    assert value == 10.0


def test_mode_mismatch_raises() -> None:
    objective = Cost(
        _target_from_yhat,
        _order_from_target,
        CostStruct(),
        mode="perhorizon",
    )

    with pytest.raises(ValueError, match="disagrees with frame conformal_mode"):
        objective.evaluate(_frame("cumulative"), _frame("cumulative")[Y])
