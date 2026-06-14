"""Tests for the regret-based tuning objective."""

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
from calibre.tuning.objectives import Cost, Regret


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


def _costs() -> CostStruct:
    return CostStruct(underage_cost=3.0, overage_cost=2.0)


def test_regret_is_realized_cost_minus_independent_oracle() -> None:
    frame = _frame()
    # Perhorizon cost is independently known: per-horizon over/underage is
    # (12-10)*2 + (20-18)*3 + (35-30)*2 = 4 + 6 + 10 = 20.
    realized = Cost(_target_from_yhat, _order_from_target, _costs()).evaluate(frame, frame[Y])
    assert realized == pytest.approx(20.0)

    # Oracle is a fixed perfect-foresight benchmark, chosen independently of
    # `realized` (not realized - delta), so a wrong realized cost would change
    # the regret rather than cancel out.
    objective = Regret(_target_from_yhat, _order_from_target, _costs(), oracle_cost=12.0)

    assert objective.evaluate(frame, frame[Y]) == pytest.approx(8.0)


def test_regret_clips_to_zero_when_realized_below_oracle() -> None:
    frame = _frame()
    objective = Regret(
        _target_from_yhat,
        _order_from_target,
        _costs(),
        oracle_cost=1000.0,
    )

    assert objective.evaluate(frame, frame[Y]) == 0.0


def test_regret_forwards_mode_to_cost() -> None:
    frame = _frame("cumulative")
    # Single cumulative window: order sum(Y_HAT)=65 vs demand sum(Y)=60 ->
    # overage (65-60)*2 = 10. With a zero oracle the regret is exactly that
    # cumulative cost, which only matches if mode reached the wrapped Cost
    # (a perhorizon Cost would instead raise on the cumulative frame).
    objective = Regret(
        _target_from_yhat,
        _order_from_target,
        _costs(),
        oracle_cost=0.0,
        mode="cumulative",
    )

    assert objective.evaluate(frame, frame[Y]) == pytest.approx(10.0)
