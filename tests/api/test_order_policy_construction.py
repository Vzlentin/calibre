"""Tests for ordering-policy construction from /order requests."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from calibre.api.main import app
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


def _calibrated_records() -> list[dict]:
    """A two-horizon calibrated frame the order endpoint can apply a policy to."""
    lower_col, upper_col = interval_column_names(0.9)
    return [
        {
            UNIQUE_ID: "A",
            DS: pd.Timestamp("2024-02-11").isoformat(),
            Y: None,
            Y_HAT: 4.0,
            H: 1,
            FORECAST_ORIGIN: pd.Timestamp("2024-02-04").isoformat(),
            MODEL_NAME: "SeasonalNaive",
            lower_col: 3.0,
            upper_col: 5.0,
        },
        {
            UNIQUE_ID: "A",
            DS: pd.Timestamp("2024-02-18").isoformat(),
            Y: None,
            Y_HAT: 5.0,
            H: 2,
            FORECAST_ORIGIN: pd.Timestamp("2024-02-04").isoformat(),
            MODEL_NAME: "SeasonalNaive",
            lower_col: 4.0,
            upper_col: 6.0,
        },
    ]


_RS_PARAMS = [{"unique_id": "A", "inventory_position": 5.0, "lead_time": 1, "review_period": 1}]
_NEWSVENDOR_PARAMS = [
    {"unique_id": "A", "underage_cost": 3.0, "overage_cost": 1.0, "inventory_position": 5.0}
]


class TestOrderEndpoint:
    def test_order_endpoint_rs_happy_path(self) -> None:
        client = TestClient(app)
        resp = client.post(
            "/order",
            json={
                "calibrated": _calibrated_records(),
                "ordering": {"policy": "rs", "coverage": 0.9, "params": _RS_PARAMS},
            },
        )
        assert resp.status_code == 200, resp.text
        orders = resp.json()["orders"]
        assert len(orders) == 1
        # target = upper bounds summed over protection window (lead 1 + review 1):
        # 5 + 6 = 11; order = max(11 - inventory(5), 0) = 6.
        assert orders[0]["target_stock_level"] == 11.0
        assert orders[0]["order_qty"] == 6.0

    def test_order_endpoint_unknown_policy_is_4xx(self) -> None:
        client = TestClient(app)
        resp = client.post(
            "/order",
            json={
                "calibrated": _calibrated_records(),
                "ordering": {"policy": "bogus", "params": _RS_PARAMS},
            },
        )
        assert resp.status_code == 400, resp.text
        assert "unknown order policy" in resp.json()["detail"]

    def test_order_endpoint_newsvendor_with_quantile_is_4xx(self) -> None:
        client = TestClient(app)
        resp = client.post(
            "/order",
            json={
                "calibrated": _calibrated_records(),
                "ordering": {
                    "policy": "newsvendor",
                    "params": _NEWSVENDOR_PARAMS,
                    "quantile": 0.9,
                },
            },
        )
        assert resp.status_code == 400, resp.text
        assert (
            "invalid ordering spec: ordering.quantile is not a valid knob for the newsvendor"
            in resp.json()["detail"]
        )

    def test_order_endpoint_missing_params_is_4xx(self) -> None:
        client = TestClient(app)
        resp = client.post(
            "/order",
            json={
                "calibrated": _calibrated_records(),
                "ordering": {"policy": "rs"},
            },
        )
        assert resp.status_code == 400, resp.text
        assert "ordering.params is required" in resp.json()["detail"]

    def test_order_endpoint_newsvendor_explicit_null_period_uses_default(self) -> None:
        """Explicit ``period: null`` is equivalent to omitting the knob.

        The factory keys optional knobs on value-is-not-None, never key presence
        (the CLI's model_dump() always emits the key as None), so the request
        succeeds with NewsvendorConfig's default period of 1.
        """
        client = TestClient(app)
        resp = client.post(
            "/order",
            json={
                "calibrated": _calibrated_records(),
                "ordering": {
                    "policy": "newsvendor",
                    "params": _NEWSVENDOR_PARAMS,
                    "period": None,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["orders"]) == 1
