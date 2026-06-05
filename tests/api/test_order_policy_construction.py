from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api.main import _build_order_policy, app
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
from calibre.ordering.policy_config import NewsvendorConfig, RsConfig, RssConfig


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


class TestBuildOrderPolicy:
    def test_rs_spec_builds_rs_config(self) -> None:
        config = _build_order_policy({"policy": "rs", "params": _RS_PARAMS, "coverage": 0.95})
        assert isinstance(config, RsConfig)
        assert config.coverage == 0.95
        assert config.quantile is None

    def test_rs_spec_with_quantile_builds_rs_config_with_quantile(self) -> None:
        config = _build_order_policy({"policy": "rs", "params": _RS_PARAMS, "quantile": 0.833})
        assert isinstance(config, RsConfig)
        assert config.quantile == pytest.approx(0.833)

    def test_rss_spec_builds_rss_config(self) -> None:
        config = _build_order_policy({"policy": "rss", "params": _RS_PARAMS})
        assert isinstance(config, RssConfig)

    def test_newsvendor_spec_builds_newsvendor_config(self) -> None:
        config = _build_order_policy(
            {"policy": "newsvendor", "params": _NEWSVENDOR_PARAMS, "period": 2}
        )
        assert isinstance(config, NewsvendorConfig)
        assert config.period == 2

    def test_newsvendor_spec_rejects_quantile_knob(self) -> None:
        with pytest.raises(ValueError, match="quantile is not a valid knob for the newsvendor"):
            _build_order_policy(
                {"policy": "newsvendor", "params": _NEWSVENDOR_PARAMS, "quantile": 0.9}
            )

    def test_rss_spec_rejects_quantile_knob(self) -> None:
        with pytest.raises(ValueError, match="quantile is not a valid knob for the rss"):
            _build_order_policy({"policy": "rss", "params": _RS_PARAMS, "quantile": 0.9})

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown order policy: 'bogus'"):
            _build_order_policy({"policy": "bogus", "params": _RS_PARAMS})


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
        assert "quantile is not a valid knob for the newsvendor" in resp.json()["detail"]
