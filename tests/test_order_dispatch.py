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
from calibre.core.order_types import (
    NewsvendorPolicyParameters,
    RsPolicyParameters,
    RssPolicyParameters,
)
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy


def _forecast_frame(
    *,
    unique_id: str,
    upper_bounds: tuple[float, ...],
    forecast_origin: pd.Timestamp = pd.Timestamp("2024-02-04"),  # noqa: B008
    model_name: str = "SeasonalNaive",
    coverage: float = 0.9,
) -> pd.DataFrame:
    lower_col, upper_col = interval_column_names(coverage)
    horizon = len(upper_bounds)
    ds = pd.date_range("2024-02-11", periods=horizon, freq="W")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: [unique_id] * horizon,
            DS: ds,
            Y: [np.nan] * horizon,
            Y_HAT: [bound - 2.0 for bound in upper_bounds],
            H: list(range(1, horizon + 1)),
            FORECAST_ORIGIN: [forecast_origin] * horizon,
            MODEL_NAME: [model_name] * horizon,
            lower_col: [bound - 5.0 for bound in upper_bounds],
            upper_col: list(upper_bounds),
        }
    )
    frame[H] = frame[H].astype("int64")
    return frame


class TestConfigValidation:
    def test_config_validates_coverage_bounds_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            OrderPolicyConfig(policy="rs", params=pd.DataFrame(), coverage=0.0)

    def test_config_validates_coverage_bounds_rejects_one(self) -> None:
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            OrderPolicyConfig(policy="rs", params=pd.DataFrame(), coverage=1.0)

    def test_config_validates_coverage_bounds_accepts_valid_value(self) -> None:
        config = OrderPolicyConfig(policy="rs", params=pd.DataFrame(), coverage=0.5)
        assert config.coverage == 0.5

    def test_config_accepts_default_coverage(self) -> None:
        config = OrderPolicyConfig(policy="rs", params=pd.DataFrame())
        assert config.coverage == 0.9

    def test_config_accepts_default_period(self) -> None:
        config = OrderPolicyConfig(policy="newsvendor", params=pd.DataFrame())
        assert config.period == 1


class TestDispatchToRsPolicy:
    def test_dispatch_routes_to_rs_policy(self) -> None:
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0, 40.0))
        params = [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ]
        config = OrderPolicyConfig(policy="rs", params=params, coverage=0.9)

        result = apply_order_policy(frame, config)

        assert not result.empty
        row = result.iloc[0]
        assert "target_stock_level" in result.columns
        assert row["target_stock_level"] == 60.0
        assert row["order_qty"] == 55.0

    def test_dispatch_rs_policy_honors_custom_coverage(self) -> None:
        # Frame carries BOTH the 0.9 and 0.95 bands with distinct values. The
        # 0.95 band sums to 60 over the protection period; the 0.9 band sums to
        # 6000 — so the target proves which coverage the dispatch actually used.
        frame = _forecast_frame(
            unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0, 40.0), coverage=0.95
        )
        lower_90, upper_90 = interval_column_names(0.9)
        frame[lower_90] = [100.0, 200.0, 300.0, 400.0]
        frame[upper_90] = [1000.0, 2000.0, 3000.0, 4000.0]
        params = [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ]
        config = OrderPolicyConfig(policy="rs", params=params, coverage=0.95)

        result = apply_order_policy(frame, config)

        row = result.iloc[0]
        assert row["protection_period"] == 3
        assert row["target_stock_level"] == 60.0
        assert row["order_qty"] == 55.0


class TestDispatchToRssPolicy:
    def test_dispatch_routes_to_rss_policy(self) -> None:
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
        params = [
            RssPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                reorder_point=20.0,
                lead_time=1,
                review_period=2,
            )
        ]
        config = OrderPolicyConfig(policy="rss", params=params, coverage=0.9)

        result = apply_order_policy(frame, config)

        assert not result.empty
        row = result.iloc[0]
        assert "target_stock_level" in result.columns
        assert row["target_stock_level"] == 60.0
        assert row["order_qty"] == 55.0

    def test_dispatch_rss_policy_no_order_above_reorder_point(self) -> None:
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
        params = [
            RssPolicyParameters(
                unique_id="SKU_001",
                inventory_position=25.0,
                reorder_point=20.0,
                lead_time=1,
                review_period=2,
            )
        ]
        config = OrderPolicyConfig(policy="rss", params=params, coverage=0.9)

        result = apply_order_policy(frame, config)

        assert result.iloc[0]["order_qty"] == 0.0


class TestDispatchToNewsvendorPolicy:
    def test_dispatch_routes_to_newsvendor_policy(self) -> None:
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
        params = [
            NewsvendorPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                underage_cost=10.0,
                overage_cost=1.0,
            )
        ]
        config = OrderPolicyConfig(policy="newsvendor", params=params, coverage=0.9, period=1)

        result = apply_order_policy(frame, config)

        assert not result.empty
        row = result.iloc[0]
        assert "critical_ratio" in result.columns
        assert "demand_quantile" in result.columns
        # critical_ratio = Cu / (Cu + Co) = 10 / 11 ≈ 0.909
        assert row["critical_ratio"] == pytest.approx(10.0 / 11.0)

    def test_dispatch_newsvendor_policy_honors_period_parameter(self) -> None:
        # Per-horizon bands differ: h=1 → [5, 10], h=2 → [15, 20]. The period
        # selects which horizon's band the demand quantile is drawn from, so the
        # two periods must land the quantile in different (non-overlapping) bands.
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
        params = [
            NewsvendorPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                underage_cost=10.0,
                overage_cost=1.0,
            )
        ]

        period1 = apply_order_policy(
            frame, OrderPolicyConfig(policy="newsvendor", params=params, coverage=0.9, period=1)
        )
        period2 = apply_order_policy(
            frame, OrderPolicyConfig(policy="newsvendor", params=params, coverage=0.9, period=2)
        )

        dq1 = period1.iloc[0]["demand_quantile"]
        dq2 = period2.iloc[0]["demand_quantile"]
        assert 5.0 <= dq1 <= 10.0
        assert 15.0 <= dq2 <= 20.0
        assert dq2 > dq1


class TestDispatchErrors:
    def test_dispatch_rejects_unknown_policy(self) -> None:
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0))
        config = OrderPolicyConfig(policy="unknown", params=pd.DataFrame(), coverage=0.9)  # type: ignore

        with pytest.raises(ValueError, match="Unknown order policy: 'unknown'"):
            apply_order_policy(frame, config)
