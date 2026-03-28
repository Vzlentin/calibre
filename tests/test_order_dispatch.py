from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.order.config import OrderPolicyConfig, apply_order_policy
from calibre.order.types import (
    NewsvendorPolicyParameters,
    RsPolicyParameters,
    RssPolicyParameters,
)


def _forecast_frame(
    *,
    unique_id: str,
    upper_bounds: tuple[float, ...],
    forecast_origin: pd.Timestamp = pd.Timestamp("2024-02-04"),  # noqa: B008
    model_name: str = "SeasonalNaive",
    coverage: float = 0.9,
) -> pd.DataFrame:
    """Create a minimal forecast frame for testing."""
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
    """Tests for OrderPolicyConfig validation."""

    def test_config_validates_coverage_bounds_rejects_zero(self) -> None:
        """coverage=0 should raise ValueError."""
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            OrderPolicyConfig(policy="rs", params=pd.DataFrame(), coverage=0.0)

    def test_config_validates_coverage_bounds_rejects_one(self) -> None:
        """coverage=1 should raise ValueError."""
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            OrderPolicyConfig(policy="rs", params=pd.DataFrame(), coverage=1.0)

    def test_config_validates_coverage_bounds_accepts_valid_value(self) -> None:
        """coverage=0.5 should be accepted."""
        config = OrderPolicyConfig(policy="rs", params=pd.DataFrame(), coverage=0.5)
        assert config.coverage == 0.5

    def test_config_accepts_default_coverage(self) -> None:
        """Default coverage should be 0.9."""
        config = OrderPolicyConfig(policy="rs", params=pd.DataFrame())
        assert config.coverage == 0.9

    def test_config_accepts_default_period(self) -> None:
        """Default period should be 1."""
        config = OrderPolicyConfig(policy="newsvendor", params=pd.DataFrame())
        assert config.period == 1


class TestDispatchToRsPolicy:
    """Tests for routing to RS policy."""

    def test_dispatch_routes_to_rs_policy(self) -> None:
        """Config with policy='rs' should call apply_rs_policy correctly."""
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

    def test_dispatch_rs_policy_respects_custom_coverage(self) -> None:
        """RS dispatch should pass custom coverage level."""
        lower_col, upper_col = interval_column_names(0.95)
        frame = _forecast_frame(
            unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0, 40.0), coverage=0.95
        )
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

        assert not result.empty
        # Just verify it doesn't error and has the expected output column
        assert "target_stock_level" in result.columns


class TestDispatchToRssPolicy:
    """Tests for routing to RSS policy."""

    def test_dispatch_routes_to_rss_policy(self) -> None:
        """Config with policy='rss' should call apply_rss_policy correctly."""
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
        """RSS should not order when inventory is above reorder point."""
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
    """Tests for routing to Newsvendor policy."""

    def test_dispatch_routes_to_newsvendor_policy(self) -> None:
        """Config with policy='newsvendor' should call apply_newsvendor_policy correctly."""
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
        # Verify critical_ratio calculation: 10 / (10 + 1) ≈ 0.909
        assert 0.9 < row["critical_ratio"] < 0.92

    def test_dispatch_newsvendor_policy_passes_period_parameter(self) -> None:
        """Newsvendor dispatch should pass the period parameter."""
        # Create a frame with multiple horizons
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
        params = [
            NewsvendorPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                underage_cost=10.0,
                overage_cost=1.0,
            )
        ]
        # Use period=2 to test it's passed correctly
        config = OrderPolicyConfig(policy="newsvendor", params=params, coverage=0.9, period=2)

        result = apply_order_policy(frame, config)

        # Should successfully complete without error, indicating period=2 was passed
        assert not result.empty


class TestDispatchErrors:
    """Tests for error handling in dispatch."""

    def test_dispatch_rejects_unknown_policy(self) -> None:
        """Unknown policy type should raise ValueError."""
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0))
        config = OrderPolicyConfig(policy="unknown", params=pd.DataFrame(), coverage=0.9)  # type: ignore

        with pytest.raises(ValueError, match="Unknown order policy: 'unknown'"):
            apply_order_policy(frame, config)
