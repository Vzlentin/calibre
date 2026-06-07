from __future__ import annotations

from dataclasses import fields

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
    quantile_column,
)
from calibre.core.order_types import (
    NewsvendorPolicyParameters,
    RsPolicyParameters,
    RssPolicyParameters,
)
from calibre.ordering.policy_config import (
    NewsvendorConfig,
    RsConfig,
    RssConfig,
    apply_order_policy,
)


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
    def test_rs_config_rejects_coverage_zero(self) -> None:
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            RsConfig(params=pd.DataFrame(), coverage=0.0)

    def test_rs_config_rejects_coverage_one(self) -> None:
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            RsConfig(params=pd.DataFrame(), coverage=1.0)

    def test_rss_config_rejects_coverage_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            RssConfig(params=pd.DataFrame(), coverage=1.0)

    def test_newsvendor_config_rejects_coverage_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="coverage must be in \\(0, 1\\)"):
            NewsvendorConfig(params=pd.DataFrame(), coverage=0.0)

    def test_rs_config_rejects_quantile_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="quantile must be in \\(0, 1\\)"):
            RsConfig(params=pd.DataFrame(), quantile=1.5)

    def test_rs_config_accepts_valid_coverage(self) -> None:
        config = RsConfig(params=pd.DataFrame(), coverage=0.5)
        assert config.coverage == 0.5

    def test_configs_accept_default_coverage(self) -> None:
        assert RsConfig(params=pd.DataFrame()).coverage == 0.9
        assert RssConfig(params=pd.DataFrame()).coverage == 0.9
        assert NewsvendorConfig(params=pd.DataFrame()).coverage == 0.9

    def test_newsvendor_config_accepts_default_period(self) -> None:
        assert NewsvendorConfig(params=pd.DataFrame()).period == 1

    def test_newsvendor_config_has_no_quantile_field(self) -> None:
        # The whole point of the deepening: a quantile knob is unrepresentable
        # on the newsvendor config — it is not even a field.
        field_names = {f.name for f in fields(NewsvendorConfig)}
        assert "quantile" not in field_names

    def test_rss_config_has_no_quantile_field(self) -> None:
        field_names = {f.name for f in fields(RssConfig)}
        assert "quantile" not in field_names


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
        config = RsConfig(params=params, coverage=0.9)

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
        config = RsConfig(params=params, coverage=0.95)

        result = apply_order_policy(frame, config)

        row = result.iloc[0]
        assert row["protection_period"] == 3
        assert row["target_stock_level"] == 60.0
        assert row["order_qty"] == 55.0

    def test_dispatch_rs_with_quantile_uses_q_column_path(self) -> None:
        # With a quantile set, the target is summed from the per-horizon
        # ``q_<p>`` column, not the conformal upper bound. Drop the conformal
        # band so only the q-column path can produce a result.
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
        lower_col, upper_col = interval_column_names(0.9)
        frame = frame.drop(columns=[lower_col, upper_col])
        frame[quantile_column(0.833)] = [12.0, 22.0, 32.0]
        params = [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ]
        config = RsConfig(params=params, quantile=0.833)

        result = apply_order_policy(frame, config)

        row = result.iloc[0]
        # protection_period = 3 → 12 + 22 + 32 = 66, order = 66 - 5 = 61
        assert row["protection_period"] == 3
        assert row["target_stock_level"] == 66.0
        assert row["order_qty"] == 61.0

    def test_dispatch_rs_without_quantile_uses_conformal_upper_bound(self) -> None:
        # Without a quantile, the target is the conformal upper bound summed over
        # the protection period and ``coverage`` selects the band.
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
        params = [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ]
        config = RsConfig(params=params, coverage=0.9)
        assert config.quantile is None

        result = apply_order_policy(frame, config)

        row = result.iloc[0]
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
        config = RssConfig(params=params, coverage=0.9)

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
        config = RssConfig(params=params, coverage=0.9)

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
        config = NewsvendorConfig(params=params, coverage=0.9, period=1)

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

        period1 = apply_order_policy(frame, NewsvendorConfig(params=params, coverage=0.9, period=1))
        period2 = apply_order_policy(frame, NewsvendorConfig(params=params, coverage=0.9, period=2))

        dq1 = period1.iloc[0]["demand_quantile"]
        dq2 = period2.iloc[0]["demand_quantile"]
        assert 5.0 <= dq1 <= 10.0
        assert 15.0 <= dq2 <= 20.0
        assert dq2 > dq1


class TestDispatchErrors:
    def test_dispatch_rejects_unknown_config_type(self) -> None:
        frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0))

        with pytest.raises(AssertionError, match="Expected code to be unreachable"):
            apply_order_policy(frame, object())  # type: ignore[arg-type]
