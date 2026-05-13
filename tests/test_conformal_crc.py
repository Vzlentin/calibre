from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeConformalRiskRuntime,
)
from calibre.conformal.partitions import series_partition
from calibre.core.forecast_frame import (
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
    quantile_column,
)
from calibre.ordering import RsPolicyParameters, apply_rs_policy


def _frame(
    *,
    unique_id: str = "A",
    origin: pd.Timestamp | None = None,
    base_values: list[float] | None = None,
    actual_values: list[float] | None = None,
) -> pd.DataFrame:
    qcol = quantile_column(0.5)
    origin = origin or pd.Timestamp("2024-01-01")
    base = base_values or [10.0, 20.0, 30.0]
    actuals = actual_values if actual_values is not None else [np.nan] * len(base)
    return pd.DataFrame(
        {
            UNIQUE_ID: [unique_id] * len(base),
            "ds": pd.date_range(origin + pd.Timedelta(weeks=1), periods=len(base), freq="W"),
            Y: actuals,
            Y_HAT: base,
            H: list(range(1, len(base) + 1)),
            FORECAST_ORIGIN: [origin] * len(base),
            MODEL_NAME: ["M"] * len(base),
            qcol: base,
        }
    )


def _runtime(*, weight_decay: float | None = None) -> CumulativeConformalRiskRuntime:
    return CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=3,
            base_column=quantile_column(0.5),
            weight_decay=weight_decay,
        )
    )


def test_cumulative_crc_runtime_emits_upper_bound_at_terminal_horizon() -> None:
    runtime = _runtime()
    lower_col, upper_col = interval_column_names(0.5)

    resolved = _frame(actual_values=[11.0, 22.0, 33.0])
    observed = runtime.observe(runtime.apply(resolved))
    assert observed[NONCONFORMITY_SCORE].dropna().tolist() == pytest.approx([6.0])

    fresh = _frame(origin=pd.Timestamp("2024-02-01"), actual_values=None)
    enriched = runtime.apply(fresh)

    assert enriched[CONFORMAL_METHOD].eq("weighted_crc").all()
    assert enriched[CONFORMAL_MODE].eq("cumulative").all()
    assert pd.isna(enriched[upper_col].iloc[0])
    assert pd.isna(enriched[upper_col].iloc[1])
    assert enriched[lower_col].iloc[2] == pytest.approx(60.0)
    assert enriched[upper_col].iloc[2] == pytest.approx(66.0)


def test_cumulative_crc_bound_feeds_rs_order_policy() -> None:
    runtime = _runtime()
    runtime.observe(runtime.apply(_frame(actual_values=[11.0, 22.0, 33.0])))

    enriched = runtime.apply(_frame(origin=pd.Timestamp("2024-02-01")))
    result = apply_rs_policy(
        enriched,
        [
            RsPolicyParameters(
                unique_id="A",
                inventory_position=10.0,
                lead_time=2,
                review_period=1,
            )
        ],
        coverage=0.5,
    )

    assert result.iloc[0]["target_stock_level"] == pytest.approx(66.0)
    assert result.iloc[0]["order_qty"] == pytest.approx(56.0)


def test_cumulative_crc_weight_decay_prefers_recent_residuals() -> None:
    runtime = _runtime(weight_decay=0.5)

    runtime.observe(runtime.apply(_frame(actual_values=[10.0, 20.0, 30.0])))
    runtime.observe(
        runtime.apply(
            _frame(
                origin=pd.Timestamp("2024-02-01"),
                actual_values=[10.0, 20.0, 40.0],
            )
        )
    )
    runtime.observe(
        runtime.apply(
            _frame(
                origin=pd.Timestamp("2024-03-01"),
                actual_values=[10.0, 20.0, 50.0],
            )
        )
    )

    fresh = runtime.apply(_frame(origin=pd.Timestamp("2024-04-01")))
    _, upper_col = interval_column_names(0.5)

    assert fresh[upper_col].iloc[2] == pytest.approx(80.0)


def test_cumulative_crc_nonexchangeable_mode_keeps_test_point_mass() -> None:
    runtime = CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=3,
            base_column=quantile_column(0.5),
            weight_decay=1.0,
            weighted_quantile_mode="nonexchangeable",
        )
    )

    runtime.observe(runtime.apply(_frame(actual_values=[10.0, 20.0, 30.0])))
    runtime.observe(
        runtime.apply(
            _frame(
                origin=pd.Timestamp("2024-02-01"),
                actual_values=[10.0, 20.0, 40.0],
            )
        )
    )

    fresh = runtime.apply(_frame(origin=pd.Timestamp("2024-03-01")))
    _, upper_col = interval_column_names(0.5)

    assert fresh[upper_col].iloc[2] == pytest.approx(70.0)
    assert runtime.get_diagnostics()["weighted_quantile_mode"] == "nonexchangeable"


def test_cumulative_crc_nonexchangeable_infinite_buffer_requires_cap() -> None:
    runtime = CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.9,
            protection_period=3,
            base_column=quantile_column(0.5),
            weight_decay=1.0,
            weighted_quantile_mode="nonexchangeable",
        )
    )
    runtime.observe(runtime.apply(_frame(actual_values=[10.0, 20.0, 30.0])))

    with pytest.raises(ValueError, match="buffer is non-finite"):
        runtime.apply(_frame(origin=pd.Timestamp("2024-02-01")))


def test_cumulative_crc_nonexchangeable_infinite_buffer_can_be_capped() -> None:
    runtime = CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.9,
            protection_period=3,
            base_column=quantile_column(0.5),
            weight_decay=1.0,
            weighted_quantile_mode="nonexchangeable",
            buffer_max=0.0,
        )
    )
    runtime.observe(runtime.apply(_frame(actual_values=[10.0, 20.0, 30.0])))

    fresh = runtime.apply(_frame(origin=pd.Timestamp("2024-02-01")))
    _, upper_col = interval_column_names(0.9)

    assert fresh[upper_col].iloc[2] == pytest.approx(60.0)


def test_cumulative_crc_can_cap_positive_buffers() -> None:
    runtime = CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.9,
            protection_period=3,
            base_column=quantile_column(0.5),
            weight_decay=None,
            buffer_max=0.0,
        )
    )
    runtime.observe(runtime.apply(_frame(actual_values=[20.0, 20.0, 30.0])))

    fresh = runtime.apply(_frame(origin=pd.Timestamp("2024-05-01")))
    _, upper_col = interval_column_names(0.9)

    assert fresh[upper_col].iloc[2] == pytest.approx(60.0)


def test_cumulative_crc_partition_key_uses_parent_residual_pool() -> None:
    runtime = CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.9,
            protection_period=3,
            base_column=quantile_column(0.5),
            partition_key=lambda row: str(row[UNIQUE_ID]).split("_")[0],
            weight_decay=None,
        )
    )
    runtime.observe(runtime.apply(_frame(unique_id="1_A", actual_values=[10.0, 20.0, 40.0])))
    runtime.observe(
        runtime.apply(
            _frame(
                unique_id="1_B",
                origin=pd.Timestamp("2024-02-01"),
                actual_values=[10.0, 20.0, 50.0],
            )
        )
    )
    runtime.observe(
        runtime.apply(
            _frame(
                unique_id="2_A",
                origin=pd.Timestamp("2024-03-01"),
                actual_values=[10.0, 20.0, 130.0],
            )
        )
    )

    fresh = runtime.apply(_frame(unique_id="1_C", origin=pd.Timestamp("2024-04-01")))
    _, upper_col = interval_column_names(0.9)

    # Parent "1" residuals are 10 and 20; at split-conformal coverage 0.9
    # the clipped finite-sample quantile is 20, not the global residual 100.
    assert fresh[upper_col].iloc[2] == pytest.approx(80.0)


def test_cumulative_crc_series_partition_can_shrink_toward_global_pool() -> None:
    runtime = CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.9,
            protection_period=3,
            base_column=quantile_column(0.5),
            partition_key=series_partition,
            weight_decay=None,
            shrinkage_strength=0.5,
        )
    )
    runtime.observe(runtime.apply(_frame(unique_id="A", actual_values=[10.0, 20.0, 40.0])))
    runtime.observe(
        runtime.apply(
            _frame(
                unique_id="B",
                origin=pd.Timestamp("2024-02-01"),
                actual_values=[10.0, 20.0, 130.0],
            )
        )
    )

    fresh = runtime.apply(_frame(unique_id="A", origin=pd.Timestamp("2024-03-01")))
    _, upper_col = interval_column_names(0.9)

    # Local residual is 10, global residual quantile is 100, shrinkage 0.5
    # produces a 55-unit buffer on a 60-unit base forecast.
    assert fresh[upper_col].iloc[2] == pytest.approx(115.0)


def test_cumulative_crc_partition_key_combined_with_weight_decay() -> None:
    runtime = CumulativeConformalRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=0.5,
            protection_period=3,
            base_column=quantile_column(0.5),
            partition_key=lambda row: str(row[UNIQUE_ID]).split("_")[0],
            weight_decay=0.5,
        )
    )
    # Older residual under parent "1": actual_sum=70, base_sum=60 -> residual 10.
    runtime.observe(runtime.apply(_frame(unique_id="1_A", actual_values=[10.0, 20.0, 40.0])))
    # Newer residual under parent "1": actual_sum=110, base_sum=60 -> residual 50.
    runtime.observe(
        runtime.apply(
            _frame(
                unique_id="1_B",
                origin=pd.Timestamp("2024-02-01"),
                actual_values=[10.0, 20.0, 80.0],
            )
        )
    )
    # Different parent "2"; residual 250 must not influence the "1_*" buffer.
    runtime.observe(
        runtime.apply(
            _frame(
                unique_id="2_A",
                origin=pd.Timestamp("2024-03-01"),
                actual_values=[10.0, 20.0, 280.0],
            )
        )
    )

    fresh = runtime.apply(_frame(unique_id="1_C", origin=pd.Timestamp("2024-04-01")))
    _, upper_col = interval_column_names(0.5)

    # Within parent "1", residuals are [10 (weight 0.5), 50 (weight 1.0)].
    # Threshold = 0.5 * 1.5 = 0.75; cumulative crosses at residual 50.
    # Upper = base 60 + buffer 50 = 110. The 250 residual under parent "2"
    # is correctly excluded.
    assert fresh[upper_col].iloc[2] == pytest.approx(110.0)
