from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.conformal.crc import CumulativeConformalRiskConfig, CumulativeConformalRiskRuntime
from calibre.contracts.forecast_frame import (
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
from calibre.order import RsPolicyParameters, apply_rs_policy


def _frame(
    *,
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
            UNIQUE_ID: ["A"] * len(base),
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
