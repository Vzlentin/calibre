from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    CONFORMAL_MODE,
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
from calibre.order import RsPolicyParameters, apply_rs_policy
from calibre.order.rss import apply_rss_policy
from calibre.order.types import RssPolicyParameters


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


def test_apply_rs_policy_returns_order_quantity_from_upper_bounds() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0, 40.0))
    params = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001"],
            "inventory_position": [5.0],
            "lead_time": [1],
            "review_period": [2],
        }
    )

    result = apply_rs_policy(frame, params)

    row = result.iloc[0]
    assert row["protection_period"] == 3
    assert row["target_stock_level"] == 60.0
    assert row["order_qty"] == 55.0


def test_apply_rs_policy_uses_per_series_parameters_and_preserves_metadata() -> None:
    frame = pd.concat(
        [
            _forecast_frame(unique_id="SKU_A", upper_bounds=(10.0, 10.0, 10.0)),
            _forecast_frame(
                unique_id="SKU_B",
                upper_bounds=(4.0, 4.0, 4.0),
                forecast_origin=pd.Timestamp("2024-03-03"),
                model_name="AutoARIMA",
            ),
        ],
        ignore_index=True,
    )

    result = apply_rs_policy(
        frame,
        [
            RsPolicyParameters(
                unique_id="SKU_A",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            ),
            RsPolicyParameters(
                unique_id="SKU_B",
                inventory_position=20.0,
                lead_time=2,
                review_period=1,
            ),
        ],
    ).sort_values(UNIQUE_ID, ignore_index=True)

    assert list(result[UNIQUE_ID]) == ["SKU_A", "SKU_B"]
    assert list(result[MODEL_NAME]) == ["SeasonalNaive", "AutoARIMA"]
    assert list(result[FORECAST_ORIGIN]) == [
        pd.Timestamp("2024-02-04"),
        pd.Timestamp("2024-03-03"),
    ]
    assert list(result["order_qty"]) == [25.0, 0.0]


def test_apply_rs_policy_uses_quantile_column_when_quantile_provided() -> None:
    # Frame with a q_0p833 column and no conformal intervals (drop them)
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    lower_col, upper_col = interval_column_names(0.9)
    frame = frame.drop(columns=[lower_col, upper_col])
    frame[quantile_column(0.833)] = [12.0, 22.0, 32.0]

    result = apply_rs_policy(
        frame,
        [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ],
        quantile=0.833,
    )

    row = result.iloc[0]
    # protection_period = 3 → 12 + 22 + 32 = 66, order = 66 - 5 = 61
    assert row["protection_period"] == 3
    assert row["target_stock_level"] == 66.0
    assert row["order_qty"] == 61.0


def test_apply_rs_policy_quantile_missing_column_raises() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))

    with pytest.raises(ValueError, match="Missing quantile column"):
        apply_rs_policy(
            frame,
            [
                RsPolicyParameters(
                    unique_id="SKU_001",
                    inventory_position=5.0,
                    lead_time=1,
                    review_period=2,
                )
            ],
            quantile=0.833,
        )


def test_apply_rs_policy_requires_interval_columns_for_requested_coverage() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    _, upper_col = interval_column_names(0.9)
    frame = frame.drop(columns=[upper_col])

    with pytest.raises(ValueError, match="Missing conformal interval columns"):
        apply_rs_policy(
            frame,
            [
                RsPolicyParameters(
                    unique_id="SKU_001",
                    inventory_position=5.0,
                    lead_time=1,
                    review_period=2,
                )
            ],
        )


def test_apply_rs_policy_rejects_protection_period_longer_than_available_horizon() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0))

    with pytest.raises(ValueError, match="Protection period 3 exceeds available horizon"):
        apply_rs_policy(
            frame,
            [
                RsPolicyParameters(
                    unique_id="SKU_001",
                    inventory_position=5.0,
                    lead_time=2,
                    review_period=1,
                )
            ],
        )


def test_apply_rs_policy_rejects_missing_horizons_with_clear_error() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    frame = frame.loc[frame[H] != 2].copy()

    with pytest.raises(ValueError, match="Missing horizons within protection period 3: \\[2\\]"):
        apply_rs_policy(
            frame,
            [
                RsPolicyParameters(
                    unique_id="SKU_001",
                    inventory_position=5.0,
                    lead_time=1,
                    review_period=2,
                )
            ],
        )


def _cumulative_forecast_frame(
    *,
    unique_id: str,
    cumulative_upper_at_terminal: float,
    horizon: int = 3,
    forecast_origin: pd.Timestamp = pd.Timestamp("2024-02-04"),  # noqa: B008
    model_name: str = "SeasonalNaive",
    coverage: float = 0.9,
) -> pd.DataFrame:
    """Frame with NaN per-horizon bounds and a single cumulative bound at h=K."""
    lower_col, upper_col = interval_column_names(coverage)
    ds = pd.date_range("2024-02-11", periods=horizon, freq="W")
    upper = [np.nan] * horizon
    lower = [np.nan] * horizon
    upper[-1] = cumulative_upper_at_terminal
    lower[-1] = -cumulative_upper_at_terminal
    frame = pd.DataFrame(
        {
            UNIQUE_ID: [unique_id] * horizon,
            DS: ds,
            Y: [np.nan] * horizon,
            Y_HAT: [10.0] * horizon,
            H: list(range(1, horizon + 1)),
            FORECAST_ORIGIN: [forecast_origin] * horizon,
            MODEL_NAME: [model_name] * horizon,
            lower_col: lower,
            upper_col: upper,
            CONFORMAL_MODE: ["cumulative"] * horizon,
        }
    )
    frame[H] = frame[H].astype("int64")
    return frame


def test_apply_rs_policy_cumulative_mode_picks_terminal_bound_without_summing() -> None:
    """A cumulative-mode frame's target = single bound at h=K, never the sum."""
    frame = _cumulative_forecast_frame(
        unique_id="SKU_001",
        cumulative_upper_at_terminal=42.0,
        horizon=3,
    )

    result = apply_rs_policy(
        frame,
        [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ],
    )
    row = result.iloc[0]
    assert row["protection_period"] == 3
    assert row["target_stock_level"] == 42.0
    assert row["order_qty"] == 37.0


def test_apply_rs_policy_perhorizon_frame_still_sums_bounds() -> None:
    """Without a CONFORMAL_MODE column, behaviour is unchanged (sum of bounds)."""
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    assert CONFORMAL_MODE not in frame.columns

    result = apply_rs_policy(
        frame,
        [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ],
    )
    assert result.iloc[0]["target_stock_level"] == 60.0


def test_apply_rs_policy_perhorizon_marker_still_sums_bounds() -> None:
    """A CONFORMAL_MODE='perhorizon' marker keeps the per-horizon summing path."""
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    frame[CONFORMAL_MODE] = "perhorizon"

    result = apply_rs_policy(
        frame,
        [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ],
    )
    assert result.iloc[0]["target_stock_level"] == 60.0


def test_apply_rs_policy_quantile_path_ignores_cumulative_marker() -> None:
    """Quantile-driven targets ignore the conformal mode marker entirely."""
    frame = _cumulative_forecast_frame(
        unique_id="SKU_001",
        cumulative_upper_at_terminal=42.0,
        horizon=3,
    )
    frame[quantile_column(0.833)] = [12.0, 22.0, 32.0]

    result = apply_rs_policy(
        frame,
        [
            RsPolicyParameters(
                unique_id="SKU_001",
                inventory_position=5.0,
                lead_time=1,
                review_period=2,
            )
        ],
        quantile=0.833,
    )
    assert result.iloc[0]["target_stock_level"] == 66.0


def test_apply_rs_policy_rejects_duplicate_horizons_per_decision_group() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    duplicate_row = frame.iloc[[0]].copy()
    frame = pd.concat([frame, duplicate_row], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate horizons for decision group: \\[1\\]"):
        apply_rs_policy(
            frame,
            [
                RsPolicyParameters(
                    unique_id="SKU_001",
                    inventory_position=5.0,
                    lead_time=1,
                    review_period=2,
                )
            ],
        )


# ── (R,s,S) policy tests ──────────────────────────────────────────────────────


def _rss_params(
    *,
    unique_id: str = "SKU_001",
    inventory_position: float,
    reorder_point: float,
    lead_time: int = 1,
    review_period: int = 2,
) -> list[RssPolicyParameters]:
    return [
        RssPolicyParameters(
            unique_id=unique_id,
            inventory_position=inventory_position,
            reorder_point=reorder_point,
            lead_time=lead_time,
            review_period=review_period,
        )
    ]


def test_apply_rss_policy_orders_when_inventory_below_reorder_point() -> None:
    # protection_period = 1 + 2 = 3, upper bounds sum to 10+20+30 = 60
    # inventory_position=5 < reorder_point=20 → order_qty = max(60 - 5, 0) = 55
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))

    result = apply_rss_policy(frame, _rss_params(inventory_position=5.0, reorder_point=20.0))

    row = result.iloc[0]
    assert row["protection_period"] == 3
    assert row["target_stock_level"] == 60.0
    assert row["order_qty"] == 55.0


def test_apply_rss_policy_no_order_when_inventory_above_reorder_point() -> None:
    # inventory_position=25 >= reorder_point=20 → order_qty = 0
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))

    result = apply_rss_policy(frame, _rss_params(inventory_position=25.0, reorder_point=20.0))

    assert result.iloc[0]["order_qty"] == 0.0


def test_apply_rss_policy_no_order_when_inventory_equals_reorder_point() -> None:
    # inventory_position=20 >= reorder_point=20 → order_qty = 0
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))

    result = apply_rss_policy(frame, _rss_params(inventory_position=20.0, reorder_point=20.0))

    assert result.iloc[0]["order_qty"] == 0.0


def test_apply_rss_policy_requires_interval_columns_for_requested_coverage() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    _, upper_col = interval_column_names(0.9)
    frame = frame.drop(columns=[upper_col])

    with pytest.raises(ValueError, match="Missing conformal interval columns"):
        apply_rss_policy(frame, _rss_params(inventory_position=5.0, reorder_point=20.0))


def test_apply_rss_policy_rejects_protection_period_longer_than_available_horizon() -> None:
    # lead_time=2, review_period=1 → protection_period=3, but only 2 horizons available
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0))

    with pytest.raises(ValueError, match="Protection period 3 exceeds available horizon"):
        apply_rss_policy(
            frame,
            _rss_params(inventory_position=5.0, reorder_point=20.0, lead_time=2, review_period=1),
        )


def test_apply_rss_policy_rejects_missing_horizons_within_protection_period() -> None:
    frame = _forecast_frame(unique_id="SKU_001", upper_bounds=(10.0, 20.0, 30.0))
    frame = frame.loc[frame[H] != 2].copy()

    with pytest.raises(ValueError, match="Missing horizons within protection period 3: \\[2\\]"):
        apply_rss_policy(frame, _rss_params(inventory_position=5.0, reorder_point=20.0))
