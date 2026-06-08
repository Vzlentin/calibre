from functools import partial

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
from calibre.evaluation.forecast_metrics import (
    compute_interval_coverage,
    compute_metrics,
    compute_row_errors,
    resolve_actuals,
)
from calibre.evaluation.point_metrics import mae, mase


def _make_ledger_df() -> pd.DataFrame:
    """Ledger with 4 forecast rows, y=NaN (unresolved)."""
    return pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001"] * 4,
            DS: pd.date_range("2024-02-25", periods=4, freq="W"),
            Y: np.nan,
            Y_HAT: [40.0, 10.0, 20.0, 30.0],
            H: [1, 2, 3, 4],
            FORECAST_ORIGIN: pd.Timestamp("2024-02-25"),
            MODEL_NAME: ["SeasonalNaive"] * 4,
        }
    )


def _make_actuals() -> pd.DataFrame:
    """Actuals covering 2024-01-07 through 2024-03-31."""
    dates = pd.date_range("2024-01-07", periods=13, freq="W")
    return pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001"] * 13,
            DS: dates,
            Y: ([10.0, 20.0, 30.0, 40.0] * 4)[:13],
        }
    )


def test_resolve_fills_y_for_past_dates():
    ledger_df = _make_ledger_df()
    actuals = _make_actuals()
    origin = pd.Timestamp("2024-03-17")

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    resolved_mask = updated[Y].notna()
    assert resolved_mask.sum() == 4


def test_resolve_leaves_future_as_nan():
    ledger_df = _make_ledger_df()
    actuals = _make_actuals()
    origin = pd.Timestamp("2024-02-25")

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    assert updated.loc[0, Y] == 40.0
    assert pd.isna(updated.loc[1, Y])
    assert len(newly_resolved) == 1


def test_resolve_handles_sparse_actuals():
    ledger_df = _make_ledger_df()
    actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001"],
            DS: [pd.Timestamp("2024-02-25")],
            Y: [40.0],
        }
    )
    origin = pd.Timestamp("2024-03-17")

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    assert updated.loc[0, Y] == 40.0
    assert pd.isna(updated.loc[1, Y])
    assert len(newly_resolved) == 1


def test_resolve_no_pending_returns_unchanged():
    ledger_df = _make_ledger_df()
    ledger_df[Y] = [40.0, 10.0, 20.0, 30.0]
    actuals = _make_actuals()
    origin = pd.Timestamp("2024-03-31")

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    assert len(newly_resolved) == 0


def test_row_errors():
    df = pd.DataFrame(
        {
            Y: [40.0, 10.0],
            Y_HAT: [38.0, 12.0],
        }
    )
    result = compute_row_errors(df)

    assert "error" in result.columns
    assert "abs_error" in result.columns
    assert "pct_error" in result.columns
    np.testing.assert_array_almost_equal(result["error"].values, [2.0, -2.0])
    np.testing.assert_array_almost_equal(result["abs_error"].values, [2.0, 2.0])


def test_compute_metrics_groups_by_uid_and_h():
    df = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "B", "B"],
            H: [1, 1, 1, 1],
            Y: [10.0, 12.0, 20.0, 22.0],
            Y_HAT: [11.0, 11.0, 21.0, 21.0],
        }
    )
    result = compute_metrics(df, metrics=[mae], group_by=[UNIQUE_ID]).set_index(UNIQUE_ID)

    assert len(result) == 2
    # Both groups have absolute errors of exactly 1 at each row → MAE == 1.0.
    assert result.loc["A", "mae"] == pytest.approx(1.0)
    assert result.loc["B", "mae"] == pytest.approx(1.0)


def test_compute_metrics_with_partial():
    df = pd.DataFrame(
        {
            UNIQUE_ID: ["A"] * 10,
            H: [1] * 10,
            Y: list(range(10)),
            Y_HAT: [x + 1 for x in range(10)],
        }
    )
    mase_52 = partial(mase, seasonality=1)
    result = compute_metrics(df, metrics=[mae, mase_52], group_by=[UNIQUE_ID])

    # Every forecast overshoots by 1, so MAE == 1.0. The seasonality=1 naive
    # benchmark (consecutive differences of y = 0..9) also has MAE 1.0, so the
    # scaled error MASE == 1.0.
    assert result["mae"].iloc[0] == pytest.approx(1.0)
    assert result["mase"].iloc[0] == pytest.approx(1.0)


def test_compute_metrics_skips_unresolved():
    df = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            H: [1, 2],
            Y: [10.0, np.nan],
            Y_HAT: [11.0, 12.0],
        }
    )
    result = compute_metrics(df, metrics=[mae], group_by=[UNIQUE_ID, H])

    assert len(result) == 1


def test_compute_metrics_adds_interval_diagnostics_when_bounds_are_provided():
    lower_col, upper_col = interval_column_names(0.9)
    df = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            H: [1, 1],
            Y: [10.0, 12.0],
            Y_HAT: [11.0, 11.0],
            lower_col: [9.0, 10.0],
            upper_col: [12.0, 13.0],
        }
    )

    result = compute_metrics(
        df,
        metrics=[mae],
        group_by=[UNIQUE_ID],
        interval_bounds=(lower_col, upper_col),
    )

    assert result["coverage"].iloc[0] == pytest.approx(1.0)
    assert result["mean_interval_width"].iloc[0] == pytest.approx(3.0)


def test_compute_interval_coverage_scores_finite_bounds_only() -> None:
    lower_col, upper_col = interval_column_names(0.9)
    df = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "A", "A"],
            H: [1, 1, 1, 1],
            MODEL_NAME: ["m"] * 4,
            Y: [10.0, 15.0, np.nan, 12.0],
            lower_col: [9.0, 16.0, 0.0, np.nan],
            upper_col: [11.0, 18.0, 20.0, 14.0],
        }
    )

    result = compute_interval_coverage(df, coverage=0.9, group_by=[UNIQUE_ID])

    row = result.iloc[0]
    assert row["total_rows"] == 4
    assert row["resolved_rows"] == 3
    assert row["unresolved_rows"] == 1
    assert row["scored_rows"] == 2
    assert row["unscored_rows"] == 1
    assert row["coverage"] == pytest.approx(0.5)
    assert row["mean_interval_width"] == pytest.approx(2.0)


def test_compute_interval_coverage_groups_by_node_level_horizon_and_model() -> None:
    lower_col, upper_col = interval_column_names(0.9)
    df = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "dept_id=D", "dept_id=D"],
            "level": ["bottom", "bottom", "dept_id", "dept_id"],
            H: [1, 2, 1, 2],
            MODEL_NAME: ["m", "m", "m", "m"],
            Y: [10.0, 12.0, 20.0, 22.0],
            lower_col: [9.0, 13.0, 19.0, 23.0],
            upper_col: [11.0, 14.0, 21.0, 24.0],
        }
    )

    result = compute_interval_coverage(
        df,
        coverage=0.9,
        group_by=[UNIQUE_ID, "level", H, MODEL_NAME],
    )

    assert list(result.columns[:4]) == [UNIQUE_ID, "level", H, MODEL_NAME]
    assert len(result) == 4
    assert set(result["scored_rows"]) == {1}
    assert result["coverage"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_compute_interval_coverage_reports_zero_scored_rows_without_false_coverage() -> None:
    lower_col, upper_col = interval_column_names(0.9)
    df = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            H: [1, 2],
            MODEL_NAME: ["m", "m"],
            Y: [10.0, 12.0],
            lower_col: [np.nan, np.inf],
            upper_col: [11.0, 13.0],
        }
    )

    result = compute_interval_coverage(df, coverage=0.9, group_by=[UNIQUE_ID])

    row = result.iloc[0]
    assert row["resolved_rows"] == 2
    assert row["scored_rows"] == 0
    assert row["unscored_rows"] == 2
    assert pd.isna(row["coverage"])
