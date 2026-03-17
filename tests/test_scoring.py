from functools import partial

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import (
    UNIQUE_ID,
    DS,
    Y,
    Y_HAT,
    H,
    FORECAST_ORIGIN,
    MODEL_NAME,
)
from calibre.engine.scoring import (
    compute_metrics,
    compute_row_errors,
    resolve_actuals,
)
from calibre.metrics import mae, mase


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
    """Actuals covering weeks 2024-01-07 through 2024-03-31."""
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
    result = compute_metrics(df, metrics=[mae], group_by=[UNIQUE_ID])

    assert len(result) == 2
    assert "mae" in result.columns


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

    assert "mae" in result.columns
    assert "mase" in result.columns


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
