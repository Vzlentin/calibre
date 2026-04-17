"""Tests for the ensemble median aggregator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    validate_forecast_frame,
)
from calibre.ensemble.median import ensemble_median


def _make_forecast_row(
    unique_id: str,
    forecast_origin: str,
    ds: str,
    h: int,
    y_hat: float,
    model_name: str,
) -> dict:
    return {
        UNIQUE_ID: unique_id,
        DS: pd.Timestamp(ds),
        Y: np.nan,
        Y_HAT: y_hat,
        H: h,
        FORECAST_ORIGIN: pd.Timestamp(forecast_origin),
        MODEL_NAME: model_name,
    }


def _make_ledger(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[Y] = df[Y].astype("float64")
    df[Y_HAT] = df[Y_HAT].astype("float64")
    df[H] = df[H].astype("int64")
    return df


class TestEnsembleMedian:
    def test_three_models_two_series(self) -> None:
        """3 models × 2 series → median computed correctly per group."""
        origin = "2024-01-01"
        ds = "2024-01-08"
        rows = []
        # Series A: model predictions 10, 20, 30 → median = 20
        for model, val in [("M1", 10.0), ("M2", 20.0), ("M3", 30.0)]:
            rows.append(_make_forecast_row("A", origin, ds, 1, val, model))
        # Series B: model predictions 5, 15, 25 → median = 15
        for model, val in [("M1", 5.0), ("M2", 15.0), ("M3", 25.0)]:
            rows.append(_make_forecast_row("B", origin, ds, 1, val, model))

        ledger = _make_ledger(rows)
        result = ensemble_median(ledger)

        assert len(result) == 2  # one row per series
        a_row = result[result[UNIQUE_ID] == "A"].iloc[0]
        b_row = result[result[UNIQUE_ID] == "B"].iloc[0]
        assert a_row[Y_HAT] == pytest.approx(20.0)
        assert b_row[Y_HAT] == pytest.approx(15.0)

    def test_single_model_passthrough(self) -> None:
        """A single model should be returned with its y_hat unchanged (median of 1 = itself)."""
        rows = [
            _make_forecast_row("X", "2024-01-01", "2024-01-08", 1, 42.0, "OnlyModel"),
            _make_forecast_row("X", "2024-01-01", "2024-01-15", 2, 88.0, "OnlyModel"),
        ]
        ledger = _make_ledger(rows)
        result = ensemble_median(ledger, name="solo")

        assert len(result) == 2
        assert set(result[MODEL_NAME].unique()) == {"solo"}
        vals = result.sort_values(H)[Y_HAT].tolist()
        assert vals == pytest.approx([42.0, 88.0])

    def test_output_passes_validate_forecast_frame(self) -> None:
        """Ensemble output must be a valid forecast frame."""
        rows = []
        for model, val in [("M1", 1.0), ("M2", 3.0)]:
            rows.append(_make_forecast_row("S1", "2024-03-01", "2024-03-08", 1, val, model))
        ledger = _make_ledger(rows)
        result = ensemble_median(ledger)
        # Should not raise
        validate_forecast_frame(result)

    def test_y_is_nan(self) -> None:
        """Ensemble output rows must have y = NaN."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 5.0, "M1")]
        ledger = _make_ledger(rows)
        result = ensemble_median(ledger)
        assert result[Y].isna().all()

    def test_model_name_default(self) -> None:
        """Default model_name is 'ensemble_median'."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 5.0, "M1")]
        ledger = _make_ledger(rows)
        result = ensemble_median(ledger)
        assert (result[MODEL_NAME] == "ensemble_median").all()

    def test_model_name_custom(self) -> None:
        """Custom name parameter is respected."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 5.0, "M1")]
        ledger = _make_ledger(rows)
        result = ensemble_median(ledger, name="my_ensemble")
        assert (result[MODEL_NAME] == "my_ensemble").all()

    def test_even_number_of_models(self) -> None:
        """With 2 models the median is the mean of the two values."""
        rows = [
            _make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 10.0, "M1"),
            _make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 20.0, "M2"),
        ]
        ledger = _make_ledger(rows)
        result = ensemble_median(ledger)
        assert result.iloc[0][Y_HAT] == pytest.approx(15.0)

    def test_multiple_horizons(self) -> None:
        """Median is computed independently per horizon."""
        origin = "2024-01-01"
        rows = []
        for h, vals in [(1, [10.0, 20.0, 30.0]), (2, [100.0, 200.0, 300.0])]:
            ds = f"2024-01-{7 + h:02d}"
            for i, v in enumerate(vals):
                rows.append(_make_forecast_row("S1", origin, ds, h, v, f"M{i + 1}"))
        ledger = _make_ledger(rows)
        result = ensemble_median(ledger).sort_values(H)

        assert result.iloc[0][Y_HAT] == pytest.approx(20.0)  # median of h=1
        assert result.iloc[1][Y_HAT] == pytest.approx(200.0)  # median of h=2

    def test_empty_ledger(self) -> None:
        """Empty ledger returns an empty DataFrame with required columns."""
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        empty[Y] = empty[Y].astype("float64")
        empty[Y_HAT] = empty[Y_HAT].astype("float64")
        empty[H] = empty[H].astype("int64")
        result = ensemble_median(empty)
        assert result.empty
        for col in REQUIRED_COLUMNS:
            assert col in result.columns
