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
    quantile_column,
    validate_forecast_frame,
)
from calibre.ensemble.median import ensemble_median
from calibre.ensemble.weighted import ensemble_inverse_error, ensemble_weighted


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


class TestEnsembleWeighted:
    def test_equal_weights_matches_mean(self) -> None:
        """Two frames with equal weights → arithmetic mean of y_hat."""
        origin = "2024-01-01"
        ds = "2024-01-08"
        rows_a = [_make_forecast_row("S1", origin, ds, 1, 10.0, "M1")]
        rows_b = [_make_forecast_row("S1", origin, ds, 1, 20.0, "M2")]
        frame_a = _make_ledger(rows_a)
        frame_b = _make_ledger(rows_b)

        result = ensemble_weighted([frame_a, frame_b], [0.5, 0.5])
        assert len(result) == 1
        assert result.iloc[0][Y_HAT] == pytest.approx(15.0)
        assert (result[MODEL_NAME] == "ensemble_weighted").all()

    def test_single_frame_weight_one(self) -> None:
        """One frame with weight 1.0 is returned unchanged."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 42.0, "M1")]
        frame = _make_ledger(rows)
        result = ensemble_weighted([frame], [1.0], name="solo")
        assert len(result) == 1
        assert result.iloc[0][Y_HAT] == pytest.approx(42.0)

    def test_weight_count_mismatch_raises(self) -> None:
        """frames and weights of different lengths raise ValueError."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 1.0, "M1")]
        frame = _make_ledger(rows)
        with pytest.raises(ValueError, match="same length"):
            ensemble_weighted([frame, frame], [0.5])

    def test_weights_not_unit_sum_raises(self) -> None:
        """Weights that do not sum to 1.0 raise ValueError."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 1.0, "M1")]
        frame = _make_ledger(rows)
        with pytest.raises(ValueError, match="sum to 1.0"):
            ensemble_weighted([frame, frame], [0.3, 0.6])

    def test_output_passes_validate_forecast_frame(self) -> None:
        """Weighted ensemble output must be a valid forecast frame."""
        rows_a = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 10.0, "M1")]
        rows_b = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 20.0, "M2")]
        result = ensemble_weighted([_make_ledger(rows_a), _make_ledger(rows_b)], [0.5, 0.5])
        validate_forecast_frame(result)

    def test_y_is_nan(self) -> None:
        """Ensemble output rows must have y = NaN."""
        rows_a = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 10.0, "M1")]
        rows_b = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 20.0, "M2")]
        result = ensemble_weighted([_make_ledger(rows_a), _make_ledger(rows_b)], [0.5, 0.5])
        assert result[Y].isna().all()

    def test_quantile_columns_ensembled(self) -> None:
        """q_* columns are combined row-wise with the same weights."""
        origin = "2024-01-01"
        ds = "2024-01-08"
        qcol = quantile_column(0.833)
        rows_a = [_make_forecast_row("S1", origin, ds, 1, 10.0, "M1")]
        rows_b = [_make_forecast_row("S1", origin, ds, 1, 20.0, "M2")]
        frame_a = _make_ledger(rows_a)
        frame_b = _make_ledger(rows_b)
        frame_a[qcol] = 100.0
        frame_b[qcol] = 200.0

        result = ensemble_weighted([frame_a, frame_b], [0.25, 0.75])
        assert result.iloc[0][qcol] == pytest.approx(175.0)

    def test_empty_frames_returns_empty_frame(self) -> None:
        """Empty list of frames returns empty DataFrame with required columns."""
        result = ensemble_weighted([], [])
        assert result.empty
        for col in REQUIRED_COLUMNS:
            assert col in result.columns

    def test_different_group_keys_raises(self) -> None:
        """Frames with different group keys raise ValueError."""
        rows_a = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 10.0, "M1")]
        rows_b = [_make_forecast_row("S2", "2024-01-01", "2024-01-08", 1, 20.0, "M2")]
        with pytest.raises(ValueError, match="group keys"):
            ensemble_weighted([_make_ledger(rows_a), _make_ledger(rows_b)], [0.5, 0.5])


class TestEnsembleInverseError:
    def test_larger_weight_to_smaller_error(self) -> None:
        """Model with smaller error gets larger weight."""
        origin = "2024-01-01"
        ds = "2024-01-08"
        rows_a = [_make_forecast_row("S1", origin, ds, 1, 10.0, "M1")]
        rows_b = [_make_forecast_row("S1", origin, ds, 1, 30.0, "M2")]
        frame_a = _make_ledger(rows_a)
        frame_b = _make_ledger(rows_b)

        # errors: M1=1.0, M2=3.0 → weights: M1=0.75, M2=0.25
        result = ensemble_inverse_error([frame_a, frame_b], [1.0, 3.0])
        expected = 10.0 * 0.75 + 30.0 * 0.25
        assert result.iloc[0][Y_HAT] == pytest.approx(expected)

    def test_error_count_mismatch_raises(self) -> None:
        """frames and errors of different lengths raise ValueError."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 1.0, "M1")]
        frame = _make_ledger(rows)
        with pytest.raises(ValueError, match="same length"):
            ensemble_inverse_error([frame, frame], [1.0])

    def test_non_positive_error_raises(self) -> None:
        """Zero or negative errors raise ValueError."""
        rows = [_make_forecast_row("S1", "2024-01-01", "2024-01-08", 1, 1.0, "M1")]
        frame = _make_ledger(rows)
        with pytest.raises(ValueError, match="all errors must be > 0"):
            ensemble_inverse_error([frame, frame], [1.0, 0.0])
