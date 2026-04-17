"""Tests for the VN2 winning approach pipeline.

Uses synthetic data since VN2 competition data may not be available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.vn2.features import build_training_frame
from benchmarks.vn2.run_winning import (
    _get_feature_cols,
    predict_global,
    train_global_models,
)


@pytest.fixture
def synthetic_sales() -> pd.DataFrame:
    """Three series, 60 weeks each with seasonal pattern."""
    dates = pd.date_range("2023-01-08", periods=60, freq="W")
    rows = []
    rng = np.random.default_rng(42)
    for uid in ["1_100", "2_200", "3_300"]:
        base = rng.uniform(5, 50)
        for i, d in enumerate(dates):
            # Seasonal pattern + noise
            seasonal = 5 * np.sin(2 * np.pi * i / 52)
            noise = rng.normal(0, 2)
            y = max(base + seasonal + noise, 0)
            rows.append({"unique_id": uid, "ds": d, "y": float(y)})
    return pd.DataFrame(rows)


@pytest.fixture
def training_frame(synthetic_sales: pd.DataFrame) -> pd.DataFrame:
    return build_training_frame(synthetic_sales)


class TestTrainGlobalModels:
    def test_trains_one_model_per_horizon(
        self,
        training_frame: pd.DataFrame,
    ) -> None:
        models = train_global_models(training_frame, horizon=3)
        assert len(models) == 3

    def test_models_can_predict(self, training_frame: pd.DataFrame) -> None:
        models = train_global_models(training_frame, horizon=3)
        feature_cols = _get_feature_cols(training_frame)
        latest = (
            training_frame.sort_values(["unique_id", "ds"])
            .groupby("unique_id", sort=False)
            .last()
            .reset_index()
        )
        X = latest[feature_cols]
        for model in models:
            preds = model.predict(X)
            assert len(preds) == len(latest)
            assert all(np.isfinite(preds))


class TestPredictGlobal:
    def test_output_shape(self, training_frame: pd.DataFrame) -> None:
        models = train_global_models(training_frame, horizon=3)
        origin = training_frame["ds"].max()
        result = predict_global(models, training_frame, pd.Timestamp(origin))

        n_series = training_frame["unique_id"].nunique()
        assert len(result) == n_series * 3  # 3 series × 3 horizons

    def test_output_columns(self, training_frame: pd.DataFrame) -> None:
        models = train_global_models(training_frame, horizon=3)
        origin = training_frame["ds"].max()
        result = predict_global(models, training_frame, pd.Timestamp(origin))

        expected_cols = {"unique_id", "ds", "y", "y_hat", "h", "forecast_origin", "model_name"}
        assert expected_cols.issubset(set(result.columns))

    def test_predictions_non_negative(
        self,
        training_frame: pd.DataFrame,
    ) -> None:
        models = train_global_models(training_frame, horizon=3)
        origin = training_frame["ds"].max()
        result = predict_global(models, training_frame, pd.Timestamp(origin))
        assert (result["y_hat"] >= 0).all()

    def test_horizons_correct(self, training_frame: pd.DataFrame) -> None:
        models = train_global_models(training_frame, horizon=3)
        origin = training_frame["ds"].max()
        result = predict_global(models, training_frame, pd.Timestamp(origin))
        assert set(result["h"].unique()) == {1, 2, 3}

    def test_model_name_is_global_lgbm(
        self,
        training_frame: pd.DataFrame,
    ) -> None:
        models = train_global_models(training_frame, horizon=3)
        origin = training_frame["ds"].max()
        result = predict_global(models, training_frame, pd.Timestamp(origin))
        assert (result["model_name"] == "global_lgbm").all()

    def test_dtypes_match_forecast_frame(
        self,
        training_frame: pd.DataFrame,
    ) -> None:
        models = train_global_models(training_frame, horizon=3)
        origin = training_frame["ds"].max()
        result = predict_global(models, training_frame, pd.Timestamp(origin))

        assert result["y_hat"].dtype == np.float64
        assert result["y"].dtype == np.float64
        assert result["h"].dtype == np.int64
        assert pd.api.types.is_datetime64_any_dtype(result["ds"])
        assert pd.api.types.is_datetime64_any_dtype(result["forecast_origin"])
