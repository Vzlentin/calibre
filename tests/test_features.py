"""Tests for calibre.forecasting.features package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.forecasting.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_series_scaling,
    add_static_features,
    add_stockout_features,
    add_time_weights,
    build_training_frame,
)


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Two series, 20 weeks each."""
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    rows = []
    for uid in ["1_100", "2_200"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": uid, "ds": d, "y": float(10 + i)})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_instock(sample_sales: pd.DataFrame) -> pd.DataFrame:
    """In-stock data: mark a few periods as out-of-stock for one series."""
    df = sample_sales[["unique_id", "ds"]].copy()
    df["in_stock"] = True
    mask = (df["unique_id"] == "1_100") & (df["ds"].dt.isocalendar().week.isin([6, 7, 8]))
    df.loc[mask, "in_stock"] = False
    return df


@pytest.fixture
def sample_static() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["1_100", "2_200"],
            "Store": [1, 2],
            "Product": [100, 200],
            "ProductGroup": ["A", "B"],
            "Department": ["Dairy", "Bakery"],
        }
    )


class TestStockoutFeatures:
    def test_without_instock_data(self, sample_sales: pd.DataFrame) -> None:
        result = add_stockout_features(sample_sales, instock=None)
        assert "in_stock" in result.columns
        assert "y_uncensored" in result.columns
        assert result["in_stock"].all()
        pd.testing.assert_series_equal(
            result["y_uncensored"],
            result["y"],
            check_names=False,
        )

    def test_with_instock_data(
        self,
        sample_sales: pd.DataFrame,
        sample_instock: pd.DataFrame,
    ) -> None:
        result = add_stockout_features(sample_sales, sample_instock)
        assert "in_stock" in result.columns
        assert "y_uncensored" in result.columns
        oos = result[~result["in_stock"]]
        if not oos.empty:
            assert (oos["y_uncensored"] >= oos["y"]).all()


class TestSeriesScaling:
    def test_adds_scaling_columns(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_series_scaling(df)
        assert {"series_mean", "series_std", "y_scaled"} <= set(result.columns)

    def test_std_floored_at_one(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_series_scaling(df)
        assert (result["series_std"] >= 1.0).all()


class TestLagFeatures:
    def test_creates_lag_columns(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_lag_features(df, lags=[1, 2, 4])
        for col in ("lag_1", "lag_2", "lag_4"):
            assert col in result.columns

    def test_lag_values_correct(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_lag_features(df, lags=[1])
        s1 = result[result["unique_id"] == "1_100"].sort_values("ds")
        assert np.isnan(s1["lag_1"].iloc[0])
        assert s1["lag_1"].iloc[1] == s1["y_uncensored"].iloc[0]


class TestRollingFeatures:
    def test_creates_rolling_columns(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_rolling_features(df, windows=[4])
        assert "rolling_mean_4" in result.columns
        assert "rolling_std_4" in result.columns


class TestCalendarFeatures:
    def test_adds_calendar_columns(self, sample_sales: pd.DataFrame) -> None:
        result = add_calendar_features(sample_sales)
        assert {"week_of_year", "month", "quarter"} <= set(result.columns)
        assert result["week_of_year"].between(1, 53).all()
        assert result["month"].between(1, 12).all()


class TestStaticFeatures:
    def test_merges_static_columns(
        self,
        sample_sales: pd.DataFrame,
        sample_static: pd.DataFrame,
    ) -> None:
        result = add_static_features(sample_sales, sample_static)
        assert "ProductGroup" in result.columns
        assert "Department" in result.columns
        assert result["ProductGroup"].dtype.name == "category"

    def test_without_static(self, sample_sales: pd.DataFrame) -> None:
        result = add_static_features(sample_sales, static=None)
        assert "ProductGroup" not in result.columns

    def test_excluded_columns_not_merged(
        self,
        sample_sales: pd.DataFrame,
        sample_static: pd.DataFrame,
    ) -> None:
        result = add_static_features(sample_sales, sample_static)
        # Store / Product are in the default exclude set
        assert "Store" not in result.columns
        assert "Product" not in result.columns


class TestTimeWeights:
    def test_adds_weight_column(self, sample_sales: pd.DataFrame) -> None:
        result = add_time_weights(sample_sales)
        assert "sample_weight" in result.columns
        assert (result["sample_weight"] > 0).all()
        assert (result["sample_weight"] <= 1.0).all()

    def test_recent_observations_weighted_higher(
        self,
        sample_sales: pd.DataFrame,
    ) -> None:
        result = add_time_weights(sample_sales)
        s1 = result[result["unique_id"] == "1_100"].sort_values("ds")
        assert s1["sample_weight"].iloc[-1] >= s1["sample_weight"].iloc[0]


class TestBuildTrainingFrame:
    def test_end_to_end(self, sample_sales: pd.DataFrame) -> None:
        result = build_training_frame(sample_sales)
        for col in (
            "y_uncensored",
            "y_scaled",
            "lag_1",
            "rolling_mean_4",
            "week_of_year",
            "sample_weight",
        ):
            assert col in result.columns
        assert len(result) == len(sample_sales)

    def test_with_all_data(
        self,
        sample_sales: pd.DataFrame,
        sample_instock: pd.DataFrame,
        sample_static: pd.DataFrame,
    ) -> None:
        result = build_training_frame(
            sample_sales,
            instock=sample_instock,
            static=sample_static,
        )
        assert "in_stock" in result.columns
        assert "ProductGroup" in result.columns
        assert "lag_52" in result.columns
