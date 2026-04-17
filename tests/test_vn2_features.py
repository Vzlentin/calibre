"""Tests for VN2 feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.vn2.features import (
    add_calendar_features,
    add_lag_features,
    add_master_features,
    add_rolling_features,
    add_series_scaling,
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
    """In-stock data: mark a few periods as out-of-stock."""
    df = sample_sales[["unique_id", "ds"]].copy()
    df["in_stock"] = True
    # Mark weeks 5-7 as OOS for series 1_100
    mask = (df["unique_id"] == "1_100") & (df["ds"].dt.isocalendar().week.isin([6, 7, 8]))
    df.loc[mask, "in_stock"] = False
    return df


@pytest.fixture
def sample_master() -> pd.DataFrame:
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
        # OOS rows should have y_uncensored >= y (imputed demand >= observed)
        oos = result[~result["in_stock"]]
        if not oos.empty:
            assert (oos["y_uncensored"] >= oos["y"]).all()


class TestSeriesScaling:
    def test_adds_scaling_columns(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_series_scaling(df)
        assert "series_mean" in result.columns
        assert "series_std" in result.columns
        assert "y_scaled" in result.columns

    def test_std_floored_at_one(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_series_scaling(df)
        assert (result["series_std"] >= 1.0).all()


class TestLagFeatures:
    def test_creates_lag_columns(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_lag_features(df, lags=[1, 2, 4])
        assert "lag_1" in result.columns
        assert "lag_2" in result.columns
        assert "lag_4" in result.columns

    def test_lag_values_correct(self, sample_sales: pd.DataFrame) -> None:
        df = add_stockout_features(sample_sales, instock=None)
        result = add_lag_features(df, lags=[1])
        s1 = result[result["unique_id"] == "1_100"].sort_values("ds")
        # lag_1 at row i should equal y_uncensored at row i-1
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
        assert "week_of_year" in result.columns
        assert "month" in result.columns
        assert "quarter" in result.columns
        assert result["week_of_year"].between(1, 53).all()
        assert result["month"].between(1, 12).all()


class TestMasterFeatures:
    def test_merges_master_columns(
        self,
        sample_sales: pd.DataFrame,
        sample_master: pd.DataFrame,
    ) -> None:
        result = add_master_features(sample_sales, sample_master)
        assert "ProductGroup" in result.columns
        assert "Department" in result.columns
        assert result["ProductGroup"].dtype.name == "category"

    def test_without_master(self, sample_sales: pd.DataFrame) -> None:
        result = add_master_features(sample_sales, master=None)
        assert "ProductGroup" not in result.columns


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
        # Last observation should have highest weight
        assert s1["sample_weight"].iloc[-1] >= s1["sample_weight"].iloc[0]


class TestBuildTrainingFrame:
    def test_end_to_end(self, sample_sales: pd.DataFrame) -> None:
        result = build_training_frame(sample_sales)
        assert "y_uncensored" in result.columns
        assert "y_scaled" in result.columns
        assert "lag_1" in result.columns
        assert "rolling_mean_4" in result.columns
        assert "week_of_year" in result.columns
        assert "sample_weight" in result.columns
        assert len(result) == len(sample_sales)

    def test_with_all_data(
        self,
        sample_sales: pd.DataFrame,
        sample_instock: pd.DataFrame,
        sample_master: pd.DataFrame,
    ) -> None:
        result = build_training_frame(
            sample_sales,
            instock=sample_instock,
            master=sample_master,
        )
        assert "in_stock" in result.columns
        assert "ProductGroup" in result.columns
        assert "lag_52" in result.columns
