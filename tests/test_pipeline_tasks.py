"""Tests for calibre.pipeline.tasks."""
from __future__ import annotations

import pandas as pd
import pytest

from calibre.pipeline.tasks import build_tasks
from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def sample_sales():
    """Create a simple long-format sales DataFrame with 3 series, 10 periods each."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = []
    for uid in ["series_a", "series_b", "series_c"]:
        for i, dt in enumerate(dates):
            data.append({"unique_id": uid, "ds": dt, "y": float(10 + i)})
    return pd.DataFrame(data)


@pytest.fixture
def model_configs():
    """Two model configs, each with backend and model keys."""
    return [
        {"backend": "statsforecast", "model": "arima"},
        {"backend": "mlforecast", "model": "xgboost"},
    ]


class TestBuildTasks:
    def test_task_count_is_series_times_configs(self, sample_sales, model_configs):
        """3 series × 2 configs = 6 tasks."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        assert len(tasks) == 6

    def test_task_type(self, sample_sales, model_configs):
        """Each task should be a ForecastTask instance."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        assert all(isinstance(task, ForecastTask) for task in tasks)

    def test_history_columns(self, sample_sales, model_configs):
        """Each task's history should have exactly [ds, y] columns (no unique_id)."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        for task in tasks:
            assert list(task.history.columns) == ["ds", "y"]

    def test_history_contains_data(self, sample_sales, model_configs):
        """Each task's history should have the same number of rows as the input series."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        # sample_sales has 10 rows per series
        for task in tasks:
            assert len(task.history) == 10

    def test_unique_ids_in_tasks(self, sample_sales, model_configs):
        """Tasks should have correct unique_ids matching the series."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        unique_ids = {task.unique_id for task in tasks}
        assert unique_ids == {"series_a", "series_b", "series_c"}

    def test_horizon_preserved(self, sample_sales, model_configs):
        """Each task should have the correct horizon."""
        horizon = 7
        tasks = build_tasks(sample_sales, model_configs, horizon=horizon)
        assert all(task.horizon == horizon for task in tasks)

    def test_model_config_preserved(self, sample_sales, model_configs):
        """Each task should have its original model_config."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        # Should have tasks with the correct model configs
        configs_in_tasks = [task.model_config for task in tasks]
        # Check that each original config appears in tasks
        for cfg in model_configs:
            assert any(tc == cfg for tc in configs_in_tasks)

    def test_series_filter_restricts_series(self, sample_sales, model_configs):
        """series_filter should restrict which unique_ids appear in output."""
        tasks = build_tasks(
            sample_sales, model_configs, horizon=5, series_filter=["series_a", "series_b"]
        )
        # 2 series × 2 configs = 4 tasks
        assert len(tasks) == 4
        unique_ids = {task.unique_id for task in tasks}
        assert unique_ids == {"series_a", "series_b"}

    def test_empty_series_filter_yields_no_tasks(self, sample_sales, model_configs):
        """series_filter=[] should yield no tasks."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5, series_filter=[])
        assert len(tasks) == 0

    def test_series_filter_none_includes_all(self, sample_sales, model_configs):
        """series_filter=None should include all series."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5, series_filter=None)
        assert len(tasks) == 6

    def test_empty_model_configs_yields_no_tasks(self, sample_sales):
        """Empty model_configs list should yield no tasks."""
        tasks = build_tasks(sample_sales, [], horizon=5)
        assert len(tasks) == 0

    def test_empty_sales_with_empty_model_configs(self):
        """Empty sales and empty model_configs should yield no tasks."""
        empty_sales = pd.DataFrame(columns=["unique_id", "ds", "y"])
        tasks = build_tasks(empty_sales, [], horizon=5)
        assert len(tasks) == 0

    def test_ds_dtype_preserved(self, sample_sales, model_configs):
        """history[ds] should be datetime64[ns]."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        for task in tasks:
            assert pd.api.types.is_datetime64_any_dtype(task.history["ds"])

    def test_y_dtype_preserved(self, sample_sales, model_configs):
        """history[y] should be float64."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        for task in tasks:
            assert task.history["y"].dtype == "float64"

    def test_history_sorted_by_ds(self, sample_sales, model_configs):
        """history should be sorted by ds."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        for task in tasks:
            expected = task.history.sort_values("ds").reset_index(drop=True)
            pd.testing.assert_frame_equal(task.history.reset_index(drop=True), expected)

    def test_single_series_single_config(self):
        """Single series and single config should yield 1 task."""
        sales = pd.DataFrame(
            {
                "unique_id": ["s1"] * 5,
                "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        config = [{"backend": "test", "model": "test_model"}]
        tasks = build_tasks(sales, config, horizon=2)
        assert len(tasks) == 1
        assert tasks[0].unique_id == "s1"
        assert tasks[0].horizon == 2
        assert tasks[0].model_config == config[0]

    def test_multiple_configs_per_series(self, sample_sales):
        """Each series should get a task for each config."""
        configs = [
            {"backend": "b1", "model": "m1"},
            {"backend": "b2", "model": "m2"},
            {"backend": "b3", "model": "m3"},
        ]
        tasks = build_tasks(sample_sales, configs, horizon=3)
        # 3 series × 3 configs = 9 tasks
        assert len(tasks) == 9
        # Each series should appear exactly 3 times (once per config)
        for uid in ["series_a", "series_b", "series_c"]:
            count = sum(1 for task in tasks if task.unique_id == uid)
            assert count == 3

    def test_forecast_origin_is_none_by_default(self, sample_sales, model_configs):
        """forecast_origin should be None by default (engine handles it)."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        assert all(task.forecast_origin is None for task in tasks)

    def test_future_x_is_none_by_default(self, sample_sales, model_configs):
        """future_x should be None by default."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        assert all(task.future_x is None for task in tasks)

    def test_series_filter_single_series(self, sample_sales, model_configs):
        """series_filter with a single series."""
        tasks = build_tasks(sample_sales, model_configs, horizon=5, series_filter=["series_c"])
        assert len(tasks) == 2  # 1 series × 2 configs
        assert all(task.unique_id == "series_c" for task in tasks)

    def test_model_config_dict_not_mutated(self, sample_sales, model_configs):
        """Original model_config dicts should not be mutated."""
        original_configs = [cfg.copy() for cfg in model_configs]
        tasks = build_tasks(sample_sales, model_configs, horizon=5)
        assert model_configs == original_configs
