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
def local_configs():
    """Two local (per-series) model configs."""
    return [
        {"backend": "statsforecast", "model": "arima"},
        {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor"},
    ]


@pytest.fixture
def global_configs():
    """One global model config."""
    return [
        {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "scope": "global"},
    ]


@pytest.fixture
def statsforecast_global_config():
    """statsforecast config with scope='global' (joint dispatch, no Fugue partitioning)."""
    return [
        {
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "season_length": 4,
            "scope": "global",
        },
    ]


class TestBuildTasksLocal:
    def test_task_count_is_series_times_configs(self, sample_sales, local_configs):
        """3 series × 2 configs = 6 tasks."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        assert len(tasks) == 6

    def test_task_type(self, sample_sales, local_configs):
        """Each task should be a ForecastTask instance."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        assert all(isinstance(task, ForecastTask) for task in tasks)

    def test_history_columns(self, sample_sales, local_configs):
        """Each task's history should have [unique_id, ds, y] columns."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        for task in tasks:
            assert list(task.history.columns) == ["unique_id", "ds", "y"]

    def test_history_single_uid(self, sample_sales, local_configs):
        """Each per-series task's history should contain exactly one unique_id."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        for task in tasks:
            assert task.history["unique_id"].nunique() == 1

    def test_history_contains_data(self, sample_sales, local_configs):
        """Each task's history should have the same number of rows as the input series."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        for task in tasks:
            assert len(task.history) == 10

    def test_unique_ids_in_tasks(self, sample_sales, local_configs):
        """Tasks should have correct unique_ids matching the series."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        unique_ids = {task.unique_id for task in tasks}
        assert unique_ids == {"series_a", "series_b", "series_c"}

    def test_horizon_preserved(self, sample_sales, local_configs):
        """Each task should have the correct horizon."""
        horizon = 7
        tasks = build_tasks(sample_sales, local_configs, horizon=horizon)
        assert all(task.horizon == horizon for task in tasks)

    def test_model_config_preserved(self, sample_sales, local_configs):
        """Each task should have its original model_config."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        configs_in_tasks = [task.model_config for task in tasks]
        for cfg in local_configs:
            assert any(tc == cfg for tc in configs_in_tasks)

    def test_series_filter_restricts_series(self, sample_sales, local_configs):
        """series_filter should restrict which unique_ids appear in output."""
        tasks = build_tasks(
            sample_sales, local_configs, horizon=5, series_filter=["series_a", "series_b"]
        )
        assert len(tasks) == 4
        unique_ids = {task.unique_id for task in tasks}
        assert unique_ids == {"series_a", "series_b"}

    def test_empty_series_filter_yields_no_tasks(self, sample_sales, local_configs):
        """series_filter=[] should yield no tasks."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5, series_filter=[])
        assert len(tasks) == 0

    def test_series_filter_none_includes_all(self, sample_sales, local_configs):
        """series_filter=None should include all series."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5, series_filter=None)
        assert len(tasks) == 6

    def test_empty_model_configs_yields_no_tasks(self, sample_sales):
        """Empty model_configs list should yield no tasks."""
        tasks = build_tasks(sample_sales, [], horizon=5)
        assert len(tasks) == 0

    def test_ds_dtype_preserved(self, sample_sales, local_configs):
        """history[ds] should be datetime64."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        for task in tasks:
            assert pd.api.types.is_datetime64_any_dtype(task.history["ds"])

    def test_y_dtype_preserved(self, sample_sales, local_configs):
        """history[y] should be float64."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        for task in tasks:
            assert task.history["y"].dtype == "float64"

    def test_history_sorted_by_ds(self, sample_sales, local_configs):
        """history should be sorted by ds within each series."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        for task in tasks:
            expected = task.history.sort_values("ds").reset_index(drop=True)
            pd.testing.assert_frame_equal(task.history.reset_index(drop=True), expected)

    def test_forecast_origin_is_none_by_default(self, sample_sales, local_configs):
        """forecast_origin should be None by default."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        assert all(task.forecast_origin is None for task in tasks)

    def test_future_x_is_none_by_default(self, sample_sales, local_configs):
        """future_x should be None by default."""
        tasks = build_tasks(sample_sales, local_configs, horizon=5)
        assert all(task.future_x is None for task in tasks)


class TestBuildTasksGlobal:
    def test_global_yields_one_task_per_config(self, sample_sales, global_configs):
        """Global backend produces one task per config regardless of series count."""
        tasks = build_tasks(sample_sales, global_configs, horizon=4)
        assert len(tasks) == 1

    def test_global_task_has_all_series(self, sample_sales, global_configs):
        """Global task history contains all unique_ids."""
        tasks = build_tasks(sample_sales, global_configs, horizon=4)
        uids = tasks[0].history["unique_id"].unique().tolist()
        assert set(uids) == {"series_a", "series_b", "series_c"}

    def test_global_task_series_filter(self, sample_sales, global_configs):
        """Global task respects series_filter."""
        tasks = build_tasks(sample_sales, global_configs, horizon=4, series_filter=["series_a"])
        assert tasks[0].history["unique_id"].unique().tolist() == ["series_a"]

    def test_mixed_local_and_global_configs(self, sample_sales, local_configs, global_configs):
        """Mix of local and global configs should produce correct task counts."""
        tasks = build_tasks(sample_sales, local_configs + global_configs, horizon=4)
        # 2 local configs × 3 series = 6 local tasks + 1 global task = 7
        assert len(tasks) == 7

    def test_global_scope_works_for_non_mlforecast_backend(
        self, sample_sales, statsforecast_global_config
    ):
        """scope='global' on statsforecast emits one task containing all series."""
        tasks = build_tasks(sample_sales, statsforecast_global_config, horizon=4)
        assert len(tasks) == 1
        assert set(tasks[0].history["unique_id"].unique()) == {
            "series_a",
            "series_b",
            "series_c",
        }
