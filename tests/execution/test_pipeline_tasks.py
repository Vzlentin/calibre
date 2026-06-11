"""Tests for calibre.execution.task_builder.

``build_tasks`` resolves local/global scope exactly once and returns a
pre-partitioned :class:`TaskGroups`. These tests lock both the emitted task set
and the global-dedup outcome (the content key, not ``id(history)``) so the
scope-once refactor stays behaviour-preserving.
"""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.core.forecast_task import ForecastTask, TaskGroups
from calibre.execution.task_builder import build_node_history, build_tasks, partition_tasks
from calibre.reconciliation.summing import (
    TOTAL_LABEL,
    build_hierarchy_index,
    build_summing_matrix,
)


@pytest.fixture
def sample_sales():
    """3 series x 10 daily periods, long format."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = []
    for uid in ["series_a", "series_b", "series_c"]:
        for i, dt in enumerate(dates):
            data.append({"unique_id": uid, "ds": dt, "y": float(10 + i)})
    return pd.DataFrame(data)


@pytest.fixture
def local_configs():
    return [
        {"backend": "statsforecast", "model": "arima"},
        {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor"},
    ]


@pytest.fixture
def global_configs():
    return [
        {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "scope": "global"},
    ]


@pytest.fixture
def statsforecast_global_config():
    return [
        {
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "season_length": 4,
            "scope": "global",
        },
    ]


@pytest.fixture
def sample_hierarchy():
    return pd.DataFrame(
        {
            "unique_id": ["series_a", "series_b", "series_c"],
            "dept_id": ["D1", "D1", "D2"],
            "store_id": ["S1", "S2", "S1"],
        }
    )


class TestBuildNodeHistory:
    def test_hierarchy_none_leaves_history_unchanged(self, sample_sales):
        out = build_node_history(sample_sales, None)
        pd.testing.assert_frame_equal(out, sample_sales)

    def test_builds_aggregate_rows_aligned_to_summing_labels(self, sample_sales, sample_hierarchy):
        out = build_node_history(sample_sales, build_hierarchy_index(sample_hierarchy))
        summing = build_summing_matrix(sample_hierarchy)

        assert out["unique_id"].unique().tolist() == list(summing.node_labels)
        assert "dept_id=D1" in out["unique_id"].unique()
        assert "store_id=S1" in out["unique_id"].unique()
        assert TOTAL_LABEL in out["unique_id"].unique()

        first_day = pd.Timestamp("2024-01-01")
        d1 = out[(out["unique_id"] == "dept_id=D1") & (out["ds"] == first_day)]["y"].iloc[0]
        total = out[(out["unique_id"] == TOTAL_LABEL) & (out["ds"] == first_day)]["y"].iloc[0]
        assert d1 == pytest.approx(20.0)
        assert total == pytest.approx(30.0)

    def test_partial_history_dates_omit_incomplete_aggregates(self, sample_sales, sample_hierarchy):
        partial = sample_sales[
            ~(
                (sample_sales["unique_id"] == "series_b")
                & (sample_sales["ds"] == pd.Timestamp("2024-01-01"))
            )
        ]
        out = build_node_history(partial, build_hierarchy_index(sample_hierarchy))

        first_day = pd.Timestamp("2024-01-01")
        d1 = out[(out["unique_id"] == "dept_id=D1") & (out["ds"] == first_day)]
        total = out[(out["unique_id"] == TOTAL_LABEL) & (out["ds"] == first_day)]
        assert d1.empty
        assert total.empty

    def test_unknown_bottom_id_raises_clear_error(self, sample_sales, sample_hierarchy):
        bad = pd.concat(
            [
                sample_sales,
                pd.DataFrame(
                    [{"unique_id": "series_z", "ds": pd.Timestamp("2024-01-01"), "y": 1.0}]
                ),
            ],
            ignore_index=True,
        )

        with pytest.raises(ValueError, match="not present in hierarchy"):
            build_node_history(bad, build_hierarchy_index(sample_hierarchy))


class TestBuildTasksLocal:
    def test_one_task_per_series_config_pair(self, sample_sales, local_configs):
        groups = build_tasks(sample_sales, local_configs, horizon=7)

        assert isinstance(groups, TaskGroups)
        assert groups.global_ == []
        assert len(groups.local) == 6  # 3 series x 2 configs
        assert {task.unique_id for task in groups.local} == {"series_a", "series_b", "series_c"}
        assert all(task.horizon == 7 for task in groups.local)
        configs_in_tasks = [task.model_config for task in groups.local]
        for cfg in local_configs:
            assert configs_in_tasks.count(cfg) == 3  # one per series

    def test_per_series_history_is_well_formed(self, sample_sales, local_configs):
        groups = build_tasks(sample_sales, local_configs, horizon=5)

        for task in groups.local:
            assert isinstance(task, ForecastTask)
            history = task.history
            assert list(history.columns) == ["unique_id", "ds", "y"]
            assert history["unique_id"].nunique() == 1  # one series per local task
            assert len(history) == 10
            assert pd.api.types.is_datetime64_any_dtype(history["ds"])
            assert history["y"].dtype == "float64"
            assert history["ds"].is_monotonic_increasing

    def test_forecast_origin_and_future_x_default_to_none(self, sample_sales, local_configs):
        groups = build_tasks(sample_sales, local_configs, horizon=5)
        assert all(task.forecast_origin is None for task in groups.local)
        assert all(task.future_x is None for task in groups.local)

    @pytest.mark.parametrize(
        ("series_filter", "expected_uids"),
        [
            (["series_a", "series_b"], {"series_a", "series_b"}),
            ([], set()),
            (None, {"series_a", "series_b", "series_c"}),
        ],
    )
    def test_series_filter_controls_included_series(
        self, sample_sales, local_configs, series_filter, expected_uids
    ):
        groups = build_tasks(sample_sales, local_configs, horizon=5, series_filter=series_filter)
        assert {task.unique_id for task in groups.local} == expected_uids
        assert len(groups.local) == len(expected_uids) * len(local_configs)

    def test_empty_model_configs_yields_no_tasks(self, sample_sales):
        groups = build_tasks(sample_sales, [], horizon=5)
        assert groups.local == []
        assert groups.global_ == []
        assert len(groups) == 0

    def test_hierarchy_creates_one_local_task_per_node(self, sample_sales, sample_hierarchy):
        node_history = build_node_history(sample_sales, build_hierarchy_index(sample_hierarchy))
        groups = build_tasks(
            node_history,
            [{"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 2}],
            horizon=2,
        )
        summing = build_summing_matrix(sample_hierarchy)
        assert [task.unique_id for task in groups.local] == list(summing.node_labels)


class TestBuildTasksGlobal:
    def test_global_yields_one_task_covering_all_series(self, sample_sales, global_configs):
        groups = build_tasks(sample_sales, global_configs, horizon=4)
        assert groups.local == []
        assert len(groups.global_) == 1
        assert set(groups.global_[0].history["unique_id"].unique()) == {
            "series_a",
            "series_b",
            "series_c",
        }

    def test_global_task_series_filter(self, sample_sales, global_configs):
        groups = build_tasks(sample_sales, global_configs, horizon=4, series_filter=["series_a"])
        assert groups.global_[0].history["unique_id"].unique().tolist() == ["series_a"]

    def test_mixed_local_and_global_configs(self, sample_sales, local_configs, global_configs):
        groups = build_tasks(sample_sales, local_configs + global_configs, horizon=4)
        # 2 local configs x 3 series = 6 local tasks + 1 global task.
        assert len(groups.local) == 6
        assert len(groups.global_) == 1
        assert len(groups) == 7
        assert len(groups.tasks) == 7

    def test_global_scope_works_for_non_mlforecast_backend(
        self, sample_sales, statsforecast_global_config
    ):
        groups = build_tasks(sample_sales, statsforecast_global_config, horizon=4)
        assert groups.local == []
        assert len(groups.global_) == 1
        assert set(groups.global_[0].history["unique_id"].unique()) == {
            "series_a",
            "series_b",
            "series_c",
        }

    def test_hierarchy_global_scope_deduplicates_node_panel(
        self, sample_sales, sample_hierarchy, statsforecast_global_config
    ):
        node_history = build_node_history(sample_sales, build_hierarchy_index(sample_hierarchy))
        groups = build_tasks(
            node_history,
            statsforecast_global_config,
            horizon=4,
        )
        assert groups.local == []
        assert len(groups.global_) == 1
        assert set(groups.global_[0].history["unique_id"].unique()) == set(
            build_summing_matrix(sample_hierarchy).node_labels
        )


class TestBuildTasksOverrides:
    def test_override_swaps_model_list_for_one_series(self, sample_sales, local_configs):
        override_cfg = [{"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4}]
        overrides = {"series_a": override_cfg}
        groups = build_tasks(sample_sales, local_configs, horizon=5, overrides=overrides)

        # series_a: 1 override config; series_b/series_c: 2 default configs each → 5 local tasks.
        assert len(groups.local) == 5
        assert groups.global_ == []

        a_tasks = [t for t in groups.local if t.unique_id == "series_a"]
        assert len(a_tasks) == 1
        assert a_tasks[0].model_config == override_cfg[0]

        b_tasks = [t for t in groups.local if t.unique_id == "series_b"]
        assert len(b_tasks) == 2

    def test_series_without_override_keep_defaults(self, sample_sales, local_configs):
        override_cfg = [{"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4}]
        overrides = {"series_a": override_cfg}
        groups = build_tasks(sample_sales, local_configs, horizon=5, overrides=overrides)

        for uid in ["series_b", "series_c"]:
            uid_tasks = [t for t in groups.local if t.unique_id == uid]
            assert len(uid_tasks) == 2
            configs = [t.model_config for t in uid_tasks]
            assert local_configs[0] in configs
            assert local_configs[1] in configs

    def test_unknown_uid_in_overrides_raises(self, sample_sales, local_configs):
        overrides = {"series_z": [{"backend": "statsforecast", "model": "SeasonalNaive"}]}
        with pytest.raises(ValueError, match="overrides contains unknown unique_id"):
            build_tasks(sample_sales, local_configs, horizon=5, overrides=overrides)

    def test_hierarchy_override_accepts_known_aggregate_label(self, sample_sales, sample_hierarchy):
        override_cfg = [{"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 2}]
        node_history = build_node_history(sample_sales, build_hierarchy_index(sample_hierarchy))
        groups = build_tasks(
            node_history,
            [{"backend": "statsforecast", "model": "Naive"}],
            horizon=2,
            overrides={"dept_id=D1": override_cfg},
        )

        aggregate_tasks = [task for task in groups.local if task.unique_id == "dept_id=D1"]
        assert len(aggregate_tasks) == 1
        assert aggregate_tasks[0].model_config == override_cfg[0]

    def test_global_override_deduplicates(self, sample_sales):
        global_cfg = [
            {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "scope": "global"}
        ]
        overrides = {"series_a": global_cfg, "series_b": global_cfg}
        groups = build_tasks(sample_sales, [], horizon=5, overrides=overrides)

        # Both uids share the same global config → exactly 1 global task.
        assert groups.local == []
        assert len(groups.global_) == 1
        assert groups.global_[0].model_config == global_cfg[0]
        assert set(groups.global_[0].history["unique_id"].unique()) == {
            "series_a",
            "series_b",
            "series_c",
        }

    def test_override_with_local_and_global_mixed(self, sample_sales, local_configs):
        global_cfg = [
            {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "scope": "global"}
        ]
        overrides = {"series_a": global_cfg}
        groups = build_tasks(sample_sales, local_configs, horizon=5, overrides=overrides)

        # series_a → 1 global task (all series); series_b/series_c → 2 local configs each = 4.
        assert len(groups.global_) == 1
        assert len(groups.local) == 4

    def test_global_override_with_list_config_deduplicates(self, sample_sales):
        global_cfg = [
            {
                "backend": "mlforecast",
                "model": "lightgbm.LGBMRegressor",
                "scope": "global",
                "features": ["rolling_mean_7", "lag_14"],
            }
        ]
        overrides = {"series_a": global_cfg, "series_b": global_cfg}
        groups = build_tasks(sample_sales, [], horizon=5, overrides=overrides)

        # Unhashable (list) values must not break dedup → exactly 1 global task.
        assert len(groups.global_) == 1
        assert groups.global_[0].model_config == global_cfg[0]


class TestGlobalDedupContentKey:
    """The content key (uid-set + config JSON + horizon), not ``id(history)``,
    drives global dedup. These lock the de-fragilized behaviour."""

    def test_key_is_stable_across_defensive_copy_of_history(self, sample_sales, global_configs):
        """``_global_dedup_key`` is keyed on content, not object identity: a
        defensive ``.copy()`` of the history frame (distinct ``id()``, identical
        content) yields the same key, so the two tasks dedup. Under the old
        ``id(history)`` key the clone produced a different key and slipped
        through as a duplicate global task.

        Scope: this is a unit test of the key function. It does not assert a
        ``build_tasks`` end-to-end path — within one ``build_tasks`` call every
        global task shares the same ``data`` object, so a copied frame never
        reaches the dedup there.
        """
        from calibre.execution.task_builder import _global_dedup_key

        groups = build_tasks(sample_sales, global_configs, horizon=4)
        assert len(groups.global_) == 1
        base = groups.global_[0]

        cloned = ForecastTask(
            history=base.history.copy(),  # different object, identical content
            horizon=base.horizon,
            model_config=dict(base.model_config),
        )
        assert id(cloned.history) != id(base.history)

        # The content key — not object identity — collapses the clone: equal
        # keys mean a `set`-based dedup keeps exactly one of the two tasks.
        assert _global_dedup_key(base) == _global_dedup_key(cloned)

    def test_content_key_distinguishes_horizon_and_config(self, sample_sales, global_configs):
        groups = build_tasks(sample_sales, global_configs, horizon=4)
        base = groups.global_[0]

        from calibre.execution.task_builder import _global_dedup_key

        diff_horizon = ForecastTask(
            history=base.history, horizon=base.horizon + 1, model_config=dict(base.model_config)
        )
        diff_config = ForecastTask(
            history=base.history,
            horizon=base.horizon,
            model_config={**base.model_config, "model": "other.Model"},
        )
        assert _global_dedup_key(base) != _global_dedup_key(diff_horizon)
        assert _global_dedup_key(base) != _global_dedup_key(diff_config)


class TestPartitionTasks:
    def test_routes_by_resolved_scope(self, sample_sales):
        local = ForecastTask(
            history=sample_sales[sample_sales["unique_id"] == "series_a"],
            horizon=4,
            model_config={"backend": "statsforecast", "model": "arima"},
        )
        global_task = ForecastTask(
            history=sample_sales,
            horizon=4,
            model_config={
                "backend": "mlforecast",
                "model": "lightgbm.LGBMRegressor",
                "scope": "global",
            },
        )
        groups = partition_tasks([local, global_task])
        assert groups.local == [local]
        assert groups.global_ == [global_task]
