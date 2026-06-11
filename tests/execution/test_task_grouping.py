from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y
from calibre.core.forecast_task import ForecastTask, TaskGroups
from calibre.execution.backend import BackendEngine, ExecutionOptions, _group_local_tasks_by_config
from calibre.execution.task_builder import partition_tasks


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    return pd.concat(
        [
            pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [10.0, 20.0] * 6}),
            pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: [5.0, 15.0] * 6}),
        ],
        ignore_index=True,
    )


def _tasks(panel: pd.DataFrame, *, grouped: bool) -> TaskGroups:
    tasks: list[ForecastTask] = []
    for _, group in panel.groupby(UNIQUE_ID, sort=False):
        tasks.append(
            ForecastTask(
                history=group.reset_index(drop=True),
                horizon=2,
                model_config={
                    "backend": "statsforecast",
                    "model": "SeasonalNaive",
                    "season_length": 2,
                },
                task_group="category_1" if grouped else None,
            )
        )
    return partition_tasks(tasks)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([UNIQUE_ID, FORECAST_ORIGIN, H]).reset_index(drop=True)


def test_group_scheduling_preserves_results() -> None:
    panel = _panel()
    origins = [pd.Timestamp("2024-03-03")]

    ungrouped = BackendEngine().execute(_tasks(panel, grouped=False), panel, origins)
    grouped = BackendEngine().execute(_tasks(panel, grouped=True), panel, origins)

    pd.testing.assert_frame_equal(
        _sorted(grouped.ledger.to_df()),
        _sorted(ungrouped.ledger.to_df()),
    )


def test_backend_consumes_partition_without_get_scope() -> None:
    """The engine reads the pre-resolved partition; it must not import or call
    ``get_scope`` (scope is resolved once, in task building)."""
    import calibre.execution.backend as backend_module

    source = (backend_module.__file__ or "").rstrip("c")
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "get_scope" not in text
    assert not hasattr(backend_module, "get_scope")


def test_local_and_global_partition_routed_separately() -> None:
    """A mixed batch partitions into the local and global groups by resolved
    scope, and the engine runs both paths from the partition."""
    panel = _panel()
    origins = [pd.Timestamp("2024-03-03")]

    local_task = ForecastTask(
        history=panel[panel[UNIQUE_ID] == "A"].reset_index(drop=True),
        horizon=2,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 2},
    )
    global_task = ForecastTask(
        history=panel,
        horizon=2,
        model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "season_length": 2,
            "scope": "global",
        },
    )
    groups = partition_tasks([local_task, global_task])
    assert groups.local == [local_task]
    assert groups.global_ == [global_task]

    result = BackendEngine().execute(groups, panel, origins)
    ledger = result.ledger.to_df()
    assert not ledger.empty
    # Local path forecasts series A; global path forecasts the full panel.
    assert set(ledger[UNIQUE_ID].unique()) == {"A", "B"}


# ---------------------------------------------------------------------------
# Chunked-local dispatch (U1 #162): grouping, chunk-count math, and the
# staging-local chunk identity must never leak into the ledger.
# ---------------------------------------------------------------------------


def _local_task(uid: str, *, season_length: int = 2) -> ForecastTask:
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    return ForecastTask(
        history=pd.DataFrame({UNIQUE_ID: uid, DS: dates, Y: [10.0, 20.0] * 6}),
        horizon=2,
        model_config={
            "backend": "statsforecast",
            "model": "SeasonalNaive",
            "season_length": season_length,
        },
    )


def test_distinct_configs_never_share_a_chunk() -> None:
    """Tasks with different resolved configs land in separate config groups."""
    tasks = [
        _local_task("A", season_length=2),
        _local_task("B", season_length=4),
        _local_task("C", season_length=2),
    ]
    groups = _group_local_tasks_by_config(tasks)

    assert len(groups) == 2
    members = {tuple(t.unique_id for t in group_tasks) for _config, group_tasks in groups}
    assert members == {("A", "C"), ("B",)}


def test_duplicate_uid_config_tasks_collapse_to_first() -> None:
    """Duplicate (uid, config, horizon) tasks dedup before chunk staging.

    Without the dedup, both copies' histories are concatenated into the staged
    chunk panel, and the worker's per-uid slice reads one series with every row
    doubled — silent forecast corruption.
    """
    first = _local_task("A")
    groups = _group_local_tasks_by_config([first, _local_task("B"), _local_task("A")])

    assert len(groups) == 1
    _config, group_tasks = groups[0]
    assert [task.unique_id for task in group_tasks] == ["A", "B"]
    assert group_tasks[0] is first


def test_same_config_different_horizons_group_separately() -> None:
    """Horizon is part of the chunk group key; the staged chunk applies one
    horizon to every member, so mixed horizons must never co-chunk."""
    groups = _group_local_tasks_by_config([_local_task("A"), replace(_local_task("B"), horizon=3)])

    assert len(groups) == 2
    horizons = {group_tasks[0].horizon for _config, group_tasks in groups}
    assert horizons == {2, 3}


def test_duplicate_local_task_yields_same_ledger_as_single_copy() -> None:
    """End-to-end lock for the doubled-history corruption: a duplicated
    (uid, config) task produces exactly the ledger of a single copy."""
    tasks = [_local_task("A"), _local_task("B")]
    panel = pd.concat([task.history for task in tasks], ignore_index=True)
    origins = [pd.Timestamp("2024-03-03")]

    single = BackendEngine(execution=ExecutionOptions(chunk_size=256))
    baseline = single.execute(partition_tasks(tasks), panel, origins).ledger.to_df()

    duplicated = BackendEngine(execution=ExecutionOptions(chunk_size=256))
    with_duplicate = duplicated.execute(
        partition_tasks([*tasks, _local_task("A")]), panel, origins
    ).ledger.to_df()

    pd.testing.assert_frame_equal(_sorted(with_duplicate), _sorted(baseline))


def test_chunk_count_is_ceil_of_series_over_chunk_size() -> None:
    """A same-config group of N series stages ceil(N / chunk_size) chunks."""
    engine = BackendEngine(execution=ExecutionOptions(chunk_size=2))
    tasks = [_local_task(f"S{i}") for i in range(5)]
    refs = engine._materialize_local_chunks(tasks, "memory://chunk-count-test/local")

    assert len(refs) == math.ceil(5 / 2)
    # Every series appears exactly once across the chunks.
    staged_uids = [uid for ref in refs for uid in ref.unique_ids]
    assert sorted(staged_uids) == ["S0", "S1", "S2", "S3", "S4"]


def test_chunk_size_one_makes_one_chunk_per_series() -> None:
    engine = BackendEngine(execution=ExecutionOptions(chunk_size=1))
    tasks = [_local_task("A"), _local_task("B"), _local_task("C")]
    refs = engine._materialize_local_chunks(tasks, "memory://chunk-one-test/local")

    assert len(refs) == 3
    assert all(len(ref.unique_ids) == 1 for ref in refs)


def test_chunked_local_ledger_uid_set_equals_real_series() -> None:
    """The synthetic chunk identity never reaches the ledger; real uids do.

    A single chunk holds three series (chunk_size large); the ledger must carry
    the three real per-series uids, not any chunk-level id.
    """
    panel = pd.concat(
        [_local_task(uid).history for uid in ("A", "B", "C")],
        ignore_index=True,
    )
    tasks = [_local_task(uid) for uid in ("A", "B", "C")]
    origins = [pd.Timestamp("2024-03-03")]

    engine = BackendEngine(execution=ExecutionOptions(chunk_size=256))
    ledger = engine.execute(partition_tasks(tasks), panel, origins).ledger.to_df()

    assert set(ledger[UNIQUE_ID].unique()) == {"A", "B", "C"}


def test_chunked_local_mixed_future_x_presence_within_a_chunk(monkeypatch) -> None:
    """A chunk where only some series carry future_x fits each series correctly.

    The chunk worker re-slices future_x per uid and passes None when a series has
    no rows — mirroring the per-series behavior, so mixed presence is harmless.
    The stub adapter records exactly what each series received: A gets its single
    promo row (sliced to A), B gets None.
    """
    from calibre.forecasting.adapter_base import ModelAdapter

    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 3
    future_x = pd.DataFrame({UNIQUE_ID: ["A"], DS: [pd.Timestamp("2024-04-07")], "promo": [1.0]})
    panel = pd.concat(
        [
            pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: pattern}),
            pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: pattern}),
        ],
        ignore_index=True,
    )
    tasks = [
        ForecastTask(
            history=pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: pattern}),
            horizon=1,
            model_config={"backend": "stub", "model": "stub_model"},
            future_x=future_x,
        ),
        ForecastTask(
            history=pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: pattern}),
            horizon=1,
            model_config={"backend": "stub", "model": "stub_model"},
        ),
    ]
    origins = [pd.Timestamp("2024-03-24")]

    received: dict[str, pd.DataFrame | None] = {}

    class _StubAdapter(ModelAdapter):
        def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
            del collect_fitted_values
            received[task.unique_id] = task.future_x

        def predict(self, task: ForecastTask) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    UNIQUE_ID: [task.unique_id],
                    DS: [pd.Timestamp("2024-04-07")],
                    "y_hat": [10.0],
                    H: [1],
                }
            )

    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

    engine = BackendEngine(execution=ExecutionOptions(chunk_size=256))
    ledger = engine.execute(partition_tasks(tasks), panel, origins).ledger.to_df()

    assert set(ledger[UNIQUE_ID].unique()) == {"A", "B"}
    # A received its single promo row sliced to A; B carried no future_x rows → None.
    assert received["A"] is not None
    assert set(received["A"][UNIQUE_ID].unique()) == {"A"}
    assert received["B"] is None
