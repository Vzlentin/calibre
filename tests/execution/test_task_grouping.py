from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y
from calibre.core.forecast_task import ForecastTask, TaskGroups
from calibre.execution.backend import BackendEngine
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
