from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    return pd.concat(
        [
            pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [10.0, 20.0] * 6}),
            pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: [5.0, 15.0] * 6}),
        ],
        ignore_index=True,
    )


def _tasks(panel: pd.DataFrame, *, grouped: bool) -> list[ForecastTask]:
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
    return tasks


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
