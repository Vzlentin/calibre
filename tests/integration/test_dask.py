from __future__ import annotations

import pandas as pd
import pytest

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


def _tasks(panel: pd.DataFrame) -> list[ForecastTask]:
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
            )
        )
    return tasks


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([UNIQUE_ID, FORECAST_ORIGIN, H]).reset_index(drop=True)


def test_dask_localcluster_matches_single_node_backend() -> None:
    distributed = pytest.importorskip("distributed")
    fugue_dask = pytest.importorskip("fugue_dask")

    panel = _panel()
    tasks = _tasks(panel)
    origins = [pd.Timestamp("2024-02-11"), pd.Timestamp("2024-02-18")]
    expected = BackendEngine(freq="W").execute(tasks, panel, origins).ledger.to_df()

    cluster = distributed.LocalCluster(processes=False, dashboard_address=None)
    client = distributed.Client(cluster)
    try:
        engine = fugue_dask.DaskExecutionEngine(client)
        actual = (
            BackendEngine(freq="W", engine=engine).execute(tasks, panel, origins).ledger.to_df()
        )
    finally:
        client.close()
        cluster.close()

    pd.testing.assert_frame_equal(_sorted(actual), _sorted(expected))
