from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from calibre.cli.commands import run
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, BackendResult


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


def _global_task(panel: pd.DataFrame) -> ForecastTask:
    return ForecastTask(
        history=panel,
        horizon=2,
        model_config={
            "backend": "mlforecast",
            "scope": "global",
            "model": "lightgbm.LGBMRegressor",
            "lags": [1, 2, 3, 4],
            "verbosity": -1,
            "n_estimators": 10,
        },
    )


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


def test_dask_global_scope_matches_single_node_backend() -> None:
    distributed = pytest.importorskip("distributed")
    fugue_dask = pytest.importorskip("fugue_dask")

    panel = _panel()
    task = _global_task(panel)
    origins = [pd.Timestamp("2024-02-11")]
    expected = BackendEngine(freq="W").execute([task], panel, origins).ledger.to_df()

    cluster = distributed.LocalCluster(processes=False, dashboard_address=None)
    client = distributed.Client(cluster)
    try:
        engine = fugue_dask.DaskExecutionEngine(client)
        actual = (
            BackendEngine(freq="W", engine=engine).execute([task], panel, origins).ledger.to_df()
        )
    finally:
        client.close()
        cluster.close()

    pd.testing.assert_frame_equal(_sorted(actual), _sorted(expected))


def _write_cli_config(tmp_path: Path, *, engine: str | None) -> Path:
    config_path = tmp_path / f"{engine or 'local'}.yaml"
    ledger_path = tmp_path / f"{engine or 'local'}.parquet"
    engine_value = "null" if engine is None else engine
    config_path.write_text(
        f"""
config_schema: "1.0"
dataset:
  adapter: vn2
  path: benchmarks/vn2/fixture
  period: 0
tasks:
  - model: SeasonalNaive
    horizon: 2
    config:
      backend: statsforecast
      season_length: 2
origins:
  start: 2024-01-29
  end: 2024-01-29
  freq: W-MON
output:
  ledger_path: {ledger_path.as_posix()}
  streaming: false
execution:
  engine: {engine_value}
  seed: 42
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_dask_config_matches_cli_single_node(tmp_path: Path) -> None:
    pytest.importorskip("distributed")
    pytest.importorskip("fugue_dask")

    local = run(_write_cli_config(tmp_path, engine=None))
    dask = run(_write_cli_config(tmp_path, engine="dask"))

    assert isinstance(local, BackendResult)
    assert isinstance(dask, BackendResult)
    pd.testing.assert_frame_equal(_sorted(dask.ledger.to_df()), _sorted(local.ledger.to_df()))
