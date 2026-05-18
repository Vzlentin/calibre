from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, ExecutionOptions

pytestmark = pytest.mark.spark


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
    return [
        ForecastTask(
            history=group.reset_index(drop=True),
            horizon=2,
            model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "season_length": 2,
            },
        )
        for _, group in panel.groupby(UNIQUE_ID, sort=False)
    ]


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([UNIQUE_ID, FORECAST_ORIGIN, H]).reset_index(drop=True)


def test_spark_local_matches_single_node_backend() -> None:
    if os.environ.get("JAVA_HOME") is None and shutil.which("java") is None:
        pytest.skip("Spark integration requires Java")
    pyspark_sql = pytest.importorskip("pyspark.sql")
    fugue_spark = pytest.importorskip("fugue_spark")

    panel = _panel()
    tasks = _tasks(panel)
    origins = [pd.Timestamp("2024-02-11"), pd.Timestamp("2024-02-18")]
    expected = BackendEngine().execute(tasks, panel, origins).ledger.to_df()

    spark = (
        pyspark_sql.SparkSession.builder.master("local[1]")
        .appName("calibre-test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        engine = fugue_spark.SparkExecutionEngine(spark)
        actual = (
            BackendEngine(execution=ExecutionOptions(engine=engine))
            .execute(tasks, panel, origins)
            .ledger.to_df()
        )
    finally:
        spark.stop()

    pd.testing.assert_frame_equal(_sorted(actual), _sorted(expected))
