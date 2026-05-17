from __future__ import annotations

import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    return pd.concat(
        [
            pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [10.0, 20.0] * 10}),
            pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: [5.0, 15.0] * 10}),
        ],
        ignore_index=True,
    )


def test_dask_quantile_columns_survive() -> None:
    """Quantile columns from local models must survive Fugue+Dask transform."""
    distributed = pytest.importorskip("distributed")
    fugue_dask = pytest.importorskip("fugue_dask")

    all_series = _panel()
    model_config = {
        "backend": "mlforecast",
        "scope": "local",
        "model": "lightgbm.LGBMRegressor",
        "objective": "quantile",
        "quantiles": [0.5, 0.833],
        "strategy": "direct",
        "lags": [1, 2],
        "verbosity": -1,
        "n_estimators": 10,
    }
    tasks = [
        ForecastTask(
            history=group.reset_index(drop=True),
            horizon=2,
            model_config=model_config,
        )
        for _, group in all_series.groupby(UNIQUE_ID, sort=False)
    ]

    origins = [pd.Timestamp("2024-04-14")]

    expected = BackendEngine(freq="W").execute(tasks, all_series, origins).ledger.to_df()
    assert "q_0p5" in expected.columns
    assert "q_0p833" in expected.columns

    cluster = distributed.LocalCluster(processes=False, dashboard_address=None)
    client = distributed.Client(cluster)
    try:
        engine = fugue_dask.DaskExecutionEngine(client)
        actual = (
            BackendEngine(freq="W", engine=engine)
            .execute(tasks, all_series, origins)
            .ledger.to_df()
        )
    finally:
        client.close()
        cluster.close()

    assert "q_0p5" in actual.columns
    assert "q_0p833" in actual.columns
    assert actual["q_0p5"].notna().all()
    assert actual["q_0p833"].notna().all()
