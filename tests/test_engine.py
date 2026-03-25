import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    UNIQUE_ID,
    DS,
    Y,
    Y_HAT,
    H,
    FORECAST_ORIGIN,
    MODEL_NAME,
)
from calibre.engine.backend import BackendEngine
from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def single_series_setup(dates, repeating_pattern):
    """Single series, single model, two origins."""
    actuals = pd.DataFrame(
        {
            "unique_id": "SKU_001",
            "ds": dates,
            "y": repeating_pattern,
        }
    )

    task = ForecastTask(
        unique_id="SKU_001",
        history=pd.DataFrame({"ds": dates, "y": repeating_pattern}),
        horizon=4,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )

    origins = [dates[7], dates[11]]

    return task, actuals, origins


def test_execute_returns_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    assert len(df) == 8


def test_forecast_frame_columns_present(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    for col in [UNIQUE_ID, DS, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]:
        assert col in df.columns


def test_partial_resolution(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()

    resolved = df[Y].notna().sum()
    unresolved = df[Y].isna().sum()
    assert resolved == 5
    assert unresolved == 3


def test_error_columns_on_resolved(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    resolved = df[df[Y].notna()]

    assert "error" in resolved.columns
    assert "abs_error" in resolved.columns
    np.testing.assert_array_almost_equal(resolved["error"].dropna().values, 0.0)


def test_model_name_stamped(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    assert (df[MODEL_NAME] == "SeasonalNaive").all()


def test_multi_series():
    """Two series, single model, single origin."""
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern_a = [10.0, 20.0, 30.0, 40.0] * 5
    pattern_b = [5.0, 15.0, 25.0, 35.0] * 5

    actuals = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern_b}),
        ],
        ignore_index=True,
    )

    tasks = [
        ForecastTask(
            unique_id="A",
            history=pd.DataFrame({"ds": dates, "y": pattern_a}),
            horizon=4,
            model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
        ),
        ForecastTask(
            unique_id="B",
            history=pd.DataFrame({"ds": dates, "y": pattern_b}),
            horizon=4,
            model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
        ),
    ]

    engine = BackendEngine(freq="W")
    ledger = engine.execute(tasks, actuals, origins=[dates[11]])

    df = ledger.to_df()
    assert len(df) == 8
    assert set(df[UNIQUE_ID].unique()) == {"A", "B"}


def test_to_parquet_roundtrip(single_series_setup, tmp_path):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    path = str(tmp_path / "backtest.parquet")
    ledger.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 8
