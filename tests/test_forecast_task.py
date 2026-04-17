import pandas as pd
import pytest

from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def history():
    return pd.DataFrame(
        {
            "unique_id": "SKU_001",
            "ds": pd.date_range("2024-01-07", periods=10, freq="W"),
            "y": range(10),
        }
    )


def test_create_task(history):
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "season_length": 4},
    )
    assert task.unique_id == "SKU_001"
    assert task.horizon == 4
    assert task.forecast_origin is None
    assert task.future_x is None


def test_frozen(history):
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive"},
    )
    with pytest.raises((AttributeError, TypeError)):
        task.horizon = 99  # type: ignore[misc]


def test_model_name_from_model_key(history):
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "season_length": 4},
    )
    assert task.model_name == "SeasonalNaive"


def test_model_name_from_name_key(history):
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "name": "SN_52", "season_length": 52},
    )
    assert task.model_name == "SN_52"


def test_with_forecast_origin(history):
    origin = pd.Timestamp("2024-03-01")
    task = ForecastTask(
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive"},
        forecast_origin=origin,
    )
    assert task.forecast_origin == origin


def test_history_must_have_unique_id_column():
    history_without_uid = pd.DataFrame(
        {"ds": pd.date_range("2024-01-07", periods=5, freq="W"), "y": range(5)}
    )
    with pytest.raises(ValueError, match="unique_id"):
        ForecastTask(history=history_without_uid, horizon=2, model_config={"model": "X"})
