import pandas as pd
import pytest

from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def history():
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-07", periods=10, freq="W"),
            "y": range(10),
        }
    )


def test_create_task(history):
    task = ForecastTask(
        unique_id="SKU_001",
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
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive"},
    )
    with pytest.raises(AttributeError):
        task.unique_id = "other"


def test_model_name_from_model_key(history):
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "season_length": 4},
    )
    assert task.model_name == "SeasonalNaive"


def test_model_name_from_name_key(history):
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "name": "SN_52", "season_length": 52},
    )
    assert task.model_name == "SN_52"


def test_with_forecast_origin(history):
    origin = pd.Timestamp("2024-03-01")
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive"},
        forecast_origin=origin,
    )
    assert task.forecast_origin == origin
