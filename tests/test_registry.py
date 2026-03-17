import pytest

from calibre.models.registry import resolve_adapter


def test_resolve_seasonal_naive():
    adapter = resolve_adapter({"model": "SeasonalNaive", "season_length": 4})
    assert type(adapter).__name__ == "StatsForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_auto_arima():
    adapter = resolve_adapter({"model": "AutoARIMA"})
    assert type(adapter).__name__ == "StatsForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_adapter({"model": "NonExistentModel"})


def test_resolve_auto_ets():
    adapter = resolve_adapter({"model": "AutoETS", "season_length": 4})
    assert type(adapter).__name__ == "StatsForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_mfles():
    adapter = resolve_adapter({"model": "MFLES"})
    assert type(adapter).__name__ == "StatsForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_lightgbm():
    adapter = resolve_adapter({"model": "LightGBM"})
    assert type(adapter).__name__ == "MLForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_xgboost():
    adapter = resolve_adapter({"model": "XGBoost"})
    assert type(adapter).__name__ == "MLForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_nhits():
    adapter = resolve_adapter({"model": "NHiTS"})
    assert type(adapter).__name__ == "NeuralForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_tide():
    adapter = resolve_adapter({"model": "TiDE"})
    assert type(adapter).__name__ == "NeuralForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_patchtst():
    adapter = resolve_adapter({"model": "PatchTST"})
    assert type(adapter).__name__ == "NeuralForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")
