import pytest

from calibre.models.registry import resolve_adapter


def test_resolve_statsforecast_backend():
    adapter = resolve_adapter({"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4})
    assert type(adapter).__name__ == "StatsForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_neuralforecast_backend():
    adapter = resolve_adapter({"backend": "neuralforecast", "model": "NHITS"})
    assert type(adapter).__name__ == "NeuralForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_mlforecast_backend():
    adapter = resolve_adapter({"backend": "mlforecast", "model": "lightgbm.LGBMRegressor"})
    assert type(adapter).__name__ == "MLForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_adapter({"backend": "unknown", "model": "SomeModel"})


def test_resolve_missing_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        resolve_adapter({"model": "SeasonalNaive"})
