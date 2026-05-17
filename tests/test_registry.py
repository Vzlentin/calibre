import pytest

from calibre.forecasting.adapter_registry import get_scope, resolve_adapter


def test_resolve_statsforecast_backend():
    adapter = resolve_adapter(
        {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4}
    )
    assert type(adapter).__name__ == "StatsForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_neuralforecast_backend():
    pytest.importorskip("neuralforecast")
    adapter = resolve_adapter({"backend": "neuralforecast", "model": "NHITS"})
    assert type(adapter).__name__ == "NeuralForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_mlforecast_backend():
    adapter = resolve_adapter({"backend": "mlforecast", "model": "lightgbm.LGBMRegressor"})
    assert type(adapter).__name__ == "MLForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_mlforecast_global_scope():
    adapter = resolve_adapter(
        {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "scope": "global"}
    )
    assert type(adapter).__name__ == "MLForecastAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_adapter({"backend": "unknown", "model": "SomeModel"})


def test_resolve_missing_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        resolve_adapter({"model": "SeasonalNaive"})


def test_get_scope_defaults_to_local():
    assert get_scope({"backend": "mlforecast", "model": "X"}) == "local"


def test_get_scope_accepts_global():
    assert get_scope({"backend": "mlforecast", "model": "X", "scope": "global"}) == "global"


def test_get_scope_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown scope"):
        get_scope({"backend": "mlforecast", "model": "X", "scope": "galactic"})
