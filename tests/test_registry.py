import pytest

from calibre.models.registry import get_adapter_cls, resolve_adapter


def test_resolve_statsforecast_backend():
    adapter = resolve_adapter(
        {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4}
    )
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


def test_resolve_mlforecast_global_backend():
    adapter = resolve_adapter({"backend": "mlforecast_global", "model": "lightgbm.LGBMRegressor"})
    assert type(adapter).__name__ == "MLForecastGlobalAdapter"
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_adapter({"backend": "unknown", "model": "SomeModel"})


def test_resolve_missing_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        resolve_adapter({"model": "SeasonalNaive"})


def test_local_adapters_are_parallel_by_uid():
    for backend in ("statsforecast", "mlforecast", "neuralforecast"):
        cls = get_adapter_cls({"backend": backend, "model": "X"})
        assert cls.PARALLEL_BY_UID is True, f"{backend} should have PARALLEL_BY_UID=True"


def test_global_adapter_is_not_parallel_by_uid():
    cls = get_adapter_cls({"backend": "mlforecast_global", "model": "lightgbm.LGBMRegressor"})
    assert cls.PARALLEL_BY_UID is False
