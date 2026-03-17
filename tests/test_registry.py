import pytest

from calibre.models.registry import resolve_adapter


def test_resolve_seasonal_naive():
    adapter = resolve_adapter({"model": "SeasonalNaive", "season_length": 4})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_auto_arima():
    adapter = resolve_adapter({"model": "AutoARIMA"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_adapter({"model": "NonExistentModel"})


def test_resolve_auto_ets():
    adapter = resolve_adapter({"model": "AutoETS", "season_length": 4})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_mfles():
    adapter = resolve_adapter({"model": "MFLES"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_lightgbm():
    adapter = resolve_adapter({"model": "LightGBM"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_xgboost():
    adapter = resolve_adapter({"model": "XGBoost"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_nhits():
    adapter = resolve_adapter({"model": "NHiTS"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_tide():
    adapter = resolve_adapter({"model": "TiDE"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_patchtst():
    adapter = resolve_adapter({"model": "PatchTST"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")
