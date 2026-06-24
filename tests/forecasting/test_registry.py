"""Tests for the forecasting-adapter registry."""

import importlib

import pytest

from calibre.forecasting import adapter_registry
from calibre.forecasting.adapter_registry import get_scope, resolve_adapter


def test_resolve_statsforecast_backend():
    adapter = resolve_adapter(
        {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4}
    )
    assert type(adapter).__name__ == "StatsForecastAdapter"


def test_resolve_statsforecast_backend_does_not_import_other_adapters(monkeypatch):
    monkeypatch.setattr(adapter_registry, "_REGISTRY", {})
    real_import_module = importlib.import_module
    imported: list[str] = []

    def tracking_import_module(name: str, package: str | None = None):
        imported.append(name)
        return real_import_module(name, package)

    monkeypatch.setattr(adapter_registry.importlib, "import_module", tracking_import_module)

    adapter = resolve_adapter(
        {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4}
    )

    assert type(adapter).__name__ == "StatsForecastAdapter"
    assert "calibre.forecasting.statsforecast_adapter" in imported
    assert "calibre.forecasting.mlforecast_adapter" not in imported
    assert "calibre.forecasting.neuralforecast_adapter" not in imported


def test_resolve_neuralforecast_backend():
    pytest.importorskip("neuralforecast")
    adapter = resolve_adapter({"backend": "neuralforecast", "model": "NHITS"})
    assert type(adapter).__name__ == "NeuralForecastAdapter"


@pytest.mark.parametrize("scope", [None, "global"])
def test_resolve_mlforecast_backend(scope):
    config = {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor"}
    if scope is not None:
        config["scope"] = scope
    adapter = resolve_adapter(config)
    assert type(adapter).__name__ == "MLForecastAdapter"


def test_resolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_adapter({"backend": "unknown", "model": "SomeModel"})


def test_resolve_missing_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        resolve_adapter({"model": "SeasonalNaive"})


def test_censoring_fit_on_non_mlforecast_backend_raises():
    # The censoring-aware fit gate is implemented only by the mlforecast adapter;
    # enabling it on another backend fails fast rather than crashing the estimator.
    with pytest.raises(ValueError, match="censoring_fit is only supported by the mlforecast"):
        resolve_adapter(
            {"backend": "statsforecast", "model": "SeasonalNaive", "censoring_fit": True}
        )


def test_censoring_fit_on_mlforecast_backend_is_allowed():
    resolve_adapter(
        {"backend": "mlforecast", "model": "lightgbm.LGBMRegressor", "censoring_fit": True}
    )


def test_get_scope_defaults_to_local():
    assert get_scope({"backend": "mlforecast", "model": "X"}) == "local"


def test_get_scope_accepts_global():
    assert get_scope({"backend": "mlforecast", "model": "X", "scope": "global"}) == "global"


def test_get_scope_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown scope"):
        get_scope({"backend": "mlforecast", "model": "X", "scope": "galactic"})
