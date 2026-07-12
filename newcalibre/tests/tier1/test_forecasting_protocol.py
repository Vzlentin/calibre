"""Exercise the adapter protocol, capability vocabulary, and backend registry.

All assertions in this module are tolerance-class-1 structural or rejection
facts; no numeric comparand is involved.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from newcalibre.domain import FittedValues, ForecastTask
from newcalibre.forecasting import (
    SEASONAL_NAIVE_BACKEND,
    AdapterCapability,
    AdapterCapabilityError,
    AdapterConfigurationError,
    AdapterRegistry,
    AdapterRegistryError,
    ForecastAdapter,
    SeasonalNaiveAdapter,
    available_backends,
    resolve_adapter,
)

pytestmark = pytest.mark.tier1


def _config(**overrides: object) -> dict[str, object]:
    return {"backend": SEASONAL_NAIVE_BACKEND, "m": 7, **overrides}


class _CapabilitySpyAdapter:
    def __init__(self, model_config: Mapping[str, object]) -> None:
        requested: set[AdapterCapability] = set()
        if model_config.get("native_quantiles") is True:
            requested.add(AdapterCapability.NATIVE_QUANTILES)
        if model_config.get("censoring_aware") is True:
            requested.add(AdapterCapability.CENSORING_AWARE_FIT)
        self._requested_capabilities = frozenset(requested)
        self.fit_calls = 0

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return self._requested_capabilities

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del task, collect_fitted_values
        self.fit_calls += 1

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        del task
        raise AssertionError("prediction is outside this registry test")

    def fitted_values(self, task: ForecastTask) -> FittedValues:
        del task
        raise AssertionError("fitted values are outside this registry test")

    def dump_state(self) -> bytes:
        raise AssertionError("state persistence is outside this registry test")

    def load_state(self, state: bytes) -> None:
        del state
        raise AssertionError("state persistence is outside this registry test")

    def update(self, task: ForecastTask) -> None:
        del task
        raise AssertionError("incremental update is outside this registry test")


def test_seasonal_naive_exposes_the_full_protocol_and_declares_no_optional_capability() -> None:
    adapter = resolve_adapter(_config())

    assert isinstance(adapter, ForecastAdapter)
    assert adapter.capabilities == frozenset()
    assert adapter.requested_capabilities == frozenset()
    assert set(AdapterCapability) == {
        AdapterCapability.FITTED_VALUES,
        AdapterCapability.NATIVE_QUANTILES,
        AdapterCapability.CENSORING_AWARE_FIT,
        AdapterCapability.INCREMENTAL_UPDATE,
        AdapterCapability.ARTIFACT_PERSISTENCE,
    }


def test_builtin_registry_contains_the_explicit_seasonal_naive_backend() -> None:
    assert "seasonal-naive" in available_backends()
    assert isinstance(resolve_adapter(_config()), SeasonalNaiveAdapter)


def test_registry_has_no_default_backend_even_with_one_available() -> None:
    with pytest.raises(
        AdapterRegistryError,
        match=r"explicit 'backend'.*available backends: seasonal-naive",
    ):
        resolve_adapter({"m": 7})


def test_registry_rejects_unknown_backend_and_lists_available_backends() -> None:
    with pytest.raises(
        AdapterRegistryError,
        match="unknown backend 'unknown'.*available backends: seasonal-naive",
    ):
        resolve_adapter({"backend": "unknown", "m": 7})


def test_registry_rejects_duplicate_backend_identifiers() -> None:
    registry = AdapterRegistry()
    registry.register(SEASONAL_NAIVE_BACKEND, SeasonalNaiveAdapter)

    with pytest.raises(AdapterRegistryError, match="already registered"):
        registry.register(SEASONAL_NAIVE_BACKEND, SeasonalNaiveAdapter)


@pytest.mark.parametrize(
    ("config", "capability"),
    [
        (_config(quantile_levels=[0.5]), AdapterCapability.NATIVE_QUANTILES),
        (_config(censoring_aware=True), AdapterCapability.CENSORING_AWARE_FIT),
    ],
)
def test_builtin_resolution_rejects_undeclared_capabilities_before_fit(
    config: Mapping[str, object], capability: AdapterCapability
) -> None:
    with pytest.raises(AdapterCapabilityError, match=capability.value):
        resolve_adapter(config)


def test_registry_validates_extension_capabilities_without_backend_special_cases() -> None:
    registry = AdapterRegistry()
    created: list[_CapabilitySpyAdapter] = []

    def factory(model_config: Mapping[str, object]) -> ForecastAdapter:
        adapter = _CapabilitySpyAdapter(model_config)
        created.append(adapter)
        return adapter

    registry.register("extension", factory)

    with pytest.raises(
        AdapterCapabilityError,
        match="censoring_aware_fit.*native_quantiles",
    ):
        registry.resolve(
            {
                "backend": "extension",
                "native_quantiles": True,
                "censoring_aware": True,
            }
        )

    assert len(created) == 1
    assert isinstance(created[0], ForecastAdapter)
    assert created[0].fit_calls == 0


@pytest.mark.parametrize("season_length", [None, 0, -1, True, 1.5])
def test_seasonal_naive_requires_a_positive_integer_season_length(
    season_length: object,
) -> None:
    with pytest.raises(AdapterConfigurationError, match="positive integer"):
        resolve_adapter(_config(m=season_length))
