"""Exercise the adapter protocol, capability vocabulary, and backend registry.

All assertions in this module are tolerance-class-1 structural or rejection
facts; no numeric comparand is involved.
"""

from __future__ import annotations

import pytest

from newcalibre.forecasting import (
    SEASONAL_NAIVE_BACKEND,
    AdapterCapability,
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


def test_seasonal_naive_exposes_the_full_protocol_and_declares_no_optional_capability() -> None:
    adapter = resolve_adapter(_config())

    assert isinstance(adapter, ForecastAdapter)
    assert adapter.capabilities == frozenset()
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


@pytest.mark.parametrize("season_length", [None, 0, -1, True, 1.5])
def test_seasonal_naive_requires_a_positive_integer_season_length(
    season_length: object,
) -> None:
    with pytest.raises(AdapterConfigurationError, match="positive integer"):
        resolve_adapter(_config(m=season_length))
