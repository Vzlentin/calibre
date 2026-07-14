"""Register VN2's local native-median seasonal-naive variant."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final, Never

import pandas as pd

from newcalibre.domain import POINT_FORECAST, ForecastTask, quantile_column, validate_forecast_frame
from newcalibre.forecasting import (
    SEASONAL_NAIVE_BACKEND,
    AdapterCapability,
    AdapterCapabilityError,
    AdapterConfigurationError,
    AdapterRegistry,
    ForecastAdapter,
    SeasonalNaiveAdapter,
)
from newcalibre.protocols.vn2._constants import VN2_SEASONAL_NAIVE_BACKEND

_MEDIAN_LEVEL: Final = 0.5


@dataclass(frozen=True, slots=True)
class _EffectiveConfig:
    season_length: int
    model_name: str
    requested_capabilities: frozenset[AdapterCapability]


class VN2SeasonalNaiveQuantileAdapter(SeasonalNaiveAdapter):
    """Add the lawful native 0.5 quantile only inside the VN2 harness.

    The underlying lookup and fit retention remain the chapter-60 brick's.
    This subclass changes neither that brick's global registration nor its
    empty capability declaration.
    """

    def __init__(self, model_config: Mapping[str, object]) -> None:
        self._vn2_config = self._effective_vn2_config(model_config)
        super().__init__(
            {
                "backend": SEASONAL_NAIVE_BACKEND,
                "m": self._vn2_config.season_length,
                "model_name": self._vn2_config.model_name,
            }
        )

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        """Declare exactly the native-quantile capability."""
        return frozenset({AdapterCapability.NATIVE_QUANTILES})

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        """Return validated requests before any fit is attempted."""
        return self._vn2_config.requested_capabilities

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        """Emit the seasonal point and its byte-equal canonical median column."""
        frame = super().predict(task)
        median = quantile_column(_MEDIAN_LEVEL)
        frame[median] = frame[POINT_FORECAST].to_numpy(copy=True)
        return validate_forecast_frame(frame, calendar=task.calendar)

    def _require_matching_config(self, task: ForecastTask) -> None:
        if self._effective_vn2_config(task.model_config) != self._vn2_config:
            raise AdapterConfigurationError(
                "task model configuration must match adapter construction configuration"
            )

    def _raise_unsupported(self, capability: AdapterCapability) -> Never:
        raise AdapterCapabilityError(
            f"backend {VN2_SEASONAL_NAIVE_BACKEND!r} does not declare "
            f"capability {capability.value!r}"
        )

    @staticmethod
    def _effective_vn2_config(model_config: Mapping[str, object]) -> _EffectiveConfig:
        if not isinstance(model_config, Mapping):
            raise AdapterConfigurationError("model configuration must be a mapping")
        backend = model_config.get("backend")
        if backend != VN2_SEASONAL_NAIVE_BACKEND:
            raise AdapterConfigurationError(
                f"model configuration backend must be {VN2_SEASONAL_NAIVE_BACKEND!r}"
            )
        season_length = model_config.get("m")
        if (
            isinstance(season_length, bool)
            or not isinstance(season_length, Integral)
            or season_length < 1
        ):
            raise AdapterConfigurationError("season length 'm' must be a positive integer")
        model_name = model_config.get("model_name", VN2_SEASONAL_NAIVE_BACKEND)
        if not isinstance(model_name, str) or not model_name:
            raise AdapterConfigurationError("model name must be a non-empty string")
        raw_levels = model_config.get("quantile_levels")
        if (
            isinstance(raw_levels, (str, bytes))
            or not isinstance(raw_levels, Sequence)
            or len(raw_levels) != 1
        ):
            raise AdapterConfigurationError(
                "VN2 seasonal native quantiles require the single 0.5 level"
            )
        level = raw_levels[0]
        if (
            isinstance(level, bool)
            or not isinstance(level, Real)
            or not math.isfinite(float(level))
            or float(level) != _MEDIAN_LEVEL
        ):
            raise AdapterConfigurationError(
                "VN2 seasonal native quantiles require the single 0.5 level"
            )
        censoring_aware = model_config.get("censoring_aware", False)
        if not isinstance(censoring_aware, bool):
            raise AdapterConfigurationError("censoring_aware must be a boolean")
        requested = {AdapterCapability.NATIVE_QUANTILES}
        if censoring_aware:
            requested.add(AdapterCapability.CENSORING_AWARE_FIT)
        return _EffectiveConfig(
            season_length=int(season_length),
            model_name=model_name,
            requested_capabilities=frozenset(requested),
        )


_VN2_ADAPTERS = AdapterRegistry()
_VN2_ADAPTERS.register(VN2_SEASONAL_NAIVE_BACKEND, VN2SeasonalNaiveQuantileAdapter)


def available_vn2_backends() -> tuple[str, ...]:
    """Return the deterministic VN2-local registry view."""
    return _VN2_ADAPTERS.available_backends


def resolve_vn2_adapter(model_config: Mapping[str, object]) -> ForecastAdapter:
    """Resolve and capability-check one VN2-local adapter before execution."""
    return _VN2_ADAPTERS.resolve(model_config)
