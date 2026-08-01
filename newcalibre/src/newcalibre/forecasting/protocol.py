"""Define the forecasting-adapter protocol and its capability vocabulary."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import pandas as pd

from newcalibre.domain import FittedValues, ForecastTask, HistoryDelta


class AdapterCapability(StrEnum):
    """Name an optional operation or fit behavior an adapter may declare."""

    FITTED_VALUES = "fitted_values"
    NATIVE_QUANTILES = "native_quantiles"
    CENSORING_AWARE_FIT = "censoring_aware_fit"
    INCREMENTAL_UPDATE = "incremental_update"
    ARTIFACT_PERSISTENCE = "artifact_persistence"


class AdapterExecutionMode(StrEnum):
    """Declare whether one semantic task may be split across series."""

    SERIES_SEPARABLE = "series-separable"
    MONOLITHIC = "monolithic"


class AdapterError(Exception):
    """Report a forecasting-adapter contract failure."""


class AdapterConfigurationError(AdapterError):
    """Report an invalid model configuration."""


class AdapterCapabilityError(AdapterError):
    """Report a request for a capability the adapter does not declare."""


class AdapterLifecycleError(AdapterError):
    """Report an adapter operation invoked before successful fitting."""


class AdapterDataError(AdapterError):
    """Report task data that cannot support the requested forecast."""


@runtime_checkable
class ForecastAdapter(Protocol):
    """Expose the complete engine-facing forecasting-adapter surface.

    Fit retention follows one rule across implementations: retain exactly the
    minimal predictive state that ``predict`` needs, and document that
    concrete state on each adapter. Local state may be per-series; global
    state may be panel-level. Whole tasks, unused history, fitted values, and
    forecast rows are not retained merely for convenience.
    """

    @property
    def execution_mode(self) -> AdapterExecutionMode:
        """Return the required physical execution declaration."""
        ...

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        """Return the optional capabilities this adapter declares."""
        ...

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        """Return capabilities requested by the construction configuration."""
        ...

    def fit(self, task: ForecastTask) -> None:
        """Fit and retain only the adapter's documented minimal predictive state."""
        ...

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        """Emit a validated forecast frame for one forecast task."""
        ...

    def fitted_values(self) -> FittedValues:
        """Emit the optional fitted-values side channel."""
        ...

    def dump_state(self) -> bytes:
        """Serialize fitted predictive state through a native artifact API."""
        ...

    def load_state(self, state: bytes) -> None:
        """Restore fitted predictive state through a native artifact API."""
        ...

    def update(self, delta: HistoryDelta) -> None:
        """Extend fitted state from only newly admissible history."""
        ...
