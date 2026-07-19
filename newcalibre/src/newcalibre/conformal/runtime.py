"""Define the stable three-verb conformal runtime protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel

from newcalibre.conformal.manifest import MethodManifest
from newcalibre.conformal.types import (
    CalibrationContext,
    CalibrationResult,
    Delivery,
    ObserveEffect,
    RuntimeContractError,
)


@runtime_checkable
class ConformalRuntime(Protocol):
    """Calibrate, apply, and observe through opaque independently stored state."""

    @property
    def manifest(self) -> MethodManifest:
        """Return this runtime's immutable method declaration."""
        ...

    @property
    def config(self) -> BaseModel:
        """Return the frozen validated runtime configuration."""
        ...

    def calibrate(
        self,
        scores: Mapping[str, Sequence[float]],
    ) -> Mapping[str, bytes]:
        """Deterministically seed independently addressable state rows."""
        ...

    def apply(
        self,
        forecasts: pd.DataFrame,
        states: Mapping[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        """Issue calibrated forecasts without consuming actual values."""
        ...

    def observe(
        self,
        delivery: Delivery,
        states: Mapping[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        """Consume one partition delivery in its supplied order."""
        ...


def require_calibration_context(
    manifest: MethodManifest,
    context: CalibrationContext | None,
    *,
    series_keys: Sequence[str],
) -> None:
    """Enforce manifest-declared context presence and exact row alignment."""
    if not isinstance(manifest, MethodManifest):
        raise RuntimeContractError("runtime manifest must be a MethodManifest")
    keys = tuple(series_keys)
    if manifest.consumes_calibration_context:
        if context is None:
            raise RuntimeContractError("calibration context is required by the manifest")
        if not isinstance(context, CalibrationContext):
            raise RuntimeContractError("calibration context must be a CalibrationContext")
        if context.series_keys != keys:
            raise RuntimeContractError("calibration context has incorrect row alignment")
    elif context is not None:
        raise RuntimeContractError("calibration context must be absent for this method")
