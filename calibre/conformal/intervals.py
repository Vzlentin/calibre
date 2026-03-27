from __future__ import annotations

from typing import Any

import numpy as np

from calibre.conformal.types import IntervalPrediction, MultiStepIntervalPrediction


def symmetric_interval(
    center: float,
    radius: float,
    alpha: float,
    issued_at: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> IntervalPrediction:
    center = float(center)
    radius = float(radius)
    return IntervalPrediction(
        center=center,
        lower=center - radius,
        upper=center + radius,
        radius=radius,
        alpha=float(alpha),
        issued_at=issued_at,
        metadata=metadata or {},
    )


def symmetric_intervals(
    center,
    radius,
    alpha,
    issued_at: int,
    metadata: dict[str, Any] | None = None,
) -> MultiStepIntervalPrediction:
    center = np.asarray(center, dtype=float)
    radius = np.asarray(radius, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if center.ndim != 1:
        raise ValueError("center must be a 1D array")
    if radius.shape != center.shape:
        raise ValueError("radius must match the shape of center")
    if alpha.shape != center.shape:
        raise ValueError("alpha must match the shape of center")
    return MultiStepIntervalPrediction(
        center=center,
        lower=center - radius,
        upper=center + radius,
        radius=radius,
        alpha=alpha,
        issued_at=issued_at,
        metadata=metadata or {},
    )
