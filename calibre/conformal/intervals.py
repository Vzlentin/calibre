"""Constructors for symmetric interval predictions around a point forecast."""

from __future__ import annotations

from typing import Any

from calibre.conformal.types import IntervalPrediction


def symmetric_interval(
    center: float,
    radius: float,
    alpha: float,
    issued_at: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> IntervalPrediction:
    """Build a single ``[center - radius, center + radius]`` interval."""
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
