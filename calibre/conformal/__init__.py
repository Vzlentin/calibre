"""Conformal calibration building blocks and runtime interface.

Exposes the low-level conformal primitives (scores, intervals, partitions,
spread, coherent draws) together with the runtime-facing entry points
re-exported from :mod:`calibre.conformal.runtime` (:class:`ConformalRuntime`,
:func:`build_symmetric_interval_runtime`) and the cumulative-risk decision
runtime in :mod:`calibre.conformal.cumulative_risk`.
"""

from calibre.conformal.coherent_draws import CoherentDraws
from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
from calibre.conformal.intervals import symmetric_interval
from calibre.conformal.partitions import (
    category_partition,
    global_partition,
    regime_partition,
    series_partition,
)
from calibre.conformal.runtime import (
    ConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
    build_symmetric_interval_runtime,
)
from calibre.conformal.scores import (
    AbsoluteErrorScore,
    absolute_error,
    absolute_error_score,
    scaled_absolute_error,
)
from calibre.conformal.spread import AnalyticRadius
from calibre.conformal.types import IntervalPrediction

__all__ = [
    "AnalyticRadius",
    "CoherentDraws",
    "ConformalRuntime",
    "CumulativeConformalRiskConfig",
    "CumulativeRiskRuntime",
    "IntervalPrediction",
    "AbsoluteErrorScore",
    "absolute_error",
    "absolute_error_score",
    "build_symmetric_interval_runtime",
    "category_partition",
    "global_partition",
    "regime_partition",
    "scaled_absolute_error",
    "series_partition",
    "SymmetricIntervalConfig",
    "SymmetricIntervalRuntime",
    "symmetric_interval",
]
