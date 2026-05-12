"""Experimental low-level conformal controllers.

These exports are algorithm-level building blocks. They are not yet the stable
pipeline-facing conformal interface for the repository. Stable runtime-facing
configuration helpers now live alongside the low-level controllers so the
engine can adopt them incrementally.
"""

from calibre.conformal.aci import (
    AdaptiveConformalInference,
    MultiStepAdaptiveConformalInference,
)
from calibre.conformal.crc import (
    CumulativeConformalRiskConfig,
    CumulativeConformalRiskRuntime,
)
from calibre.conformal.intervals import symmetric_interval, symmetric_intervals
from calibre.conformal.mscp import (
    CumulativeSplitConformalInference,
    MultiStepSplitConformalInference,
)
from calibre.conformal.partitions import (
    category_partition,
    global_partition,
    regime_partition,
    series_partition,
)
from calibre.conformal.policies import OnlineConformalController
from calibre.conformal.runtime import (
    ConformalPolicyConfig,
    ConformalRuntime,
    ConformalRuntimeLike,
    deserialize_calibration_state,
    serialize_calibration_state,
)
from calibre.conformal.scores import absolute_error, scaled_absolute_error
from calibre.conformal.types import IntervalPrediction, MultiStepIntervalPrediction

__all__ = [
    "AdaptiveConformalInference",
    "ConformalPolicyConfig",
    "ConformalRuntime",
    "ConformalRuntimeLike",
    "CumulativeConformalRiskConfig",
    "CumulativeConformalRiskRuntime",
    "CumulativeSplitConformalInference",
    "IntervalPrediction",
    "MultiStepAdaptiveConformalInference",
    "MultiStepSplitConformalInference",
    "MultiStepIntervalPrediction",
    "OnlineConformalController",
    "absolute_error",
    "category_partition",
    "deserialize_calibration_state",
    "global_partition",
    "regime_partition",
    "scaled_absolute_error",
    "serialize_calibration_state",
    "series_partition",
    "symmetric_interval",
    "symmetric_intervals",
]
