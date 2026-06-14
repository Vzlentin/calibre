"""Tests for the PartitionedConformalRuntime protocol.

The backend's _persist_conformal_state previously duck-typed
``getattr(runtime, "get_partition_states", None)``; it now gates on
``isinstance(runtime, PartitionedConformalRuntime)``. These tests pin the
behaviour that gate depends on: the real runtime must be recognized (else
persistence silently falls back to a single blob), and an arbitrary object
must not be.
"""

from __future__ import annotations

import pandas as pd

from calibre.conformal.runtime import (
    PartitionedConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)


def _config() -> SymmetricIntervalConfig:
    return SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)


def _runtime() -> SymmetricIntervalRuntime:
    return SymmetricIntervalRuntime(_config())


def _two_horizon_window() -> pd.DataFrame:
    lower, upper = interval_column_names(0.9)
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            MODEL_NAME: ["stub", "stub"],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-07")] * 2,
            DS: pd.to_datetime(["2024-01-14", "2024-01-21"]),
            H: [1, 2],
            Y: [2.0, 3.0],
            Y_HAT: [2.5, 5.0],
            lower: [1.0, 1.0],
            upper: [4.0, 9.0],
        }
    )


def test_symmetric_interval_runtime_is_partitioned():
    runtime = _runtime()
    assert isinstance(runtime, PartitionedConformalRuntime)
    # The gate must resolve to the real method, not just structural truthiness;
    # a fresh runtime has no partitions, so the backend falls back to one blob.
    assert runtime.get_partition_states() == {}


def test_object_without_get_partition_states_is_not_partitioned():
    class _NotPartitioned:
        def get_resume_state(self) -> dict:
            return {}

    assert not isinstance(_NotPartitioned(), PartitionedConformalRuntime)


def test_get_partition_states_splits_observed_history_by_partition():
    runtime = _runtime()
    runtime.observe(_two_horizon_window())

    states = runtime.get_partition_states()

    # One persistable state per (model, horizon) partition, each carrying only
    # its own calibration scores (|y - y_hat| per horizon).
    assert set(states) == {"stub:h1:__global__", "stub:h2:__global__"}
    for key, state in states.items():
        assert state["partition"] == key
        assert set(state["calibrator"]["score_history"]) == {key}
    assert states["stub:h1:__global__"]["calibrator"]["score_history"][
        "stub:h1:__global__"
    ].tolist() == [0.5]
    assert states["stub:h2:__global__"]["calibrator"]["score_history"][
        "stub:h2:__global__"
    ].tolist() == [2.0]


def test_partition_states_round_trip_through_factory():
    runtime = _runtime()
    runtime.observe(_two_horizon_window())
    states = runtime.get_partition_states()

    rebuilt = SymmetricIntervalRuntime.from_partition_states(_config(), states)

    rebuilt_states = rebuilt.get_partition_states()
    assert set(rebuilt_states) == set(states)
    for key in states:
        assert (
            rebuilt_states[key]["calibrator"]["score_history"][key].tolist()
            == states[key]["calibrator"]["score_history"][key].tolist()
        )
