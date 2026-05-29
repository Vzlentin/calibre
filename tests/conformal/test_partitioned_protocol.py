"""Tests for the PartitionedConformalRuntime protocol (roadmap P1.5).

The backend's _persist_conformal_state previously duck-typed
``getattr(runtime, "get_partition_states", None)``; it now gates on
``isinstance(runtime, PartitionedConformalRuntime)``. These tests pin the
behaviour that gate depends on: the real runtime must be recognized (else
persistence silently falls back to a single blob), and an arbitrary object
must not be.
"""

from __future__ import annotations

from calibre.conformal.runtime import (
    PartitionedConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)


def _runtime() -> SymmetricIntervalRuntime:
    return SymmetricIntervalRuntime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )


def test_symmetric_interval_runtime_is_partitioned():
    runtime = _runtime()
    assert isinstance(runtime, PartitionedConformalRuntime)
    # The gate must resolve to the real method, not just structural truthiness.
    assert runtime.get_partition_states() == {}


def test_object_without_get_partition_states_is_not_partitioned():
    class _NotPartitioned:
        def get_resume_state(self) -> dict:
            return {}

    assert not isinstance(_NotPartitioned(), PartitionedConformalRuntime)
