"""Exercise deterministic lifecycle aggregation and memory evidence."""

from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pandas as pd
import pytest

from newcalibre.benchmarking import (
    LifecycleCollector,
    MemoryMonitor,
    MemorySample,
    ProcessResidentSample,
    ProfileError,
    aggregate_profile,
)
from newcalibre.engine import Phase, PhaseEvent, PhaseStatus


class Clock:
    """Return deterministic injected monotonic values."""

    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _events(origin: pd.Timestamp) -> tuple[PhaseEvent, ...]:
    return tuple(
        PhaseEvent(phase, origin, status)
        for phase in Phase
        for status in (PhaseStatus.STARTED, PhaseStatus.FINISHED)
    )


def _memory() -> tuple[MemorySample, ...]:
    processes = (
        ProcessResidentSample(10, "driver", 100),
        ProcessResidentSample(11, "ray-control", 200),
        ProcessResidentSample(12, "object-store", 300),
        ProcessResidentSample(13, "worker", 400),
    )
    return (
        MemorySample(processes, peak_job_memory_bytes=800, cgroup_identity="/gate-c"),
        MemorySample(processes, peak_job_memory_bytes=900, cgroup_identity="/gate-c"),
    )


def _profile(*, hidden_gap: bool = False) -> dict[str, object]:
    origin = pd.Timestamp("2026-01-01")
    timestamps = [float(value) for index in range(7) for value in (index + 1, index + 2)]
    if hidden_gap:
        timestamps[4:] = [value + 2.0 for value in timestamps[4:]]
    collector = LifecycleCollector(clock=Clock(*timestamps))
    for event in _events(origin):
        collector(event)
    return aggregate_profile(
        attempt_id="attempt-a",
        expected_origins=(origin,),
        wall_start=0.0,
        wall_end=9.0 + (2.0 if hidden_gap else 0.0),
        lifecycle=collector.records,
        memory_samples=_memory(),
        sampling_interval_seconds=0.1,
        dispatch={
            "logical_shards": 16,
            "numeric_threads_per_worker": 1,
            "origin_count": 1,
            "shard_dispatch_count": 16,
            "workers": 16,
        },
        scaling=(
            (1000, 16, 10.0, 900, 16),
            (10000, 16, 20.0, 1800, 16),
            (30490, 16, 30.0, 2700, 16),
        ),
        concurrency=((1, 4.0, 16), (16, 1.0, 16)),
    )


def test_aggregate_profile_reconciles_disjoint_lifecycle_and_memory() -> None:
    """Aggregate exact phases and independent process/job peaks."""
    profile = _profile()
    assert profile["valid"] is True
    assert profile["timing"]["reconciliation_percent"] == 100.0  # type: ignore[index]
    assert profile["memory"]["peak_process_resident_bytes"] == 1000  # type: ignore[index]
    assert profile["memory"]["peak_job_memory_bytes"] == 900  # type: ignore[index]


def test_hidden_span_below_reconciliation_threshold_invalidates_attempt() -> None:
    """Invalidate unattributed wall time instead of passing a budget verdict."""
    profile = _profile(hidden_gap=True)
    assert profile["valid"] is False
    assert profile["budgets"]["reconciliation_percent"]["passed"] is False  # type: ignore[index]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "failed"])
def test_lifecycle_validation_rejects_malformed_sequences(mutation: str) -> None:
    """Reject incomplete, duplicate, foreign, or failed lifecycle streams."""
    origin = pd.Timestamp("2026-01-01")
    events = list(_events(origin))
    if mutation == "missing":
        events.pop()
    elif mutation == "duplicate":
        events.insert(1, events[0])
    elif mutation == "foreign":
        events[0] = replace(events[0], origin=pd.Timestamp("2026-01-02"))
    else:
        events[1] = replace(events[1], status=PhaseStatus.FAILED)
    collector = LifecycleCollector(clock=Clock(*range(1, len(events) + 1)))
    for event in events:
        collector(event)
    with pytest.raises(ProfileError):
        aggregate_profile(
            attempt_id="attempt-a",
            expected_origins=(origin,),
            wall_start=0.0,
            wall_end=20.0,
            lifecycle=collector.records,
            memory_samples=_memory(),
            sampling_interval_seconds=0.1,
            dispatch={
                "logical_shards": 16,
                "numeric_threads_per_worker": 1,
                "origin_count": 1,
                "shard_dispatch_count": 16,
                "workers": 16,
            },
            scaling=((1000, 16, 1.0, 1, 16), (10000, 16, 1.0, 1, 16), (30490, 16, 1.0, 1, 16)),
            concurrency=((1, 1.0, 16), (16, 1.0, 16)),
        )


@pytest.mark.parametrize(
    "samples",
    [
        (
            MemorySample(
                (ProcessResidentSample(10, "driver", 100),),
                peak_job_memory_bytes=100,
                cgroup_identity="/gate-c",
            ),
        ),
        (
            MemorySample(
                _memory()[0].processes,
                peak_job_memory_bytes=100,
                cgroup_identity="/gate-c",
                complete=False,
            ),
        ),
    ],
)
def test_memory_inventory_faults_invalidate_attempt(samples: tuple[MemorySample, ...]) -> None:
    """Invalidate missing roles and incomplete memory sampling."""
    origin = pd.Timestamp("2026-01-01")
    collector = LifecycleCollector(clock=Clock(*range(1, 15)))
    for event in _events(origin):
        collector(event)
    profile = aggregate_profile(
        attempt_id="attempt-a",
        expected_origins=(origin,),
        wall_start=0.0,
        wall_end=15.0,
        lifecycle=collector.records,
        memory_samples=samples,
        sampling_interval_seconds=0.1,
        dispatch={
            "logical_shards": 16,
            "numeric_threads_per_worker": 1,
            "origin_count": 1,
            "shard_dispatch_count": 16,
            "workers": 16,
        },
        scaling=((1000, 16, 1.0, 1, 16), (10000, 16, 1.0, 1, 16), (30490, 16, 1.0, 1, 16)),
        concurrency=((1, 1.0, 16), (16, 1.0, 16)),
    )
    assert profile["valid"] is False


def test_memory_monitor_marks_a_vanished_required_process() -> None:
    """Invalidate a process inventory that disappears while sampling is active."""
    completed = Event()

    class Reader:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self) -> MemorySample:
            self.calls += 1
            processes = _memory()[0].processes
            if self.calls > 1:
                processes = processes[:-1]
                completed.set()
            return MemorySample(
                processes,
                peak_job_memory_bytes=900,
                cgroup_identity="/gate-c",
            )

    monitor = MemoryMonitor(Reader(), interval_seconds=0.001)
    monitor.start()
    assert completed.wait(timeout=1.0)
    monitor.stop()

    assert monitor.failure is None
    assert monitor.samples[-1].vanished_pids == (13,)


def test_memory_monitor_records_sampler_failure() -> None:
    """Retain sampler failure as invalidating attempt evidence."""

    class Reader:
        def sample(self) -> MemorySample:
            raise RuntimeError("fixture sampler failed")

    monitor = MemoryMonitor(Reader(), interval_seconds=0.1)
    monitor.start()
    monitor.stop()

    assert monitor.samples == ()
    assert monitor.failure == "fixture sampler failed"


def test_concurrency_runner_terminates_work_at_the_hard_timeout() -> None:
    """Terminate the Fit/Predict child instead of waiting past 60 seconds."""
    script = Path(__file__).parents[2] / "scripts" / "m5_benchmark.py"
    bounded = runpy.run_path(str(script))["run_bounded_concurrency"]

    class Process:
        def __init__(self) -> None:
            self.alive = True
            self.terminated = False

        def start(self) -> None:
            pass

        def join(self, _timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

    process = Process()

    class Context:
        def Queue(self) -> object:
            return object()

        def Process(self, **_kwargs: object) -> Process:
            return process

    bounded.__globals__["multiprocessing"] = SimpleNamespace(get_context=lambda _kind: Context())

    with pytest.raises(ProfileError, match="exceeded 60 seconds"):
        bounded(Path("profile.yaml"), concurrency=1)

    assert process.terminated
