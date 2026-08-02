"""Aggregate and publish harness-owned standard profiling evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

from newcalibre.benchmarking.environment import (
    REQUIRED_PROCESS_ROLES,
    EnvironmentError,
    MemorySample,
    validate_environment,
)
from newcalibre.engine import RAY_WORKER_THREAD_POLICY, Phase, PhaseEvent, PhaseStatus

_PHASES = tuple(Phase)
_JOB_MEMORY_LIMIT_BYTES = 32_000_000_000


class ProfileError(ValueError):
    """Report invalid lifecycle, measurement, or profile artifact evidence."""


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    """Bind one engine lifecycle event to a harness-owned monotonic time."""

    event: PhaseEvent
    timestamp: float


class LifecycleCollector:
    """Timestamp engine lifecycle events without exposing a clock to the engine."""

    def __init__(self, *, clock: Callable[[], float]) -> None:
        if not callable(clock):
            raise TypeError("lifecycle collector clock must be callable")
        self._clock = clock
        self._records: list[LifecycleRecord] = []
        self._failure: str | None = None

    @property
    def records(self) -> tuple[LifecycleRecord, ...]:
        """Return the immutable event record accumulated so far."""
        return tuple(self._records)

    @property
    def failure(self) -> str | None:
        """Return the reporter failure that invalidates this attempt, if any."""
        return self._failure

    def __call__(self, event: PhaseEvent) -> None:
        """Timestamp one engine event and remember any reporter failure."""
        try:
            if not isinstance(event, PhaseEvent):
                raise TypeError("lifecycle reporter requires a PhaseEvent")
            timestamp = _finite_nonnegative(self._clock(), name="lifecycle timestamp")
            self._records.append(LifecycleRecord(event, timestamp))
        except Exception as error:
            self._failure = str(error)
            raise


def aggregate_profile(
    *,
    attempt_id: str,
    expected_origins: Sequence[pd.Timestamp],
    wall_start: float,
    wall_end: float,
    lifecycle: Sequence[LifecycleRecord],
    memory_samples: Sequence[MemorySample],
    sampling_interval_seconds: float,
    dispatch: Mapping[str, object],
    scaling: Sequence[tuple[int, int, float, int, int]],
    concurrency: Sequence[tuple[int, float, int, Mapping[str, str]]],
    reporter_failure: str | None = None,
    sampler_failure: str | None = None,
) -> dict[str, object]:
    """Aggregate ordinary attempt facts and recompute every budget verdict."""
    _text(attempt_id, name="attempt identity")
    start = _finite_nonnegative(wall_start, name="wall start")
    end = _finite_nonnegative(wall_end, name="wall end")
    if end < start:
        raise ProfileError("wall end precedes wall start")
    origins = tuple(expected_origins)
    if not origins or len(set(origins)) != len(origins):
        raise ProfileError("expected origins must be non-empty and unique")
    if any(not isinstance(origin, pd.Timestamp) for origin in origins):
        raise ProfileError("expected origins must contain pandas timestamps")
    records = tuple(lifecycle)
    timings = _aggregate_lifecycle(origins, records, wall_start=start, wall_end=end)
    memory, memory_reasons = _aggregate_memory(
        tuple(memory_samples), interval=sampling_interval_seconds
    )
    dispatch_payload = _validate_dispatch(dict(dispatch), expected_origin_count=len(origins))
    scaling_payload = _scaling_payload(tuple(scaling))
    concurrency_payload = _concurrency_payload(tuple(concurrency))
    reasons = list(memory_reasons)
    if reporter_failure:
        reasons.append(f"lifecycle reporter failed: {reporter_failure}")
    if sampler_failure:
        reasons.append(f"memory sampler failed: {sampler_failure}")
    budgets = _budgets(timings=timings, memory=memory)
    if not budgets["reconciliation_percent"]["passed"]:
        reasons.append("wall-time reconciliation is below 99 percent")
    valid = not reasons
    if not valid:
        for budget in budgets.values():
            budget["passed"] = False
    return {
        "schema": "calibre-performance-profile",
        "schema_version": 1,
        "attempt_id": attempt_id,
        "valid": valid,
        "invalid_reasons": sorted(set(reasons), key=str.encode),
        "timing": timings,
        "memory": memory,
        "dispatch": dispatch_payload,
        "scaling": scaling_payload,
        "concurrency": concurrency_payload,
        "budgets": budgets,
    }


def validate_profile(
    value: object,
    *,
    environment: object,
) -> dict[str, object]:
    """Validate an exact profile and recompute all derived fields and verdicts."""
    try:
        environment_payload = validate_environment(environment)
    except EnvironmentError as error:
        raise ProfileError(str(error)) from error
    root = _mapping(
        value,
        keys={
            "schema",
            "schema_version",
            "attempt_id",
            "valid",
            "invalid_reasons",
            "timing",
            "memory",
            "dispatch",
            "scaling",
            "concurrency",
            "budgets",
        },
        name="profile",
    )
    if root["schema"] != "calibre-performance-profile" or root["schema_version"] != 1:
        raise ProfileError("profile schema is unsupported")
    attempt_id = _text(root["attempt_id"], name="profile attempt identity")
    if attempt_id != environment_payload["attempt_id"]:
        raise ProfileError("profile and environment attempt identities differ")
    if not isinstance(root["valid"], bool):
        raise ProfileError("profile validity must be boolean")
    reasons = root["invalid_reasons"]
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise ProfileError("profile invalid reasons must be non-empty strings")
    if bool(reasons) == root["valid"]:
        raise ProfileError("profile validity and invalid reasons disagree")
    timing = _validate_timing(root["timing"])
    memory = _validate_memory(root["memory"])
    timing_origins = timing["origins"]
    assert isinstance(timing_origins, list)
    dispatch = _validate_dispatch(
        _mapping(
            root["dispatch"],
            keys={
                "logical_shards",
                "numeric_threads_per_worker",
                "origin_count",
                "shard_dispatch_count",
                "workers",
            },
            name="profile dispatch",
        ),
        expected_origin_count=len(timing_origins),
    )
    _validate_environment_binding(environment_payload, memory=memory, dispatch=dispatch)
    scaling = _validate_scaling_payload(root["scaling"])
    full = scaling[-1]
    if not math.isclose(
        cast(float, full["wall_seconds"]),
        cast(float, timing["wall_seconds"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProfileError("full scaling wall duration differs from primary timing")
    if full["peak_job_memory_bytes"] != memory["peak_job_memory_bytes"]:
        raise ProfileError("full scaling memory peak differs from primary memory")
    if full["dispatch_count"] != dispatch["shard_dispatch_count"]:
        raise ProfileError("full scaling dispatch count differs from primary dispatch")
    _validate_concurrency_payload(root["concurrency"])
    sampler = cast(dict[str, object], environment_payload["sampler"])
    expected_valid = not reasons and memory["complete"] is True and sampler["complete"] is True
    if root["valid"] is not expected_valid:
        raise ProfileError("profile validity disagrees with completeness facts")
    expected_budgets = _budgets(timings=timing, memory=memory)
    if reasons:
        for budget in expected_budgets.values():
            budget["passed"] = False
    budgets = root["budgets"]
    if budgets != expected_budgets:
        raise ProfileError("profile budget verdicts do not match ordinary measurements")
    return root


def _aggregate_lifecycle(
    origins: tuple[pd.Timestamp, ...],
    records: tuple[LifecycleRecord, ...],
    *,
    wall_start: float,
    wall_end: float,
) -> dict[str, object]:
    expected = tuple(
        (origin, phase, status)
        for origin in origins
        for phase in _PHASES
        for status in (PhaseStatus.STARTED, PhaseStatus.FINISHED)
    )
    actual = tuple(
        (record.event.origin, record.event.phase, record.event.status) for record in records
    )
    if actual != expected:
        raise ProfileError("lifecycle events are incomplete, failed, foreign, or out of order")
    timestamps = tuple(record.timestamp for record in records)
    if any(
        not math.isfinite(value) or value < wall_start or value > wall_end for value in timestamps
    ):
        raise ProfileError("lifecycle timestamps fall outside the attempt wall span")
    if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:], strict=False)):
        raise ProfileError("lifecycle timestamps must be monotonic")
    origin_payloads: list[dict[str, object]] = []
    durations: list[float] = []
    offset = 0
    for origin in origins:
        stages: dict[str, float] = {}
        for phase in _PHASES:
            started = records[offset].timestamp
            finished = records[offset + 1].timestamp
            duration = finished - started
            stages[phase.value] = duration
            durations.append(duration)
            offset += 2
        origin_payloads.append({"origin": origin.isoformat(), "stages": stages})
    pre_origin = records[0].timestamp - wall_start
    close = wall_end - records[-1].timestamp
    attributed = math.fsum((pre_origin, close, *durations))
    wall = wall_end - wall_start
    if wall <= 0.0:
        raise ProfileError("attempt wall duration must be positive")
    reconciliation = 100.0 * attributed / wall
    return {
        "wall_seconds": wall,
        "pre_origin_seconds": pre_origin,
        "close_seconds": close,
        "origins": origin_payloads,
        "attributed_seconds": attributed,
        "reconciliation_percent": reconciliation,
    }


def _aggregate_memory(
    samples: tuple[MemorySample, ...], *, interval: float
) -> tuple[dict[str, object], tuple[str, ...]]:
    sampling_interval = _positive_number(interval, name="sampling interval")
    reasons: list[str] = []
    if not samples:
        raise ProfileError("memory sampling produced no observations")
    peaks: dict[tuple[int, str], int] = {}
    identities = {sample.cgroup_identity for sample in samples}
    if len(identities) != 1:
        reasons.append("cgroup identity changed during sampling")
    for sample in samples:
        if not sample.complete:
            reasons.append("memory sampling was incomplete")
        if sample.vanished_pids:
            reasons.append("a required process vanished during sampling")
        for process in sample.processes:
            key = (process.pid, process.role)
            peaks[key] = max(peaks.get(key, 0), process.resident_bytes)
    roles = {role for _pid, role in peaks}
    missing = REQUIRED_PROCESS_ROLES - roles
    if missing:
        reasons.append("missing required process roles: " + ", ".join(sorted(missing)))
    processes = [
        {"pid": pid, "role": role, "peak_process_resident_bytes": peak}
        for (pid, role), peak in sorted(
            peaks.items(), key=lambda item: (item[0][1].encode(), item[0][0])
        )
    ]
    return (
        {
            "sampling_interval_seconds": sampling_interval,
            "complete": not reasons,
            "processes": processes,
            "peak_process_resident_bytes": sum(peaks.values()),
            "peak_job_memory_bytes": max(sample.peak_job_memory_bytes for sample in samples),
            "cgroup_identity": next(iter(identities)) if len(identities) == 1 else "invalid",
        },
        tuple(reasons),
    )


def _scaling_payload(
    values: tuple[tuple[int, int, float, int, int], ...],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = [
        {
            "series_count": series_count,
            "workers": workers,
            "wall_seconds": wall_seconds,
            "peak_job_memory_bytes": peak_job_memory_bytes,
            "dispatch_count": dispatch_count,
        }
        for series_count, workers, wall_seconds, peak_job_memory_bytes, dispatch_count in values
    ]
    _validate_scaling_payload(payload)
    return payload


def _concurrency_payload(
    values: tuple[tuple[int, float, int, Mapping[str, str]], ...],
) -> dict[str, object]:
    payload = {
        "series_count": 1000,
        "origin_count": 1,
        "logical_shards": 16,
        "phases": ["Fit", "Predict"],
        "timeout_seconds": 60.0,
        "runs": [
            {
                "concurrency": concurrency,
                "wall_seconds": wall,
                "dispatch_count": count,
                "thread_policy": dict(policy),
            }
            for concurrency, wall, count, policy in values
        ],
    }
    _validate_concurrency_payload(payload)
    return payload


def _validate_timing(value: object) -> dict[str, object]:
    timing = _mapping(
        value,
        keys={
            "wall_seconds",
            "pre_origin_seconds",
            "close_seconds",
            "origins",
            "attributed_seconds",
            "reconciliation_percent",
        },
        name="profile timing",
    )
    wall = _positive_number(timing["wall_seconds"], name="wall duration")
    pre = _finite_nonnegative(timing["pre_origin_seconds"], name="pre-origin duration")
    close = _finite_nonnegative(timing["close_seconds"], name="close duration")
    origins = timing["origins"]
    if not isinstance(origins, list) or not origins:
        raise ProfileError("profile timing requires origins")
    durations: list[float] = []
    identities: list[str] = []
    for origin_value in origins:
        origin = _mapping(origin_value, keys={"origin", "stages"}, name="origin timing")
        identities.append(_text(origin["origin"], name="timing origin"))
        stages = _mapping(
            origin["stages"], keys={phase.value for phase in _PHASES}, name="origin stages"
        )
        durations.extend(
            _finite_nonnegative(stages[phase.value], name=f"{phase.value} duration")
            for phase in _PHASES
        )
    if len(set(identities)) != len(identities):
        raise ProfileError("profile timing origins must be unique")
    attributed = math.fsum((pre, close, *durations))
    if not math.isclose(
        _finite_nonnegative(timing["attributed_seconds"], name="attributed duration"),
        attributed,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProfileError("profile attributed duration does not match stage facts")
    reconciliation = 100.0 * attributed / wall
    if not math.isclose(
        _finite_nonnegative(timing["reconciliation_percent"], name="reconciliation percent"),
        reconciliation,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ProfileError("profile reconciliation does not match timing facts")
    return timing


def _validate_memory(value: object) -> dict[str, object]:
    memory = _mapping(
        value,
        keys={
            "sampling_interval_seconds",
            "complete",
            "processes",
            "peak_process_resident_bytes",
            "peak_job_memory_bytes",
            "cgroup_identity",
        },
        name="profile memory",
    )
    _positive_number(memory["sampling_interval_seconds"], name="sampling interval")
    if not isinstance(memory["complete"], bool):
        raise ProfileError("memory completeness must be boolean")
    processes = memory["processes"]
    if not isinstance(processes, list) or not processes:
        raise ProfileError("profile memory requires process peaks")
    peaks: list[int] = []
    identities: list[tuple[int, str]] = []
    roles: set[str] = set()
    for process_value in processes:
        process = _mapping(
            process_value,
            keys={"pid", "role", "peak_process_resident_bytes"},
            name="process peak",
        )
        pid = _positive_integer(process["pid"], name="process pid")
        role = _text(process["role"], name="process role")
        if role not in REQUIRED_PROCESS_ROLES:
            raise ProfileError("profile contains an unknown process role")
        peak = _nonnegative_integer(
            process["peak_process_resident_bytes"], name="process resident peak"
        )
        identities.append((pid, role))
        roles.add(role)
        peaks.append(peak)
    if len(set(identities)) != len(identities):
        raise ProfileError("profile process peaks contain duplicate identities")
    if memory["complete"] and roles != REQUIRED_PROCESS_ROLES:
        raise ProfileError("complete profile memory is missing required process roles")
    if _nonnegative_integer(
        memory["peak_process_resident_bytes"], name="summed process resident peak"
    ) != sum(peaks):
        raise ProfileError("summed process resident peak does not match process facts")
    _nonnegative_integer(memory["peak_job_memory_bytes"], name="job memory peak")
    identity = _text(memory["cgroup_identity"], name="profile cgroup identity")
    if not identity.startswith("/"):
        raise ProfileError("profile cgroup identity must be absolute")
    return memory


def _validate_dispatch(
    value: dict[str, object], *, expected_origin_count: int
) -> dict[str, object]:
    required = {
        "logical_shards",
        "numeric_threads_per_worker",
        "origin_count",
        "shard_dispatch_count",
        "workers",
    }
    if set(value) != required:
        raise ProfileError("profile dispatch must contain exact fields")
    if _positive_integer(value["logical_shards"], name="logical shards") != 16:
        raise ProfileError("profile requires 16 logical shards")
    if _positive_integer(value["workers"], name="workers") != 16:
        raise ProfileError("profile requires 16 workers")
    if (
        _positive_integer(value["numeric_threads_per_worker"], name="numeric threads per worker")
        != 1
    ):
        raise ProfileError("profile requires one numeric thread per worker")
    if _positive_integer(value["origin_count"], name="origin count") != expected_origin_count:
        raise ProfileError("dispatch origin count does not match timing origins")
    if _positive_integer(value["shard_dispatch_count"], name="dispatch count") < 16:
        raise ProfileError("profile dispatch count is incomplete")
    return value


def _validate_scaling_payload(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProfileError("scaling profile requires exactly three population points")
    expected_counts = (1000, 10000, 30490)
    points: list[dict[str, object]] = []
    for point_value, expected_count in zip(value, expected_counts, strict=True):
        point = _mapping(
            point_value,
            keys={
                "series_count",
                "workers",
                "wall_seconds",
                "peak_job_memory_bytes",
                "dispatch_count",
            },
            name="scaling point",
        )
        if _positive_integer(point["series_count"], name="scaling series count") != expected_count:
            raise ProfileError("scaling populations must be nested 1k, 10k, and full M5")
        if _positive_integer(point["workers"], name="scaling workers") != 16:
            raise ProfileError("every ordinary scaling point requires 16 workers")
        _positive_number(point["wall_seconds"], name="scaling wall duration")
        _nonnegative_integer(point["peak_job_memory_bytes"], name="scaling job memory peak")
        _positive_integer(point["dispatch_count"], name="scaling dispatch count")
        points.append(point)
    return points


def _validate_concurrency_payload(value: object) -> None:
    payload = _mapping(
        value,
        keys={
            "series_count",
            "origin_count",
            "logical_shards",
            "phases",
            "timeout_seconds",
            "runs",
        },
        name="concurrency profile",
    )
    if payload["series_count"] != 1000 or payload["origin_count"] != 1:
        raise ProfileError("concurrency profile requires 1,000 series and one origin")
    if payload["logical_shards"] != 16 or payload["phases"] != ["Fit", "Predict"]:
        raise ProfileError("concurrency profile is not the fixed Fit/Predict fan-out")
    if payload["timeout_seconds"] != 60.0:
        raise ProfileError("concurrency profile requires a hard 60-second cap")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != 2:
        raise ProfileError("concurrency profile requires exactly two runs")
    for run_value, expected in zip(runs, (1, 16), strict=True):
        run = _mapping(
            run_value,
            keys={"concurrency", "wall_seconds", "dispatch_count", "thread_policy"},
            name="concurrency run",
        )
        if _positive_integer(run["concurrency"], name="concurrency") != expected:
            raise ProfileError("concurrency runs must compare one with 16")
        if _positive_number(run["wall_seconds"], name="concurrency wall duration") > 60.0:
            raise ProfileError("concurrency run exceeded its hard timeout")
        if _positive_integer(run["dispatch_count"], name="concurrency dispatch count") != 16:
            raise ProfileError("concurrency run must dispatch all 16 logical shards")
        policy = run["thread_policy"]
        if policy != dict(RAY_WORKER_THREAD_POLICY):
            raise ProfileError("concurrency run did not observe the one-thread policy")


def _budgets(
    *, timings: dict[str, object], memory: dict[str, object]
) -> dict[str, dict[str, object]]:
    wall = _positive_number(timings["wall_seconds"], name="wall duration")
    pre = _finite_nonnegative(timings["pre_origin_seconds"], name="pre-origin duration")
    job = _nonnegative_integer(memory["peak_job_memory_bytes"], name="job memory peak")
    reconciliation = _finite_nonnegative(
        timings["reconciliation_percent"], name="reconciliation percent"
    )
    return {
        "wall_seconds": {"limit": 900.0, "passed": wall <= 900.0},
        "pre_origin_seconds": {"limit": 60.0, "passed": pre <= 60.0},
        "peak_job_memory_bytes": {
            "limit": _JOB_MEMORY_LIMIT_BYTES,
            "passed": job <= _JOB_MEMORY_LIMIT_BYTES,
        },
        "reconciliation_percent": {"minimum": 99.0, "passed": reconciliation >= 99.0},
    }


def _validate_environment_binding(
    environment: dict[str, object],
    *,
    memory: dict[str, object],
    dispatch: dict[str, object],
) -> None:
    cgroup = cast(dict[str, object], environment["cgroup"])
    if cgroup["identity"] != memory["cgroup_identity"]:
        raise ProfileError("profile and environment cgroup identities differ")
    execution = cast(dict[str, object], environment["execution"])
    for key in ("logical_shards", "numeric_threads_per_worker", "workers"):
        if execution[key] != dispatch[key]:
            raise ProfileError("profile and environment execution facts differ")
    sampler = cast(dict[str, object], environment["sampler"])
    if sampler["complete"] != memory["complete"]:
        raise ProfileError("profile and environment sampler completeness differs")
    if sampler["interval_seconds"] != memory["sampling_interval_seconds"]:
        raise ProfileError("profile and environment sampling interval differs")


def _mapping(value: object, *, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProfileError(f"{name} must contain exact fields")
    if any(not isinstance(key, str) for key in value):
        raise ProfileError(f"{name} fields must be strings")
    return cast(dict[str, object], value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{name} must be non-empty text")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProfileError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProfileError(f"{name} must be a non-negative integer")
    return value


def _positive_number(value: object, *, name: str) -> float:
    number = _finite_nonnegative(value, name=name)
    if number <= 0.0:
        raise ProfileError(f"{name} must be positive")
    return number


def _finite_nonnegative(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProfileError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ProfileError(f"{name} must be finite and non-negative")
    return number


__all__ = [
    "LifecycleCollector",
    "LifecycleRecord",
    "ProfileError",
    "aggregate_profile",
    "validate_profile",
]
