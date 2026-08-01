"""Define typed deterministic forecast work and serial placement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

import pandas as pd

from newcalibre.domain import (
    HORIZON_STEP,
    ORIGIN,
    SERIES_KEY,
    CycleToken,
    ForecastTask,
    SessionIdentity,
    validate_forecast_frame,
)
from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.forecasting import AdapterExecutionMode


class ForecastDispatchError(RuntimeError):
    """Report invalid logical work, placement, or result envelopes."""


@dataclass(frozen=True, slots=True)
class ForecastExecutionBudget:
    """Fix the physical execution budget for one dispatch backend."""

    logical_shards: int
    concurrency: int
    numeric_threads_per_worker: int = 1
    retries: int = 0

    def __post_init__(self) -> None:
        for name in ("logical_shards", "concurrency", "numeric_threads_per_worker"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ForecastDispatchError(f"forecast execution {name} must be positive")
        if self.concurrency > self.logical_shards:
            raise ForecastDispatchError("forecast concurrency cannot exceed logical shards")
        if self.numeric_threads_per_worker != 1:
            raise ForecastDispatchError("forecast workers require exactly one numeric thread")
        if not isinstance(self.retries, Integral) or isinstance(self.retries, bool):
            raise ForecastDispatchError("forecast retries must be an integer")
        if self.retries != 0:
            raise ForecastDispatchError("forecast dispatch requires zero retries")


@dataclass(frozen=True, slots=True)
class ForecastShard:
    """Carry one dispatch-private contiguous series slice."""

    key: str
    work_key: str
    backend: str
    ordinal: int
    session: SessionIdentity
    token: CycleToken
    semantic_task_identity: str
    task: ForecastTask
    series_keys: tuple[str, ...]
    empty: bool = False
    prior_native_state: bytes | None = None
    prior_time_bound: int | None = None
    fit_time_bound: int | None = None


@dataclass(frozen=True, slots=True)
class ForecastWork:
    """Carry one semantic task as typed, placement-ready forecast work."""

    key: str
    backend: str
    session: SessionIdentity
    token: CycleToken
    task: ForecastTask
    execution_mode: AdapterExecutionMode
    budget: ForecastExecutionBudget
    shards: tuple[ForecastShard, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.execution_mode, AdapterExecutionMode):
            raise ForecastDispatchError("forecast work requires an adapter execution mode")
        expected_count = (
            self.budget.logical_shards
            if self.execution_mode is AdapterExecutionMode.SERIES_SEPARABLE
            else 1
        )
        if len(self.shards) != expected_count:
            raise ForecastDispatchError("forecast work has the wrong logical shard count")
        if tuple(shard.ordinal for shard in self.shards) != tuple(range(expected_count)):
            raise ForecastDispatchError("forecast shard ordinals must be canonical and complete")


@dataclass(frozen=True, slots=True)
class ForecastResultEnvelope:
    """Return one unpublished shard result with attributable identity."""

    work_key: str
    shard_key: str
    backend: str
    ordinal: int
    session: SessionIdentity
    token: CycleToken
    semantic_task_identity: str
    series_keys: tuple[str, ...]
    frame: pd.DataFrame
    native_state: bytes | None
    fit_time_bound: int


@runtime_checkable
class ForecastShardExecutor(Protocol):
    """Execute one typed forecast shard without publishing its effects."""

    def run_shard(self, shard: ForecastShard) -> ForecastResultEnvelope:
        """Execute one validated shard and return an unpublished envelope."""
        ...


@runtime_checkable
class DispatchBackend(Protocol):
    """Place typed forecast work behind a complete-result barrier."""

    @property
    def backend(self) -> str:
        """Return the stable backend identity."""
        ...

    @property
    def budget(self) -> ForecastExecutionBudget:
        """Return the fixed execution budget."""
        ...

    def dispatch(
        self,
        work: ForecastWork,
        executor: ForecastShardExecutor,
    ) -> tuple[ForecastResultEnvelope, ...]:
        """Execute all logical shards and return only complete outcomes."""
        ...


class InProcessDispatch:
    """Execute canonical forecast shards serially in the current process."""

    def __init__(self, *, logical_shards: int = 1) -> None:
        self._budget = ForecastExecutionBudget(
            logical_shards=logical_shards,
            concurrency=1,
        )

    @property
    def backend(self) -> str:
        """Return the serial backend identity."""
        return "in-process"

    @property
    def budget(self) -> ForecastExecutionBudget:
        """Return the immutable serial execution budget."""
        return self._budget

    def dispatch(
        self,
        work: ForecastWork,
        executor: ForecastShardExecutor,
    ) -> tuple[ForecastResultEnvelope, ...]:
        """Execute every shard once in canonical ordinal order."""
        _require_backend_work(self, work)
        return tuple(executor.run_shard(shard) for shard in work.shards)


def canonical_shard_ranges(
    *,
    series_count: int,
    shard_count: int,
) -> tuple[tuple[int, int], ...]:
    """Partition canonical series into near-equal contiguous ranges."""
    if not isinstance(series_count, Integral) or isinstance(series_count, bool) or series_count < 1:
        raise ValueError("series count must be positive")
    if not isinstance(shard_count, Integral) or isinstance(shard_count, bool) or shard_count < 1:
        raise ValueError("shard count must be positive")
    quotient, remainder = divmod(int(series_count), int(shard_count))
    ranges: list[tuple[int, int]] = []
    start = 0
    for ordinal in range(int(shard_count)):
        stop = start + quotient + (1 if ordinal < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def build_forecast_work(
    *,
    backend: str,
    budget: ForecastExecutionBudget,
    session: SessionIdentity,
    token: CycleToken,
    task: ForecastTask,
    execution_mode: AdapterExecutionMode,
    prior_states: dict[str, tuple[bytes, int, int]] | None = None,
) -> ForecastWork:
    """Derive typed logical shards from one semantic forecast task."""
    if not isinstance(backend, str) or not backend:
        raise ForecastDispatchError("forecast backend identity must be non-empty")
    if token.session != session or token.origin != task.origin:
        raise ForecastDispatchError("forecast work token does not match its task")
    shard_count = (
        budget.logical_shards if execution_mode is AdapterExecutionMode.SERIES_SEPARABLE else 1
    )
    ranges = canonical_shard_ranges(
        series_count=len(task.series_keys),
        shard_count=shard_count,
    )
    work_key = _work_key(backend=backend, session=session, token=token, task=task)
    shards: list[ForecastShard] = []
    states = prior_states or {}
    for ordinal, (start, stop) in enumerate(ranges):
        empty = start == stop
        shard_task = task if empty else task._series_slice(start, stop)
        series_keys = task.series_keys[start:stop]
        shard_key = _shard_key(
            task=task,
            ordinal=ordinal,
            series_keys=series_keys,
            series_start=start,
            series_stop=stop,
        )
        prior = states.get(shard_key)
        shards.append(
            ForecastShard(
                key=shard_key,
                work_key=work_key,
                backend=backend,
                ordinal=ordinal,
                session=session,
                token=token,
                semantic_task_identity=task.identity,
                task=shard_task,
                series_keys=series_keys,
                empty=empty,
                prior_native_state=None if prior is None else prior[0],
                prior_time_bound=None if prior is None else prior[1],
                fit_time_bound=None if prior is None else prior[2],
            )
        )
    if states and set(states) != {shard.key for shard in shards}:
        raise ForecastDispatchError("committed shard states do not match canonical work")
    return ForecastWork(
        key=work_key,
        backend=backend,
        session=session,
        token=token,
        task=task,
        execution_mode=execution_mode,
        budget=budget,
        shards=tuple(shards),
    )


def validate_forecast_envelopes(
    work: ForecastWork,
    envelopes: tuple[ForecastResultEnvelope, ...],
) -> tuple[pd.DataFrame, tuple[ForecastResultEnvelope, ...]]:
    """Validate complete attributed results and merge in canonical row order."""
    by_ordinal: dict[int, ForecastResultEnvelope] = {}
    expected_by_ordinal = {shard.ordinal: shard for shard in work.shards}
    for envelope in envelopes:
        if not isinstance(envelope, ForecastResultEnvelope):
            raise ForecastDispatchError("forecast result must be a typed envelope")
        if envelope.ordinal in by_ordinal:
            raise ForecastDispatchError("forecast result contains a duplicate ordinal")
        shard = expected_by_ordinal.get(envelope.ordinal)
        if shard is None:
            raise ForecastDispatchError("forecast result contains a foreign ordinal")
        if (
            envelope.work_key != work.key
            or envelope.shard_key != shard.key
            or envelope.backend != work.backend
            or envelope.session != work.session
            or envelope.token != work.token
            or envelope.semantic_task_identity != work.task.identity
            or envelope.series_keys != shard.series_keys
        ):
            raise ForecastDispatchError("forecast result envelope does not match expected work")
        frame = validate_forecast_frame(envelope.frame, calendar=work.task.calendar)
        expected_rows = tuple(
            (series_key, step)
            for series_key in shard.series_keys
            for step in range(1, shard.task.horizon + 1)
        )
        actual_rows = tuple(zip(frame[SERIES_KEY], frame[HORIZON_STEP], strict=True))
        if actual_rows != expected_rows or not frame[ORIGIN].eq(work.task.origin).all():
            raise ForecastDispatchError("forecast result rows do not canonically own their shard")
        by_ordinal[envelope.ordinal] = ForecastResultEnvelope(
            work_key=envelope.work_key,
            shard_key=envelope.shard_key,
            backend=envelope.backend,
            ordinal=envelope.ordinal,
            session=envelope.session,
            token=envelope.token,
            semantic_task_identity=envelope.semantic_task_identity,
            series_keys=envelope.series_keys,
            frame=frame,
            native_state=envelope.native_state,
            fit_time_bound=envelope.fit_time_bound,
        )
    if set(by_ordinal) != set(expected_by_ordinal):
        raise ForecastDispatchError("forecast result is missing one or more logical shards")
    ordered = tuple(by_ordinal[ordinal] for ordinal in range(len(work.shards)))
    merged = pd.concat((envelope.frame for envelope in ordered), ignore_index=True)
    return validate_forecast_frame(merged, calendar=work.task.calendar), ordered


def _require_backend_work(backend: DispatchBackend, work: ForecastWork) -> None:
    if work.backend != backend.backend or work.budget != backend.budget:
        raise ForecastDispatchError("forecast work does not match its dispatch backend")


def _work_key(
    *,
    backend: str,
    session: SessionIdentity,
    token: CycleToken,
    task: ForecastTask,
) -> str:
    payload = canonical_json_bytes(
        {
            "backend": backend,
            "session": session.value,
            "task": task.identity,
            "token": {
                "attempt": token.attempt,
                "controller_nonce": token.controller_nonce,
                "revision": token.revision,
            },
        },
        path="forecast work key",
    )
    return f"forecast-work:{hashlib.sha256(payload).hexdigest()}"


def _shard_key(
    *,
    task: ForecastTask,
    ordinal: int,
    series_keys: tuple[str, ...],
    series_start: int,
    series_stop: int,
) -> str:
    payload = canonical_json_bytes(
        {
            "model_config": dict(task.model_config),
            "ordinal": ordinal,
            "panel_identity": task.cursor.panel_identity,
            "scope": task.scope.value,
            "series_keys": list(series_keys),
            "series_start": task.cursor.series_start + series_start,
            "series_stop": task.cursor.series_start + series_stop,
        },
        path="forecast shard key",
    )
    return f"forecast-shard:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "DispatchBackend",
    "ForecastDispatchError",
    "ForecastExecutionBudget",
    "ForecastResultEnvelope",
    "ForecastShard",
    "ForecastShardExecutor",
    "ForecastWork",
    "InProcessDispatch",
    "build_forecast_work",
    "canonical_shard_ranges",
    "validate_forecast_envelopes",
]
