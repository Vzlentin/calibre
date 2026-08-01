"""Place typed forecast shards on cached workers on one local Ray node."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import pandas as pd
import ray

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    Calendar,
    CycleToken,
    ForecastTask,
    HistoryCursor,
    HistoryDelta,
    HistoryView,
    Scope,
    SessionIdentity,
)
from newcalibre.engine.dispatch import (
    ForecastDispatchError,
    ForecastExecutionBudget,
    ForecastResultEnvelope,
    ForecastShard,
    ForecastShardExecutor,
    ForecastWork,
    require_backend_work,
)

RAY_BACKEND: Final = "ray"
_WORKER_ENV: Final = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_ACTOR_OPTIONS: Final = {
    "max_restarts": 0,
    "max_task_retries": 0,
    "num_cpus": 1,
    "num_gpus": 0,
}
_METHOD_OPTIONS: Final = {"max_task_retries": 0}


@dataclass(frozen=True, slots=True)
class ForecastTransportAudit:
    """Record whether one shard request transported history or only its delta."""

    ordinal: int
    cache_hit: bool
    history_rows: int
    delta_rows: int


@dataclass(frozen=True, slots=True)
class _CacheRecord:
    state_digest: str
    time_bound: int


@dataclass(frozen=True, slots=True)
class _ShardRequest:
    work_key: str
    shard_key: str
    backend: str
    ordinal: int
    session: SessionIdentity
    token: CycleToken
    semantic_task_identity: str
    series_keys: tuple[str, ...]
    empty: bool
    panel_identity: str
    series_start: int
    series_stop: int
    start_cursor: HistoryCursor
    end_cursor: HistoryCursor
    history_frame: pd.DataFrame | None
    delta_frame: pd.DataFrame | None
    future_exogenous: pd.DataFrame | None
    horizon: int
    origin: pd.Timestamp
    calendar: Calendar
    model_config: dict[str, object]
    scope: Scope
    cache_hit: bool
    prior_native_state: bytes | None
    prior_time_bound: int | None
    fit_time_bound: int | None


class _WorkerHistoryStorage:
    """Retain one shard's staged history across actor calls."""

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        identity: str,
        series_start: int,
        series_keys: tuple[str, ...],
        calendar: Calendar,
    ) -> None:
        self._frames = [frame]
        self._series_start = series_start
        self._local_series_keys = series_keys
        self._calendar = calendar
        self.identity = identity
        self.series_keys = ("",) * series_start + series_keys

    def append(self, frame: pd.DataFrame) -> None:
        """Append one already validated contiguous history delta."""
        if not frame.empty:
            self._frames.append(frame)

    def materialize(
        self,
        *,
        series_start: int,
        series_stop: int,
        time_start: int,
        time_stop: int,
    ) -> pd.DataFrame:
        """Materialize canonical rows from the cached shard only."""
        local_start = series_start - self._series_start
        local_stop = series_stop - self._series_start
        if local_start < 0 or local_stop > len(self._local_series_keys):
            raise ForecastDispatchError("worker history request escapes its cached shard")
        keys = self._local_series_keys[local_start:local_stop]
        if not keys:
            return self._frames[0].iloc[0:0].copy(deep=True).reset_index(drop=True)
        phase = self._calendar.phase
        if phase is None:
            raise ForecastDispatchError("worker history requires a bound calendar")
        start = self._calendar.advance(phase, time_start)
        stop = self._calendar.advance(phase, time_stop)
        parts: list[pd.DataFrame] = []
        for key in keys:
            for frame in self._frames:
                selected = frame.loc[
                    frame[SERIES_KEY].eq(key)
                    & frame["timestamp"].ge(start)
                    & frame["timestamp"].lt(stop)
                ]
                if not selected.empty:
                    parts.append(selected)
        if not parts:
            return self._frames[0].iloc[0:0].copy(deep=True).reset_index(drop=True)
        return pd.concat(parts, ignore_index=True).copy(deep=True)


class _ForecastWorker:
    """Own one ordinal's staged history and non-authoritative model state cache."""

    def __init__(self, ordinal: int, executor: ForecastShardExecutor) -> None:
        self._ordinal = ordinal
        self._executor = executor
        self._storage: _WorkerHistoryStorage | None = None
        self._cursor: HistoryCursor | None = None
        self._native_state: bytes | None = None
        self._fit_time_bound: int | None = None

    def run(self, request: _ShardRequest) -> ForecastResultEnvelope:
        """Reconstruct a shard from cached history plus its current delta."""
        if request.ordinal != self._ordinal:
            raise ForecastDispatchError("Ray worker received a foreign shard ordinal")
        if request.empty:
            return _empty_envelope(request)
        if request.cache_hit:
            storage = self._require_cache(request)
            assert request.delta_frame is not None
            storage.append(request.delta_frame)
            prior_native_state = self._native_state
            prior_time_bound = self._cursor.time_bound if self._cursor is not None else None
            fit_time_bound = self._fit_time_bound
        else:
            history = request.history_frame
            if history is None:
                raise ForecastDispatchError("Ray cache miss omitted staged shard history")
            storage = _WorkerHistoryStorage(
                frame=history,
                identity=request.panel_identity,
                series_start=request.series_start,
                series_keys=request.series_keys,
                calendar=request.calendar,
            )
            self._storage = storage
            prior_native_state = request.prior_native_state
            prior_time_bound = request.prior_time_bound
            fit_time_bound = request.fit_time_bound
        view = HistoryView._from_storage(storage, cursor=request.end_cursor)
        delta = HistoryDelta._from_storage(
            storage,
            start_cursor=request.start_cursor,
            end_cursor=request.end_cursor,
        )
        task = ForecastTask._from_components(
            history=view,
            delta=delta,
            cursor=request.end_cursor,
            future_exogenous=request.future_exogenous,
            horizon=request.horizon,
            origin=request.origin,
            calendar=request.calendar,
            model_config=request.model_config,
            scope=request.scope,
            series_keys=request.series_keys,
        )
        shard = ForecastShard(
            key=request.shard_key,
            work_key=request.work_key,
            backend=request.backend,
            ordinal=request.ordinal,
            session=request.session,
            token=request.token,
            semantic_task_identity=request.semantic_task_identity,
            task=task,
            series_keys=request.series_keys,
            prior_native_state=prior_native_state,
            prior_time_bound=prior_time_bound,
            fit_time_bound=fit_time_bound,
        )
        result = self._executor.run_shard(shard)
        self._cursor = request.end_cursor
        self._native_state = result.native_state
        self._fit_time_bound = result.fit_time_bound
        return result

    def _require_cache(self, request: _ShardRequest) -> _WorkerHistoryStorage:
        storage = self._storage
        if (
            storage is None
            or self._cursor != request.start_cursor
            or self._native_state is None
            or self._fit_time_bound is None
        ):
            raise ForecastDispatchError("Ray worker cache does not match committed shard state")
        return storage


_REMOTE_WORKER = ray.remote(**_ACTOR_OPTIONS)(_ForecastWorker)


class RayDispatch:
    """Execute exactly 16 canonical shards on fixed cached local Ray workers."""

    def __init__(
        self,
        *,
        logical_shards: int = 16,
        workers: int = 16,
        numeric_threads_per_worker: int = 1,
        retries: int = 0,
    ) -> None:
        self._budget = ForecastExecutionBudget(
            logical_shards=logical_shards,
            concurrency=workers,
            numeric_threads_per_worker=numeric_threads_per_worker,
            retries=retries,
        )
        if self._budget.logical_shards != 16 or self._budget.concurrency != 16:
            raise ForecastDispatchError("M5 Ray dispatch requires exactly 16 shards and workers")
        self._owns_runtime = False
        self._actors: tuple[object, ...] | None = None
        self._cache: dict[int, _CacheRecord] = {}
        self._transport_audit: tuple[ForecastTransportAudit, ...] = ()

    @property
    def backend(self) -> str:
        """Return the stable Ray backend identity."""
        return RAY_BACKEND

    @property
    def budget(self) -> ForecastExecutionBudget:
        """Return the fixed 16-worker execution budget."""
        return self._budget

    @property
    def transport_audit(self) -> tuple[ForecastTransportAudit, ...]:
        """Return the latest immutable history-versus-delta transport audit."""
        return self._transport_audit

    def dispatch(
        self,
        work: ForecastWork,
        executor: ForecastShardExecutor,
    ) -> tuple[ForecastResultEnvelope, ...]:
        """Collect every cached-worker outcome before selecting one failure."""
        require_backend_work(self, work)
        if len(work.shards) != 16:
            raise ForecastDispatchError("Ray forecast work requires exactly 16 logical shards")
        self._ensure_runtime()
        actors = self._ensure_workers(executor)
        requests = tuple(self._request(shard) for shard in work.shards)
        self._transport_audit = tuple(
            ForecastTransportAudit(
                ordinal=request.ordinal,
                cache_hit=request.cache_hit,
                history_rows=(0 if request.history_frame is None else len(request.history_frame)),
                delta_rows=0 if request.delta_frame is None else len(request.delta_frame),
            )
            for request in requests
        )
        reference_to_shard = {
            actor.run.options(**_METHOD_OPTIONS).remote(request): shard
            for actor, request, shard in zip(actors, requests, work.shards, strict=True)
        }
        pending = list(reference_to_shard)
        ready, pending = ray.wait(pending, num_returns=len(pending))
        if pending:
            raise ForecastDispatchError("Ray complete barrier left pending forecast shards")
        successes: list[ForecastResultEnvelope] = []
        failures: list[tuple[ForecastShard, BaseException]] = []
        for reference in ready:
            shard = reference_to_shard[reference]
            try:
                successes.append(ray.get(reference))
            except BaseException as error:
                failures.append((shard, error))
        if failures:
            shard, error = min(failures, key=lambda failure: failure[0].ordinal)
            cause = (
                error.as_instanceof_cause()
                if isinstance(error, ray.exceptions.RayTaskError)
                else error
            )
            raise ForecastDispatchError(
                "forecast shard failed: "
                f"key={shard.key} ordinal={shard.ordinal} backend={self.backend}; "
                f"cause={cause}"
            ) from cause
        for result in successes:
            if result.native_state is not None:
                self._cache[result.ordinal] = _CacheRecord(
                    state_digest=_state_digest(result.native_state),
                    time_bound=work.shards[result.ordinal].task.cursor.time_bound,
                )
        return tuple(successes)

    def shutdown(self) -> None:
        """Destroy owned actors and shut down only an owned Ray runtime."""
        if self._actors is not None and ray.is_initialized():
            for actor in self._actors:
                ray.kill(actor, no_restart=True)
        self._actors = None
        self._cache.clear()
        self._transport_audit = ()
        if self._owns_runtime and ray.is_initialized():
            ray.shutdown()
        self._owns_runtime = False

    def _request(self, shard: ForecastShard) -> _ShardRequest:
        cached = self._cache.get(shard.ordinal)
        cache_hit = (
            not shard.empty
            and cached is not None
            and shard.prior_native_state is not None
            and cached.state_digest == _state_digest(shard.prior_native_state)
            and cached.time_bound == shard.prior_time_bound
        )
        return _ShardRequest(
            work_key=shard.work_key,
            shard_key=shard.key,
            backend=shard.backend,
            ordinal=shard.ordinal,
            session=shard.session,
            token=shard.token,
            semantic_task_identity=shard.semantic_task_identity,
            series_keys=shard.series_keys,
            empty=shard.empty,
            panel_identity=shard.task.cursor.panel_identity,
            series_start=shard.task.cursor.series_start,
            series_stop=shard.task.cursor.series_stop,
            start_cursor=shard.task.delta.start_cursor,
            end_cursor=shard.task.cursor,
            history_frame=None if cache_hit or shard.empty else shard.task.history.materialize(),
            delta_frame=shard.task.delta.materialize() if cache_hit else None,
            future_exogenous=shard.task.future_exogenous,
            horizon=shard.task.horizon,
            origin=shard.task.origin,
            calendar=shard.task.calendar,
            model_config=dict(shard.task.model_config),
            scope=shard.task.scope,
            cache_hit=cache_hit,
            prior_native_state=None if cache_hit else shard.prior_native_state,
            prior_time_bound=None if cache_hit else shard.prior_time_bound,
            fit_time_bound=None if cache_hit else shard.fit_time_bound,
        )

    def _ensure_runtime(self) -> None:
        if ray.is_initialized():
            return
        ray.init(
            include_dashboard=False,
            num_cpus=self._budget.concurrency,
            runtime_env={"env_vars": dict(_WORKER_ENV)},
        )
        self._owns_runtime = True

    def _ensure_workers(
        self,
        executor: ForecastShardExecutor,
    ) -> tuple[object, ...]:
        if self._actors is None:
            self._actors = tuple(
                _REMOTE_WORKER.remote(ordinal, executor)
                for ordinal in range(self._budget.logical_shards)
            )
        return self._actors


def _empty_envelope(request: _ShardRequest) -> ForecastResultEnvelope:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(dtype="string"),
            TARGET_TIMESTAMP: pd.Series(dtype="datetime64[ns]"),
            ACTUAL_VALUE: pd.Series(dtype="float64"),
            POINT_FORECAST: pd.Series(dtype="float64"),
            HORIZON_STEP: pd.Series(dtype="int64"),
            ORIGIN: pd.Series(dtype="datetime64[ns]"),
            MODEL_NAME: pd.Series(dtype="string"),
        }
    )
    return ForecastResultEnvelope(
        work_key=request.work_key,
        shard_key=request.shard_key,
        backend=request.backend,
        ordinal=request.ordinal,
        session=request.session,
        token=request.token,
        semantic_task_identity=request.semantic_task_identity,
        series_keys=(),
        frame=frame,
        native_state=b"",
        fit_time_bound=request.end_cursor.time_bound,
    )


def _state_digest(state: bytes) -> str:
    return hashlib.sha256(state).hexdigest()


__all__ = ["RAY_BACKEND", "ForecastTransportAudit", "RayDispatch"]
