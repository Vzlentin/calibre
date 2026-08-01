"""Place typed forecast shards on one deterministic local Ray node."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Final

import pandas as pd
import ray
from ray.exceptions import RayTaskError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from newcalibre.domain import (
    TIMESTAMP,
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
    _require_backend_work,
)
from newcalibre.forecasting import AdapterExecutionMode

RAY_BACKEND: Final = "ray"
RAY_WORKER_THREAD_POLICY: Final = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_RAY_LOGICAL_SHARDS: Final = 16
# This spelling resolves to 127.0.0.1 without triggering Ray's exact-string rewrite.
_LOOPBACK_ADDRESS: Final = "127.0.0.01"
_ACTOR_OPTIONS: Final = {
    "max_restarts": 0,
    "max_task_retries": 0,
    "num_cpus": 1,
    "num_gpus": 0,
    "runtime_env": {"env_vars": dict(RAY_WORKER_THREAD_POLICY)},
}


@dataclass(frozen=True, slots=True)
class _CachedHistory:
    """Retain one materialized shard without its driver-side staged panel."""

    frame: pd.DataFrame = field(repr=False, compare=False)
    calendar: Calendar
    identity: str
    series_keys: tuple[str, ...]
    series_start: int
    series_stop: int

    @classmethod
    def stage(cls, task: ForecastTask) -> _CachedHistory:
        """Materialize the full shard exactly once inside its ordinal worker."""
        cursor = task.cursor
        return cls(
            frame=task.history.materialize(),
            calendar=task.calendar,
            identity=cursor.panel_identity,
            series_keys=("",) * cursor.series_start + task.series_keys,
            series_start=cursor.series_start,
            series_stop=cursor.series_stop,
        )

    def advance(self, delta: pd.DataFrame) -> _CachedHistory:
        """Append only the newly admissible driver transport."""
        frame = self.frame if delta.empty else pd.concat((self.frame, delta), ignore_index=True)
        return _CachedHistory(
            frame=frame,
            calendar=self.calendar,
            identity=self.identity,
            series_keys=self.series_keys,
            series_start=self.series_start,
            series_stop=self.series_stop,
        )

    def materialize(
        self,
        *,
        series_start: int,
        series_stop: int,
        time_start: int,
        time_stop: int,
    ) -> pd.DataFrame:
        """Materialize one cached time interval for the fixed shard range."""
        if (series_start, series_stop) != (self.series_start, self.series_stop):
            raise ForecastDispatchError("cached Ray history received a foreign series range")
        phase = self.calendar.phase
        if phase is None:
            raise ForecastDispatchError("cached Ray history requires a bound calendar")
        start = self.calendar.advance(phase, time_start)
        stop = self.calendar.advance(phase, time_stop)
        selected = self.frame[self.frame[TIMESTAMP].ge(start) & self.frame[TIMESTAMP].lt(stop)]
        return selected.copy(deep=True).reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class _HistoryDeltaTransport:
    """Carry only one shard delta and current-origin task metadata."""

    key: str
    work_key: str
    backend: str
    ordinal: int
    session: SessionIdentity
    token: CycleToken
    semantic_task_identity: str
    task_identity: str
    series_keys: tuple[str, ...]
    start_cursor: HistoryCursor
    end_cursor: HistoryCursor
    delta: pd.DataFrame = field(repr=False, compare=False)
    future_exogenous: pd.DataFrame | None = field(repr=False, compare=False)
    horizon: int
    origin: pd.Timestamp
    calendar: Calendar
    model_config: dict[str, object]
    scope: Scope

    @classmethod
    def from_shard(cls, shard: ForecastShard) -> _HistoryDeltaTransport:
        """Detach a current-origin update from the full staged-history plane."""
        if shard.prior_native_state is None or shard.prior_time_bound is None:
            raise ForecastDispatchError("cached Ray update requires committed shard state")
        return cls(
            key=shard.key,
            work_key=shard.work_key,
            backend=shard.backend,
            ordinal=shard.ordinal,
            session=shard.session,
            token=shard.token,
            semantic_task_identity=shard.semantic_task_identity,
            task_identity=shard.task.identity,
            series_keys=shard.series_keys,
            start_cursor=shard.task.delta.start_cursor,
            end_cursor=shard.task.cursor,
            delta=shard.task.delta.materialize(),
            future_exogenous=shard.task.future_exogenous,
            horizon=shard.task.horizon,
            origin=shard.task.origin,
            calendar=shard.task.calendar,
            model_config=dict(shard.task.model_config),
            scope=shard.task.scope,
        )


class _OrdinalWorker:
    """Cache one ordinal's staged history and reconstructible model state."""

    def __init__(self, ordinal: int) -> None:
        self._ordinal = ordinal
        self._executor: ForecastShardExecutor | None = None
        self._history: _CachedHistory | None = None
        self._cursor: HistoryCursor | None = None
        self._result: ForecastResultEnvelope | None = None

    def stage(
        self,
        executor: ForecastShardExecutor,
        shard: ForecastShard,
    ) -> tuple[ForecastResultEnvelope, str]:
        """Stage full history or reconstruct from a committed checkpoint."""
        self._require_ordinal(shard.ordinal)
        history = None if shard.empty else _CachedHistory.stage(shard.task)
        result = executor.run_shard(shard)
        self._executor = executor
        self._history = history
        self._cursor = shard.task.cursor
        self._result = result
        return result, "stage"

    def advance(
        self,
        update: _HistoryDeltaTransport,
    ) -> tuple[ForecastResultEnvelope, str]:
        """Advance cached history and state from delta-only transport."""
        self._require_ordinal(update.ordinal)
        executor = self._executor
        history = self._history
        cursor = self._cursor
        previous = self._result
        if executor is None or history is None or cursor is None or previous is None:
            raise ForecastDispatchError("Ray ordinal worker has no staged cache")
        if update.start_cursor != cursor or update.key != previous.shard_key:
            raise ForecastDispatchError("Ray ordinal worker cache is not committed input state")
        history = history.advance(update.delta)
        view = HistoryView._from_storage(history, cursor=update.end_cursor)
        delta = HistoryDelta._from_storage(
            history,
            start_cursor=update.start_cursor,
            end_cursor=update.end_cursor,
        )
        task = ForecastTask._from_components(
            history=view,
            delta=delta,
            cursor=update.end_cursor,
            future_exogenous=update.future_exogenous,
            horizon=update.horizon,
            origin=update.origin,
            calendar=update.calendar,
            model_config=update.model_config,
            scope=update.scope,
            series_keys=update.series_keys,
        )
        if task.identity != update.task_identity:
            raise ForecastDispatchError("cached Ray task reconstruction changed semantic identity")
        shard = ForecastShard(
            key=update.key,
            work_key=update.work_key,
            backend=update.backend,
            ordinal=update.ordinal,
            session=update.session,
            token=update.token,
            semantic_task_identity=update.semantic_task_identity,
            task=task,
            series_keys=update.series_keys,
            prior_native_state=previous.native_state,
            prior_time_bound=update.start_cursor.time_bound,
            fit_time_bound=previous.fit_time_bound,
        )
        result = executor.run_shard(shard)
        self._history = history
        self._cursor = update.end_cursor
        self._result = result
        return result, "delta"

    def _require_ordinal(self, ordinal: int) -> None:
        if ordinal != self._ordinal:
            raise ForecastDispatchError("Ray ordinal worker received foreign physical work")


_REMOTE_WORKER = ray.remote(_OrdinalWorker)


@dataclass(frozen=True, slots=True)
class _CachedResult:
    """Track the last successful result eligible for committed delta transport."""

    shard_key: str
    cursor: HistoryCursor
    native_state: bytes | None


class RayDispatch:
    """Execute fixed logical work on stable one-thread local Ray actors."""

    def __init__(
        self,
        *,
        logical_shards: int = _RAY_LOGICAL_SHARDS,
        workers: int = _RAY_LOGICAL_SHARDS,
        numeric_threads_per_worker: int = 1,
        retries: int = 0,
    ) -> None:
        self._budget = ForecastExecutionBudget(
            logical_shards=logical_shards,
            concurrency=workers,
            numeric_threads_per_worker=numeric_threads_per_worker,
            retries=retries,
        )
        if (
            self._budget.logical_shards != _RAY_LOGICAL_SHARDS
            or self._budget.concurrency != _RAY_LOGICAL_SHARDS
        ):
            raise ForecastDispatchError("M5 Ray dispatch requires exactly 16 shards and workers")
        self._owns_runtime = False
        self._workers: dict[int, object] = {}
        self._cached_results: dict[int, _CachedResult] = {}
        self._transport_modes: list[tuple[int, str]] = []

    @property
    def backend(self) -> str:
        """Return the stable Ray backend identity."""
        return RAY_BACKEND

    @property
    def budget(self) -> ForecastExecutionBudget:
        """Return the fixed 16-worker execution budget."""
        return self._budget

    def dispatch(
        self,
        work: ForecastWork,
        executor: ForecastShardExecutor,
    ) -> tuple[ForecastResultEnvelope, ...]:
        """Collect every outcome before selecting the lowest failed ordinal."""
        _require_backend_work(self, work)
        expected_count = (
            _RAY_LOGICAL_SHARDS
            if work.execution_mode is AdapterExecutionMode.SERIES_SEPARABLE
            else 1
        )
        if len(work.shards) != expected_count:
            raise ForecastDispatchError("Ray forecast work has the wrong logical shard count")
        self._ensure_runtime()
        references: dict[object, ForecastShard] = {}
        for shard in work.shards:
            worker = self._worker(shard.ordinal)
            if self._can_advance(shard):
                reference = worker.advance.remote(_HistoryDeltaTransport.from_shard(shard))
            else:
                reference = worker.stage.remote(executor, shard)
            references[reference] = shard

        successes: list[tuple[ForecastShard, ForecastResultEnvelope, str]] = []
        failures: list[tuple[ForecastShard, BaseException]] = []
        ready, _pending = ray.wait(list(references), num_returns=len(references))
        for reference in ready:
            shard = references[reference]
            try:
                envelope, mode = ray.get(reference)
                successes.append((shard, envelope, mode))
            except BaseException as error:
                failures.append((shard, error))
        if failures:
            self._drop_workers()
            shard, error = min(failures, key=lambda failure: failure[0].ordinal)
            cause = error.as_instanceof_cause() if isinstance(error, RayTaskError) else error
            raise ForecastDispatchError(
                "forecast shard failed: "
                f"key={shard.key} ordinal={shard.ordinal} backend={self.backend}; "
                f"cause={cause}"
            ) from cause

        for shard, envelope, _mode in successes:
            self._cached_results[shard.ordinal] = _CachedResult(
                shard_key=shard.key,
                cursor=shard.task.cursor,
                native_state=envelope.native_state,
            )
        self._transport_modes.extend(
            sorted(
                ((shard.ordinal, mode) for shard, _envelope, mode in successes),
                key=lambda item: item[0],
            )
        )
        return tuple(envelope for _shard, envelope, _mode in successes)

    def shutdown(self) -> None:
        """Shut down only the Ray runtime initialized by this backend."""
        self._workers.clear()
        self._cached_results.clear()
        if self._owns_runtime and ray.is_initialized():
            ray.shutdown()
        self._owns_runtime = False

    def _ensure_runtime(self) -> None:
        if ray.is_initialized():
            if not self._owns_runtime:
                raise ForecastDispatchError("Ray dispatch refuses an ambient runtime")
            self._require_local_runtime()
            return
        ray.init(
            address="local",
            _node_ip_address=_LOOPBACK_ADDRESS,
            include_dashboard=False,
            num_cpus=self._budget.concurrency,
            runtime_env={"env_vars": dict(RAY_WORKER_THREAD_POLICY)},
        )
        self._owns_runtime = True
        try:
            self._require_local_runtime()
        except Exception:
            ray.shutdown()
            self._owns_runtime = False
            raise

    def _require_local_runtime(self) -> None:
        nodes = tuple(node for node in ray.nodes() if node.get("Alive"))
        if len(nodes) != 1 or nodes[0].get("NodeManagerAddress") != _LOOPBACK_ADDRESS:
            raise ForecastDispatchError("Ray dispatch requires one loopback-only local node")

    def _worker(self, ordinal: int):
        worker = self._workers.get(ordinal)
        if worker is not None:
            return worker
        node_id = ray.get_runtime_context().get_node_id()
        worker = _REMOTE_WORKER.options(
            **_ACTOR_OPTIONS,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(ordinal)
        self._workers[ordinal] = worker
        return worker

    def _can_advance(self, shard: ForecastShard) -> bool:
        cached = self._cached_results.get(shard.ordinal)
        return (
            cached is not None
            and not shard.empty
            and shard.ordinal in self._workers
            and shard.key == cached.shard_key
            and shard.prior_native_state is not None
            and shard.prior_native_state == cached.native_state
            and shard.prior_time_bound == cached.cursor.time_bound
            and shard.task.delta.start_cursor == cached.cursor
        )

    def _drop_workers(self) -> None:
        workers = tuple(self._workers.values())
        self._workers.clear()
        self._cached_results.clear()
        for worker in workers:
            # A failed or lost actor is already unavailable for cache reuse.
            with suppress(Exception):
                ray.kill(worker, no_restart=True)


__all__ = ["RAY_BACKEND", "RAY_WORKER_THREAD_POLICY", "RayDispatch"]
