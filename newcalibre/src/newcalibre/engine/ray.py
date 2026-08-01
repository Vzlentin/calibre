"""Place typed forecast shards on one deterministic local Ray node."""

from __future__ import annotations

from typing import Final

import ray
from ray.exceptions import RayTaskError

from newcalibre.engine.dispatch import (
    ForecastDispatchError,
    ForecastExecutionBudget,
    ForecastResultEnvelope,
    ForecastShard,
    ForecastShardExecutor,
    ForecastWork,
    _require_backend_work,
)

RAY_BACKEND: Final = "ray"
_RAY_LOGICAL_SHARDS: Final = 16
_WORKER_ENV: Final = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_REMOTE_OPTIONS: Final = {
    "max_retries": 0,
    "num_cpus": 1,
    "num_gpus": 0,
}


def _execute_shard(
    executor: ForecastShardExecutor,
    shard: ForecastShard,
) -> ForecastResultEnvelope:
    """Execute one shard inside its one-thread worker process."""
    return executor.run_shard(shard)


_REMOTE_EXECUTE = ray.remote(_execute_shard)


class RayDispatch:
    """Execute exactly 16 canonical shards on fixed local Ray resources."""

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
        if len(work.shards) != _RAY_LOGICAL_SHARDS:
            raise ForecastDispatchError("Ray forecast work requires exactly 16 logical shards")
        self._ensure_runtime()
        reference_to_shard = {
            _REMOTE_EXECUTE.options(**_REMOTE_OPTIONS).remote(executor, shard): shard
            for shard in work.shards
        }
        successes: list[ForecastResultEnvelope] = []
        failures: list[tuple[ForecastShard, BaseException]] = []
        ready, _pending = ray.wait(
            list(reference_to_shard),
            num_returns=len(reference_to_shard),
        )
        for reference in ready:
            shard = reference_to_shard[reference]
            try:
                successes.append(ray.get(reference))
            except BaseException as error:
                failures.append((shard, error))
        if failures:
            shard, error = min(failures, key=lambda failure: failure[0].ordinal)
            cause = error.as_instanceof_cause() if isinstance(error, RayTaskError) else error
            raise ForecastDispatchError(
                "forecast shard failed: "
                f"key={shard.key} ordinal={shard.ordinal} backend={self.backend}; "
                f"cause={cause}"
            ) from cause
        return tuple(successes)

    def shutdown(self) -> None:
        """Shut down only the Ray runtime initialized by this backend."""
        if self._owns_runtime and ray.is_initialized():
            ray.shutdown()
        self._owns_runtime = False

    def _ensure_runtime(self) -> None:
        if ray.is_initialized():
            return
        ray.init(
            include_dashboard=False,
            num_cpus=self._budget.concurrency,
            runtime_env={"env_vars": dict(_WORKER_ENV)},
        )
        self._owns_runtime = True


__all__ = ["RAY_BACKEND", "RayDispatch"]
