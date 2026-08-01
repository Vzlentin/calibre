"""Exercise deterministic Ray resources and failure selection."""

from __future__ import annotations

import inspect
import os
import time

import pandas as pd
import pytest

import newcalibre.engine.ray as ray_backend
from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    CycleToken,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import (
    ForecastDispatchError,
    ForecastLifecycle,
    IndexedPanel,
    RayDispatch,
)
from newcalibre.forecasting import resolve_adapter

pytestmark = pytest.mark.tier2

_THREAD_VARIABLES = {
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
}


def _work():
    calendar = Calendar("D", phase=pd.Timestamp("2026-01-01"))
    series = tuple(f"s-{ordinal:02d}" for ordinal in range(16))
    frame = pd.DataFrame.from_records(
        [
            {SERIES_KEY: key, TIMESTAMP: timestamp, OBSERVED_VALUE: float(ordinal + day)}
            for ordinal, key in enumerate(series)
            for day, timestamp in enumerate(pd.date_range("2026-01-01", periods=4, freq="D"))
        ]
    ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"})
    panel = Panel.from_frame(frame, calendar=calendar, target_support=TargetSupport.REAL)
    session = SessionIdentity.derive(
        tenant="ray-failure",
        series_keys=series,
        calendar=calendar,
        horizon=2,
        model_config={"backend": "seasonal-naive", "m": 2},
    )
    task = IndexedPanel.from_panel(panel).tasks(
        origin=pd.Timestamp("2026-01-04"),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config={"backend": "seasonal-naive", "m": 2},
    )[0]
    token = CycleToken(session, task.origin, 1, 1, "0" * 32)
    lifecycle = ForecastLifecycle(adapter_resolver=resolve_adapter)
    dispatch = RayDispatch()
    work = lifecycle.prepare_work(
        session=session,
        task=task,
        token=token,
        checkpoints={},
        checkpoint_indexes={},
        backend=dispatch.backend,
        budget=dispatch.budget,
    )
    return dispatch, lifecycle, work


def test_ray_reports_the_lowest_failed_ordinal_after_the_complete_barrier() -> None:
    """Choose attributable failure independently of completion order."""

    class FailingExecutor:
        def __init__(self, lifecycle: ForecastLifecycle) -> None:
            self._lifecycle = lifecycle

        def run_shard(self, shard):
            time.sleep((16 - shard.ordinal) / 1000)
            if shard.ordinal in {2, 7}:
                raise RuntimeError(f"failure-{shard.ordinal}")
            if any(os.environ.get(name) != "1" for name in _THREAD_VARIABLES):
                raise RuntimeError("numeric worker was not pinned to one thread")
            return self._lifecycle.run_shard(shard)

    dispatch, lifecycle, work = _work()
    try:
        with pytest.raises(ForecastDispatchError, match=r"ordinal=2") as caught:
            dispatch.dispatch(work, FailingExecutor(lifecycle))
    finally:
        dispatch.shutdown()

    assert caught.value.__cause__ is not None
    assert "failure-2" in str(caught.value)


def test_ray_options_pin_resources_threads_and_zero_retries() -> None:
    """Keep all Ray resource and retry policy explicit at its sole adapter."""
    assert ray_backend._REMOTE_OPTIONS == {
        "max_retries": 0,
        "num_cpus": 1,
        "num_gpus": 0,
    }
    assert set(ray_backend._WORKER_ENV) == _THREAD_VARIABLES
    assert set(ray_backend._WORKER_ENV.values()) == {"1"}
    source = inspect.getsource(ray_backend)
    assert "max_retries" in source
    assert "max_task_retries" not in source
