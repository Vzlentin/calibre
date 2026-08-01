"""Exercise deterministic Ray resources and failure selection."""

from __future__ import annotations

import inspect
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

import newcalibre.engine.ray as ray_backend
from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    CycleToken,
    ForecastTask,
    HierarchyIndex,
    HistoryDelta,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import (
    Engine,
    ForecastDispatchError,
    ForecastLifecycle,
    IndexedPanel,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    InProcessDispatch,
    OriginIntent,
    OriginRequest,
    PhaseError,
    RayDispatch,
    Spine,
)
from newcalibre.forecasting import (
    AdapterCapability,
    AdapterExecutionMode,
    SeasonalNaiveAdapter,
    resolve_adapter,
)

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


class _ControlledSeasonalAdapter:
    """Expose deterministic test-controlled failures around the real adapter."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        self._config = dict(model_config)
        self._delegate = SeasonalNaiveAdapter(model_config)

    @property
    def execution_mode(self) -> AdapterExecutionMode:
        mode = self._config.get("execution_mode", "series-separable")
        return AdapterExecutionMode(mode)

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return self._delegate.capabilities

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return self._delegate.requested_capabilities

    def fit(self, task: ForecastTask) -> None:
        self._delegate.fit(task)

    def predict(self, task: ForecastTask):
        origin = self._config.get("failure_origin")
        series = self._config.get("failure_series")
        if origin == str(task.origin.date()) and series in task.series_keys:
            raise RuntimeError(f"controlled failure for {series}")
        kill_origin = self._config.get("worker_loss_origin")
        kill_series = self._config.get("worker_loss_series")
        if kill_origin == str(task.origin.date()) and kill_series in task.series_keys:
            witness = self._config.get("attempt_witness")
            if not isinstance(witness, str):
                raise RuntimeError("worker-loss witness is missing")
            with Path(witness).open("a", encoding="utf-8") as stream:
                stream.write("attempt\n")
            os._exit(23)
        return self._delegate.predict(task)

    def dump_state(self) -> bytes:
        return self._delegate.dump_state()

    def load_state(self, state: bytes) -> None:
        self._delegate.load_state(state)

    def fitted_values(self):
        return self._delegate.fitted_values()

    def update(self, delta: HistoryDelta) -> None:
        self._delegate.update(delta)


class _SafeSeasonalAdapter(_ControlledSeasonalAdapter):
    """Ignore injected failures while preserving the exact adapter contract."""

    def predict(self, task: ForecastTask):
        return self._delegate.predict(task)


def _controlled_resolver(model_config: Mapping[str, object]):
    return _ControlledSeasonalAdapter(model_config)


def _safe_resolver(model_config: Mapping[str, object]):
    return _SafeSeasonalAdapter(model_config)


def _engine_world(
    model_config: Mapping[str, object],
    *,
    tenant: str,
    dispatch: InProcessDispatch | RayDispatch,
    safe: bool = False,
):
    ray_backend.ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    calendar = Calendar("D", phase=pd.Timestamp("2026-01-01"))
    series = tuple(f"s-{ordinal:02d}" for ordinal in range(16))
    frame = pd.DataFrame.from_records(
        [
            {SERIES_KEY: key, TIMESTAMP: timestamp, OBSERVED_VALUE: float(ordinal + day)}
            for ordinal, key in enumerate(series)
            for day, timestamp in enumerate(pd.date_range("2026-01-01", periods=7, freq="D"))
        ]
    ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"})
    panel = Panel.from_frame(frame, calendar=calendar, target_support=TargetSupport.REAL)
    session = SessionIdentity.derive(
        tenant=tenant,
        series_keys=series,
        calendar=calendar,
        horizon=2,
        model_config=model_config,
    )
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=calendar,
        actuals=panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    engine = Engine(
        session=session,
        panel_source=InMemoryPanelSource(panel),
        run_store=store,
        dispatch_backend=dispatch,
        hierarchy=HierarchyIndex.flat(series),
        adapter_resolver=_safe_resolver if safe else _controlled_resolver,
        orderer=None,
    )
    return panel, session, store, engine


def _run_origin(engine: Engine, store: InMemoryIndexedRunStore, origin: str) -> None:
    timestamp = pd.Timestamp(origin)
    snapshot = store.open(OriginIntent(store.session, timestamp))
    Spine(engine).run_origin(
        OriginRequest(session=store.session, origin=timestamp, scope=Scope.GLOBAL),
        snapshot=snapshot,
    )


def _committed_state(store: InMemoryIndexedRunStore) -> tuple[object, ...]:
    return (
        store.revision,
        store.resume_marker,
        tuple(store.receipts.items()),
        tuple(store.states.items()),
        tuple(store.checkpoints.items()),
        tuple(store.checkpoint_indexes.items()),
        store.forecasts,
        store.orders,
        store.settlements,
        store.observed_history,
        store.pending_observations,
        store.earliest_origin,
        store.latest_origin,
        store.forecast_origin_count,
    )


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
    assert ray_backend._ACTOR_OPTIONS == {
        "max_restarts": 0,
        "max_task_retries": 0,
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": ray_backend._WORKER_ENV},
    }
    assert set(ray_backend._WORKER_ENV) == _THREAD_VARIABLES
    assert set(ray_backend._WORKER_ENV.values()) == {"1"}
    source = inspect.getsource(ray_backend)
    assert 'address="local"' in source
    assert "_node_ip_address=_LOOPBACK_ADDRESS" in source
    assert "NodeAffinitySchedulingStrategy" in source


def test_ray_executes_monolithic_work_as_one_production_item() -> None:
    """Run a real monolithic adapter through Engine and the Ray backend."""
    dispatch = RayDispatch()
    _panel, session, store, engine = _engine_world(
        {
            "backend": "seasonal-naive",
            "execution_mode": "monolithic",
            "m": 2,
        },
        tenant="ray-monolithic",
        dispatch=dispatch,
    )
    try:
        _run_origin(engine, store, "2026-01-04")
    finally:
        dispatch.shutdown()

    assert store.session == session
    assert len(store.forecasts) == 32
    assert dispatch._transport_modes == [(0, "stage")]


def test_ray_refuses_an_incompatible_ambient_runtime() -> None:
    """Fail closed when another owner already initialized Ray."""
    ray_backend.ray.init(
        address="local",
        _node_ip_address="127.0.0.1",
        include_dashboard=False,
        num_cpus=1,
    )
    dispatch, lifecycle, work = _work()
    try:
        with pytest.raises(ForecastDispatchError, match="ambient runtime"):
            dispatch.dispatch(work, lifecycle)
    finally:
        dispatch.shutdown()
        ray_backend.ray.shutdown()


def test_ray_shard_failure_preserves_every_committed_surface() -> None:
    """Publish no sibling result across the Engine transaction boundary."""
    dispatch = RayDispatch()
    _panel, _session, store, engine = _engine_world(
        {
            "backend": "seasonal-naive",
            "execution_mode": "series-separable",
            "failure_origin": "2026-01-05",
            "failure_series": "s-02",
            "m": 2,
        },
        tenant="ray-transaction-failure",
        dispatch=dispatch,
    )
    try:
        _run_origin(engine, store, "2026-01-04")
        committed = _committed_state(store)

        with pytest.raises(PhaseError, match=r"ordinal=2"):
            _run_origin(engine, store, "2026-01-05")
    finally:
        dispatch.shutdown()

    assert _committed_state(store) == committed


def test_worker_loss_runs_once_and_reconstructs_from_committed_checkpoint(
    tmp_path: Path,
) -> None:
    """Require a fresh deterministic attempt after an unretried actor loss."""
    witness = tmp_path / "attempts.txt"
    config = {
        "backend": "seasonal-naive",
        "execution_mode": "series-separable",
        "m": 2,
        "attempt_witness": str(witness),
        "worker_loss_origin": "2026-01-05",
        "worker_loss_series": "s-03",
    }
    dispatch = RayDispatch()
    panel, session, store, engine = _engine_world(
        config,
        tenant="ray-worker-loss",
        dispatch=dispatch,
    )
    try:
        _run_origin(engine, store, "2026-01-04")
        committed = _committed_state(store)

        with pytest.raises(PhaseError, match=r"ordinal=3"):
            _run_origin(engine, store, "2026-01-05")
    finally:
        dispatch.shutdown()

    assert witness.read_text(encoding="utf-8").splitlines() == ["attempt"]
    assert _committed_state(store) == committed

    fresh_dispatch = RayDispatch()
    resumed = Engine(
        session=session,
        panel_source=InMemoryPanelSource(panel),
        run_store=store,
        dispatch_backend=fresh_dispatch,
        hierarchy=HierarchyIndex.flat(session.series_keys),
        adapter_resolver=_safe_resolver,
        orderer=None,
    )
    try:
        _run_origin(resumed, store, "2026-01-05")
    finally:
        fresh_dispatch.shutdown()

    reference_dispatch = InProcessDispatch(logical_shards=16)
    _reference_panel, _reference_session, reference_store, reference_engine = _engine_world(
        config,
        tenant="ray-worker-loss",
        dispatch=reference_dispatch,
        safe=True,
    )
    _run_origin(reference_engine, reference_store, "2026-01-04")
    _run_origin(reference_engine, reference_store, "2026-01-05")

    assert _committed_state(store) == _committed_state(reference_store)
