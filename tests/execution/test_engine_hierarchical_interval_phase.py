"""Tests for the hierarchical-interval phase of the engine."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    DS,
    FITTED_Y_HAT,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.core.forecast_task import ForecastTask
from calibre.execution.actuals import FrameActualsSource
from calibre.execution.backend import (
    BackendEngine,
    ConformalOptions,
    HierarchicalIntervalEngineOptions,
    ReconciliationOptions,
    _with_group_tag,
)
from calibre.execution.ledger import InMemoryLedger, InMemoryOrderLedger
from calibre.execution.task_builder import partition_tasks
from calibre.forecasting.adapter_base import ModelAdapter, build_fitted_values_frame
from calibre.ordering.policy_config import RsConfig
from calibre.reconciliation import HierarchicalIntervalContext, ReconciliationContext
from calibre.reconciliation.summing import (
    HierarchyIndex,
    build_hierarchy_index,
    build_summing_matrix,
)


@contextmanager
def _materialize_refs_for(engine: BackendEngine, tasks: list[ForecastTask]):
    groups = partition_tasks(tasks)
    local_tasks = [_with_group_tag(t) for t in groups.local]
    direct_tasks = [_with_group_tag(t) for t in groups.global_]
    with engine._task_staging_prefix() as staging_prefix:
        chunk_refs = engine._materialize_local_chunks(local_tasks, f"{staging_prefix}/local")
        direct_refs = engine._materialize_task_refs(direct_tasks, f"{staging_prefix}/global")
        yield chunk_refs, direct_refs


class _SimpleAdapter(ModelAdapter):
    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del task, collect_fitted_values

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        uid = str(task.history[UNIQUE_ID].iloc[0])
        last_ds = task.history[DS].max()
        return pd.DataFrame(
            [
                {UNIQUE_ID: uid, DS: last_ds + pd.Timedelta(days=h), Y_HAT: 10.0 + h, H: h}
                for h in range(1, task.horizon + 1)
            ]
        )

    def fitted_values(self, task: ForecastTask) -> pd.DataFrame:
        raw = task.history[[UNIQUE_ID, DS, Y]].copy()
        raw[FITTED_Y_HAT] = raw[Y].astype(float) - 0.5
        return build_fitted_values_frame(raw, model_name=task.model_name)


class _NoFittedAdapter(_SimpleAdapter):
    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del task
        if collect_fitted_values:
            raise AssertionError("flat path should not request fitted values")


class _SpyReconciler:
    requires_fitted_values = False

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        del hierarchy, context
        self.calls += 1
        return frame


class _SpyPhase:
    requires_fitted_values = True

    def __init__(self, *, explode: bool = False) -> None:
        self.calls = 0
        self.fitted_values: pd.DataFrame | None = None
        self.explode = explode

    def apply(
        self,
        frame: pd.DataFrame,
        hierarchy_index: HierarchyIndex,
        context: HierarchicalIntervalContext,
    ) -> pd.DataFrame:
        del hierarchy_index
        self.calls += 1
        self.fitted_values = context.fitted_values
        if self.explode:
            raise RuntimeError("fused intervals exploded")
        lower_col, upper_col = interval_column_names(0.9)
        out = frame.copy()
        out[lower_col] = out[Y_HAT].astype(float) - 1.0
        out[upper_col] = out[Y_HAT].astype(float) + 1.0
        return out


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame({UNIQUE_ID: ["a", "b"], "dept": ["D", "D"]})


def _node_history() -> pd.DataFrame:
    hierarchy = _hierarchy()
    summing = build_summing_matrix(hierarchy)
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    bottom = pd.DataFrame(
        [
            {UNIQUE_ID: uid, DS: ds, Y: float(idx + step + 1)}
            for idx, uid in enumerate(summing.bottom_ids)
            for step, ds in enumerate(dates)
        ]
    )
    from calibre.execution.task_builder import build_node_history

    return build_node_history(bottom, build_hierarchy_index(hierarchy))


def _tasks(history: pd.DataFrame, *, horizon: int = 1) -> list[ForecastTask]:
    return [
        ForecastTask(
            history=group.reset_index(drop=True),
            horizon=horizon,
            model_config={"backend": "statsforecast", "model": "stub", "name": "m"},
        )
        for _, group in history.groupby(UNIQUE_ID, sort=False)
    ]


def _actuals(history: pd.DataFrame) -> pd.DataFrame:
    return history[[UNIQUE_ID, DS, Y]].copy()


def test_default_route_still_calls_reconcile_and_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _NoFittedAdapter(model_config),
    )
    history = _node_history()
    reconciler = _SpyReconciler()
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(
            reconciler=reconciler, hierarchy_index=build_hierarchy_index(_hierarchy())
        )
    )
    calls = {"calibrate": 0}

    def _calibrate(
        self: BackendEngine,
        origin_preds: pd.DataFrame,
        conformal_runtime: Any,
        fitted_context: Any,
    ):
        del self, conformal_runtime, fitted_context
        calls["calibrate"] += 1
        return origin_preds

    engine._calibrate = MethodType(_calibrate, engine)

    with _materialize_refs_for(engine, _tasks(history)) as (chunk_refs, direct_refs):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=FrameActualsSource(_actuals(history)),
            origin=pd.Timestamp("2024-01-04"),
            conformal_runtime=None,
            chunk_refs=chunk_refs,
            direct_refs=direct_refs,
        )

    assert reconciler.calls == 1
    assert calls["calibrate"] == 1


def test_fused_phase_bypasses_reconcile_and_calibrate_and_commits_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _SimpleAdapter(model_config),
    )
    seen_order_uids: list[str] = []

    def _fake_apply_order_policy(order_frame: pd.DataFrame, config: Any) -> pd.DataFrame:
        del config
        seen_order_uids.extend(order_frame[UNIQUE_ID].astype(str).tolist())
        return pd.DataFrame({UNIQUE_ID: seen_order_uids, "order_qty": [0.0] * len(order_frame)})

    monkeypatch.setattr("calibre.execution.backend.apply_order_policy", _fake_apply_order_policy)
    history = _node_history()
    phase = _SpyPhase()
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(hierarchy_index=build_hierarchy_index(_hierarchy())),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(phase=phase),
        order=RsConfig(params=pd.DataFrame()),
    )
    engine._reconcile = MethodType(
        lambda self, origin_preds, context: (_ for _ in ()).throw(
            AssertionError("reconcile should be bypassed")
        ),
        engine,
    )
    engine._calibrate = MethodType(
        lambda self, origin_preds, runtime: (_ for _ in ()).throw(
            AssertionError("calibrate should be bypassed")
        ),
        engine,
    )
    ledger = InMemoryLedger()
    order_ledger = InMemoryOrderLedger()

    with _materialize_refs_for(engine, _tasks(history)) as (chunk_refs, direct_refs):
        engine.run_origin(
            ledger=ledger,
            order_ledger=order_ledger,
            actuals=FrameActualsSource(_actuals(history)),
            origin=pd.Timestamp("2024-01-04"),
            conformal_runtime=None,
            chunk_refs=chunk_refs,
            direct_refs=direct_refs,
        )

    assert phase.calls == 1
    assert phase.fitted_values is not None
    assert not ledger.to_df().empty
    assert seen_order_uids == ["a", "b"]


def test_fused_phase_rejects_conformal_runtime() -> None:
    with pytest.raises(ValueError, match="cannot be combined with conformal runtime"):
        BackendEngine(
            conformal=ConformalOptions(runtime=object()),  # type: ignore[arg-type]
            reconciliation=ReconciliationOptions(
                hierarchy_index=build_hierarchy_index(_hierarchy())
            ),
            hierarchical_intervals=HierarchicalIntervalEngineOptions(phase=_SpyPhase()),
        )


def test_fused_phase_rejects_point_reconciler() -> None:
    with pytest.raises(ValueError, match="cannot be combined with point reconciliation"):
        BackendEngine(
            reconciliation=ReconciliationOptions(
                reconciler=_SpyReconciler(),
                hierarchy_index=build_hierarchy_index(_hierarchy()),
            ),
            hierarchical_intervals=HierarchicalIntervalEngineOptions(phase=_SpyPhase()),
        )


def test_fused_phase_failure_names_phase_and_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _SimpleAdapter(model_config),
    )
    history = _node_history()
    origin = pd.Timestamp("2024-01-04")
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(hierarchy_index=build_hierarchy_index(_hierarchy())),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(phase=_SpyPhase(explode=True)),
    )

    with (
        _materialize_refs_for(engine, _tasks(history)) as (chunk_refs, direct_refs),
        pytest.raises(
            RuntimeError,
            match=rf"HierarchicalIntervals phase failed at origin {origin}",
        ),
    ):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=FrameActualsSource(_actuals(history)),
            origin=origin,
            conformal_runtime=None,
            chunk_refs=chunk_refs,
            direct_refs=direct_refs,
        )


def test_flat_default_path_does_not_request_fitted_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _NoFittedAdapter(model_config),
    )
    history = _node_history()
    bottom_history = history[history[UNIQUE_ID].isin(["a", "b"])].copy()
    engine = BackendEngine()

    with _materialize_refs_for(engine, _tasks(bottom_history)) as (chunk_refs, direct_refs):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=FrameActualsSource(_actuals(bottom_history)),
            origin=pd.Timestamp("2024-01-04"),
            conformal_runtime=None,
            chunk_refs=chunk_refs,
            direct_refs=direct_refs,
        )


def test_initial_ledger_resume_skips_fused_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _SimpleAdapter(model_config),
    )
    history = _node_history()
    initial = pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(["a"], dtype="object"),
            DS: pd.to_datetime(["2024-01-05"]),
            Y: np.array([np.nan], dtype=np.float64),
            Y_HAT: np.array([1.0], dtype=np.float64),
            H: np.array([1], dtype=np.int64),
            FORECAST_ORIGIN: pd.to_datetime(["2024-01-04"]),
            MODEL_NAME: pd.Series(["m"], dtype="object"),
        }
    )
    phase = _SpyPhase(explode=True)
    engine = BackendEngine(
        conformal=ConformalOptions(initial_ledger=initial),
        reconciliation=ReconciliationOptions(hierarchy_index=build_hierarchy_index(_hierarchy())),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(phase=phase),
    )

    result = engine.execute(
        partition_tasks(_tasks(history)),
        _actuals(history),
        [pd.Timestamp("2024-01-04")],
    )

    assert phase.calls == 0
    assert len(result.ledger.to_df()) == 1
