"""Phase-level tests for the Reconcile seam (U6).

The Reconcile phase sits between Predict and Calibrate: it rewrites point
``y_hat`` to be coherent across the hierarchy, then conformal calibrates per
node. These tests drive the phase directly (mirroring ``test_engine_phases.py``)
and pin the three no-op guards plus the before-Calibrate ordering.
"""

from contextlib import contextmanager

import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.core.forecast_frame import UNIQUE_ID, Y_HAT
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import (
    BackendEngine,
    ConformalOptions,
    ReconciliationOptions,
    _with_group_tag,
)
from calibre.execution.ledger import InMemoryLedger
from calibre.execution.task_builder import partition_tasks
from calibre.reconciliation import BottomUpReconciler, NoOpReconciler


@contextmanager
def _materialize_refs_for(engine, tasks):
    groups = partition_tasks(tasks)
    parallel_tasks = [_with_group_tag(t) for t in groups.local]
    direct_tasks = [_with_group_tag(t) for t in groups.global_]
    with engine._task_staging_prefix() as staging_prefix:
        parallel_refs = engine._materialize_task_refs(parallel_tasks, f"{staging_prefix}/local")
        direct_refs = engine._materialize_task_refs(direct_tasks, f"{staging_prefix}/global")
        yield parallel_refs, direct_refs


def _periodic_task(horizon=2):
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    return (
        ForecastTask(
            history=pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern}),
            horizon=horizon,
            model_config={
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "season_length": 4,
            },
        ),
        dates,
        pattern,
    )


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame({UNIQUE_ID: ["SKU_001"], "store_id": ["S1"]})


class _SpyReconciler:
    """Records the frame it was handed, then returns it unchanged."""

    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def __call__(self, frame: pd.DataFrame, hierarchy: pd.DataFrame | None) -> pd.DataFrame:
        self.seen = frame.copy()
        return frame


def test_reconcile_noop_without_reconciler() -> None:
    """No reconciler configured -> identity (mirrors calibrate-no-runtime)."""
    engine = BackendEngine()
    frame = pd.DataFrame({UNIQUE_ID: ["A"], Y_HAT: [1.0]})
    out = engine._reconcile(frame)
    pd.testing.assert_frame_equal(out, frame)


def test_reconcile_noop_with_noop_reconciler_and_hierarchy() -> None:
    """The no-op reconciler with a non-None hierarchy is a strict identity."""
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(reconciler=NoOpReconciler(), hierarchy=_hierarchy())
    )
    frame = pd.DataFrame({UNIQUE_ID: ["SKU_001"], Y_HAT: [3.0]})
    out = engine._reconcile(frame)
    pd.testing.assert_frame_equal(out, frame)


def test_reconcile_noop_when_hierarchy_none_even_with_real_strategy() -> None:
    """hierarchy=None short-circuits a real strategy to identity (R3, R11)."""
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(reconciler=BottomUpReconciler(), hierarchy=None)
    )
    frame = pd.DataFrame({UNIQUE_ID: ["SKU_001"], Y_HAT: [3.0]})
    out = engine._reconcile(frame)
    pd.testing.assert_frame_equal(out, frame)


def test_reconcile_noop_on_empty_predictions() -> None:
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(
            reconciler=BottomUpReconciler(), hierarchy=_hierarchy()
        )
    )
    empty = pd.DataFrame(columns=[UNIQUE_ID, Y_HAT])
    out = engine._reconcile(empty)
    assert out.empty


def test_reconcile_runs_before_calibrate_on_raw_yhat() -> None:
    """The reconciler sees raw point forecasts with no interval columns yet (R8)."""
    task, dates, _pattern = _periodic_task()
    runtime = SymmetricIntervalRuntime(
        SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    )
    spy = _SpyReconciler()
    engine = BackendEngine(
        conformal=ConformalOptions(runtime=runtime),
        reconciliation=ReconciliationOptions(reconciler=spy, hierarchy=_hierarchy()),
    )
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})
    lower_col, upper_col = runtime.interval_columns

    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=actuals,
            origin=dates[11],
            conformal_runtime=runtime,
            parallel_refs=parallel_refs,
            direct_refs=direct_refs,
        )

    assert spy.seen is not None
    assert Y_HAT in spy.seen.columns
    assert lower_col not in spy.seen.columns
    assert upper_col not in spy.seen.columns


def test_reconcile_phase_failure_names_phase_and_origin() -> None:
    task, dates, _pattern = _periodic_task()

    def _boom(frame, hierarchy):
        raise RuntimeError("reconcile exploded")

    engine = BackendEngine(
        reconciliation=ReconciliationOptions(reconciler=_boom, hierarchy=_hierarchy())
    )
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": _pattern})
    origin = dates[11]

    with (
        _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs),
        pytest.raises(RuntimeError, match=rf"Reconcile phase failed at origin {origin}"),
    ):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=actuals,
            origin=origin,
            conformal_runtime=None,
            parallel_refs=parallel_refs,
            direct_refs=direct_refs,
        )
