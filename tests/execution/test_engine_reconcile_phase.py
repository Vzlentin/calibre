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
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import (
    BackendEngine,
    ConformalOptions,
    ReconciliationOptions,
    _with_group_tag,
)
from calibre.execution.ledger import InMemoryLedger, InMemoryOrderLedger
from calibre.execution.task_builder import build_node_history, partition_tasks
from calibre.forecasting.adapter_base import ModelAdapter
from calibre.ordering.policy_config import RsConfig
from calibre.reconciliation import NixtlaReconciler, NoOpReconciler, ReconciliationContext


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


class _NoFittedRequestAdapter(ModelAdapter):
    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del task
        if collect_fitted_values:
            raise AssertionError("no-residual path should not request fitted values")

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id],
                DS: [task.history[DS].max() + pd.Timedelta(weeks=1)],
                Y_HAT: [1.0],
                H: [1],
            }
        )


class _SpyReconciler:
    """Records the frame it was handed, then returns it unchanged."""

    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    requires_fitted_values = False

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        del hierarchy, context
        self.seen = frame.copy()
        return frame


class _NeedsFittedSpy:
    requires_fitted_values = True

    def __init__(self) -> None:
        self.fitted: pd.DataFrame | None = None

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        del hierarchy
        self.fitted = context.fitted_values
        return frame


def _boom_if_called(
    frame: pd.DataFrame,
    hierarchy: pd.DataFrame | None,
    context: ReconciliationContext,
) -> pd.DataFrame:
    del frame, hierarchy, context
    raise AssertionError("reconciler should not be called")


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


def test_reconcile_noop_when_hierarchy_none_without_calling_reconciler() -> None:
    """hierarchy=None short-circuits before delegating (R3, R11)."""
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(reconciler=_boom_if_called, hierarchy=None)
    )
    frame = pd.DataFrame({UNIQUE_ID: ["SKU_001"], Y_HAT: [3.0]})
    out = engine._reconcile(frame)
    pd.testing.assert_frame_equal(out, frame)


def test_no_hierarchy_path_does_not_request_fitted_values(monkeypatch) -> None:
    task, dates, pattern = _periodic_task(horizon=1)
    monkeypatch.setattr(
        "calibre.execution.backend.resolve_adapter",
        lambda model_config: _NoFittedRequestAdapter(model_config),
    )
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(
            reconciler=NixtlaReconciler("mint_shrink"),
            hierarchy=None,
        )
    )
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})

    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=actuals,
            origin=dates[11],
            conformal_runtime=None,
            parallel_refs=parallel_refs,
            direct_refs=direct_refs,
        )


def test_residual_reconcile_receives_fitted_context() -> None:
    task, dates, pattern = _periodic_task(horizon=1)
    spy = _NeedsFittedSpy()
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(reconciler=spy, hierarchy=_hierarchy())
    )
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})

    with _materialize_refs_for(engine, [task]) as (parallel_refs, direct_refs):
        engine.run_origin(
            ledger=InMemoryLedger(),
            order_ledger=None,
            actuals=actuals,
            origin=dates[11],
            conformal_runtime=None,
            parallel_refs=parallel_refs,
            direct_refs=direct_refs,
        )

    assert spy.fitted is not None
    assert not spy.fitted.empty
    assert set(spy.fitted[MODEL_NAME]) == {"SeasonalNaive"}


def test_reconcile_noop_on_empty_predictions() -> None:
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(reconciler=_boom_if_called, hierarchy=_hierarchy())
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

    def _boom(frame, hierarchy, context):
        del frame, hierarchy, context
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


def test_resolve_due_fills_aggregate_actuals_from_node_history() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A", "B"], "dept_id": ["D", "D"]})
    bottom_actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: pd.to_datetime(["2024-01-08", "2024-01-08"]),
            Y: [2.0, 3.0],
        }
    )
    actuals = build_node_history(bottom_actuals, hierarchy)
    ledger = InMemoryLedger()
    ledger.append(
        pd.DataFrame(
            {
                UNIQUE_ID: pd.Series(["A", "B", "dept_id=D", "__total__"], dtype="object"),
                DS: pd.to_datetime(["2024-01-08"] * 4),
                Y: [float("nan")] * 4,
                Y_HAT: [2.0, 3.0, 6.0, 6.0],
                H: [1] * 4,
                FORECAST_ORIGIN: pd.to_datetime(["2024-01-07"] * 4),
                MODEL_NAME: pd.Series(["m"] * 4, dtype="object"),
            }
        )
    )

    BackendEngine()._resolve_due(ledger, actuals, pd.Timestamp("2024-01-08"), None)
    resolved = ledger.to_df().set_index(UNIQUE_ID)

    assert resolved.loc["dept_id=D", Y] == pytest.approx(5.0)
    assert resolved.loc["__total__", Y] == pytest.approx(5.0)


def test_order_phase_filters_aggregate_rows_when_hierarchy_present(monkeypatch) -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A", "B"], "dept_id": ["D", "D"]})
    frame = pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(["A", "B", "dept_id=D", "__total__"], dtype="object"),
            DS: pd.to_datetime(["2024-01-08"] * 4),
            Y: [float("nan")] * 4,
            Y_HAT: [2.0, 3.0, 5.0, 5.0],
            H: [1] * 4,
            FORECAST_ORIGIN: pd.to_datetime(["2024-01-07"] * 4),
            MODEL_NAME: pd.Series(["m"] * 4, dtype="object"),
        }
    )
    seen: dict[str, list[str]] = {}

    def _fake_apply_order_policy(order_frame, config):
        del config
        seen["uids"] = order_frame[UNIQUE_ID].astype(str).tolist()
        return pd.DataFrame({UNIQUE_ID: seen["uids"], "order_qty": [0.0] * len(order_frame)})

    monkeypatch.setattr("calibre.execution.backend.apply_order_policy", _fake_apply_order_policy)
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(
            reconciler=NixtlaReconciler("bottom_up"), hierarchy=hierarchy
        ),
        order=RsConfig(params=pd.DataFrame()),
    )

    engine._order(frame, InMemoryOrderLedger())

    assert seen["uids"] == ["A", "B"]
