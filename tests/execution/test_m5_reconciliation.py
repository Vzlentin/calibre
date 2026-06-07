"""End-to-end reconciliation seam checks on the M5 fixture and the VN2 no-op.

The seam runs on real M5 hierarchy attributes and receives independently
forecast aggregate-node rows. It is also byte-identical on the
``hierarchy=None`` path VN2 takes, which is what protects the
``total_cost=4992.20`` benchmark gate (verified independently by
``tests/benchmarks/test_vn2_regression.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from calibre.cli.commands import _load_dataset
from calibre.cli.config import load_config_from_mapping
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import (
    BackendEngine,
    ConformalOptions,
    ExecutionOptions,
    ReconciliationOptions,
)
from calibre.execution.task_builder import build_node_history, build_tasks
from calibre.forecasting.adapter_base import ModelAdapter
from calibre.reconciliation import NixtlaReconciler, NoOpReconciler
from calibre.reconciliation.summing import build_summing_matrix

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"


class _CountingReconciler:
    """Wraps a reconciler and counts how many times the phase invoked it."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def __call__(self, frame: pd.DataFrame, hierarchy: pd.DataFrame | None) -> pd.DataFrame:
        self.calls += 1
        return self.inner(frame, hierarchy)


def _m5_bundle_tasks_origins():
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "execution": {"backend": "local", "seed": 42},
        }
    )
    bundle = _load_dataset(config)
    node_history = build_node_history(bundle.history, bundle.hierarchy)
    model_configs = [task.resolved_model_config() for task in config.tasks]
    tasks = build_tasks(node_history, model_configs, 1)
    origins = config.origins.to_list()
    return bundle, node_history, tasks, origins


def _run_m5(bundle, actuals, tasks, origins, reconciler) -> pd.DataFrame:
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        conformal=ConformalOptions(
            config=SymmetricIntervalConfig(
                method="aci", coverage=0.9, calibration_window=4, gamma=0.05
            )
        ),
        reconciliation=ReconciliationOptions(reconciler=reconciler, hierarchy=bundle.hierarchy),
    )
    try:
        result = engine.execute(tasks, actuals, origins)
    finally:
        engine.close()
    return result.ledger.to_df()


def _assert_node_rows_coherent(frame: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
    summing = build_summing_matrix(hierarchy)
    for _, group in frame.groupby([MODEL_NAME, FORECAST_ORIGIN, H], sort=False):
        values = group.set_index(UNIQUE_ID)[Y_HAT]
        bottom = values.reindex(summing.bottom_ids).to_numpy(dtype=np.float64)
        actual = values.reindex(summing.node_labels).to_numpy(dtype=np.float64)
        np.testing.assert_allclose(actual, summing.S @ bottom, rtol=1e-10, atol=1e-10)


class _NonAdditiveAdapter(ModelAdapter):
    """Deterministic test adapter whose aggregate forecasts are not additive."""

    def fit(self, task: ForecastTask) -> None:
        pass

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        last_ds = task.history[DS].max()
        yhat = float(task.history[Y].mean() ** 2)
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id],
                DS: [last_ds + pd.Timedelta(days=1)],
                Y_HAT: [yhat],
                H: [1],
            }
        )


def test_summing_matrix_built_from_real_m5_attributes() -> None:
    bundle, _actuals, _tasks, _origins = _m5_bundle_tasks_origins()
    summing = build_summing_matrix(bundle.hierarchy)
    # S is derived from the M5 attribute columns generically (R4).
    assert "cat_id=HOBBIES" in summing.node_labels
    assert "state_id=CA" in summing.node_labels
    assert "store_id=CA_1" in summing.node_labels
    assert summing.n_bottom == int(bundle.hierarchy[UNIQUE_ID].nunique())


def test_m5_ols_run_completes_and_reconcile_executes() -> None:
    bundle, actuals, tasks, origins = _m5_bundle_tasks_origins()
    spy = _CountingReconciler(NixtlaReconciler("ols"))
    ledger = _run_m5(bundle, actuals, tasks, origins, spy)
    # The reconcile phase actually ran on M5 (not silently skipped).
    assert spy.calls >= 1
    assert not ledger.empty
    _assert_node_rows_coherent(ledger, bundle.hierarchy)


def test_m5_bottom_yhat_identical_to_noop_run() -> None:
    """bottom_up keeps bottom y_hat unchanged while overwriting aggregates."""
    bundle, actuals, tasks, origins = _m5_bundle_tasks_origins()
    keys = [UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME, H]
    bottom_ids = set(build_summing_matrix(bundle.hierarchy).bottom_ids)

    none_run = _run_m5(bundle, actuals, tasks, origins, NoOpReconciler()).sort_values(keys)
    bottom_up_run = _run_m5(
        bundle, actuals, tasks, origins, NixtlaReconciler("bottom_up")
    ).sort_values(keys)
    none_bottom = none_run[none_run[UNIQUE_ID].isin(bottom_ids)]
    bottom_up_bottom = bottom_up_run[bottom_up_run[UNIQUE_ID].isin(bottom_ids)]

    np.testing.assert_array_equal(
        bottom_up_bottom[Y_HAT].to_numpy(),
        none_bottom[Y_HAT].to_numpy(),
    )
    _assert_node_rows_coherent(bottom_up_run, bundle.hierarchy)


def test_m5_conformal_interval_columns_present_after_reconcile() -> None:
    """Conformal still emits per-node marginal interval columns (R9)."""
    bundle, actuals, tasks, origins = _m5_bundle_tasks_origins()
    ledger = _run_m5(bundle, actuals, tasks, origins, NixtlaReconciler("ols"))
    lower_col, upper_col = interval_column_names(0.9)
    assert lower_col in ledger.columns
    assert upper_col in ledger.columns
    aggregate_ids = set(build_summing_matrix(bundle.hierarchy).node_labels) - set(
        build_summing_matrix(bundle.hierarchy).bottom_ids
    )
    assert not ledger[ledger[UNIQUE_ID].isin(aggregate_ids)].empty


def test_m5_ols_moves_bottom_forecasts_from_divergent_node_bases(monkeypatch) -> None:
    monkeypatch.setattr(
        "calibre.execution.backend.resolve_adapter",
        lambda model_config: _NonAdditiveAdapter(model_config),
    )
    bundle, actuals, tasks, origins = _m5_bundle_tasks_origins()
    summing = build_summing_matrix(bundle.hierarchy)
    keys = [UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME, H]

    base = _run_m5(bundle, actuals, tasks, origins, NoOpReconciler()).sort_values(keys)
    values = base.set_index(UNIQUE_ID)[Y_HAT]
    bottom_base = values.reindex(summing.bottom_ids).to_numpy(dtype=np.float64)
    node_base = values.reindex(summing.node_labels).to_numpy(dtype=np.float64)

    assert not np.allclose(summing.S @ bottom_base, node_base)

    ols = _run_m5(bundle, actuals, tasks, origins, NixtlaReconciler("ols")).sort_values(keys)
    ols_values = ols.set_index(UNIQUE_ID)[Y_HAT]
    ols_bottom = ols_values.reindex(summing.bottom_ids).to_numpy(dtype=np.float64)
    assert not np.allclose(ols_bottom, bottom_base)
    _assert_node_rows_coherent(ols, bundle.hierarchy)

    bottom_up = _run_m5(bundle, actuals, tasks, origins, NixtlaReconciler("bottom_up")).sort_values(
        keys
    )
    bottom_up_values = bottom_up.set_index(UNIQUE_ID)[Y_HAT]
    bottom_up_bottom = bottom_up_values.reindex(summing.bottom_ids).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(bottom_up_bottom, bottom_base, rtol=1e-10, atol=1e-10)
    _assert_node_rows_coherent(bottom_up, bundle.hierarchy)


def test_reconcile_byte_identical_when_hierarchy_none() -> None:
    """VN2 path: hierarchy=None makes the reconcile phase byte-identical (R11)."""
    preds = pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(["s1", "s2"], dtype="object"),
            "ds": pd.to_datetime(["2024-01-14", "2024-01-14"]),
            "y": np.array([np.nan, np.nan], dtype="float64"),
            Y_HAT: np.array([12.0, 7.0], dtype="float64"),
            H: np.array([1, 1], dtype="int64"),
            FORECAST_ORIGIN: pd.to_datetime(["2024-01-07", "2024-01-07"]),
            MODEL_NAME: pd.Series(["m", "m"], dtype="object"),
        }
    )
    engine = BackendEngine(
        reconciliation=ReconciliationOptions(reconciler=NixtlaReconciler("ols"), hierarchy=None)
    )
    out = engine._reconcile(preds)
    pd.testing.assert_frame_equal(out, preds)
