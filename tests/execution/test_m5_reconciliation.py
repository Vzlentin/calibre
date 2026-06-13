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
import pytest

from calibre.cli.commands import _load_dataset
from calibre.cli.config import load_config_from_mapping
from calibre.conformal.runtime import SymmetricIntervalConfig
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
from calibre.evaluation.forecast_metrics import compute_metrics, resolve_actuals
from calibre.evaluation.point_metrics import mae
from calibre.execution.backend import (
    BackendEngine,
    ConformalOptions,
    ExecutionOptions,
    HierarchicalIntervalEngineOptions,
    ReconciliationOptions,
)
from calibre.execution.task_builder import build_node_history, build_tasks
from calibre.forecasting.adapter_base import ModelAdapter, build_fitted_values_frame
from calibre.ordering.policy_config import RsConfig
from calibre.reconciliation import (
    BottomUpReconciler,
    HierarchicalIntervalContext,
    HierarchicalIntervalOptions,
    NixtlaHierarchicalIntervalPhase,
    NixtlaReconciler,
    NoOpReconciler,
    ReconciliationContext,
)
from calibre.reconciliation.summing import build_hierarchy_index, build_summing_matrix
from tests.infra import closed_form_min_trace

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"


class _CountingReconciler:
    """Wraps a reconciler and counts how many times the phase invoked it."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0
        self.requires_fitted_values = inner.requires_fitted_values

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        self.calls += 1
        return self.inner(frame, hierarchy, context)


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
    node_history = build_node_history(bundle.history, build_hierarchy_index(bundle.hierarchy))
    model_configs = [task.resolved_model_config() for task in config.tasks]
    tasks = build_tasks(node_history, model_configs, 1)
    origins = config.origins.to_list()
    return bundle, node_history, tasks, origins


def _m5_bottom_tasks(bundle):
    """Bottom-only tasks for the native bottom_up path (no aggregate tasks)."""
    return build_tasks(
        bundle.history,
        [
            {
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "name": "SeasonalNaive",
                "season_length": 7,
            }
        ],
        1,
    )


def _run_m5(bundle, actuals, tasks, origins, reconciler) -> pd.DataFrame:
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        conformal=ConformalOptions(
            config=SymmetricIntervalConfig(
                method="aci", coverage=0.9, calibration_window=4, gamma=0.05
            )
        ),
        reconciliation=ReconciliationOptions(
            reconciler=reconciler, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
    )
    try:
        result = engine.execute(tasks, actuals, origins)
    finally:
        engine.close()
    return result.ledger.to_df()


def _run_m5_hierarchical_intervals(
    bundle,
    actuals,
    tasks,
    origins,
    *,
    strategy: str = "bottom_up",
) -> pd.DataFrame:
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        reconciliation=ReconciliationOptions(
            reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(
            phase=NixtlaHierarchicalIntervalPhase(
                HierarchicalIntervalOptions(
                    method="nixtla_conformal",
                    coverage=0.9,
                    strategy=strategy,
                )
            )
        ),
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

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del collect_fitted_values
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


class _ResidualAdapter(ModelAdapter):
    """Deterministic adapter with non-additive in-sample residuals per node."""

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del task, collect_fitted_values

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        uid = str(task.history[UNIQUE_ID].iloc[0])
        node_offset = float(sum(ord(char) for char in uid) % 17)
        last_ds = task.history[DS].max()
        base = float(task.history[Y].mean() + 0.3 * node_offset)
        return pd.DataFrame(
            [
                {
                    UNIQUE_ID: uid,
                    DS: last_ds + pd.Timedelta(days=h),
                    Y_HAT: base + 0.1 * h,
                    H: h,
                }
                for h in range(1, task.horizon + 1)
            ]
        )

    def fitted_values(self, task: ForecastTask) -> pd.DataFrame:
        uid = str(task.history[UNIQUE_ID].iloc[0])
        node_offset = float(sum(ord(char) for char in uid) % 17)
        raw = task.history[[UNIQUE_ID, DS, Y]].copy()
        t = np.arange(len(raw), dtype=np.float64)
        residual = (
            np.sin(t * (0.13 + node_offset * 0.01))
            + np.cos(t * (0.07 + node_offset * 0.02))
            + node_offset * 0.05
        )
        raw[FITTED_Y_HAT] = raw[Y].to_numpy(dtype=np.float64) - residual
        return build_fitted_values_frame(raw, model_name=task.model_name)


def _synthetic_m5_node_history(hierarchy: pd.DataFrame) -> pd.DataFrame:
    bottom_ids = hierarchy[UNIQUE_ID].astype(str).tolist()
    dates = pd.date_range("2011-01-01", periods=80, freq="D")
    rows = [
        {
            UNIQUE_ID: uid,
            DS: ds,
            Y: float((idx + 1) * 2 + (step % 7) + 0.1 * step),
        }
        for idx, uid in enumerate(bottom_ids)
        for step, ds in enumerate(dates)
    ]
    return build_node_history(pd.DataFrame(rows), build_hierarchy_index(hierarchy))


def _synthetic_residual_tasks(node_history: pd.DataFrame, *, horizon: int = 1):
    return build_tasks(
        node_history,
        [
            {
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "name": "residual_stub",
                "season_length": 7,
            }
        ],
        horizon,
    )


# ---------------------------------------------------------------------------
# Sparse-vs-dense agreement pins (#168). The expectations are the dense MinT
# closed form S (S' W^-1 S)^-1 S' W^-1 y, captured against the dense path and
# held within a bicgstab-generous tolerance so the same pins prove the sparse
# MinTraceSparse path agrees with the old dense closed-form results.
# ---------------------------------------------------------------------------

# Generous enough for an iterative bicgstab solve (atol=1e-5 on the normal
# equations), deliberately not float-roundoff tight; a wrong weight vector or
# projection produces O(1) relative errors and still fails these pins.
_SOLVER_RTOL = 1e-3
_SOLVER_ATOL = 1e-6


def _divergent_node_values(summing) -> np.ndarray:
    """A deliberately incoherent node vector aligned to ``summing.node_labels``."""
    rng = np.random.default_rng(7)
    bottom = rng.uniform(5.0, 20.0, size=summing.n_bottom)
    values = np.empty(summing.n_nodes, dtype=np.float64)
    values[: summing.n_bottom] = bottom
    coherent_aggregates = summing.S[summing.n_bottom :] @ bottom
    perturbation = rng.uniform(0.8, 1.2, size=coherent_aggregates.size)
    values[summing.n_bottom :] = coherent_aggregates * perturbation + 1.0
    return values


def _m5_hierarchy_frame() -> pd.DataFrame:
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
    assert bundle.hierarchy is not None
    return bundle.hierarchy


def _node_point_frame(node_labels: tuple[str, ...], values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(list(node_labels), dtype="object"),
            DS: pd.to_datetime(["2011-02-01"] * len(node_labels)),
            Y: np.full(len(node_labels), np.nan, dtype=np.float64),
            Y_HAT: np.asarray(values, dtype=np.float64),
            H: np.ones(len(node_labels), dtype=np.int64),
            FORECAST_ORIGIN: pd.to_datetime(["2011-01-31"] * len(node_labels)),
            MODEL_NAME: pd.Series(["m"] * len(node_labels), dtype="object"),
        }
    )


def _node_residual_matrix(n_nodes: int, periods: int) -> np.ndarray:
    """Deterministic per-node residuals with clearly nonzero means and variances."""
    t = np.arange(periods, dtype=np.float64)
    node = np.arange(n_nodes, dtype=np.float64)[:, None]
    return 0.5 + 0.1 * node + 0.3 * np.sin(t[None, :] + node)


def _node_fitted_frame(node_labels: tuple[str, ...], periods: int = 10) -> pd.DataFrame:
    residuals = _node_residual_matrix(len(node_labels), periods)
    dates = pd.date_range("2011-01-10", periods=periods, freq="D")
    rows = []
    for node_idx, label in enumerate(node_labels):
        for step, ds in enumerate(dates):
            y = 10.0 + node_idx + step
            rows.append(
                {
                    UNIQUE_ID: label,
                    DS: ds,
                    Y: y,
                    MODEL_NAME: "m",
                    FITTED_Y_HAT: y - residuals[node_idx, step],
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("strategy", ["ols", "wls_struct"])
def test_point_min_trace_agrees_with_dense_closed_form(strategy: str) -> None:
    pytest.importorskip("hierarchicalforecast.methods")
    hierarchy = _m5_hierarchy_frame()
    summing = build_summing_matrix(hierarchy)
    base = _divergent_node_values(summing)
    w_diag = (
        np.ones(summing.n_nodes) if strategy == "ols" else summing.S @ np.ones(summing.n_bottom)
    )
    expected = closed_form_min_trace(summing.S, w_diag, base)

    out = NixtlaReconciler(strategy)(
        _node_point_frame(summing.node_labels, base),
        build_hierarchy_index(hierarchy),
        ReconciliationContext(),
    )

    values = out.set_index(UNIQUE_ID)[Y_HAT].reindex(summing.node_labels).to_numpy(np.float64)
    np.testing.assert_allclose(values, expected, rtol=_SOLVER_RTOL, atol=_SOLVER_ATOL)


def test_point_wls_var_agrees_with_dense_closed_form() -> None:
    pytest.importorskip("hierarchicalforecast.methods")
    hierarchy = _m5_hierarchy_frame()
    summing = build_summing_matrix(hierarchy)
    base = _divergent_node_values(summing)
    periods = 10
    residuals = _node_residual_matrix(summing.n_nodes, periods)
    # MinTraceSparse wls_var weights: unbiased residual variance per node
    # (nanvar, ddof=1). This is a documented estimator change from the dense
    # MinTrace path, which weighted by the mean squared residual + 2e-8
    # jitter â€” at this fixture the two estimators diverge by ~1.5% in the
    # reconciled values, well past the solver tolerance, so this pin proves
    # the sparse path carries the upstream sparse estimator.
    w_diag = residuals.var(axis=1, ddof=1)
    expected = closed_form_min_trace(summing.S, w_diag, base)

    out = NixtlaReconciler("wls_var")(
        _node_point_frame(summing.node_labels, base),
        build_hierarchy_index(hierarchy),
        ReconciliationContext(fitted_values=_node_fitted_frame(summing.node_labels, periods)),
    )

    values = out.set_index(UNIQUE_ID)[Y_HAT].reindex(summing.node_labels).to_numpy(np.float64)
    np.testing.assert_allclose(values, expected, rtol=_SOLVER_RTOL, atol=_SOLVER_ATOL)


def _zero_variance_node_fitted_frame(
    node_labels: tuple[str, ...], periods: int = 10
) -> pd.DataFrame:
    """Like ``_node_fitted_frame`` but one node has CONSTANT in-sample residuals
    (zero unbiased variance), the degenerate case the sparse ``wls_var`` rejects."""
    frame = _node_fitted_frame(node_labels, periods)
    degenerate = node_labels[0]
    mask = frame[UNIQUE_ID] == degenerate
    # A constant residual of 1.0 for every period -> nanvar(ddof=1) == 0.0.
    frame.loc[mask, FITTED_Y_HAT] = frame.loc[mask, Y].to_numpy(dtype=np.float64) - 1.0
    return frame


def test_point_wls_var_rejects_zero_variance_node_the_dense_path_tolerated() -> None:
    """Characterization (#168): sparse ``wls_var`` weights by unbiased residual
    variance and requires it strictly positive per node, so a node with constant
    in-sample residuals raises upstream's positive-definite error — surfaced WITH
    cross-section identity. The old dense jittered estimator tolerated this; the
    sparse path intentionally does not. This pins the documented divergence."""
    pytest.importorskip("hierarchicalforecast.methods")
    hierarchy = _m5_hierarchy_frame()
    summing = build_summing_matrix(hierarchy)
    base = _divergent_node_values(summing)

    with pytest.raises(RuntimeError, match=r"strategy='wls_var'.*model_name="):
        NixtlaReconciler("wls_var")(
            _node_point_frame(summing.node_labels, base),
            build_hierarchy_index(hierarchy),
            ReconciliationContext(
                fitted_values=_zero_variance_node_fitted_frame(summing.node_labels)
            ),
        )


@pytest.mark.parametrize("strategy", ["ols", "wls_struct"])
def test_fused_min_trace_point_output_agrees_with_dense_closed_form(strategy: str) -> None:
    """The fused phase's reconciled mean through MinTraceSparse + sparse S_df
    matches the dense MinT closed form within solver tolerance (#168)."""
    pytest.importorskip("hierarchicalforecast.core")
    hierarchy = _m5_hierarchy_frame()
    summing = build_summing_matrix(hierarchy)
    base = _divergent_node_values(summing)
    w_diag = (
        np.ones(summing.n_nodes) if strategy == "ols" else summing.S @ np.ones(summing.n_bottom)
    )
    expected = closed_form_min_trace(summing.S, w_diag, base)

    phase = NixtlaHierarchicalIntervalPhase(
        HierarchicalIntervalOptions(method="nixtla_conformal", coverage=0.9, strategy=strategy)
    )
    out = phase.apply(
        _node_point_frame(summing.node_labels, base),
        build_hierarchy_index(hierarchy),
        HierarchicalIntervalContext(fitted_values=_node_fitted_frame(summing.node_labels)),
    )

    values = out.set_index(UNIQUE_ID)[Y_HAT].reindex(summing.node_labels).to_numpy(np.float64)
    np.testing.assert_allclose(values, expected, rtol=_SOLVER_RTOL, atol=_SOLVER_ATOL)


def test_fused_bottom_up_point_output_equals_bottom_sums() -> None:
    """Fused bottom_up through BottomUpSparse + sparse S_df: the reconciled
    mean is exactly the aggregated bottom block of the base forecasts."""
    pytest.importorskip("hierarchicalforecast.core")
    hierarchy = _m5_hierarchy_frame()
    summing = build_summing_matrix(hierarchy)
    base = _divergent_node_values(summing)
    expected = summing.S @ base[: summing.n_bottom]

    phase = NixtlaHierarchicalIntervalPhase(
        HierarchicalIntervalOptions(method="nixtla_conformal", coverage=0.9, strategy="bottom_up")
    )
    out = phase.apply(
        _node_point_frame(summing.node_labels, base),
        build_hierarchy_index(hierarchy),
        HierarchicalIntervalContext(fitted_values=_node_fitted_frame(summing.node_labels)),
    )

    values = out.set_index(UNIQUE_ID)[Y_HAT].reindex(summing.node_labels).to_numpy(np.float64)
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-10)


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
    """Native bottom_up keeps bottom y_hat unchanged while synthesizing aggregates."""
    bundle, actuals, tasks, origins = _m5_bundle_tasks_origins()
    keys = [UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME, H]
    bottom_ids = set(build_summing_matrix(bundle.hierarchy).bottom_ids)

    none_run = _run_m5(bundle, actuals, tasks, origins, NoOpReconciler()).sort_values(keys)
    bottom_up_run = _run_m5(
        bundle, actuals, _m5_bottom_tasks(bundle), origins, BottomUpReconciler()
    ).sort_values(keys)
    none_bottom = none_run[none_run[UNIQUE_ID].isin(bottom_ids)]
    bottom_up_bottom = bottom_up_run[bottom_up_run[UNIQUE_ID].isin(bottom_ids)]

    np.testing.assert_array_equal(
        bottom_up_bottom[Y_HAT].to_numpy(),
        none_bottom[Y_HAT].to_numpy(),
    )
    # Aggregate rows exist in the ledger even though no aggregate task ran.
    assert not bottom_up_run[~bottom_up_run[UNIQUE_ID].isin(bottom_ids)].empty
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


@pytest.mark.parametrize("strategy", ["ols", "wls_struct"])
def test_m5_min_trace_moves_bottom_forecasts_from_divergent_node_bases(
    monkeypatch, strategy: str
) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
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

    reconciled = _run_m5(bundle, actuals, tasks, origins, NixtlaReconciler(strategy)).sort_values(
        keys
    )
    reconciled_values = reconciled.set_index(UNIQUE_ID)[Y_HAT]
    reconciled_bottom = reconciled_values.reindex(summing.bottom_ids).to_numpy(dtype=np.float64)
    assert not np.allclose(reconciled_bottom, bottom_base)
    _assert_node_rows_coherent(reconciled, bundle.hierarchy)

    bottom_up = _run_m5(
        bundle, actuals, _m5_bottom_tasks(bundle), origins, BottomUpReconciler()
    ).sort_values(keys)
    bottom_up_values = bottom_up.set_index(UNIQUE_ID)[Y_HAT]
    bottom_up_bottom = bottom_up_values.reindex(summing.bottom_ids).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(bottom_up_bottom, bottom_base, rtol=1e-10, atol=1e-10)
    _assert_node_rows_coherent(bottom_up, bundle.hierarchy)


@pytest.mark.parametrize("strategy", ["mint_shrink", "wls_var", "erm"])
def test_m5_residual_strategies_return_coherent_multi_horizon_forecasts(
    monkeypatch, strategy: str
) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _ResidualAdapter(model_config),
    )
    bundle, _actuals, _tasks, _origins = _m5_bundle_tasks_origins()
    node_history = _synthetic_m5_node_history(bundle.hierarchy)
    tasks = _synthetic_residual_tasks(node_history, horizon=2)
    origins = [pd.Timestamp("2011-03-15")]

    ledger = _run_m5(bundle, node_history, tasks, origins, NixtlaReconciler(strategy))

    assert not ledger.empty
    assert set(ledger[H]) == {1, 2}
    _assert_node_rows_coherent(ledger, bundle.hierarchy)


def test_m5_hierarchical_conformal_intervals_emit_node_bounds_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("hierarchicalforecast.core")
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _ResidualAdapter(model_config),
    )
    bundle, _actuals, _tasks, _origins = _m5_bundle_tasks_origins()
    node_history = _synthetic_m5_node_history(bundle.hierarchy)
    tasks = _synthetic_residual_tasks(node_history, horizon=2)
    origins = [pd.Timestamp("2011-03-15")]
    lower_col, upper_col = interval_column_names(0.9)

    ledger = _run_m5_hierarchical_intervals(bundle, node_history, tasks, origins)

    assert {lower_col, upper_col}.issubset(ledger.columns)
    summing = build_summing_matrix(bundle.hierarchy)
    assert set(ledger[UNIQUE_ID]) == set(summing.node_labels)
    _assert_node_rows_coherent(ledger, bundle.hierarchy)

    width_gaps: list[float] = []
    for _, group in ledger.groupby([MODEL_NAME, FORECAST_ORIGIN, H], sort=False):
        by_uid = group.set_index(UNIQUE_ID)
        parent_width = float(
            by_uid.loc["__total__", upper_col] - by_uid.loc["__total__", lower_col]
        )
        bottom_bounds = by_uid.loc[list(summing.bottom_ids), [lower_col, upper_col]]
        child_width_sum = float((bottom_bounds[upper_col] - bottom_bounds[lower_col]).sum())
        width_gaps.append(abs(parent_width - child_width_sum))
    assert any(gap > 1e-9 for gap in width_gaps)

    resolved, _new = resolve_actuals(ledger, node_history, pd.Timestamp("2011-03-17"))
    metrics = compute_metrics(
        resolved,
        metrics=[mae],
        group_by=[UNIQUE_ID],
        interval_bounds=(lower_col, upper_col),
    )
    assert {"coverage", "mean_interval_width"}.issubset(metrics.columns)


def test_hierarchical_interval_ordering_uses_bottom_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _ResidualAdapter(model_config),
    )
    seen: dict[str, list[str]] = {}

    def _fake_apply_order_policy(order_frame, config):
        del config
        seen["uids"] = order_frame[UNIQUE_ID].astype(str).tolist()
        return pd.DataFrame({UNIQUE_ID: seen["uids"], "order_qty": [0.0] * len(order_frame)})

    monkeypatch.setattr("calibre.execution.backend.apply_order_policy", _fake_apply_order_policy)
    bundle, _actuals, _tasks, _origins = _m5_bundle_tasks_origins()
    node_history = _synthetic_m5_node_history(bundle.hierarchy)
    tasks = _synthetic_residual_tasks(node_history, horizon=1)
    phase = NixtlaHierarchicalIntervalPhase(
        HierarchicalIntervalOptions(method="nixtla_conformal", coverage=0.9, strategy="bottom_up")
    )
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        reconciliation=ReconciliationOptions(
            hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(phase=phase),
        order=RsConfig(params=pd.DataFrame()),
    )
    try:
        engine.execute(tasks, node_history, [pd.Timestamp("2011-03-15")])
    finally:
        engine.close()

    assert seen["uids"] == list(build_summing_matrix(bundle.hierarchy).bottom_ids)


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
        reconciliation=ReconciliationOptions(
            reconciler=NixtlaReconciler("ols"), hierarchy_index=None
        )
    )
    out = engine._reconcile(preds, ReconciliationContext())
    pd.testing.assert_frame_equal(out, preds)
