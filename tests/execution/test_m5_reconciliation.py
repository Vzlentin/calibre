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
    BackendResult,
    ConformalOptions,
    ExecutionOptions,
    HierarchicalIntervalEngineOptions,
    LedgerOutputOptions,
    ReconciliationOptions,
)
from calibre.execution.ledger import resolved_ledger_uri
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
    backend: str = "local",
    max_concurrency: int | None = None,
    cpu_per_task: float | None = None,
    output_path: Path | None = None,
) -> BackendResult:
    """Run the fused hierarchical-interval phase and return the ``BackendResult``.

    ``backend``/``max_concurrency``/``cpu_per_task`` drive serial-vs-parallel
    dispatch; ``output_path`` enables the streaming sink so a finalized
    ``.resolved.parquet`` artifact is produced. Returns the ``BackendResult`` so
    tests reach ``.ledger.to_df()`` explicitly.
    """
    output = (
        LedgerOutputOptions(forecast_path=str(output_path), streaming=True)
        if output_path is not None
        else LedgerOutputOptions()
    )
    engine = BackendEngine(
        execution=ExecutionOptions(
            freq="D",
            backend=backend,
            seed=42,
            max_concurrency=max_concurrency,
            cpu_per_task=cpu_per_task,
            ray_threshold=1,
        ),
        output=output,
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
        return engine.execute(tasks, actuals, origins)
    finally:
        engine.close()


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
    """Like ``_node_fitted_frame`` but one node has constant in-sample residuals.

    Zero unbiased variance is the degenerate case the sparse ``wls_var`` rejects.
    """
    frame = _node_fitted_frame(node_labels, periods)
    degenerate = node_labels[0]
    mask = frame[UNIQUE_ID] == degenerate
    # A constant residual of 1.0 for every period -> nanvar(ddof=1) == 0.0.
    frame.loc[mask, FITTED_Y_HAT] = frame.loc[mask, Y].to_numpy(dtype=np.float64) - 1.0
    return frame


def test_point_wls_var_rejects_zero_variance_node_the_dense_path_tolerated() -> None:
    """Sparse ``wls_var`` rejects a node with constant in-sample residuals.

    It weights by unbiased residual variance and requires it strictly positive
    per node, so such a node raises upstream's positive-definite error — surfaced
    with cross-section identity. The old dense jittered estimator tolerated this;
    the sparse path intentionally does not. This pins the documented divergence.
    """
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
    """The fused phase's reconciled mean matches the dense MinT closed form.

    MinTraceSparse + sparse S_df agree with the dense form within solver tolerance.
    """
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
    """Fused bottom_up reconciles to the aggregated bottom block.

    Through BottomUpSparse + sparse S_df, the reconciled mean is exactly the
    aggregated bottom block of the base forecasts.
    """
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
    # Two origins exercise the cross-origin summing-matrix cache READ (#218)
    # through the real 1e-10 coherence oracle below.
    origins = [pd.Timestamp("2011-03-15"), pd.Timestamp("2011-03-16")]
    lower_col, upper_col = interval_column_names(0.9)

    ledger = _run_m5_hierarchical_intervals(bundle, node_history, tasks, origins).ledger.to_df()

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


# ---------------------------------------------------------------------------
# Fused hierarchical-interval parallelization byte-identity gate (#219).
#
# The parallel path (Ray bounded sliding window over origins) MUST produce
# byte-for-byte identical ledger output to the serial path. The serial baseline
# is pinned to ``backend="ray", max_concurrency=1`` (window-of-1), NOT
# ``backend="local"`` — so the no-resort ``check_exact=True`` gate isolates the
# interval-dispatch window as the only variable and does not inherit a latent
# local-vs-ray Predict concat-order delta. The gate is parametrized over
# ``{bottom_up, wls_struct}``: ``wls_struct`` is dense BLAS, whose reductions are
# thread-count sensitive, so it exercises the serial-vs-worker thread-symmetry
# vector that ``bottom_up`` (sparse-sum, thread-invariant) would mask.
# ---------------------------------------------------------------------------

_FUSED_BYTE_STRATEGIES = ["bottom_up", "wls_struct"]
# Three consecutive origins at horizon 2 give a multi-origin due-window overlap:
# each origin's h=2 row becomes due one origin later, so ``_resolve_due``
# carry-forward (not just append order) is exercised across the window.
_FUSED_ORIGINS = [
    pd.Timestamp("2011-03-15"),
    pd.Timestamp("2011-03-16"),
    pd.Timestamp("2011-03-17"),
]


def _ca_subset_hierarchy(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """CA-only bottom rows — a smaller real-attribute hierarchy for the byte gate."""
    subset = hierarchy[hierarchy["state_id"] == "CA"].reset_index(drop=True)
    assert not subset.empty
    return subset


class _FusedBundle:
    """Minimal bundle stand-in exposing the ``.hierarchy`` the helper reads."""

    def __init__(self, hierarchy: pd.DataFrame) -> None:
        self.hierarchy = hierarchy


def _fused_residual_fixture():
    """CA-subset hierarchy + real-adapter tasks for the fused byte gate.

    The real statsforecast SeasonalNaive (no monkeypatched stub) is used so the
    Predict phase computes byte-identically on both the driver (``backend=local``)
    and a Ray worker — a driver-only ``resolve_adapter`` monkeypatch is invisible
    to the worker process and would make local and ray diverge. SeasonalNaive
    supplies the in-sample fitted values the fused phase requires natively.
    """
    full_hierarchy = _m5_hierarchy_frame()
    hierarchy = _ca_subset_hierarchy(full_hierarchy)
    bundle = _FusedBundle(hierarchy)
    node_history = _synthetic_m5_node_history(hierarchy)
    tasks = _synthetic_residual_tasks(node_history, horizon=2)
    return bundle, node_history, tasks


def _ledger_sha256(frame: pd.DataFrame) -> str:
    import hashlib

    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def _run_m5_hierarchical_intervals_resumed(
    bundle,
    actuals,
    tasks,
    initial_ledger: pd.DataFrame,
    *,
    strategy: str,
    max_concurrency: int,
) -> pd.DataFrame:
    """Resume a fused run from ``initial_ledger`` over the full origin set."""
    engine = BackendEngine(
        execution=ExecutionOptions(
            freq="D",
            backend="ray",
            seed=42,
            ray_threshold=1,
            max_concurrency=max_concurrency,
            cpu_per_task=1.0,
        ),
        reconciliation=ReconciliationOptions(
            reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(
            phase=NixtlaHierarchicalIntervalPhase(
                HierarchicalIntervalOptions(
                    method="nixtla_conformal", coverage=0.9, strategy=strategy
                )
            )
        ),
        conformal=ConformalOptions(initial_ledger=initial_ledger),
    )
    try:
        return engine.execute(tasks, actuals, _FUSED_ORIGINS).ledger.to_df()
    finally:
        engine.close()


@pytest.mark.parametrize("strategy", _FUSED_BYTE_STRATEGIES)
def test_fused_parallel_ledger_byte_identical_to_serial(strategy: str) -> None:
    """T1: N=2 parallel ledger is byte-identical to the window-of-1 serial baseline."""
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()

    serial = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy=strategy,
        backend="ray",
        max_concurrency=1,
        cpu_per_task=1.0,
    ).ledger.to_df()
    parallel = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy=strategy,
        backend="ray",
        max_concurrency=2,
        cpu_per_task=1.0,
    ).ledger.to_df()

    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    assert _ledger_sha256(serial) == _ledger_sha256(parallel)
    lower_col, upper_col = interval_column_names(0.9)
    np.testing.assert_array_equal(
        serial[[lower_col, upper_col]].to_numpy(), parallel[[lower_col, upper_col]].to_numpy()
    )


def test_fused_parallel_ledger_byte_identical_with_default_cpu_per_task() -> None:
    """T1c: byte-identity holds at the production default ``cpu_per_task=None``.

    The #133 run leaves ``cpu_per_task=None`` (so ``thread_budget(None) == 1``),
    unlike the other byte gates that pin ``cpu_per_task=1.0``. Gate the default so
    a regression in the unpinned BLAS-budget path can't slip the byte guarantee.
    """
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()

    serial = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=1,
        cpu_per_task=None,
    ).ledger.to_df()
    parallel = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=2,
        cpu_per_task=None,
    ).ledger.to_df()

    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    assert _ledger_sha256(serial) == _ledger_sha256(parallel)


def test_fused_local_vs_ray_sorted_sanity() -> None:
    """T1b: local-vs-(ray window-of-1) equality up to a sort (separate concern)."""
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()
    keys = [UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME, H]

    local = _run_m5_hierarchical_intervals(
        bundle, node_history, tasks, _FUSED_ORIGINS, strategy="wls_struct", backend="local"
    ).ledger.to_df()
    ray_serial = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=1,
        cpu_per_task=1.0,
    ).ledger.to_df()

    pd.testing.assert_frame_equal(
        local.sort_values(keys).reset_index(drop=True),
        ray_serial.sort_values(keys).reset_index(drop=True),
    )


@pytest.mark.parametrize("strategy", _FUSED_BYTE_STRATEGIES)
def test_fused_parallel_resolved_parquet_byte_identical(strategy: str, tmp_path: Path) -> None:
    """T2: finalized streamed ``.resolved.parquet`` is byte-identical serial-vs-parallel."""
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()
    serial_path = tmp_path / "serial.parquet"
    parallel_path = tmp_path / "parallel.parquet"

    _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy=strategy,
        backend="ray",
        max_concurrency=1,
        cpu_per_task=1.0,
        output_path=serial_path,
    )
    _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy=strategy,
        backend="ray",
        max_concurrency=2,
        cpu_per_task=1.0,
        output_path=parallel_path,
    )

    serial_resolved = resolved_ledger_uri(serial_path)
    parallel_resolved = resolved_ledger_uri(parallel_path)
    assert Path(serial_resolved).read_bytes() == Path(parallel_resolved).read_bytes()


def test_fused_parallel_vs_parallel_determinism() -> None:
    """T3: two independent N=2 wls_struct runs are byte-identical (dense BLAS)."""
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()

    first = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=2,
        cpu_per_task=1.0,
    ).ledger.to_df()
    second = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=2,
        cpu_per_task=1.0,
    ).ledger.to_df()

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert _ledger_sha256(first) == _ledger_sha256(second)


def test_fused_window_commits_in_origin_order_under_out_of_order_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4: consumer commits head-first even when workers finish out of order.

    Stub ``_dispatch_origin_intervals`` to return refs whose ``ray.get`` would
    "complete" in reverse order; the consumer must still append origins in
    origins-list order, and origin t+1's due frame must contain exactly origin
    t's appended-then-due rows (carry-forward), not just final-frame order.
    """
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()

    serial = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="bottom_up",
        backend="ray",
        max_concurrency=1,
        cpu_per_task=1.0,
    ).ledger.to_df()

    committed_origins: list[pd.Timestamp] = []
    real_finish = BackendEngine._finish_origin

    def _spy_finish(self, ledger, order_ledger, actuals, origin, origin_preds, runtime):
        committed_origins.append(pd.Timestamp(origin))
        return real_finish(self, ledger, order_ledger, actuals, origin, origin_preds, runtime)

    monkeypatch.setattr(BackendEngine, "_finish_origin", _spy_finish)
    parallel = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="bottom_up",
        backend="ray",
        max_concurrency=3,
        cpu_per_task=1.0,
    ).ledger.to_df()

    # Commits fire strictly in origins-list order regardless of worker completion.
    assert committed_origins == _FUSED_ORIGINS
    # And carry-forward state is identical to the serial path.
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)


def test_fused_resume_happy_path_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5: resumed origins are never dispatched; resumed parallel == resumed serial."""
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()

    # Build the initial ledger covering the first two origins (the "done" prefix).
    prefix = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS[:2],
        strategy="wls_struct",
        backend="ray",
        max_concurrency=1,
        cpu_per_task=1.0,
    ).ledger.to_df()

    dispatched: list[pd.Timestamp] = []
    real_dispatch = BackendEngine._dispatch_origin_intervals

    def _spy_dispatch(self, origin_preds, context):
        # Record the origin via the predictions' forecast_origin column.
        dispatched.append(pd.Timestamp(origin_preds[FORECAST_ORIGIN].iloc[0]))
        return real_dispatch(self, origin_preds, context)

    def _run_resumed(backend_kwargs: dict) -> pd.DataFrame:
        engine = BackendEngine(
            execution=ExecutionOptions(
                freq="D", backend="ray", seed=42, ray_threshold=1, **backend_kwargs
            ),
            reconciliation=ReconciliationOptions(
                reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
            ),
            hierarchical_intervals=HierarchicalIntervalEngineOptions(
                phase=NixtlaHierarchicalIntervalPhase(
                    HierarchicalIntervalOptions(
                        method="nixtla_conformal", coverage=0.9, strategy="wls_struct"
                    )
                )
            ),
            conformal=ConformalOptions(initial_ledger=prefix),
        )
        try:
            return engine.execute(tasks, node_history, _FUSED_ORIGINS).ledger.to_df()
        finally:
            engine.close()

    resumed_serial = _run_resumed({"max_concurrency": 1, "cpu_per_task": 1.0})
    monkeypatch.setattr(BackendEngine, "_dispatch_origin_intervals", _spy_dispatch)
    resumed_parallel = _run_resumed({"max_concurrency": 2, "cpu_per_task": 1.0})

    # Completed origins (the first two) are never dispatched — only the third.
    assert _FUSED_ORIGINS[0] not in dispatched
    assert _FUSED_ORIGINS[1] not in dispatched
    assert _FUSED_ORIGINS[2] in dispatched
    pd.testing.assert_frame_equal(resumed_serial, resumed_parallel, check_exact=True)


def test_fused_max_concurrency_one_collapses_to_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T6: ``max_concurrency=1`` fused-ray run takes the serial path (window-of-1)."""
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()

    # The window-of-1 collapse: _fused_parallel_window returns None at N==1, so the
    # serial generator drives the loop (no interval task is ever dispatched).
    dispatched = 0
    real_dispatch = BackendEngine._dispatch_origin_intervals

    def _counting_dispatch(self, origin_preds, context):
        nonlocal dispatched
        dispatched += 1
        return real_dispatch(self, origin_preds, context)

    monkeypatch.setattr(BackendEngine, "_dispatch_origin_intervals", _counting_dispatch)
    serial_local = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=1,
        cpu_per_task=1.0,
    ).ledger.to_df()

    assert dispatched == 0  # N==1 path never dispatches an interval task
    assert not serial_local.empty


def test_fused_window_holds_more_than_one_task_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T6b: an N=2 run over 3 origins genuinely keeps >1 interval task in flight.

    Positive proof the bounded window is NOT silently degenerating to serial: the
    byte gates would all still pass at peak-in-flight==1. Track the live count of
    tasks dispatched-but-not-yet-consumed (the in-flight deque depth) by
    incrementing on :meth:`_dispatch_origin_intervals` and decrementing on
    :meth:`_get_origin_intervals`, and assert the high-water mark reaches the
    window depth. Forcing the window to serial drops the peak to 0 and fails this.
    """
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()

    in_flight = 0
    peak_in_flight = 0
    real_dispatch = BackendEngine._dispatch_origin_intervals
    real_get = BackendEngine._get_origin_intervals  # already a plain function (staticmethod)

    def _tracking_dispatch(self, origin_preds, context):
        nonlocal in_flight, peak_in_flight
        ref = real_dispatch(self, origin_preds, context)
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        return ref

    def _tracking_get(ref):
        nonlocal in_flight
        try:
            return real_get(ref)
        finally:
            in_flight -= 1

    monkeypatch.setattr(BackendEngine, "_dispatch_origin_intervals", _tracking_dispatch)
    monkeypatch.setattr(BackendEngine, "_get_origin_intervals", staticmethod(_tracking_get))

    ledger = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=2,
        cpu_per_task=1.0,
    ).ledger.to_df()

    assert not ledger.empty
    # Window depth is 2 and there are 3 origins, so the producer fills the window
    # before the first drain — the deque must hold 2 tasks at peak.
    assert peak_in_flight >= 2


def test_fused_producer_failure_resumes_with_identical_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T7: a worker failing on origin i commits 0..i-1, re-raises named, resumes clean."""
    pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    bundle, node_history, tasks = _fused_residual_fixture()
    fail_origin = _FUSED_ORIGINS[1]

    # Fail the SECOND origin's interval task. Origin 0 must commit first; origin 1's
    # ray.get raises, surfaced named-by-origin via the HierarchicalIntervals _phase.
    real_dispatch = BackendEngine._dispatch_origin_intervals

    def _maybe_failing_dispatch(self, origin_preds, context):
        origin = pd.Timestamp(origin_preds[FORECAST_ORIGIN].iloc[0])
        if origin == fail_origin:
            import ray

            @ray.remote
            def _boom() -> pd.DataFrame:
                raise RuntimeError("injected interval failure")

            return _boom.remote()
        return real_dispatch(self, origin_preds, context)

    monkeypatch.setattr(BackendEngine, "_dispatch_origin_intervals", _maybe_failing_dispatch)

    engine = BackendEngine(
        execution=ExecutionOptions(
            freq="D", backend="ray", seed=42, ray_threshold=1, max_concurrency=3, cpu_per_task=1.0
        ),
        reconciliation=ReconciliationOptions(
            reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(
            phase=NixtlaHierarchicalIntervalPhase(
                HierarchicalIntervalOptions(
                    method="nixtla_conformal", coverage=0.9, strategy="wls_struct"
                )
            )
        ),
    )
    last_ledger: pd.DataFrame | None = None
    try:
        with pytest.raises(RuntimeError, match=r"HierarchicalIntervals phase failed at origin"):
            for result in engine.iter_origins(tasks, node_history, _FUSED_ORIGINS):
                # Each yield carries the cumulative ledger; the last one before the
                # raise is origin 0's committed state.
                last_ledger = result.ledger.to_df()
    finally:
        engine.close()

    assert last_ledger is not None
    committed = {pd.Timestamp(o) for o in last_ledger[FORECAST_ORIGIN].unique()}
    # Origin 0 committed; the failing origin and beyond never reached the ledger.
    assert _FUSED_ORIGINS[0] in committed
    assert fail_origin not in committed
    assert _FUSED_ORIGINS[2] not in committed

    # A resumed run (with origin 0 done) completes byte-identically to a clean run.
    # Undo the failing-dispatch patch so the resume uses the real worker path.
    monkeypatch.undo()
    resumed = _run_m5_hierarchical_intervals_resumed(
        bundle, node_history, tasks, last_ledger, strategy="wls_struct", max_concurrency=2
    )
    clean = _run_m5_hierarchical_intervals(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        strategy="wls_struct",
        backend="ray",
        max_concurrency=1,
        cpu_per_task=1.0,
    ).ledger.to_df()
    # Sort is structural, not a loosened byte gate: resume prepends initial_ledger
    # (origin 0's committed rows) so the resumed frame's row ORDER differs from a
    # clean run. This asserts the resumed run reproduces the same ROWSET; the
    # window-vs-serial byte-identity guarantee is pinned by the T1/T4/T5 sha256
    # gates, not here.
    keys = [UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME, H]
    pd.testing.assert_frame_equal(
        resumed.sort_values(keys).reset_index(drop=True),
        clean.sort_values(keys).reset_index(drop=True),
    )


def test_get_origin_intervals_reraises_wrapper_when_cause_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T7b: a RayTaskError with no ``cause`` is re-raised intact, not unwrapped.

    The unwrap path re-raises ``exc.cause`` so ``_phase`` can name the origin, but
    a wrapper with ``cause is None`` has nothing to unwrap — assert the original
    ``RayTaskError`` propagates unchanged (the :func:`backend._get_origin_intervals`
    fallback ``raise``).
    """
    ray = pytest.importorskip("ray")
    from ray.exceptions import RayTaskError

    # A RayTaskError whose .cause we force to None (the unwrappable-cause branch).
    wrapper = RayTaskError("compute_origin_intervals", "traceback-string", cause=RuntimeError("x"))
    wrapper.cause = None

    def _raise_wrapper(_ref: object) -> object:
        raise wrapper

    monkeypatch.setattr(ray, "get", _raise_wrapper)
    with pytest.raises(RayTaskError) as excinfo:
        BackendEngine._get_origin_intervals(object())
    assert excinfo.value is wrapper


def test_fused_cache_and_ref_rebuild_after_owned_runtime_shutdown() -> None:
    """T8: a fresh runtime re-``ray.put``s the index and rebuilds the remote handle."""
    ray = pytest.importorskip("ray")
    pytest.importorskip("hierarchicalforecast.core")
    if ray.is_initialized():
        ray.shutdown()
    bundle, node_history, tasks = _fused_residual_fixture()

    results: list[pd.DataFrame] = []
    for _ in range(2):
        engine = BackendEngine(
            execution=ExecutionOptions(
                freq="D",
                backend="ray",
                seed=42,
                ray_threshold=1,
                max_concurrency=2,
                cpu_per_task=1.0,
            ),
            reconciliation=ReconciliationOptions(
                reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
            ),
            hierarchical_intervals=HierarchicalIntervalEngineOptions(
                phase=NixtlaHierarchicalIntervalPhase(
                    HierarchicalIntervalOptions(
                        method="nixtla_conformal", coverage=0.9, strategy="wls_struct"
                    )
                )
            ),
        )
        try:
            result = engine.execute(tasks, node_history, _FUSED_ORIGINS)
            # The owned runtime cached a fresh index ref this acquisition.
            assert engine._hierarchy_index_ref is not None  # noqa: SLF001
            results.append(result.ledger.to_df())
        finally:
            engine.close()
            # Shutdown nulls the cached ref/handle so the next run rebuilds them.
            assert engine._hierarchy_index_ref is None  # noqa: SLF001
            assert engine._remote_compute_origin_intervals is None  # noqa: SLF001
        if ray.is_initialized():
            ray.shutdown()

    pd.testing.assert_frame_equal(results[0], results[1], check_exact=True)


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
