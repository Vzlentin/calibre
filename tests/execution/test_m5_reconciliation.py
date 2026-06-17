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
    CONFORMAL_METHOD,
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
from calibre.execution.actuals import HierarchyActualsSource
from calibre.execution.backend import (
    BackendEngine,
    BackendResult,
    ConformalOptions,
    ExecutionOptions,
    LedgerOutputOptions,
    ReconciliationOptions,
)
from calibre.execution.ledger import resolved_ledger_uri
from calibre.execution.task_builder import build_node_history, build_tasks
from calibre.forecasting.adapter_base import ModelAdapter, build_fitted_values_frame
from calibre.reconciliation import (
    BottomUpReconciler,
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


def _run_m5_coherent(bundle, actuals, tasks, origins) -> pd.DataFrame:
    """Run the coherent-draws conformal path (reconciliation: none)."""
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        conformal=ConformalOptions(
            config=SymmetricIntervalConfig(
                method="aci",
                coverage=0.9,
                calibration_window=3,
                gamma=0.05,
                spread="coherent_draws",
                draw_count=128,
                draw_seed=11,
            )
        ),
        reconciliation=ReconciliationOptions(
            reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
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


def test_m5_coherent_draws_emit_finite_node_bounds_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end: predict -> (reconciliation: none) -> CoherentDraws calibrate.
    # The coherent spread produces lo/hi for every node and the marginal
    # held-out calibrator gates emission once it is ready.
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _ResidualAdapter(model_config),
    )
    bundle, _actuals, _tasks, _origins = _m5_bundle_tasks_origins()
    node_history = _synthetic_m5_node_history(bundle.hierarchy)
    tasks = _synthetic_residual_tasks(node_history, horizon=2)
    origins = [pd.Timestamp(d) for d in ("2011-03-15", "2011-03-16", "2011-03-17", "2011-03-18")]
    lower_col, upper_col = interval_column_names(0.9)

    ledger = _run_m5_coherent(bundle, node_history, tasks, origins)

    assert not ledger.empty
    assert {lower_col, upper_col}.issubset(ledger.columns)
    assert ledger[CONFORMAL_METHOD].eq("aci").all()
    summing = build_summing_matrix(bundle.hierarchy)
    assert set(ledger[UNIQUE_ID]) == set(summing.node_labels)
    # Once the calibrator is ready, emitted bounds are finite and ordered.
    issued = ledger[ledger[lower_col].notna()]
    assert not issued.empty
    assert np.isfinite(issued[lower_col]).all()
    assert np.isfinite(issued[upper_col]).all()
    assert (issued[upper_col] >= issued[lower_col]).all()


def test_m5_coherent_draws_aggregate_brackets_member_center_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real-data smoke (NOT the coherence pin — that is the exact quantile-of-
    # summed-member-draws assertion in tests/conformal/test_coherent_draws.py):
    # on a real M5 run every aggregate node's emitted interval brackets the sum
    # of its member centers, the necessary consequence of reconciled draws being
    # centered on the summed member centers.
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _ResidualAdapter(model_config),
    )
    bundle, _actuals, _tasks, _origins = _m5_bundle_tasks_origins()
    node_history = _synthetic_m5_node_history(bundle.hierarchy)
    tasks = _synthetic_residual_tasks(node_history, horizon=1)
    origins = [pd.Timestamp(d) for d in ("2011-03-15", "2011-03-16", "2011-03-17", "2011-03-18")]
    lower_col, upper_col = interval_column_names(0.9)
    summing = build_summing_matrix(bundle.hierarchy)

    ledger = _run_m5_coherent(bundle, node_history, tasks, origins)
    issued = ledger[ledger[lower_col].notna()]
    assert not issued.empty

    # For each cross-section, the total interval must contain the sum of member
    # centers (the reconciled draws are centered on the summed member centers).
    checked = 0
    for _, group in issued.groupby([MODEL_NAME, FORECAST_ORIGIN, H], sort=False):
        by_uid = group.set_index(UNIQUE_ID)
        if "__total__" not in by_uid.index:
            continue
        present_bottom = [b for b in summing.bottom_ids if b in by_uid.index]
        if not present_bottom:
            continue
        member_center_sum = float(by_uid.loc[present_bottom, Y_HAT].sum())
        total_lo = float(by_uid.loc["__total__", lower_col])
        total_hi = float(by_uid.loc["__total__", upper_col])
        assert total_lo <= member_center_sum <= total_hi
        checked += 1
    assert checked > 0


def test_m5_coherent_draws_run_is_byte_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # R7 determinism: a seeded coherent run reproduces its ledger byte-for-byte.
    monkeypatch.setattr(
        "calibre.execution.prediction.resolve_adapter",
        lambda model_config: _ResidualAdapter(model_config),
    )
    bundle, _actuals, _tasks, _origins = _m5_bundle_tasks_origins()
    node_history = _synthetic_m5_node_history(bundle.hierarchy)
    origins = [pd.Timestamp("2011-03-15"), pd.Timestamp("2011-03-16")]

    first = _run_m5_coherent(
        bundle, node_history, _synthetic_residual_tasks(node_history, horizon=1), origins
    )
    second = _run_m5_coherent(
        bundle, node_history, _synthetic_residual_tasks(node_history, horizon=1), origins
    )
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Coherent across-origin parallelization byte-identity gate.
#
# The parallel path (Ray bounded sliding window over origins) MUST produce
# byte-for-byte identical ledger output to the serial path. The serial baseline
# is pinned to ``backend="ray", max_concurrency=1`` (window-of-1), NOT
# ``backend="local"`` — so the no-resort ``check_exact=True`` gate isolates the
# coherent-dispatch window as the only variable and does not inherit a latent
# local-vs-ray Predict concat-order delta.
# ---------------------------------------------------------------------------

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
    """CA-subset hierarchy + real-adapter tasks for the across-origin byte gate.

    The real statsforecast SeasonalNaive (no monkeypatched stub) is used so the
    Predict phase computes byte-identically on both the driver (``backend=local``)
    and a Ray worker — a driver-only ``resolve_adapter`` monkeypatch is invisible
    to the worker process and would make local and ray diverge. SeasonalNaive
    supplies the in-sample fitted values the coherent slab requires natively.
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


# ---------------------------------------------------------------------------
# Coherent across-origin parallel harness byte-identity gate (#233).
#
# The across-origin acceptance suite for the coherent
# (SymmetricIntervalRuntime + CoherentDraws) path through the method-agnostic
# harness. Both A/B sides are TWO ``ray`` runs — ``max_concurrency=1`` (serial,
# window-of-1 collapse) vs ``max_concurrency=2`` (parallel) — both driven by a
# WORKER-IMPORTABLE real adapter (real ``SeasonalNaive``), never a
# ``backend="local"`` + monkeypatched side: a driver-only ``resolve_adapter``
# patch is invisible to ray Predict workers and would silently force a sort. The
# coherent slab is the RNG-seeded draw+reconcile compute (no model fit), so the
# tier is cheap despite covering ledger sha256 + ``.resolved.parquet`` bytes +
# resume + the joint-store + sigma-lag resume surface.
# ---------------------------------------------------------------------------


def _coherent_config() -> SymmetricIntervalConfig:
    """The per-horizon coherent-draws config the byte gate drives (the IN path)."""
    return SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=3,
        gamma=0.05,
        spread="coherent_draws",
        draw_count=128,
        draw_seed=11,
    )


def _run_coherent_parallel(
    bundle,
    actuals,
    tasks,
    origins,
    *,
    backend: str = "ray",
    max_concurrency: int | None = None,
    cpu_per_task: float | None = None,
    output_path: Path | None = None,
    initial_ledger: pd.DataFrame | None = None,
) -> BackendResult:
    """Run the coherent path through the method-agnostic harness.

    Threads the coherent-draws conformal config + the hierarchy index (which
    supplies ``S``) with ``reconciler=None``, so ``_origin_compute_target`` routes
    it onto the coherent worker. ``backend``/``max_concurrency``/``cpu_per_task``
    select serial (window-of-1) vs parallel dispatch and the BLAS budget; the
    worker-importable real ``SeasonalNaive`` in ``tasks`` natively supplies the
    fitted-value sidecar, so a parallel ``ray`` run is byte-identical to a
    window-of-1 ``ray`` run (no monkeypatch).
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
        conformal=ConformalOptions(config=_coherent_config(), initial_ledger=initial_ledger),
        reconciliation=ReconciliationOptions(
            reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
    )
    try:
        return engine.execute(tasks, actuals, origins)
    finally:
        engine.close()


def _assert_coherent_spread_engaged(ledger: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
    """Fail a green-but-hollow gate: the coherent draw spread must actually engage.

    Asserts at least one issued cross-section is non-degenerate (positive total
    width) AND non-additive (the total band width differs from the sum of member
    band widths) — the property only reconciled draw quantiles produce. A silent
    fall-back to additive/``SeasonalNaive`` point geometry would have an additive
    total width and would slip a byte gate that only checks serial==parallel.
    """
    lower_col, upper_col = interval_column_names(0.9)
    summing = build_summing_matrix(hierarchy)
    issued = ledger[ledger[lower_col].notna()]
    assert not issued.empty
    non_degenerate = False
    non_additive = False
    for _, group in issued.groupby([MODEL_NAME, FORECAST_ORIGIN, H], sort=False):
        by_uid = group.set_index(UNIQUE_ID)
        if "__total__" not in by_uid.index:
            continue
        total_width = float(by_uid.loc["__total__", upper_col] - by_uid.loc["__total__", lower_col])
        present_bottom = [b for b in summing.bottom_ids if b in by_uid.index]
        if not present_bottom:
            continue
        child_width_sum = float(
            (by_uid.loc[present_bottom, upper_col] - by_uid.loc[present_bottom, lower_col]).sum()
        )
        if total_width > 1e-9:
            non_degenerate = True
        if abs(total_width - child_width_sum) > 1e-9:
            non_additive = True
    assert non_degenerate, "coherent intervals degenerate to a point — spread not engaged"
    assert non_additive, "coherent total width is additive — reconciled draws not engaged"


def test_coherent_runtime_rejects_point_reconciler_at_construction() -> None:
    """T0: coherent-draws conformal + a point reconciler is rejected by the engine.

    The serial path reconciles centers BEFORE Calibrate, but the parallel
    coherent drain feeds raw centers into the draw slab, so a configured reconciler
    would diverge serial-vs-parallel. The CLI validator already rejects this combo;
    the engine boundary must too. Constructing the engine with a coherent config and
    a center-rewriting point reconciler raises; the ``none`` strategy (a NoOp
    pass-through) is allowed (see
    :func:`test_coherent_runtime_accepts_noop_reconciler_at_construction`).
    """
    bundle, _node_history, _tasks = _fused_residual_fixture()
    with pytest.raises(ValueError, match="coherent-draws conformal cannot be combined"):
        BackendEngine(
            execution=ExecutionOptions(freq="D", backend="local", seed=42),
            conformal=ConformalOptions(config=_coherent_config()),
            reconciliation=ReconciliationOptions(
                reconciler=BottomUpReconciler(),
                hierarchy_index=build_hierarchy_index(bundle.hierarchy),
            ),
        )


def test_coherent_runtime_accepts_noop_reconciler_at_construction() -> None:
    """T0b: coherent-draws conformal + the ``none`` (NoOp) reconciler is allowed.

    ``reconciliation: strategy: none`` resolves to a NoOpReconciler — a non-None
    pass-through that leaves centers untouched — which is the supported coherent
    wiring (the hierarchy still supplies S). The engine boundary must NOT reject it;
    the guard rejects only a center-rewriting point reconciler. This pins against the
    over-broad ``reconciler is not None`` check, which rejected the real CLI coherent
    config (``strategy: none`` never produces a literal ``None`` reconciler).
    """
    bundle, _node_history, _tasks = _fused_residual_fixture()
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        conformal=ConformalOptions(config=_coherent_config()),
        reconciliation=ReconciliationOptions(
            reconciler=NoOpReconciler(),
            hierarchy_index=build_hierarchy_index(bundle.hierarchy),
        ),
    )
    assert isinstance(engine.reconciler, NoOpReconciler)


@pytest.mark.parametrize("cpu_per_task", [None, 1.0])
def test_coherent_parallel_ledger_byte_identical_to_serial(cpu_per_task: float | None) -> None:
    """T1: N=2 coherent parallel ledger is byte-identical to the window-of-1 serial.

    Both sides are ``ray`` runs (worker-side real adapter). Both ``cpu_per_task``
    variants (``None`` and ``1.0``) map to ``thread_budget == 1`` on this
    thread-invariant n_bottom=2 fixture, so this gate does not by itself exercise a
    thread asymmetry; the ``threadpool_limits`` wrapper presence on BOTH the serial
    apply and the worker is pinned independently by
    :func:`test_coherent_apply_and_worker_share_thread_budget`. Here the variants
    pin that both budgets feed through cleanly and the ``None`` variant fixes the
    production default.
    """
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()

    serial = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=1, cpu_per_task=cpu_per_task
    ).ledger.to_df()
    parallel = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=2, cpu_per_task=cpu_per_task
    ).ledger.to_df()

    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    assert _ledger_sha256(serial) == _ledger_sha256(parallel)
    lower_col, upper_col = interval_column_names(0.9)
    np.testing.assert_array_equal(
        serial[[lower_col, upper_col]].to_numpy(), parallel[[lower_col, upper_col]].to_numpy()
    )
    # The gate is not green-but-hollow: the coherent draw spread genuinely engages.
    _assert_coherent_spread_engaged(serial, bundle.hierarchy)


def test_coherent_mid_stream_calibrator_update_is_real() -> None:
    """T1b: a prior-origin row resolves mid-stream, so the byte gate is a real guard.

    If no due rows resolved between consecutive origins, no ``calibrator.update``
    would fire mid-stream and every byte gate would pass even with the drain-time
    ordering wrong (no calibrator mutation to reorder). Spy ``calibrator.update``
    and assert at least one fires inside a ``_resolve_open`` that PRECEDES a later
    origin's apply in the 3-origin/horizon-2 fixture — pinning T1 as a genuine
    state-ordering guard, not a coincidence.
    """
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()

    from calibre.conformal.calibrators import RollingQuantileCalibrator

    updates_in_resolve_open = 0
    in_resolve_open = False
    real_update = RollingQuantileCalibrator.update
    real_resolve_open = BackendEngine._resolve_open

    def _spy_update(self, *args, **kwargs):
        nonlocal updates_in_resolve_open
        if in_resolve_open:
            updates_in_resolve_open += 1
        return real_update(self, *args, **kwargs)

    def _spy_resolve_open(self, ledger, actuals, origin, runtime):
        nonlocal in_resolve_open
        in_resolve_open = True
        try:
            return real_resolve_open(self, ledger, actuals, origin, runtime)
        finally:
            in_resolve_open = False

    import unittest.mock as mock

    with (
        mock.patch.object(RollingQuantileCalibrator, "update", _spy_update),
        mock.patch.object(BackendEngine, "_resolve_open", _spy_resolve_open),
    ):
        _run_coherent_parallel(
            bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=2, cpu_per_task=1.0
        )

    assert updates_in_resolve_open > 0, (
        "no calibrator.update fired during a ResolveOpen — the fixture never resolves a "
        "prior-origin row mid-stream, so T1 would be green for the wrong reason"
    )


@pytest.mark.parametrize("cpu_per_task", [None, 1.0])
def test_coherent_apply_and_worker_share_thread_budget(
    monkeypatch: pytest.MonkeyPatch, cpu_per_task: float | None
) -> None:
    """T1c: BOTH the serial apply and the coherent worker wrap under the same budget.

    The byte gates run a thread-invariant n_bottom=2 fixture where both
    ``cpu_per_task`` variants map to ``thread_budget == 1``, so dropping either
    ``threadpool_limits(thread_budget(cpu_per_task))`` wrapper — the serial
    ``_calibrate`` apply OR the ``compute_origin_coherent`` worker — would stay
    green. This pins the wrapper presence on BOTH sides independent of fixture
    scale: spy the exact ``threadpool_limits`` symbol each module imports and assert
    each side calls it with ``limits == thread_budget(cpu_per_task)``.
    """
    pytest.importorskip("ray")
    from calibre.execution import backend as backend_mod
    from calibre.execution import origin_compute as origin_mod
    from calibre.execution.origin_compute import compute_origin_coherent
    from calibre.execution.threading import thread_budget

    bundle, node_history, tasks = _fused_residual_fixture()
    expected_limit = thread_budget(cpu_per_task)

    def _make_spy(real, recorder):
        def _spy(*args, **kwargs):
            recorder.append(kwargs.get("limits"))
            return real(*args, **kwargs)

        return _spy

    # Serial apply side: a ``backend="local"`` coherent run routes through
    # :meth:`BackendEngine._calibrate`, which wraps the apply in the budget.
    apply_limits: list[object] = []
    monkeypatch.setattr(
        backend_mod, "threadpool_limits", _make_spy(backend_mod.threadpool_limits, apply_limits)
    )
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42, cpu_per_task=cpu_per_task),
        conformal=ConformalOptions(config=_coherent_config()),
        reconciliation=ReconciliationOptions(
            reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
    )
    try:
        engine.execute(tasks, node_history, _FUSED_ORIGINS)
    finally:
        engine.close()
    assert expected_limit in apply_limits, (
        "the serial coherent _calibrate apply did not wrap in "
        "threadpool_limits(thread_budget(cpu_per_task))"
    )

    # Worker side: capture the real per-origin inputs + run-constant S a coherent
    # dispatch would ship, then invoke ``compute_origin_coherent`` in-process under
    # the spy to assert it wraps under the same budget.
    captured: list[tuple] = []
    real_dispatch = BackendEngine._dispatch_origin_coherent

    def _capturing_dispatch(self, inputs, summing):
        captured.append((inputs, summing))
        return real_dispatch(self, inputs, summing)

    monkeypatch.setattr(BackendEngine, "_dispatch_origin_coherent", _capturing_dispatch)
    _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=2, cpu_per_task=cpu_per_task
    )
    assert captured, "no coherent origin was dispatched — cannot exercise the worker budget"

    worker_limits: list[object] = []
    monkeypatch.setattr(
        origin_mod, "threadpool_limits", _make_spy(origin_mod.threadpool_limits, worker_limits)
    )
    inputs, summing = captured[0]
    compute_origin_coherent(inputs, summing, cpu_per_task)
    assert expected_limit in worker_limits, (
        "compute_origin_coherent did not wrap in threadpool_limits(thread_budget(cpu_per_task))"
    )


def test_coherent_parallel_resolved_parquet_byte_identical(tmp_path: Path) -> None:
    """T2: finalized streamed ``.resolved.parquet`` is byte-identical serial-vs-parallel."""
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()
    serial_path = tmp_path / "serial.parquet"
    parallel_path = tmp_path / "parallel.parquet"

    _run_coherent_parallel(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        max_concurrency=1,
        cpu_per_task=1.0,
        output_path=serial_path,
    )
    _run_coherent_parallel(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        max_concurrency=2,
        cpu_per_task=1.0,
        output_path=parallel_path,
    )

    serial_resolved = resolved_ledger_uri(serial_path)
    parallel_resolved = resolved_ledger_uri(parallel_path)
    assert Path(serial_resolved).read_bytes() == Path(parallel_resolved).read_bytes()


def test_coherent_max_concurrency_one_takes_serial_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """T3a: ``max_concurrency=1`` coherent run takes the serial path (no dispatch).

    The window-of-1 collapse: ``_parallel_origin_window`` returns ``None`` at N==1,
    so the serial generator drives the loop and the coherent slab is never
    dispatched to a worker.
    """
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()

    coherent_dispatched = 0
    real_dispatch = BackendEngine._dispatch_origin_coherent

    def _counting_dispatch(self, inputs, summing):
        nonlocal coherent_dispatched
        coherent_dispatched += 1
        return real_dispatch(self, inputs, summing)

    monkeypatch.setattr(BackendEngine, "_dispatch_origin_coherent", _counting_dispatch)
    ledger = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=1, cpu_per_task=1.0
    ).ledger.to_df()

    assert coherent_dispatched == 0  # N==1 collapses to serial; no worker dispatch
    assert not ledger.empty


def test_coherent_predict_overlaps_draw_slab(monkeypatch: pytest.MonkeyPatch) -> None:
    """T3b: Predict(N+1) is dispatched before slab(N) is consumed.

    Drain-time dispatch makes the *draw-slab* peak-in-flight 1 by construction, so
    the genuine overlap to prove is that the next origin's Predict fires before the
    current origin's slab is drained — NOT a ``slab_in_flight >= 2`` assertion,
    which is unsatisfiable under the byte-safe design and would pressure the
    implementation back toward the unsafe eager-producer shape. Record the event
    order of Predict-dispatch and slab-``ray.get`` and assert origin N+1's Predict
    precedes origin N's slab consumption.
    """
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()

    events: list[str] = []
    real_predict = BackendEngine._predict
    real_get = BackendEngine._get_origin_coherent

    def _spy_predict(self, chunk_refs, direct_refs, origin):
        events.append(f"predict:{pd.Timestamp(origin)}")
        return real_predict(self, chunk_refs, direct_refs, origin)

    def _spy_get(ref):
        events.append("slab_get")
        return real_get(ref)

    monkeypatch.setattr(BackendEngine, "_predict", _spy_predict)
    monkeypatch.setattr(BackendEngine, "_get_origin_coherent", staticmethod(_spy_get))
    _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=2, cpu_per_task=1.0
    )

    # With a window of 2 the producer runs Predict for origins 0 and 1 before
    # draining origin 0's slab: the second Predict must precede the first slab_get.
    first_slab = events.index("slab_get")
    predicts_before_first_slab = [e for e in events[:first_slab] if e.startswith("predict:")]
    assert len(predicts_before_first_slab) >= 2, (
        f"Predict(N+1) did not overlap slab(N): events={events}"
    )


def test_coherent_window_commits_in_origin_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """T4: the coherent consumer commits head-first in origins-list order.

    Spy ``_finish_origin``; assert commits fire in origins-list order regardless of
    worker completion order, and that the carry-forward state is byte-identical to
    the window-of-1 serial baseline. Proves the ``_APPEND_SEQ`` head-first drain
    holds for the coherent path.
    """
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()

    serial = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=1, cpu_per_task=1.0
    ).ledger.to_df()

    committed_origins: list[pd.Timestamp] = []
    real_finish = BackendEngine._finish_origin

    def _spy_finish(self, ledger, order_ledger, actuals, origin, origin_preds, runtime):
        committed_origins.append(pd.Timestamp(origin))
        return real_finish(self, ledger, order_ledger, actuals, origin, origin_preds, runtime)

    monkeypatch.setattr(BackendEngine, "_finish_origin", _spy_finish)
    parallel = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=3, cpu_per_task=1.0
    ).ledger.to_df()

    assert committed_origins == _FUSED_ORIGINS
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)


def test_coherent_parallel_vs_parallel_determinism() -> None:
    """T5: two independent N=2 coherent runs are byte-identical (seeded slab)."""
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()

    first = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=2, cpu_per_task=1.0
    ).ledger.to_df()
    second = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=2, cpu_per_task=1.0
    ).ledger.to_df()

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert _ledger_sha256(first) == _ledger_sha256(second)


def test_coherent_resume_happy_path_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """T6a: resumed origins are never re-dispatched; resumed parallel == resumed serial.

    Build an ``initial_ledger`` prefix over the first origins, spy the coherent
    dispatch to assert resumed origins are NEVER re-dispatched, and assert the
    resumed serial and resumed parallel runs are byte-identical (this half stays
    exact, no sort).
    """
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()

    prefix = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS[:2], max_concurrency=1, cpu_per_task=1.0
    ).ledger.to_df()

    dispatched: list[pd.Timestamp] = []
    real_dispatch = BackendEngine._dispatch_origin_coherent

    def _spy_dispatch(self, inputs, summing):
        dispatched.append(pd.Timestamp(inputs.context.frame[FORECAST_ORIGIN].iloc[0]))
        return real_dispatch(self, inputs, summing)

    resumed_serial = _run_coherent_parallel(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        max_concurrency=1,
        cpu_per_task=1.0,
        initial_ledger=prefix,
    ).ledger.to_df()
    monkeypatch.setattr(BackendEngine, "_dispatch_origin_coherent", _spy_dispatch)
    resumed_parallel = _run_coherent_parallel(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        max_concurrency=2,
        cpu_per_task=1.0,
        initial_ledger=prefix,
    ).ledger.to_df()

    assert _FUSED_ORIGINS[0] not in dispatched
    assert _FUSED_ORIGINS[1] not in dispatched
    assert _FUSED_ORIGINS[2] in dispatched
    pd.testing.assert_frame_equal(resumed_serial, resumed_parallel, check_exact=True)


def test_coherent_resume_joint_store_and_sigma_lag_survive(tmp_path: Path) -> None:
    """T6b: the ``::joint::`` store + one-origin sigma lag survive resume byte-identically.

    Persist coherent state through the SQL state store across a split run and assert
    the resumed final origin reproduces the uninterrupted run's rows exactly. A
    faithful resume carries BOTH halves of the durable state: the conformal state
    (calibrator + the ``::joint::`` node-vector store, rehydrated via
    ``from_partition_states``) AND the ledger prefix — the latter holds the still-open
    multi-horizon rows from the first leg, so an h2 row predicted in the prefix
    resolves at the resumed origin and advances the same calibrator window the
    uninterrupted run sees at apply. Resuming over the full origins list (completed
    origins skipped) with that prefix is what production resume does; a state-only
    resume would strand those pending rows and miss one h2 update, so the exact
    compare below is the teeth on a complete resume — the coherent path keeps the
    runtime live across resume. ``backend="local"`` keeps the persistence surface
    (not the ray window) as the variable under test.
    """
    pytest.importorskip("ray")
    from calibre.storage.models import Base
    from calibre.storage.postgres import (
        ConformalStateRepo,
        RunRepo,
        make_engine,
        make_session_factory,
        session_scope,
    )
    from calibre.storage.session import derive_session_id
    from calibre.storage.state import SqlConformalStateStore

    bundle, node_history, tasks = _fused_residual_fixture()
    config = _coherent_config()
    hierarchy_index = build_hierarchy_index(bundle.hierarchy)

    def _engine(conformal: ConformalOptions) -> BackendEngine:
        return BackendEngine(
            execution=ExecutionOptions(freq="D", backend="local", seed=42),
            conformal=conformal,
            reconciliation=ReconciliationOptions(reconciler=None, hierarchy_index=hierarchy_index),
        )

    uninterrupted = _engine(ConformalOptions(config=config)).execute(
        tasks, node_history, _FUSED_ORIGINS
    )

    db_engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'resume.db').as_posix()}")
    Base.metadata.create_all(db_engine)
    factory = make_session_factory(db_engine)
    session_id = derive_session_id(
        "tenant",
        ["coherent"],
        {"model": "SeasonalNaive"},
        {
            "method": config.method,
            "coverage": config.coverage,
            "calibration_window": config.calibration_window,
            "gamma": config.gamma,
            "spread": config.spread,
        },
    )

    with session_scope(factory) as session:
        first_run = RunRepo(session).create(config={"run": 1})
        prefix = (
            _engine(
                ConformalOptions(
                    config=config,
                    run_id=first_run.id,
                    state_store=SqlConformalStateStore(
                        ConformalStateRepo(session), session_id=session_id
                    ),
                )
            )
            .execute(tasks, node_history, _FUSED_ORIGINS[:2])
            .ledger.to_df()
        )

    with session_scope(factory) as session:
        second_run = RunRepo(session).create(config={"run": 2})
        # Resume over the full origins list: completed origins are skipped, but the
        # ledger prefix's still-open multi-horizon rows resolve at the resumed origin
        # so the calibrator window advances exactly as the uninterrupted run's does.
        resumed = _engine(
            ConformalOptions(
                config=config,
                run_id=second_run.id,
                initial_ledger=prefix,
                state_store=SqlConformalStateStore(
                    ConformalStateRepo(session), session_id=session_id
                ),
            )
        ).execute(tasks, node_history, _FUSED_ORIGINS)

    sort_cols = [UNIQUE_ID, FORECAST_ORIGIN, H]
    lower_col, upper_col = interval_column_names(0.9)
    expected = uninterrupted.ledger.to_df()
    expected = expected[expected[FORECAST_ORIGIN] == _FUSED_ORIGINS[2]]
    expected = expected.sort_values(sort_cols).reset_index(drop=True)
    actual = resumed.ledger.to_df()
    actual = actual[actual[FORECAST_ORIGIN] == _FUSED_ORIGINS[2]]
    actual = actual.sort_values(sort_cols).reset_index(drop=True)
    # The resumed final origin reproduces the uninterrupted run: same kappa-widened
    # bounds prove the joint store + sigma lag rehydrated correctly. Exact compare
    # (matching T6a) — a SQL-round-trip or kappa-recompute drift must not slip a
    # near-equal tolerance.
    pd.testing.assert_frame_equal(
        actual[[lower_col, upper_col]], expected[[lower_col, upper_col]], check_exact=True
    )


def test_coherent_producer_failure_resumes_with_identical_rowset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T6c: a worker failing on origin 1 commits origin 0; a resumed run reproduces the rowset.

    Inject a coherent-slab worker failure at origin 1; assert origin 0 committed and
    the failing origin + beyond never reached the ledger. The resumed run reproduces
    the same rowset. The crash-resume rowset compare is sorted on BOTH sides
    symmetrically (post-crash rowset-equivalence only) — acceptable ONLY because
    T1/T2/T5 pin the byte guarantee on a worker-side adapter; not a loosened gate.
    """
    pytest.importorskip("ray")
    bundle, node_history, tasks = _fused_residual_fixture()
    fail_origin = _FUSED_ORIGINS[1]

    real_dispatch = BackendEngine._dispatch_origin_coherent

    def _maybe_failing_dispatch(self, inputs, summing):
        origin = pd.Timestamp(inputs.context.frame[FORECAST_ORIGIN].iloc[0])
        if origin == fail_origin:
            import ray

            @ray.remote
            def _boom():
                raise RuntimeError("injected coherent slab failure")

            return _boom.remote()
        return real_dispatch(self, inputs, summing)

    monkeypatch.setattr(BackendEngine, "_dispatch_origin_coherent", _maybe_failing_dispatch)

    engine = BackendEngine(
        execution=ExecutionOptions(
            freq="D", backend="ray", seed=42, ray_threshold=1, max_concurrency=3, cpu_per_task=1.0
        ),
        conformal=ConformalOptions(config=_coherent_config()),
        reconciliation=ReconciliationOptions(
            reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
        ),
    )
    last_ledger: pd.DataFrame | None = None
    try:
        with pytest.raises(RuntimeError, match=r"Calibrate phase failed at origin"):
            for result in engine.iter_origins(tasks, node_history, _FUSED_ORIGINS):
                last_ledger = result.ledger.to_df()
    finally:
        engine.close()

    assert last_ledger is not None
    committed = {pd.Timestamp(o) for o in last_ledger[FORECAST_ORIGIN].unique()}
    assert _FUSED_ORIGINS[0] in committed
    assert fail_origin not in committed
    assert _FUSED_ORIGINS[2] not in committed

    monkeypatch.undo()
    resumed = _run_coherent_parallel(
        bundle,
        node_history,
        tasks,
        _FUSED_ORIGINS,
        max_concurrency=2,
        cpu_per_task=1.0,
        initial_ledger=last_ledger,
    ).ledger.to_df()
    clean = _run_coherent_parallel(
        bundle, node_history, tasks, _FUSED_ORIGINS, max_concurrency=1, cpu_per_task=1.0
    ).ledger.to_df()
    # The post-crash check is ROWSET equivalence on the calibrator-INDEPENDENT
    # structural columns (keys + centers + actuals), sorted on both sides
    # symmetrically. The interval VALUES legitimately differ: this resume carries no
    # state store, so the resumed run rebuilds the online calibrator cold from the
    # prefix ledger and reaches a different (warm-vs-cold) radius than the clean run.
    # The serial==parallel byte guarantee is pinned by T1/T2/T5 on a worker-side
    # adapter; T6b separately proves the joint-store + sigma lag survive a *stateful*
    # resume byte-identically. This half asserts only that the same rows reach the
    # ledger after a crash + resume.
    keys = [UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME, H]
    structural = [UNIQUE_ID, FORECAST_ORIGIN, MODEL_NAME, H, Y_HAT, Y]
    pd.testing.assert_frame_equal(
        resumed.sort_values(keys).reset_index(drop=True)[structural],
        clean.sort_values(keys).reset_index(drop=True)[structural],
    )


def test_analytic_and_cumulative_configs_never_take_parallel_path() -> None:
    """T8: analytic-spread + cumulative-coherent configs route serial (isolation lock).

    Structural VN2-class isolation, independent of which backend VN2 runs: assert
    ``_parallel_origin_window`` / ``_origin_compute_target`` return ``None`` for an
    analytic-spread ``backend="ray"`` config (so VN2's atomic apply is untouched)
    AND for a cumulative-mode coherent config (so cumulative-coherent never reaches
    the per-horizon-only worker).
    """
    hierarchy = _ca_subset_hierarchy(_m5_hierarchy_frame())
    hierarchy_index = build_hierarchy_index(hierarchy)

    analytic = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="ray", seed=42, ray_threshold=1),
        conformal=ConformalOptions(
            config=SymmetricIntervalConfig(method="aci", coverage=0.9, spread="analytic")
        ),
    )
    try:
        assert analytic._origin_compute_target is None  # noqa: SLF001
        assert analytic._parallel_origin_window() is None  # noqa: SLF001
    finally:
        analytic.close()

    cumulative = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="ray", seed=42, ray_threshold=1),
        conformal=ConformalOptions(
            config=SymmetricIntervalConfig(
                method="mscp",
                coverage=0.9,
                mode="cumulative",
                protection_period=2,
                spread="coherent_draws",
                draw_count=64,
            )
        ),
        reconciliation=ReconciliationOptions(reconciler=None, hierarchy_index=hierarchy_index),
    )
    try:
        # Cumulative-coherent is OUT of the parallel path — it must route serial so a
        # mscp+coherent+ray run never hits the per-horizon-only worker.
        assert cumulative._origin_compute_target is None  # noqa: SLF001
        assert cumulative._parallel_origin_window() is None  # noqa: SLF001
    finally:
        cumulative.close()


def test_get_origin_coherent_reraises_wrapper_when_cause_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RayTaskError with no ``cause`` is re-raised intact, not unwrapped.

    The unwrap path re-raises ``exc.cause`` so ``_phase`` can name the origin, but a
    wrapper with ``cause is None`` has nothing to unwrap — assert the original
    ``RayTaskError`` propagates unchanged via :func:`backend._get_origin_coherent`'s
    fallback ``raise``.
    """
    ray = pytest.importorskip("ray")
    from ray.exceptions import RayTaskError

    # A RayTaskError whose .cause we force to None (the unwrappable-cause branch).
    wrapper = RayTaskError("compute_origin_coherent", "traceback-string", cause=RuntimeError("x"))
    wrapper.cause = None

    def _raise_wrapper(_ref: object) -> object:
        raise wrapper

    monkeypatch.setattr(ray, "get", _raise_wrapper)
    with pytest.raises(RayTaskError) as excinfo:
        BackendEngine._get_origin_coherent(object())
    assert excinfo.value is wrapper


def test_coherent_cache_and_ref_rebuild_after_owned_runtime_shutdown() -> None:
    """A fresh runtime re-``ray.put``s ``S`` and rebuilds the coherent remote handle.

    The cached summing ref and coherent remote handle are non-``None`` after a run
    on an owned runtime, and ``close()`` nulls both so the next acquisition rebuilds
    them against the new runtime (a stale ref would point at an object the new
    runtime never plasma-stored). Two runs must produce byte-identical ledgers.
    """
    ray = pytest.importorskip("ray")
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
            conformal=ConformalOptions(config=_coherent_config()),
            reconciliation=ReconciliationOptions(
                reconciler=None, hierarchy_index=build_hierarchy_index(bundle.hierarchy)
            ),
        )
        try:
            result = engine.execute(tasks, node_history, _FUSED_ORIGINS)
            # The owned runtime cached a fresh summing ref + coherent handle this run.
            assert engine._summing_ref is not None  # noqa: SLF001
            assert engine._remote_compute_origin_coherent is not None  # noqa: SLF001
            results.append(result.ledger.to_df())
        finally:
            engine.close()
            # Shutdown nulls the cached ref/handle so the next run rebuilds them.
            assert engine._summing_ref is None  # noqa: SLF001
            assert engine._remote_compute_origin_coherent is None  # noqa: SLF001
        if ray.is_initialized():
            ray.shutdown()

    pd.testing.assert_frame_equal(results[0], results[1], check_exact=True)


# ---------------------------------------------------------------------------
# #210: point bottom_up aggregate-actuals precompute byte-identity gate.
#
# Warming HierarchyActualsSource._lookup_cache for the whole window up front must
# be a pure performance change: the resolved ledger (y values + complete-set
# membership of (node, ds) pairs) must be byte-for-byte identical to the lazy
# (unseeded) run. These gates run the *point bottom_up* path (the only place
# HierarchyActualsSource is live) on a CA-subset hierarchy, NOT the
# FrameActualsSource fixtures the across-origin byte gates use.
# ---------------------------------------------------------------------------

_PRECOMPUTE_ORIGINS = [
    pd.Timestamp("2011-03-15"),
    pd.Timestamp("2011-03-16"),
    pd.Timestamp("2011-03-17"),
]
_PRECOMPUTE_HORIZON = 2


def _ca_subset_bottom_history(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Bottom-level history for the CA-subset hierarchy (no aggregate rows)."""
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
    return pd.DataFrame(rows)


def _precompute_window_ds() -> list[pd.Timestamp]:
    """The (origins x horizon) prediction-date superset the run resolves.

    Mirrors ``hierarchy_preparation._evaluation_window_ds``: the span starts at
    each origin and runs one step past the horizon, bracketing the adapter's h=1
    anchor (here h=1 lands on the origin itself).
    """
    dates: pd.DatetimeIndex = pd.DatetimeIndex([])
    for origin in _PRECOMPUTE_ORIGINS:
        dates = dates.union(pd.date_range(origin, periods=_PRECOMPUTE_HORIZON + 1, freq="D"))
    return list(dates)


def _run_point_bottom_up(
    hierarchy: pd.DataFrame,
    bottom_history: pd.DataFrame,
    *,
    precompute: bool,
    output_path: Path | None = None,
) -> BackendResult:
    """Run the point bottom_up path; optionally pre-seed the actuals window cache.

    Builds bottom-only tasks + a lazy ``HierarchyActualsSource`` + a
    ``BottomUpReconciler`` (the live ``full.yaml`` shape) and runs three
    consecutive origins at horizon 2 with conformal on, so ``_resolve_due`` fires
    at both ResolveOpen and Commit and new ``(node, ds)`` pairs enter per origin.
    """
    index = build_hierarchy_index(hierarchy)
    tasks = build_tasks(
        bottom_history,
        [
            {
                "backend": "statsforecast",
                "model": "SeasonalNaive",
                "name": "SeasonalNaive",
                "season_length": 7,
            }
        ],
        _PRECOMPUTE_HORIZON,
    )
    actuals = HierarchyActualsSource(bottom_history, index)
    if precompute:
        actuals.precompute(index.node_labels, _precompute_window_ds())

    output = (
        LedgerOutputOptions(forecast_path=str(output_path), streaming=True)
        if output_path is not None
        else LedgerOutputOptions()
    )
    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        output=output,
        conformal=ConformalOptions(
            config=SymmetricIntervalConfig(
                method="aci", coverage=0.9, calibration_window=4, gamma=0.05
            )
        ),
        reconciliation=ReconciliationOptions(
            reconciler=BottomUpReconciler(), hierarchy_index=index
        ),
    )
    try:
        return engine.execute(tasks, actuals, _PRECOMPUTE_ORIGINS)
    finally:
        engine.close()


def test_precompute_on_vs_off_ledger_byte_identical() -> None:
    """T1: precompute-ON ledger is byte-identical to the lazy (OFF) run."""
    hierarchy = _ca_subset_hierarchy(_m5_hierarchy_frame())
    bottom_history = _ca_subset_bottom_history(hierarchy)

    off = _run_point_bottom_up(hierarchy, bottom_history, precompute=False).ledger.to_df()
    on = _run_point_bottom_up(hierarchy, bottom_history, precompute=True).ledger.to_df()

    pd.testing.assert_frame_equal(off, on, check_exact=True)
    assert _ledger_sha256(off) == _ledger_sha256(on)


def test_precompute_seeds_superset_of_due_ds(monkeypatch: pytest.MonkeyPatch) -> None:
    """T1 guard: the seeded ds set is a superset of the ds the resolve actually requests.

    Spies on the pending ``ds`` each ``resolve`` is asked to fill (what
    ``due_frame`` produces), not the whole ledger. A future freq/adapter change
    that breaks the superset property fails loudly here rather than silently
    falling back to lazy (byte-identical but a perf regression).
    """
    hierarchy = _ca_subset_hierarchy(_m5_hierarchy_frame())
    bottom_history = _ca_subset_bottom_history(hierarchy)

    requested_ds: set[pd.Timestamp] = set()
    real_resolve = HierarchyActualsSource.resolve

    def _spy_resolve(self, ledger_df, current_origin):
        pending = ledger_df[ledger_df[Y].isna() & (ledger_df[DS] <= current_origin)]
        requested_ds.update(pd.to_datetime(pending[DS]).unique())
        return real_resolve(self, ledger_df, current_origin)

    monkeypatch.setattr(HierarchyActualsSource, "resolve", _spy_resolve)
    _run_point_bottom_up(hierarchy, bottom_history, precompute=False)

    seeded_ds = set(pd.to_datetime(pd.Index(_precompute_window_ds())))
    assert requested_ds
    assert requested_ds <= seeded_ds


def test_precompute_on_vs_off_resolved_parquet_byte_identical(tmp_path: Path) -> None:
    """T2: finalized streamed ``.resolved.parquet`` is byte-identical ON vs OFF."""
    hierarchy = _ca_subset_hierarchy(_m5_hierarchy_frame())
    bottom_history = _ca_subset_bottom_history(hierarchy)
    off_path = tmp_path / "off.parquet"
    on_path = tmp_path / "on.parquet"

    _run_point_bottom_up(hierarchy, bottom_history, precompute=False, output_path=off_path)
    _run_point_bottom_up(hierarchy, bottom_history, precompute=True, output_path=on_path)

    off_resolved = resolved_ledger_uri(off_path)
    on_resolved = resolved_ledger_uri(on_path)
    assert Path(off_resolved).read_bytes() == Path(on_resolved).read_bytes()


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
