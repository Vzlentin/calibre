from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from calibre.core.forecast_frame import (
    DS,
    FITTED_Y_HAT,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.reconciliation.nixtla_adapter import (
    NixtlaReconciler,
    _cache_key,
    _from_nixtla_layout,
    _make_checked_min_trace_sparse,
    _to_nixtla_layout,
)
from calibre.reconciliation.protocols import ReconciliationContext
from calibre.reconciliation.summing import (
    SparseSummingMatrix,
    SummingMatrix,
    build_hierarchy_index,
    build_summing_matrix,
    sparse_summing_matrix_from_index,
)


class _CountingMethod:
    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(
        self, *, S: np.ndarray, y_hat: np.ndarray, tags: dict[str, np.ndarray]
    ) -> _CountingMethod:
        del S, y_hat, tags
        self.fit_calls += 1
        return self

    def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]:
        del S
        return {"mean": y_hat}


def _counting_factory() -> tuple[list[_CountingMethod], Callable[[], _CountingMethod]]:
    methods: list[_CountingMethod] = []

    def _factory() -> _CountingMethod:
        method = _CountingMethod()
        methods.append(method)
        return method

    return methods, _factory


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["a", "b", "c"],
            "dept": ["D1", "D1", "D2"],
            "store": ["S1", "S2", "S1"],
        }
    )


def _coherent_base(summing: SummingMatrix, bottom: list[float]) -> np.ndarray:
    return summing.S @ np.array(bottom, dtype=np.float64)


def _require_hierarchy_extra() -> None:
    pytest.importorskip("hierarchicalforecast.methods")


@pytest.mark.parametrize("strategy", ["ols", "wls_struct"])
def test_min_trace_strategy_reconciles_small_lattice_to_coherent_vector(strategy: str) -> None:
    _require_hierarchy_extra()
    summing = build_summing_matrix(_hierarchy()).subset(["a", "b"])
    base = _coherent_base(summing, [4.0, 5.0])
    base[summing.node_labels.index("dept=D1")] = 20.0
    base[summing.total_index] = 30.0

    reconciled = NixtlaReconciler(strategy).reconcile_vector(base, summing)

    np.testing.assert_allclose(
        reconciled,
        summing.S @ reconciled[: summing.n_bottom],
        rtol=1e-6,
        atol=1e-6,
    )


def test_s_layout_conversion_round_trips_to_identity_first_order() -> None:
    summing = build_summing_matrix(_hierarchy()).subset(["a", "c"])
    base = _coherent_base(summing, [2.0, 7.0])

    layout = _to_nixtla_layout(base, summing)
    round_tripped = _from_nixtla_layout(layout.y_hat, layout)

    np.testing.assert_array_equal(layout.S[-summing.n_bottom :], np.eye(summing.n_bottom))
    np.testing.assert_array_equal(round_tripped, base)


@pytest.mark.parametrize("strategy", ["ols", "wls_struct"])
def test_sparse_roster_factory_constructs_checked_min_trace_sparse(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    """ols/wls_struct are sparse-capable: the factory builds the Calibre-owned
    checked subclass of ``MinTraceSparse`` (method, num_threads=1), never the
    dense ``MinTrace``."""
    captured: list[tuple[str, int]] = []

    class _FakeMinTraceSparse(_CountingMethod):
        def __init__(self, *, method: str, num_threads: int) -> None:
            super().__init__()
            captured.append((method, num_threads))

    class _FailDenseMinTrace(_CountingMethod):
        def __init__(self, *, method: str, num_threads: int) -> None:
            del method, num_threads
            raise AssertionError("dense MinTrace must not be constructed for the sparse roster")

    def _fake_import_module(name: str) -> SimpleNamespace:
        assert name == "hierarchicalforecast.methods"
        return SimpleNamespace(
            BottomUp=_CountingMethod,
            BottomUpSparse=_CountingMethod,
            MinTrace=_FailDenseMinTrace,
            MinTraceSparse=_FakeMinTraceSparse,
            ERM=_CountingMethod,
        )

    monkeypatch.setattr(
        "calibre.reconciliation.nixtla_adapter.importlib.import_module",
        _fake_import_module,
    )
    summing = SummingMatrix(
        S=np.eye(2, dtype=np.float64),
        bottom_ids=("a", "b"),
        node_labels=("a", "b"),
    )

    NixtlaReconciler(strategy).reconcile_vector(np.array([1.0, 2.0]), summing)

    assert captured == [(strategy, 1)]


@pytest.mark.parametrize(
    ("strategy", "expected_constructor"),
    [("mint_shrink", "MinTrace"), ("wls_var", "MinTraceSparse")],
)
def test_residual_factory_splits_dense_and_sparse_min_trace(
    monkeypatch: pytest.MonkeyPatch, strategy: str, expected_constructor: str
) -> None:
    """wls_var has an upstream sparse variant; mint_shrink does not and keeps
    the dense ``MinTrace`` path unchanged."""
    captured: list[tuple[str, str, int]] = []

    class _FakeMinTrace(_ResidualMethod):
        def __init__(self, *, method: str, num_threads: int) -> None:
            super().__init__()
            captured.append(("MinTrace", method, num_threads))

    class _FakeMinTraceSparse(_ResidualMethod):
        def __init__(self, *, method: str, num_threads: int) -> None:
            super().__init__()
            captured.append(("MinTraceSparse", method, num_threads))

    def _fake_import_module(name: str) -> SimpleNamespace:
        assert name == "hierarchicalforecast.methods"
        return SimpleNamespace(
            BottomUp=_CountingMethod,
            BottomUpSparse=_CountingMethod,
            MinTrace=_FakeMinTrace,
            MinTraceSparse=_FakeMinTraceSparse,
            ERM=_ResidualMethod,
        )

    monkeypatch.setattr(
        "calibre.reconciliation.nixtla_adapter.importlib.import_module",
        _fake_import_module,
    )

    NixtlaReconciler(strategy)(
        _tiny_forecast_frame(),
        build_hierarchy_index(_tiny_hierarchy()),
        ReconciliationContext(fitted_values=_tiny_fitted_values()),
    )

    assert captured == [(expected_constructor, strategy, 1)]


def test_erm_factory_uses_closed_method(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class _FakeERM(_ResidualMethod):
        def __init__(self, *, method: str) -> None:
            super().__init__()
            captured.append(method)

    def _fake_import_module(name: str) -> SimpleNamespace:
        assert name == "hierarchicalforecast.methods"
        return SimpleNamespace(BottomUp=_CountingMethod, MinTrace=_ResidualMethod, ERM=_FakeERM)

    monkeypatch.setattr(
        "calibre.reconciliation.nixtla_adapter.importlib.import_module",
        _fake_import_module,
    )

    NixtlaReconciler("erm")(
        _tiny_forecast_frame(),
        build_hierarchy_index(_tiny_hierarchy()),
        ReconciliationContext(fitted_values=_tiny_fitted_values()),
    )

    assert captured == ["closed"]


def test_projection_cache_reuses_fit_per_bottom_signature() -> None:
    full = build_summing_matrix(_hierarchy())
    first = full.subset(["a", "b"])
    second = full.subset(["a", "c"])
    methods, factory = _counting_factory()
    reconciler = NixtlaReconciler("ols", method_factory=factory)

    reconciler.reconcile_vector(_coherent_base(first, [1.0, 2.0]), first)
    reconciler.reconcile_vector(_coherent_base(first, [3.0, 4.0]), first)
    reconciler.reconcile_vector(_coherent_base(second, [5.0, 6.0]), second)

    assert len(methods) == 2
    assert sum(method.fit_calls for method in methods) == 2


def test_projection_cache_separates_same_labels_with_different_s_matrices() -> None:
    first = SummingMatrix(
        S=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        bottom_ids=("a", "b"),
        node_labels=("a", "b", "__total__", "group=G"),
    )
    second = SummingMatrix(
        S=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        bottom_ids=first.bottom_ids,
        node_labels=first.node_labels,
    )
    methods, factory = _counting_factory()
    reconciler = NixtlaReconciler("ols", method_factory=factory)

    reconciler.reconcile_vector(_coherent_base(first, [1.0, 2.0]), first)
    reconciler.reconcile_vector(_coherent_base(second, [3.0, 4.0]), second)

    assert len(methods) == 2


def test_projection_cache_evicts_oldest_signature_when_bounded() -> None:
    full = build_summing_matrix(_hierarchy())
    first = full.subset(["a", "b"])
    second = full.subset(["a", "c"])
    methods, factory = _counting_factory()
    reconciler = NixtlaReconciler("ols", method_factory=factory, max_cache_size=1)

    reconciler.reconcile_vector(_coherent_base(first, [1.0, 2.0]), first)
    reconciler.reconcile_vector(_coherent_base(second, [3.0, 4.0]), second)
    reconciler.reconcile_vector(_coherent_base(first, [5.0, 6.0]), first)

    assert len(methods) == 3


def test_incoherent_nixtla_output_raises_clear_error() -> None:
    class _IncoherentMethod(_CountingMethod):
        def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]:
            del S, y_hat
            return {"mean": np.array([[0.0], [1.0], [1.0]], dtype=np.float64)}

    summing = SummingMatrix(
        S=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        bottom_ids=("a", "b"),
        node_labels=("a", "b", "__total__"),
    )
    reconciler = NixtlaReconciler("ols", method_factory=_IncoherentMethod)

    with pytest.raises(ValueError, match="incoherent forecast vector"):
        reconciler.reconcile_vector(np.array([1.0, 2.0, 3.0], dtype=np.float64), summing)


def test_missing_hierarchy_extra_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(name: str) -> Any:
        del name
        raise ImportError("missing")

    monkeypatch.setattr(
        "calibre.reconciliation.nixtla_adapter.importlib.import_module",
        _raise_import_error,
    )

    summing = SummingMatrix(
        S=np.eye(2, dtype=np.float64),
        bottom_ids=("a", "b"),
        node_labels=("a", "b"),
    )
    reconciler = NixtlaReconciler("ols")

    with pytest.raises(RuntimeError, match=r"Install calibre with the 'hierarchy' extra"):
        reconciler.reconcile_vector(np.array([1.0, 2.0], dtype=np.float64), summing)


class _ResidualMethod:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.y_insample: np.ndarray | None = None
        self.y_hat_insample: np.ndarray | None = None

    def fit(
        self,
        *,
        S: np.ndarray,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
    ) -> _ResidualMethod:
        del S, y_hat, tags
        self.fit_calls += 1
        self.y_insample = y_insample
        self.y_hat_insample = y_hat_insample
        return self

    def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]:
        bottom = y_hat[-S.shape[1] :, 0]
        return {"mean": (S @ bottom).reshape(-1, 1)}


def _residual_factory() -> tuple[list[_ResidualMethod], Callable[[], _ResidualMethod]]:
    methods: list[_ResidualMethod] = []

    def _factory() -> _ResidualMethod:
        method = _ResidualMethod()
        methods.append(method)
        return method

    return methods, _factory


def _tiny_hierarchy() -> pd.DataFrame:
    return pd.DataFrame({UNIQUE_ID: ["a", "b"], "dept": ["D", "D"]})


def _tiny_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: pd.Series(["a", "b", "dept=D", "__total__"], dtype="object"),
            DS: pd.to_datetime(["2024-01-03"] * 4),
            Y: np.array([np.nan] * 4, dtype=np.float64),
            Y_HAT: np.array([2.0, 3.0, 20.0, 30.0], dtype=np.float64),
            H: np.array([1] * 4, dtype=np.int64),
            FORECAST_ORIGIN: pd.to_datetime(["2024-01-02"] * 4),
            MODEL_NAME: pd.Series(["m"] * 4, dtype="object"),
        }
    )


def _tiny_fitted_values() -> pd.DataFrame:
    rows = []
    for ds, values, fits in [
        ("2024-01-01", {"a": 1.0, "b": 2.0, "dept=D": 3.0, "__total__": 3.0}, [1.1, 1.8, 2.9, 2.9]),
        ("2024-01-02", {"a": 2.0, "b": 3.0, "dept=D": 5.0, "__total__": 5.0}, [2.2, 2.7, 4.9, 4.9]),
    ]:
        for uid, fitted in zip(values, fits, strict=True):
            rows.append(
                {
                    UNIQUE_ID: uid,
                    DS: pd.Timestamp(ds),
                    Y: values[uid],
                    MODEL_NAME: "m",
                    FITTED_Y_HAT: fitted,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("strategy", ["mint_shrink", "wls_var", "erm"])
def test_residual_strategies_use_fitted_context_and_return_coherent_frame(strategy: str) -> None:
    methods, factory = _residual_factory()
    frame = _tiny_forecast_frame()
    summing = build_summing_matrix(_tiny_hierarchy())

    out = NixtlaReconciler(strategy, method_factory=factory)(
        frame,
        build_hierarchy_index(_tiny_hierarchy()),
        ReconciliationContext(fitted_values=_tiny_fitted_values()),
    )

    assert len(methods) == 1
    assert methods[0].fit_calls == 1
    assert methods[0].y_insample.shape == (summing.n_nodes, 2)
    assert methods[0].y_hat_insample.shape == (summing.n_nodes, 2)
    values = out.set_index(UNIQUE_ID)[Y_HAT].reindex(summing.node_labels).to_numpy(np.float64)
    np.testing.assert_allclose(values, summing.S @ values[: summing.n_bottom])


def test_residual_strategy_requires_fitted_context() -> None:
    with pytest.raises(ValueError, match="requires in-sample fitted values"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualMethod)(
            _tiny_forecast_frame(),
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(),
        )


def test_residual_strategy_names_missing_fitted_node() -> None:
    fitted = _tiny_fitted_values()
    fitted = fitted[fitted[UNIQUE_ID] != "dept=D"]

    with pytest.raises(ValueError, match="dept=D"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualMethod)(
            _tiny_forecast_frame(),
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(fitted_values=fitted),
        )


def test_residual_strategy_rejects_misaligned_fitted_timestamps() -> None:
    fitted = _tiny_fitted_values()
    fitted = fitted[~((fitted[UNIQUE_ID] == "a") & (fitted[DS] == pd.Timestamp("2024-01-02")))]

    with pytest.raises(ValueError, match="misaligned"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualMethod)(
            _tiny_forecast_frame(),
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(fitted_values=fitted),
        )


def test_mint_cov_is_rejected_before_runtime() -> None:
    with pytest.raises(ValueError, match="ill-conditioned covariance"):
        NixtlaReconciler("mint_cov")


def test_reconciliation_rejects_quantile_columns_under_active_hierarchy() -> None:
    frame = _tiny_forecast_frame()
    frame["q_0p5"] = np.array([2.0, 3.0, 5.0, 5.0], dtype=np.float64)

    with pytest.raises(ValueError, match="quantile columns remain unreconciled"):
        NixtlaReconciler("ols", method_factory=_CountingMethod)(
            frame,
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(),
        )


def test_residual_sidecar_lookup_is_not_horizon_specific() -> None:
    fitted = _tiny_fitted_values()
    frame_h1 = _tiny_forecast_frame()
    frame_h2 = _tiny_forecast_frame()
    frame_h2[H] = 2
    frame = pd.concat([frame_h1, frame_h2], ignore_index=True)
    methods, factory = _residual_factory()

    NixtlaReconciler("mint_shrink", method_factory=factory)(
        frame,
        build_hierarchy_index(_tiny_hierarchy()),
        ReconciliationContext(fitted_values=fitted),
    )

    assert len(methods) == 2
    assert methods[0].y_hat_insample is not None
    assert methods[1].y_hat_insample is not None
    np.testing.assert_allclose(methods[1].y_hat_insample, methods[0].y_hat_insample)


class _ResidualFitFailure(_ResidualMethod):
    def fit(
        self,
        *,
        S: np.ndarray,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
    ) -> _ResidualMethod:
        del S, y_hat, tags, y_insample, y_hat_insample
        raise ValueError("library fit failed")


class _ResidualPredictFailure(_ResidualMethod):
    def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]:
        del S, y_hat
        raise ValueError("library predict failed")


class _ResidualMissingMean(_ResidualMethod):
    def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]:
        del S, y_hat
        return {"median": np.zeros((4, 1), dtype=np.float64)}


class _ResidualWrongShape(_ResidualMethod):
    def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]:
        del S, y_hat
        return {"mean": np.zeros((3, 1), dtype=np.float64)}


def test_residual_fit_failure_is_wrapped_with_context() -> None:
    with pytest.raises(
        RuntimeError,
        match="fit failed .*strategy='mint_shrink'.*model_name='m'.*forecast_origin=.*h=1",
    ):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualFitFailure)(
            _tiny_forecast_frame(),
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )


def test_residual_predict_failure_is_wrapped_with_context() -> None:
    with pytest.raises(
        RuntimeError,
        match="predict failed .*strategy='mint_shrink'.*model_name='m'.*forecast_origin=.*h=1",
    ):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualPredictFailure)(
            _tiny_forecast_frame(),
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )


def test_residual_predict_requires_mean_key() -> None:
    with pytest.raises(ValueError, match="missing 'mean'.*model_name='m'.*forecast_origin=.*h=1"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualMissingMean)(
            _tiny_forecast_frame(),
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )


def test_residual_predict_requires_expected_mean_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(3, 1\); expected \(4, 1\)"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualWrongShape)(
            _tiny_forecast_frame(),
            build_hierarchy_index(_tiny_hierarchy()),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )


# ---------------------------------------------------------------------------
# Sparse point path (#168): producer-selection seam, csr layout, cache key,
# and the bicgstab convergence guard.
# ---------------------------------------------------------------------------


def _four_bottom_hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["a", "b", "c", "d"],
            "dept": ["D1", "D1", "D2", "D2"],
            "store": ["S1", "S2", "S1", "S2"],
        }
    )


def _coherent_tiny_forecast_frame() -> pd.DataFrame:
    frame = _tiny_forecast_frame()
    frame[Y_HAT] = np.array([2.0, 3.0, 5.0, 5.0], dtype=np.float64)
    return frame


@pytest.mark.parametrize("strategy", ["ols", "wls_struct", "wls_var"])
def test_sparse_roster_build_summing_selects_sparse_producer(strategy: str) -> None:
    index = build_hierarchy_index(_hierarchy())
    assert isinstance(NixtlaReconciler(strategy).build_summing(index), SparseSummingMatrix)


@pytest.mark.parametrize("strategy", ["mint_shrink", "erm"])
def test_dense_roster_build_summing_keeps_dense_producer(strategy: str) -> None:
    index = build_hierarchy_index(_hierarchy())
    assert isinstance(NixtlaReconciler(strategy).build_summing(index), SummingMatrix)


def test_dense_producer_never_invoked_for_sparse_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seam liveness: a roster strategy must never materialize the dense S."""

    def _fail(hierarchy_index):
        raise AssertionError("dense summing producer must not run for a sparse-capable strategy")

    monkeypatch.setattr("calibre.reconciliation.apply.summing_matrix_from_index", _fail)
    _methods, factory = _counting_factory()

    out = NixtlaReconciler("ols", method_factory=factory)(
        _coherent_tiny_forecast_frame(),
        build_hierarchy_index(_tiny_hierarchy()),
        ReconciliationContext(),
    )

    assert len(out) == 4


def test_sparse_layout_keeps_csr_and_identity_last_block() -> None:
    summing = sparse_summing_matrix_from_index(build_hierarchy_index(_hierarchy()))
    base = summing.S @ np.array([1.0, 2.0, 3.0])

    layout = _to_nixtla_layout(base, summing)

    assert isinstance(layout.S, sparse.csr_array)
    np.testing.assert_array_equal(
        layout.S[-summing.n_bottom :].toarray(), np.eye(summing.n_bottom)
    )
    np.testing.assert_array_equal(_from_nixtla_layout(layout.y_hat, layout), base)


def test_sparse_projection_cache_hits_same_hierarchy_and_misses_different() -> None:
    methods, factory = _counting_factory()
    reconciler = NixtlaReconciler("ols", method_factory=factory)
    first = sparse_summing_matrix_from_index(build_hierarchy_index(_hierarchy()))
    # A fresh producer call over equal hierarchy facts: distinct object, same key.
    rebuilt = sparse_summing_matrix_from_index(build_hierarchy_index(_hierarchy()))
    other = sparse_summing_matrix_from_index(build_hierarchy_index(_four_bottom_hierarchy()))

    reconciler.reconcile_vector(first.S @ np.array([1.0, 2.0, 3.0]), first)
    reconciler.reconcile_vector(rebuilt.S @ np.array([4.0, 5.0, 6.0]), rebuilt)
    assert len(methods) == 1

    reconciler.reconcile_vector(other.S @ np.array([1.0, 2.0, 3.0, 4.0]), other)
    assert len(methods) == 2


def test_sparse_cache_key_is_deterministic_across_construction_orders() -> None:
    summing = sparse_summing_matrix_from_index(build_hierarchy_index(_hierarchy()))
    coo = summing.S.tocoo()
    order = np.random.default_rng(0).permutation(coo.nnz)
    shuffled = SparseSummingMatrix(
        S=sparse.csr_array(
            (coo.data[order], (coo.row[order], coo.col[order])), shape=summing.S.shape
        ),
        bottom_ids=summing.bottom_ids,
        node_labels=summing.node_labels,
    )

    assert _cache_key(summing) == _cache_key(shuffled)


def test_starved_bicgstab_solve_raises_convergence_guard() -> None:
    """KTD-4: hierarchicalforecast silently returns the best-effort iterate and
    the output is S-coherent by construction, so the Calibre-owned checked
    method is the only convergence signal — a starved solve must raise."""
    _require_hierarchy_extra()
    summing = sparse_summing_matrix_from_index(build_hierarchy_index(_four_bottom_hierarchy()))
    base = np.arange(1.0, summing.n_nodes + 1.0, dtype=np.float64)
    layout = _to_nixtla_layout(base, summing)
    method = _make_checked_min_trace_sparse("wls_struct", bicgstab_maxiter=1)
    method.fit(S=layout.S, y_hat=layout.y_hat, tags=layout.tags)

    with pytest.raises(RuntimeError, match="bicgstab solve failed"):
        method.predict(S=layout.S, y_hat=layout.y_hat)


def test_unstarved_checked_solve_converges_and_matches_closed_form() -> None:
    """The checked variant changes nothing on a healthy solve: it agrees with
    the dense closed form S (S' W^-1 S)^-1 S' W^-1 y within solver tolerance."""
    _require_hierarchy_extra()
    index = build_hierarchy_index(_four_bottom_hierarchy())
    summing = sparse_summing_matrix_from_index(index)
    dense = build_summing_matrix(_four_bottom_hierarchy())
    base = np.arange(1.0, summing.n_nodes + 1.0, dtype=np.float64)

    reconciled = NixtlaReconciler("wls_struct").reconcile_vector(base, summing)

    w_diag = dense.S @ np.ones(dense.n_bottom)
    weighted = dense.S.T / w_diag
    bottom = np.linalg.solve(weighted @ dense.S, weighted @ base)
    np.testing.assert_allclose(reconciled, dense.S @ bottom, rtol=1e-3, atol=1e-6)
