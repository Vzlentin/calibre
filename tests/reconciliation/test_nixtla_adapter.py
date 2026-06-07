from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
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
)
from calibre.reconciliation.nixtla_adapter import (
    NixtlaReconciler,
    _from_nixtla_layout,
    _to_nixtla_layout,
)
from calibre.reconciliation.protocols import ReconciliationContext
from calibre.reconciliation.summing import SummingMatrix, build_summing_matrix


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


def test_bottom_up_bottom_only_cross_section_keeps_bottom_block() -> None:
    _require_hierarchy_extra()
    summing = SummingMatrix(
        S=np.eye(2, dtype=np.float64),
        bottom_ids=("a", "b"),
        node_labels=("a", "b"),
    )
    base = np.array([4.0, 5.0], dtype=np.float64)

    reconciled = NixtlaReconciler("bottom_up").reconcile_vector(base, summing)

    np.testing.assert_allclose(reconciled[: summing.n_bottom], base)


def test_s_layout_conversion_round_trips_to_identity_first_order() -> None:
    summing = build_summing_matrix(_hierarchy()).subset(["a", "c"])
    base = _coherent_base(summing, [2.0, 7.0])

    layout = _to_nixtla_layout(base, summing)
    round_tripped = _from_nixtla_layout(layout.y_hat, layout)

    np.testing.assert_array_equal(layout.S[-summing.n_bottom :], np.eye(summing.n_bottom))
    np.testing.assert_array_equal(round_tripped, base)


@pytest.mark.parametrize("strategy", ["ols", "wls_struct"])
def test_min_trace_factory_passes_strategy_and_single_thread(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    captured: list[tuple[str, int]] = []

    class _FakeMinTrace(_CountingMethod):
        def __init__(self, *, method: str, num_threads: int) -> None:
            super().__init__()
            captured.append((method, num_threads))

    def _fake_import_module(name: str) -> SimpleNamespace:
        assert name == "hierarchicalforecast.methods"
        return SimpleNamespace(
            BottomUp=_CountingMethod,
            MinTrace=_FakeMinTrace,
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
    ("strategy", "expected_method"),
    [("mint_shrink", "mint_shrink"), ("wls_var", "wls_var")],
)
def test_residual_min_trace_factory_passes_strategy_and_single_thread(
    monkeypatch: pytest.MonkeyPatch, strategy: str, expected_method: str
) -> None:
    captured: list[tuple[str, int]] = []

    class _FakeMinTrace(_ResidualMethod):
        def __init__(self, *, method: str, num_threads: int) -> None:
            super().__init__()
            captured.append((method, num_threads))

    def _fake_import_module(name: str) -> SimpleNamespace:
        assert name == "hierarchicalforecast.methods"
        return SimpleNamespace(
            BottomUp=_CountingMethod,
            MinTrace=_FakeMinTrace,
            ERM=_ResidualMethod,
        )

    monkeypatch.setattr(
        "calibre.reconciliation.nixtla_adapter.importlib.import_module",
        _fake_import_module,
    )

    NixtlaReconciler(strategy)(
        _tiny_forecast_frame(),
        _tiny_hierarchy(),
        ReconciliationContext(fitted_values=_tiny_fitted_values()),
    )

    assert captured == [(expected_method, 1)]


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
        _tiny_hierarchy(),
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
                    H: 1,
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
        _tiny_hierarchy(),
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
            _tiny_hierarchy(),
            ReconciliationContext(),
        )


def test_residual_strategy_names_missing_fitted_node() -> None:
    fitted = _tiny_fitted_values()
    fitted = fitted[fitted[UNIQUE_ID] != "dept=D"]

    with pytest.raises(ValueError, match="dept=D"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualMethod)(
            _tiny_forecast_frame(),
            _tiny_hierarchy(),
            ReconciliationContext(fitted_values=fitted),
        )


def test_residual_strategy_rejects_misaligned_fitted_timestamps() -> None:
    fitted = _tiny_fitted_values()
    fitted = fitted[~((fitted[UNIQUE_ID] == "a") & (fitted[DS] == pd.Timestamp("2024-01-02")))]

    with pytest.raises(ValueError, match="misaligned"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualMethod)(
            _tiny_forecast_frame(),
            _tiny_hierarchy(),
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
            _tiny_hierarchy(),
            ReconciliationContext(),
        )


def test_residual_sidecar_lookup_is_horizon_specific() -> None:
    h1 = _tiny_fitted_values()
    h2 = h1.copy()
    h2[H] = 2
    h2[FITTED_Y_HAT] = h2[FITTED_Y_HAT] + 100.0
    fitted = pd.concat([h1, h2], ignore_index=True)
    frame_h1 = _tiny_forecast_frame()
    frame_h2 = _tiny_forecast_frame()
    frame_h2[H] = 2
    frame = pd.concat([frame_h1, frame_h2], ignore_index=True)
    methods, factory = _residual_factory()

    NixtlaReconciler("mint_shrink", method_factory=factory)(
        frame,
        _tiny_hierarchy(),
        ReconciliationContext(fitted_values=fitted),
    )

    assert len(methods) == 2
    assert methods[0].y_hat_insample is not None
    assert methods[1].y_hat_insample is not None
    np.testing.assert_allclose(methods[1].y_hat_insample - methods[0].y_hat_insample, 100.0)


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
        match="fit failed .*strategy='mint_shrink'.*model_name='m'.*h=1",
    ):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualFitFailure)(
            _tiny_forecast_frame(),
            _tiny_hierarchy(),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )


def test_residual_predict_failure_is_wrapped_with_context() -> None:
    with pytest.raises(
        RuntimeError,
        match="predict failed .*strategy='mint_shrink'.*model_name='m'.*h=1",
    ):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualPredictFailure)(
            _tiny_forecast_frame(),
            _tiny_hierarchy(),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )


def test_residual_predict_requires_mean_key() -> None:
    with pytest.raises(ValueError, match="missing 'mean'.*model_name='m'.*h=1"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualMissingMean)(
            _tiny_forecast_frame(),
            _tiny_hierarchy(),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )


def test_residual_predict_requires_expected_mean_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(3, 1\); expected \(4, 1\)"):
        NixtlaReconciler("mint_shrink", method_factory=_ResidualWrongShape)(
            _tiny_forecast_frame(),
            _tiny_hierarchy(),
            ReconciliationContext(fitted_values=_tiny_fitted_values()),
        )
