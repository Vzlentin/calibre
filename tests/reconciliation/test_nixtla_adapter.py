from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from calibre.reconciliation.nixtla_adapter import (
    NixtlaReconciler,
    _from_nixtla_layout,
    _to_nixtla_layout,
)
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


@pytest.mark.parametrize("strategy", ["ols", "wls_struct"])
def test_min_trace_strategy_reconciles_small_lattice_to_coherent_vector(strategy: str) -> None:
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


def test_missing_hierarchy_extra_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(name: str) -> Any:
        del name
        raise ImportError("missing")

    monkeypatch.setattr(
        "calibre.reconciliation.nixtla_adapter.importlib.import_module",
        _raise_import_error,
    )

    with pytest.raises(RuntimeError, match=r"Install calibre with the 'hierarchy' extra"):
        NixtlaReconciler("ols")
