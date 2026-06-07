"""Nixtla-backed point reconciler behind Calibre's reconciliation seam."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np

from calibre.reconciliation.apply import VectorReconciler
from calibre.reconciliation.summing import SummingMatrix

NixtlaStrategy = Literal["bottom_up", "ols", "wls_struct"]

_SUPPORTED_STRATEGIES = frozenset({"bottom_up", "ols", "wls_struct"})
_COHERENCE_RTOL = 1e-6
_COHERENCE_ATOL = 1e-6
_DEFAULT_MAX_CACHE_SIZE = 128


class _NixtlaMethod(Protocol):
    def fit(
        self,
        *,
        S: np.ndarray,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
    ) -> _NixtlaMethod: ...

    def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class _NixtlaLayout:
    S: np.ndarray
    y_hat: np.ndarray
    tags: dict[str, np.ndarray]
    inverse_permutation: np.ndarray


_CacheKey = tuple[tuple[str, ...], tuple[str, ...]]


def _load_method_classes() -> tuple[type[Any], type[Any]]:
    try:
        methods = importlib.import_module("hierarchicalforecast.methods")
    except ImportError as exc:  # pragma: no cover - covered with import monkeypatch
        raise RuntimeError(
            "hierarchicalforecast is not installed. Install calibre with the "
            "'hierarchy' extra: pip install calibre[hierarchy]"
        ) from exc
    return methods.BottomUp, methods.MinTrace


def _default_method_factory(strategy: NixtlaStrategy) -> Callable[[], _NixtlaMethod]:
    bottom_up_cls, min_trace_cls = _load_method_classes()
    if strategy == "bottom_up":
        return lambda: cast(_NixtlaMethod, bottom_up_cls())
    return lambda: cast(_NixtlaMethod, min_trace_cls(method=strategy, num_threads=1))


def _to_nixtla_layout(base: np.ndarray, summing: SummingMatrix) -> _NixtlaLayout:
    """Convert Calibre's identity-first S layout to Nixtla's identity-last layout."""
    bottom_idx = np.arange(summing.n_bottom, dtype=np.int64)
    aggregate_idx = np.arange(summing.n_bottom, summing.n_nodes, dtype=np.int64)
    permutation = np.concatenate([aggregate_idx, bottom_idx])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=np.int64)

    tags = {
        "aggregate": np.arange(aggregate_idx.size, dtype=np.int64),
        "bottom": np.arange(aggregate_idx.size, summing.n_nodes, dtype=np.int64),
    }
    return _NixtlaLayout(
        S=np.asarray(summing.S[permutation], dtype=np.float64),
        y_hat=np.asarray(base[permutation], dtype=np.float64).reshape(-1, 1),
        tags=tags,
        inverse_permutation=inverse,
    )


def _from_nixtla_layout(y_hat: np.ndarray, layout: _NixtlaLayout) -> np.ndarray:
    return np.asarray(y_hat, dtype=np.float64).reshape(-1)[layout.inverse_permutation]


def _cache_key(summing: SummingMatrix) -> _CacheKey:
    return (tuple(summing.bottom_ids), tuple(summing.node_labels))


class NixtlaReconciler(VectorReconciler):
    """Delegate point reconciliation projection math to ``hierarchicalforecast``."""

    def __init__(
        self,
        strategy: NixtlaStrategy,
        *,
        method_factory: Callable[[], _NixtlaMethod] | None = None,
        max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    ) -> None:
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"unsupported Nixtla reconciliation strategy: {strategy!r}. "
                f"Available: {sorted(_SUPPORTED_STRATEGIES)}"
            )
        if max_cache_size < 1:
            raise ValueError("max_cache_size must be at least 1")
        self.strategy = strategy
        self._method_factory = method_factory or _default_method_factory(strategy)
        self._max_cache_size = max_cache_size
        self._cache: OrderedDict[_CacheKey, _NixtlaMethod] = OrderedDict()

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        layout = _to_nixtla_layout(base, summing)
        signature = _cache_key(summing)
        reconciler = self._cache.get(signature)
        if reconciler is None:
            reconciler = self._method_factory()
            reconciler.fit(S=layout.S, y_hat=layout.y_hat, tags=layout.tags)
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)
            self._cache[signature] = reconciler
        else:
            self._cache.move_to_end(signature)

        predicted = reconciler.predict(S=layout.S, y_hat=layout.y_hat)["mean"]
        reconciled = _from_nixtla_layout(predicted, layout)
        coherent = summing.S @ reconciled[: summing.n_bottom]
        if not np.allclose(reconciled, coherent, rtol=_COHERENCE_RTOL, atol=_COHERENCE_ATOL):
            raise ValueError("Nixtla reconciliation produced an incoherent forecast vector")
        return reconciled
