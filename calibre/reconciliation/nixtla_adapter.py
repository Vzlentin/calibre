"""Nixtla-backed point reconciler behind Calibre's reconciliation seam."""

from __future__ import annotations

import hashlib
import importlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, get_args

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    FITTED_Y_HAT,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    H,
    Y,
    validate_fitted_values_frame,
)
from calibre.reconciliation.apply import (
    ReconciliationCrossSection,
    VectorReconciler,
    reject_quantile_columns,
)
from calibre.reconciliation.protocols import ReconciliationContext
from calibre.reconciliation.summing import SummingMatrix

NixtlaStrategy = Literal[
    "bottom_up",
    "ols",
    "wls_struct",
    "mint_shrink",
    "wls_var",
    "erm",
]

NIXTLA_STRATEGIES = cast(tuple[NixtlaStrategy, ...], get_args(NixtlaStrategy))
# Point reconciliation strategies served by Nixtla. ``bottom_up`` stays in
# ``NIXTLA_STRATEGIES`` for the fused hierarchical-interval phase, but point
# bottom_up is Calibre's native ``BottomUpReconciler`` — it needs no aggregate
# base forecasts, so routing it through this harness would silently re-impose
# the eager node-history/task cost.
NIXTLA_POINT_STRATEGIES = cast(
    tuple[NixtlaStrategy, ...],
    tuple(strategy for strategy in NIXTLA_STRATEGIES if strategy != "bottom_up"),
)
_SUPPORTED_STRATEGIES = frozenset(NIXTLA_POINT_STRATEGIES)
_RESIDUAL_STRATEGIES = frozenset({"mint_shrink", "wls_var", "erm"})
_UNSUPPORTED_STRATEGY_MESSAGES = {
    "mint_cov": (
        "mint_cov is not exposed by Calibre because the full M5 lattice produces "
        "ill-conditioned covariance estimates; use mint_shrink, wls_var, or erm"
    ),
    "bottom_up": (
        "point bottom_up reconciliation is served by Calibre's native "
        "BottomUpReconciler (resolve_reconciler('bottom_up')), not the Nixtla "
        "harness; hierarchical_intervals.strategy='bottom_up' remains a Nixtla path"
    ),
}
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
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
    ) -> _NixtlaMethod: ...

    def predict(self, *, S: np.ndarray, y_hat: np.ndarray) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class _NixtlaLayout:
    S: np.ndarray
    y_hat: np.ndarray
    tags: dict[str, np.ndarray]
    inverse_permutation: np.ndarray


_CacheKey = tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], bytes]
_InsampleCacheKey = tuple[str, _CacheKey]


def _load_method_classes() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        methods = importlib.import_module("hierarchicalforecast.methods")
    except ImportError as exc:  # pragma: no cover - covered with import monkeypatch
        raise RuntimeError(
            "hierarchicalforecast is not installed. Install calibre with the "
            "'hierarchy' extra: pip install calibre[hierarchy]"
        ) from exc
    return methods.BottomUp, methods.MinTrace, methods.ERM


def _default_method_factory(strategy: NixtlaStrategy) -> Callable[[], _NixtlaMethod]:
    def _make_method() -> _NixtlaMethod:
        bottom_up_cls, min_trace_cls, erm_cls = _load_method_classes()
        if strategy == "bottom_up":
            return cast(_NixtlaMethod, bottom_up_cls())
        if strategy == "erm":
            return cast(_NixtlaMethod, erm_cls(method="closed"))
        return cast(_NixtlaMethod, min_trace_cls(method=strategy, num_threads=1))

    return _make_method


def make_nixtla_method(strategy: NixtlaStrategy) -> _NixtlaMethod:
    """Build the Nixtla method object for Calibre's supported strategy names."""

    return _default_method_factory(strategy)()


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
    S = np.ascontiguousarray(summing.S, dtype=np.float64)
    digest = hashlib.blake2b(S.tobytes(), digest_size=16).digest()
    return (tuple(summing.bottom_ids), tuple(summing.node_labels), S.shape, digest)


class NixtlaReconciler(VectorReconciler):
    """Delegate point reconciliation projection math to ``hierarchicalforecast``."""

    def __init__(
        self,
        strategy: str,
        *,
        method_factory: Callable[[], _NixtlaMethod] | None = None,
        max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    ) -> None:
        strategy_key = str(strategy).strip().lower()
        if strategy_key in _UNSUPPORTED_STRATEGY_MESSAGES:
            raise ValueError(_UNSUPPORTED_STRATEGY_MESSAGES[strategy_key])
        if strategy_key not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"unsupported Nixtla reconciliation strategy: {strategy!r}. "
                f"Available: {sorted(_SUPPORTED_STRATEGIES)}"
            )
        if max_cache_size < 1:
            raise ValueError("max_cache_size must be at least 1")
        self.strategy = cast(NixtlaStrategy, strategy_key)
        self.requires_fitted_values = strategy_key in _RESIDUAL_STRATEGIES
        self._method_factory = method_factory or _default_method_factory(self.strategy)
        self._max_cache_size = max_cache_size
        self._cache: OrderedDict[_CacheKey, _NixtlaMethod] = OrderedDict()

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        if hierarchy is not None and not frame.empty:
            reject_quantile_columns(frame, strategy="Nixtla")
        return super().__call__(frame, hierarchy, context)

    def prepare_reconcile(
        self,
        summing: SummingMatrix,
        context: ReconciliationContext,
    ) -> _ResidualReconcileState | None:
        del summing
        if not self.requires_fitted_values:
            return None
        return _ResidualReconcileState(
            fitted_values=_validated_fitted_context(context),
            insample_cache={},
        )

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

    def reconcile_cross_section(self, cross_section: ReconciliationCrossSection) -> np.ndarray:
        if not self.requires_fitted_values:
            return super().reconcile_cross_section(cross_section)
        if not isinstance(cross_section.state, _ResidualReconcileState):
            raise TypeError("Residual reconciliation state was not prepared")
        return self._reconcile_residual_cross_section(cross_section, cross_section.state)

    def _reconcile_residual_cross_section(
        self,
        cross_section: ReconciliationCrossSection,
        state: _ResidualReconcileState,
    ) -> np.ndarray:
        group = cross_section.group
        subset = cross_section.subset
        model_name = str(group[MODEL_NAME].iloc[0])
        forecast_origin = group[FORECAST_ORIGIN].iloc[0]
        horizon = int(group[H].iloc[0])
        layout = _to_nixtla_layout(cross_section.base, subset)
        cache_key = (model_name, _cache_key(subset))
        residual_layout = state.insample_cache.get(cache_key)
        if residual_layout is None:
            y_insample, y_hat_insample = _insample_arrays(
                state.fitted_values,
                subset,
                model_name,
            )
            residual_layout = _to_nixtla_insample_layout(
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
                inverse_permutation=layout.inverse_permutation,
            )
            state.insample_cache[cache_key] = residual_layout
        reconciler = self._method_factory()
        try:
            reconciler.fit(
                S=layout.S,
                y_hat=layout.y_hat,
                y_insample=residual_layout.y_insample,
                y_hat_insample=residual_layout.y_hat_insample,
                tags=layout.tags,
            )
        except Exception as exc:
            raise RuntimeError(
                "Nixtla residual reconciliation fit failed "
                f"for strategy={self.strategy!r}, model_name={model_name!r}, "
                f"forecast_origin={forecast_origin!r}, h={horizon}"
            ) from exc
        predicted = _predict_residual_mean(
            reconciler,
            layout,
            strategy=self.strategy,
            model_name=model_name,
            forecast_origin=forecast_origin,
            horizon=horizon,
        )
        reconciled = _from_nixtla_layout(predicted, layout)
        coherent = subset.S @ reconciled[: subset.n_bottom]
        if not np.allclose(reconciled, coherent, rtol=_COHERENCE_RTOL, atol=_COHERENCE_ATOL):
            raise ValueError("Nixtla reconciliation produced an incoherent forecast vector")
        return reconciled


def _predict_residual_mean(
    reconciler: _NixtlaMethod,
    layout: _NixtlaLayout,
    *,
    strategy: str,
    model_name: str,
    forecast_origin: object,
    horizon: int,
) -> np.ndarray:
    try:
        predicted = reconciler.predict(S=layout.S, y_hat=layout.y_hat)
    except Exception as exc:
        raise RuntimeError(
            "Nixtla residual reconciliation predict failed "
            f"for strategy={strategy!r}, model_name={model_name!r}, "
            f"forecast_origin={forecast_origin!r}, h={horizon}"
        ) from exc
    if "mean" not in predicted:
        raise ValueError(
            "Nixtla residual reconciliation predict result missing 'mean' "
            f"for strategy={strategy!r}, model_name={model_name!r}, "
            f"forecast_origin={forecast_origin!r}, h={horizon}"
        )
    mean = np.asarray(predicted["mean"], dtype=np.float64)
    if mean.shape != layout.y_hat.shape:
        raise ValueError(
            "Nixtla residual reconciliation predict returned 'mean' with shape "
            f"{mean.shape}; expected {layout.y_hat.shape} "
            f"for strategy={strategy!r}, model_name={model_name!r}, "
            f"forecast_origin={forecast_origin!r}, h={horizon}"
        )
    return mean


@dataclass(frozen=True, slots=True)
class _InsampleLayout:
    y_insample: np.ndarray
    y_hat_insample: np.ndarray


@dataclass(slots=True)
class _ResidualReconcileState:
    fitted_values: pd.DataFrame
    insample_cache: dict[_InsampleCacheKey, _InsampleLayout]


def _validated_fitted_context(context: ReconciliationContext) -> pd.DataFrame:
    fitted = context.fitted_values
    if fitted is None or fitted.empty:
        raise ValueError(
            "Residual reconciliation strategy requires in-sample fitted values; "
            "no fitted-value sidecar was provided"
        )
    validate_fitted_values_frame(fitted)
    return fitted


def _insample_arrays(
    fitted_values: pd.DataFrame,
    summing: SummingMatrix,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset = fitted_values[fitted_values[MODEL_NAME].astype(str) == model_name].copy()
    if subset.empty:
        raise ValueError(f"Missing fitted values for model_name={model_name!r}")
    subset[UNIQUE_ID] = subset[UNIQUE_ID].astype(str)
    node_ids = set(summing.node_labels)
    unknown = set(subset[UNIQUE_ID]) - node_ids
    if unknown:
        subset = subset[~subset[UNIQUE_ID].isin(unknown)].copy()

    grouped_ds = subset.groupby(UNIQUE_ID, sort=False)[DS].agg(lambda values: set(values))
    ds_by_node = {label: grouped_ds.get(label, set()) for label in summing.node_labels}
    missing_nodes = sorted(label for label, ds_values in ds_by_node.items() if not ds_values)
    if missing_nodes:
        raise ValueError(
            "Missing fitted values for hierarchy node(s): "
            f"{missing_nodes} and model_name={model_name!r}"
        )
    common_ds = set.intersection(*ds_by_node.values())
    if not common_ds:
        raise ValueError(
            "Fitted-value timestamps do not overlap across hierarchy nodes "
            f"for model_name={model_name!r}"
        )
    mismatched = {
        label: sorted(str(value) for value in ds_values.symmetric_difference(common_ds))
        for label, ds_values in ds_by_node.items()
        if ds_values != common_ds
    }
    if mismatched:
        raise ValueError(
            "Fitted-value timestamps are misaligned across hierarchy nodes "
            f"for model_name={model_name!r}: {mismatched}"
        )

    ordered_ds = sorted(common_ds)
    rows = subset[subset[DS].isin(ordered_ds)].copy()
    y_wide = rows.pivot(index=DS, columns=UNIQUE_ID, values=Y).reindex(
        index=ordered_ds,
        columns=summing.node_labels,
    )
    fitted_wide = rows.pivot(index=DS, columns=UNIQUE_ID, values=FITTED_Y_HAT).reindex(
        index=ordered_ds,
        columns=summing.node_labels,
    )
    if y_wide.isna().any().any() or fitted_wide.isna().any().any():
        raise ValueError(
            "Fitted-value sidecar cannot be widened without missing "
            f"(unique_id, ds, model_name) keys for model_name={model_name!r}"
        )
    return y_wide.to_numpy(dtype=np.float64).T, fitted_wide.to_numpy(dtype=np.float64).T


def _to_nixtla_insample_layout(
    *,
    y_insample: np.ndarray,
    y_hat_insample: np.ndarray,
    inverse_permutation: np.ndarray,
) -> _InsampleLayout:
    permutation = np.empty_like(inverse_permutation)
    permutation[inverse_permutation] = np.arange(inverse_permutation.size, dtype=np.int64)
    return _InsampleLayout(
        y_insample=np.asarray(y_insample[permutation], dtype=np.float64),
        y_hat_insample=np.asarray(y_hat_insample[permutation], dtype=np.float64),
    )
