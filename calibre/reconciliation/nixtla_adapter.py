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
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

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
from calibre.reconciliation.summing import (
    HierarchyIndex,
    SparseSummingMatrix,
    SummingMatrixLike,
    sparse_summing_matrix_from_index,
)

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
# Strategies with an upstream sparse implementation: ``BottomUpSparse`` serves
# the fused bottom_up path (the point adapter never requests bottom_up — the
# native BottomUpReconciler owns that), and ``MinTraceSparse`` serves exactly
# its allowed-method set ols/wls_struct/wls_var. ``erm`` and ``mint_shrink``
# have no sparse variant upstream, so they keep the dense path and its
# documented memory ceiling. Single source of truth consumed by the point
# producer-selection seam (``NixtlaReconciler.build_summing``), the fused
# phase's S build, and the preflight memory estimate.
NIXTLA_SPARSE_STRATEGIES: frozenset[NixtlaStrategy] = frozenset(
    {"bottom_up", "ols", "wls_struct", "wls_var"}
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

# scipy's LinearOperator dispatches through __new__ to a private subclass when
# called as LinearOperator(shape, matvec=...); the declared __init__ signature
# (dtype, shape) confuses static analysis, so the functional form goes through
# this alias.
_linear_operator = cast(Any, sparse_linalg.LinearOperator)


class _NixtlaMethod(Protocol):
    def fit(
        self,
        *,
        S: np.ndarray | sparse.csr_array,
        y_hat: np.ndarray,
        tags: dict[str, np.ndarray],
        y_insample: np.ndarray | None = None,
        y_hat_insample: np.ndarray | None = None,
    ) -> _NixtlaMethod: ...

    def predict(
        self, *, S: np.ndarray | sparse.csr_array, y_hat: np.ndarray
    ) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class _NixtlaLayout:
    S: np.ndarray | sparse.csr_array
    y_hat: np.ndarray
    tags: dict[str, np.ndarray]
    inverse_permutation: np.ndarray


_CacheKey = tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], bytes]
_InsampleCacheKey = tuple[str, _CacheKey]


def _load_methods_module() -> Any:
    try:
        return importlib.import_module("hierarchicalforecast.methods")
    except ImportError as exc:  # pragma: no cover - covered with import monkeypatch
        raise RuntimeError(
            "hierarchicalforecast is not installed. Install calibre with the "
            "'hierarchy' extra: pip install calibre[hierarchy]"
        ) from exc


def _make_checked_min_trace_sparse(
    strategy: NixtlaStrategy,
    *,
    bicgstab_maxiter: int | None = None,
) -> _NixtlaMethod:
    """Build a ``MinTraceSparse`` whose P-action raises on a bad bicgstab solve.

    hierarchicalforecast 1.5.1 discards bicgstab's ``exit_code`` inside
    ``MinTraceSparse._get_PW_matrices`` and silently returns the best-effort
    iterate, and the reconciled output ``S @ (P @ y_hat)`` is coherent **by
    construction**, so neither the adapter's S-coherence self-check nor a
    completed run can detect a non-converged solve. Calibre owns the guard
    instead: this subclass rebuilds the upstream P-action with the same
    operators and tolerance, and raises when bicgstab reports non-convergence
    (positive exit code) or breakdown (negative exit code).
    ``bicgstab_maxiter`` exists so tests can starve the solver and prove the
    guard fires; production callers leave it at scipy's default.
    """
    methods = _load_methods_module()

    class _CheckedMinTraceSparse(methods.MinTraceSparse):
        def __init__(self) -> None:
            super().__init__(method=strategy, num_threads=1)

        def _get_PW_matrices(
            self,
            S: Any,
            y_hat: np.ndarray,
            y_insample: np.ndarray | None = None,
            y_hat_insample: np.ndarray | None = None,
        ) -> tuple[Any, Any]:
            # Reuse the upstream weight derivation (and its validation); only
            # the returned P-action is replaced with the checked variant.
            _unchecked_p, W = super()._get_PW_matrices(
                S=S,
                y_hat=y_hat,
                y_insample=y_insample,
                y_hat_insample=y_hat_insample,
            )
            # csr_matrix (not csr_array) deliberately mirrors the upstream
            # MinTraceSparse P-action arithmetic this closure replaces — the
            # checked variant must replay the exact same operators.
            S_csr = sparse.csr_matrix(S)
            w_diag = np.asarray(W.diagonal(), dtype=np.float64)
            R = sparse.csr_matrix(
                S_csr.T @ sparse.spdiags(np.reciprocal(w_diag), 0, w_diag.size, w_diag.size)
            )

            def checked_p_action(y: np.ndarray) -> np.ndarray:
                b = R @ y
                A = _linear_operator((b.size, b.size), matvec=lambda v: R @ (S_csr @ v))
                x_tilde, exit_code = sparse_linalg.bicgstab(
                    A, b, atol=1e-5, maxiter=bicgstab_maxiter
                )
                if exit_code != 0:
                    reason = (
                        "no convergence within maxiter"
                        if exit_code > 0
                        else "breakdown or illegal input"
                    )
                    raise RuntimeError(
                        f"MinTraceSparse({strategy!r}) bicgstab solve failed: "
                        f"exit_code={exit_code} ({reason}). hierarchicalforecast "
                        "returns the best-effort iterate silently and the reconciled "
                        "output is coherent by construction, so this guard is the "
                        "only convergence signal for the sparse path."
                    )
                return x_tilde

            P = _linear_operator((S_csr.shape[1], y_hat.shape[0]), matvec=checked_p_action)
            return P, W

    return cast(_NixtlaMethod, _CheckedMinTraceSparse())


def _default_method_factory(strategy: NixtlaStrategy) -> Callable[[], _NixtlaMethod]:
    def _make_method() -> _NixtlaMethod:
        methods = _load_methods_module()
        if strategy == "bottom_up":
            # Fused-phase-only mapping: NIXTLA_POINT_STRATEGIES excludes
            # bottom_up, so the sparse variant here affects only the fused
            # interval caller.
            return cast(_NixtlaMethod, methods.BottomUpSparse())
        if strategy == "erm":
            return cast(_NixtlaMethod, methods.ERM(method="closed"))
        if strategy in NIXTLA_SPARSE_STRATEGIES:
            return _make_checked_min_trace_sparse(strategy)
        # mint_shrink has no sparse implementation upstream; dense MinTrace stays.
        return cast(_NixtlaMethod, methods.MinTrace(method=strategy, num_threads=1))

    return _make_method


def make_nixtla_method(strategy: NixtlaStrategy) -> _NixtlaMethod:
    """Build the Nixtla method object for Calibre's supported strategy names."""

    return _default_method_factory(strategy)()


def _to_nixtla_layout(base: np.ndarray, summing: SummingMatrixLike) -> _NixtlaLayout:
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
    if isinstance(summing, SparseSummingMatrix):
        # Row-permuting a csr_array keeps it csr (no densify); the sparse
        # methods coerce to csr_matrix internally.
        S: np.ndarray | sparse.csr_array = summing.S[permutation]
    else:
        S = np.asarray(summing.S[permutation], dtype=np.float64)
    return _NixtlaLayout(
        S=S,
        y_hat=np.asarray(base[permutation], dtype=np.float64).reshape(-1, 1),
        tags=tags,
        inverse_permutation=inverse,
    )


def _from_nixtla_layout(y_hat: np.ndarray, layout: _NixtlaLayout) -> np.ndarray:
    return np.asarray(y_hat, dtype=np.float64).reshape(-1)[layout.inverse_permutation]


def _cache_key(summing: SummingMatrixLike) -> _CacheKey:
    if isinstance(summing, SparseSummingMatrix):
        # Structure fully determines the sparse matrix: every stored value is
        # exactly 1.0 (asserted by the producer), so the digest hashes only
        # the csr index arrays (~850 KB at full M5) instead of the dense
        # bytes (7.6 GiB) the dense branch below would hash.
        S = summing.S
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(S.indices.tobytes())
        hasher.update(S.indptr.tobytes())
        digest = hasher.digest()
        shape = (int(S.shape[0]), int(S.shape[1]))
        return (tuple(summing.bottom_ids), tuple(summing.node_labels), shape, digest)
    dense = np.ascontiguousarray(summing.S, dtype=np.float64)
    digest = hashlib.blake2b(dense.tobytes(), digest_size=16).digest()
    return (tuple(summing.bottom_ids), tuple(summing.node_labels), dense.shape, digest)


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
        hierarchy_index: HierarchyIndex | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        if hierarchy_index is not None and not frame.empty:
            reject_quantile_columns(frame, strategy="Nixtla")
        return super().__call__(frame, hierarchy_index, context)

    def build_summing(self, hierarchy_index: HierarchyIndex) -> SummingMatrixLike:
        """Producer-selection seam: csr S for the sparse-capable roster.

        The base harness builds the summing matrix upstream of any
        per-strategy code, so the roster is consulted here — without this
        override the full dense S would be materialized before
        ``reconcile_vector`` ever runs and the sparse methods would never see
        a csr input.
        """
        if self.strategy in NIXTLA_SPARSE_STRATEGIES:
            return sparse_summing_matrix_from_index(hierarchy_index)
        return super().build_summing(hierarchy_index)

    def prepare_reconcile(
        self,
        summing: SummingMatrixLike,
        context: ReconciliationContext,
    ) -> _ResidualReconcileState | None:
        del summing
        if not self.requires_fitted_values:
            return None
        return _ResidualReconcileState(
            fitted_values=_validated_fitted_context(context),
            insample_cache={},
        )

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrixLike) -> np.ndarray:
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
        # Alignment guard only: for the sparse methods the output is coherent
        # by construction (S @ (P @ y_hat)), so this check is not a
        # solver-quality signal — convergence is guarded inside the checked
        # sparse method itself.
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
        # Alignment guard only (see reconcile_vector): not a solver-quality
        # signal for the sparse methods.
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
    summing: SummingMatrixLike,
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
