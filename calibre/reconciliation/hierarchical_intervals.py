"""Fused hierarchical conformal interval phase.

This module owns Calibre's interim Nixtla hierarchical interval path. It is
separate from the point-only ``Reconciler`` seam because Nixtla conformal
intervals need future forecasts, in-sample fitted values, hierarchy metadata,
and interval normalization in one operation.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    DS,
    FITTED_Y_HAT,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    Y,
    interval_column_names,
    validate_fitted_values_frame,
)
from calibre.reconciliation.nixtla_adapter import (
    NIXTLA_SPARSE_STRATEGIES,
    NIXTLA_STRATEGIES,
    NixtlaStrategy,
    make_nixtla_method,
)
from calibre.reconciliation.summing import (
    HierarchyIndex,
    SparseSummingMatrix,
    SummingMatrixLike,
    sparse_summing_matrix_from_index,
    summing_matrix_from_index,
)

HierarchicalIntervalMethod = Literal["nixtla_conformal"]
_NIXTLA_MODEL_COL = "__calibre_model__"
_METADATA_METHOD = "nixtla_conformal"
_METADATA_MODE = "hierarchical_marginal"
_GROUP_KEYS = [MODEL_NAME, FORECAST_ORIGIN]


@dataclass(frozen=True, slots=True)
class HierarchicalIntervalOptions:
    """Options for the fused hierarchical interval phase."""

    method: HierarchicalIntervalMethod = "nixtla_conformal"
    coverage: float = 0.9
    strategy: NixtlaStrategy = "bottom_up"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method != "nixtla_conformal":
            raise ValueError("hierarchical interval method must be 'nixtla_conformal'")
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must be between 0 and 1")
        strategy = str(self.strategy).strip().lower()
        if strategy not in NIXTLA_STRATEGIES:
            raise ValueError(
                f"unsupported Nixtla hierarchical interval strategy: {self.strategy!r}. "
                f"Available: {sorted(NIXTLA_STRATEGIES)}"
            )
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "strategy", cast(NixtlaStrategy, strategy))


@dataclass(frozen=True)
class HierarchicalIntervalContext:
    """Per-origin sidecars required by the fused interval phase."""

    fitted_values: pd.DataFrame | None = None


class HierarchicalIntervalPhase(Protocol):
    """Fused hierarchy + interval phase used between Predict and Order."""

    requires_fitted_values: bool

    def apply(
        self,
        frame: pd.DataFrame,
        hierarchy_index: HierarchyIndex,
        context: HierarchicalIntervalContext,
    ) -> pd.DataFrame: ...


@lru_cache(maxsize=1)
def _load_reconciliation_class() -> type[Any]:
    try:
        core = importlib.import_module("hierarchicalforecast.core")
    except ImportError as exc:  # pragma: no cover - covered with import monkeypatch
        raise RuntimeError(
            "hierarchicalforecast is not installed. Install calibre with the "
            "'hierarchy' extra: uv sync --extra hierarchy"
        ) from exc
    return core.HierarchicalReconciliation


def _make_reconciliation(method: Any) -> Any:
    return _load_reconciliation_class()([method])


def _level_value(coverage: float) -> float:
    return coverage * 100.0


class NixtlaHierarchicalIntervalPhase:
    """Apply Nixtla marginal conformal intervals and normalize to Calibre columns."""

    requires_fitted_values = True

    def __init__(self, options: HierarchicalIntervalOptions) -> None:
        self.options = options

    def apply(
        self,
        frame: pd.DataFrame,
        hierarchy_index: HierarchyIndex,
        context: HierarchicalIntervalContext,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        fitted = _validated_fitted_values(context)
        fitted_by_model = _fitted_values_by_model(fitted)
        # Producer-selection mirror of the point seam: the sparse-capable
        # roster (bottom_up/ols/wls_struct/wls_var via BottomUpSparse /
        # MinTraceSparse) never materializes the dense S; erm and mint_shrink
        # have no upstream sparse variant and keep the dense build.
        summing = (
            sparse_summing_matrix_from_index(hierarchy_index)
            if self.options.strategy in NIXTLA_SPARSE_STRATEGIES
            else summing_matrix_from_index(hierarchy_index)
        )
        parts = [
            self._apply_model_group(group, summing, fitted_by_model)
            for _, group in frame.groupby(_GROUP_KEYS, sort=False)
        ]
        return pd.concat(parts, ignore_index=True)

    def _apply_model_group(
        self,
        group: pd.DataFrame,
        summing: SummingMatrixLike,
        fitted_by_model: Mapping[str, pd.DataFrame],
    ) -> pd.DataFrame:
        model_name = str(group[MODEL_NAME].iloc[0])
        subset = _subset_for_group(group, summing)
        y_hat_df = _to_nixtla_forecast_df(group, subset)
        y_df = _to_nixtla_fitted_df(fitted_by_model, subset, model_name)
        method = make_nixtla_method(self.options.strategy)
        reconciliation = _make_reconciliation(method)
        forecast_origin = group[FORECAST_ORIGIN].iloc[0]
        try:
            reconciled = reconciliation.reconcile(
                Y_hat_df=y_hat_df,
                S_df=_to_s_df(subset),
                tags=_to_tags(subset),
                Y_df=y_df,
                level=[_level_value(self.options.coverage)],
                intervals_method="conformal",
                seed=self.options.seed,
                is_balanced=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Nixtla hierarchical conformal reconciliation failed "
                f"for strategy={self.options.strategy!r}, model_name={model_name!r}, "
                f"forecast_origin={forecast_origin!r}, coverage={self.options.coverage!r}"
            ) from exc
        return _normalize_nixtla_output(
            group,
            reconciled,
            coverage=self.options.coverage,
        )


def _validated_fitted_values(context: HierarchicalIntervalContext) -> pd.DataFrame:
    fitted = context.fitted_values
    if fitted is None or fitted.empty:
        raise ValueError(
            "Hierarchical conformal intervals require in-sample fitted values; "
            "no fitted-value sidecar was provided"
        )
    validate_fitted_values_frame(fitted)
    return fitted


def _fitted_values_by_model(fitted_values: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(model_name): group.copy()
        for model_name, group in fitted_values.groupby(MODEL_NAME, sort=False)
    }


def _subset_for_group(group: pd.DataFrame, summing: SummingMatrixLike) -> SummingMatrixLike:
    keyed = group[[UNIQUE_ID, DS]].copy()
    keyed[UNIQUE_ID] = keyed[UNIQUE_ID].astype(str)
    keyed[DS] = pd.to_datetime(keyed[DS]).astype("datetime64[ns]")
    duplicates = keyed[keyed.duplicated([UNIQUE_ID, DS], keep=False)]
    if not duplicates.empty:
        duplicate_keys = [
            tuple(row) for row in duplicates.drop_duplicates().itertuples(index=False, name=None)
        ]
        raise ValueError(
            f"forecast contains duplicate hierarchy node rows for a timestamp: {duplicate_keys}"
        )
    supplied = set(keyed[UNIQUE_ID])
    known = set(summing.node_labels)
    unknown = supplied - known
    if unknown:
        raise ValueError(
            "forecast contains hierarchy node(s) not present in the summing matrix: "
            f"{sorted(unknown)}"
        )
    supplied_bottom = [uid for uid in summing.bottom_ids if uid in supplied]
    if not supplied_bottom:
        raise ValueError("forecast cross-section contains no bottom-level hierarchy rows")
    subset = summing.subset(supplied_bottom)
    required = set(subset.node_labels)
    ids_by_ds = {
        ds: set(ds_group[UNIQUE_ID].astype(str)) for ds, ds_group in keyed.groupby(DS, sort=False)
    }
    missing_by_ds = {
        str(ds): sorted(required - ids) for ds, ids in ids_by_ds.items() if required - ids
    }
    if missing_by_ds:
        context = {
            key: group[key].iloc[0]
            for key in _GROUP_KEYS
            if key in group.columns and not group.empty
        }
        raise ValueError(
            "forecast cross-section missing required hierarchy node forecast(s): "
            f"{missing_by_ds} for {context}"
        )
    extra_by_ds = {
        str(ds): sorted(ids - required) for ds, ids in ids_by_ds.items() if ids - required
    }
    if extra_by_ds:
        raise ValueError(
            "forecast contains aggregate node row(s) outside the present bottom subset: "
            f"{extra_by_ds}"
        )
    return subset


def _to_s_df(summing: SummingMatrixLike) -> pd.DataFrame:
    row_order = _nixtla_node_order(summing)
    index_by_label = {label: index for index, label in enumerate(summing.node_labels)}
    permutation = [index_by_label[label] for label in row_order]
    if isinstance(summing, SparseSummingMatrix):
        # Sparse[float64, 0] value columns are the zero-copy precondition for
        # hierarchicalforecast's `.sparse.to_coo()` -> csr path; a plain dense
        # DataFrame would silently densify the full S transiently before
        # re-sparsifying — the exact allocation this path exists to avoid.
        matrix = pd.DataFrame.sparse.from_spmatrix(
            summing.S[permutation], columns=list(summing.bottom_ids)
        )
    else:
        matrix = pd.DataFrame(summing.S[permutation], columns=list(summing.bottom_ids))
    matrix.insert(0, UNIQUE_ID, list(row_order))
    return matrix


def _to_tags(summing: SummingMatrixLike) -> dict[str, np.ndarray]:
    aggregate_labels = np.array(summing.node_labels[summing.n_bottom :], dtype=object)
    return {
        "aggregate": aggregate_labels,
        "bottom": np.array(summing.bottom_ids, dtype=object),
    }


def _nixtla_node_order(summing: SummingMatrixLike) -> tuple[str, ...]:
    """Return Nixtla's expected row order: aggregates first, bottom block last."""
    return (*summing.node_labels[summing.n_bottom :], *summing.bottom_ids)


def _to_nixtla_forecast_df(group: pd.DataFrame, summing: SummingMatrixLike) -> pd.DataFrame:
    ordered = group.copy()
    ordered[UNIQUE_ID] = ordered[UNIQUE_ID].astype(str)
    ordered[DS] = pd.to_datetime(ordered[DS]).astype("datetime64[ns]")
    order = {label: index for index, label in enumerate(_nixtla_node_order(summing))}
    ordered["_node_order"] = ordered[UNIQUE_ID].map(order).astype("int64")
    ordered = ordered.sort_values([DS, "_node_order"], kind="stable")
    return ordered[[UNIQUE_ID, DS, Y_HAT]].rename(columns={Y_HAT: _NIXTLA_MODEL_COL})


def _to_nixtla_fitted_df(
    fitted_by_model: Mapping[str, pd.DataFrame],
    summing: SummingMatrixLike,
    model_name: str,
) -> pd.DataFrame:
    subset = fitted_by_model.get(model_name)
    if subset is None or subset.empty:
        raise ValueError(f"Missing fitted values for model_name={model_name!r}")
    subset = subset.copy()
    subset[UNIQUE_ID] = subset[UNIQUE_ID].astype(str)
    unknown = set(subset[UNIQUE_ID]) - set(summing.node_labels)
    if unknown:
        raise ValueError(
            "Fitted-value sidecar contains hierarchy node(s) not present in the "
            f"active summing matrix for model_name={model_name!r}: {sorted(unknown)}"
        )
    missing_keys = _missing_fitted_keys(subset, summing, model_name)
    if missing_keys:
        preview = missing_keys[:10]
        suffix = "..." if len(missing_keys) > len(preview) else ""
        raise ValueError(
            f"Missing fitted-value sidecar key(s) (unique_id, ds, model_name): {preview}{suffix}"
        )
    order = {label: index for index, label in enumerate(_nixtla_node_order(summing))}
    subset["_node_order"] = subset[UNIQUE_ID].map(order).astype("int64")
    subset = subset.sort_values([DS, "_node_order"], kind="stable")
    return subset[[UNIQUE_ID, DS, Y, FITTED_Y_HAT]].rename(
        columns={FITTED_Y_HAT: _NIXTLA_MODEL_COL}
    )


def _missing_fitted_keys(
    fitted_values: pd.DataFrame,
    summing: SummingMatrixLike,
    model_name: str,
) -> list[tuple[str, str, str]]:
    fitted_values[DS] = pd.to_datetime(fitted_values[DS]).astype("datetime64[ns]")
    observed_ds = sorted(fitted_values[DS].dropna().unique())
    observed = {
        (str(row.unique_id), pd.Timestamp(str(row.ds)))
        for row in fitted_values[[UNIQUE_ID, DS]].itertuples(index=False)
    }
    return [
        (node, str(pd.Timestamp(str(ds))), model_name)
        for ds in observed_ds
        for node in summing.node_labels
        if (node, pd.Timestamp(str(ds))) not in observed
    ]


def _normalize_nixtla_output(
    original: pd.DataFrame,
    nixtla_output: pd.DataFrame,
    *,
    coverage: float,
) -> pd.DataFrame:
    rec_col = _single_reconciled_column(nixtla_output)
    nixtla_lower = _single_interval_output_column(nixtla_output, f"{rec_col}-lo-")
    nixtla_upper = _single_interval_output_column(nixtla_output, f"{rec_col}-hi-")

    normalized = nixtla_output[[UNIQUE_ID, DS, rec_col, nixtla_lower, nixtla_upper]].copy()
    normalized[UNIQUE_ID] = normalized[UNIQUE_ID].astype(str)
    normalized[DS] = pd.to_datetime(normalized[DS]).astype("datetime64[ns]")
    normalized = normalized.set_index([UNIQUE_ID, DS])

    lower_col, upper_col = interval_column_names(coverage)
    result = original.copy()
    lookup_index = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                UNIQUE_ID: result[UNIQUE_ID].astype(str),
                DS: pd.to_datetime(result[DS]).astype("datetime64[ns]"),
            }
        )
    )
    result[Y_HAT] = normalized.loc[lookup_index, rec_col].to_numpy(dtype=np.float64)
    result[lower_col] = normalized.loc[lookup_index, nixtla_lower].to_numpy(dtype=np.float64)
    result[upper_col] = normalized.loc[lookup_index, nixtla_upper].to_numpy(dtype=np.float64)
    result[CONFORMAL_METHOD] = _METADATA_METHOD
    result[CONFORMAL_ALPHA] = float(1.0 - coverage)
    result[CONFORMAL_MODE] = _METADATA_MODE
    return result


def _single_reconciled_column(frame: pd.DataFrame) -> str:
    prefix = f"{_NIXTLA_MODEL_COL}/"
    candidates = [
        str(column)
        for column in frame.columns
        if (
            str(column).startswith(prefix)
            and "-lo-" not in str(column)
            and "-hi-" not in str(column)
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Nixtla hierarchical conformal output must contain exactly one reconciled "
            f"forecast column, got {candidates}"
        )
    return candidates[0]


def _single_interval_output_column(frame: pd.DataFrame, prefix: str) -> str:
    candidates = [str(column) for column in frame.columns if str(column).startswith(prefix)]
    if len(candidates) != 1:
        raise ValueError(
            "Nixtla hierarchical conformal output must contain exactly one interval "
            f"column for prefix {prefix!r}, got {candidates}"
        )
    return candidates[0]
