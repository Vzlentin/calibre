"""Cross-section apply harness shared by every vector-based reconciler.

:class:`VectorReconciler` implements the frame-to-frame mechanics the
``Reconciler`` Protocol requires once, so each concrete strategy only supplies a
pure ``reconcile_vector`` hook. The harness groups a forecast frame into
``(model_name, forecast_origin, h)`` cross-sections, aligns each supplied node
vector to a subset summing matrix, delegates to the strategy, and writes the
reconciled node forecasts back **in place** while preserving the frame's node
row-set, order, and dtypes.

The base vector handed to ``reconcile_vector`` spans **all supplied required
nodes** (bottom + aggregates), aligned to the applicable
``summing.node_labels`` subset. Aggregate entries come from forecast-frame rows,
not from ``S @ bottom``; missing required nodes fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    is_quantile_column,
)
from calibre.reconciliation.protocols import ReconciliationContext
from calibre.reconciliation.summing import SummingMatrix, build_summing_matrix

_GROUP_KEYS = [MODEL_NAME, FORECAST_ORIGIN, H]
_ORDER_COL = "__calibre_reconcile_order__"


def reject_quantile_columns(frame: pd.DataFrame, *, strategy: str) -> None:
    """Point reconciliation rewrites ``y_hat`` only; quantile columns would drift."""
    quantile_cols = [str(column) for column in frame.columns if is_quantile_column(str(column))]
    if quantile_cols:
        raise ValueError(
            f"{strategy} hierarchy reconciliation only reconciles the point forecast column "
            f"{Y_HAT!r}; quantile columns remain unreconciled and are not supported: "
            f"{quantile_cols}"
        )


@dataclass(frozen=True, slots=True)
class ReconciliationCrossSection:
    """One forecast cross-section aligned to a summing-matrix subset."""

    group: pd.DataFrame
    summing: SummingMatrix
    subset: SummingMatrix
    base: np.ndarray
    context: ReconciliationContext
    state: Any = None


class VectorReconciler:
    """Base class turning a per-cross-section vector transform into a Reconciler.

    Subclasses implement :meth:`reconcile_vector`, mapping a node-level base
    vector (aligned to ``summing.node_labels``) to a coherent node-level vector
    of the same length.
    """

    requires_fitted_values = False

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        if hierarchy is None or frame.empty:
            return frame
        summing = build_summing_matrix(hierarchy)
        state = self.prepare_reconcile(summing, context)
        order_col = _ORDER_COL
        while order_col in frame.columns:
            order_col = f"_{order_col}"
        ordered = frame.copy()
        ordered[order_col] = np.arange(len(ordered), dtype=np.int64)
        parts = [
            self._reconcile_group(group, summing, context=context, state=state)
            for _, group in ordered.groupby(_GROUP_KEYS, sort=False)
        ]
        return pd.concat(parts).sort_values(order_col, kind="stable").drop(columns=order_col)

    def prepare_reconcile(
        self,
        summing: SummingMatrix,
        context: ReconciliationContext,
    ) -> Any:
        """Prepare state shared across cross-sections for one frame apply."""
        del summing, context
        return None

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        """Map a node-level base vector to a coherent node-level vector.

        ``base`` and the return value are both length ``summing.n_nodes`` and
        aligned to ``summing.node_labels`` (bottom identity block first).
        """
        raise NotImplementedError

    def reconcile_cross_section(self, cross_section: ReconciliationCrossSection) -> np.ndarray:
        """Map one aligned cross-section to a coherent vector.

        The default preserves the simple vector strategy surface. Subclasses
        that need sidecar/context data can override this richer hook.
        """
        return self.reconcile_vector(cross_section.base, cross_section.subset)

    def _base_vector(
        self,
        group: pd.DataFrame,
        summing: SummingMatrix,
    ) -> tuple[SummingMatrix, np.ndarray]:
        uid_str = group[UNIQUE_ID].astype(str)
        duplicates = uid_str[uid_str.duplicated()].unique()
        if len(duplicates) > 0:
            raise ValueError(
                "forecast cross-section contains duplicate hierarchy node rows: "
                f"{sorted(duplicates)}"
            )

        supplied = set(uid_str)
        known = set(summing.node_labels)
        unknown = supplied - known
        if unknown:
            raise ValueError(
                "forecast contains hierarchy node(s) not present in the summing matrix: "
                f"{sorted(unknown)}"
            )

        bottom_ids = set(summing.bottom_ids)
        supplied_bottom = [uid for uid in uid_str if uid in bottom_ids]
        if not supplied_bottom:
            raise ValueError("forecast cross-section contains no bottom-level hierarchy rows")

        subset = summing.subset(supplied_bottom)
        required = set(subset.node_labels)
        missing = required - supplied
        if missing:
            context = {
                key: group[key].iloc[0]
                for key in _GROUP_KEYS
                if key in group.columns and not group.empty
            }
            raise ValueError(
                "forecast cross-section missing required hierarchy node forecast(s): "
                f"{sorted(missing)} for {context}"
            )
        extra = supplied - required
        if extra:
            raise ValueError(
                "forecast contains aggregate node row(s) outside the present bottom subset: "
                f"{sorted(extra)}"
            )

        yhat_by_id = dict(zip(uid_str, group[Y_HAT].astype(np.float64), strict=True))
        base = np.array([yhat_by_id[label] for label in subset.node_labels], dtype=np.float64)
        return subset, base

    def _coherent_vector(
        self,
        group: pd.DataFrame,
        summing: SummingMatrix,
        *,
        context: ReconciliationContext,
        state: Any,
    ) -> tuple[SummingMatrix, np.ndarray]:
        subset, base = self._base_vector(group, summing)
        cross_section = ReconciliationCrossSection(
            group=group,
            summing=summing,
            subset=subset,
            base=base,
            context=context,
            state=state,
        )
        return subset, self.reconcile_cross_section(cross_section)

    def _reconcile_group(
        self,
        group: pd.DataFrame,
        summing: SummingMatrix,
        *,
        context: ReconciliationContext,
        state: Any,
    ) -> pd.DataFrame:
        subset, coherent = self._coherent_vector(
            group,
            summing,
            context=context,
            state=state,
        )
        reconciled_by_node = dict(zip(subset.node_labels, coherent, strict=True))
        result = group.copy()
        result[Y_HAT] = result[UNIQUE_ID].astype(str).map(reconciled_by_node).astype(np.float64)
        return result
