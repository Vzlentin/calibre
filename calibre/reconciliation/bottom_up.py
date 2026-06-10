"""Native bottom-up point reconciler.

Point ``bottom_up`` is just ``aggregate = S @ bottom``: it needs no independent
aggregate base forecasts and no projection math. This reconciler therefore
consumes **bottom-only** forecast rows and synthesizes the aggregate node rows
itself, so a bottom-up run never pays for aggregate forecast tasks or eager
node-history expansion. MinT-style strategies (``ols``, ``wls_struct``,
``mint_shrink``, ``wls_var``, ``erm``) stay on the Nixtla-backed vector
reconciler with independent bottom + aggregate base forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.reconciliation.apply import reject_quantile_columns
from calibre.reconciliation.protocols import ReconciliationContext
from calibre.reconciliation.summing import SummingMatrix, build_summing_matrix

_GROUP_KEYS = [MODEL_NAME, FORECAST_ORIGIN, H]


class BottomUpReconciler:
    """Synthesize coherent aggregate node rows from bottom-only point forecasts.

    Each ``(model_name, forecast_origin, h)`` cross-section must contain only
    bottom-level hierarchy rows; aggregate rows are emitted by this reconciler
    as ``S @ bottom``, appended in canonical node order after the (unchanged)
    bottom rows. An aggregate is synthesized only when all of its bottom
    members are present in the cross-section — the same completeness rule
    ``HierarchyActualsSource`` applies to aggregate actuals. Synthesized rows
    copy the cross-section's frame columns, so the forecast-frame contract is
    preserved.
    """

    requires_fitted_values = False

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        del context
        if hierarchy is None or frame.empty:
            return frame
        reject_quantile_columns(frame, strategy="bottom_up")
        summing = build_summing_matrix(hierarchy)
        parts = [
            self._expand_group(group, summing)
            for _, group in frame.groupby(_GROUP_KEYS, sort=False)
        ]
        return pd.concat(parts, ignore_index=True)

    def _expand_group(self, group: pd.DataFrame, summing: SummingMatrix) -> pd.DataFrame:
        uid_str = group[UNIQUE_ID].astype(str)
        duplicates = uid_str[uid_str.duplicated()].unique()
        if len(duplicates) > 0:
            raise ValueError(
                "forecast cross-section contains duplicate hierarchy node rows: "
                f"{sorted(duplicates)}"
            )
        non_bottom = sorted(set(uid_str) - set(summing.bottom_ids))
        if non_bottom:
            raise ValueError(
                "native bottom_up reconciliation consumes bottom-level forecast rows only; "
                f"got non-bottom node row(s): {non_bottom}"
            )
        if group[DS].nunique() != 1:
            raise ValueError(
                "bottom_up forecast cross-section must share a single target date; "
                f"got {sorted(str(value) for value in group[DS].unique())}"
            )

        subset = summing.subset(list(uid_str))
        yhat_by_id = dict(zip(uid_str, group[Y_HAT].astype(np.float64), strict=True))
        bottom = np.array([yhat_by_id[uid] for uid in subset.bottom_ids], dtype=np.float64)

        # Synthesize an aggregate only when every bottom member of that node is
        # present in the cross-section. A partial member sum would later be
        # resolved against the complete-member actual (HierarchyActualsSource
        # only resolves aggregates whose full member set is observed), silently
        # undercounting the forecast; suppressing the node keeps forecast and
        # actual completeness rules aligned.
        full_member_counts = dict(
            zip(summing.node_labels, summing.S.sum(axis=1), strict=True)
        )
        aggregate_rows = subset.S[subset.n_bottom :]
        subset_labels = subset.node_labels[subset.n_bottom :]
        complete = np.array(
            [
                present_members == full_member_counts[label]
                for label, present_members in zip(
                    subset_labels, aggregate_rows.sum(axis=1), strict=True
                )
            ],
            dtype=bool,
        )
        aggregates = (aggregate_rows @ bottom)[complete]
        aggregate_labels = [
            label for label, keep in zip(subset_labels, complete, strict=True) if keep
        ]

        template_rows = group.iloc[[0] * len(aggregate_labels)].copy()
        template_rows[UNIQUE_ID] = list(aggregate_labels)
        template_rows[Y_HAT] = aggregates.astype(np.float64)
        if Y in template_rows.columns:
            template_rows[Y] = np.nan
        return pd.concat([group, template_rows], ignore_index=True)
