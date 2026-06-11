"""Native bottom-up point reconciler.

Point ``bottom_up`` is just per-node sums of bottom member forecasts: it needs
no independent aggregate base forecasts and no projection math. This
reconciler therefore consumes **bottom-only** forecast rows and synthesizes
the aggregate node rows itself, so a bottom-up run never pays for aggregate
forecast tasks or eager node-history expansion. MinT-style strategies
(``ols``, ``wls_struct``, ``mint_shrink``, ``wls_var``, ``erm``) stay on the
Nixtla-backed vector reconciler with independent bottom + aggregate base
forecasts.
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
from calibre.reconciliation.summing import TOTAL_LABEL, HierarchyIndex, build_hierarchy_index

_GROUP_KEYS = [MODEL_NAME, FORECAST_ORIGIN, H]


class BottomUpReconciler:
    """Synthesize coherent aggregate node rows from bottom-only point forecasts.

    Each ``(model_name, forecast_origin, h)`` cross-section must contain only
    bottom-level hierarchy rows; aggregate rows are synthesized as the sum of
    their bottom members, appended in canonical node order after the
    (unchanged) bottom rows. An aggregate is synthesized only when all of its
    bottom members are present in the cross-section — the same completeness
    rule ``HierarchyActualsSource`` applies to aggregate actuals. Synthesized
    rows copy the cross-section's frame columns, so the forecast-frame
    contract is preserved.

    Aggregation is grouped member sums over the hierarchy attribute columns,
    deliberately not a dense ``S @ bottom`` product: a dense summing matrix at
    M5 scale (~33.5k nodes x ~30.5k bottoms) costs ~8 GiB plus a full copy per
    cross-section, which cannot run on commodity memory. A NaN bottom forecast
    still poisons every aggregate that contains it, matching the dense
    product's semantics.
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
        index = build_hierarchy_index(hierarchy)
        attrs = index.frame.set_index(UNIQUE_ID)
        expected_members = index.expected_members()
        parts = [
            self._expand_group(group, index, attrs, expected_members)
            for _, group in frame.groupby(_GROUP_KEYS, sort=False)
        ]
        return pd.concat(parts, ignore_index=True)

    def _expand_group(
        self,
        group: pd.DataFrame,
        index: HierarchyIndex,
        attrs: pd.DataFrame,
        expected_members: dict[str, pd.Series],
    ) -> pd.DataFrame:
        uid_str = group[UNIQUE_ID].astype(str)
        duplicates = uid_str[uid_str.duplicated()].unique()
        if len(duplicates) > 0:
            raise ValueError(
                "forecast cross-section contains duplicate hierarchy node rows: "
                f"{sorted(duplicates)}"
            )
        non_bottom = sorted(set(uid_str) - set(index.bottom_ids))
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

        yhat = group[Y_HAT].to_numpy(dtype=np.float64)
        nan_mask = np.isnan(yhat)
        members = attrs.loc[uid_str.to_list()]

        # Synthesize an aggregate only when every bottom member of that node is
        # present in the cross-section. A partial member sum would later be
        # resolved against the complete-member actual (HierarchyActualsSource
        # only resolves aggregates whose full member set is observed), silently
        # undercounting the forecast; suppressing the node keeps forecast and
        # actual completeness rules aligned.
        aggregate_labels: list[str] = []
        aggregate_values: list[float] = []
        for col in index.attr_cols:
            cross_section = pd.DataFrame(
                {
                    "value": members[col].astype(str).to_numpy(),
                    "yhat": yhat,
                    "isnan": nan_mask,
                }
            )
            agg = cross_section.groupby("value", sort=True).agg(
                total=("yhat", "sum"),
                present=("yhat", "size"),
                nans=("isnan", "sum"),
            )
            expected = expected_members[col]
            for value, row in agg.iterrows():
                if row["present"] != expected[str(value)]:
                    continue
                aggregate_labels.append(f"{col}={value}")
                aggregate_values.append(np.nan if row["nans"] else float(row["total"]))
        if len(group) == len(index.bottom_ids):
            aggregate_labels.append(TOTAL_LABEL)
            aggregate_values.append(np.nan if nan_mask.any() else float(yhat.sum()))

        template_rows = group.iloc[[0] * len(aggregate_labels)].copy()
        template_rows[UNIQUE_ID] = aggregate_labels
        template_rows[Y_HAT] = np.asarray(aggregate_values, dtype=np.float64)
        if Y in template_rows.columns:
            template_rows[Y] = np.nan
        return pd.concat([group, template_rows], ignore_index=True)
