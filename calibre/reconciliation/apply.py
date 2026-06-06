"""Cross-section apply harness shared by every vector-based reconciler.

:class:`VectorReconciler` implements the frame-to-frame mechanics the
``Reconciler`` Protocol requires once, so each concrete strategy only supplies a
pure ``reconcile_vector`` hook. The harness groups a forecast frame into
``(model_name, forecast_origin, h)`` cross-sections (KTD2), aligns each present
bottom vector to a subset summing matrix, lifts it to a coherent node-level base
vector, delegates to the strategy, and writes the reconciled bottom forecasts
back **in place** — preserving the frame's node row-set, order, and dtypes (KTD1).

The base vector handed to ``reconcile_vector`` spans **all** nodes (bottom +
aggregates), aligned to ``summing.node_labels``. On a bottom-only frame the
aggregate entries are the exact sums of their members, so the base vector is
already coherent and every strategy is an exact no-op on the bottom block —
strategy *divergence* requires independent aggregate base forecasts (a documented
follow-up). The full coherent all-levels vector is available via
:meth:`reconciled_all_levels` for tests and the Wave 4 UI, but is never injected
into the forecast frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
)
from calibre.reconciliation.summing import SummingMatrix, build_summing_matrix

NODE_LABEL = "node_label"
_GROUP_KEYS = [MODEL_NAME, FORECAST_ORIGIN, H]


class VectorReconciler:
    """Base class turning a per-cross-section vector transform into a Reconciler.

    Subclasses implement :meth:`reconcile_vector`, mapping a node-level base
    vector (aligned to ``summing.node_labels``) to a coherent node-level vector
    of the same length.
    """

    def __call__(self, frame: pd.DataFrame, hierarchy: pd.DataFrame | None) -> pd.DataFrame:
        if hierarchy is None or frame.empty:
            return frame
        summing = build_summing_matrix(hierarchy)
        parts = [
            self._reconcile_group(group, summing)
            for _, group in frame.groupby(_GROUP_KEYS, sort=False)
        ]
        return pd.concat(parts).sort_index()

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        """Map a node-level base vector to a coherent node-level vector.

        ``base`` and the return value are both length ``summing.n_nodes`` and
        aligned to ``summing.node_labels`` (bottom identity block first).
        """
        raise NotImplementedError

    def reconciled_all_levels(
        self, frame: pd.DataFrame, hierarchy: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Return the full coherent all-levels vector per cross-section.

        Long frame with columns ``[model_name, forecast_origin, h, node_label,
        y_hat]`` covering bottom **and** aggregate nodes. This is the Wave 4
        level-propagation surface; it is intentionally *not* written back into the
        forecast frame (KTD1).
        """
        if hierarchy is None or frame.empty:
            return pd.DataFrame(columns=[*_GROUP_KEYS, NODE_LABEL, Y_HAT])
        summing = build_summing_matrix(hierarchy)
        records: list[pd.DataFrame] = []
        for keys, group in frame.groupby(_GROUP_KEYS, sort=False):
            subset, coherent = self._coherent_vector(group, summing)
            model_name, origin, horizon = keys
            records.append(
                pd.DataFrame(
                    {
                        MODEL_NAME: model_name,
                        FORECAST_ORIGIN: origin,
                        H: horizon,
                        NODE_LABEL: list(subset.node_labels),
                        Y_HAT: coherent,
                    }
                )
            )
        return pd.concat(records, ignore_index=True)

    def _coherent_vector(
        self, group: pd.DataFrame, summing: SummingMatrix
    ) -> tuple[SummingMatrix, np.ndarray]:
        uid_str = group[UNIQUE_ID].astype(str)
        subset = summing.subset(uid_str.tolist())
        yhat_by_id = dict(zip(uid_str, group[Y_HAT].astype(np.float64), strict=True))
        bottom = np.array([yhat_by_id[uid] for uid in subset.bottom_ids], dtype=np.float64)
        base = subset.S @ bottom
        return subset, self.reconcile_vector(base, subset)

    def _reconcile_group(self, group: pd.DataFrame, summing: SummingMatrix) -> pd.DataFrame:
        subset, coherent = self._coherent_vector(group, summing)
        reconciled_bottom = dict(zip(subset.bottom_ids, coherent[: subset.n_bottom], strict=True))
        result = group.copy()
        result[Y_HAT] = result[UNIQUE_ID].astype(str).map(reconciled_bottom).astype(np.float64)
        return result
