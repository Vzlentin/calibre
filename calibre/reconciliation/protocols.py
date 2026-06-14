"""Reconciler protocol and the per-origin context threaded into reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from calibre.reconciliation.summing import HierarchyIndex


@dataclass(frozen=True)
class ReconciliationContext:
    """Per-origin sidecars required by strategy-specific reconciliation methods.

    ``fitted_values`` is the in-sample residual source for strategies such as
    MinT residual covariance. Rows are keyed by
    ``(unique_id, ds, model_name)`` because fitted values are historical
    in-sample predictions, not horizon-specific future forecast rows.
    """

    fitted_values: pd.DataFrame | None = None


class Reconciler(Protocol):
    """Point-forecast reconciliation seam (architecture §9).

    A ``Reconciler`` makes the point forecasts in a forecast frame coherent
    across a cross-sectional hierarchy. The call signature is fixed by the
    architecture: a forecast frame plus an optional prebuilt
    :class:`HierarchyIndex` in, a forecast frame (same node row-set, reconciled
    ``y_hat``) out. The index is the single one run preparation built and
    threaded through the engine — no reconciler rebuilds it. A ``None`` index is
    a no-op pass-through.

    Strategy-specific inputs such as MinT residual covariance are threaded
    through ``ReconciliationContext`` so the forecast-frame ledger stays a future
    forecast ledger. Reconciliation callers must pass the context explicitly,
    even when the current strategy ignores it.
    """

    requires_fitted_values: bool

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy_index: HierarchyIndex | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame: ...
