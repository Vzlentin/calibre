from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from calibre.reconciliation.protocols import ReconciliationContext

if TYPE_CHECKING:
    from calibre.reconciliation.summing import HierarchyIndex


class NoOpReconciler:
    """Identity reconciler — returns the forecast frame unchanged.

    The default registered strategy (R3). It never inspects ``hierarchy_index``:
    a flat-panel run (or any run that does not opt into reconciliation) passes
    through byte-identically, which is what keeps the VN2 baseline safe by
    construction.
    """

    requires_fitted_values = False

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy_index: HierarchyIndex | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        del hierarchy_index, context
        return frame
