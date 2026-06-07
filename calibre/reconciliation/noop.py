from __future__ import annotations

import pandas as pd

from calibre.reconciliation.protocols import ReconciliationContext


class NoOpReconciler:
    """Identity reconciler — returns the forecast frame unchanged.

    The default registered strategy (R3). It never inspects ``hierarchy``: a
    flat-panel run (or any run that does not opt into reconciliation) passes
    through byte-identically, which is what keeps the VN2 baseline safe by
    construction.
    """

    requires_fitted_values = False

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        del hierarchy, context
        return frame
