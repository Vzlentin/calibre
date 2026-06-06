from __future__ import annotations

import pandas as pd


class NoOpReconciler:
    """Identity reconciler — returns the forecast frame unchanged.

    The default registered strategy (R3). It never inspects ``hierarchy``: a
    flat-panel run (or any run that does not opt into reconciliation) passes
    through byte-identically, which is what keeps the VN2 baseline safe by
    construction.
    """

    def __call__(self, frame: pd.DataFrame, hierarchy: pd.DataFrame | None) -> pd.DataFrame:
        del hierarchy
        return frame
