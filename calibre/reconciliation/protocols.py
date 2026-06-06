from __future__ import annotations

from typing import Protocol

import pandas as pd


class Reconciler(Protocol):
    """Point-forecast reconciliation seam (architecture §9).

    A ``Reconciler`` makes the point forecasts in a forecast frame coherent
    across a cross-sectional hierarchy. The call signature is fixed by the
    architecture: a forecast frame plus an optional hierarchy attribute frame in,
    a forecast frame (same node row-set, reconciled ``y_hat``) out. A ``None``
    hierarchy is a no-op pass-through.

    Strategy-specific inputs (MinT's residual covariance, top-down's proportion
    source) live on the concrete instance, built by the registry from config, so
    the seam signature stays identical to §9 (KTD5).
    """

    def __call__(self, frame: pd.DataFrame, hierarchy: pd.DataFrame | None) -> pd.DataFrame: ...
