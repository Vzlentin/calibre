from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class ReconciliationContext:
    """Sidecar data required by strategy-specific reconciliation methods."""

    fitted_values: pd.DataFrame | None = None


class Reconciler(Protocol):
    """Point-forecast reconciliation seam (architecture §9).

    A ``Reconciler`` makes the point forecasts in a forecast frame coherent
    across a cross-sectional hierarchy. The call signature is fixed by the
    architecture: a forecast frame plus an optional hierarchy attribute frame in,
    a forecast frame (same node row-set, reconciled ``y_hat``) out. A ``None``
    hierarchy is a no-op pass-through.

    Strategy-specific inputs such as MinT residual covariance are threaded
    through ``ReconciliationContext`` so the forecast-frame ledger stays a future
    forecast ledger.
    """

    requires_fitted_values: bool

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: pd.DataFrame | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame: ...
