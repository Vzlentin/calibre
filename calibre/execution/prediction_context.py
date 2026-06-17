"""Per-origin fitted-value sidecar carried across the predict→calibrate seam."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FittedValueContext:
    """Fitted-value sidecar :meth:`_run_origin_predict` returns each origin.

    Carries the in-sample fitted values produced alongside this origin's point
    forecasts. The shared predict→Reconcile→Calibrate path threads it through:
    :meth:`_reconcile` reads ``.fitted_values`` for residual reconcilers and
    :meth:`_calibrate` stages it onto a draws-based conformal runtime.
    """

    fitted_values: pd.DataFrame | None = None
