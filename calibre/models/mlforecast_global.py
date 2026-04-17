from __future__ import annotations

from calibre.models.mlforecast import MLForecastAdapter


class MLForecastGlobalAdapter(MLForecastAdapter):
    """Cross-series MLForecast adapter. Fits one model jointly across all uids."""

    PARALLEL_BY_UID = False
