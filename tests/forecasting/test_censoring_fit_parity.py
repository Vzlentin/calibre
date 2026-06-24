"""C1 gating parity: the engine's censoring-aware fit target equals the benchmark's.

This is checkpoint C1, the acceptance gate for issue #260. It captures the *actual* frame
``MLForecastAdapter.fit`` hands to the underlying mlforecast ``.fit`` (via a spy
on the MLForecast class) for the VN2 winning config with ``censoring_fit`` on,
and asserts its ``[unique_id, ds, y]`` subset equals
``benchmarks.vn2.data.prepare_model_history(..., cumulative_target=False)``.

The test must NOT re-call ``add_stockout_features`` itself — that would compare
the helper to itself and prove nothing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from benchmarks.vn2.config import BEST_CONFIG, HORIZON, LEAD_TIME, REVIEW_PERIOD
from benchmarks.vn2.data import load_instock, prepare_model_history
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.data_loading import load_period
from calibre.forecasting.mlforecast_adapter import MLForecastAdapter

DATA_DIR = Path(__file__).parents[2] / "data" / "vn2"
PROTECTION_PERIOD = LEAD_TIME + REVIEW_PERIOD


def test_engine_fit_target_matches_benchmark_prepare_model_history(monkeypatch) -> None:
    sales = load_period(DATA_DIR, 0)
    instock = load_instock(DATA_DIR, series_filter=None)
    assert instock is not None, "VN2 in-stock fixture is required for the C1 parity check"

    task = ForecastTask(
        history=sales,
        horizon=HORIZON,
        model_config={**BEST_CONFIG, "censoring_fit": True},
        censoring=instock,
    )

    captured: dict[str, pd.DataFrame] = {}
    mock_instance = MagicMock()

    def _record_fit(df, **_kwargs):
        captured["fit_df"] = df

    mock_instance.fit.side_effect = _record_fit
    monkeypatch.setattr(
        "calibre.forecasting.mlforecast_adapter.MLForecast",
        MagicMock(return_value=mock_instance),
    )

    MLForecastAdapter(task.model_config).fit(task)

    actual = (
        captured["fit_df"][[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    )

    expected = prepare_model_history(
        sales,
        instock,
        protection_period=PROTECTION_PERIOD,
        cumulative_target=False,
    )[[UNIQUE_ID, DS, Y]]
    expected = expected.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    # The adapter casts the target to float32 before handing it to mlforecast.
    expected[Y] = expected[Y].astype("float32")

    pd.testing.assert_frame_equal(actual, expected)
