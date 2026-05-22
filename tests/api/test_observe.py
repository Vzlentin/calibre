from __future__ import annotations

import pandas as pd
import pytest

from calibre.api import main as api_main
from calibre.api.lifecycle import FitRecord, LifecycleStore
from calibre.api.main import _run_observe_job
from calibre.core.forecast_frame import (
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.core.run_status import RunStatus


@pytest.fixture(autouse=True)
def _reset_store(monkeypatch):
    store = LifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", store)
    return store


def _perhorizon_fit_record(session_id: str = "s1") -> FitRecord:
    return FitRecord(
        fit_id="f1",
        session_id=session_id,
        tenant="t",
        sku_set=["sku-a"],
        forecaster_config={"backend": "stub"},
        horizon=2,
        freq="D",
        history=pd.DataFrame(),
        future_x=None,
        conformal_config={"method": "mscp", "coverage": 0.9, "calibration_window": 4},
        status=RunStatus.SUCCEEDED,
    )


def _cumulative_fit_record(session_id: str = "s2") -> FitRecord:
    return FitRecord(
        fit_id="f2",
        session_id=session_id,
        tenant="t",
        sku_set=["sku-b"],
        forecaster_config={"backend": "stub"},
        horizon=2,
        freq="D",
        history=pd.DataFrame(),
        future_x=None,
        conformal_config={
            "method": "mscp",
            "coverage": 0.9,
            "calibration_window": 4,
            "mode": "cumulative",
            "protection_period": 2,
        },
        status=RunStatus.SUCCEEDED,
    )


def _calibrated_perhorizon(lower_col: str, upper_col: str) -> pd.DataFrame:
    """Two rows: h=1 has actuals available, h=2 does not."""
    origin = pd.Timestamp("2024-01-01")
    return pd.DataFrame(
        {
            UNIQUE_ID: ["sku-a", "sku-a"],
            DS: [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["m1", "m1"],
            H: [1, 2],
            Y: [float("nan"), float("nan")],
            Y_HAT: [9.0, 10.0],
            lower_col: [7.0, 8.0],
            upper_col: [11.0, 12.0],
            CONFORMAL_METHOD: ["mscp", "mscp"],
            CONFORMAL_MODE: ["perhorizon", "perhorizon"],
            CONFORMAL_ALPHA: [0.1, 0.1],
        }
    )


def _calibrated_cumulative(lower_col: str, upper_col: str) -> pd.DataFrame:
    """Two rows: h=1 has no interval columns (intermediate), h=2 is terminal."""
    origin = pd.Timestamp("2024-01-01")
    return pd.DataFrame(
        {
            UNIQUE_ID: ["sku-b", "sku-b"],
            DS: [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["m1", "m1"],
            H: [1, 2],
            Y: [float("nan"), float("nan")],
            Y_HAT: [9.0, 10.0],
            # Only terminal row (h=2) has finite interval values
            lower_col: [float("nan"), 15.0],
            upper_col: [float("nan"), 22.0],
            CONFORMAL_METHOD: ["mscp", "mscp"],
            CONFORMAL_MODE: ["cumulative", "cumulative"],
            CONFORMAL_ALPHA: [0.1, 0.1],
        }
    )


def test_observe_perhorizon_drops_unresolved_rows(_reset_store):
    lower_col, upper_col = interval_column_names(0.9)
    record = _perhorizon_fit_record()
    record.last_calibrated = _calibrated_perhorizon(lower_col, upper_col)
    _reset_store.put_fit(record)

    # Only provide actual for h=1; h=2 remains unresolved
    actuals = [{"unique_id": "sku-a", "ds": "2024-01-02", "y": 10.0}]
    _run_observe_job("s1", actuals)

    state = _reset_store.get_conformal_state("s1")
    # Some partition state must have been written (observation happened)
    assert state, "expected conformal state to be updated after observation"


def test_observe_cumulative_does_not_drop_intermediate_rows(_reset_store):
    """Cumulative mode: _run_observe_job must not dropna before routing to observe_cumulative.

    The old code dropped rows without interval columns, which stripped the
    intermediate h=1 row that observe_cumulative needs to check window completeness.
    """
    lower_col, upper_col = interval_column_names(0.9)
    record = _cumulative_fit_record()
    record.last_calibrated = _calibrated_cumulative(lower_col, upper_col)
    _reset_store.put_fit(record)

    # Provide actuals for BOTH rows so the window is complete
    actuals = [
        {"unique_id": "sku-b", "ds": "2024-01-02", "y": 8.0},
        {"unique_id": "sku-b", "ds": "2024-01-03", "y": 9.0},
    ]
    _run_observe_job("s2", actuals)

    state = _reset_store.get_conformal_state("s2")
    assert state, (
        "cumulative observe must update conformal state when window is complete; "
        "if state is empty, intermediate rows were likely dropped before dispatch"
    )
