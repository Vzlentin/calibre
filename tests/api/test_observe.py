"""Behavioral tests for /observe conformal-mode dispatch (roadmap P0.1).

Regression guard for lessons.md §40: ``_run_observe_job`` used to drop rows with
NaN interval bounds before calling ``runtime.observe``. Cumulative mode emits
NaN bounds on a window's intermediate horizons *by construction*, so that dropna
silently killed online recalibration for cumulative deployments. These assert
the production path routes observations by mode: cumulative keeps the whole
completed window (NaN-bound rows included); per-horizon still drops rows without
resolved bounds + actuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.api import main as api_main
from calibre.api.lifecycle import FitRecord, LifecycleStore
from calibre.core.forecast_frame import (
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

LOWER, UPPER = interval_column_names(0.9)
ORIGIN = pd.Timestamp("2024-02-04")
WINDOW_DS = [ORIGIN + pd.Timedelta(weeks=h) for h in (1, 2, 3)]


class _RecordingRuntime:
    """Captures exactly which rows reach ``observe`` for a given mode."""

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self.observed: list[pd.DataFrame] = []

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def interval_columns(self) -> tuple[str, str]:
        return (LOWER, UPPER)

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame:
        self.observed.append(resolved.copy())
        return resolved

    def get_partition_states(self) -> dict[str, dict]:
        return {}


def _calibrated_window(bounds_at_each_horizon: bool) -> pd.DataFrame:
    """One (uid, model, origin) window of 3 horizons.

    ``bounds_at_each_horizon=False`` mimics cumulative output: only the final
    horizon carries finite bounds; intermediate horizons are NaN.
    """
    if bounds_at_each_horizon:
        lo, hi = [4.0, 4.0, 5.0], [14.0, 14.0, 15.0]
    else:
        lo, hi = [np.nan, np.nan, 5.0], [np.nan, np.nan, 15.0]
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "A"],
            DS: WINDOW_DS,
            FORECAST_ORIGIN: [ORIGIN, ORIGIN, ORIGIN],
            MODEL_NAME: ["m", "m", "m"],
            H: [1, 2, 3],
            Y: [np.nan, np.nan, np.nan],
            Y_HAT: [10.0, 10.0, 10.0],
            LOWER: lo,
            UPPER: hi,
        }
    )


def _actual_records(ds_values: list[pd.Timestamp]) -> list[dict]:
    return [{UNIQUE_ID: "A", "ds": ds.strftime("%Y-%m-%d"), "y": 9.0} for ds in ds_values]


@pytest.fixture
def session(monkeypatch):
    """A fit record installed in a fresh lifecycle store; returns the session id."""
    store = LifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", store)
    session_id = "sess-observe"

    def _make(mode: str, calibrated: pd.DataFrame) -> _RecordingRuntime:
        record = FitRecord(
            fit_id="fit-1",
            session_id=session_id,
            tenant="acme",
            sku_set=["A"],
            forecaster_config={"backend": "statsforecast", "model": "Naive"},
            horizon=3,
            freq="W-SUN",
            history=pd.DataFrame(),
            future_x=None,
            conformal_config={"method": "mscp", "mode": mode},
            status=RunStatus.SUCCEEDED,
            last_calibrated=calibrated,
        )
        store.put_fit(record)
        runtime = _RecordingRuntime(mode)
        monkeypatch.setattr(api_main, "_runtime_for_session", lambda _record: runtime)
        return runtime

    return session_id, _make


def test_observe_cumulative_keeps_nan_bound_intermediate_rows(session):
    """Cumulative: the whole completed window reaches observe, NaN bounds and all."""
    session_id, make = session
    runtime = make("cumulative", _calibrated_window(bounds_at_each_horizon=False))

    api_main._run_observe_job(session_id, _actual_records(WINDOW_DS))

    assert len(runtime.observed) == 1, "completed window should be observed"
    observed = runtime.observed[0]
    assert len(observed) == 3, "all 3 horizons (incl. NaN-bound rows) must survive"
    assert observed[LOWER].isna().sum() == 2, "intermediate NaN bounds must be preserved"
    assert sorted(observed[H].tolist()) == [1, 2, 3]


def test_observe_cumulative_incomplete_window_observes_nothing(session):
    """Cumulative: a window missing an actual is not yet ready — nothing observed."""
    session_id, make = session
    runtime = make("cumulative", _calibrated_window(bounds_at_each_horizon=False))

    # Only the first two horizons have actuals; the window is incomplete.
    api_main._run_observe_job(session_id, _actual_records(WINDOW_DS[:2]))

    # An incomplete window is never handed to observe — not "empty or nothing".
    assert runtime.observed == []


def test_observe_perhorizon_drops_unresolved_rows(session):
    """Per-horizon: only rows with resolved bounds + actuals are observed."""
    session_id, make = session
    runtime = make("perhorizon", _calibrated_window(bounds_at_each_horizon=True))

    # Actual only for the first horizon; h=2,3 stay unresolved (no actual).
    api_main._run_observe_job(session_id, _actual_records(WINDOW_DS[:1]))

    assert len(runtime.observed) == 1
    observed = runtime.observed[0]
    assert observed[H].tolist() == [1], "only the resolved horizon should be observed"


def test_observe_calibrated_without_y_column_does_not_crash(session):
    """A hand-crafted /calibrate payload may omit y; observe adds it, not KeyError.

    The old code handled a missing y explicitly; routing through the dispatch
    must not regress that — _fill_actuals needs the column to exist.
    """
    session_id, make = session
    calibrated = _calibrated_window(bounds_at_each_horizon=True).drop(columns=[Y])
    runtime = make("perhorizon", calibrated)

    api_main._run_observe_job(session_id, _actual_records(WINDOW_DS[:1]))

    assert len(runtime.observed) == 1
    observed = runtime.observed[0]
    assert observed[H].tolist() == [1]
    assert observed[Y].tolist() == [9.0], "actual should be filled into the added y column"
