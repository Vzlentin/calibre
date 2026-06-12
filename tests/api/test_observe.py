"""Behavioral tests for /observe conformal-mode dispatch (roadmap P0.1).

Regression guard for lessons.md §40: the observe job used to drop rows with
NaN interval bounds before calling ``runtime.observe``. Cumulative mode emits
NaN bounds on a window's intermediate horizons *by construction*, so that dropna
silently killed online recalibration for cumulative deployments. These assert
the production path routes observations by mode: cumulative keeps the whole
completed window (NaN-bound rows included); per-horizon still drops rows without
resolved bounds + actuals.

These pins drive the observe job through a stable test wrapper
``observe_for_test(session_id, records, *, store, runtime)`` rather than the
module internals directly. The wrapper is the migration seam: today it delegates
to the private observe job with the runtime stubbed at the module boundary; once
the job is promoted to ``run_observe_job(session_id, records, *, store)`` the
wrapper's internals swap to the public function (the runtime is rehydrated
inside it) while these call sites stay byte-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.api import main as api_main
from calibre.api.lifecycle import FitRecord, LifecycleStore
from calibre.conformal.runtime import (
    SymmetricIntervalConfig,
    build_symmetric_interval_runtime,
)
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


def observe_for_test(
    session_id: str,
    records: list[dict],
    *,
    store,
    runtime,
) -> None:
    """Stable observe-job seam for the characterization pins.

    U1 end-state: the ``session`` fixture has already bound ``store`` and
    ``runtime`` onto the module seam (``_LIFECYCLE_STORE`` /
    ``_runtime_for_session``) via ``monkeypatch``, so this just drives the
    private observe job. U3 swaps this body for
    ``api_main.run_observe_job(session_id, records, store=store)`` (the runtime
    is rehydrated inside it) and the fixture stops monkeypatching, leaving every
    call site below unchanged.
    """
    del store, runtime  # bound on the module seam by the session fixture in U1
    api_main._run_observe_job(session_id, records)


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
    """A fit record installed in a fresh lifecycle store; returns the session id.

    Yields ``(session_id, store, make)`` where ``make(mode, calibrated)`` installs
    the fit and returns the recording runtime the wrapper observes through. In U1
    the fixture binds the store and runtime onto the module seam via
    ``monkeypatch`` (auto-restored), so the seam is the only module state touched;
    ``observe_for_test`` then just drives the private job. U3 removes this binding
    once the job takes an injected store directly.
    """
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

    return session_id, store, _make


def test_observe_cumulative_keeps_nan_bound_intermediate_rows(session):
    """Cumulative: the whole completed window reaches observe, NaN bounds and all."""
    session_id, store, make = session
    runtime = make("cumulative", _calibrated_window(bounds_at_each_horizon=False))

    observe_for_test(session_id, _actual_records(WINDOW_DS), store=store, runtime=runtime)

    assert len(runtime.observed) == 1, "completed window should be observed"
    observed = runtime.observed[0]
    assert len(observed) == 3, "all 3 horizons (incl. NaN-bound rows) must survive"
    assert observed[LOWER].isna().sum() == 2, "intermediate NaN bounds must be preserved"
    assert sorted(observed[H].tolist()) == [1, 2, 3]


def test_observe_cumulative_incomplete_window_observes_nothing(session):
    """Cumulative: a window missing an actual is not yet ready — nothing observed."""
    session_id, store, make = session
    runtime = make("cumulative", _calibrated_window(bounds_at_each_horizon=False))

    # Only the first two horizons have actuals; the window is incomplete.
    observe_for_test(session_id, _actual_records(WINDOW_DS[:2]), store=store, runtime=runtime)

    # An incomplete window is never handed to observe — not "empty or nothing".
    assert runtime.observed == []


def test_observe_perhorizon_drops_unresolved_rows(session):
    """Per-horizon: only rows with resolved bounds + actuals are observed."""
    session_id, store, make = session
    runtime = make("perhorizon", _calibrated_window(bounds_at_each_horizon=True))

    # Actual only for the first horizon; h=2,3 stay unresolved (no actual).
    observe_for_test(session_id, _actual_records(WINDOW_DS[:1]), store=store, runtime=runtime)

    assert len(runtime.observed) == 1
    observed = runtime.observed[0]
    assert observed[H].tolist() == [1], "only the resolved horizon should be observed"


def test_observe_failure_is_logged_not_raised_and_state_not_persisted(session, caplog):
    """A structurally bad window fails loudly in observe; the job boundary
    records it with session context, does not re-raise, and durable conformal
    state is untouched (upsert runs only on success)."""
    import logging

    session_id, store, make = session
    runtime = make("cumulative", _calibrated_window(bounds_at_each_horizon=False))

    def _raise(resolved: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("Duplicate H values in cumulative observe window")

    runtime.observe = _raise
    # Non-empty states prove the skip: were upsert reached, this would persist.
    runtime.get_partition_states = lambda: {"m:cumulative:__global__": {"scores": [1.0]}}

    with caplog.at_level(logging.ERROR):
        observe_for_test(session_id, _actual_records(WINDOW_DS), store=store, runtime=runtime)

    assert "observe job failed" in caplog.text
    assert store.get_conformal_state(session_id) == {}


def test_observe_success_persists_partition_states(session):
    """The inverse lock: a successful observe upserts non-empty partition states."""
    session_id, store, make = session
    runtime = make("cumulative", _calibrated_window(bounds_at_each_horizon=False))
    runtime.get_partition_states = lambda: {"m:cumulative:__global__": {"scores": [1.0]}}

    observe_for_test(session_id, _actual_records(WINDOW_DS), store=store, runtime=runtime)

    assert store.get_conformal_state(session_id) == {"m:cumulative:__global__": {"scores": [1.0]}}


def test_observe_calibrated_without_y_column_does_not_crash(session):
    """A hand-crafted /calibrate payload may omit y; observe adds it, not KeyError.

    The old code handled a missing y explicitly; routing through the dispatch
    must not regress that — _fill_actuals needs the column to exist.
    """
    session_id, store, make = session
    calibrated = _calibrated_window(bounds_at_each_horizon=True).drop(columns=[Y])
    runtime = make("perhorizon", calibrated)

    observe_for_test(session_id, _actual_records(WINDOW_DS[:1]), store=store, runtime=runtime)

    assert len(runtime.observed) == 1
    observed = runtime.observed[0]
    assert observed[H].tolist() == [1]
    assert observed[Y].tolist() == [9.0], "actual should be filled into the added y column"


def test_observe_end_to_end_through_real_runtime(session, monkeypatch):
    """End-to-end characterization through a REAL SymmetricIntervalRuntime.

    The six routing pins above stub the conformal arithmetic with
    ``_RecordingRuntime``, so they prove the row-routing seam but not the
    apply/observe math. This pin drives a real perhorizon mscp runtime through
    /calibrate's apply + the observe job's observe and asserts the persisted
    nonconformity scores — |actual - y_hat| per resolved horizon — so the real
    conformal path is pinned before any code moves.
    """
    session_id, store, _make = session
    config = SymmetricIntervalConfig(method="mscp", coverage=0.9, mode="perhorizon")
    runtime = build_symmetric_interval_runtime(config)
    monkeypatch.setattr(api_main, "_runtime_for_session", lambda _record: runtime)

    forecast = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "A"],
            DS: WINDOW_DS,
            FORECAST_ORIGIN: [ORIGIN, ORIGIN, ORIGIN],
            MODEL_NAME: ["m", "m", "m"],
            H: [1, 2, 3],
            Y: [np.nan, np.nan, np.nan],
            Y_HAT: [10.0, 10.0, 10.0],
        }
    )
    calibrated = runtime.apply(forecast)

    record = FitRecord(
        fit_id="fit-real",
        session_id=session_id,
        tenant="acme",
        sku_set=["A"],
        forecaster_config={"backend": "statsforecast", "model": "Naive"},
        horizon=3,
        freq="W-SUN",
        history=pd.DataFrame(),
        future_x=None,
        conformal_config={"method": "mscp", "coverage": 0.9, "mode": "perhorizon"},
        status=RunStatus.SUCCEEDED,
        last_calibrated=calibrated,
    )
    store.put_fit(record)

    # Actual y=9.0 at every horizon -> nonconformity |9 - 10| = 1.0 per horizon.
    observe_for_test(
        session_id,
        _actual_records(WINDOW_DS),
        store=store,
        runtime=runtime,
    )

    state = store.get_conformal_state(session_id)
    assert set(state) == {"m:h1:__global__", "m:h2:__global__", "m:h3:__global__"}
    for partition in state:
        scores = state[partition]["calibrator"]["score_history"][partition]
        assert [float(s) for s in scores] == [1.0]
