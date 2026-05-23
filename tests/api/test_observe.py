from __future__ import annotations

import pandas as pd

from calibre.api import main as api_main
from calibre.api.lifecycle import FitRecord, LifecycleStore, MemoryLifecycleStore
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


class _RuntimeSpy:
    interval_columns = interval_column_names(0.9)

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.observed: list[pd.DataFrame] = []

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame:
        self.observed.append(resolved.copy())
        return resolved

    def get_partition_states(self) -> dict[str, dict[str, object]]:
        return {"partition": {"observed_batches": len(self.observed)}}


def _record(session_id: str) -> FitRecord:
    return FitRecord(
        fit_id=LifecycleStore.new_fit_id(),
        session_id=session_id,
        tenant="tenant",
        sku_set=["A"],
        forecaster_config={"model": "stub"},
        horizon=2,
        freq="D",
        conformal_config={"method": "mscp", "coverage": 0.9},
    )


def _calibrated_frame(*, cumulative: bool) -> pd.DataFrame:
    lower_col, upper_col = interval_column_names(0.9)
    origin = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            DS: [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            Y: [float("nan"), float("nan")],
            Y_HAT: [10.0, 20.0],
            H: [1, 2],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["stub", "stub"],
            lower_col: [9.0, 18.0],
            upper_col: [11.0, 22.0],
        }
    )
    if cumulative:
        frame.loc[frame[H] == 1, [lower_col, upper_col]] = float("nan")
    return frame


def test_observe_cumulative_does_not_drop_intermediate_rows(monkeypatch) -> None:
    store = MemoryLifecycleStore()
    runtime = _RuntimeSpy("cumulative")
    session_id = "session-cumulative"
    record = _record(session_id)
    store.put_fit(record)
    store.put_fit_frame(record.fit_id, "last_calibrated", _calibrated_frame(cumulative=True))
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", store)
    monkeypatch.setattr(api_main, "_runtime_for_session", lambda record: runtime)

    api_main._run_observe_job(
        session_id,
        [
            {UNIQUE_ID: "A", DS: "2024-01-02", Y: 12.0},
            {UNIQUE_ID: "A", DS: "2024-01-03", Y: 21.0},
        ],
    )

    assert len(runtime.observed) == 1
    observed = runtime.observed[0].sort_values(H).reset_index(drop=True)
    assert observed[H].tolist() == [1, 2]
    assert observed[Y].tolist() == [12.0, 21.0]
    assert store.get_conformal_state(session_id) == {"partition": {"observed_batches": 1}}


def test_observe_perhorizon_drops_unresolved_rows(monkeypatch) -> None:
    store = MemoryLifecycleStore()
    runtime = _RuntimeSpy("perhorizon")
    session_id = "session-perhorizon"
    record = _record(session_id)
    store.put_fit(record)
    store.put_fit_frame(record.fit_id, "last_calibrated", _calibrated_frame(cumulative=False))
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", store)
    monkeypatch.setattr(api_main, "_runtime_for_session", lambda record: runtime)

    api_main._run_observe_job(
        session_id,
        [{UNIQUE_ID: "A", DS: "2024-01-02", Y: 12.0}],
    )

    assert len(runtime.observed) == 1
    observed = runtime.observed[0]
    assert observed[H].tolist() == [1]
    assert observed[Y].tolist() == [12.0]
