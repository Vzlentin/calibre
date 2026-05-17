from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.core.forecast_frame import (
    CALIBRATION_STATE,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine
from calibre.storage.state import RUNTIME_PARTITION


class _MemoryStateStore:
    def __init__(self) -> None:
        self.states: dict[tuple[UUID, str], dict] = {}

    def get(self, run_id: UUID, partition: str = RUNTIME_PARTITION) -> dict | None:
        return self.states.get((run_id, partition))

    def upsert(self, run_id: UUID, partition: str, state: dict) -> None:
        self.states[(run_id, partition)] = dict(state)


def _frame(origin: pd.Timestamp, y_hat: float, y: float | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            "ds": [origin + pd.Timedelta(weeks=1)],
            Y: [np.nan if y is None else float(y)],
            Y_HAT: [float(y_hat)],
            H: [1],
            FORECAST_ORIGIN: [origin],
            MODEL_NAME: ["stub"],
        }
    )


def test_symmetric_runtime_from_state_restores_calibrator_and_controller() -> None:
    config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=5,
        gamma=0.05,
    )
    runtime = SymmetricIntervalRuntime(config)

    first = runtime.apply(_frame(pd.Timestamp("2024-01-01"), 10.0))
    first[Y] = [12.0]
    observed = runtime.observe(first)
    assert observed[NONCONFORMITY_SCORE].iloc[0] == pytest.approx(2.0)

    second = runtime.apply(_frame(pd.Timestamp("2024-01-08"), 11.0))
    state_payload = str(second[CALIBRATION_STATE].iloc[0])
    resumed = SymmetricIntervalRuntime.from_state(config, state_payload)

    original_diag = runtime.get_diagnostics()
    resumed_diag = resumed.get_diagnostics()
    assert resumed_diag["issued_count"] == original_diag["issued_count"] - 1
    assert resumed_diag["controller"]["current_alpha"] == pytest.approx(
        original_diag["controller"]["current_alpha"]
    )
    np.testing.assert_allclose(
        resumed_diag["calibrator"]["score_history"]["stub:h1:__global__"],
        np.asarray([2.0]),
    )


def test_backend_restores_conformal_runtime_from_state_store() -> None:
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 3
    pattern[8] = 99.0
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})
    task = ForecastTask(
        history=actuals,
        horizon=1,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    run_id = uuid4()
    store = _MemoryStateStore()

    BackendEngine(
        freq="W",
        conformal_config=config,
        run_id=run_id,
        conformal_state_store=store,
    ).execute([task], actuals, origins=[dates[7], dates[8]])

    assert store.get(run_id, RUNTIME_PARTITION) is not None

    resumed = BackendEngine(
        freq="W",
        conformal_config=config,
        run_id=run_id,
        conformal_state_store=store,
    ).execute([task], actuals, origins=[dates[9]])

    frame = resumed.ledger.to_df()
    lower_col, upper_col = config.interval_columns
    assert (frame[upper_col] - frame[lower_col]).iloc[0] > 0.0
