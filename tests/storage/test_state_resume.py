from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.core.forecast_frame import (
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, ConformalOptions
from calibre.execution.task_builder import partition_tasks
from calibre.storage.models import Base
from calibre.storage.objstore import read_initial_ledger, write_ledger_shard
from calibre.storage.postgres import (
    ConformalStateRepo,
    ForecastPointerRepo,
    RunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)
from calibre.storage.state import RUNTIME_PARTITION, SqlConformalStateStore


class _MemoryStateStore:
    def __init__(self) -> None:
        self.states: dict[tuple[UUID, str], dict] = {}

    def get(self, run_id: UUID, partition: str = RUNTIME_PARTITION) -> dict | None:
        return self.states.get((run_id, partition))

    def list_for_run(self, run_id: UUID) -> dict[str, dict]:
        return {
            partition: state
            for (stored_run_id, partition), state in self.states.items()
            if stored_run_id == run_id
        }

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

    state_payload = runtime.get_resume_state()
    resumed = SymmetricIntervalRuntime.from_state(config, state_payload)

    original_diag = runtime.get_diagnostics()
    resumed_diag = resumed.get_diagnostics()
    assert resumed_diag["issued_count"] == original_diag["issued_count"]
    assert resumed_diag["controller"]["current_alpha"] == pytest.approx(
        original_diag["controller"]["current_alpha"]
    )
    assert "alpha_history" not in state_payload["controller"]
    assert "error_history" not in state_payload["controller"]
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
        conformal=ConformalOptions(config=config, run_id=run_id, state_store=store),
    ).execute(partition_tasks([task]), actuals, origins=[dates[7], dates[8]])

    persisted = store.list_for_run(run_id)
    assert "SeasonalNaive:h1:__global__" in persisted
    persisted_state = persisted["SeasonalNaive:h1:__global__"]
    assert RUNTIME_PARTITION not in persisted
    assert "alpha_history" not in persisted_state["controller"]
    assert "error_history" not in persisted_state["controller"]

    resumed = BackendEngine(
        conformal=ConformalOptions(config=config, run_id=run_id, state_store=store),
    ).execute(partition_tasks([task]), actuals, origins=[dates[9]])

    frame = resumed.ledger.to_df()
    lower_col, upper_col = config.interval_columns
    assert (frame[upper_col] - frame[lower_col]).iloc[0] > 0.0


def test_backend_replays_initial_ledger_for_byte_identical_resume() -> None:
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": [10.0, 20.0, 30.0, 40.0] * 3})
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
    origins = [dates[7], dates[8], dates[9]]

    uninterrupted = BackendEngine(conformal=ConformalOptions(config=config)).execute(
        partition_tasks([task]),
        actuals,
        origins=origins,
    )

    run_id = uuid4()
    store = _MemoryStateStore()
    interrupted = BackendEngine(
        conformal=ConformalOptions(config=config, run_id=run_id, state_store=store),
    ).execute(partition_tasks([task]), actuals, origins=origins[:2])

    resumed = BackendEngine(
        conformal=ConformalOptions(
            config=config,
            run_id=run_id,
            state_store=store,
            initial_ledger=interrupted.ledger.to_df(),
        ),
    ).execute(partition_tasks([task]), actuals, origins=origins[2:])

    sort_cols = [UNIQUE_ID, FORECAST_ORIGIN, H]
    expected = uninterrupted.ledger.to_df().sort_values(sort_cols).reset_index(drop=True)
    actual = resumed.ledger.to_df().sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_backend_resumes_from_db_state_and_artifact_pointer(tmp_path) -> None:
    dates = pd.date_range("2024-01-07", periods=12, freq="W")
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": [10.0, 20.0, 30.0, 40.0] * 3})
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
    origins = [dates[7], dates[8], dates[9]]

    uninterrupted = BackendEngine(conformal=ConformalOptions(config=config)).execute(
        partition_tasks([task]),
        actuals,
        origins=origins,
    )

    db = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'resume.db').as_posix()}")
    Base.metadata.create_all(db)
    factory = make_session_factory(db)
    ledger_path = tmp_path / "ledger.parquet"

    with session_scope(factory) as session:
        run = RunRepo(session).create(config={"name": "resume-test"})
        run_id = run.id
        interrupted = BackendEngine(
            conformal=ConformalOptions(
                config=config,
                run_id=run_id,
                state_store=SqlConformalStateStore(ConformalStateRepo(session)),
            ),
        ).execute(partition_tasks([task]), actuals, origins=origins[:2])
        pointer = write_ledger_shard(interrupted.ledger.to_df(), str(ledger_path))
        ForecastPointerRepo(session).upsert(
            run_id,
            "ledger",
            str(pointer["uri"]),
            int(pointer["byte_size"]),
        )

    with session_scope(factory) as session:
        initial_ledger = read_initial_ledger(ForecastPointerRepo(session), run_id)
        assert initial_ledger is not None
        resumed = BackendEngine(
            conformal=ConformalOptions(
                config=config,
                run_id=run_id,
                state_store=SqlConformalStateStore(ConformalStateRepo(session)),
                initial_ledger=initial_ledger,
            ),
        ).execute(partition_tasks([task]), actuals, origins=origins[2:])

    sort_cols = [UNIQUE_ID, FORECAST_ORIGIN, H]
    expected = uninterrupted.ledger.to_df().sort_values(sort_cols).reset_index(drop=True)
    actual = resumed.ledger.to_df().sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_backend_resumes_across_deferred_cumulative_window(tmp_path) -> None:
    """A kill with a cumulative window mid-flight resumes to the exact
    uninterrupted result: deferred y-NaN rows re-enter the open set via the
    initial ledger, the window completes at its terminal origin, and it scores
    exactly once (the restored calibrator state carries no score for it yet).
    """
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 5
    actuals = pd.DataFrame({"unique_id": "SKU_001", "ds": dates, "y": pattern})
    task = ForecastTask(
        history=actuals,
        horizon=3,
        model_config={"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 4},
    )
    config = SymmetricIntervalConfig(
        method="mscp",
        coverage=0.9,
        calibration_window=10,
        mode="cumulative",
        protection_period=3,
    )
    # Weekly origins one step apart: the dates[7] window completes only at
    # dates[9], so killing after two origins leaves it mid-flight (deferred).
    origins = [dates[7], dates[8], dates[9], dates[10], dates[11]]

    uninterrupted = BackendEngine(conformal=ConformalOptions(config=config)).execute(
        partition_tasks([task]), actuals, origins=origins
    )

    db = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'resume-cumulative.db').as_posix()}")
    Base.metadata.create_all(db)
    factory = make_session_factory(db)
    ledger_path = tmp_path / "ledger-cumulative.parquet"

    with session_scope(factory) as session:
        run = RunRepo(session).create(config={"name": "resume-cumulative-test"})
        run_id = run.id
        interrupted = BackendEngine(
            conformal=ConformalOptions(
                config=config,
                run_id=run_id,
                state_store=SqlConformalStateStore(ConformalStateRepo(session)),
            ),
        ).execute(partition_tasks([task]), actuals, origins=origins[:2])
        part_one = interrupted.ledger.to_df()
        # The seam this test exists for: the part-1 ledger carries deferred
        # in-window rows (y NaN) from the still-incomplete window.
        deferred = part_one[(part_one[H] <= 3) & part_one[Y].isna()]
        assert not deferred.empty, "kill point must leave a window mid-flight"
        pointer = write_ledger_shard(part_one, str(ledger_path))
        ForecastPointerRepo(session).upsert(
            run_id, "ledger", str(pointer["uri"]), int(pointer["byte_size"])
        )

    with session_scope(factory) as session:
        initial_ledger = read_initial_ledger(ForecastPointerRepo(session), run_id)
        assert initial_ledger is not None
        resumed = BackendEngine(
            conformal=ConformalOptions(
                config=config,
                run_id=run_id,
                state_store=SqlConformalStateStore(ConformalStateRepo(session)),
                initial_ledger=initial_ledger,
            ),
        ).execute(partition_tasks([task]), actuals, origins=origins[2:])

    sort_cols = [UNIQUE_ID, FORECAST_ORIGIN, H]
    expected = uninterrupted.ledger.to_df().sort_values(sort_cols).reset_index(drop=True)
    actual = resumed.ledger.to_df().sort_values(sort_cols).reset_index(drop=True)

    # Full-frame equality covers exactly-once scoring: a double-observed window
    # would drift nonconformity_score; a lost window would leave it NaN.
    pd.testing.assert_frame_equal(actual, expected)
    assert expected[NONCONFORMITY_SCORE].notna().any()
