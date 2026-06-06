from __future__ import annotations

from uuid import uuid4

import pandas as pd

from calibre.conformal import SymmetricIntervalConfig
from calibre.core.forecast_frame import FORECAST_ORIGIN, UNIQUE_ID, H
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine, ConformalOptions
from calibre.execution.task_builder import partition_tasks
from calibre.storage.models import Base
from calibre.storage.postgres import (
    ConformalStateRepo,
    RunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)
from calibre.storage.session import derive_session_id
from calibre.storage.state import SqlConformalStateStore


def test_same_session_id_resumes_across_runs(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'state.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    session_id = derive_session_id("tenant", ["sku"], {"model": "m"}, {"method": "aci"})

    with session_scope(factory) as session:
        first_run = RunRepo(session).create(config={"run": 1})
        SqlConformalStateStore(
            ConformalStateRepo(session),
            session_id=session_id,
        ).upsert(first_run.id, "sku:model:h1", {"score": 1.0})

    with session_scope(factory) as session:
        second_run = RunRepo(session).create(config={"run": 2})
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=session_id)

        assert store.get(second_run.id, "sku:model:h1") == {"score": 1.0}


def test_different_session_id_starts_fresh(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'state.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    with session_scope(factory) as session:
        run_id = RunRepo(session).create(config={}).id
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=uuid4().hex)
        store.upsert(run_id, "sku:model:h1", {"score": 1.0})

    with session_scope(factory) as session:
        run_id = RunRepo(session).create(config={}).id
        store = SqlConformalStateStore(ConformalStateRepo(session), session_id=uuid4().hex)

        assert store.get(run_id, "sku:model:h1") is None


def test_same_session_id_hydrates_backend_state_across_runs(tmp_path) -> None:
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
    session_id = derive_session_id(
        "tenant",
        ["SKU_001"],
        task.model_config,
        {
            "method": config.method,
            "coverage": config.coverage,
            "calibration_window": config.calibration_window,
            "gamma": config.gamma,
        },
    )

    uninterrupted = BackendEngine(conformal=ConformalOptions(config=config)).execute(
        [task],
        actuals,
        origins=origins,
    )

    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'resume.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    with session_scope(factory) as session:
        first_run = RunRepo(session).create(config={"run": 1})
        BackendEngine(
            conformal=ConformalOptions(
                config=config,
                run_id=first_run.id,
                state_store=SqlConformalStateStore(
                    ConformalStateRepo(session),
                    session_id=session_id,
                ),
            ),
        ).execute(partition_tasks([task]), actuals, origins=origins[:2])

    with session_scope(factory) as session:
        second_run = RunRepo(session).create(config={"run": 2})
        resumed = BackendEngine(
            conformal=ConformalOptions(
                config=config,
                run_id=second_run.id,
                state_store=SqlConformalStateStore(
                    ConformalStateRepo(session),
                    session_id=session_id,
                ),
            ),
        ).execute(partition_tasks([task]), actuals, origins=origins[2:])

    sort_cols = [UNIQUE_ID, FORECAST_ORIGIN, H]
    expected = uninterrupted.ledger.to_df()
    expected = expected[expected[FORECAST_ORIGIN] == origins[2]]
    expected = expected.sort_values(sort_cols).reset_index(drop=True)
    actual = resumed.ledger.to_df().sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)
