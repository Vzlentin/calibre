"""Tests for pending-observation deferral and resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

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
from calibre.core.forecast_task import TaskGroups
from calibre.execution.decision_loop import DecisionLoop, DecisionLoopConfig
from calibre.storage.models import Base
from calibre.storage.postgres import (
    PendingObservationRepo,
    make_engine,
    make_session_factory,
    session_scope,
)


def _conformal_frame(origin: pd.Timestamp) -> pd.DataFrame:
    lower_col, upper_col = interval_column_names(0.9)
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            DS: [origin + pd.Timedelta(weeks=1), origin + pd.Timedelta(weeks=2)],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["model", "model"],
            H: [1, 2],
            Y: [float("nan"), float("nan")],
            Y_HAT: [10.0, 20.0],
            lower_col: [8.0, 18.0],
            upper_col: [12.0, 22.0],
        }
    )


def _loop(
    *,
    repo: PendingObservationRepo,
    session_id: str,
    runtime: MagicMock,
    origin: pd.Timestamp,
    actual: float,
) -> DecisionLoop:
    engine = MagicMock()
    engine.freq = "W-MON"
    result = MagicMock()
    result.ledger.to_df.return_value = pd.DataFrame()
    engine.execute.return_value = result
    lower_col, upper_col = interval_column_names(0.9)
    runtime.interval_columns = (lower_col, upper_col)
    runtime.mode = "perhorizon"

    return DecisionLoop(
        engine=engine,
        simulator=MagicMock(),
        build_round_tasks=lambda round_num: (TaskGroups(), origin, pd.DataFrame()),
        policy=lambda frame: {},
        get_actuals=lambda round_num: {"A": actual},
        config=DecisionLoopConfig(n_rounds=1),
        runtime=runtime,
        pending_observation_repo=repo,
        session_id=session_id,
    )


def test_pending_persists_across_process_restart(tmp_path) -> None:
    engine = make_engine(f"sqlite+pysqlite:///{(tmp_path / 'pending.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    session_id = "session"
    first_origin = pd.Timestamp("2024-01-01")

    with session_scope(factory) as session:
        runtime = MagicMock()
        runtime.apply.return_value = _conformal_frame(first_origin)
        observed: list[pd.DataFrame] = []
        runtime.observe.side_effect = lambda frame: observed.append(frame.reset_index(drop=True))
        _loop(
            repo=PendingObservationRepo(session),
            session_id=session_id,
            runtime=runtime,
            origin=first_origin,
            actual=10.0,
        ).run()

        # Only the h=1 forecast has resolved (its actual landed this round); it
        # is the exact row handed to the runtime, carrying its resolved actual.
        [first_observed] = observed
        assert first_observed[H].tolist() == [1]
        assert first_observed[Y].tolist() == [10.0]
        assert first_observed[Y_HAT].tolist() == [10.0]

        # The still-unresolved h=2 forecast must outlive the process restart.
        pending = PendingObservationRepo(session).list_for_session(session_id)
        assert [(row.uid, row.h, row.y_hat) for row in pending] == [("A", 2, 20.0)]

    with session_scope(factory) as session:
        runtime = MagicMock()
        runtime.apply.return_value = _conformal_frame(first_origin).iloc[0:0].copy()
        observed = []
        runtime.observe.side_effect = lambda frame: observed.append(frame.reset_index(drop=True))
        _loop(
            repo=PendingObservationRepo(session),
            session_id=session_id,
            runtime=runtime,
            origin=first_origin + pd.Timedelta(weeks=1),
            actual=20.0,
        ).run()

        # The persisted h=2 row is rehydrated and observed with its now-resolved
        # actual, then cleared from the pending store.
        [resumed_observed] = observed
        assert resumed_observed[H].tolist() == [2]
        assert resumed_observed[Y].tolist() == [20.0]
        assert resumed_observed[Y_HAT].tolist() == [20.0]
        assert PendingObservationRepo(session).list_for_session(session_id) == []
