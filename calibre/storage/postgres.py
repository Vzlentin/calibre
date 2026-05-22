from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.core.run_status import RunStatus
from calibre.storage.models import (
    ConformalState,
    ForecastPointer,
    PendingObservation,
    Run,
    TuningRun,
)


def database_url() -> str | None:
    return os.environ.get("CALIBRE_DATABASE_URL")


def make_engine(url: str, **kwargs) -> Engine:
    return create_engine(url, future=True, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class RunRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, *, config: dict, idempotency_key: str | None = None) -> Run:
        if idempotency_key is not None:
            existing = self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
        run = Run(config=config, idempotency_key=idempotency_key, status=RunStatus.QUEUED.value)
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: UUID) -> Run | None:
        return self.session.get(Run, run_id)

    def get_by_idempotency_key(self, key: str) -> Run | None:
        return self.session.scalar(select(Run).where(Run.idempotency_key == key))

    def set_status(
        self,
        run_id: UUID,
        status: RunStatus | str,
        *,
        error: str | None = None,
    ) -> None:
        parsed_status = RunStatus(status)
        run = self.get(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        run.status = parsed_status.value
        run.error = error
        if parsed_status in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
            run.finished_at = datetime.now(UTC)
        else:
            run.finished_at = None


class ConformalStateRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, session_id: str, partition: str) -> dict | None:
        row = self.session.get(ConformalState, (session_id, partition))
        return dict(row.state) if row is not None else None

    def list_for_session(self, session_id: str) -> dict[str, dict]:
        rows = self.session.scalars(
            select(ConformalState).where(ConformalState.session_id == session_id)
        )
        return {row.partition: dict(row.state) for row in rows}

    def upsert(self, session_id: str, partition: str, state: dict, *, run_id: UUID) -> None:
        row = self.session.get(ConformalState, (session_id, partition))
        if row is None:
            self.session.add(
                ConformalState(
                    session_id=session_id,
                    run_id=run_id,
                    partition=partition,
                    state=dict(state),
                )
            )
        else:
            row.run_id = run_id
            row.state = dict(state)


class PendingObservationRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_for_session(self, session_id: str) -> list[PendingObservation]:
        return list(
            self.session.scalars(
                select(PendingObservation).where(PendingObservation.session_id == session_id)
            )
        )

    def to_frames(
        self,
        session_id: str,
        *,
        lower_col: str,
        upper_col: str,
    ) -> list[pd.DataFrame]:
        rows = self.list_for_session(session_id)
        if not rows:
            return []
        frame = pd.DataFrame(
            [
                {
                    UNIQUE_ID: row.uid,
                    DS: pd.Timestamp(row.ds),
                    FORECAST_ORIGIN: pd.Timestamp(row.origin),
                    MODEL_NAME: row.model_name,
                    H: int(row.h),
                    Y: float("nan"),
                    Y_HAT: row.y_hat,
                    lower_col: row.lo,
                    upper_col: row.hi,
                }
                for row in rows
            ]
        )
        sorted_frame = frame.sort_values([UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN, H])
        return [sorted_frame.reset_index(drop=True)]

    def upsert_frame(
        self,
        session_id: str,
        frame: pd.DataFrame,
        *,
        lower_col: str,
        upper_col: str,
    ) -> None:
        if frame.empty:
            return
        missing = {
            UNIQUE_ID,
            DS,
            FORECAST_ORIGIN,
            MODEL_NAME,
            H,
            Y_HAT,
            lower_col,
            upper_col,
        } - set(frame.columns)
        if missing:
            raise ValueError(f"Pending observation frame missing columns: {sorted(missing)}")

        dialect = self.session.get_bind().dialect.name
        for _, row in frame.iterrows():
            uid = str(row[UNIQUE_ID])
            model_name = str(row[MODEL_NAME])
            origin = pd.Timestamp(row[FORECAST_ORIGIN]).to_pydatetime()
            horizon = int(row[H])
            ds = pd.Timestamp(row[DS]).to_pydatetime()
            lo = None if pd.isna(row[lower_col]) else float(row[lower_col])
            hi = None if pd.isna(row[upper_col]) else float(row[upper_col])
            y_hat = None if pd.isna(row[Y_HAT]) else float(row[Y_HAT])
            values: dict[str, Any] = {
                "session_id": session_id,
                "uid": uid,
                "model_name": model_name,
                "origin": origin,
                "h": horizon,
                "ds": ds,
                "lo": lo,
                "hi": hi,
                "y_hat": y_hat,
            }
            if dialect == "sqlite":
                stmt = sqlite_insert(PendingObservation).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["session_id", "uid", "model_name", "origin", "h"],
                    set_={
                        "ds": stmt.excluded.ds,
                        "lo": stmt.excluded.lo,
                        "hi": stmt.excluded.hi,
                        "y_hat": stmt.excluded.y_hat,
                    },
                )
                self.session.execute(stmt)
                continue
            existing = self.session.get(
                PendingObservation,
                (session_id, uid, model_name, origin, horizon),
            )
            if existing is None:
                self.session.add(
                    PendingObservation(
                        session_id=session_id,
                        uid=uid,
                        model_name=model_name,
                        origin=origin,
                        h=horizon,
                        ds=ds,
                        lo=lo,
                        hi=hi,
                        y_hat=y_hat,
                    )
                )
            else:
                existing.ds = ds
                existing.lo = lo
                existing.hi = hi
                existing.y_hat = y_hat

    def replace_session(
        self,
        session_id: str,
        frames: list[pd.DataFrame],
        *,
        lower_col: str,
        upper_col: str,
    ) -> None:
        from sqlalchemy import delete

        self.session.execute(
            delete(PendingObservation).where(PendingObservation.session_id == session_id)
        )
        for frame in frames:
            self.upsert_frame(
                session_id,
                frame,
                lower_col=lower_col,
                upper_col=upper_col,
            )


class TuningRunRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, session_id: str, unique_id: str) -> TuningRun | None:
        return self.session.get(TuningRun, (session_id, unique_id))

    def list_for_session(self, session_id: str) -> list[TuningRun]:
        return list(
            self.session.scalars(select(TuningRun).where(TuningRun.session_id == session_id))
        )

    def upsert(
        self,
        session_id: str,
        unique_id: str,
        *,
        candidate: dict,
        score: float | None,
    ) -> None:
        row = self.session.get(TuningRun, (session_id, unique_id))
        if row is None:
            self.session.add(
                TuningRun(
                    session_id=session_id,
                    unique_id=unique_id,
                    candidate=dict(candidate),
                    score=score,
                    finished_at=datetime.now(UTC),
                )
            )
        else:
            row.candidate = dict(candidate)
            row.score = score
            row.finished_at = datetime.now(UTC)


class ForecastPointerRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, run_id: UUID, kind: str) -> ForecastPointer | None:
        return self.session.get(ForecastPointer, (run_id, kind))

    def list_for_run(self, run_id: UUID) -> list[ForecastPointer]:
        return list(
            self.session.scalars(select(ForecastPointer).where(ForecastPointer.run_id == run_id))
        )

    def upsert(self, run_id: UUID, kind: str, uri: str, byte_size: int) -> None:
        row = self.session.get(ForecastPointer, (run_id, kind))
        if row is None:
            self.session.add(
                ForecastPointer(run_id=run_id, kind=kind, uri=uri, byte_size=byte_size)
            )
        else:
            row.uri = uri
            row.byte_size = int(byte_size)
