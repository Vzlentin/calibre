"""SqlLifecycleStore round-trip + restart survival (roadmap P0.2).

The in-memory store loses everything on restart and is invisible across
workers. These tests assert the SQL store persists fit/tune records, frames
(as parquet by reference), and session-owned conformal state across a simulated
restart — a brand-new store/engine on the same database + artifact base.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from calibre.api.lifecycle import FitRecord, TuneRecord
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, Y
from calibre.core.run_status import RunStatus
from calibre.storage.lifecycle_repo import SqlLifecycleStore
from calibre.storage.models import Base, LifecycleFitRecord
from calibre.storage.postgres import make_engine, make_session_factory, session_scope


def _stores(tmp_path):
    """A factory returning fresh stores over the same db + artifact base."""
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'lc.db').as_posix()}"
    artifact_base = str(tmp_path / "artifacts")
    Base.metadata.create_all(make_engine(db_url))

    def make() -> SqlLifecycleStore:
        return SqlLifecycleStore(make_session_factory(make_engine(db_url)), artifact_base)

    return make


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "A"],
            DS: pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-21"]),
            Y: [1.0, 2.0, 3.0],
        }
    )


def _fit_record() -> FitRecord:
    return FitRecord(
        fit_id="fit-1",
        session_id="sess-1",
        tenant="acme",
        sku_set=["A"],
        forecaster_config={"backend": "statsforecast", "model": "Naive"},
        horizon=2,
        freq="W-SUN",
        history=_history(),
        future_x=None,
        conformal_config={"method": "aci", "coverage": 0.9},
        status=RunStatus.QUEUED,
    )


def test_fit_survives_restart_with_frames(tmp_path):
    make = _stores(tmp_path)
    make().put_fit(_fit_record())

    # Restart: brand-new store/engine on the same db + artifact base.
    loaded = make().get_fit("fit-1")
    assert loaded is not None
    assert loaded.tenant == "acme"
    assert loaded.sku_set == ["A"]
    assert loaded.conformal_config == {"method": "aci", "coverage": 0.9}
    assert loaded.status == RunStatus.QUEUED
    assert loaded.future_x is None  # absent frame stays absent
    pd.testing.assert_frame_equal(loaded.history.reset_index(drop=True), _history())


def test_update_fit_persists_status_and_new_frame(tmp_path):
    make = _stores(tmp_path)
    make().put_fit(_fit_record())

    forecast = pd.DataFrame({UNIQUE_ID: ["A"], DS: pd.to_datetime(["2024-01-28"]), Y_HAT: [4.0]})
    make().update_fit("fit-1", status=RunStatus.SUCCEEDED, last_forecast=forecast)

    loaded = make().get_fit("fit-1")
    assert loaded is not None
    assert loaded.status == RunStatus.SUCCEEDED
    assert loaded.last_forecast is not None
    pd.testing.assert_frame_equal(loaded.last_forecast.reset_index(drop=True), forecast)


def test_conformal_state_is_session_owned_and_survives_restart(tmp_path):
    make = _stores(tmp_path)
    make().upsert_conformal_state("sess-1", {"p1": {"score": 1.0}, "p2": {"score": 2.0}})

    assert make().get_conformal_state("sess-1") == {
        "p1": {"score": 1.0},
        "p2": {"score": 2.0},
    }

    # Upsert overwrites an existing partition, leaves others.
    make().upsert_conformal_state("sess-1", {"p1": {"score": 9.0}})
    assert make().get_conformal_state("sess-1") == {
        "p1": {"score": 9.0},
        "p2": {"score": 2.0},
    }


def test_lookups_by_session_and_tenant(tmp_path):
    make = _stores(tmp_path)
    store = make()
    store.put_fit(_fit_record())

    assert {r.fit_id for r in make().fits_for_session("sess-1")} == {"fit-1"}
    assert make().first_fit_for_session("sess-1").fit_id == "fit-1"
    assert make().first_fit_for_session("missing") is None
    assert {r.fit_id for r in make().fits_for_tenant_uid("acme", "A")} == {"fit-1"}
    assert make().fits_for_tenant_uid("acme", "ZZZ") == []


def test_study_round_trip_and_update(tmp_path):
    make = _stores(tmp_path)
    make().put_study(
        TuneRecord(
            study_id="study-1",
            session_id="sess-1",
            tenant="acme",
            sku_set=["A"],
            status=RunStatus.QUEUED,
        )
    )
    make().update_study(
        "study-1",
        status=RunStatus.SUCCEEDED,
        best_candidates={"A": {"model_config": {"season_length": 4}}},
    )

    loaded = make().get_study("study-1")
    assert loaded is not None
    assert loaded.status == RunStatus.SUCCEEDED
    assert loaded.best_candidates == {"A": {"model_config": {"season_length": 4}}}


def test_first_fit_uses_creation_order_not_fit_id(tmp_path):
    """A session can hold several fits; "first" must be the earliest created,
    not the lexicographically smallest (random) fit_id."""
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'lc.db').as_posix()}"
    Base.metadata.create_all(make_engine(db_url))
    factory = make_session_factory(make_engine(db_url))

    earlier = datetime(2024, 1, 1, tzinfo=UTC)
    # The earlier-created fit deliberately has the LARGER fit_id, so ordering by
    # fit_id alone would wrongly pick the later one.
    with session_scope(factory) as session:
        for fit_id, created in [("zzz", earlier), ("aaa", earlier + timedelta(hours=1))]:
            session.add(
                LifecycleFitRecord(
                    fit_id=fit_id,
                    session_id="s",
                    tenant="t",
                    sku_set=["A"],
                    forecaster_config={},
                    horizon=1,
                    freq="W",
                    conformal_config=None,
                    status="queued",
                    artifact_urls={},
                    frame_uris={},
                    created_at=created,
                )
            )

    store = SqlLifecycleStore(make_session_factory(make_engine(db_url)), str(tmp_path / "art"))
    first = store.first_fit_for_session("s")
    assert first is not None
    assert first.fit_id == "zzz", "first_fit should be the earliest created, not min fit_id"
